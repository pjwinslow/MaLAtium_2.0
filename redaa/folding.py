"""ESMFold2-Fast folding engine.

Replaces the V9 ``facebook/esmfold_v1`` path.  ESMFold2-Fast is the
speed-optimised, single-sequence member of the ESMFold2 family: 24 folding
blocks, no MSA conditioning, an ESM-C backbone in bf16 and an fp32 structure
trunk.  It is loaded through ``transformers.AutoModel`` with
``trust_remote_code=True``, which pulls the FastPLMs implementation shipped
inside the model repository.

Three things make this fast in a search loop, in descending order of impact:

1. **Nothing is folded twice.**  Every prediction goes through
   :class:`~redaa.foldcache.FoldCache`, so repeated sequences -- common under
   restricted alphabets, and across replicates -- are free.
2. **Screening folds are cheap.**  ESMFold2's structure module is a diffusion
   sampler, so ``num_sampling_steps`` trades wall time against precision
   directly.  Deciding "right fold or wrong fold" does not need the step count
   that a reported number needs, so the two run at different settings.
3. **No disk round-trip.**  V9 wrote every prediction to a PDB file and then
   re-parsed it (four times) with BioPython to get CA coordinates.  Here the
   coordinates come straight out of the returned PDB text in memory, and a
   file is written only when we actually want to keep that structure.

The model's exact keyword arguments vary between ESMFold2 releases, so the
call is built by introspecting the bound method and passing only what it
accepts.  That keeps the pipeline working across point releases instead of
dying on an unexpected-keyword error.
"""

from __future__ import annotations

import inspect
import os
import time
from dataclasses import dataclass

import numpy as np

from . import structure as st
from .foldcache import FoldCache, fold_key


@dataclass
class FoldResult:
    """One ESMFold2 prediction, reduced to what the search actually uses."""

    sequence: str
    plddt: float | None          # 0-100 scale, as in the AlphaFold convention
    ptm: float | None
    ca: np.ndarray               # (L, 3) CA coordinates
    pdb_text: str | None = None  # only populated on a cache miss
    # (L, 3, 3) N/CA/C per residue.  Only parsed and cached when a functional
    # site is configured, because every fold pays for it and a site-blind run
    # has no use for it.  None means "not requested or not available", which
    # the site metrics report as unmeasurable rather than as a failure.
    frames: np.ndarray | None = None

    @property
    def ok(self) -> bool:
        return self.ca is not None and len(self.ca) >= 3

    @property
    def site_frames(self) -> np.ndarray | None:
        """``(L, 4, 3)`` N/CA/C plus derived virtual CB, or None."""
        if self.frames is None:
            return None
        return st.frames_with_cb(self.frames)


def _to_float(value) -> float | None:
    """Best-effort scalar extraction from a tensor, array or number."""
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().to("cpu").float()
        arr = np.asarray(value, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return float(arr.mean())
    except Exception:
        return None


def normalise_plddt(value: float | None) -> float | None:
    """Put pLDDT on the 0-100 scale regardless of which scale the model used.

    ESMFold v1 reported 0-1 and V9 multiplied by 100; ESMFold2 releases have
    used both conventions.  Rescaling on magnitude rather than on a hardcoded
    assumption means the acceptance thresholds keep meaning the same thing.
    """
    if value is None:
        return None
    return value * 100.0 if value <= 1.5 else value


class FoldEngine:
    """Thin, cached wrapper around an ESMFold2-Fast checkpoint."""

    def __init__(
        self,
        cfg,
        cache: FoldCache | None = None,
        verbose: bool = True,
        want_frames: bool = False,
    ):
        self.cfg = cfg
        # Set once by the pipeline when a functional site is configured, so
        # every fold in the run carries the backbone frame the site metrics
        # need without each caller opting in.
        self.want_frames = bool(want_frames)
        # Set if the model's output turns out to have no backbone frame, so
        # the warning is printed once rather than per fold.
        self.frames_unavailable = False
        self.cache = cache if cache is not None else FoldCache(None, enabled=False)
        self.verbose = verbose
        self.model = None
        self.device = "cpu"
        self._fold_fn = None
        self._fold_name = ""
        self._fold_kwargs: dict[str, object] = {}
        self._steps_name: str | None = "num_sampling_steps"
        self._pdb_fn = None
        self.n_folded = 0
        self.fold_seconds = 0.0

    # -- setup ------------------------------------------------------------
    def load(self) -> None:
        import torch
        from transformers import AutoModel

        use_gpu = torch.cuda.is_available()
        self.device = "cuda:0" if use_gpu else "cpu"

        if use_gpu:
            # TF32 costs nothing in accuracy terms for a structure module and
            # is a large free win on A100/H100 tensor cores.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        kwargs: dict[str, object] = {"trust_remote_code": True}
        if use_gpu:
            kwargs["device_map"] = {"": self.device}
            kwargs["esmc_precision"] = self.cfg.fold_precision

        if self.verbose:
            print(f"Loading {self.cfg.fold_model_id} (trust_remote_code=True)...")

        try:
            model = AutoModel.from_pretrained(self.cfg.fold_model_id, **kwargs)
        except (TypeError, ValueError) as exc:
            # Older or stripped-down revisions may not accept esmc_precision at
            # all, and a revision that does may reject the particular value --
            # 'fp16' is worth trying on a pre-Ampere GPU with no bf16, but not
            # every build offers it.  Either way, retry on the default.
            if "esmc_precision" not in kwargs:
                raise
            print(
                f"  esmc_precision={self.cfg.fold_precision!r} rejected "
                f"({exc}); retrying with the model default"
            )
            kwargs.pop("esmc_precision", None)
            model = AutoModel.from_pretrained(self.cfg.fold_model_id, **kwargs)

        model = model.eval()
        if use_gpu and not hasattr(model, "hf_device_map"):
            model = model.to(self.device)
        self.model = model

        # Resolve the single-protein entry point.  Different ESMFold2 releases
        # name it fold_protein or infer_protein.
        for name in ("fold_protein", "infer_protein", "fold_sequence"):
            fn = getattr(model, name, None)
            if callable(fn):
                self._fold_fn = fn
                self._fold_name = name
                break
        if self._fold_fn is None:
            raise RuntimeError(
                f"{self.cfg.fold_model_id} exposes no fold_protein/infer_protein "
                "method; check the installed FastPLMs revision."
            )

        self._fold_kwargs = self._supported_kwargs(self._fold_fn)
        self._steps_name = self._resolve_steps_name()

        for name in ("result_to_pdb", "output_to_pdb", "to_pdb"):
            fn = getattr(model, name, None)
            if callable(fn):
                self._pdb_fn = fn
                break
        if self._pdb_fn is None:
            raise RuntimeError(
                f"{self.cfg.fold_model_id} exposes no result_to_pdb method; "
                "cannot recover coordinates."
            )

        if self.cfg.use_torch_compile:
            try:
                self.model = torch.compile(self.model, dynamic=True)
                if self.verbose:
                    print("  torch.compile enabled (first fold pays the warm-up)")
            except Exception as exc:  # noqa: BLE001
                print(f"  torch.compile unavailable ({exc}); continuing eager")

        if self.verbose:
            where = "GPU" if use_gpu else "CPU (expect minutes per sequence)"
            print(
                f"  ready on {where} via {self._fold_name}(), "
                f"screen={self.cfg.fold_steps_screen} steps, "
                f"final={self.cfg.fold_steps_final} steps"
            )

    def _supported_kwargs(self, fn) -> dict[str, object]:
        """Map our settings onto whichever keywords *fn* actually accepts."""
        wanted = {
            "num_loops": self.cfg.fold_loops,
            "num_recycles": self.cfg.fold_loops,
            "num_diffusion_samples": 1,
            "seed": self.cfg.fold_seed,
            "verbose": False,
        }
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return {"seed": self.cfg.fold_seed}
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            # Accepts **kwargs; still avoid guessing at recycles vs loops.
            return {
                k: v for k, v in wanted.items() if k in params or k != "num_recycles"
            }
        return {k: v for k, v in wanted.items() if k in params}

    def _resolve_steps_name(self) -> str | None:
        """Name the sampler-step argument the way this revision spells it.

        Getting this wrong would silently fall back to the model's default step
        count and quietly erase the screen/final distinction, which is one of
        the larger speedups here -- so when the signature only shows
        ``**kwargs`` we pass the documented name rather than nothing.
        Resolved once at load time, not per fold.
        """
        try:
            params = inspect.signature(self._fold_fn).parameters
        except (TypeError, ValueError):
            return "num_sampling_steps"
        for name in ("num_sampling_steps", "num_steps", "sampling_steps"):
            if name in params:
                return name
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return "num_sampling_steps"
        print(
            "  NOTE: this ESMFold2 revision exposes no sampler-step argument, "
            "so FOLD_STEPS_SCREEN/FOLD_STEPS_FINAL have no effect"
        )
        return None

    def _steps_kwarg(self, steps: int) -> dict[str, object]:
        if self._steps_name is None:
            return {}
        return {self._steps_name: steps}

    # -- folding ----------------------------------------------------------
    def fold(
        self,
        sequence: str,
        steps: int | None = None,
        want_frames: bool | None = None,
    ) -> FoldResult | None:
        """Fold one sequence, consulting the cache first.

        Returns ``None`` if the prediction failed outright; the search treats
        that as "no structural model produced", exactly as V9 did.

        ``want_frames`` asks for the backbone frame as well as the CA trace.
        It defaults to whatever the engine was constructed with, so a
        site-aware run gets frames everywhere without every call site having
        to remember to ask.
        """
        if not sequence:
            return None
        steps = self.cfg.fold_steps_screen if steps is None else steps
        want_frames = self.want_frames if want_frames is None else bool(want_frames)

        key = fold_key(
            sequence,
            self.cfg.fold_model_id,
            steps,
            self.cfg.fold_loops,
            self.cfg.fold_seed,
        )
        cached = self.cache.get(key, need_frames=want_frames)
        if cached is not None:
            plddt, ptm, ca, frames = cached
            return FoldResult(sequence, plddt, ptm, ca, frames=frames)

        if len(sequence) > self.cfg.fold_max_residues:
            print(
                f"    skipping fold: {len(sequence)} residues exceeds "
                f"FOLD_MAX_RESIDUES={self.cfg.fold_max_residues}"
            )
            return None

        if self.model is None:
            self.load()

        import torch

        call_kwargs = dict(self._fold_kwargs)
        call_kwargs.update(self._steps_kwarg(steps))

        started = time.time()
        try:
            with torch.inference_mode():
                result = self._fold_fn(sequence, **call_kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"    ESMFold2 OOM at L={len(sequence)}; skipping this sequence")
            return None
        except Exception as exc:  # noqa: BLE001
            print(f"    ESMFold2 error: {exc}")
            return None
        finally:
            self.fold_seconds += time.time() - started
        self.n_folded += 1

        pdb_text = self._result_pdb_text(result)
        if not pdb_text:
            print("    ESMFold2 returned no PDB text for this sequence")
            return None

        frames = None
        if want_frames:
            records = st.parse_residues_from_text(pdb_text)
            full, valid = st.frames_from_records(records)
            # Store N/CA/C only; CB is reconstructed on read, so the cache
            # does not carry a derived quantity.
            frames = full[:, :3, :]
            ca = full[:, 1, :]
            if not valid.any():
                # The model returned coordinates but no complete N-CA-C for
                # any residue. Failing every fold from here would bury the
                # cause under thousands of identical messages, so fall back to
                # the CA trace -- the fold gate still works -- and say once
                # that site geometry is not measurable. The pipeline checks
                # this after the native fold and stops there.
                if not self.frames_unavailable:
                    self.frames_unavailable = True
                    print(
                        "    WARNING: this model's PDB output carries no "
                        "N/CA/C backbone, so functional-site geometry cannot "
                        "be measured from it. Falling back to the CA trace."
                    )
                frames = None
                ca, _ = st.parse_ca_from_text(pdb_text)
            else:
                ca = ca[valid]
                frames = frames[valid]
        else:
            ca, _ = st.parse_ca_from_text(pdb_text)
        if len(ca) < 3:
            print("    ESMFold2 output had no usable CA trace")
            return None

        plddt = normalise_plddt(_to_float(getattr(result, "plddt", None)))
        ptm = _to_float(getattr(result, "ptm", None))

        self.cache.put(key, plddt, ptm, ca, frames)
        return FoldResult(
            sequence, plddt, ptm, ca, pdb_text=pdb_text, frames=frames
        )

    def _result_pdb_text(self, result) -> str | None:
        try:
            out = self._pdb_fn(result)
        except Exception as exc:  # noqa: BLE001
            print(f"    result_to_pdb failed: {exc}")
            return None
        if isinstance(out, str):
            return out
        if isinstance(out, (list, tuple)) and out:
            return out[0] if isinstance(out[0], str) else None
        return None

    def fold_many(
        self, sequences: list[str], steps: int | None = None
    ) -> list[FoldResult | None]:
        """Fold several sequences, cheapest-first.

        Cached sequences are resolved without touching the GPU.  The rest are
        folded shortest-first, so if a wall-clock limit cuts the batch short we
        have spent the time on the most predictions rather than the fewest.
        ESMFold2-Fast's documented interface takes one protein per call --
        ``fold()`` is for multi-chain complexes, not independent monomers, so
        it is deliberately not used to batch here.
        """
        steps = self.cfg.fold_steps_screen if steps is None else steps
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        out: list[FoldResult | None] = [None] * len(sequences)
        for i in order:
            out[i] = self.fold(sequences[i], steps=steps)
        return out

    # -- helpers ----------------------------------------------------------
    def write_pdb(self, result: FoldResult, path: str) -> bool:
        """Persist a prediction, if we still hold its text.

        A cache hit carries coordinates but not the original PDB text, so a
        structure that must be kept on disk is re-folded.  That only happens
        for champion sequences, which is a handful per domain.
        """
        text = result.pdb_text
        if text is None:
            refold = self.fold(result.sequence, steps=self.cfg.fold_steps_final)
            text = refold.pdb_text if refold is not None else None
        if not text:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
        return True

    def stats(self) -> str:
        avg = (self.fold_seconds / self.n_folded) if self.n_folded else 0.0
        return (
            f"{self.n_folded} folds computed in {self.fold_seconds / 60.0:.1f} min "
            f"({avg:.2f} s/fold); {self.cache.stats()}"
        )
