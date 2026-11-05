"""In-process LigandMPNN sequence design.

This is the single largest speedup over V9, and it is almost entirely
bookkeeping rather than cleverness.

V9 called ProteinMPNN by shelling out to ``protein_mpnn_run.py`` once per
candidate alphabet:

    subprocess.run("python protein_mpnn_run.py --pdb_path ... --bias_AA_jsonl ...")

Every one of those calls paid for a fresh interpreter, a fresh ``import torch``,
a fresh CUDA context, a fresh checkpoint load, a fresh PDB parse and a fresh
featurisation -- several seconds of fixed cost -- to do roughly 50 ms of actual
GPU work.  It then wrote a FASTA file which the parent process immediately read
back and deleted.  A single replicate makes somewhere between 50 and several
hundred of those calls, so for a ten-replicate batch the overhead dominates
everything else the script does apart from folding.

Here the checkpoint is loaded once per process and -- the part that matters --
the backbone is parsed and featurised **once per domain**.  The backbone never
changes during a search; only the amino-acid bias vector does.  So every
subsequent candidate alphabet is a bias-tensor rebuild plus one
``model.sample()`` call on already-resident features.

Alphabet restriction uses the same mechanism as V9 (a large negative logit bias
on disallowed residues), applied directly to ``feature_dict["bias"]`` instead
of being routed through a JSON file on disk.  One deliberate difference: the
unknown token ``X`` is biased out too.  V9's ``bias.jsonl`` only covered the 20
standard residues, which left ``X`` samplable and meant downstream code had to
tolerate it.
"""

from __future__ import annotations

import importlib
import inspect
import os
import statistics
import sys
import time
from dataclasses import dataclass

NUM_TOKENS = 21  # 20 standard residues + X
_BIG_NEGATIVE = -1e8  # matches LigandMPNN's own omit-residue penalty


@dataclass
class MPNNDesigns:
    """Result of one design call."""

    mean_score: float
    pairs: list[tuple[float, str]]  # (score, sequence), ascending; lower is better

    @property
    def sequences(self) -> list[str]:
        return [s for _, s in self.pairs]


def _enforce_fixed(seq: str, fixed: dict[int, str]) -> tuple[str, int]:
    """Force *fixed* residues into *seq*; return the string and how many moved."""
    chars = list(seq)
    moved = 0
    for pos, aa in fixed.items():
        if 0 <= pos < len(chars) and chars[pos] != aa:
            chars[pos] = aa
            moved += 1
    return "".join(chars), moved


def _default_checkpoint(repo: str) -> str:
    return os.path.join(repo, "model_params", "proteinmpnn_v_48_020.pt")


class MPNNEngine:
    """Persistent LigandMPNN model with per-domain feature caching."""

    def __init__(self, cfg, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        self.model = None
        self.device = None
        self._mods: dict[str, object] = {}
        self._features: dict[str, dict] = {}
        self._native: dict[str, str] = {}
        self._base_chain_mask: dict[str, object] = {}
        self._ligand_atoms: dict[str, int] = {}
        self._atom_context_num = 1
        self.n_forced_positions = 0
        self.warned_chain_mask = False
        self.n_calls = 0
        self.n_sequences = 0
        self.seconds = 0.0

    # -- setup ------------------------------------------------------------
    def load(self) -> None:
        import torch

        repo = self.cfg.mpnn_repo or os.environ.get("LIGANDMPNN_REPO", "")
        if repo:
            repo = os.path.abspath(os.path.expanduser(repo))
            if repo not in sys.path:
                sys.path.insert(0, repo)

        try:
            data_utils = importlib.import_module("data_utils")
            model_utils = importlib.import_module("model_utils")
        except ImportError as exc:
            raise RuntimeError(
                "Could not import LigandMPNN's data_utils/model_utils "
                f"({exc}).\nSet MPNN_REPO to your LigandMPNN checkout, e.g.\n"
                "  export MPNN_REPO=/projects/standard/seeligb/shared/LigandMPNN\n"
                "This pipeline drives LigandMPNN in-process; the older "
                "ProteinMPNN CLI layout (protein_mpnn_run.py) is not used."
            ) from exc

        self._mods = {"data_utils": data_utils, "model_utils": model_utils}

        ckpt_path = self.cfg.mpnn_checkpoint or _default_checkpoint(repo or ".")
        if not os.path.isfile(ckpt_path):
            raise RuntimeError(
                f"LigandMPNN checkpoint not found: {ckpt_path}\n"
                "Set MPNN_CHECKPOINT, or run LigandMPNN's get_model_params.sh."
            )

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # torch >= 2.6 defaults weights_only=True, which rejects these
        # checkpoints' plain-Python metadata entries.
        try:
            checkpoint = torch.load(
                ckpt_path, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=self.device)

        model_type = self.cfg.mpnn_model_type
        self._atom_context_num = int(checkpoint.get("atom_context_num", 1) or 1)

        self.model = model_utils.ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=checkpoint["num_edges"],
            device=self.device,
            atom_context_num=self._atom_context_num,
            model_type=model_type,
            ligand_mpnn_use_side_chain_context=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        if self.verbose:
            print(
                f"  LigandMPNN ready: {model_type} from "
                f"{os.path.basename(ckpt_path)} on {self.device} "
                f"(k_neighbors={checkpoint['num_edges']})"
            )

    # -- features ---------------------------------------------------------
    def features_for(self, pdb_path: str) -> dict:
        """Parse and featurise *pdb_path* once, then reuse it forever.

        The backbone is fixed for the whole search, so this is the work V9
        repeated on every single MPNN invocation.
        """
        cached = self._features.get(pdb_path)
        if cached is not None:
            return cached

        if self.model is None:
            self.load()

        data_utils = self._mods["data_utils"]

        parse_kwargs = {"device": self.device}
        params = inspect.signature(data_utils.parse_PDB).parameters
        if "parse_all_atoms" in params:
            parse_kwargs["parse_all_atoms"] = False
        if "parse_atoms_with_zero_occupancy" in params:
            parse_kwargs["parse_atoms_with_zero_occupancy"] = False
        parsed = data_utils.parse_PDB(pdb_path, **parse_kwargs)
        protein_dict = parsed[0]

        # Ligand context is on exactly when the checkpoint is a ligand model.
        # Turning it on for protein_mpnn would ask the network for a tensor it
        # was never trained to read; leaving it off for ligand_mpnn would load
        # the ligand-aware weights and then hide the ligand from them, which
        # is the more insidious mistake because it still runs.
        want_ligand = str(self.cfg.mpnn_model_type).strip() == "ligand_mpnn"

        n_ligand_atoms = 0
        for key in ("Y", "ligand_atoms", "xyz_37"):
            arr = protein_dict.get(key) if hasattr(protein_dict, "get") else None
            if key == "Y" and arr is not None:
                try:
                    n_ligand_atoms = int(arr.shape[0])
                except Exception:  # noqa: BLE001
                    n_ligand_atoms = 0
                break
        self._ligand_atoms[pdb_path] = n_ligand_atoms

        if want_ligand and n_ligand_atoms == 0:
            print(
                "    WARNING: MPNN_MODEL_TYPE=ligand_mpnn but this input has "
                "no ligand atoms. Designs will be made against a bare "
                "backbone. Keep the ligand in the prepared PDB "
                "(SITE_KEEP_LIGANDS) or use protein_mpnn."
            )

        feat_kwargs = {}
        fparams = inspect.signature(data_utils.featurize).parameters
        if "cutoff_for_score" in fparams:
            feat_kwargs["cutoff_for_score"] = 8.0
        if "use_atom_context" in fparams:
            feat_kwargs["use_atom_context"] = bool(want_ligand and n_ligand_atoms)
        if "number_of_ligand_atoms" in fparams:
            feat_kwargs["number_of_ligand_atoms"] = self._atom_context_num
        if "model_type" in fparams:
            feat_kwargs["model_type"] = self.cfg.mpnn_model_type

        feature_dict = data_utils.featurize(protein_dict, **feat_kwargs)
        self._features[pdb_path] = feature_dict

        # The native sequence as *MPNN* indexes it.  The site layer indexes
        # positions by parsing the same file independently, so these two must
        # agree; the pipeline compares them and refuses to run a site-aware
        # search if they do not.  An undetected off-by-one here would fix the
        # wrong residues and the run would look entirely normal.
        restype_int_to_str = data_utils.restype_int_to_str
        try:
            s_native = feature_dict["S"][0].detach().to("cpu").numpy()
            self._native[pdb_path] = "".join(
                restype_int_to_str[int(a)] for a in s_native
            )
        except Exception:  # noqa: BLE001
            self._native[pdb_path] = ""

        cm = feature_dict.get("chain_mask")
        self._base_chain_mask[pdb_path] = None if cm is None else cm.clone()
        return feature_dict

    def native_sequence(self, pdb_path: str) -> str:
        """The native sequence in MPNN's own residue order."""
        if pdb_path not in self._native:
            self.features_for(pdb_path)
        return self._native.get(pdb_path, "")

    def ligand_atom_count(self, pdb_path: str) -> int:
        if pdb_path not in self._ligand_atoms:
            self.features_for(pdb_path)
        return self._ligand_atoms.get(pdb_path, 0)

    def length_of(self, pdb_path: str) -> int:
        return int(self.features_for(pdb_path)["X"].shape[1])

    def release(self, pdb_path: str) -> None:
        """Drop one domain's cached features, freeing its VRAM."""
        self._features.pop(pdb_path, None)
        self._native.pop(pdb_path, None)
        self._base_chain_mask.pop(pdb_path, None)
        self._ligand_atoms.pop(pdb_path, None)

    # -- design -----------------------------------------------------------
    def _bias_tensor(
        self,
        allowed_aas,
        length: int,
        fixed: dict[int, str] | None = None,
    ):
        """Build a ``[1, L, 21]`` logit bias allowing only *allowed_aas*.

        Everything outside the allowed set -- including ``X`` -- gets a large
        negative bias, so sampled sequences are guaranteed to be in-alphabet
        and the search never has to discard out-of-alphabet designs.

        *fixed* maps a position to the single residue allowed there, which is
        how functional-site positions are held at their native identity.  Those
        positions are pinned through the bias as well as through the chain
        mask, so a site residue outside the current alphabet -- the normal
        case, and the whole point -- cannot be sampled away by any code path.
        """
        import torch

        restype_str_to_int = self._mods["data_utils"].restype_str_to_int
        vec = torch.full(
            [NUM_TOKENS], _BIG_NEGATIVE, device=self.device, dtype=torch.float32
        )
        for aa in allowed_aas:
            idx = restype_str_to_int.get(aa)
            if idx is not None:
                vec[idx] = 0.0
        bias = vec[None, None, :].repeat(1, length, 1)

        for pos, aa in (fixed or {}).items():
            if not 0 <= pos < length:
                continue
            idx = restype_str_to_int.get(aa)
            bias[0, pos, :] = _BIG_NEGATIVE
            if idx is not None:
                bias[0, pos, idx] = 0.0
        return bias

    def _chain_mask_for(self, pdb_path: str, length: int, fixed):
        """``[1, L]`` mask: 1 where MPNN should design, 0 where it must not.

        LigandMPNN reads this as "redesign these positions and keep the native
        residue elsewhere", which is the mechanism its own ``--fixed_residues``
        flag uses.
        """
        import torch

        base = self._base_chain_mask.get(pdb_path)
        if base is None:
            mask = torch.ones([1, length], device=self.device, dtype=torch.float32)
        else:
            mask = base.clone().to(torch.float32)
        if fixed:
            for pos in fixed:
                if 0 <= pos < mask.shape[1]:
                    mask[0, pos] = 0.0
        return mask

    def design(
        self,
        pdb_path: str,
        allowed_aas,
        num_seq: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        fixed: dict[int, str] | None = None,
    ) -> MPNNDesigns:
        """Sample sequences restricted to *allowed_aas*.

        Scores are mean per-residue negative log-likelihoods -- the same
        quantity ProteinMPNN writes as ``score=`` in its FASTA headers -- so
        lower is better and the numbers stay comparable with V9 output.
        Duplicate sequences are collapsed to their best score, which matters
        at small alphabets where MPNN repeats itself often and every duplicate
        would otherwise cost a fold.

        *fixed* maps 0-based positions to the residue that must appear there.
        Those positions are excluded from design three ways over: the chain
        mask tells the model not to redesign them, the bias forbids anything
        else there, and the sampled string is checked and corrected afterwards.
        The belt-and-braces is deliberate -- ``chain_mask`` semantics have
        moved between LigandMPNN revisions, and this pipeline cannot verify
        which one is installed until it runs. If the correction ever has work
        to do, that is reported once rather than passing silently, because it
        means the installed revision does not honour the mask as assumed and
        every score in the run is computed over the wrong set of positions.

        Note that scores with *fixed* positions are means over the **designed**
        positions only, since the fixed ones are not predictions. That is the
        right denominator for comparing alphabets within a site-aware run, but
        it is not directly comparable with a site-blind run's scores.
        """
        import torch

        if self.model is None:
            self.load()

        allowed = list(allowed_aas)
        if not allowed:
            return MPNNDesigns(999.0, [])

        num_seq = self.cfg.mpnn_num_seq if num_seq is None else num_seq
        temperature = self.cfg.mpnn_temp if temperature is None else temperature

        data_utils = self._mods["data_utils"]
        get_score = data_utils.get_score
        restype_int_to_str = data_utils.restype_int_to_str

        feature_dict = self.features_for(pdb_path)
        length = int(feature_dict["X"].shape[1])

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        fixed = {int(k): v for k, v in (fixed or {}).items() if 0 <= int(k) < length}
        feature_dict["bias"] = self._bias_tensor(allowed, length, fixed)
        feature_dict["temperature"] = float(temperature)
        feature_dict["symmetry_residues"] = [[]]
        feature_dict["symmetry_weights"] = [[]]
        feature_dict["chain_mask"] = self._chain_mask_for(pdb_path, length, fixed)

        score_mask = feature_dict["mask"] * feature_dict["chain_mask"]

        best: dict[str, float] = {}
        scores: list[float] = []
        batch = max(1, min(num_seq, self.cfg.mpnn_batch_size))
        started = time.time()

        remaining = num_seq
        with torch.inference_mode():
            while remaining > 0:
                this_batch = min(batch, remaining)
                remaining -= this_batch
                feature_dict["batch_size"] = this_batch
                feature_dict["randn"] = torch.randn(
                    [this_batch, feature_dict["mask"].shape[1]], device=self.device
                )
                try:
                    out = self.model.sample(feature_dict)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if batch == 1:
                        raise
                    # Halve and retry rather than lose the whole call.
                    remaining += this_batch
                    batch = max(1, batch // 2)
                    print(f"    MPNN OOM; retrying with batch_size={batch}")
                    continue

                loss, _ = get_score(out["S"], out["log_probs"], score_mask)
                loss_np = loss.detach().to("cpu").numpy()
                s_np = out["S"].detach().to("cpu").numpy()
                for i in range(s_np.shape[0]):
                    seq = "".join(restype_int_to_str[int(a)] for a in s_np[i])
                    if fixed:
                        seq, n_forced = _enforce_fixed(seq, fixed)
                        if n_forced:
                            self.n_forced_positions += n_forced
                            if not self.warned_chain_mask:
                                self.warned_chain_mask = True
                                print(
                                    "    WARNING: this LigandMPNN revision "
                                    "redesigned positions that were marked "
                                    "fixed; the sequences are being corrected "
                                    "here, but MPNN scores are then computed "
                                    "over a different position set than "
                                    "intended. Treat the scores, not the "
                                    "structures, as suspect."
                                )
                    score = float(loss_np[i])
                    scores.append(score)
                    if seq not in best or score < best[seq]:
                        best[seq] = score

        self.seconds += time.time() - started
        self.n_calls += 1
        self.n_sequences += len(scores)

        pairs = sorted((sc, sq) for sq, sc in best.items())
        mean_score = statistics.mean(scores) if scores else 999.0
        return MPNNDesigns(mean_score, pairs)

    def stats(self) -> str:
        per_call = (self.seconds / self.n_calls) if self.n_calls else 0.0
        extra = ""
        if self.n_forced_positions:
            extra = (
                f", {self.n_forced_positions} fixed-position corrections "
                "(chain_mask not honoured)"
            )
        return (
            f"{self.n_calls} MPNN calls, {self.n_sequences} sequences, "
            f"{self.seconds:.1f} s total ({per_call * 1000:.0f} ms/call){extra}"
        )
