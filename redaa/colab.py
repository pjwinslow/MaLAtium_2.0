"""Helpers for running the search in Google Colab.

Colab differs from an MSI batch node in ways that matter here:

* **The GPU you get is not the GPU you asked for.**  A free session is usually
  a T4, which is Turing and has no bfloat16 -- and ESMFold2's ESM-C backbone
  loads in bf16 by default.  :func:`detect_gpu` and :func:`recommend` surface
  that before you wait twenty minutes to hit it.
* **Sessions end without warning** at roughly 12 hours, sooner if idle.  The
  pipeline already checkpoints per replicate and caches every fold, so the fix
  is to put the output directory and the fold cache on Drive and let a
  reconnected session resume.
* **Disk is ephemeral and small.**  Model weights are the bulk of it; keeping
  the HuggingFace cache on Drive makes a reconnect cheap.

Nothing in this module is Colab-specific at import time, so it is importable
and testable anywhere.
"""

from __future__ import annotations

import os
import shutil
import sys
import zipfile
from dataclasses import dataclass, field

from . import fetch as F


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass
class GpuInfo:
    available: bool = False
    name: str = "none"
    total_gb: float = 0.0
    capability: tuple[int, int] = (0, 0)
    bf16: bool = False

    @property
    def label(self) -> str:
        if not self.available:
            return "no GPU"
        return (
            f"{self.name} ({self.total_gb:.0f} GB, "
            f"sm_{self.capability[0]}{self.capability[1]}, "
            f"bf16 {'yes' if self.bf16 else 'NO'})"
        )


def detect_gpu() -> GpuInfo:
    """Describe the attached accelerator, or report its absence."""
    try:
        import torch
    except ImportError:
        return GpuInfo()
    if not torch.cuda.is_available():
        return GpuInfo()
    props = torch.cuda.get_device_properties(0)
    try:
        bf16 = bool(torch.cuda.is_bf16_supported())
    except Exception:
        bf16 = props.major >= 8
    return GpuInfo(
        available=True,
        name=props.name,
        total_gb=props.total_memory / 1024**3,
        capability=(props.major, props.minor),
        bf16=bf16,
    )


@dataclass
class Recommendation:
    env: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def recommend(gpu: GpuInfo | None = None) -> Recommendation:
    """Suggest settings for the GPU this session actually got.

    Returns environment-variable overrides plus plain-language notes.  These
    are starting points, not measurements -- the comments say which way each
    one errs.
    """
    gpu = detect_gpu() if gpu is None else gpu
    rec = Recommendation()

    if not gpu.available:
        rec.warnings.append(
            "No GPU attached. Runtime > Change runtime type > T4 GPU (or "
            "better) -- ESMFold on CPU is minutes per sequence, which makes "
            "even the smoke test impractical."
        )
        return rec

    rec.notes.append(f"GPU: {gpu.label}")

    if not gpu.bf16:
        rec.warnings.append(
            f"{gpu.name} has no bfloat16 (needs sm_80+). ESMFold2's ESM-C "
            "backbone loads in bf16 by default, so this may fail or fall back "
            "to a slow path. If loading errors out, set FOLD_PRECISION to "
            "'fp16' or 'fp32', or pick a smaller ESMFold2 variant below. An "
            "L4 or A100 runtime avoids the issue entirely."
        )

    # Length cap, erring low: a fold that OOMs costs the time it ran for and
    # then still has to be skipped.
    if gpu.total_gb < 17:
        rec.env["FOLD_MAX_RESIDUES"] = "320"
        rec.warnings.append(
            f"Only {gpu.total_gb:.0f} GB of VRAM (a T4 or V100 class card). "
            "Capping inputs at 320 residues; larger domains will be skipped "
            "rather than run until they OOM. Raise FOLD_MAX_RESIDUES if your "
            "domains fit."
        )
    elif gpu.total_gb < 26:
        rec.env["FOLD_MAX_RESIDUES"] = "512"
        rec.notes.append("Capping inputs at 512 residues for this VRAM budget.")
    else:
        rec.env["FOLD_MAX_RESIDUES"] = "1024"

    # A Colab session is hours, not days, so the MSI defaults (10 replicates,
    # 9 final folds) do not fit. Fewer replicates with the same gate is the
    # right trade: you get real answers for a few domains instead of a
    # truncated first pass over many.
    rec.env["NUM_REPLICATES"] = "3"
    rec.env["CHUNK_SIZE"] = "100"
    rec.env["NUM_SEQS_TO_FOLD"] = "5"
    rec.notes.append(
        "Defaulting to 3 replicates and 5 final folds -- a Colab session is "
        "hours, not days. Raise NUM_REPLICATES if you have a long session and "
        "want MSI-comparable statistics."
    )

    # Stop with 20 minutes to spare so the summaries get written even if the
    # session is about to be reclaimed.
    rec.env["MAX_WALL_SECONDS"] = "39600"
    rec.env["AUTO_RESUME"] = "1"
    return rec


def apply_env(env: dict[str, str], override: bool = False) -> dict[str, str]:
    """Push recommendations into ``os.environ``, reporting what was set.

    Existing values win unless *override*, so a setting you typed into a form
    cell is never silently replaced by a recommendation.
    """
    applied = {}
    for key, value in env.items():
        if override or key not in os.environ:
            os.environ[key] = str(value)
            applied[key] = str(value)
    return applied


def hf_cache_dir() -> str:
    """Where huggingface_hub will actually put model weights."""
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(key):
            return os.environ[key]
    if os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "hub")
    return os.path.expanduser("~/.cache/huggingface/hub")


def on_drive(path: str) -> bool:
    """Whether *path* lives on a mounted Google Drive."""
    return os.path.abspath(path).startswith("/content/drive")


def use_drive(
    base: str = "/content/drive/MyDrive/redaa",
    hf_cache: bool = False,
) -> dict[str, str]:
    """Point outputs and the fold cache at Google Drive.

    Call this *after* mounting Drive.  Surviving a disconnect is the point:
    completed replicates are checkpointed and every fold is cached, so a
    reconnected session resumes instead of starting over.

    Model weights deliberately do **not** go to Drive.  safetensors reads
    weights by memory-mapping them, and Drive's FUSE mount supports neither
    mmap nor reliable multi-gigabyte writes, so a model cached there fails to
    load with ``SafetensorError: Error while deserializing header: header too
    large`` -- the first eight bytes come back as garbage.  Re-downloading a
    few gigabytes after a restart is the cheaper problem, so ``hf_cache``
    defaults to False and should stay there.
    """
    if not os.path.isdir(os.path.dirname(base.rstrip("/"))):
        raise RuntimeError(
            f"{base} is not reachable -- mount Drive first:\n"
            "  from google.colab import drive; drive.mount('/content/drive')"
        )
    os.makedirs(base, exist_ok=True)
    env = {
        "OUTPUT_ROOT": os.path.join(base, "outputs"),
        "FOLD_CACHE_PATH": os.path.join(base, "fold_cache.sqlite"),
    }
    if hf_cache:
        env["HF_HOME"] = os.path.join(base, "hf")
        print(
            "WARNING: hf_cache=True puts model weights on Drive, where "
            "safetensors cannot mmap them. Expect "
            "'SafetensorError: header too large' at load time."
        )
    elif on_drive(hf_cache_dir()):
        # A previous session may have pointed it at Drive; undo that, because
        # anything already cached there is unreadable.
        env["HF_HOME"] = "/content/hf"
        print(
            "Model cache was pointing at Drive, which cannot serve "
            "memory-mapped weights -- moved it to /content/hf.\n"
            "Run Step 8 with REPAIR_HF_CACHE if a model fails to load."
        )
    for path in (env["OUTPUT_ROOT"], env.get("HF_HOME", "")):
        if path:
            os.makedirs(path, exist_ok=True)
    return apply_env(env, override=True)


# ---------------------------------------------------------------------------
# Model cache integrity
# ---------------------------------------------------------------------------

_SAFETENSORS_HEADER_MAX = 512 * 1024 * 1024   # no real header is this big


def check_safetensors(path: str) -> tuple[bool, str]:
    """Validate a ``.safetensors`` file's header without loading it.

    The format is an 8-byte little-endian header length, then that many bytes
    of JSON, then the tensor data.  A truncated or FUSE-mangled file shows up
    immediately as a header length that cannot fit in the file -- which is
    exactly the "header too large" that safetensors reports, except here we
    can say which file and why.
    """
    import json
    import struct

    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return False, f"unreadable: {exc}"
    if size < 9:
        return False, f"only {size} bytes -- download never started"

    try:
        with open(path, "rb") as fh:
            raw = fh.read(8)
            if len(raw) < 8:
                return False, "could not read the 8-byte header length"
            (header_len,) = struct.unpack("<Q", raw)
            if header_len == 0:
                return False, "header length is zero"
            if header_len > _SAFETENSORS_HEADER_MAX or header_len + 8 > size:
                return False, (
                    f"header claims {header_len} bytes but the file is {size} "
                    "-- truncated, or read through a mount that cannot serve it"
                )
            header = fh.read(header_len)
            if len(header) < header_len:
                return False, "header is shorter than it claims"
            json.loads(header.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"header does not parse: {exc}"
    return True, "ok"


def scan_hf_cache(cache_dir: str | None = None) -> list[tuple[str, str]]:
    """Return ``(path, reason)`` for every broken weight file in the cache."""
    cache_dir = cache_dir or hf_cache_dir()
    broken: list[tuple[str, str]] = []
    if not os.path.isdir(cache_dir):
        return broken
    for root, _dirs, names in os.walk(cache_dir):
        for name in names:
            if not name.endswith(".safetensors"):
                continue
            path = os.path.join(root, name)
            if os.path.islink(path) and not os.path.exists(path):
                broken.append((path, "dangling symlink"))
                continue
            ok, reason = check_safetensors(path)
            if not ok:
                broken.append((path, reason))
    return broken


def repair_hf_cache(
    model_id: str | None = None, cache_dir: str | None = None
) -> list[str]:
    """Delete cached entries so the next load re-downloads them.

    With *model_id*, removes just that model's directory; otherwise removes
    every model directory containing a broken weight file.  Returns what was
    removed.
    """
    cache_dir = cache_dir or hf_cache_dir()
    if not os.path.isdir(cache_dir):
        return []

    removed: list[str] = []
    if model_id:
        slug = "models--" + model_id.replace("/", "--")
        target = os.path.join(cache_dir, slug)
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
            removed.append(target)
        return removed

    for path, _reason in scan_hf_cache(cache_dir):
        # Walk up to the models--owner--name directory and drop the lot; a
        # partially repaired snapshot is worse than no snapshot.
        current = path
        while current.startswith(cache_dir) and not os.path.basename(
            current
        ).startswith("models--"):
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        target = current if os.path.basename(current).startswith("models--") else path
        if target not in removed and os.path.exists(target):
            shutil.rmtree(target, ignore_errors=True) if os.path.isdir(
                target
            ) else os.remove(target)
            removed.append(target)
    return removed


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def gather_inputs(
    specs: str = "",
    files: list[str] | None = None,
    input_dir: str = "/content/redaa_inputs",
    chain: str | None = None,
    cache_dir: str | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Turn accessions and/or local files into pipeline-ready input PDBs.

    *specs* is free text -- newline-, comma- or space-separated, ``#`` comments
    allowed -- in any form :func:`redaa.fetch.parse_spec` accepts. *files* are
    paths already on disk (uploaded, or in a Drive folder); they go through the
    same cleaning so an uploaded raw RCSB entry behaves like a fetched one.

    *chain* overrides the chain for every spec that does not name one itself,
    which is what you want for a list of entries you know are homodimers.

    Returns one report dict per prepared input; failures are reported and
    skipped rather than aborting the batch, so one dead accession in a list of
    twenty does not cost you the run.
    """
    os.makedirs(input_dir, exist_ok=True)
    cache_dir = cache_dir or os.path.join(input_dir, "_raw")

    parsed: list[F.StructureSpec] = []
    errors: list[str] = []

    if specs.strip():
        tokens, spec_errors = F.parse_spec_list(specs)
        errors.extend(spec_errors)
        for token in tokens:
            if chain and token.chain is None and token.source != "cath":
                token.chain = chain
            parsed.append(token)

    for path in files or []:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if name.lower().endswith((".pdb", ".ent", ".cif", ".mmcif")):
                    parsed.append(F.parse_spec(os.path.join(path, name)))
            continue
        try:
            spec = F.parse_spec(path)
        except ValueError as exc:
            errors.append(f"{path}: {exc}")
            continue
        if chain:
            spec.chain = chain
        parsed.append(spec)

    reports: list[dict] = []
    for spec in parsed:
        try:
            report = F.fetch_and_prepare(spec, input_dir, cache_dir=cache_dir)
        except (F.FetchError, ValueError) as exc:
            errors.append(f"{spec.raw}: {exc}")
            if verbose:
                print(f"  SKIP  {spec.raw}: {exc}")
            continue
        reports.append(report)
        if verbose:
            extra = []
            if report["dropped_incomplete"]:
                extra.append(f"{report['dropped_incomplete']} incomplete dropped")
            if report["gaps"]:
                extra.append(f"{report['gaps']} chain gap(s)")
            suffix = f"  [{', '.join(extra)}]" if extra else ""
            print(
                f"  ok    {report['name']}: chain {report['chain']}, "
                f"{report['n_residues']} residues{suffix}"
            )

    if verbose:
        print(f"\n{len(reports)} input(s) ready in {input_dir}")
        if errors:
            print(f"{len(errors)} failed:")
            for line in errors:
                print(f"  {line}")
    return reports


def describe_inputs(reports: list[dict]) -> str:
    """A small fixed-width table of prepared inputs, for a notebook cell."""
    if not reports:
        return "no inputs prepared"
    rows = [("name", "source", "chain", "residues", "dropped", "gaps")]
    for r in reports:
        rows.append(
            (
                r["name"], r["source"], r["chain"],
                str(r["n_residues"]), str(r["dropped_incomplete"]), str(r["gaps"]),
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    out = []
    for i, row in enumerate(rows):
        out.append("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def zip_results(run_dir: str, out_path: str | None = None) -> str:
    """Zip a run directory for download, leaving the fold cache behind.

    The cache can be far larger than the results and is worth keeping on Drive
    rather than pulling through the browser.
    """
    if not os.path.isdir(run_dir):
        raise RuntimeError(f"not a directory: {run_dir}")
    out_path = out_path or f"{run_dir.rstrip('/')}_results.zip"
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, names in os.walk(run_dir):
            for name in names:
                if name.endswith((".sqlite", ".sqlite-wal", ".sqlite-shm")):
                    continue
                full = os.path.join(root, name)
                zf.write(full, os.path.relpath(full, os.path.dirname(run_dir)))
    return out_path


def latest_run_dir(output_root: str | None = None) -> str | None:
    """Most recently modified run directory under *output_root*."""
    output_root = output_root or os.environ.get("OUTPUT_ROOT", "outputs")
    if not os.path.isdir(output_root):
        return None
    candidates = [
        os.path.join(output_root, name)
        for name in os.listdir(output_root)
        if os.path.isdir(os.path.join(output_root, name))
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def summary_rows(run_dir: str) -> list[dict]:
    """Read ``BATCH_SUMMARY.csv`` from *run_dir*."""
    import csv

    path = os.path.join(run_dir, "BATCH_SUMMARY.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def disk_free_gb(path: str = "/content") -> float:
    """Free space in GB, falling back to the filesystem root off Colab."""
    for candidate in (path, os.getcwd(), "/"):
        try:
            return shutil.disk_usage(candidate).free / 1024**3
        except OSError:
            continue
    return float("nan")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

# Packages whose version changing under a running kernel breaks things in
# confusing ways. Colab preinstalls all of them, so a `pip install --upgrade`
# can swap one out from under code that already imported it; the symptom is
# never "wrong version", it is an ABI error, a bogus AttributeError or a
# segfault somewhere unrelated.
_CORE_PACKAGES = (
    "numpy", "torch", "transformers", "tokenizers", "accelerate",
    "huggingface_hub",
)


def stale_imports() -> list[tuple[str, str, str]]:
    """Packages imported at one version but installed at another.

    Returns ``(name, imported_version, installed_version)`` triples.  A
    non-empty result means the kernel must be restarted before anything else
    is believable -- this is the Colab failure mode where ``pip install`` has
    replaced a package the session had already loaded.
    """
    import importlib.metadata as md

    out: list[tuple[str, str, str]] = []
    for name in _CORE_PACKAGES:
        module = sys.modules.get(name)
        if module is None:
            continue
        imported = getattr(module, "__version__", None)
        if not imported:
            continue
        try:
            installed = md.version(name)
        except Exception:
            continue
        if installed != imported:
            out.append((name, imported, installed))
    return out


def diagnose(try_model_load: bool = False) -> dict:
    """Collect everything needed to explain a failed run, in one place.

    ``try_model_load=True`` additionally attempts to construct both engines,
    which is the slow part but is what distinguishes "the environment is
    wrong" from "the model will not load on this GPU".
    """
    import importlib
    import importlib.metadata as md
    import platform

    report: dict = {"ok": [], "problems": [], "info": {}}

    def info(key, value):
        report["info"][key] = value

    def ok(line):
        report["ok"].append(line)

    def problem(line):
        report["problems"].append(line)

    info("python", platform.python_version())
    info("platform", platform.platform())

    # -- torch and the GPU ------------------------------------------------
    try:
        import torch

        info("torch", torch.__version__)
        info("torch_cuda_build", getattr(torch.version, "cuda", None))
        if torch.cuda.is_available():
            gpu = detect_gpu()
            info("gpu", gpu.label)
            info("gpu_name", gpu.name)
            info("gpu_total_gb", round(gpu.total_gb, 1))
            info("gpu_capability", "sm_%d%d" % gpu.capability)
            info("bf16", gpu.bf16)
            ok("CUDA is available: " + gpu.label)
            try:
                free, total = torch.cuda.mem_get_info()
                info("gpu_free_gb", round(free / 1024**3, 1))
                if free / total < 0.5:
                    problem(
                        "Only %.1f of %.1f GB of VRAM is free -- something else "
                        "is holding it. Runtime > Restart session."
                        % (free / 1024**3, total / 1024**3)
                    )
            except Exception:
                pass
        else:
            problem(
                "torch reports no CUDA device. Runtime > Change runtime type > "
                "GPU, then re-run Step 1."
            )
    except Exception as exc:  # noqa: BLE001
        problem("torch is not importable: %r" % (exc,))

    # -- the rest of the stack --------------------------------------------
    for name, required in (
        ("transformers", True),
        ("accelerate", True),
        ("huggingface_hub", True),
        ("numpy", True),
        ("tmtools", False),
        ("prody", False),
        ("ml_collections", False),
    ):
        try:
            info(name, md.version(name.replace("_", "-")))
            ok("%s %s" % (name, report["info"][name]))
        except Exception:
            try:
                info(name, md.version(name))
                ok("%s %s" % (name, report["info"][name]))
            except Exception:
                (problem if required else ok)(
                    "%s is not installed%s"
                    % (name, " -- re-run Step 1" if required else " (optional)")
                )

    # -- the thing that silently poisons a Colab session -------------------
    stale = stale_imports()
    info("stale_imports", stale)
    if stale:
        detail = ", ".join(
            "%s imported %s but installed %s" % t for t in stale
        )
        problem(
            "A package was replaced after this kernel imported it (%s). "
            "Runtime > Restart session, then re-run from Step 2 -- do NOT "
            "re-run the pip install." % detail
        )

    # -- this repository ---------------------------------------------------
    redaa_ok = True
    for module in (
        "redaa.config", "redaa.structure", "redaa.fetch",
        "redaa.folding", "redaa.mpnn", "redaa.pipeline",
    ):
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            redaa_ok = False
            problem("cannot import %s: %r" % (module, exc))
    if redaa_ok:
        ok("redaa package imports cleanly")

    # -- LigandMPNN --------------------------------------------------------
    mpnn_repo = os.environ.get("MPNN_REPO", "")
    ckpt = os.environ.get("MPNN_CHECKPOINT", "")
    info("MPNN_REPO", mpnn_repo or "(unset)")
    info("MPNN_CHECKPOINT", ckpt or "(unset)")
    if not mpnn_repo or not os.path.exists(os.path.join(mpnn_repo, "data_utils.py")):
        problem(
            "LigandMPNN not found at MPNN_REPO=%r -- re-run Step 1"
            % (mpnn_repo,)
        )
    else:
        ok("LigandMPNN checkout present")
    if not ckpt or not os.path.exists(ckpt):
        problem("MPNN checkpoint missing at %r -- re-run Step 1" % (ckpt,))
    elif os.path.getsize(ckpt) < 1_000_000:
        problem(
            "MPNN checkpoint is only %d bytes, so the download was truncated "
            "-- delete it and re-run Step 1" % os.path.getsize(ckpt)
        )
    else:
        ok("MPNN checkpoint %.0f MB" % (os.path.getsize(ckpt) / 1e6))

    # -- model weight cache ------------------------------------------------
    cache = hf_cache_dir()
    info("hf_cache", cache)
    if on_drive(cache):
        problem(
            "The model cache is on Google Drive (%s). safetensors reads "
            "weights by memory-mapping them and Drive's FUSE mount cannot "
            "serve that, which fails as 'SafetensorError: header too large'. "
            "Re-run Step 3 to move it to /content/hf, then Step 8 with "
            "REPAIR_HF_CACHE to clear what was cached there." % cache
        )
    broken = scan_hf_cache(cache)
    info("hf_cache_broken", [b[0] for b in broken])
    if broken:
        for path, reason in broken[:5]:
            problem("corrupt weight file %s -- %s" % (path, reason))
        problem(
            "Run Step 8 with REPAIR_HF_CACHE to delete the broken entries so "
            "the next load re-downloads them."
        )
    elif os.path.isdir(cache):
        ok("model cache at %s has no corrupt weight files" % cache)

    # -- run configuration -------------------------------------------------
    from .config import ENV_VARS

    info("env", {k: os.environ[k] for k in ENV_VARS if k in os.environ})
    info("disk_free_gb", round(disk_free_gb("/content"), 1))

    # -- optionally load the models ---------------------------------------
    if try_model_load:
        from .config import RedAAConfig

        cfg = RedAAConfig()
        try:
            from .mpnn import MPNNEngine

            MPNNEngine(cfg, verbose=False).load()
            ok("LigandMPNN loads")
        except Exception as exc:  # noqa: BLE001
            problem("LigandMPNN failed to load: %r" % (exc,))
        try:
            from .folding import FoldEngine

            FoldEngine(cfg, verbose=False).load()
            ok("%s loads" % cfg.fold_model_id)
        except Exception as exc:  # noqa: BLE001
            problem("%s failed to load: %r" % (cfg.fold_model_id, exc))

    return report


def print_diagnosis(try_model_load: bool = False) -> dict:
    """Human-readable :func:`diagnose`, for pasting into a bug report."""
    report = diagnose(try_model_load=try_model_load)

    print("=== environment ===")
    for key, value in report["info"].items():
        if key in ("env", "stale_imports"):
            continue
        print("  %-18s %s" % (key, value))

    env = report["info"].get("env") or {}
    print("\n=== run configuration (%d set) ===" % len(env))
    for key in sorted(env):
        print("  %-24s %s" % (key, env[key]))

    print("\n=== checks passed (%d) ===" % len(report["ok"]))
    for line in report["ok"]:
        print("  ok    " + line)

    if report["problems"]:
        print("\n=== PROBLEMS (%d) ===" % len(report["problems"]))
        for line in report["problems"]:
            print("  !!    " + line)
    else:
        print("\nNo problems detected.")
    return report


# ---------------------------------------------------------------------------
# Run logging
# ---------------------------------------------------------------------------
#
# Colab truncates long cell output and drops it entirely on a disconnect, so
# the console is the worst possible place to keep the record of a run. Worse,
# the output you most need is the output of a run that crashed -- which is
# exactly when you are least likely to still have the cell contents.
#
# capture_run_log() tees everything to a file while still printing live, adds
# the environment and the traceback, and leaves behind one attachable report.


class _Tee:
    """Write to the real stream and to a file at the same time."""

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, data):
        self._stream.write(data)
        try:
            self._handle.write(data)
        except Exception:
            pass          # a failed log write must never break the run
        return len(data)

    def flush(self):
        try:
            self._stream.flush()
        finally:
            try:
                self._handle.flush()
            except Exception:
                pass

    def isatty(self):
        return bool(getattr(self._stream, "isatty", lambda: False)())

    def __getattr__(self, name):
        return getattr(self._stream, name)


def log_dir() -> str:
    """Where to keep run logs: beside the outputs, so Drive keeps them."""
    root = os.environ.get("OUTPUT_ROOT", "/content/redaa_outputs")
    path = os.path.join(os.path.dirname(root.rstrip("/")) or root, "logs")
    os.makedirs(path, exist_ok=True)
    return path


def _env_block() -> list[str]:
    report = diagnose()
    lines = ["=== environment ==="]
    for key, value in report["info"].items():
        if key in ("env", "hf_cache_broken"):
            continue
        lines.append("  %-18s %s" % (key, value))

    env = report["info"].get("env") or {}
    lines.append("")
    lines.append("=== run configuration (%d set) ===" % len(env))
    for key in sorted(env):
        lines.append("  %-24s %s" % (key, env[key]))

    lines.append("")
    lines.append("=== checks ===")
    for line in report["ok"]:
        lines.append("  ok    " + line)
    for line in report["problems"]:
        lines.append("  !!    " + line)
    return lines


def write_report(
    log_path: str | None = None,
    run_dir: str | None = None,
    out_path: str | None = None,
    max_log_lines: int = 1500,
) -> str:
    """Bundle the environment, the console log and the results into one file.

    This is the file to attach to a bug report or drop into a chat.  A very
    long console log is elided in the middle rather than dropped, because the
    interesting parts of a failed run are the setup at the top and the
    traceback at the bottom; the untouched log stays on disk beside it.
    """
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_path or os.path.join(log_dir(), "report_%s.txt" % stamp)

    parts: list[str] = [
        "RedAA finder run report",
        "generated %s" % datetime.now().isoformat(timespec="seconds"),
        "",
    ]
    parts += _env_block()

    run_dir = run_dir or latest_run_dir()
    if run_dir and os.path.isdir(run_dir):
        parts += ["", "=== run directory ===", "  " + run_dir]
        rows = summary_rows(run_dir)
        parts.append("")
        parts.append("=== BATCH_SUMMARY.csv (%d row(s)) ===" % len(rows))
        keep = [
            "pdb", "replicate", "status", "final_alphabet", "alphabet_size",
            "pruned_alphabet", "pruned_alphabet_size", "best_seq_tm",
            "best_seq_rmsd", "best_seq_plddt", "elapsed_seconds",
        ]
        for row in rows:
            parts.append(
                "  " + "  ".join(
                    "%s=%s" % (k, row[k]) for k in keep if row.get(k)
                )
            )
        flagged = os.path.join(run_dir, "FLAGGED_DOMAINS.csv")
        if os.path.exists(flagged):
            parts += ["", "=== FLAGGED_DOMAINS.csv ==="]
            with open(flagged) as fh:
                parts += ["  " + ln.rstrip() for ln in fh]

    if log_path and os.path.exists(log_path):
        with open(log_path, errors="replace") as fh:
            lines = fh.read().splitlines()
        parts += ["", "=== console log: %s (%d lines) ===" % (log_path, len(lines))]
        if len(lines) <= max_log_lines:
            parts += lines
        else:
            head = max_log_lines // 2
            tail = max_log_lines - head
            parts += lines[:head]
            parts.append(
                "... %d lines elided; the complete log is at %s ..."
                % (len(lines) - max_log_lines, log_path)
            )
            parts += lines[-tail:]

    with open(out_path, "w") as fh:
        fh.write("\n".join(parts) + "\n")
    return out_path


class capture_run_log:
    """Context manager that records a run to disk as well as to the cell.

    On the way out it writes an attachable report whether the block succeeded
    or raised, and prints where both files went.  A traceback is written into
    the log *before* being re-raised, so a crash is captured rather than lost.
    """

    def __init__(self, path: str | None = None, quiet: bool = False):
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = path or os.path.join(log_dir(), "run_%s.log" % stamp)
        self.report_path: str | None = None
        self.quiet = quiet
        self._handle = None
        self._saved: tuple = ()

    def __enter__(self):
        from datetime import datetime

        self._handle = open(self.path, "w", buffering=1)
        self._handle.write(
            "RedAA run log, started %s\n\n"
            % datetime.now().isoformat(timespec="seconds")
        )
        self._handle.write("\n".join(_env_block()))
        self._handle.write("\n\n=== console ===\n")
        self._saved = (sys.stdout, sys.stderr)
        sys.stdout = _Tee(sys.stdout, self._handle)
        sys.stderr = _Tee(sys.stderr, self._handle)
        return self

    def __exit__(self, exc_type, exc, tb):
        import traceback

        if exc is not None and self._handle is not None:
            try:
                self._handle.write("\n=== traceback ===\n")
                self._handle.write(
                    "".join(traceback.format_exception(exc_type, exc, tb))
                )
            except Exception:
                pass

        if self._saved:
            sys.stdout, sys.stderr = self._saved
        if self._handle is not None:
            try:
                self._handle.flush()
                self._handle.close()
            except Exception:
                pass
            self._handle = None

        try:
            self.report_path = write_report(log_path=self.path)
        except Exception as report_exc:  # noqa: BLE001
            print("could not write the report: %r" % (report_exc,))

        if not self.quiet:
            size = (
                os.path.getsize(self.report_path) / 1024
                if self.report_path and os.path.exists(self.report_path)
                else 0
            )
            print("\n" + "-" * 70)
            if exc is not None:
                print("The run raised %s -- it is captured in the files below."
                      % exc_type.__name__)
            print("Full log : %s" % self.path)
            print("Report   : %s  (%.0f KB)" % (self.report_path, size))
            print("Attach the report to a chat or a bug report; it carries the")
            print("environment, the settings, the results so far and the log.")
            print("-" * 70)
        return False   # never swallow the exception


def latest_report() -> str | None:
    """Newest report file written by :func:`write_report`."""
    folder = log_dir()
    reports = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.startswith("report_") and name.endswith(".txt")
    ]
    return max(reports, key=os.path.getmtime) if reports else None
