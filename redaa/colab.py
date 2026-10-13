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


def use_drive(
    base: str = "/content/drive/MyDrive/redaa",
    hf_cache: bool = True,
) -> dict[str, str]:
    """Point outputs, the fold cache and model downloads at Google Drive.

    Call this *after* mounting Drive.  Surviving a disconnect is the whole
    point: completed replicates are checkpointed and every fold is cached, so
    a reconnected session resumes instead of starting over.
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
    for path in (env["OUTPUT_ROOT"], env.get("HF_HOME", "")):
        if path:
            os.makedirs(path, exist_ok=True)
    return apply_env(env, override=True)


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
