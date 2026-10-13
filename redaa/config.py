"""Configuration for the reduced-alphabet search.

Every knob is overridable from the environment so a SLURM array job can be
retargeted without editing the source.  ``RedAAConfig.describe()`` prints the
resolved values into the job log, which is what you want when you come back to
a six-month-old output directory and need to know what produced it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

ALL_AAS = tuple("ACDEFGHIKLMNPQRSTVWY")


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class RedAAConfig:
    """Resolved run configuration."""

    # ---------------- input / output ----------------
    input_pdb_dir: str = field(
        default_factory=lambda: _env_str("INPUT_PDB_DIR", "input_pdbs")
    )
    output_root: str = field(default_factory=lambda: _env_str("OUTPUT_ROOT", "outputs"))
    run_name: str = field(default_factory=lambda: _env_str("RUN_NAME", ""))
    chunk_index: int = field(default_factory=lambda: _env_int("CHUNK_INDEX", 0))
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 10))
    resume_from: str = field(default_factory=lambda: _env_str("RESUME_FROM", ""))
    auto_resume: bool = field(default_factory=lambda: _env_bool("AUTO_RESUME", True))

    # ---------------- ESMFold2-Fast ----------------
    fold_model_id: str = field(
        default_factory=lambda: _env_str("FOLD_MODEL_ID", "Synthyra/ESMFold2-Fast")
    )
    fold_precision: str = field(
        default_factory=lambda: _env_str("FOLD_PRECISION", "auto")
    )
    # ESMFold2's structure module is a diffusion sampler, so the step count is
    # a direct speed/quality dial.  Screening folds -- the overwhelming
    # majority -- only ever need to tell "right fold" from "wrong fold", so
    # they run short; the folds that end up in the published numbers run long.
    fold_steps_screen: int = field(
        default_factory=lambda: _env_int("FOLD_STEPS_SCREEN", 50)
    )
    fold_steps_final: int = field(
        default_factory=lambda: _env_int("FOLD_STEPS_FINAL", 200)
    )
    fold_loops: int = field(default_factory=lambda: _env_int("FOLD_LOOPS", 1))
    fold_seed: int = field(default_factory=lambda: _env_int("FOLD_SEED", 7))
    fold_batch_size: int = field(default_factory=lambda: _env_int("FOLD_BATCH_SIZE", 4))
    fold_max_residues: int = field(
        default_factory=lambda: _env_int("FOLD_MAX_RESIDUES", 1024)
    )
    use_torch_compile: bool = field(
        default_factory=lambda: _env_bool("USE_TORCH_COMPILE", False)
    )

    # ---------------- LigandMPNN ----------------
    # Path to a LigandMPNN checkout (contains data_utils.py / model_utils.py).
    mpnn_repo: str = field(default_factory=lambda: _env_str("MPNN_REPO", ""))
    mpnn_checkpoint: str = field(
        default_factory=lambda: _env_str("MPNN_CHECKPOINT", "")
    )
    mpnn_model_type: str = field(
        default_factory=lambda: _env_str("MPNN_MODEL_TYPE", "protein_mpnn")
    )
    mpnn_num_seq: int = field(default_factory=lambda: _env_int("MPNN_NUM_SEQ", 25))
    mpnn_batch_size: int = field(
        default_factory=lambda: _env_int("MPNN_BATCH_SIZE", 25)
    )
    mpnn_temp: float = field(default_factory=lambda: _env_float("MPNN_TEMP", 0.5))
    mpnn_baseline_temp: float = field(
        default_factory=lambda: _env_float("MPNN_BASELINE_TEMP", 0.25)
    )

    # ---------------- search ----------------
    num_replicates: int = field(default_factory=lambda: _env_int("NUM_REPLICATES", 10))
    starting_alphabet: tuple[str, ...] = ("A", "G")
    use_hydrophobic_seeding: bool = field(
        default_factory=lambda: _env_bool("USE_HYDROPHOBIC_SEEDING", True)
    )
    dynamic_seeding: bool = field(
        default_factory=lambda: _env_bool("DYNAMIC_SEEDING", False)
    )
    dynamic_seed_size: int = field(
        default_factory=lambda: _env_int("DYNAMIC_SEED_SIZE", 2)
    )
    baseline_tolerance: float = field(
        default_factory=lambda: _env_float("BASELINE_TOLERANCE", 1.50)
    )
    esm_start_after_n_aas: int = field(
        default_factory=lambda: _env_int("ESM_START_AFTER_N_AAS", 3)
    )

    # Two-stage candidate screening.  V9 folded `num_seqs_quick_validate`
    # sequences for all ~17 candidates every round.  Here every candidate gets
    # a cheap first look, then only the most promising few get the full
    # per-candidate budget -- the decision is made on the same evidence for
    # the candidates that could actually win, at roughly half the folds.
    screen_folds_stage1: int = field(
        default_factory=lambda: _env_int("SCREEN_FOLDS_STAGE1", 1)
    )
    screen_folds_stage2: int = field(
        default_factory=lambda: _env_int("SCREEN_FOLDS_STAGE2", 3)
    )
    screen_stage2_keep: int = field(
        default_factory=lambda: _env_int("SCREEN_STAGE2_KEEP", 5)
    )
    num_seqs_to_fold: int = field(
        default_factory=lambda: _env_int("NUM_SEQS_TO_FOLD", 9)
    )
    max_identity_for_fold: float = field(
        default_factory=lambda: _env_float("MAX_IDENTITY_FOR_FOLD", 0.70)
    )

    # ---------------- acceptance ----------------
    min_tm_score: float = field(default_factory=lambda: _env_float("MIN_TM_SCORE", 0.51))
    use_plddt_gate: bool = field(
        default_factory=lambda: _env_bool("USE_PLDDT_GATE", False)
    )
    min_plddt_disorder: float = field(
        default_factory=lambda: _env_float("MIN_PLDDT_DISORDER", 30.0)
    )
    max_rmsd_threshold: float = field(
        default_factory=lambda: _env_float("MAX_RMSD_THRESHOLD", 3.0)
    )
    native_plddt_min: float = field(
        default_factory=lambda: _env_float("NATIVE_PLDDT_MIN", 50.0)
    )
    native_tm_min: float = field(
        default_factory=lambda: _env_float("NATIVE_TM_MIN", 0.5)
    )

    # ---------------- backward pruning ----------------
    do_backward_pruning: bool = field(
        default_factory=lambda: _env_bool("DO_BACKWARD_PRUNING", True)
    )
    pruning_min_size: int = field(
        default_factory=lambda: _env_int("PRUNING_MIN_SIZE", 3)
    )
    exhaustive_pruning_threshold: int = field(
        default_factory=lambda: _env_int("EXHAUSTIVE_PRUNING_THRESHOLD", 8)
    )
    # Downward closure: if alphabet S cannot encode the fold, assume no subset
    # of S can either, and skip those subsets.  This is the Apriori argument
    # and it collapses most of the exhaustive search.  Set False for a fully
    # unconditional sweep.
    pruning_downward_closure: bool = field(
        default_factory=lambda: _env_bool("PRUNING_DOWNWARD_CLOSURE", True)
    )
    prune_folds: int = field(default_factory=lambda: _env_int("PRUNE_FOLDS", 3))

    # ---------------- caching / housekeeping ----------------
    fold_cache_path: str = field(
        default_factory=lambda: _env_str("FOLD_CACHE_PATH", "")
    )
    use_fold_cache: bool = field(
        default_factory=lambda: _env_bool("USE_FOLD_CACHE", True)
    )
    save_fold_pdbs: bool = field(
        default_factory=lambda: _env_bool("SAVE_FOLD_PDBS", False)
    )
    # Wall-clock budget in seconds.  The run stops between replicates once it
    # is exceeded, so a preempted or time-limited job exits with a usable
    # checkpoint instead of being killed mid-write.
    max_wall_seconds: int = field(
        default_factory=lambda: _env_int("MAX_WALL_SECONDS", 0)
    )
    test_mode: bool = False

    def apply_test_mode(self) -> None:
        """Shrink every budget for a ~5 minute single-domain smoke test."""
        self.test_mode = True
        self.num_replicates = 1
        self.chunk_index = 0
        self.chunk_size = 1
        self.mpnn_num_seq = 50
        self.num_seqs_to_fold = 2
        self.screen_folds_stage1 = 1
        self.screen_folds_stage2 = 1
        self.screen_stage2_keep = 2
        self.prune_folds = 1
        self.fold_steps_screen = min(self.fold_steps_screen, 25)
        self.fold_steps_final = min(self.fold_steps_final, 50)

    def describe(self) -> str:
        lines = []
        for f in fields(self):
            lines.append(f"  {f.name:<30s} {getattr(self, f.name)!r}")
        return "\n".join(lines)
