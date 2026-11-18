"""The batch scripts, checked against the code they drive.

These are shell scripts, so nothing imports them and no test run exercises
them -- which is how they fell a long way behind. The committed batch job
still pinned MPNN_CHECKPOINT to the ProteinMPNN file (the exact mistake fixed
in the Colab path, which silently hands a ligand_mpnn run the wrong weights),
knew nothing about SEARCH_MODE or any SITE_* setting, and defaulted to letting
ten array tasks each try to download 25 GB of model weights.

A queue wait is a slow way to discover a stale script, so the checks here are
the cheap ones that catch drift: every pipeline variable a script sets really
is one the configuration reads, and the specific mistakes already made are
asserted against by name.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARRAY = os.path.join(REPO_ROOT, "slurm", "redaa_array.slurm")
PREPARE = os.path.join(REPO_ROOT, "slurm", "prepare_msi.sh")

# Prefixes that belong to this pipeline rather than to a driver or a tool.
OURS = (
    "SITE_", "FOLD_", "MPNN_", "CHUNK_", "NUM_", "SEARCH_", "MAX_WALL",
    "AUTO_", "INPUT_", "OUTPUT_", "DO_", "MIN_", "USE_", "SCREEN_",
    "PRUNE_", "PRUNING_", "RUN_NAME", "RESUME_", "SAVE_", "BASELINE_",
    "DYNAMIC_", "ESM_", "EXHAUSTIVE_", "NATIVE_",
)


def _read(path: str) -> str:
    with open(path) as fh:
        return fh.read()


def _exported(text: str) -> set[str]:
    return set(re.findall(r"export ([A-Z][A-Z0-9_]+)=", text))


def test_both_scripts_are_valid_bash():
    for path in (ARRAY, PREPARE):
        result = subprocess.run(
            ["bash", "-n", path], capture_output=True, text=True
        )
        assert result.returncode == 0, f"{path}: {result.stderr}"


def test_every_pipeline_variable_set_is_one_the_config_reads():
    """Catches a renamed or mistyped setting, which would otherwise be ignored.

    An unrecognised name in a batch script is silent: the run proceeds on the
    default and the result looks fine.
    """
    from redaa.config import ENV_VARS

    known = set(ENV_VARS)
    for path in (ARRAY, PREPARE):
        ours = {
            name for name in _exported(_read(path))
            if name.startswith(OURS)
        }
        unknown = sorted(ours - known)
        assert not unknown, f"{path} sets unknown settings: {unknown}"


def test_the_array_job_does_not_pin_the_mpnn_checkpoint():
    """The checkpoint must come from MPNN_MODEL_TYPE.

    protein_mpnn, ligand_mpnn and soluble_mpnn load different weights with
    different atom_context_num. Pinning one means an ATP run quietly gets the
    ProteinMPNN file and dies in load_state_dict, with an error about tensor
    shapes that never mentions the checkpoint.
    """
    text = _read(ARRAY)
    assert "export MPNN_MODEL_TYPE=" in text
    assert "proteinmpnn_v_48_020.pt" not in text, (
        "the array job pins a checkpoint again"
    )
    # Mentioning it to say it is deliberately unset is fine; exporting is not.
    assert "export MPNN_CHECKPOINT=" not in text


def test_the_array_job_is_offline_by_default():
    """Ten tasks each discovering a 25 GB download is not a plan.

    Compute nodes usually cannot reach the internet at all, so the weights are
    staged by prepare_msi.sh and the job runs offline.
    """
    text = _read(ARRAY)
    match = re.search(r'export HF_HUB_OFFLINE="\$\{HF_HUB_OFFLINE:-(\d)\}"', text)
    assert match, "HF_HUB_OFFLINE is not set with a default"
    assert match.group(1) == "1", "the array job defaults to online"


def test_the_array_job_exposes_the_search_modes_and_site_settings():
    text = _read(ARRAY)
    for mode in ("alphabet_only", "site_aware", "scaffold_then_site"):
        assert mode in text, mode
    for name in (
        "SEARCH_MODE", "SITE_PRESET", "SITE_SPEC", "SITE_RESIDUE_POLICY",
        "SITE_GATE", "SITE_MAX_BB_RMSD", "SITE_MAX_LOCAL_RMSD",
        "SITE_MAX_CB_DEV", "SITE_KEEP_LIGANDS",
    ):
        assert f"export {name}=" in text, name
    # Both policies have to be findable by someone reading the script.
    assert "exempt" in text and "in_alphabet" in text


def test_the_array_job_checks_its_settings_before_loading_models():
    """A bad combination otherwise wastes the whole allocation."""
    text = _read(ARRAY)
    assert "--print-config" in text
    config_at = text.index("--print-config")
    run_at = text.index("srun python run_redaa.py")
    assert config_at < run_at


def test_the_array_job_still_handles_the_checkpointed_exit():
    text = _read(ARRAY)
    assert "--signal=B:USR1@300" in text
    assert "MAX_WALL_SECONDS" in text
    assert 'status" -eq 2' in text


def test_the_fold_cache_is_per_task_by_default():
    """A shared SQLite cache over a parallel filesystem is a gamble.

    WAL mode expects POSIX locking that a networked filesystem may not provide
    faithfully across nodes. The saving from sharing is real, so the script
    explains the trade rather than hiding it -- but the default is safe.
    """
    text = _read(ARRAY)
    export_line = next(
        line for line in text.splitlines()
        if line.startswith("export FOLD_CACHE_PATH=")
    )
    assert "SLURM_ARRAY_TASK_ID" in export_line, export_line
    # And the reason is written down, so nobody "optimises" it back.
    assert "locking" in text


def test_prepare_stages_every_checkpoint_the_model_types_need():
    """Not a hardcoded list: it has drifted from the code once already."""
    text = _read(PREPARE)
    assert "DEFAULT_CHECKPOINTS" in text
    assert "CHECKPOINT_BASE_URL" in text
    # And it must not reintroduce hardcoded filenames.
    assert "proteinmpnn_v_48_020.pt" not in text


def test_prepare_handles_what_a_compute_node_cannot():
    text = _read(PREPARE)
    # The three things that need the network.
    assert "snapshot_download" in text          # ~25 GB of weights
    assert "model_dependency_plan" in text      # the model's own imports
    assert "--fetch-only" in text               # input structures
    # And it must point at the one step it cannot do: a GPU fold.
    assert "--preflight" in text
    assert "srun" in text
    # The data file that only appears on the first fold is called out, since
    # caching the weights does not cover it.
    assert "first fold" in text.lower() or "FIRST FOLD" in text


def test_prepare_does_not_guess_the_torch_build():
    """Installing the wrong CUDA build is a slow way to find out.

    torch has to match the module the batch script loads, and only the user
    knows which that is.
    """
    text = _read(PREPARE)
    assert "pytorch.org" in text
    assert "index-url" in text
    # It checks rather than installs.
    assert "torch is not importable" in text


def main():
    failures = []
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc!r}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
