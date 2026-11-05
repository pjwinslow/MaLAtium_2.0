"""Tests for the Colab notebook.

A Colab runtime restart wipes ``/content``.  Before the bootstrap existed, a
cell that had worked ten minutes earlier would fail with a bare
``ModuleNotFoundError: No module named 'redaa'`` and no hint that the fix was
to re-run the install cell.  These tests pin down the three things that stops:

* every post-install cell carries the bootstrap,
* the bootstrap repairs sys.path, the working directory and the environment
  when the checkout is present,
* and detects, reports and re-clones when it is not.

Plus the structural invariants: the cells are valid Python, Colab's form
widgets still render, and the committed notebook matches its generator.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

NOTEBOOK = os.path.join(REPO_ROOT, "colab", "RedAA_Colab.ipynb")
BUILDER = os.path.join(REPO_ROOT, "colab", "build_notebook.py")


def _cells():
    with open(NOTEBOOK) as fh:
        return json.load(fh)["cells"]


def _code_cells():
    return [
        "".join(c["source"]) for c in _cells() if c["cell_type"] == "code"
    ]


def _bootstrap_source(cell: str) -> str:
    """The bootstrap block, lifted out of a cell."""
    start = cell.index("# --- bootstrap")
    end = cell.index("# ---------", start + 10)
    return cell[start : cell.index("\n", end) + 1]


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_notebook_is_valid_and_declares_a_gpu():
    with open(NOTEBOOK) as fh:
        nb = json.load(fh)
    assert nb["nbformat"] == 4
    assert nb["metadata"]["accelerator"] == "GPU"
    assert len(nb["cells"]) >= 8


def test_every_code_cell_is_valid_python():
    for i, src in enumerate(_code_cells()):
        try:
            ast.parse(src)
        except SyntaxError as exc:
            raise AssertionError(f"cell {i} line {exc.lineno}: {exc.msg}") from exc


def test_param_blocks_stay_above_the_first_real_statement():
    """Colab only renders a form for #@param lines that precede other code."""
    for i, src in enumerate(_code_cells()):
        lines = src.splitlines()
        first_real = next(
            (
                j
                for j, line in enumerate(lines)
                if line.strip()
                and not line.strip().startswith("#")
                and "#@param" not in line
            ),
            len(lines),
        )
        for j, line in enumerate(lines):
            if "#@param" in line:
                assert j < first_real, (
                    f"cell {i}: #@param on line {j} comes after code on "
                    f"line {first_real}; its widget would not render"
                )


def test_notebook_matches_its_generator():
    """Guards against hand-editing the JSON instead of the builder.

    Compares the committed notebook with what the builder produces in memory,
    rather than shelling out to git, so this does not depend on the working
    tree being clean.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_redaa_nb_builder", BUILDER)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)   # module-level guard means no file write

    with open(NOTEBOOK) as fh:
        on_disk = json.load(fh)
    generated = builder.build()

    assert len(generated["cells"]) == len(on_disk["cells"]), (
        f"builder makes {len(generated['cells'])} cells, "
        f"the notebook has {len(on_disk['cells'])} -- regenerate with "
        "`python colab/build_notebook.py`"
    )
    for i, (want, have) in enumerate(zip(generated["cells"], on_disk["cells"])):
        assert want["cell_type"] == have["cell_type"], i
        assert "".join(want["source"]) == "".join(have["source"]), (
            f"cell {i} differs from the builder's output -- edit "
            "colab/build_notebook.py and regenerate, do not edit the JSON"
        )


# ---------------------------------------------------------------------------
# The bootstrap
# ---------------------------------------------------------------------------


def test_install_cell_has_no_bootstrap_but_verifies_itself():
    install = _code_cells()[0]
    assert "Step 1" in install
    assert "# --- bootstrap" not in install  # it *is* the installer
    # A partial install must not report success.
    assert "INSTALL INCOMPLETE" in install
    assert "raise SystemExit" in install
    for checked in ("redaa.config", "data_utils.py", "transformers"):
        assert checked in install, checked


def test_every_cell_after_the_installer_carries_the_bootstrap():
    cells = _code_cells()
    assert len(cells) >= 7
    for i, src in enumerate(cells[1:], start=1):
        title = next(
            (l for l in src.splitlines() if l.startswith("#@title")), f"cell {i}"
        )
        assert "# --- bootstrap" in src, f"{title} has no bootstrap"
        assert "import redaa.colab as rc" in src, f"{title} does not import rc"


def test_bootstrap_repairs_path_cwd_and_env_when_the_checkout_exists():
    boot = _bootstrap_source(_code_cells()[1])
    tmp = tempfile.mkdtemp(prefix="redaa_nb_")
    cwd = os.getcwd()
    saved = {k: os.environ.get(k) for k in ("MPNN_REPO", "MPNN_CHECKPOINT")}
    try:
        repo = os.path.join(tmp, "MaLAtium_2.0")
        shutil.copytree(os.path.join(REPO_ROOT, "redaa"), os.path.join(repo, "redaa"))
        for key in saved:
            os.environ.pop(key, None)

        scope: dict = {}
        exec(
            boot.replace('"/content/MaLAtium_2.0"', repr(repo)).replace(
                '"/content/LigandMPNN"', repr(os.path.join(tmp, "LigandMPNN"))
            ),
            scope,
        )

        assert repo in sys.path
        assert os.path.realpath(os.getcwd()) == os.path.realpath(repo)
        assert os.environ["MPNN_REPO"] == os.path.join(tmp, "LigandMPNN")
        assert os.environ["MPNN_CHECKPOINT"].endswith("proteinmpnn_v_48_020.pt")
        assert hasattr(scope["rc"], "detect_gpu")
    finally:
        os.chdir(cwd)
        while repo in sys.path:
            sys.path.remove(repo)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(tmp, ignore_errors=True)


def test_bootstrap_does_not_overwrite_env_the_install_cell_set():
    boot = _bootstrap_source(_code_cells()[1])
    tmp = tempfile.mkdtemp(prefix="redaa_nb_env_")
    cwd = os.getcwd()
    saved = os.environ.get("MPNN_REPO")
    try:
        repo = os.path.join(tmp, "MaLAtium_2.0")
        shutil.copytree(os.path.join(REPO_ROOT, "redaa"), os.path.join(repo, "redaa"))
        os.environ["MPNN_REPO"] = "/my/own/ligandmpnn"
        exec(boot.replace('"/content/MaLAtium_2.0"', repr(repo)), {})
        assert os.environ["MPNN_REPO"] == "/my/own/ligandmpnn"
    finally:
        os.chdir(cwd)
        while repo in sys.path:
            sys.path.remove(repo)
        if saved is None:
            os.environ.pop("MPNN_REPO", None)
        else:
            os.environ["MPNN_REPO"] = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_bootstrap_detects_a_wiped_runtime_and_reclones():
    """The case that produced the bare ModuleNotFoundError."""
    boot = _bootstrap_source(_code_cells()[1])
    tmp = tempfile.mkdtemp(prefix="redaa_nb_wipe_")
    cwd = os.getcwd()
    repo = os.path.join(tmp, "MaLAtium_2.0")   # deliberately absent
    calls: list[list[str]] = []
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd) if isinstance(cmd, (list, tuple)) else [cmd])
        if isinstance(cmd, (list, tuple)) and cmd[0] == "git":
            # Stand in for a successful clone.
            shutil.copytree(
                os.path.join(REPO_ROOT, "redaa"), os.path.join(repo, "redaa")
            )
        return real_run([sys.executable, "-c", ""], capture_output=True)

    printed: list[str] = []
    try:
        subprocess.run = fake_run
        scope = {"print": lambda *a, **k: printed.append(" ".join(map(str, a)))}
        exec(boot.replace('"/content/MaLAtium_2.0"', repr(repo)), scope)

        message = "\n".join(printed)
        assert "runtime was restarted" in message, message
        assert "Step 1" in message, message       # tells them what to do next
        assert any(c[0] == "git" and "clone" in c for c in calls), calls
        clone = next(c for c in calls if c[0] == "git")
        assert "--branch" in clone and "--depth" in clone
        assert hasattr(scope["rc"], "detect_gpu")   # importable again
    finally:
        subprocess.run = real_run
        os.chdir(cwd)
        while repo in sys.path:
            sys.path.remove(repo)
        shutil.rmtree(tmp, ignore_errors=True)


def test_bootstrap_clone_uses_the_pinned_branch_by_default():
    """With no globals from Step 1, the branch must still be baked in."""
    boot = _bootstrap_source(_code_cells()[1])
    assert "__BRANCH__" not in boot, "placeholder was not substituted"
    assert "__REPO_URL__" not in boot
    assert "github.com" in boot
    assert 'globals().get("BRANCH"' in boot


def test_run_cell_refuses_to_start_without_inputs_configured():
    # Found by title, not step number: inserting a step renumbers the rest and
    # a number-keyed lookup would silently start checking the wrong cell.
    run_cell = next(c for c in _code_cells() if "Run the search" in c)
    assert "INPUT_PDB_DIR" in run_cell
    assert "raise SystemExit" in run_cell
    # TEST_MODE comes from the settings cell; a restart clears it.
    assert 'globals().get("TEST_MODE"' in run_cell


def test_site_step_precedes_the_run_and_offers_both_modes():
    cells = _code_cells()
    site_idx = next(
        i for i, c in enumerate(cells) if "Mode and functional site" in c
    )
    run_idx = next(i for i, c in enumerate(cells) if "Run the search" in c)
    assert site_idx < run_idx, "the site must be configured before the run"

    site = cells[site_idx]
    # Every mode reachable from the form, and the presets offered by name.
    for mode in ("alphabet_only", "site_aware", "scaffold_then_site"):
        assert mode in site, mode
    for key in ("atp", "tev"):
        assert f'"{key}"' in site, key
    assert "rc.configure_site(" in site
    # The site is resolved and shown against the real inputs before any GPU
    # time is spent, which is the point of having the step at all.
    assert "rc.preview_sites(" in site


def test_inputs_step_can_keep_ligands_and_load_a_preset():
    inputs = next(c for c in _code_cells() if "Fetch and/or upload inputs" in c)
    assert "KEEP_LIGANDS" in inputs
    assert "keep_ligands=" in inputs
    assert "rc.preset(" in inputs


def test_results_step_shows_site_columns_only_in_a_site_run():
    results = next(c for c in _code_cells() if "Step 8" in c)
    assert "search_mode" in results
    assert "best_seq_site_bb_rmsd" in results
    assert "graft_final_alphabet" in results
    assert "rc.describe_site_results(" in results


def test_steps_are_numbered_consecutively_from_one():
    import re as _re

    numbers = []
    for cell in _code_cells():
        m = _re.search(r"#@title Step (\d+)", cell)
        if m:
            numbers.append(int(m.group(1)))
    assert numbers == sorted(numbers), numbers
    assert numbers == list(range(1, len(numbers) + 1)), numbers


def main():
    failures = []
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            import traceback

            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc!r}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
