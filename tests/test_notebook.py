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


def _snapshot_redaa_modules() -> dict:
    """Remember which redaa modules are loaded, and from where.

    Several tests exec the notebook's bootstrap against a throwaway copy of
    the package, which caches `redaa` with its __path__ inside a temp
    directory that is then deleted. Restoring sys.path and cwd is not enough:
    a stale package entry makes a later `import redaa.<anything>` in an
    unrelated test fail with ModuleNotFoundError, so the suite's result starts
    depending on the order tests happen to run in.
    """
    return {
        name: mod for name, mod in sys.modules.items()
        if name == "redaa" or name.startswith("redaa.")
    }


def _restore_redaa_modules(saved: dict) -> None:
    for name in [
        n for n in list(sys.modules)
        if n == "redaa" or n.startswith("redaa.")
    ]:
        del sys.modules[name]
    sys.modules.update(saved)
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
    saved_modules = _snapshot_redaa_modules()
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
        # MPNN_CHECKPOINT must stay unset: the engine resolves the checkpoint
        # from MPNN_MODEL_TYPE, and pinning one here would hand a ligand_mpnn
        # run the ProteinMPNN weights -- which fails at load with a
        # tensor-shape error that says nothing about the real mistake.
        assert "MPNN_CHECKPOINT" not in os.environ
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
        _restore_redaa_modules(saved_modules)
        shutil.rmtree(tmp, ignore_errors=True)


def test_bootstrap_does_not_overwrite_env_the_install_cell_set():
    boot = _bootstrap_source(_code_cells()[1])
    tmp = tempfile.mkdtemp(prefix="redaa_nb_env_")
    cwd = os.getcwd()
    saved = os.environ.get("MPNN_REPO")
    saved_modules = _snapshot_redaa_modules()
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
        _restore_redaa_modules(saved_modules)
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
    saved_modules = _snapshot_redaa_modules()
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
        _restore_redaa_modules(saved_modules)
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
    """Step numbers must run 1..N in order, allowing lettered sub-steps.

    "6b" is deliberate: the preflight slots between settings and the run
    without renumbering everything after it (and the troubleshooting prose
    that refers to those numbers).
    """
    import re as _re

    labels = []
    for cell in _code_cells():
        m = _re.search(r"#@title Step (\d+)([a-z]?)", cell)
        if m:
            labels.append((int(m.group(1)), m.group(2)))

    assert labels == sorted(labels), labels
    majors = [n for n, suffix in labels if not suffix]
    assert majors == list(range(1, len(majors) + 1)), majors
    # A sub-step must attach to a major step that exists.
    for number, suffix in labels:
        if suffix:
            assert number in majors, (number, suffix)


def test_report_push_cell_never_leaks_the_token():
    """The optional push cell handles a credential, so it gets its own checks."""
    cell = next(c for c in _code_cells() if "Step 10" in c)
    # Read from Colab's secret store, never a literal in the notebook.
    assert 'userdata.get("GITHUB_TOKEN")' in cell
    assert "ghp_" not in cell and "github_pat_" not in cell
    # Never printed, and scrubbed out of any error text that is.
    assert "print(token" not in cell
    assert 'replace(token, "***")' in cell
    # Absent secret is a no-op with an explanation, not a crash.
    assert "if not token:" in cell
    assert "nothing was pushed" in cell
    # Only the small text artefacts leave the session.
    assert "BATCH_SUMMARY.csv" in cell
    assert "fold_cache" not in cell
    # A separate branch and a throwaway clone, so code cannot be touched.
    assert "REPORTS_BRANCH" in cell
    assert "--orphan" in cell
    assert "_report_push" in cell


def test_baked_branch_matches_this_checkout():
    """The notebook must clone the branch it was generated from.

    This is the bug that made the committed notebook unusable once: a
    hardcoded BRANCH kept pointing at the branch it was first built on, so
    Step 1 cloned a `redaa` that did not match the cells around it -- the
    notebook offered a functional-site step and the import had never heard of
    one. The failure is invisible until someone runs it in Colab, so it is
    worth a test.
    """
    import subprocess

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo, capture_output=True, text=True,
    )
    branch = out.stdout.strip()
    if out.returncode != 0 or not branch or branch == "HEAD":
        return  # not a usable git checkout; nothing to compare against

    boot = next(c for c in _code_cells() if "Step 1" in c)
    assert f'BRANCH   = "{branch}"' in boot, (
        f"notebook clones a different branch than this checkout ({branch}); "
        "re-run colab/build_notebook.py"
    )


def test_step1_can_clone_a_private_repo_without_leaking_the_token():
    """The repository is private, so an anonymous clone cannot work.

    Step 1 is the first thing anyone runs, and a bare git failure here
    ("could not read Username", or exit 128) says nothing about the cause.
    """
    boot = next(c for c in _code_cells() if "Step 1" in c)
    # Token comes from Colab's secret store, never a literal.
    assert 'userdata.get("GITHUB_TOKEN")' in boot
    assert "ghp_" not in boot and "github_pat_" not in boot
    assert "x-access-token" in boot
    # The clone command is echoed, so it must be echoed without the token.
    assert "quiet_cmd" in boot
    assert "print(GH_TOKEN" not in boot
    assert "scrub(" in boot
    # A failure explains the likely cause and the fix rather than the raw
    # git error.
    assert "GITHUB_TOKEN secret is set" in boot
    assert "Contents: Read" in boot
    # Works unauthenticated too, for when the repo is public.
    assert "if GH_TOKEN and REPO_URL.startswith" in boot


def test_step1_installs_a_checkpoint_for_every_model_type():
    """A ligand-binding run needs ligand weights, not the ProteinMPNN file.

    The ATP preset sets MPNN_MODEL_TYPE=ligand_mpnn. Those are different
    weights with a different atom_context_num, so if Step 1 only fetched
    proteinmpnn_v_48_020.pt the ATP case would die at load_state_dict with a
    tensor-shape error that never mentions the checkpoint.
    """
    from redaa.mpnn import DEFAULT_CHECKPOINTS

    boot = next(c for c in _code_cells() if "Step 1" in c)
    assert "DEFAULT_CHECKPOINTS" in boot, "Step 1 should fetch every default"
    # And it must not pin one, or the model type cannot choose.
    assert 'os.environ.pop("MPNN_CHECKPOINT"' in boot
    assert len(set(DEFAULT_CHECKPOINTS.values())) >= 2


def test_step1_makes_the_repo_importable_before_importing_from_it():
    """Ordering bug that only ever shows up in Colab.

    Step 1 clones the repo and later imports from it. If the sys.path setup
    comes after that import, the cell dies with
    "ModuleNotFoundError: No module named 'redaa'" -- and nothing in a local
    test run notices, because locally the repo is already importable.
    """
    boot = next(c for c in _code_cells() if "#@title Step 1" in c)
    lines = boot.splitlines()

    def first(pred):
        return next((i for i, l in enumerate(lines) if pred(l)), None)

    path_at = first(lambda l: "sys.path.insert(0, REPO_DIR)" in l)
    chdir_at = first(lambda l: "os.chdir(REPO_DIR)" in l)
    import_at = first(
        lambda l: l.startswith("from redaa") or l.startswith("import redaa")
    )
    assert path_at is not None, "Step 1 never puts the checkout on sys.path"
    assert import_at is not None, "expected Step 1 to import from redaa"
    assert path_at < import_at, (
        f"sys.path set at line {path_at} but redaa imported at line "
        f"{import_at}; the cell would fail with ModuleNotFoundError"
    )
    assert chdir_at is None or chdir_at < import_at

    # A re-run re-clones under already-imported modules, so stale entries
    # must be dropped or the import silently uses deleted files.
    stale_at = first(lambda l: "del sys.modules[_stale]" in l)
    assert stale_at is not None and stale_at < import_at


def test_no_cell_imports_redaa_before_the_bootstrap_runs():
    """Every later cell gets the checkout back after a runtime restart.

    A restart wipes /content, so each cell re-establishes the checkout and
    sys.path first. An import placed above that bootstrap would fail on
    exactly the restart the bootstrap exists to survive.
    """
    for cell in _code_cells():
        if "#@title Step 1" in cell:
            continue  # covered by the test above
        lines = cell.splitlines()
        import_at = next(
            (i for i, l in enumerate(lines)
             if l.startswith("from redaa") or l.startswith("import redaa")),
            None,
        )
        if import_at is None:
            continue
        boot_at = next(
            (i for i, l in enumerate(lines) if "sys.path.insert(0, _REPO)" in l),
            None,
        )
        assert boot_at is not None, (
            f"cell imports redaa at line {import_at} with no bootstrap: "
            + lines[0]
        )
        assert boot_at < import_at, lines[0]


def test_step1_runs_up_to_the_checkpoint_block_from_a_clean_interpreter():
    """Execute Step 1 the way Colab does, with nothing importable up front.

    The structural tests above check the order of lines; this one runs the
    cell. It is the only test that does, and it is here because the ordering
    bug it catches reproduced exactly as reported --
    "ModuleNotFoundError: No module named 'redaa'" -- while every other test
    passed, since locally the package is importable before the cell starts.

    The clone, pip and wget calls are stubbed; the cell is cut off at the
    checkpoint download, which is as far as it can go without the network.
    """
    src = next(c for c in _code_cells() if "#@title Step 1" in c)
    cut = src.index("CKPT_DIR = os.path.join")
    src = src[:cut]

    tmp = tempfile.mkdtemp(prefix="redaa_nb_step1_")
    repo = os.path.join(tmp, "MaLAtium_2.0")
    mpnn = os.path.join(tmp, "LigandMPNN")
    src = src.replace('REPO_DIR = "/content/MaLAtium_2.0"', f"REPO_DIR = {repo!r}")
    src = src.replace('MPNN_DIR = "/content/LigandMPNN"', f"MPNN_DIR = {mpnn!r}")

    cwd = os.getcwd()
    saved_modules = _snapshot_redaa_modules()
    saved_path = list(sys.path)
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        text = str(cmd)
        if "clone" in text and "MaLAtium" in text:
            shutil.copytree(
                os.path.join(REPO_ROOT, "redaa"), os.path.join(repo, "redaa")
            )
        elif "clone" in text:
            os.makedirs(mpnn, exist_ok=True)
            open(os.path.join(mpnn, "data_utils.py"), "w").close()

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Result()

    try:
        subprocess.run = fake_run
        # The whole point: redaa must not be importable when the cell starts.
        _restore_redaa_modules({})
        scope: dict = {"print": lambda *a, **k: None}
        exec(src, scope)

        # It got past the import that used to fail, and the values are the
        # ones the engine will resolve against.
        assert "DEFAULT_CHECKPOINTS" in scope
        assert len(set(scope["DEFAULT_CHECKPOINTS"].values())) >= 2
        assert repo in sys.path
    finally:
        subprocess.run = real_run
        os.chdir(cwd)
        sys.path[:] = saved_path
        _restore_redaa_modules(saved_modules)
        shutil.rmtree(tmp, ignore_errors=True)


def test_preflight_cell_comes_before_the_run_and_reuses_its_models():
    cells = _code_cells()
    pre_at = next(i for i, c in enumerate(cells) if "Preflight" in c)
    run_at = next(i for i, c in enumerate(cells) if "Run the search" in c)
    assert pre_at < run_at, "preflight must come before the run"

    pre = cells[pre_at]
    assert ".preflight()" in pre
    # It must stop the user rather than let them start a long run anyway.
    assert "raise SystemExit" in pre
    # And hand the loaded models on, or it would double the model-load cost.
    assert "_REDAA_ENGINES" in pre
    assert "_REDAA_ENGINES" in cells[run_at]


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
