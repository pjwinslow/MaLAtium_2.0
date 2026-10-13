"""Tests for the Colab helpers: GPU recommendations, input gathering, results."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redaa import colab as C  # noqa: E402
from redaa import fetch as F  # noqa: E402
from tests.test_fetch import _StubHTTP, _two_chain_pdb, _with_stub  # noqa: E402


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def test_no_gpu_is_reported_as_a_blocker():
    rec = C.recommend(C.GpuInfo(available=False))
    assert rec.env == {}
    assert any("Change runtime type" in w for w in rec.warnings)


def test_t4_is_warned_about_bf16_and_capped_on_length():
    t4 = C.GpuInfo(True, "Tesla T4", 15.0, (7, 5), bf16=False)
    rec = C.recommend(t4)
    assert rec.env["FOLD_MAX_RESIDUES"] == "320"
    assert any("bfloat16" in w for w in rec.warnings)
    assert any("FOLD_PRECISION" in w for w in rec.warnings)


def test_a100_gets_the_full_length_budget_and_no_bf16_warning():
    a100 = C.GpuInfo(True, "NVIDIA A100-SXM4-40GB", 40.0, (8, 0), bf16=True)
    rec = C.recommend(a100)
    assert rec.env["FOLD_MAX_RESIDUES"] == "1024"
    assert not any("bfloat16" in w for w in rec.warnings)
    # A Colab session is hours, so the MSI replicate count is not the default.
    assert rec.env["NUM_REPLICATES"] == "3"
    assert int(rec.env["MAX_WALL_SECONDS"]) > 0
    assert rec.env["AUTO_RESUME"] == "1"


def test_l4_sits_between_the_two():
    l4 = C.GpuInfo(True, "NVIDIA L4", 22.5, (8, 9), bf16=True)
    assert C.recommend(l4).env["FOLD_MAX_RESIDUES"] == "512"


def test_gpu_label_is_human_readable():
    t4 = C.GpuInfo(True, "Tesla T4", 15.0, (7, 5), bf16=False)
    assert "Tesla T4" in t4.label and "sm_75" in t4.label and "bf16 NO" in t4.label
    assert C.GpuInfo().label == "no GPU"


def test_apply_env_does_not_clobber_an_explicit_setting():
    key = "REDAA_TEST_APPLY_ENV"
    os.environ[key] = "mine"
    try:
        assert C.apply_env({key: "theirs"}) == {}
        assert os.environ[key] == "mine"
        assert C.apply_env({key: "theirs"}, override=True) == {key: "theirs"}
        assert os.environ[key] == "theirs"
    finally:
        os.environ.pop(key, None)


def test_use_drive_refuses_when_drive_is_not_mounted():
    try:
        C.use_drive("/definitely/not/mounted/redaa")
    except RuntimeError as exc:
        assert "mount Drive first" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_use_drive_points_everything_at_the_given_base():
    tmp = tempfile.mkdtemp()
    saved = {k: os.environ.get(k) for k in
             ("OUTPUT_ROOT", "FOLD_CACHE_PATH", "HF_HOME")}
    try:
        base = os.path.join(tmp, "redaa")
        env = C.use_drive(base)
        assert env["OUTPUT_ROOT"] == os.path.join(base, "outputs")
        assert env["FOLD_CACHE_PATH"] == os.path.join(base, "fold_cache.sqlite")
        assert env["HF_HOME"] == os.path.join(base, "hf")
        assert os.path.isdir(env["OUTPUT_ROOT"])
        assert os.path.isdir(env["HF_HOME"])
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Input gathering
# ---------------------------------------------------------------------------


def test_gather_inputs_fetches_and_prepares_several_specs():
    stub = _StubHTTP({
        "https://files.rcsb.org/download/1ABC.pdb": _two_chain_pdb().encode(),
        "https://files.rcsb.org/download/2DEF.pdb": _two_chain_pdb().encode(),
    })
    tmp = tempfile.mkdtemp()
    try:
        reports = _with_stub(
            stub,
            lambda: C.gather_inputs("1abc_A, 2def_B", input_dir=tmp, verbose=False),
        )
        assert [r["name"] for r in reports] == ["1ABC_A", "2DEF_B"]
        assert [r["chain"] for r in reports] == ["A", "B"]
        assert all(os.path.exists(r["path"]) for r in reports)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gather_inputs_applies_the_chain_override_only_where_needed():
    stub = _StubHTTP({
        "https://files.rcsb.org/download/1ABC.pdb": _two_chain_pdb().encode(),
        "https://files.rcsb.org/download/2DEF.pdb": _two_chain_pdb().encode(),
    })
    tmp = tempfile.mkdtemp()
    try:
        reports = _with_stub(
            stub,
            lambda: C.gather_inputs(
                "1abc 2def_A", input_dir=tmp, chain="B", verbose=False
            ),
        )
        by_name = {r["name"]: r for r in reports}
        assert by_name["1ABC_B"]["chain"] == "B"   # override applied
        assert by_name["2DEF_A"]["chain"] == "A"   # explicit chain respected
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gather_inputs_skips_failures_and_keeps_going():
    stub = _StubHTTP({
        "https://files.rcsb.org/download/2DEF.pdb": _two_chain_pdb().encode(),
    })
    tmp = tempfile.mkdtemp()
    try:
        reports = _with_stub(
            stub,
            lambda: C.gather_inputs(
                "1abc, 2def, notanid", input_dir=tmp, verbose=False
            ),
        )
        # 1abc 404s and 'notanid' will not parse; 2def still gets prepared.
        assert [r["name"] for r in reports] == ["2DEF"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gather_inputs_accepts_local_files_and_directories():
    tmp = tempfile.mkdtemp()
    try:
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        for name in ("one.pdb", "two.pdb", "notes.txt"):
            with open(os.path.join(src, name), "w") as fh:
                fh.write(_two_chain_pdb() if name.endswith(".pdb") else "ignore me")

        out = os.path.join(tmp, "inputs")
        reports = C.gather_inputs(files=[src], input_dir=out, verbose=False)
        assert sorted(r["name"] for r in reports) == ["one", "two"]
        assert all(r["source"] == "file" for r in reports)

        single = C.gather_inputs(
            files=[os.path.join(src, "one.pdb")],
            input_dir=os.path.join(tmp, "inputs2"),
            chain="B",
            verbose=False,
        )
        assert single[0]["chain"] == "B"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_describe_inputs_renders_a_table():
    stub = _StubHTTP({
        "https://files.rcsb.org/download/1ABC.pdb": _two_chain_pdb().encode(),
    })
    tmp = tempfile.mkdtemp()
    try:
        reports = _with_stub(
            stub, lambda: C.gather_inputs("1abc", input_dir=tmp, verbose=False)
        )
        table = C.describe_inputs(reports)
        lines = table.splitlines()
        assert lines[0].split()[:3] == ["name", "source", "chain"]
        assert set(lines[1]) <= {"-", " "}
        assert "1ABC" in lines[2]
        assert C.describe_inputs([]) == "no inputs prepared"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def test_zip_results_includes_outputs_and_excludes_the_fold_cache():
    tmp = tempfile.mkdtemp()
    try:
        run_dir = os.path.join(tmp, "outputs", "run_20260101_000000")
        os.makedirs(os.path.join(run_dir, "domA"))
        with open(os.path.join(run_dir, "BATCH_SUMMARY.csv"), "w") as fh:
            fh.write("pdb\ndomA\n")
        with open(os.path.join(run_dir, "fold_cache.sqlite"), "wb") as fh:
            fh.write(b"\x00" * 2048)
        with open(os.path.join(run_dir, "domA", "domain_info.txt"), "w") as fh:
            fh.write("info")

        import zipfile

        path = C.zip_results(run_dir)
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        assert any(n.endswith("BATCH_SUMMARY.csv") for n in names)
        assert any(n.endswith("domA/domain_info.txt") for n in names)
        assert not any(n.endswith(".sqlite") for n in names)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_latest_run_dir_and_summary_rows():
    tmp = tempfile.mkdtemp()
    try:
        root = os.path.join(tmp, "outputs")
        assert C.latest_run_dir(root) is None
        older = os.path.join(root, "run_a")
        newer = os.path.join(root, "run_b")
        os.makedirs(older)
        os.makedirs(newer)
        os.utime(older, (1_600_000_000, 1_600_000_000))
        os.utime(newer, (1_700_000_000, 1_700_000_000))
        assert C.latest_run_dir(root) == newer

        with open(os.path.join(newer, "BATCH_SUMMARY.csv"), "w") as fh:
            fh.write("pdb,pruned_alphabet\n1ABC_A,GLV\n")
        rows = C.summary_rows(newer)
        assert rows == [{"pdb": "1ABC_A", "pruned_alphabet": "GLV"}]
        assert C.summary_rows(older) == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_fetch_only_cli_prepares_inputs_and_exits_zero():
    import run_redaa

    stub = _StubHTTP({
        "https://files.rcsb.org/download/1ABC.pdb": _two_chain_pdb().encode(),
    })
    tmp = tempfile.mkdtemp()
    try:
        code = _with_stub(
            stub,
            lambda: run_redaa.main(
                ["--fetch", "1abc_A", "--fetch-only", "--input-dir", tmp]
            ),
        )
        assert code == 0
        assert os.path.exists(os.path.join(tmp, "1ABC_A.pdb"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fetch_only_cli_fails_when_nothing_could_be_prepared():
    import run_redaa

    stub = _StubHTTP({})
    tmp = tempfile.mkdtemp()
    try:
        code = _with_stub(
            stub,
            lambda: run_redaa.main(
                ["--fetch", "1abc", "--fetch-only", "--input-dir", tmp]
            ),
        )
        assert code == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_diagnose_reports_a_bare_environment_as_problems_not_a_crash():
    """The diagnostic must survive the very situation it exists to explain."""
    report = C.diagnose()
    assert set(report) == {"ok", "problems", "info"}
    assert report["info"]["python"]
    # torch is absent in this test environment, which is a problem, not a crash.
    assert any("torch" in p for p in report["problems"])
    assert "redaa package imports cleanly" in report["ok"]
    assert isinstance(report["info"]["env"], dict)


def test_diagnose_names_the_fix_for_each_missing_piece():
    saved = {k: os.environ.get(k) for k in ("MPNN_REPO", "MPNN_CHECKPOINT")}
    try:
        os.environ["MPNN_REPO"] = "/nope"
        os.environ["MPNN_CHECKPOINT"] = "/nope/ckpt.pt"
        report = C.diagnose()
        mpnn = [p for p in report["problems"] if "LigandMPNN" in p]
        ckpt = [p for p in report["problems"] if "checkpoint" in p]
        assert mpnn and "Step 1" in mpnn[0], mpnn
        assert ckpt and "Step 1" in ckpt[0], ckpt
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_diagnose_flags_a_truncated_checkpoint_separately_from_a_missing_one():
    tmp = tempfile.mkdtemp()
    saved = os.environ.get("MPNN_CHECKPOINT")
    try:
        path = os.path.join(tmp, "proteinmpnn_v_48_020.pt")
        with open(path, "wb") as fh:
            fh.write(b"\x00" * 128)      # a failed download, not a model
        os.environ["MPNN_CHECKPOINT"] = path
        report = C.diagnose()
        hit = [p for p in report["problems"] if "truncated" in p]
        assert hit, report["problems"]
        assert "delete it" in hit[0]
    finally:
        if saved is None:
            os.environ.pop("MPNN_CHECKPOINT", None)
        else:
            os.environ["MPNN_CHECKPOINT"] = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_stale_imports_is_empty_for_a_consistent_session():
    assert C.stale_imports() == []


def test_stale_imports_catches_a_package_swapped_under_the_kernel():
    """The Colab failure mode where pip replaces an already-imported package.

    Nothing reports a version mismatch at the time; you get an ABI error or a
    nonsense AttributeError somewhere unrelated. Detecting it is the only way
    to tell the user to restart rather than keep debugging.
    """
    import numpy

    real = numpy.__version__
    try:
        numpy.__version__ = "0.0.1-not-what-is-installed"
        stale = C.stale_imports()
        names = [t[0] for t in stale]
        assert "numpy" in names, stale
        entry = next(t for t in stale if t[0] == "numpy")
        assert entry[1] == "0.0.1-not-what-is-installed"
        assert entry[2] == real

        report = C.diagnose()
        hit = [p for p in report["problems"] if "Restart session" in p]
        assert hit, report["problems"]
        assert "do NOT" in hit[0] or "Do NOT" in hit[0]
    finally:
        numpy.__version__ = real


def test_print_diagnosis_emits_every_section():
    import contextlib, io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        C.print_diagnosis()
    out = buf.getvalue()
    for section in ("=== environment ===", "run configuration",
                    "checks passed", "PROBLEMS"):
        assert section in out, section
    assert "python" in out


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
