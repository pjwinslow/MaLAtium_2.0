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
        assert os.path.isdir(env["OUTPUT_ROOT"])
        # Model weights are deliberately NOT sent to Drive; see
        # test_use_drive_keeps_model_weights_off_drive_by_default.
        assert "HF_HOME" not in env

        opted_in = C.use_drive(base, hf_cache=True)
        assert opted_in["HF_HOME"] == os.path.join(base, "hf")
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


# ---------------------------------------------------------------------------
# Model cache integrity
# ---------------------------------------------------------------------------


def _safetensors(path, header_len=None, header=None, data=b"\x00" * 4):
    import json, struct

    header = header if header is not None else json.dumps(
        {"t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    n = len(header) if header_len is None else header_len
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", n) + header + data)
    return path


def test_check_safetensors_accepts_a_well_formed_file():
    tmp = tempfile.mkdtemp()
    try:
        assert C.check_safetensors(_safetensors(os.path.join(tmp, "m.safetensors"))) == (
            True, "ok",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_check_safetensors_reproduces_the_header_too_large_failure():
    """The exact failure seen on Colab, with a message that says why."""
    tmp = tempfile.mkdtemp()
    try:
        path = _safetensors(
            os.path.join(tmp, "m.safetensors"), header_len=10**12, header=b"junk"
        )
        ok, reason = C.check_safetensors(path)
        assert not ok
        assert "header claims" in reason and "truncated" in reason
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_check_safetensors_catches_truncation_and_bad_json():
    tmp = tempfile.mkdtemp()
    try:
        empty = os.path.join(tmp, "a.safetensors")
        open(empty, "wb").write(b"abc")
        ok, reason = C.check_safetensors(empty)
        assert not ok and "never started" in reason

        zero = _safetensors(os.path.join(tmp, "b.safetensors"), header_len=0, header=b"")
        ok, reason = C.check_safetensors(zero)
        assert not ok and "zero" in reason

        junk = _safetensors(
            os.path.join(tmp, "c.safetensors"), header=b"not json at all"
        )
        ok, reason = C.check_safetensors(junk)
        assert not ok and "does not parse" in reason

        missing = os.path.join(tmp, "nope.safetensors")
        assert C.check_safetensors(missing)[0] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_scan_and_repair_remove_the_whole_broken_model_directory():
    tmp = tempfile.mkdtemp()
    try:
        hub = os.path.join(tmp, "hub")
        good_dir = os.path.join(hub, "models--ok--model", "snapshots", "r1")
        bad_dir = os.path.join(hub, "models--broken--model", "snapshots", "r1")
        _safetensors(os.path.join(good_dir, "model.safetensors"))
        _safetensors(
            os.path.join(bad_dir, "model.safetensors"), header_len=10**12, header=b"x"
        )

        broken = C.scan_hf_cache(hub)
        assert len(broken) == 1 and "broken--model" in broken[0][0]

        removed = C.repair_hf_cache(cache_dir=hub)
        assert removed == [os.path.join(hub, "models--broken--model")], removed
        # A partially repaired snapshot is worse than none, so the whole
        # model directory goes -- and the healthy model is untouched.
        assert not os.path.exists(os.path.join(hub, "models--broken--model"))
        assert os.path.exists(os.path.join(good_dir, "model.safetensors"))
        assert C.scan_hf_cache(hub) == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_repair_can_target_one_model_by_id():
    tmp = tempfile.mkdtemp()
    try:
        hub = os.path.join(tmp, "hub")
        target = os.path.join(hub, "models--Synthyra--ESMFold2-Fast")
        _safetensors(os.path.join(target, "snapshots", "r1", "model.safetensors"))
        other = os.path.join(hub, "models--other--model")
        _safetensors(os.path.join(other, "snapshots", "r1", "model.safetensors"))

        removed = C.repair_hf_cache("Synthyra/ESMFold2-Fast", cache_dir=hub)
        assert removed == [target]
        assert os.path.exists(other)
        assert C.repair_hf_cache("Synthyra/ESMFold2-Fast", cache_dir=hub) == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_use_drive_keeps_model_weights_off_drive_by_default():
    """Weights on Drive are unreadable, so the default must not put them there."""
    tmp = tempfile.mkdtemp()
    saved = {k: os.environ.get(k) for k in
             ("OUTPUT_ROOT", "FOLD_CACHE_PATH", "HF_HOME")}
    try:
        base = os.path.join(tmp, "redaa")
        env = C.use_drive(base)
        assert "OUTPUT_ROOT" in env and "FOLD_CACHE_PATH" in env
        assert env.get("HF_HOME") != os.path.join(base, "hf")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp, ignore_errors=True)


def test_hf_cache_dir_honours_the_env_and_detects_drive():
    saved = {k: os.environ.get(k) for k in
             ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME")}
    try:
        for key in saved:
            os.environ.pop(key, None)
        os.environ["HF_HOME"] = "/content/drive/MyDrive/redaa/hf"
        assert C.hf_cache_dir() == "/content/drive/MyDrive/redaa/hf/hub"
        assert C.on_drive(C.hf_cache_dir())

        os.environ["HF_HUB_CACHE"] = "/content/hf/hub"
        assert C.hf_cache_dir() == "/content/hf/hub"
        assert not C.on_drive(C.hf_cache_dir())
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_diagnose_flags_a_cache_sitting_on_drive():
    saved = os.environ.get("HF_HOME")
    try:
        os.environ["HF_HOME"] = "/content/drive/MyDrive/redaa/hf"
        report = C.diagnose()
        hit = [p for p in report["problems"] if "Google Drive" in p]
        assert hit, report["problems"]
        assert "header too large" in hit[0]
        assert "REPAIR_HF_CACHE" in hit[0]
    finally:
        if saved is None:
            os.environ.pop("HF_HOME", None)
        else:
            os.environ["HF_HOME"] = saved


# ---------------------------------------------------------------------------
# Run logging
# ---------------------------------------------------------------------------


def _with_output_root(tmp, fn):
    saved = os.environ.get("OUTPUT_ROOT")
    os.environ["OUTPUT_ROOT"] = os.path.join(tmp, "outputs")
    try:
        return fn()
    finally:
        if saved is None:
            os.environ.pop("OUTPUT_ROOT", None)
        else:
            os.environ["OUTPUT_ROOT"] = saved


def test_log_dir_sits_beside_the_outputs_so_drive_keeps_it():
    tmp = tempfile.mkdtemp()
    try:
        folder = _with_output_root(tmp, C.log_dir)
        assert folder == os.path.join(tmp, "logs"), folder
        assert os.path.isdir(folder)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_capture_tees_output_to_the_cell_and_to_a_file():
    import contextlib, io

    tmp = tempfile.mkdtemp()
    try:
        cell = io.StringIO()

        def run():
            with contextlib.redirect_stdout(cell):
                with C.capture_run_log() as log:
                    print("a line from the run")
            return log

        log = _with_output_root(tmp, run)
        # The cell still saw it...
        assert "a line from the run" in cell.getvalue()
        # ...and so did the file.
        assert "a line from the run" in open(log.path).read()
        assert "=== console ===" in open(log.path).read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_capture_records_a_crash_and_still_re_raises():
    """The run that dies is the one whose output you most need."""
    import contextlib, io

    tmp = tempfile.mkdtemp()
    try:
        holder = {}

        def run():
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    with C.capture_run_log() as log:
                        holder["log"] = log
                        print("got as far as here")
                        raise RuntimeError("header too large")
                except RuntimeError as exc:
                    holder["raised"] = exc

        _with_output_root(tmp, run)

        assert isinstance(holder.get("raised"), RuntimeError), holder
        body = open(holder["log"].path).read()
        assert "got as far as here" in body
        assert "=== traceback ===" in body
        assert "header too large" in body

        report = open(holder["log"].report_path).read()
        assert "header too large" in report
        assert "=== environment ===" in report
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_capture_restores_the_streams_even_on_a_crash():
    import contextlib, io

    tmp = tempfile.mkdtemp()
    before = (sys.stdout, sys.stderr)
    try:
        def run():
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    with C.capture_run_log():
                        raise ValueError("boom")
                except ValueError:
                    pass

        _with_output_root(tmp, run)
        assert (sys.stdout, sys.stderr) == before
    finally:
        sys.stdout, sys.stderr = before
        shutil.rmtree(tmp, ignore_errors=True)


def test_report_includes_the_environment_settings_and_results():
    tmp = tempfile.mkdtemp()
    try:
        run_dir = os.path.join(tmp, "outputs", "batch_chunk0_x")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "BATCH_SUMMARY.csv"), "w") as fh:
            fh.write("pdb,replicate,pruned_alphabet,best_seq_tm\n"
                     "6WVS_A,1,GLV,0.72\n")
        with open(os.path.join(run_dir, "FLAGGED_DOMAINS.csv"), "w") as fh:
            fh.write("pdb,native_tm,reason\n6WVS_A,0.41,TM=0.410 < 0.5\n")

        log = os.path.join(tmp, "run.log")
        with open(log, "w") as fh:
            fh.write("line one\nline two\n")

        path = _with_output_root(
            tmp, lambda: C.write_report(log_path=log, run_dir=run_dir)
        )
        body = open(path).read()
        assert "=== environment ===" in body
        assert "=== run configuration" in body
        assert "BATCH_SUMMARY.csv (1 row(s))" in body
        assert "pruned_alphabet=GLV" in body
        assert "FLAGGED_DOMAINS.csv" in body and "TM=0.410" in body
        assert "line one" in body and "line two" in body
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_report_elides_the_middle_of_a_huge_log_not_the_ends():
    """A failed run's useful parts are the setup and the traceback."""
    tmp = tempfile.mkdtemp()
    try:
        log = os.path.join(tmp, "run.log")
        with open(log, "w") as fh:
            fh.write("FIRST\n")
            fh.write("".join("filler %d\n" % i for i in range(5000)))
            fh.write("LAST\n")

        path = _with_output_root(
            tmp, lambda: C.write_report(log_path=log, max_log_lines=100)
        )
        body = open(path).read()
        assert "FIRST" in body
        assert "LAST" in body
        assert "lines elided" in body
        assert log in body            # points at the complete log
        assert "filler 2500" not in body
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_latest_report_picks_the_newest():
    tmp = tempfile.mkdtemp()
    try:
        def run():
            assert C.latest_report() is None
            older = C.write_report()
            os.utime(older, (1_600_000_000, 1_600_000_000))
            newer = C.write_report(out_path=os.path.join(C.log_dir(),
                                                         "report_newer.txt"))
            return older, newer, C.latest_report()

        older, newer, latest = _with_output_root(tmp, run)
        assert latest == newer != older
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Functional-site configuration and preview
# ---------------------------------------------------------------------------


def _site_pdb(path, seq="ACDEFGHIK", ligand=None):
    aa3 = {
        "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
        "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    }
    lines = []
    n = 1
    for i, aa in enumerate(seq):
        for name, dx in ((" N  ", -0.6), (" CA ", 0.0), (" C  ", 0.7)):
            lines.append(
                f"ATOM  {n:>5d} {name} {aa3[aa]} A{i + 1:>4d}    "
                f"{3.8 * i + dx:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00  0.00"
            )
            n += 1
    lines.append("TER")
    if ligand:
        comp, x = ligand
        lines.append(
            f"HETATM{n:>5d}  C1  {comp:>3s} A 900    "
            f"{x:>8.3f}{2.0:>8.3f}{0.0:>8.3f}  1.00  0.00"
        )
    lines.append("END")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _clear_site_env():
    for key in list(os.environ):
        if key.startswith("SITE_") or key in ("SEARCH_MODE", "MPNN_MODEL_TYPE"):
            del os.environ[key]


def test_configure_site_sets_the_environment():
    _clear_site_env()
    try:
        applied = C.configure_site(mode="site_aware", spec="H46 D81 C151")
        assert applied["SEARCH_MODE"] == "site_aware"
        assert os.environ["SEARCH_MODE"] == "site_aware"
        assert os.environ["SITE_SPEC"] == "H46 D81 C151"
        assert os.environ["SITE_RESIDUE_POLICY"] == "exempt"
        assert os.environ["SITE_GATE"] == "1"
    finally:
        _clear_site_env()


def test_configure_site_preset_fills_blanks_but_explicit_values_win():
    _clear_site_env()
    try:
        applied = C.configure_site(mode="site_aware", preset="atp")
        # The preset supplies the spec, its ligand and a ligand-aware model.
        assert applied["SITE_SPEC"] == C.preset("atp").site_spec
        assert "ATP" in applied["SITE_KEEP_LIGANDS"]
        assert applied["MPNN_MODEL_TYPE"] == "ligand_mpnn"

        override = C.configure_site(
            mode="site_aware", preset="atp", spec="R82 R84",
            mpnn_model_type="protein_mpnn",
        )
        assert override["SITE_SPEC"] == "R82 R84"
        assert override["MPNN_MODEL_TYPE"] == "protein_mpnn"
    finally:
        _clear_site_env()


def test_configure_site_rejects_a_bad_mode_or_policy():
    _clear_site_env()
    try:
        for kwargs in ({"mode": "nope"}, {"mode": "site_aware", "policy": "nope"}):
            try:
                C.configure_site(**kwargs)
            except ValueError:
                continue
            raise AssertionError(f"{kwargs} should have been rejected")
    finally:
        _clear_site_env()


def test_preview_site_resolves_against_the_real_file():
    tmp = tempfile.mkdtemp()
    try:
        path = _site_pdb(os.path.join(tmp, "x.pdb"))
        out = C.preview_site(path, "C2 F5 H7")
        assert "3 position(s)" in out
        assert "C2" in out and "F5" in out and "H7" in out
        assert "looks usable" in out
        assert "ligands in file: none" in out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_preview_site_reports_a_numbering_mismatch_loudly():
    tmp = tempfile.mkdtemp()
    try:
        path = _site_pdb(os.path.join(tmp, "x.pdb"))
        out = C.preview_site(path, "H2")  # residue 2 is C
        assert "NUMBERING MISMATCH" in out
        assert "will refuse this domain" in out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_preview_site_reports_an_empty_resolution():
    tmp = tempfile.mkdtemp()
    try:
        path = _site_pdb(os.path.join(tmp, "x.pdb"))
        out = C.preview_site(path, "A99")
        assert "will skip this domain" in out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_preview_site_shows_a_ligand_derived_pocket_and_its_coverage():
    tmp = tempfile.mkdtemp()
    try:
        path = _site_pdb(os.path.join(tmp, "x.pdb"), ligand=("ATP", 7.6))
        out = C.preview_site(path, "ligand:ATP@4.5")
        assert "ATP (1 atoms)" in out
        assert "A from ligand" in out
        assert "% of the chain" in out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_preview_site_handles_a_bad_spec_and_a_missing_file():
    assert "cannot parse" in C.preview_site(__file__, "not-a-spec")
    assert "no such file" in C.preview_site("/nope/nope.pdb", "H46")


def test_preview_sites_covers_every_prepared_input():
    tmp = tempfile.mkdtemp()
    try:
        _site_pdb(os.path.join(tmp, "a.pdb"))
        _site_pdb(os.path.join(tmp, "b.pdb"))
        out = C.preview_sites(tmp, "C2")
        assert "a.pdb" in out and "b.pdb" in out
        assert "no prepared inputs" in C.preview_sites(
            os.path.join(tmp, "empty"), "C2"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_describe_site_results_is_silent_for_an_alphabet_only_run():
    assert C.describe_site_results([{"search_mode": "alphabet_only"}]) == ""
    assert C.describe_site_results([]) == ""


def test_describe_site_results_spells_out_exempt_positions():
    rows = [{
        "search_mode": "site_aware", "pdb": "d1",
        "pruned_alphabet": "GLV", "site_n_positions": "6",
        "site_residues": "H46+D81", "site_residue_types": "CDH",
        "site_policy": "exempt", "best_seq_site_ok": "1",
        "best_seq_site_bb_rmsd": "0.70",
    }]
    out = C.describe_site_results(rows)
    # The headline number has to carry its caveat: 3 letters plus 6 positions
    # that are not letters is not the same claim as 3 letters.
    assert "3 letter(s)" in out
    assert "6 exempt position(s)" in out
    assert "in_alphabet" in out
    assert "site preserved in 1/1" in out


def test_describe_site_results_reports_what_the_graft_cost():
    rows = [{
        "search_mode": "scaffold_then_site", "pdb": "d1",
        "pruned_alphabet": "GLVR", "site_n_positions": "3",
        "site_residues": "H46", "site_residue_types": "CDH",
        "site_policy": "exempt", "best_seq_site_ok": "1",
        "graft_direct_pass": "0", "graft_added_aas": "R", "graft_passed": "1",
    }]
    out = C.describe_site_results(rows)
    assert "carried the site unchanged in 0/1" in out
    assert "cost extra letters" in out
    assert "site_aware" in out


def test_configure_site_always_decides_the_model_type():
    """A notebook's cells get re-run in any order.

    Configuring the ATP preset once leaves MPNN_MODEL_TYPE=ligand_mpnn in the
    environment. If a later TEV configuration stays silent on it, the ligand
    model is used for a case with no ligand -- which is exactly the
    misconfiguration that reached a real run.
    """
    _clear_site_env()
    try:
        C.configure_site(mode="site_aware", preset="atp")
        assert os.environ["MPNN_MODEL_TYPE"] == "ligand_mpnn"

        # Same session, now a catalytic-site case with no preset and no
        # explicit model type: it must not inherit ligand_mpnn.
        applied = C.configure_site(mode="site_aware", spec="H46 D81 C151")
        assert applied["MPNN_MODEL_TYPE"] == "protein_mpnn"
        assert os.environ["MPNN_MODEL_TYPE"] == "protein_mpnn"

        # The TEV preset says so for itself.
        applied = C.configure_site(mode="site_aware", preset="tev")
        assert applied["MPNN_MODEL_TYPE"] == "protein_mpnn"
    finally:
        _clear_site_env()


def test_preview_site_refuses_a_ligand_model_on_a_ligand_free_input():
    tmp = tempfile.mkdtemp()
    try:
        path = _site_pdb(os.path.join(tmp, "x.pdb"))  # no ligand
        out = C.preview_site(path, "C2 F5", mpnn_model_type="ligand_mpnn")
        assert "no ligand" in out
        assert "protein_mpnn" in out
        assert "will refuse this domain" in out

        # The same input is fine for a protein model.
        ok = C.preview_site(path, "C2 F5", mpnn_model_type="protein_mpnn")
        assert "looks usable" in ok

        # And a ligand model is fine once there is a ligand.
        with_lig = _site_pdb(os.path.join(tmp, "y.pdb"), ligand=("ATP", 7.6))
        ok2 = C.preview_site(
            with_lig, "ligand:ATP@4.5", mpnn_model_type="ligand_mpnn"
        )
        assert "looks usable" in ok2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
