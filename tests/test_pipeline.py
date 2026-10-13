"""End-to-end orchestration test with stub engines.

Exercises the whole pipeline -- domain setup, chemical requirements, the
native round-trip check, forward greedy growth, two-stage screening, full
validation, backward pruning, checkpointing and the summary writers -- against
stand-in ESMFold2 and LigandMPNN engines.  No GPU, no torch, no model weights,
so this can run on a login node before anything is queued.

The synthetic problem has a known answer: a design folds correctly exactly
when its alphabet contains G, V and L.  The forward search should therefore
grow to a 4-letter alphabet (A and G seed, V from hydrophobic seeding, then L),
and backward pruning should drop the A to land on GLV.
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redaa.config import RedAAConfig  # noqa: E402
from redaa.folding import FoldResult, normalise_plddt  # noqa: E402
from redaa.mpnn import MPNNDesigns  # noqa: E402
from redaa.pipeline import Pipeline  # noqa: E402

REQUIRED = set("GVL")
NATIVE = ("ACDEFGHIKLMNPQRSTVWY" * 5)[:100]


def _helix(n):
    t = np.arange(n)
    ang = np.deg2rad(100.0) * t
    return np.stack([2.3 * np.cos(ang), 2.3 * np.sin(ang), 1.5 * t], axis=1)


def _write_reference(path, seq=NATIVE):
    aa1_to_3 = {
        "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
        "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
        "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
        "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
    }
    xyz = _helix(len(seq))
    lines = []
    for i, (aa, (x, y, z)) in enumerate(zip(seq, xyz), start=1):
        lines.append(
            f"ATOM  {i:>5d}  CA  {aa1_to_3[aa]} A{i:>4d}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00"
        )
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


class StubMPNN:
    """Deterministic sequence sampler over the allowed alphabet."""

    def __init__(self, length=len(NATIVE), cfg=None):
        self.length = length
        self.calls = 0
        self.cfg = cfg

    def load(self):
        pass

    def release(self, _path):
        pass

    def stats(self):
        return f"{self.calls} stub calls"

    def design(self, _pdb, allowed, num_seq=None, temperature=None, seed=None):
        self.calls += 1
        allowed = sorted(set(allowed))
        rng = np.random.default_rng(abs(hash(("".join(allowed), seed))) % 2**32)
        n = num_seq or 25
        pairs = []
        for i in range(n):
            seq = "".join(rng.choice(allowed, size=self.length))
            pairs.append((1.0 + i * 0.02, seq))
        # Monotone in alphabet size so the baseline-tolerance stop behaves.
        mean = 3.0 - 0.1 * len(allowed)
        return MPNNDesigns(mean, sorted(pairs))


class StubFolder:
    """Folds to the reference when the alphabet is sufficient, else garbage."""

    def __init__(self, ref_xyz, cfg=None):
        self.ref = ref_xyz
        self.calls = 0
        self.seen: list[str] = []
        self.cfg = cfg
        self.cache = None

    def load(self):
        pass

    def stats(self):
        return f"{self.calls} stub folds ({len(set(self.seen))} unique)"

    def fold(self, sequence, steps=None):
        self.calls += 1
        self.seen.append(sequence)
        if sequence == NATIVE or REQUIRED <= set(sequence):
            rng = np.random.default_rng(len(sequence))
            return FoldResult(
                sequence, 88.0, 0.9,
                self.ref + rng.normal(scale=0.2, size=self.ref.shape),
                pdb_text="ATOM\nEND\n",
            )
        rng = np.random.default_rng(abs(hash(sequence)) % 2**32)
        return FoldResult(
            sequence, 42.0, 0.2,
            rng.normal(scale=25.0, size=self.ref.shape),
            pdb_text="ATOM\nEND\n",
        )

    def fold_many(self, sequences, steps=None):
        return [self.fold(s, steps=steps) for s in sequences]

    def write_pdb(self, _result, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            fh.write("ATOM\nEND\n")
        return True


class StubPipeline(Pipeline):
    def setup(self):
        from redaa.foldcache import FoldCache
        from redaa import structure as st

        self.run_dir = self.resolve_run_dir()
        os.makedirs(self.run_dir, exist_ok=True)
        self.cache = FoldCache(None, enabled=False)
        ref_xyz, _ = st.parse_ca_from_file(self.ref_path)
        self.folder = StubFolder(np.array(ref_xyz))
        self.mpnn = StubMPNN()


def _make_config(tmp, **over):
    cfg = RedAAConfig()
    cfg.input_pdb_dir = os.path.join(tmp, "inputs")
    cfg.output_root = os.path.join(tmp, "outputs")
    cfg.auto_resume = False
    cfg.resume_from = ""
    cfg.run_name = "unittest"
    cfg.num_replicates = 1
    cfg.chunk_index = 0
    cfg.chunk_size = 1
    cfg.use_fold_cache = False
    cfg.exhaustive_pruning_threshold = 8
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _run(tmp, **over):
    os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
    ref = os.path.join(tmp, "inputs", "testdomA00.pdb")
    _write_reference(ref)
    cfg = _make_config(tmp, **over)
    pipe = StubPipeline(cfg)
    pipe.ref_path = ref
    code = pipe.run()
    return pipe, code


def _read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------


def test_full_run_finds_the_known_minimal_alphabet():
    tmp = tempfile.mkdtemp(prefix="redaa_e2e_")
    try:
        pipe, code = _run(tmp)
        assert code == 0, code
        rows = [r for r in pipe.results if "elapsed_seconds" in r]
        assert len(rows) == 1, pipe.results
        row = rows[0]
        # Forward search grows A,G -> +V (hydrophobic seed) -> +L.
        assert set(row["final_alphabet"]) >= REQUIRED, row["final_alphabet"]
        # Pruning drops whatever is not required.
        assert row["pruned_alphabet"] == "GLV", row["pruned_alphabet"]
        assert row["pruned_alphabet_size"] == 3
        assert row["aas_removed_pruning"] >= 1
        assert row["best_sequence"] and REQUIRED <= set(row["best_sequence"])
        assert row["best_seq_tm"] > 0.5
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_writes_all_expected_artifacts():
    tmp = tempfile.mkdtemp(prefix="redaa_art_")
    try:
        pipe, _ = _run(tmp)
        run_dir = pipe.run_dir
        domain_dir = os.path.join(run_dir, "testdomA00")
        for rel in (
            "BATCH_SUMMARY.csv",
            "ALL_BEST_SEQUENCES.fasta",
            "testdomA00/domain_info.txt",
            "testdomA00/native_esmfold.pdb",
            "testdomA00/testdomA00_best_sequence.fasta",
            "testdomA00/run_1/COMPLETED",
            "testdomA00/run_1/alphabet_building_log.csv",
            "testdomA00/run_1/esmfold_validation_log.csv",
        ):
            assert os.path.exists(os.path.join(run_dir, rel)), rel

        summary = _read_csv(os.path.join(run_dir, "BATCH_SUMMARY.csv"))
        assert summary[0]["pdb"] == "testdomA00"
        assert summary[0]["pruned_alphabet"] == "GLV"

        # Column layout must stay compatible with the V9 analysis scripts.
        build = _read_csv(os.path.join(domain_dir, "run_1/alphabet_building_log.csv"))
        assert list(build[0].keys()) == [
            "Stage", "Round", "Candidate_AA", "MPNN_Score",
            "Current_Set", "Improvement_%", "Status",
        ]
        assert build[0]["Stage"] == "BASELINE"
        stages = {r["Stage"] for r in build}
        assert {"BASELINE", "INIT", "HYDROPHOBIC", "OPEN", "PRUNING"} <= stages

        val = _read_csv(os.path.join(domain_dir, "run_1/esmfold_validation_log.csv"))
        assert list(val[0].keys()) == [
            "Stage", "Round", "Alphabet", "Mean_RMSD", "Min_RMSD",
            "Mean_TM", "Max_TM", "Mean_pLDDT", "Max_pLDDT",
            "Num_Pass_All", "Successful_Folds", "Total_Attempted",
            "Validation_Status",
        ]
        assert any(r["Stage"] == "FULL_VALIDATION" for r in val)
        assert any(r["Validation_Status"] == "PASSED" for r in val)

        fasta = open(os.path.join(run_dir, "ALL_BEST_SEQUENCES.fasta")).read()
        assert fasta.startswith(">testdomA00_rep1 ")
        assert "alphabet=" in fasta and "TM=" in fasta
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_two_stage_screening_costs_fewer_folds_than_folding_every_candidate():
    """The screened run must be cheaper and reach the same alphabet."""
    results = {}
    for label, over in (
        ("screened", {}),
        # stage1 == stage2 reproduces V9's behaviour: every candidate folded
        # at the full per-candidate budget, every round.
        ("exhaustive", {"screen_folds_stage1": 3, "screen_folds_stage2": 3}),
    ):
        tmp = tempfile.mkdtemp(prefix=f"redaa_{label}_")
        try:
            pipe, _ = _run(tmp, **over)
            row = [r for r in pipe.results if "elapsed_seconds" in r][0]
            results[label] = (pipe.folder.calls, row["pruned_alphabet"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    assert results["screened"][1] == results["exhaustive"][1] == "GLV", results
    assert results["screened"][0] < results["exhaustive"][0], results


def test_downward_closure_reduces_folds_without_changing_the_answer():
    outcomes = {}
    for closure in (True, False):
        tmp = tempfile.mkdtemp(prefix=f"redaa_cl{closure}_")
        try:
            pipe, _ = _run(tmp, pruning_downward_closure=closure)
            row = [r for r in pipe.results if "elapsed_seconds" in r][0]
            outcomes[closure] = (
                pipe.folder.calls, row["pruned_alphabet"], row["pruning_trials"]
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    assert outcomes[True][1] == outcomes[False][1] == "GLV", outcomes
    assert outcomes[True][2] <= outcomes[False][2], outcomes


def test_completed_replicates_are_skipped_on_resume():
    tmp = tempfile.mkdtemp(prefix="redaa_resume_")
    try:
        os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
        ref = os.path.join(tmp, "inputs", "testdomA00.pdb")
        _write_reference(ref)

        first = StubPipeline(_make_config(tmp))
        first.ref_path = ref
        first.run()
        run_dir = first.run_dir

        second = StubPipeline(_make_config(tmp, resume_from=run_dir))
        second.ref_path = ref
        second.run()

        assert second.run_dir == run_dir
        assert [r.get("status") for r in second.results] == ["resumed"]
        # A resumed replicate must not re-run the search; only the native
        # round-trip check folds.
        assert second.folder.calls <= 2, second.folder.calls
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_weak_native_round_trip_is_flagged_but_still_searched():
    tmp = tempfile.mkdtemp(prefix="redaa_flag_")
    try:
        os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
        ref = os.path.join(tmp, "inputs", "baddomA00.pdb")
        _write_reference(ref)
        cfg = _make_config(tmp)
        pipe = StubPipeline(cfg)
        pipe.ref_path = ref

        original_setup = pipe.setup

        def setup_with_bad_native():
            original_setup()
            real_fold = pipe.folder.fold

            def fold(sequence, steps=None):
                result = real_fold(sequence, steps=steps)
                if sequence == NATIVE:
                    # Native does not reproduce the reference at all.
                    rng = np.random.default_rng(1)
                    result.ca = rng.normal(scale=30.0, size=pipe.folder.ref.shape)
                    result.plddt = 20.0
                return result

            pipe.folder.fold = fold
            pipe.folder.fold_many = lambda seqs, steps=None: [
                fold(s, steps=steps) for s in seqs
            ]

        pipe.setup = setup_with_bad_native
        pipe.run()

        assert len(pipe.flagged) == 1, pipe.flagged
        assert pipe.flagged[0]["pdb"] == "baddomA00"
        assert "TM=" in pipe.flagged[0]["reason"]
        assert os.path.exists(os.path.join(pipe.run_dir, "FLAGGED_DOMAINS.csv"))
        assert os.path.exists(
            os.path.join(
                pipe.run_dir, "baddomA00",
                "FLAGGED_native_esmfold_mismatch.txt",
            )
        )
        # The search still ran: the domain has a real result row.
        rows = [r for r in pipe.results if "elapsed_seconds" in r]
        assert len(rows) == 1 and rows[0]["pruned_alphabet"] == "GLV"
        assert "flagged" in rows[0]["status"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_wall_clock_budget_stops_cleanly_and_reports_resumable():
    tmp = tempfile.mkdtemp(prefix="redaa_budget_")
    try:
        os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
        ref = os.path.join(tmp, "inputs", "testdomA00.pdb")
        _write_reference(ref)
        cfg = _make_config(tmp, max_wall_seconds=1)
        pipe = StubPipeline(cfg)
        pipe.ref_path = ref
        pipe.started -= 10  # pretend the budget is already blown
        code = pipe.run()
        assert code == 2, code
        # Summaries are still written, so the run is inspectable.
        assert os.path.exists(os.path.join(pipe.run_dir, "BATCH_SUMMARY.csv"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_setup_reuses_injected_engines_when_the_models_match():
    """A notebook re-running the run cell should not reload ESMFold2.

    Reuse is checked against the real Pipeline.setup, not the stub's: if the
    reuse branch were not taken, setup would try to build a genuine FoldEngine
    and fail for want of torch.
    """
    tmp = tempfile.mkdtemp(prefix="redaa_reuse_")
    try:
        os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
        ref = os.path.join(tmp, "inputs", "testdomA00.pdb")
        _write_reference(ref)
        cfg = _make_config(tmp)

        folder = StubFolder(_helix(len(NATIVE)), cfg=cfg)
        mpnn = StubMPNN(cfg=cfg)
        pipe = Pipeline(cfg, folder=folder, mpnn=mpnn)
        pipe.setup()

        assert pipe.folder is folder
        assert pipe.mpnn is mpnn
        # The engine is repointed at this run's cache rather than keeping a
        # stale one from the previous run.
        assert pipe.folder.cache is pipe.cache
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_setup_reloads_when_the_requested_model_changed():
    """Changing the model in a form cell must not silently reuse the old one."""
    import redaa.pipeline as P

    tmp = tempfile.mkdtemp(prefix="redaa_noreuse_")
    built = []

    class _Sentinel:
        def __init__(self, cfg, cache=None):
            built.append(cfg.fold_model_id)
            self.cfg = cfg
            self.cache = cache

        def load(self):
            pass

    real_fold, real_mpnn = P.FoldEngine, P.MPNNEngine
    try:
        os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
        _write_reference(os.path.join(tmp, "inputs", "testdomA00.pdb"))

        old_cfg = _make_config(tmp)
        old_cfg.fold_model_id = "Synthyra/ESMFold2-Fast"
        new_cfg = _make_config(tmp)
        new_cfg.fold_model_id = "Synthyra/ESMFold2-600"

        P.FoldEngine = _Sentinel
        P.MPNNEngine = _Sentinel
        pipe = Pipeline(
            new_cfg,
            folder=StubFolder(_helix(len(NATIVE)), cfg=old_cfg),
            mpnn=StubMPNN(cfg=old_cfg),
        )
        pipe.setup()
        assert built == ["Synthyra/ESMFold2-600"], built
        assert isinstance(pipe.folder, _Sentinel)
    finally:
        P.FoldEngine, P.MPNNEngine = real_fold, real_mpnn
        shutil.rmtree(tmp, ignore_errors=True)


def test_plddt_is_rescaled_to_the_zero_to_hundred_convention():
    assert normalise_plddt(0.87) == 87.0
    assert normalise_plddt(87.0) == 87.0
    assert normalise_plddt(1.0) == 100.0
    assert normalise_plddt(None) is None


def main():
    failures = []
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            import traceback

            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc!r}", flush=True)
            traceback.print_exc()
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
