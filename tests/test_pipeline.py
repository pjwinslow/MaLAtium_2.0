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


AA1_TO_3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}


def _backbone(n):
    """N, CA and C for a synthetic helix.

    The site metrics need a real backbone frame, not just a CA trace, so the
    reference carries all three.  The CA positions are unchanged from the
    CA-only version this replaced, which is why the pre-existing fold tests
    see exactly the same geometry.
    """
    ca = _helix(n)
    nxt = np.vstack([ca[1:], ca[-1] + (ca[-1] - ca[-2])])
    prv = np.vstack([ca[0] - (ca[1] - ca[0]), ca[:-1]])
    # Place N towards the previous CA and C towards the next one, at
    # bond-length scale, so the derived virtual CB points somewhere sensible.
    n_xyz = ca + 0.46 * (prv - ca) / np.linalg.norm(prv - ca, axis=1, keepdims=True) * 3.2
    c_xyz = ca + 0.48 * (nxt - ca) / np.linalg.norm(nxt - ca, axis=1, keepdims=True) * 3.2
    return n_xyz, ca, c_xyz


def _write_reference(path, seq=NATIVE, ligand=None):
    n_xyz, ca, c_xyz = _backbone(len(seq))
    lines = []
    serial = 1
    for i, aa in enumerate(seq):
        for name, xyz in (("N", n_xyz[i]), ("CA", ca[i]), ("C", c_xyz[i])):
            lines.append(
                f"ATOM  {serial:>5d}  {name:<3s} {AA1_TO_3[aa]} A{i + 1:>4d}    "
                f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}  1.00  0.00"
            )
            serial += 1
    lines.append("TER")
    if ligand:
        comp, centre, n_atoms = ligand
        for k in range(n_atoms):
            xyz = np.asarray(centre) + np.array([0.9 * k, 0.0, 0.0])
            lines.append(
                f"HETATM{serial:>5d}  C{k:<2d} {comp:>3s} A{900:>4d}    "
                f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}  1.00  0.00"
            )
            serial += 1
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


class StubMPNN:
    """Deterministic sequence sampler over the allowed alphabet."""

    def __init__(self, length=len(NATIVE), cfg=None, loaded=True):
        self.length = length
        self.calls = 0
        self.cfg = cfg
        self.fixed_seen: list[dict] = []
        self.alphabets_seen: list[str] = []
        self.ligand_atoms = 0
        # The real engine counts positions it had to correct after sampling,
        # which preflight reports on.
        self.n_forced_positions = 0
        # Stands in for a loaded checkpoint; None means load() never ran.
        self.model = object() if loaded else None

    def load(self):
        pass

    def release(self, _path):
        pass

    def stats(self):
        return f"{self.calls} stub calls"

    def native_sequence(self, _pdb):
        return NATIVE[: self.length]

    def length_of(self, _pdb):
        return self.length

    def ligand_atom_count(self, _pdb):
        return self.ligand_atoms

    def design(
        self, _pdb, allowed, num_seq=None, temperature=None, seed=None,
        fixed=None,
    ):
        self.calls += 1
        self.fixed_seen.append(dict(fixed or {}))
        allowed = sorted(set(allowed))
        self.alphabets_seen.append("".join(allowed))
        rng = np.random.default_rng(abs(hash(("".join(allowed), seed))) % 2**32)
        n = num_seq or 25
        pairs = []
        for i in range(n):
            chars = list(rng.choice(allowed, size=self.length))
            # A real LigandMPNN run keeps the native residue wherever the
            # chain mask says not to design; the stub does the same so the
            # pipeline's fixed-position handling is exercised rather than
            # merely called.
            for pos, aa in (fixed or {}).items():
                if 0 <= pos < len(chars):
                    chars[pos] = aa
            pairs.append((1.0 + i * 0.02, "".join(chars)))
        # Monotone in alphabet size so the baseline-tolerance stop behaves.
        mean = 3.0 - 0.1 * len(allowed)
        return MPNNDesigns(mean, sorted(pairs))


class StubFolder:
    """Folds to the reference when the alphabet is sufficient, else garbage."""

    def __init__(self, ref_xyz, cfg=None, loaded=True, ref_frames=None,
                 want_frames=False):
        self.ref = ref_xyz
        self.calls = 0
        self.seen: list[str] = []
        self.cfg = cfg
        self.cache = None
        # (L, 3, 3) N/CA/C of the reference, so a "correct" fold can hand back
        # a backbone frame and the site metrics have something to measure.
        self.ref_frames = ref_frames
        self.want_frames = want_frames
        # Sequences the stub should fold correctly in every respect except the
        # site, used to prove the site gate rejects on its own.
        self.break_site_for: set[str] = set()
        # Stands in for a loaded model; None means load() never ran.
        self.model = object() if loaded else None

    def load(self):
        pass

    def stats(self):
        return f"{self.calls} stub folds ({len(set(self.seen))} unique)"

    def _frames(self, scale, seed, break_site=False):
        if self.ref_frames is None or not self.want_frames:
            return None
        rng = np.random.default_rng(seed)
        out = self.ref_frames + rng.normal(scale=scale, size=self.ref_frames.shape)
        if break_site:
            # Move a handful of residues a long way without touching the rest,
            # which is exactly the failure the site gate exists to catch: the
            # global fold still superposes well, the site does not.
            out[1:8] += 7.0
        return out

    def fold(self, sequence, steps=None, want_frames=None):
        self.calls += 1
        self.seen.append(sequence)
        if sequence == NATIVE or REQUIRED <= set(sequence):
            rng = np.random.default_rng(len(sequence))
            break_site = sequence in self.break_site_for
            return FoldResult(
                sequence, 88.0, 0.9,
                self.ref + rng.normal(scale=0.2, size=self.ref.shape),
                pdb_text="ATOM\nEND\n",
                frames=self._frames(0.2, len(sequence), break_site),
            )
        rng = np.random.default_rng(abs(hash(sequence)) % 2**32)
        return FoldResult(
            sequence, 42.0, 0.2,
            rng.normal(scale=25.0, size=self.ref.shape),
            pdb_text="ATOM\nEND\n",
            frames=self._frames(25.0, abs(hash(sequence)) % 2**32),
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
        ref_xyz, ref_seq = st.parse_ca_from_file(self.ref_path)
        records = st.parse_residues_from_file(self.ref_path)
        frames, _valid = st.frames_from_records(records)
        self.folder = StubFolder(
            np.array(ref_xyz),
            ref_frames=frames[:, :3, :],
            want_frames=self.cfg.site_enabled,
        )
        self.mpnn = StubMPNN(length=len(ref_seq))


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


def _run(tmp, ligand=None, **over):
    os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
    ref = os.path.join(tmp, "inputs", "testdomA00.pdb")
    _write_reference(ref, ligand=ligand)
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
            "Site_BB_RMSD", "Site_Local_RMSD", "Site_CB_Dev", "Num_Site_Pass",
        ]
        # An alphabet_only run still writes the site columns, left blank, so
        # one reader handles every mode.
        assert all(r["Num_Site_Pass"] == "" for r in val)
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


def test_setup_does_not_reuse_an_engine_whose_load_failed():
    """A failed run leaves its engine behind; it must not be treated as loaded.

    Observed on Colab: a run died inside the ESMFold2 load, the half-built
    engine was cached for the next run, and setup announced "reusing the model
    already in memory" and skipped load() -- so the real error resurfaced much
    later, from inside the first fold, instead of at setup.
    """
    import redaa.pipeline as P

    tmp = tempfile.mkdtemp(prefix="redaa_deadengine_")
    built = []

    class _Sentinel:
        def __init__(self, cfg, cache=None, want_frames=False):
            built.append(type(self).__name__)
            self.cfg = cfg
            self.cache = cache
            self.model = object()

        def load(self):
            pass

    real_fold, real_mpnn = P.FoldEngine, P.MPNNEngine
    try:
        os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
        _write_reference(os.path.join(tmp, "inputs", "testdomA00.pdb"))
        cfg = _make_config(tmp)

        P.FoldEngine = _Sentinel
        P.MPNNEngine = _Sentinel
        dead_folder = StubFolder(_helix(len(NATIVE)), cfg=cfg, loaded=False)
        dead_mpnn = StubMPNN(cfg=cfg, loaded=False)
        pipe = Pipeline(cfg, folder=dead_folder, mpnn=dead_mpnn)
        pipe.setup()

        assert pipe.folder is not dead_folder
        assert pipe.mpnn is not dead_mpnn
        assert len(built) == 2, built
    finally:
        P.FoldEngine, P.MPNNEngine = real_fold, real_mpnn
        shutil.rmtree(tmp, ignore_errors=True)


def test_setup_reloads_when_the_requested_model_changed():
    """Changing the model in a form cell must not silently reuse the old one."""
    import redaa.pipeline as P

    tmp = tempfile.mkdtemp(prefix="redaa_noreuse_")
    built = []

    class _Sentinel:
        def __init__(self, cfg, cache=None, want_frames=False):
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




# ---------------------------------------------------------------------------
# Functional-site modes
# ---------------------------------------------------------------------------
#
# NATIVE is "ACDEFGHIKLMNPQRSTVWY" repeated, numbered from 1, so residue 2 is
# C, residue 3 is D and residue 7 is H -- three residues whose identities are
# all outside the GVL folding alphabet this synthetic problem resolves to.
# That is the interesting case: the site can only survive if its positions are
# genuinely exempt from the alphabet restriction.

SITE = "C2 D3 H7"
SITE_INDICES = {1: "C", 2: "D", 6: "H"}


def test_site_aware_mode_holds_the_site_fixed_throughout():
    with tempfile.TemporaryDirectory(prefix="redaa_siteaware_") as tmp:
        pipe, code = _run(
            tmp, search_mode="site_aware", site_spec=SITE,
            num_seqs_to_fold=3, prune_folds=1,
        )
        assert code == 0

        # Every MPNN call in a site_aware run carries the fixed positions --
        # including the 20-residue baseline, so its score is computed over the
        # same designed positions as the restricted alphabets it is compared
        # against.
        assert pipe.mpnn.fixed_seen, "no MPNN calls recorded"
        assert all(f == SITE_INDICES for f in pipe.mpnn.fixed_seen), (
            [f for f in pipe.mpnn.fixed_seen if f != SITE_INDICES][:3]
        )

        row = _read_csv(os.path.join(pipe.run_dir, "BATCH_SUMMARY.csv"))[0]
        assert row["search_mode"] == "site_aware"
        assert row["site_residues"] == "C2+D3+H7"
        assert row["site_residue_types"] == "CDH"
        assert row["site_n_positions"] == "3"
        assert row["site_policy"] == "exempt"

        # The winning sequence really does carry the native site residues.
        seq = row["best_sequence"]
        assert seq, "no best sequence"
        for idx, aa in SITE_INDICES.items():
            assert seq[idx] == aa, (idx, seq[idx], aa)

        # And the site was measured, not merely declared.
        assert row["best_seq_site_bb_rmsd"] not in ("", "N/A")
        assert row["best_seq_site_ok"] == "1"

        val = _read_csv(
            os.path.join(pipe.run_dir, "testdomA00/run_1/esmfold_validation_log.csv")
        )
        measured = [r for r in val if r["Site_BB_RMSD"] not in ("", "N/A")]
        assert measured, "no site geometry recorded in the validation log"


def test_exempt_policy_keeps_site_residues_out_of_the_alphabet():
    with tempfile.TemporaryDirectory(prefix="redaa_exempt_") as tmp:
        pipe, code = _run(
            tmp, search_mode="site_aware", site_spec=SITE,
            site_residue_policy="exempt", num_seqs_to_fold=3, prune_folds=1,
        )
        assert code == 0
        row = _read_csv(os.path.join(pipe.run_dir, "BATCH_SUMMARY.csv"))[0]
        # The site's C, D and H are held at their positions without being
        # letters of the alphabet: the reported alphabet describes the
        # designable positions only.
        assert set(row["pruned_alphabet"]) & set("CDH") == set(), row
        # The folding alphabet is still the one this synthetic problem needs.
        assert row["pruned_alphabet"] == "GLV", row["pruned_alphabet"]


def test_in_alphabet_policy_protects_the_site_residue_types():
    with tempfile.TemporaryDirectory(prefix="redaa_inalpha_") as tmp:
        pipe, code = _run(
            tmp, search_mode="site_aware", site_spec=SITE,
            site_residue_policy="in_alphabet", num_seqs_to_fold=3,
            prune_folds=1,
        )
        assert code == 0
        row = _read_csv(os.path.join(pipe.run_dir, "BATCH_SUMMARY.csv"))[0]
        assert row["site_policy"] == "in_alphabet"
        # Under this policy the alphabet has to be one a real protein could be
        # written in, so the site's residue types are in it and survive
        # pruning.
        for aa in "CDH":
            assert aa in row["final_alphabet"], (aa, row["final_alphabet"])
            assert aa in row["pruned_alphabet"], (aa, row["pruned_alphabet"])


def test_scaffold_then_site_searches_blind_then_grafts():
    with tempfile.TemporaryDirectory(prefix="redaa_graft_") as tmp:
        pipe, code = _run(
            tmp, search_mode="scaffold_then_site", site_spec=SITE,
            num_seqs_to_fold=3, prune_folds=1,
        )
        assert code == 0

        # The search itself was blind to the site; only the graft phase fixes
        # positions. Both facts have to hold, or the mode is not what it says.
        assert any(f == {} for f in pipe.mpnn.fixed_seen), "search was not blind"
        assert any(f == SITE_INDICES for f in pipe.mpnn.fixed_seen), (
            "graft phase never fixed the site"
        )
        # Blind calls come first.
        first_fixed = next(
            i for i, f in enumerate(pipe.mpnn.fixed_seen) if f
        )
        assert all(f == {} for f in pipe.mpnn.fixed_seen[:first_fixed])

        row = _read_csv(os.path.join(pipe.run_dir, "BATCH_SUMMARY.csv"))[0]
        assert row["search_mode"] == "scaffold_then_site"
        assert row["graft_start_alphabet"], row
        assert row["graft_final_alphabet"], row
        assert row["graft_passed"] in ("0", "1")
        assert row["graft_direct_pass"] in ("0", "1")
        assert int(row["graft_final_size"]) >= int(row["graft_start_size"])

        val = _read_csv(
            os.path.join(pipe.run_dir, "testdomA00/run_1/esmfold_validation_log.csv")
        )
        assert any(r["Stage"].startswith("GRAFT") for r in val), (
            sorted({r["Stage"] for r in val})
        )
        build = _read_csv(
            os.path.join(pipe.run_dir, "testdomA00/run_1/alphabet_building_log.csv")
        )
        assert any(r["Stage"] == "GRAFT" for r in build)


def test_alphabet_only_mode_is_unchanged_by_the_site_machinery():
    with tempfile.TemporaryDirectory(prefix="redaa_blind_") as tmp:
        pipe, code = _run(tmp)
        assert code == 0
        assert all(f == {} for f in pipe.mpnn.fixed_seen)
        row = _read_csv(os.path.join(pipe.run_dir, "BATCH_SUMMARY.csv"))[0]
        assert row["search_mode"] == "alphabet_only"
        assert row["site_residues"] == ""
        assert row["best_seq_site_ok"] == ""
        assert row["pruned_alphabet"] == "GLV"


def test_site_gate_rejects_a_fold_that_folds_but_breaks_the_site():
    """The gate has to be able to fail on site geometry alone.

    The stub folds these sequences correctly in every global sense -- the CA
    trace still superposes on the reference -- while displacing the site
    residues. Only a site-specific check can tell the difference, which is
    the entire reason the site metrics exist.
    """
    from redaa import sites as sitemod
    from redaa import structure as st
    from redaa.search import Validator

    with tempfile.TemporaryDirectory(prefix="redaa_gate_") as tmp:
        ref = os.path.join(tmp, "ref.pdb")
        _write_reference(ref)
        cfg = _make_config(tmp, search_mode="site_aware", site_spec=SITE)
        ref_xyz, ref_seq = st.parse_ca_from_file(ref)
        records = st.parse_residues_from_file(ref, atom_names=None,
                                              include_hetatm=True)
        frames, _ = st.frames_from_records(records)
        site = sitemod.resolve_site(sitemod.parse_site_spec(SITE), records)
        assert site.indices == (1, 2, 6)

        folder = StubFolder(
            np.array(ref_xyz), ref_frames=frames[:, :3, :], want_frames=True
        )
        good = "G" * len(ref_seq)
        design = list(good)
        for idx, aa in SITE_INDICES.items():
            design[idx] = aa
        for i, aa in enumerate("VL"):
            design[50 + i] = aa          # satisfy the folder's GVL rule
        design = "".join(design)
        pairs = [(1.0, design)]

        v = Validator(cfg, folder, np.array(ref_xyz), ref_seq,
                      site=site, ref_frames=frames)

        # Site intact -> passes.
        ok = v.validate(set("GVL"), pairs, 1)
        assert ok.passed, ok.metrics
        assert ok.metrics.num_site_pass == 1
        assert ok.best.site_ok is True

        # Same fold, site displaced -> fails, and fails on the site.
        folder.break_site_for = {design}
        folder.cache = None
        bad = v.validate(set("GVL"), pairs, 1)
        assert not bad.passed, bad.metrics
        assert bad.metrics.num_site_pass == 0
        assert bad.metrics.max_tm is not None and bad.metrics.max_tm > 0.4, (
            "the global fold should still look fine; that is the point"
        )
        assert bad.metrics.site_bb_rmsd > cfg.site_max_bb_rmsd

        # With the gate switched off the same fold is accepted, and the site
        # numbers are still recorded.
        cfg.site_gate = False
        reported = v.validate(set("GVL"), pairs, 1)
        assert reported.passed
        assert reported.metrics.site_bb_rmsd is not None


def test_exempt_positions_do_not_make_designs_look_out_of_alphabet():
    """Without this, every site-aware design would be discarded.

    The in-alphabet filter rejects sequences containing residues outside the
    current alphabet. A site held at its native identity is outside it by
    construction, so the filter has to skip exempt positions.
    """
    from redaa.search import in_alphabet

    seq = "GGCGGGG"
    assert not in_alphabet(seq, set("G"), frozenset())
    assert in_alphabet(seq, set("G"), frozenset({2}))
    assert not in_alphabet(seq, set("G"), frozenset({3}))


def test_numbering_mismatch_aborts_the_domain_instead_of_guessing():
    with tempfile.TemporaryDirectory(prefix="redaa_mismatch_") as tmp:
        # Residue 2 is C, not H.
        pipe, code = _run(tmp, search_mode="site_aware", site_spec="H2")
        assert code == 0
        row = _read_csv(os.path.join(pipe.run_dir, "BATCH_SUMMARY.csv"))[0]
        assert row["status"].startswith("skipped_site_unresolved"), row["status"]
        assert "numbering" in row["status"].lower()
        # No replicate ran.
        assert not os.path.exists(
            os.path.join(pipe.run_dir, "testdomA00/run_1/COMPLETED")
        )


def test_site_that_resolves_to_nothing_is_skipped():
    with tempfile.TemporaryDirectory(prefix="redaa_empty_") as tmp:
        pipe, code = _run(tmp, search_mode="site_aware", site_spec="999")
        assert code == 0
        row = _read_csv(os.path.join(pipe.run_dir, "BATCH_SUMMARY.csv"))[0]
        assert row["status"] == "skipped_site_empty", row["status"]


def test_mpnn_residue_order_disagreement_is_fatal():
    """A silent index mismatch here would fix the wrong residues.

    The site layer and MPNN parse the same file independently. If they
    disagree about residue order, every fixed position is wrong and the run
    still completes, so the pipeline compares the two sequences up front.
    """
    with tempfile.TemporaryDirectory(prefix="redaa_order_") as tmp:
        os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
        ref = os.path.join(tmp, "inputs", "testdomA00.pdb")
        _write_reference(ref)
        cfg = _make_config(tmp, search_mode="site_aware", site_spec=SITE)

        class ReorderedMPNN(StubMPNN):
            def native_sequence(self, _pdb):
                base = NATIVE[: self.length]
                return base[::-1]

        class P(StubPipeline):
            def setup(self):
                super().setup()
                self.mpnn = ReorderedMPNN(length=len(NATIVE))

        pipe = P(cfg)
        pipe.ref_path = ref
        code = pipe.run()
        assert code == 0
        row = _read_csv(os.path.join(pipe.run_dir, "BATCH_SUMMARY.csv"))[0]
        assert row["status"].startswith("skipped_site_unresolved"), row["status"]
        assert "orders this structure's residues differently" in row["status"]


def test_ligand_derived_site_resolves_through_the_pipeline():
    with tempfile.TemporaryDirectory(prefix="redaa_lig_") as tmp:
        # A ligand parked beside the helix at the residue-3 end.
        n_xyz, ca, c_xyz = _backbone(len(NATIVE))
        centre = tuple(ca[2] + np.array([0.0, 2.2, 0.0]))
        pipe, code = _run(
            tmp, ligand=("ATP", centre, 4),
            search_mode="site_aware", site_spec="ligand:ATP@5.0",
            num_seqs_to_fold=3, prune_folds=1,
        )
        assert code == 0
        row = _read_csv(os.path.join(pipe.run_dir, "BATCH_SUMMARY.csv"))[0]
        assert row["status"] not in ("skipped_site_empty",), row["status"]
        assert int(row["site_n_positions"]) >= 1, row
        assert "ligand:ATP@5" in row["site_spec"], row["site_spec"]
        # Whatever the shell turned out to be, those positions were fixed.
        fixed = [f for f in pipe.mpnn.fixed_seen if f]
        assert fixed, "ligand-derived site never reached MPNN"
        assert all(f == fixed[0] for f in fixed)


def test_site_run_stops_early_if_the_model_has_no_backbone_frames():
    """One clear failure, not thousands of blank site columns.

    If the folding model's PDB output carries no N/CA/C, site geometry is
    unmeasurable -- and with the gate on, every candidate would fail for a
    reason unrelated to the science. The native fold is already paid for, so
    that is where this gets caught.
    """
    with tempfile.TemporaryDirectory(prefix="redaa_noframes_") as tmp:
        os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
        ref = os.path.join(tmp, "inputs", "testdomA00.pdb")
        _write_reference(ref)
        cfg = _make_config(tmp, search_mode="site_aware", site_spec=SITE)

        class CaOnlyPipeline(StubPipeline):
            def setup(self):
                super().setup()
                # A model that returns coordinates but no backbone frame.
                self.folder.ref_frames = None
                self.folder.want_frames = True

        pipe = CaOnlyPipeline(cfg)
        pipe.ref_path = ref
        code = pipe.run()
        assert code == 0
        row = _read_csv(os.path.join(pipe.run_dir, "BATCH_SUMMARY.csv"))[0]
        assert row["status"] == "skipped_no_backbone_frames_for_site", row["status"]
        # It stopped before burning a replicate on it.
        assert not os.path.exists(
            os.path.join(pipe.run_dir, "testdomA00/run_1/COMPLETED")
        )


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def _preflight(tmp, **over):
    os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
    ref = os.path.join(tmp, "inputs", "testdomA00.pdb")
    _write_reference(ref)
    cfg = _make_config(tmp, **over)
    pipe = StubPipeline(cfg)
    pipe.ref_path = ref
    return pipe, pipe.preflight()


def test_preflight_passes_on_a_healthy_setup():
    with tempfile.TemporaryDirectory(prefix="redaa_pf_") as tmp:
        pipe, code = _preflight(tmp)
        assert code == 0
        # It really did one of each rather than reporting success blindly.
        assert pipe.mpnn.calls >= 1
        assert pipe.folder.calls >= 1


def test_preflight_checks_the_site_when_one_is_configured():
    with tempfile.TemporaryDirectory(prefix="redaa_pf_site_") as tmp:
        pipe, code = _preflight(
            tmp, search_mode="site_aware", site_spec=SITE
        )
        assert code == 0
        # The fixed-position design happened, so the mechanism was exercised.
        assert any(f == SITE_INDICES for f in pipe.mpnn.fixed_seen)


def test_preflight_fails_with_no_inputs_instead_of_starting():
    with tempfile.TemporaryDirectory(prefix="redaa_pf_empty_") as tmp:
        os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
        cfg = _make_config(tmp)
        pipe = StubPipeline(cfg)
        pipe.ref_path = os.path.join(tmp, "inputs", "nothing.pdb")
        _write_reference(pipe.ref_path)
        os.remove(pipe.ref_path)
        assert pipe.preflight() == 1


def test_preflight_reports_a_bad_site_spec_without_running_anything():
    with tempfile.TemporaryDirectory(prefix="redaa_pf_bad_") as tmp:
        # Residue 2 is C, not H.
        pipe, code = _preflight(tmp, search_mode="site_aware", site_spec="H2")
        assert code == 1
        # It stopped at the site, before folding anything.
        assert pipe.folder.calls == 0


def test_preflight_catches_a_featurise_contract_break():
    """The bug that reached Colab: featurize raising on a missing key.

    Preflight exists so this surfaces in a minute rather than forty.
    """
    with tempfile.TemporaryDirectory(prefix="redaa_pf_feat_") as tmp:
        os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
        ref = os.path.join(tmp, "inputs", "testdomA00.pdb")
        _write_reference(ref)
        cfg = _make_config(tmp)

        class BrokenMPNN(StubMPNN):
            def design(self, *a, **k):
                raise KeyError("chain_mask")

        class P(StubPipeline):
            def setup(self):
                super().setup()
                self.mpnn = BrokenMPNN(length=len(NATIVE))

        pipe = P(cfg)
        pipe.ref_path = ref
        assert pipe.preflight() == 1


def test_preflight_catches_a_fold_model_with_no_backbone_frames():
    with tempfile.TemporaryDirectory(prefix="redaa_pf_frames_") as tmp:
        os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
        ref = os.path.join(tmp, "inputs", "testdomA00.pdb")
        _write_reference(ref)
        cfg = _make_config(tmp, search_mode="site_aware", site_spec=SITE)

        class CaOnly(StubPipeline):
            def setup(self):
                super().setup()
                self.folder.ref_frames = None
                self.folder.want_frames = True

        pipe = CaOnly(cfg)
        pipe.ref_path = ref
        assert pipe.preflight() == 1


def test_preflight_reports_why_the_fold_failed_not_just_that_it_did():
    """fold() returns None for five different reasons.

    Each one prints itself, but a caller holding only None could say no more
    than "produced no usable model" -- so the cause sat one line up the log,
    detached from the check that failed.
    """
    with tempfile.TemporaryDirectory(prefix="redaa_pf_why_") as tmp:
        os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
        ref = os.path.join(tmp, "inputs", "testdomA00.pdb")
        _write_reference(ref)
        cfg = _make_config(tmp)

        class MuteFolder(StubFolder):
            def fold(self, sequence, steps=None, want_frames=None):
                self.last_error = "CUDA out of memory at L=221. Lower ..."
                return None

        class P(StubPipeline):
            def setup(self):
                super().setup()
                broken = MuteFolder(self.folder.ref, want_frames=False)
                broken.last_error = ""
                self.folder = broken

        pipe = P(cfg)
        pipe.ref_path = ref

        import io, contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = pipe.preflight()
        out = buf.getvalue()
        assert code == 1
        # The reason travels with the check.
        assert "CUDA out of memory" in out, out[-600:]
        # And the attempt is described before it is made, so an opaque model
        # error is still interpretable.
        assert "precision=" in out and "FOLD_MAX_RESIDUES=" in out


def test_precision_hint_names_the_t4_problem_only_for_dtype_errors():
    from redaa.config import RedAAConfig
    from redaa.folding import FoldEngine

    cfg = RedAAConfig()
    cfg.fold_precision = "auto"
    engine = FoldEngine(cfg)
    hint = engine._precision_hint(RuntimeError("\"triu_tril_cuda_template\" not implemented for 'BFloat16'"))
    assert "FOLD_PRECISION" in hint and "T4" in hint
    # Not every error is a precision error.
    assert engine._precision_hint(ValueError("no such file")) == ""
    # If the user already moved off auto, say something useful instead.
    cfg.fold_precision = "fp16"
    assert "already" in FoldEngine(cfg)._precision_hint(RuntimeError("bad dtype"))


def test_auto_precision_follows_what_the_gpu_actually_supports():
    """"auto" has to mean auto, or the default is a trap.

    FOLD_PRECISION="auto" used to be handed to the model verbatim. A revision
    that rejects it raises, and the fallback is the *model default* -- bf16 --
    so a pre-Ampere GPU loaded happily and failed on the first forward pass.
    Asking the device costs nothing.
    """
    import types

    from redaa.folding import FoldEngine

    def _fake_torch(bf16, name):
        mod = types.ModuleType("torch")
        mod.cuda = types.SimpleNamespace(
            is_bf16_supported=lambda: bf16,
            get_device_name=lambda i: name,
            is_available=lambda: True,
        )
        return mod

    saved = sys.modules.get("torch")
    cfg = _make_config(tempfile.mkdtemp(prefix="redaa_prec_"))
    try:
        cfg.fold_precision = "auto"

        sys.modules["torch"] = _fake_torch(True, "NVIDIA A100-SXM4-40GB")
        precision, why = FoldEngine(cfg)._resolve_precision(True)
        assert precision == "bf16", (precision, why)

        sys.modules["torch"] = _fake_torch(False, "Tesla T4")
        precision, why = FoldEngine(cfg)._resolve_precision(True)
        assert precision == "fp16", (precision, why)
        assert "T4" in why and "bfloat16" in why

        # CPU gets fp32, not a half precision it cannot use well.
        assert FoldEngine(cfg)._resolve_precision(False)[0] == "fp32"

        # An explicit choice is never second-guessed.
        cfg.fold_precision = "bf16"
        precision, why = FoldEngine(cfg)._resolve_precision(True)
        assert precision == "bf16" and "requested" in why

        # A device that cannot answer falls back to the safe option rather
        # than raising during load.
        broken = types.ModuleType("torch")
        broken.cuda = types.SimpleNamespace(
            is_bf16_supported=lambda: (_ for _ in ()).throw(RuntimeError("x")),
            get_device_name=lambda i: (_ for _ in ()).throw(RuntimeError("x")),
            is_available=lambda: True,
        )
        sys.modules["torch"] = broken
        cfg.fold_precision = "auto"
        assert FoldEngine(cfg)._resolve_precision(True)[0] == "fp16"
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


def test_preflight_names_the_install_for_a_missing_model_dependency():
    """ESMFold2's implementation arrives with the weights and brings imports.

    "No module named 'biotite'" came from inside the model, on the first fold,
    after 25 GB had downloaded. Preflight checks the model card's list before
    folding so the pip line is in hand first.
    """
    with tempfile.TemporaryDirectory(prefix="redaa_pf_dep_") as tmp:
        os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
        ref = os.path.join(tmp, "inputs", "testdomA00.pdb")
        _write_reference(ref)
        cfg = _make_config(tmp)

        from redaa import colab as rc
        from redaa.folding import FoldEngine

        class QuietEngine(FoldEngine):
            """A real FoldEngine -- so the check applies -- that loads nothing.

            The dependency check deliberately runs only for the real engine,
            since a model card's list says nothing about an injected stub. A
            subclass is the honest way to exercise that branch.
            """

            def __init__(self, cfg, ref):
                super().__init__(cfg, verbose=False)
                self.model = object()
                self._ref = ref

            def load(self):
                pass

            def stats(self):
                return "quiet"

            def fold(self, sequence, steps=None, want_frames=None):
                return FoldResult(sequence, 88.0, 0.9, self._ref)

            def fold_many(self, sequences, steps=None):
                return [self.fold(s) for s in sequences]

        saved = rc.model_requirements
        try:
            rc.model_requirements = lambda model_id, revision="main": (
                [("redaa-absent-xyz", ">=1.0")], "stub card"
            )

            class P(StubPipeline):
                def setup(self):
                    super().setup()
                    self.folder = QuietEngine(self.cfg, self.folder.ref)

            pipe = P(cfg)
            pipe.ref_path = ref

            import contextlib
            import io

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                pipe.preflight()
            out = buf.getvalue()
            assert "fold model dependencies" in out
            assert "redaa-absent-xyz" in out
            assert "pip install redaa-absent-xyz>=1.0" in out
        finally:
            rc.model_requirements = saved


def test_missing_module_hint_maps_import_names_to_distributions():
    from redaa.config import RedAAConfig
    from redaa.folding import FoldEngine

    engine = FoldEngine(RedAAConfig())
    hint = engine._missing_module_hint(
        ModuleNotFoundError("No module named 'biotite'", name="biotite")
    )
    assert "pip install biotite" in hint
    # Import name and distribution name differ for these.
    assert "pip install biopython" in engine._missing_module_hint(
        ModuleNotFoundError("No module named 'Bio'", name="Bio")
    )
    assert "pip install msgpack-numpy" in engine._missing_module_hint(
        ModuleNotFoundError("x", name="msgpack_numpy")
    )
    # Only import errors get this treatment.
    assert engine._missing_module_hint(RuntimeError("bf16")) == ""


def test_preflight_skips_the_dependency_check_for_an_injected_engine():
    """A model card's dependency list says nothing about a stub.

    The check must apply to the engine that actually downloads and runs the
    model card's code, and be silent otherwise -- not fail a run whose folder
    was supplied by the caller.
    """
    with tempfile.TemporaryDirectory(prefix="redaa_pf_skip_") as tmp:
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _pipe, code = _preflight(tmp)
        out = buf.getvalue()
        assert code == 0
        assert "fold model dependencies" not in out


def test_preflight_warns_but_does_not_fail_on_a_weak_native_round_trip():
    """Machinery working and answers meaning something are different claims.

    If ESMFold2 cannot reproduce the reference from its own native sequence,
    no reduced-alphabet design's pass or fail on that domain can be trusted --
    but the pipeline is not broken, so this warns rather than stopping a run.
    """
    with tempfile.TemporaryDirectory(prefix="redaa_pf_weak_") as tmp:
        os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
        ref = os.path.join(tmp, "inputs", "testdomA00.pdb")
        _write_reference(ref)
        cfg = _make_config(tmp)

        class NativeFailsFolder(StubFolder):
            def fold(self, sequence, steps=None, want_frames=None):
                if sequence == NATIVE:
                    # Folds, but nowhere near the reference.
                    rng = np.random.default_rng(1)
                    return FoldResult(
                        sequence, 41.0, 0.2,
                        rng.normal(scale=30.0, size=self.ref.shape),
                    )
                return super().fold(sequence, steps=steps)

        class P(StubPipeline):
            def setup(self):
                super().setup()
                self.folder = NativeFailsFolder(self.folder.ref)

        pipe = P(cfg)
        pipe.ref_path = ref

        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = pipe.preflight()
        out = buf.getvalue()

        # Warned, not failed: the machinery is fine.
        assert code == 0, out[-800:]
        assert "[warn] native round-trip" in out, out[-800:]
        assert "NATIVE_TM_MIN" in out
        assert "unreliable" in out
        # And it is repeated in the summary, so it cannot scroll away.
        assert "worth knowing before a long run" in out


def test_preflight_flags_site_thresholds_the_model_cannot_reach():
    """A gate tighter than the native fold is unreachable for any design.

    Discovering that after a long run would waste it, and would look like the
    alphabet failing when it is really the threshold.
    """
    with tempfile.TemporaryDirectory(prefix="redaa_pf_thresh_") as tmp:
        cfg_over = {"search_mode": "site_aware", "site_spec": SITE,
                    "site_max_bb_rmsd": 0.001, "site_max_local_rmsd": 0.001,
                    "site_max_cb_dev": 0.001}
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _pipe, code = _preflight(tmp, **cfg_over)
        out = buf.getvalue()
        assert code == 0, out[-800:]
        assert "[warn] site thresholds are reachable" in out, out[-900:]
        assert "no design can" in out
        assert "Loosen" in out


def test_a_failing_verdict_names_the_gate_that_bound():
    """"maxTM=0.960 ... fail" is a puzzle, and it was asked about.

    The fold gate and the site gate are separate per-fold conditions. A design
    can fold beautifully and still not hold the site -- which is the point of
    a site gate -- but nothing on the console line said so.
    """
    from redaa import sites as sitemod
    from redaa import structure as st
    from redaa.search import Validator
    import redaa.pipeline as P

    with tempfile.TemporaryDirectory(prefix="redaa_reason_") as tmp:
        ref = os.path.join(tmp, "ref.pdb")
        _write_reference(ref)
        cfg = _make_config(tmp, search_mode="site_aware", site_spec=SITE)
        ref_xyz, ref_seq = st.parse_ca_from_file(ref)
        records = st.parse_residues_from_file(
            ref, atom_names=None, include_hetatm=True
        )
        frames, _ = st.frames_from_records(records)
        site = sitemod.resolve_site(sitemod.parse_site_spec(SITE), records)

        folder = StubFolder(
            np.array(ref_xyz), ref_frames=frames[:, :3, :], want_frames=True
        )
        design = list("G" * len(ref_seq))
        for idx, aa in SITE_INDICES.items():
            design[idx] = aa
        for i, aa in enumerate("VL"):
            design[50 + i] = aa
        design = "".join(design)
        pairs = [(1.0, design)]
        v = Validator(cfg, folder, np.array(ref_xyz), ref_seq,
                      site=site, ref_frames=frames)

        good = v.validate(set("GVL"), pairs, 1)
        assert good.passed and good.reason == ""
        assert P._verdict(good) == "PASS"

        # Folds just as well, site displaced.
        folder.break_site_for = {design}
        bad = v.validate(set("GVL"), pairs, 1)
        assert not bad.passed
        assert bad.reason == "site", bad.reason
        assert P._verdict(bad) == "fail[site]"
        # The fold really was fine, so TM cannot be what the reader blames.
        assert bad.metrics.max_tm > cfg.min_tm_score
        # And the site numbers are on the line now.
        fields = P._site_fields(bad.metrics)
        assert "siteBB=" in fields and "cbDev=" in fields

        # With the gate off, the same fold is accepted and nothing is blamed.
        cfg.site_gate = False
        reported = v.validate(set("GVL"), pairs, 1)
        assert reported.passed and reported.reason == ""


def test_a_fold_failure_is_blamed_on_the_fold_not_the_site():
    from redaa import sites as sitemod
    from redaa import structure as st
    from redaa.search import Validator

    with tempfile.TemporaryDirectory(prefix="redaa_reason2_") as tmp:
        ref = os.path.join(tmp, "ref.pdb")
        _write_reference(ref)
        cfg = _make_config(tmp, search_mode="site_aware", site_spec=SITE)
        ref_xyz, ref_seq = st.parse_ca_from_file(ref)
        records = st.parse_residues_from_file(
            ref, atom_names=None, include_hetatm=True
        )
        frames, _ = st.frames_from_records(records)
        site = sitemod.resolve_site(sitemod.parse_site_spec(SITE), records)
        folder = StubFolder(
            np.array(ref_xyz), ref_frames=frames[:, :3, :], want_frames=True
        )
        v = Validator(cfg, folder, np.array(ref_xyz), ref_seq,
                      site=site, ref_frames=frames)

        # A sequence the stub folds to garbage: no G/V/L requirement met.
        junk = "A" * len(ref_seq)
        chars = list(junk)
        for idx, aa in SITE_INDICES.items():
            chars[idx] = aa
        out = v.validate(set("A") | set(SITE_INDICES.values()), [(1.0, "".join(chars))], 1)
        assert not out.passed
        assert out.reason in ("TM", "TM and site"), out.reason


def test_no_site_configured_still_reports_a_plain_verdict():
    """The site columns must not appear in an alphabet_only run."""
    import redaa.pipeline as P
    from redaa.search import Metrics, Validation

    metrics = Metrics(max_tm=0.2, min_rmsd=20.0)
    assert P._site_fields(metrics) == ""
    assert P._verdict(Validation(passed=False, reason="TM")) == "fail[TM]"
    assert P._verdict(Validation(passed=True)) == "PASS"


def test_measure_only_frees_the_site_but_still_measures_it():
    """Ligand-shaped design without preserving any native identity.

    The companion to MPNN_MODEL_TYPE=ligand_mpnn: the ligand shapes the design
    through atom context, nothing is held native, and the pocket geometry is
    still gated -- otherwise there is no evidence the ligand did anything.
    """
    with tempfile.TemporaryDirectory(prefix="redaa_measure_") as tmp:
        pipe, code = _preflight(
            tmp, search_mode="site_aware", site_spec=SITE,
            site_residue_policy="measure_only",
        )
        assert code == 0
        # Nothing was fixed.
        assert all(f == {} for f in pipe.mpnn.fixed_seen), [
            f for f in pipe.mpnn.fixed_seen if f
        ][:3]


def test_measure_only_does_not_require_the_native_identity():
    """Requiring it would fail every candidate by definition."""
    from redaa import sites as sitemod
    from redaa import structure as st
    from redaa.search import Validator

    with tempfile.TemporaryDirectory(prefix="redaa_measure2_") as tmp:
        ref = os.path.join(tmp, "ref.pdb")
        _write_reference(ref)
        ref_xyz, ref_seq = st.parse_ca_from_file(ref)
        records = st.parse_residues_from_file(
            ref, atom_names=None, include_hetatm=True
        )
        frames, _ = st.frames_from_records(records)
        site = sitemod.resolve_site(sitemod.parse_site_spec(SITE), records)
        folder = StubFolder(
            np.array(ref_xyz), ref_frames=frames[:, :3, :], want_frames=True
        )

        # A design with the WRONG residues at the site, folding correctly.
        chars = list("G" * len(ref_seq))
        for i, aa in enumerate("VL"):
            chars[50 + i] = aa
        design = "".join(chars)
        assert not sitemod.site_identity_kept(site, design)
        pairs = [(1.0, design)]

        strict = _make_config(tmp, search_mode="site_aware", site_spec=SITE)
        v = Validator(strict, folder, np.array(ref_xyz), ref_seq,
                      site=site, ref_frames=frames)
        assert v.require_identity
        assert not v.validate(set("GVL"), pairs, 1).passed

        free = _make_config(tmp, search_mode="site_aware", site_spec=SITE,
                            site_residue_policy="measure_only")
        v2 = Validator(free, folder, np.array(ref_xyz), ref_seq,
                       site=site, ref_frames=frames)
        assert not v2.require_identity
        out = v2.validate(set("GVL"), pairs, 1)
        assert out.passed, out.reason
        # The geometry was still measured, not waived.
        assert out.metrics.site_bb_rmsd is not None

        # Site positions are not exempt either: the design must be fully
        # in-alphabet, since nothing is held native.
        assert v2.exempt == frozenset()


def test_measure_only_is_rejected_with_scaffold_then_site():
    """There is nothing to graft, so the combination is meaningless."""
    from redaa.config import RedAAConfig

    cfg = RedAAConfig()
    cfg.search_mode = "scaffold_then_site"
    cfg.site_spec = "H46"
    cfg.site_residue_policy = "measure_only"
    problems = cfg.validate()
    assert any("nothing to graft" in p for p in problems), problems


# ---------------------------------------------------------------------------
# Starting alphabet, exclusions, and inverse folding
# ---------------------------------------------------------------------------


def test_the_search_starts_from_the_configured_alphabet():
    with tempfile.TemporaryDirectory(prefix="redaa_start_") as tmp:
        pipe, code = _run(tmp, starting_alphabet=("G", "S"))
        assert code == 0
        build = _read_csv(
            os.path.join(pipe.run_dir, "testdomA00/run_1/alphabet_building_log.csv")
        )
        init = next(r for r in build if r["Stage"] == "INIT")
        assert init["Current_Set"] == "GS", init


def test_excluded_residues_never_appear_anywhere_in_a_run():
    """Not offered, not seeded, and not in the baseline either."""
    barred = ("C", "W", "M", "H", "Y")
    with tempfile.TemporaryDirectory(prefix="redaa_excl_") as tmp:
        pipe, code = _run(tmp, exclude_aas=barred)
        assert code == 0
        row = _read_csv(os.path.join(pipe.run_dir, "BATCH_SUMMARY.csv"))[0]
        assert not set(row["final_alphabet"]) & set(barred), row
        assert not set(row["pruned_alphabet"]) & set(barred), row

        build = _read_csv(
            os.path.join(pipe.run_dir, "testdomA00/run_1/alphabet_building_log.csv")
        )
        offered = {r["Candidate_AA"] for r in build}
        assert not offered & set(barred), sorted(offered & set(barred))

        # The baseline must be the reachable ceiling, not an unreachable
        # 20-letter score. The stub's mean score falls with alphabet size, so
        # a 15-letter baseline scores higher than a 20-letter one would.
        assert any(r["Stage"] == "BASELINE" for r in build)
        design_calls = [set(a) for a in pipe.mpnn.alphabets_seen]
        assert design_calls, "no MPNN calls recorded"
        assert not any(c & set(barred) for c in design_calls), (
            "MPNN was asked for an excluded residue"
        )


def test_excluding_a_residue_the_structure_needs_warns_and_wins():
    """Config validation cannot know about a structure's disulfides."""
    with tempfile.TemporaryDirectory(prefix="redaa_exclchem_") as tmp:
        os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
        ref = os.path.join(tmp, "inputs", "testdomA00.pdb")
        _write_reference(ref)
        cfg = _make_config(tmp, exclude_aas=("C",))

        class NeedsCys(StubPipeline):
            def setup(self):
                super().setup()

        import contextlib
        import io

        import redaa.pipeline as P

        real = P.analyze_requirements

        class _Req:
            required_aas = ("C",)
            n_disulfides = 1
            cis_pro_count = 0
            notes = ("1 disulfide bond",)

        try:
            P.analyze_requirements = lambda _p: _Req()
            pipe = NeedsCys(cfg)
            pipe.ref_path = ref
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                pipe.run()
            out = buf.getvalue()
            assert "EXCLUDE_AAS bars C" in out, out[-700:]
            assert "exclusion wins" in out
            row = _read_csv(os.path.join(pipe.run_dir, "BATCH_SUMMARY.csv"))[0]
            assert "C" not in row["final_alphabet"], row
        finally:
            P.analyze_requirements = real


def test_an_excluded_site_residue_is_fatal_under_in_alphabet():
    with tempfile.TemporaryDirectory(prefix="redaa_exclsite_") as tmp:
        pipe, code = _run(
            tmp, search_mode="site_aware", site_spec=SITE,
            site_residue_policy="in_alphabet", exclude_aas=("C",),
        )
        assert code == 0
        row = _read_csv(os.path.join(pipe.run_dir, "BATCH_SUMMARY.csv"))[0]
        assert row["status"].startswith("skipped_excluded_site_residue"), row
        # ... and harmless under exempt, where it is not an alphabet letter.
    with tempfile.TemporaryDirectory(prefix="redaa_exclsite2_") as tmp:
        pipe, code = _run(
            tmp, search_mode="site_aware", site_spec=SITE,
            site_residue_policy="exempt", exclude_aas=("C",),
            num_seqs_to_fold=3, prune_folds=1,
        )
        assert code == 0
        row = _read_csv(os.path.join(pipe.run_dir, "BATCH_SUMMARY.csv"))[0]
        assert not row["status"].startswith("skipped"), row


def test_inverse_fold_uses_the_alphabet_the_run_settled_on():
    with tempfile.TemporaryDirectory(prefix="redaa_inv_") as tmp:
        pipe, _ = _run(tmp)
        summary = _read_csv(os.path.join(pipe.run_dir, "BATCH_SUMMARY.csv"))[0]
        expected = summary["pruned_alphabet"]

        pipe.cfg.resume_from = pipe.run_dir     # as the notebook does
        second = StubPipeline(pipe.cfg)
        second.ref_path = pipe.ref_path
        csv_path = second.inverse_fold(num_seq=20, n_folds=4)
        rows = _read_csv(csv_path)
        assert rows, "no sequences produced"
        # Pruned, not forward: pruning is what makes the answer minimal.
        assert all(r["alphabet"] == expected for r in rows), rows[0]
        assert all(set(r["sequence"]) <= set(expected) for r in rows)
        # Ranked best first, and every row carries its own numbers.
        assert [int(r["rank"]) for r in rows] == sorted(int(r["rank"]) for r in rows)
        assert all(r["tm"] and r["rmsd"] and r["length"] for r in rows)
        fasta = csv_path.replace("INVERSE_FOLD.csv", "INVERSE_FOLD.fasta")
        assert os.path.exists(fasta)
        assert open(fasta).read().startswith(">testdomA00_rank1 ")


def test_inverse_fold_honours_an_explicit_alphabet_and_the_site():
    with tempfile.TemporaryDirectory(prefix="redaa_inv2_") as tmp:
        pipe, _ = _run(tmp, search_mode="site_aware", site_spec=SITE,
                       num_seqs_to_fold=3, prune_folds=1)
        pipe.cfg.resume_from = pipe.run_dir
        second = StubPipeline(pipe.cfg)
        second.ref_path = pipe.ref_path
        csv_path = second.inverse_fold(
            alphabet="GLVA", num_seq=20, n_folds=4
        )
        rows = _read_csv(csv_path)
        assert all(r["alphabet"] == "AGLV" for r in rows), rows[0]["alphabet"]
        # The site was held, and measured.
        for row in rows:
            seq = row["sequence"]
            for idx, aa in SITE_INDICES.items():
                assert seq[idx] == aa, (idx, seq[idx], aa)
            assert row["site_bb_rmsd"] not in ("", "N/A")
            assert row["site_ok"] in ("0", "1")


def test_inverse_fold_without_a_summary_or_an_alphabet_says_so():
    with tempfile.TemporaryDirectory(prefix="redaa_inv3_") as tmp:
        os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
        ref = os.path.join(tmp, "inputs", "testdomA00.pdb")
        _write_reference(ref)
        pipe = StubPipeline(_make_config(tmp))
        pipe.ref_path = ref
        try:
            pipe.inverse_fold(num_seq=5, n_folds=2)
        except SystemExit as exc:
            assert "produced nothing" in str(exc) or "BATCH_SUMMARY" in str(exc)
        else:
            raise AssertionError("expected a clear refusal")


def test_an_exempt_site_holding_an_excluded_residue_is_called_out():
    """The trap for "remove this residue from the protein entirely".

    Exempt positions keep their NATIVE identity, so an exempt site residue
    whose type is barred puts that residue straight back into the sequence --
    at the positions that matter most, and without a word.
    """
    import contextlib
    import io

    with tempfile.TemporaryDirectory(prefix="redaa_exempt_excl_") as tmp:
        os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
        ref = os.path.join(tmp, "inputs", "testdomA00.pdb")
        _write_reference(ref)
        # SITE holds C2, D3 and H7; bar C, which one of them is.
        cfg = _make_config(
            tmp, search_mode="site_aware", site_spec=SITE,
            site_residue_policy="exempt", exclude_aas=("C",),
            num_seqs_to_fold=3, prune_folds=1,
        )
        pipe = StubPipeline(cfg)
        pipe.ref_path = ref
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pipe.run()
        out = buf.getvalue()
        assert "EXCLUDE_AAS bars C" in out, out[-900:]
        assert "still contain" in out
        assert "measure_only" in out

        # And the CSV says so rather than leaving it to be assumed.
        row = _read_csv(os.path.join(pipe.run_dir, "BATCH_SUMMARY.csv"))[0]
        assert row["excluded_aas"] == "C"
        assert row["excluded_present"] == "C", row
        assert int(row["excluded_count"]) >= 1


def test_a_clean_exclusion_reports_nothing_present():
    with tempfile.TemporaryDirectory(prefix="redaa_excl_clean_") as tmp:
        pipe, code = _run(tmp, exclude_aas=("C", "W"))
        assert code == 0
        row = _read_csv(os.path.join(pipe.run_dir, "BATCH_SUMMARY.csv"))[0]
        assert row["excluded_aas"] == "CW"
        assert row["excluded_present"] == "", row
        assert int(row["excluded_count"]) == 0
        assert "C" not in row["best_sequence"] and "W" not in row["best_sequence"]


def test_inverse_fold_reports_excluded_survivors_per_sequence():
    with tempfile.TemporaryDirectory(prefix="redaa_inv_excl_") as tmp:
        pipe, _ = _run(tmp, exclude_aas=("C", "W", "M"))
        pipe.cfg.resume_from = pipe.run_dir
        second = StubPipeline(pipe.cfg)
        second.ref_path = pipe.ref_path
        rows = _read_csv(second.inverse_fold(num_seq=20, n_folds=4))
        assert rows
        for row in rows:
            assert row["excluded_present"] == "", row
            assert int(row["excluded_count"]) == 0
            assert not set(row["sequence"]) & set("CWM")


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
