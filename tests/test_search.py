"""Unit tests for sequence selection, pruning and chemical requirements."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redaa import search as se  # noqa: E402
from redaa.chemistry import analyze_requirements, dihedral  # noqa: E402
from redaa.config import RedAAConfig  # noqa: E402

import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# Sequence selection
# ---------------------------------------------------------------------------


def _pairs(n, length=40, seed=0):
    """n synthetic (score, sequence) pairs, ascending by score."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        seq = "".join(rng.choice(list("AGVL"), size=length))
        out.append((1.0 + i * 0.01, seq))
    return out


def test_selection_prefixes_are_nested():
    """Every smaller budget must be a strict prefix of a larger one.

    This is what makes two-stage screening nearly free: the single fold spent
    looking at a candidate is reused verbatim when that candidate is promoted
    to the full budget, because it is the same sequence.
    """
    pairs = _pairs(30, seed=1)
    full = se.select_sequences(pairs, 9)
    for k in (1, 2, 3, 6, 9):
        assert se.select_sequences(pairs, k) == full[:k], k


def test_selection_spans_the_score_distribution():
    """A 3-sequence budget must not be the three best MPNN scores.

    MPNN scores naturalness under the full 20-residue model, not foldability
    under a restricted alphabet, so top-k by score under-samples exactly the
    region we care about.
    """
    pairs = _pairs(30, seed=2)
    chosen = se.select_sequences(pairs, 3)
    by_seq = {seq: score for score, seq in pairs}
    ranks = sorted(sorted(by_seq.values()).index(by_seq[s]) for s in chosen)
    assert ranks[0] < 10, ranks     # something from the best third
    assert ranks[-1] >= 20, ranks   # something from the worst third


def test_selection_respects_the_budget_and_dedupes():
    pairs = _pairs(30, seed=3)
    assert len(se.select_sequences(pairs, 5)) == 5
    assert len(set(se.select_sequences(pairs, 9))) == 9
    assert se.select_sequences(pairs, 0) == []
    assert se.select_sequences([], 5) == []


def test_selection_returns_everything_when_pool_is_small():
    pairs = _pairs(2, seed=4)
    assert len(se.select_sequences(pairs, 9)) == 2


def test_diversity_filter_drops_near_duplicates_and_keeps_the_best():
    base = "AGVL" * 10
    near = base[:-2] + "AG"           # 95% identical
    far = "VLAG" * 10
    pairs = [(1.0, base), (1.1, near), (1.2, far)]
    kept = se.diversity_filter(pairs, max_identity=0.70)
    kept_seqs = [s for _, s in kept]
    assert base in kept_seqs          # best score always survives
    assert near not in kept_seqs
    assert far in kept_seqs


def test_metrics_rank_key_prefers_high_tm_then_low_rmsd():
    a = se.Metrics(max_tm=0.60, min_rmsd=3.0)
    b = se.Metrics(max_tm=0.60, min_rmsd=1.0)
    c = se.Metrics(max_tm=0.70, min_rmsd=9.0)
    assert b.rank_key() > a.rank_key()
    assert c.rank_key() > b.rank_key()
    assert se.Metrics().rank_key() < a.rank_key()


# ---------------------------------------------------------------------------
# Backward pruning
# ---------------------------------------------------------------------------


def _prune_with(alphabet, required, cfg, protected=None):
    """Prune, treating any alphabet that contains *required* as passing."""
    tested: list[frozenset] = []

    def trial_fn(subset, _label):
        tested.append(frozenset(subset))
        ok = set(required) <= set(subset)
        return ok, se.Metrics(max_tm=0.9 if ok else 0.1, min_rmsd=1.0 if ok else 9.0)

    pruned, removed, trials = se.backward_prune(
        cfg, alphabet, trial_fn, protected=protected
    )
    return pruned, removed, trials, tested


def test_exhaustive_pruning_finds_the_minimal_alphabet():
    cfg = RedAAConfig()
    cfg.exhaustive_pruning_threshold = 8
    cfg.pruning_downward_closure = False
    pruned, removed, _, _ = _prune_with(list("AGVL"), "GVL", cfg)
    assert pruned == list("GLV"), pruned
    assert removed == 1


def test_downward_closure_gives_the_same_answer_with_fewer_trials():
    """Closure must prune work, not change the result."""
    cfg_off = RedAAConfig()
    cfg_off.exhaustive_pruning_threshold = 8
    cfg_off.pruning_downward_closure = False
    cfg_on = RedAAConfig()
    cfg_on.exhaustive_pruning_threshold = 8
    cfg_on.pruning_downward_closure = True

    alphabet = list("AGVLST")
    off = _prune_with(alphabet, "GVL", cfg_off)
    on = _prune_with(alphabet, "GVL", cfg_on)

    assert on[0] == off[0] == list("GLV"), (on[0], off[0])
    assert on[2] < off[2], (on[2], off[2])


def test_closure_never_tests_a_subset_of_a_failed_set():
    cfg = RedAAConfig()
    cfg.exhaustive_pruning_threshold = 8
    cfg.pruning_downward_closure = True
    _pruned, _removed, _trials, tested = _prune_with(list("AGVLS"), "GVL", cfg)

    failed = [s for s in tested if not {"G", "V", "L"} <= s]
    for candidate in tested:
        for bad in failed:
            assert not (candidate < bad), (
                f"tested {sorted(candidate)}, a strict subset of failed "
                f"{sorted(bad)}"
            )


def test_pruning_keeps_protected_residues():
    cfg = RedAAConfig()
    cfg.exhaustive_pruning_threshold = 8
    pruned, _removed, _trials, tested = _prune_with(
        list("ACGVL"), "GVL", cfg, protected={"C"}
    )
    assert "C" in pruned, pruned
    for subset in tested:
        assert "C" in subset, sorted(subset)


def test_greedy_path_used_above_the_exhaustive_threshold():
    cfg = RedAAConfig()
    cfg.exhaustive_pruning_threshold = 4
    cfg.pruning_downward_closure = True
    pruned, removed, _, _ = _prune_with(list("AGVLST"), "GVL", cfg)
    assert set("GVL") <= set(pruned), pruned
    assert removed >= 1
    # Greedy removes one or two at a time and may stop above the true minimum,
    # but it must never drop a required residue.
    assert len(pruned) <= 6


def test_pruning_stops_when_nothing_can_be_removed():
    cfg = RedAAConfig()
    cfg.exhaustive_pruning_threshold = 8
    pruned, removed, _, _ = _prune_with(list("GVL"), "GVL", cfg)
    assert pruned == list("GLV")
    assert removed == 0


# ---------------------------------------------------------------------------
# Chemical requirements
# ---------------------------------------------------------------------------


def _atom(serial, name, resname, resseq, xyz, chain="A"):
    x, y, z = xyz
    return (
        f"ATOM  {serial:>5d} {name:^4s} {resname:>3s} {chain}{resseq:>4d}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00"
    )


def test_dihedral_matches_known_angles():
    p0 = np.array([1.0, 1.0, 0.0])
    p1 = np.array([0.0, 1.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3_cis = np.array([1.0, 0.0, 0.0])
    p3_trans = np.array([-1.0, 0.0, 0.0])
    assert abs(dihedral(p0, p1, p2, p3_cis) - 0.0) < 1e-6
    assert abs(abs(dihedral(p0, p1, p2, p3_trans)) - 180.0) < 1e-6


def test_detects_disulfide_and_requires_cys(tmp_path="/tmp"):
    path = os.path.join(tmp_path, "redaa_ss_test.pdb")
    lines = [
        _atom(1, "CA", "CYS", 1, (0.0, 0.0, 0.0)),
        _atom(2, "SG", "CYS", 1, (0.0, 0.0, 1.0)),
        _atom(3, "CA", "CYS", 2, (5.0, 0.0, 0.0)),
        _atom(4, "SG", "CYS", 2, (0.0, 0.0, 3.05)),  # 2.05 A from the first SG
        "END",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    req = analyze_requirements(path)
    assert req.n_disulfides == 1, req
    assert req.required_aas == ["C"], req
    os.remove(path)


def test_ignores_distant_cysteine_pairs():
    path = "/tmp/redaa_nos_test.pdb"
    lines = [
        _atom(1, "SG", "CYS", 1, (0.0, 0.0, 0.0)),
        _atom(2, "SG", "CYS", 2, (0.0, 0.0, 6.0)),
        "END",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    req = analyze_requirements(path)
    assert req.n_disulfides == 0
    assert req.required_aas == []
    os.remove(path)


def test_detects_cis_proline_and_ignores_trans():
    for omega_is_cis, expect in ((True, 1), (False, 0)):
        path = f"/tmp/redaa_pro_{omega_is_cis}.pdb"
        # omega is CA(i-1)-C(i-1)-N(i)-CA(i); place CA(i) to make it 0 or 180.
        ca_i = (1.0, 0.0, 0.0) if omega_is_cis else (-1.0, 0.0, 0.0)
        lines = [
            _atom(1, "CA", "ALA", 1, (1.0, 1.0, 0.0)),
            _atom(2, "C", "ALA", 1, (0.0, 1.0, 0.0)),
            _atom(3, "N", "PRO", 2, (0.0, 0.0, 0.0)),
            _atom(4, "CA", "PRO", 2, ca_i),
            "END",
        ]
        with open(path, "w") as fh:
            fh.write("\n".join(lines))
        req = analyze_requirements(path)
        assert req.cis_pro_count == expect, (omega_is_cis, req)
        assert ("P" in req.required_aas) == bool(expect), req
        os.remove(path)


# ---------------------------------------------------------------------------
# MPNN checkpoint resolution
# ---------------------------------------------------------------------------


def test_checkpoint_name_is_per_model_type():
    from redaa.mpnn import DEFAULT_CHECKPOINTS, checkpoint_name

    # Exact names, taken from LigandMPNN's run.py argparse defaults. Pinned
    # rather than pattern-matched because the upstream spelling is not what
    # you would guess: model_type "soluble_mpnn" loads "solublempnn_...",
    # with no underscore. A prefix assertion let that typo through once.
    assert checkpoint_name("protein_mpnn") == "proteinmpnn_v_48_020.pt"
    assert checkpoint_name("ligand_mpnn") == "ligandmpnn_v_32_010_25.pt"
    assert checkpoint_name("soluble_mpnn") == "solublempnn_v_48_020.pt"
    # Each type maps to a distinct file; they are not interchangeable.
    assert len(set(DEFAULT_CHECKPOINTS.values())) == len(DEFAULT_CHECKPOINTS)
    # An unknown type falls back rather than crashing.
    assert checkpoint_name("something_else") == "proteinmpnn_v_48_020.pt"


def test_default_checkpoint_path_follows_the_model_type():
    import os as _os

    from redaa.mpnn import _default_checkpoint

    p_protein = _default_checkpoint("/repo", "protein_mpnn")
    p_ligand = _default_checkpoint("/repo", "ligand_mpnn")
    assert p_protein != p_ligand
    assert _os.path.dirname(p_protein).endswith("model_params")


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
