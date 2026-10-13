"""Unit tests for the NumPy structural metrics.

These run without torch, BioPython or a GPU, so the metric layer can be
checked on a login node (or in CI) before a GPU job is ever queued.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redaa import structure as st  # noqa: E402


def _pdb_line(serial, atom, resname, chain, resseq, xyz, altloc=" ", record="ATOM  "):
    x, y, z = xyz
    return (
        f"{record}{serial:>5d} {atom:^4s}{altloc}{resname:>3s} {chain}"
        f"{resseq:>4d}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00"
    )


def _helix(n, seed=0):
    """An ideal alpha-helix CA trace: 1.5 A rise, 100 degrees per residue."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    ang = np.deg2rad(100.0) * t
    xyz = np.stack([2.3 * np.cos(ang), 2.3 * np.sin(ang), 1.5 * t], axis=1)
    return xyz + rng.normal(scale=1e-9, size=xyz.shape)


def _random_rigid(xyz, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return xyz @ q.T + rng.normal(scale=10.0, size=3)


def test_parse_filters_and_order():
    lines = [
        "MODEL        1",
        _pdb_line(1, "CA", "GLY", "A", 2, (1.0, 2.0, 3.0)),
        _pdb_line(2, "CB", "GLY", "A", 2, (9.0, 9.0, 9.0)),       # not CA
        _pdb_line(3, "CA", "ALA", "A", 1, (4.0, 5.0, 6.0)),       # out of order
        _pdb_line(4, "CA", "SER", "A", 3, (7.0, 8.0, 9.0), altloc="A"),
        _pdb_line(5, "CA", "SER", "A", 3, (0.0, 0.0, 0.0), altloc="B"),  # altloc B
        _pdb_line(6, "CA", "HOH", "A", 4, (1.0, 1.0, 1.0), record="HETATM"),
        _pdb_line(7, "CA", "MSE", "A", 5, (2.0, 2.0, 2.0), record="HETATM"),
        "ENDMDL",
        _pdb_line(8, "CA", "TRP", "A", 6, (5.0, 5.0, 5.0)),       # second model
    ]
    coords, seq = st.parse_ca_from_text("\n".join(lines))
    assert seq == "AGS", seq
    assert coords.shape == (3, 3)
    # Sorted by residue number, so ALA(1) comes first despite its file position.
    assert np.allclose(coords[0], [4.0, 5.0, 6.0])
    assert np.allclose(coords[2], [7.0, 8.0, 9.0])


def test_parse_multi_chain_keeps_chain_order():
    lines = [
        _pdb_line(1, "CA", "GLY", "B", 1, (0.0, 0.0, 0.0)),
        _pdb_line(2, "CA", "ALA", "A", 1, (1.0, 0.0, 0.0)),
        _pdb_line(3, "CA", "SER", "B", 2, (2.0, 0.0, 0.0)),
    ]
    _, seq = st.parse_ca_from_text("\n".join(lines))
    assert seq == "GSA", seq


def test_kabsch_recovers_rigid_transform():
    ref = _helix(60)
    moved = _random_rigid(ref, seed=3)
    assert st.kabsch_rmsd(moved, ref) < 1e-8


def test_rmsd_matches_hand_computation_on_pure_translation():
    ref = _helix(30)
    # A translation is removed exactly by superposition, so RMSD must be ~0.
    assert st.kabsch_rmsd(ref + np.array([3.0, -2.0, 7.0]), ref) < 1e-9
    # A uniform outward displacement along z cannot be superposed away.
    noisy = ref.copy()
    noisy[:, 2] += np.where(np.arange(30) % 2 == 0, 1.0, -1.0)
    assert abs(st.kabsch_rmsd(noisy, ref) - 1.0) < 1e-3


def test_tm_score_identity_is_one():
    ref = _helix(120)
    assert abs(st.tm_score(_random_rigid(ref, seed=5), ref) - 1.0) < 1e-6


def test_tm_d0_matches_published_formula():
    for length in (20, 50, 100, 250):
        expected = max(0.5, 1.24 * (length - 15) ** (1 / 3) - 1.8)
        assert abs(st.tm_d0(length) - expected) < 1e-12
    assert st.tm_d0(10) == 0.5


def test_tm_score_decreases_with_noise_and_stays_in_range():
    rng = np.random.default_rng(11)
    ref = _helix(150)
    scores = []
    for sigma in (0.0, 0.5, 1.5, 4.0, 15.0):
        pred = ref + rng.normal(scale=sigma, size=ref.shape)
        tm = st.tm_score(pred, ref)
        assert 0.0 <= tm <= 1.0
        scores.append(tm)
    assert scores == sorted(scores, reverse=True), scores
    assert scores[0] > 0.99
    assert scores[-1] < 0.3


def test_tm_score_beats_single_superposition_on_a_hinge():
    """A domain-swing case: a global fit is dragged off the matched core.

    The seed-refined search should lock onto the well-superposed half and
    report a materially higher TM-score than one global Kabsch fit does.
    """
    ref = _helix(160)
    pred = ref.copy()
    pred[80:] += np.array([60.0, 0.0, 0.0])  # swing the C-terminal half away

    refined = st.tm_score(pred, ref)

    rot, trans = st.kabsch(pred, ref)
    dist = np.linalg.norm(st.apply_transform(pred, rot, trans) - ref, axis=1)
    d0 = st.tm_d0(len(ref))
    global_only = float((1.0 / (1.0 + (dist / d0) ** 2)).sum() / len(ref))

    assert refined > global_only + 0.25, (refined, global_only)
    # Half the chain matches well, so the score should sit near 0.5.
    assert 0.40 < refined < 0.60, refined


def test_align_pair_handles_internal_gap_without_frameshift():
    ref = _helix(40)
    ref_seq = "ACDEFGHIKL" * 4
    drop = list(range(0, 15)) + list(range(20, 40))  # prediction misses 15..19
    pred = ref[drop]
    pred_seq = "".join(ref_seq[i] for i in drop)

    ref_p, pred_p, l_ref = st.align_pair(ref, ref_seq, pred, pred_seq)
    assert l_ref == 40
    assert len(ref_p) == len(pred_p) == 35
    # Correct pairing means the paired atoms are already identical.
    assert np.allclose(ref_p, pred_p)
    assert st.kabsch_rmsd(pred_p, ref_p) < 1e-8


def test_align_pair_normalises_tm_by_full_reference_length():
    ref = _helix(100)
    ref_seq = "ACDEFGHIKL" * 10
    keep = list(range(50))
    pred, pred_seq = ref[keep], ref_seq[:50]

    ref_p, pred_p, l_ref = st.align_pair(ref, ref_seq, pred, pred_seq)
    assert l_ref == 100
    tm = st.tm_score(pred_p, ref_p, l_ref=l_ref)
    # A perfect match over half the reference scores about 0.5, not 1.0.
    assert 0.45 < tm < 0.55, tm


def test_compare_end_to_end_without_tmtools():
    ref = _helix(90)
    ref_seq = "ACDEFGHIKL" * 9
    pred = _random_rigid(ref, seed=7)
    rmsd, tm = st.compare(ref, ref_seq, pred, ref_seq, use_tmtools=False)
    assert rmsd is not None and rmsd < 1e-8
    assert tm is not None and abs(tm - 1.0) < 1e-6


def test_compare_returns_none_for_degenerate_input():
    empty = np.zeros((0, 3))
    assert st.compare(empty, "", empty, "") == (None, None)


def main():
    failures = []
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
