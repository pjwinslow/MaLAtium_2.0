"""Functional-site specs, resolution, metrics and presets.

No GPU, no torch, no model weights, no network: the structures are written
here as small synthetic PDBs, so this runs anywhere.

The emphasis is on the failure modes that would otherwise be invisible.  A
site spec resolved against the wrong numbering, or against a file whose
residue order differs from MPNN's, produces a run that completes normally and
reports a meaningless alphabet -- so the tests below care at least as much
about mismatches being caught as about the happy path working.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redaa import sites as S  # noqa: E402
from redaa import structure as st  # noqa: E402

AA1_TO_3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}


def _write(path, seq, start=1, chain="A", ligand=None, skip_c=()):
    """A straight-line backbone, one residue per 3.8 A, plus optional ligand.

    ``skip_c`` omits the C atom of those residue numbers, which is how a
    residue with an incomplete backbone is simulated.
    """
    lines = []
    serial = 1
    for i, aa in enumerate(seq):
        num = start + i
        x = 3.8 * i
        for name, dx, dy in (("N", -0.6, 0.0), ("CA", 0.0, 0.0), ("C", 0.7, 0.5)):
            if name == "C" and num in skip_c:
                continue
            lines.append(
                f"ATOM  {serial:>5d}  {name:<3s} {AA1_TO_3[aa]} {chain}{num:>4d}    "
                f"{x + dx:>8.3f}{dy:>8.3f}{0.0:>8.3f}  1.00  0.00"
            )
            serial += 1
    lines.append("TER")
    for comp, centre, n_atoms in ligand or []:
        for k in range(n_atoms):
            cx, cy, cz = centre
            lines.append(
                f"HETATM{serial:>5d}  C{k:<2d} {comp:>3s} {chain}{900:>4d}    "
                f"{cx + 0.6 * k:>8.3f}{cy:>8.3f}{cz:>8.3f}  1.00  0.00"
            )
            serial += 1
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _records(path):
    return st.parse_residues_from_file(
        path, atom_names=None, include_hetatm=True
    )


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------


def test_parse_residue_forms():
    spec = S.parse_site_spec("H46, D81 C151")
    assert [t.resseq for t in spec.residues] == [46, 81, 151]
    assert [t.expect_aa for t in spec.residues] == ["H", "D", "C"]
    assert spec.ligands == ()


def test_parse_chain_icode_and_bare_numbers():
    spec = S.parse_site_spec("A/H46 B:81 151 46A")
    assert [(t.chain, t.resseq, t.icode, t.expect_aa) for t in spec.residues] == [
        ("A", 46, "", "H"),
        ("B", 81, "", None),
        (None, 151, "", None),
        (None, 46, "A", None),
    ]


def test_parse_negative_numbering():
    # TEV's deposited construct numbers its purification tag -8..-1, so
    # negative author numbering has to parse.
    spec = S.parse_site_spec("-8 -1 5")
    assert [t.resseq for t in spec.residues] == [-8, -1, 5]


def test_parse_ligand_and_cutoff():
    spec = S.parse_site_spec("ligand:atp ligand:ADP@6.5")
    assert [(l.comp_id, l.cutoff) for l in spec.ligands] == [
        ("ATP", 4.5), ("ADP", 6.5)
    ]


def test_parse_range_and_comments():
    spec = S.parse_site_spec("range:10-13  # the oxyanion loop\nH46")
    assert [t.resseq for t in spec.residues] == [10, 11, 12, 13, 46]


def test_parse_default_chain_applies():
    spec = S.parse_site_spec("46 B/81", default_chain="A")
    assert [t.chain for t in spec.residues] == ["A", "B"]


def test_parse_deduplicates_keeping_the_asserted_identity():
    spec = S.parse_site_spec("H46 46")
    assert len(spec.residues) == 1
    assert spec.residues[0].expect_aa == "H"


def test_parse_rejects_nonsense():
    for bad in ("Z46", "ligand:ATP@99", "range:20-10", "hello", "ligand:"):
        try:
            S.parse_site_spec(bad)
        except S.SiteError:
            continue
        raise AssertionError(f"{bad!r} should not have parsed")


def test_empty_spec_is_falsey():
    assert not S.parse_site_spec("   ")
    assert not S.parse_site_spec("")


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_resolve_maps_author_numbering_to_sequence_index():
    with tempfile.TemporaryDirectory() as tmp:
        # Numbering starts at 20, so index == resseq - 20. A spec written in
        # author numbering must not be read as a 0-based index.
        p = _write(os.path.join(tmp, "x.pdb"), "ACDEFGHIK", start=20)
        site = S.resolve_site(S.parse_site_spec("D22 G25"), _records(p))
        assert site.indices == (2, 5)
        assert site.native_aas == "DG"
        assert not site.warnings, site.warnings


def test_resolve_flags_numbering_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(os.path.join(tmp, "x.pdb"), "ACDEFGHIK", start=1)
        # Residue 3 is D, not H.
        site = S.resolve_site(S.parse_site_spec("H3"), _records(p))
        assert any(w.startswith("NUMBERING MISMATCH") for w in site.warnings), (
            site.warnings
        )


def test_resolve_reports_missing_residue():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(os.path.join(tmp, "x.pdb"), "ACDEFG", start=1)
        site = S.resolve_site(S.parse_site_spec("A99"), _records(p))
        assert len(site) == 0
        assert any("not in the prepared structure" in w for w in site.warnings)


def test_contact_shell_finds_the_pocket_and_orders_by_distance():
    with tempfile.TemporaryDirectory() as tmp:
        # Residues sit at x = 0, 3.8, 7.6, ... ; the ligand sits beside the
        # third one, so that residue and its neighbours are the shell.
        p = _write(
            os.path.join(tmp, "x.pdb"), "AAAAAAAA",
            ligand=[("ATP", (7.6, 2.0, 0.0), 1)],
        )
        recs = _records(p)
        site = S.resolve_site(S.parse_site_spec("ligand:ATP@4.0"), recs)
        assert site.indices, "expected a non-empty shell"
        assert 2 in site.indices
        assert all(p_.role == S.ROLE_CONTACT for p_ in site.positions)
        assert site.ligand_atom_counts == {"ATP": 1}
        # A resolved site is ordered by sequence position, which is what the
        # metrics and the CSV want...
        assert list(site.indices) == sorted(site.indices)
        # ... while contact_shell itself ranks by distance, which is what the
        # SITE_CONTACT_MAX truncation relies on to keep the closest residues.
        found, _warns = S.contact_shell(recs, ("ATP",), 4.0)
        dists = [d for _i, d, _n in found]
        assert dists == sorted(dists), dists
        assert all(d > 0 for d in dists)


def test_contact_shell_cutoff_widens_the_shell():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(
            os.path.join(tmp, "x.pdb"), "AAAAAAAA",
            ligand=[("ATP", (7.6, 2.0, 0.0), 1)],
        )
        recs = _records(p)
        narrow = S.resolve_site(S.parse_site_spec("ligand:ATP@3.0"), recs)
        wide = S.resolve_site(S.parse_site_spec("ligand:ATP@7.0"), recs)
        assert set(narrow.indices) <= set(wide.indices)
        assert len(wide) > len(narrow)


def test_contact_max_keeps_the_closest():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(
            os.path.join(tmp, "x.pdb"), "AAAAAAAAAA",
            ligand=[("ATP", (11.4, 1.0, 0.0), 3)],
        )
        recs = _records(p)
        full = S.resolve_site(S.parse_site_spec("ligand:ATP@8.0"), recs)
        capped = S.resolve_site(
            S.parse_site_spec("ligand:ATP@8.0"), recs, contact_max=2
        )
        assert len(capped) == 2
        assert set(capped.indices) <= set(full.indices)
        assert any("kept the 2 closest" in w for w in capped.warnings)


def test_missing_ligand_is_reported_not_guessed():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(os.path.join(tmp, "x.pdb"), "AAAA")
        site = S.resolve_site(S.parse_site_spec("ligand:ATP"), _records(p))
        assert len(site) == 0
        assert any("no atoms found" in w for w in site.warnings)


def test_solvent_is_never_taken_for_a_ligand():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(
            os.path.join(tmp, "x.pdb"), "AAAA",
            ligand=[("HOH", (3.8, 1.5, 0.0), 1), ("GOL", (3.8, 1.5, 0.0), 3)],
        )
        recs = _records(p)
        # Water never survives parsing at all; glycerol is excluded from the
        # auto-detected candidates but can still be named on purpose.
        assert all(r.resname != "HOH" for r in recs)
        assert S.ligand_groups(recs) == []
        assert len(S.ligand_groups(recs, ("GOL",))) == 1


def test_named_residue_keeps_catalytic_role_inside_a_shell():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(
            os.path.join(tmp, "x.pdb"), "AACAAAAA",
            ligand=[("ATP", (7.6, 2.0, 0.0), 1)],
        )
        site = S.resolve_site(
            S.parse_site_spec("C3 ligand:ATP@4.0"), _records(p)
        )
        by_index = {p_.index: p_ for p_ in site.positions}
        assert by_index[2].role == S.ROLE_CATALYTIC
        # It still picks up the measured distance to the ligand.
        assert by_index[2].min_distance is not None


def test_oversized_site_warns():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(os.path.join(tmp, "x.pdb"), "ACDEFGHIKL")
        site = S.resolve_site(S.parse_site_spec("range:1-10"), _records(p))
        assert len(site) == 10
        assert any("site covers" in w for w in site.warnings), site.warnings


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _frames(path):
    recs = st.parse_residues_from_file(path)
    frames, _ = st.frames_from_records(recs)
    return frames


def test_site_metrics_are_zero_against_itself():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(os.path.join(tmp, "x.pdb"), "ACDEFGHIK")
        site = S.resolve_site(S.parse_site_spec("C2 F5 H7"), _records(p))
        f = _frames(p)
        m = S.measure_site(site, f, f, "ACDEFGHIK")
        assert m.bb_rmsd < 1e-6 and m.local_rmsd < 1e-6
        assert m.cb_max_dev < 1e-6
        assert m.identity_kept


def test_site_metrics_ignore_rigid_body_motion():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(os.path.join(tmp, "x.pdb"), "ACDEFGHIK")
        site = S.resolve_site(S.parse_site_spec("C2 F5 H7"), _records(p))
        f = _frames(p)
        theta = 0.7
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ])
        moved = (f.reshape(-1, 3) @ rot + np.array([11.0, -4.0, 2.0])).reshape(f.shape)
        m = S.measure_site(site, f, moved)
        # A whole-structure rotation and translation is not a site failure.
        assert m.bb_rmsd < 1e-6, m.bb_rmsd
        assert m.local_rmsd < 1e-6, m.local_rmsd
        assert m.cb_max_dev < 1e-6, m.cb_max_dev


def test_local_and_global_distinguish_placement_from_distortion():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(os.path.join(tmp, "x.pdb"), "ACDEFGHIKLMNPQ")
        site = S.resolve_site(S.parse_site_spec("C2 F5 H7"), _records(p))
        f = _frames(p)

        # Move the site rigidly: its internal geometry is intact (local ~0)
        # but it is in the wrong place (global large).
        shifted = f.copy()
        for i in site.indices:
            shifted[i] += 3.0
        m = S.measure_site(site, f, shifted)
        assert m.bb_rmsd > 2.0, m.bb_rmsd
        assert m.local_rmsd < 1e-6, m.local_rmsd

        # Pull one site residue away from the others: internal geometry is
        # now wrong too, and the distance matrix notices.
        warped = f.copy()
        warped[site.indices[1]] += 3.0
        m2 = S.measure_site(site, f, warped)
        assert m2.local_rmsd > 0.5, m2.local_rmsd
        assert m2.cb_max_dev > 1.0, m2.cb_max_dev


def test_identity_loss_is_detected_and_repairable():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(os.path.join(tmp, "x.pdb"), "ACDEFGHIK")
        site = S.resolve_site(S.parse_site_spec("C2 F5 H7"), _records(p))
        broken = "AADEAGAIK"  # C2, F5 and H7 all replaced by A
        assert not S.site_identity_kept(site, broken)
        repaired = S.apply_site_to_sequence(site, broken)
        assert S.site_identity_kept(site, repaired)
        assert repaired == "ACDEFGHIK"
        # Only the site positions are touched.
        other = S.apply_site_to_sequence(site, "GGGGGGGGG")
        assert other == "GCGGFGHGG"


def test_gate_thresholds_and_switching_checks_off():
    m = S.SiteMetrics(bb_rmsd=1.2, local_rmsd=0.8, cb_max_dev=1.0, n_positions=3)
    assert m.passed(1.5, 1.0, 1.5)
    assert not m.passed(1.0, 1.0, 1.5)  # bb too high
    assert not m.passed(1.5, 0.5, 1.5)  # local too high
    assert not m.passed(1.5, 1.0, 0.5)  # cb dev too high
    # A zero limit switches that check off rather than making it impossible.
    assert m.passed(0.0, 0.0, 0.0)

    lost = S.SiteMetrics(bb_rmsd=0.1, local_rmsd=0.1, cb_max_dev=0.1,
                         identity_kept=False)
    assert not lost.passed(1.5, 1.0, 1.5)
    assert lost.passed(1.5, 1.0, 1.5, require_identity=False)


def test_unmeasurable_site_is_not_a_pass():
    # No frames available: every requested check is unmeasurable, which must
    # read as "did not pass", never as "nothing to object to".
    blank = S.SiteMetrics(n_positions=3)
    assert not blank.passed(1.5, 1.0, 1.5)
    # ... unless every check is switched off, in which case there is nothing
    # being asked for.
    assert blank.passed(0.0, 0.0, 0.0)


def test_incomplete_backbone_at_a_site_residue_is_unmeasurable():
    with tempfile.TemporaryDirectory() as tmp:
        # Residue 5 has no C atom, so it has no frame.
        p = _write(os.path.join(tmp, "x.pdb"), "ACDEFGHIK", skip_c=(5,))
        recs = _records(p)
        site = S.resolve_site(S.parse_site_spec("F5"), recs)
        frames, valid = st.frames_from_records(recs)
        assert not valid[4]
        m = S.measure_site(site, frames, frames)
        assert m.bb_rmsd is None
        assert not m.passed(1.5, 1.0, 1.5)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_presets_exist_and_parse():
    assert set(S.PRESETS) == {"atp", "tev"}
    for key in S.PRESETS:
        pre = S.preset(key)
        spec = S.parse_site_spec(pre.site_spec)
        assert spec, key
        assert pre.fetch_spec and pre.title and pre.notes
        assert pre.mpnn_model_type in ("protein_mpnn", "soluble_mpnn", "ligand_mpnn")
        # A ligand-aware model is only meaningful if a ligand is kept.
        if pre.mpnn_model_type == "ligand_mpnn":
            assert pre.keep_ligands, key
        assert key.lower() in S.describe_presets().lower()


def test_atp_preset_is_a_ligand_derived_pocket():
    pre = S.preset("atp")
    spec = S.parse_site_spec(pre.site_spec)
    assert [l.comp_id for l in spec.ligands] == ["ATP"]
    assert "ATP" in pre.keep_ligands
    assert pre.mpnn_model_type == "ligand_mpnn"


def test_tev_preset_names_the_triad_with_expected_identities():
    pre = S.preset("tev")
    spec = S.parse_site_spec(pre.site_spec)
    named = {(t.expect_aa, t.resseq) for t in spec.residues}
    # The catalytic triad, asserted by identity so a numbering change fails
    # loudly instead of silently fixing the wrong residues.
    assert {("H", 46), ("D", 81), ("C", 151)} <= named
    # The tag is excluded by the fetch spec's residue range.
    assert pre.fetch_spec.endswith(":1-221")


def test_unknown_preset_names_the_alternatives():
    try:
        S.preset("nope")
    except S.SiteError as exc:
        assert "atp" in str(exc) and "tev" in str(exc)
    else:
        raise AssertionError("expected SiteError")


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
            import traceback

            traceback.print_exc()
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
