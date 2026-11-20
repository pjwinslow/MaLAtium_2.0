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
import shutil
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
        bad = [w for w in site.warnings if w.startswith("NUMBERING MISMATCH")]
        assert bad, site.warnings
        # The message has to answer the question it provokes -- what IS this
        # structure? -- or every mismatch costs a round trip.
        message = bad[0]
        assert "has D3 at that position" in message
        assert "Around it:" in message
        assert "A1" in message and "F5" in message, message
        # And where the expected residue actually is, if anywhere.
        assert "H appears at H7" in message, message


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


def test_common_crystallisation_additives_are_not_ligand_candidates():
    """Imidazole and friends turn up beside real ligands constantly.

    A GlyRS structure offered IMD as a binding candidate next to its actual
    adenylate analogue. The list is for *auto-detection* only -- anything can
    still be named explicitly -- so it is better to be generous here than to
    advertise a buffer as a pocket.
    """
    for code in ("IMD", "BTB", "B3P", "MRD", "SPD", "SPM", "DTT", "GOL",
                 "EDO", "MES", "SO4", "HOH"):
        assert code in S.SOLVENT_COMPONENTS, code
    # Real ligands must not have been swept up with them.
    for code in ("ATP", "ADP", "G5A", "6NT", "B12", "NAD", "FAD", "HEM"):
        assert code not in S.SOLVENT_COMPONENTS, code
        assert code not in S.ION_COMPONENTS, code


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
    assert set(S.PRESETS) == {"atp", "tev", "glyrs", "alars", "alars_cat"}
    from redaa.config import parse_site_policy

    for key in S.PRESETS:
        pre = S.preset(key)
        spec = S.parse_site_spec(pre.site_spec)
        assert spec, key
        assert pre.fetch_spec and pre.title and pre.notes
        assert pre.mpnn_model_type in ("protein_mpnn", "soluble_mpnn", "ligand_mpnn")
        # A ligand-aware model is only meaningful if a ligand is kept, or a
        # partner chain is -- either gives it atoms to design against.
        if pre.mpnn_model_type == "ligand_mpnn":
            assert pre.keep_ligands or pre.keep_partners, key
        # A spec cannot name a partner chain the preset does not retain: the
        # shell would resolve to nothing and the site would silently shrink.
        for tok in spec.partners:
            assert tok.chain in [c.upper() for c in pre.keep_partners], key
        # Conversely, retaining a chain and never referring to it would carry
        # a tRNA through every fold for no reason.
        for chain in pre.keep_partners:
            assert any(t.chain == chain.upper() for t in spec.partners), key
        # Every recommended setting has to be one the config layer can take,
        # with a value it accepts. A preset that cannot be applied is worse
        # than no preset: it fails after the structure has been fetched.
        from redaa.config import RedAAConfig

        cfg = RedAAConfig()
        for env_key, value in pre.settings:
            assert env_key in RedAAConfig.ENV_TO_FIELD, (key, env_key)
            cfg.apply_env_setting(env_key, value)
        assert cfg.validate() == [] or all(
            "SITE_SPEC" in p or "needs a site" in p for p in cfg.validate()
        ), (key, cfg.validate())
        assert key.lower() in S.describe_presets().lower()


def test_glyrs_preset_pairs_chain_a_with_the_trna_it_binds():
    """The pairing is the whole point, and getting it backwards is silent.

    In 5E6M each tRNA binds predominantly one subunit: chain E binds protein
    A, chain C binds protein B. Fetching chain A and keeping chain C would
    resolve a small, wrong interface and still run to completion.
    """
    pre = S.preset("glyrs")
    assert pre.fetch_spec.endswith("_A")
    spec = S.parse_site_spec(pre.site_spec)
    assert [t.chain for t in spec.partners] == ["E"]
    assert [t.comp_id for t in spec.ligands] == ["GAP"]
    settings = dict(pre.settings)
    # One blanket policy would force 18 letters before the search starts.
    assert "interface=measure_only" in settings["SITE_RESIDUE_POLICY"]
    # The interface spans the whole chain, so it is measured, not gated.
    assert settings["SITE_GATE_ROLES"] == "catalytic,contact"
    # The worst-pair CB check grows with site size; the RMS one does not.
    assert float(settings["SITE_MAX_CB_DEV"]) == 0.0
    assert float(settings["SITE_MAX_CB_RMS_DEV"]) > 0.0


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


# ---------------------------------------------------------------------------
# Polymer partners and per-role policies
# ---------------------------------------------------------------------------


def _write_with_partner(path, seq, partner_chain="E", partner_at=(1, 2)):
    """A chain plus a short nucleic partner written as HETATM, as prepare does."""
    _write(path, seq)
    lines = open(path).read().splitlines()
    lines = [l for l in lines if l != "END"]
    serial = 5000
    for n, i in enumerate(partner_at, start=1):
        for k, name in enumerate(("P", "C1'", "N9")):
            lines.append(
                f"HETATM{serial:>5d}  {name:<3s}   G {partner_chain}{n:>4d}    "
                f"{3.8 * i:>8.3f}{3.0:>8.3f}{0.3 * k:>8.3f}  1.00  0.00"
            )
            serial += 1
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def test_parse_partner_token_forms():
    spec = S.parse_site_spec("partner:E partner=c@6.5")
    assert [(t.chain, t.cutoff) for t in spec.partners] == [("E", 4.5), ("C", 6.5)]
    assert spec.describe() == "partner:E@4.5 partner:C@6.5"
    assert bool(S.parse_site_spec("partner:E"))


def test_partner_cutoff_out_of_range_is_rejected():
    for text in ("partner:E@0.5", "partner:E@40"):
        try:
            S.parse_site_spec(text)
        except S.SiteError as exc:
            assert "out of range" in str(exc), exc
        else:
            raise AssertionError(f"{text} should have been rejected")


def test_partner_shell_finds_the_interface():
    tmp = tempfile.mkdtemp()
    try:
        path = _write_with_partner(os.path.join(tmp, "p.pdb"), "ACDEFGH")
        recs = _records(path)
        assert sum(len(g.atoms) for g in S.partner_atoms(recs, "E")) == 6
        # Chain letters are matched case-insensitively, as elsewhere.
        assert S.partner_atoms(recs, "e")
        # 3.5 A, not 4: at 4 the straight-line fixture puts residue 0 in the
        # shell too, which says nothing about partner handling.
        found, warns = S.partner_shell(recs, "E", cutoff=3.5)
        assert warns == []
        assert sorted(i for i, _d, _n in found) == [1, 2]
        assert all(2.5 < d < 2.7 for _i, d, _n in found), found
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_partner_shell_says_so_when_the_chain_was_never_retained():
    """The quiet failure this replaces: an interface silently resolving to nothing."""
    tmp = tempfile.mkdtemp()
    try:
        path = _write(os.path.join(tmp, "p.pdb"), "ACDEFGH",
                      ligand=[("ATP", (3.8, 3.0, 0.0), 3)])
        found, warns = S.partner_shell(_records(path), "E")
        assert found == []
        assert len(warns) == 1
        assert "no retained atoms" in warns[0], warns
        assert "SITE_KEEP_PARTNERS" in warns[0], warns
        # Naming what IS retained is what turns this into a fixable message.
        assert "retained non-polymer chains: A" in warns[0], warns
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_site_gives_partner_contacts_the_interface_role():
    tmp = tempfile.mkdtemp()
    try:
        path = _write_with_partner(os.path.join(tmp, "p.pdb"), "ACDEFGH")
        recs = _records(path)
        site = S.resolve_site(S.parse_site_spec("partner:E@3.5"), recs)
        assert site.indices == (1, 2)
        assert {p.role for p in site.positions} == {S.ROLE_INTERFACE}
        assert site.interface_indices == (1, 2)
        assert site.partners == ("E",)
        assert site.partner_atom_counts == {"E": 6}
        assert "interface residues to chain E" in site.describe()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_catalytic_residue_in_the_interface_keeps_its_role():
    """Roles decide policy, so the stronger claim has to win."""
    tmp = tempfile.mkdtemp()
    try:
        path = _write_with_partner(os.path.join(tmp, "p.pdb"), "ACDEFGH")
        site = S.resolve_site(
            S.parse_site_spec("C2 partner:E@3.5"), _records(path)
        )
        by_index = {p.index: p for p in site.positions}
        assert by_index[1].role == S.ROLE_CATALYTIC
        assert by_index[1].min_distance is not None  # gained the measurement
        assert by_index[2].role == S.ROLE_INTERFACE
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _plan_fixture(tmp):
    path = _write_with_partner(
        os.path.join(tmp, "p.pdb"), "ACDEFGH", partner_at=(4, 5)
    )
    # ATP beside residues 0-1, partner beside residues 4-5.
    lines = open(path).read().splitlines()
    lines = [l for l in lines if l != "END"]
    serial = 7000
    for k in range(3):
        lines.append(
            f"HETATM{serial + k:>5d}  C{k:<2d} ATP A 900    "
            f"{0.6 * k:>8.3f}{3.0:>8.3f}{0.0:>8.3f}  1.00  0.00"
        )
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return S.resolve_site(
        S.parse_site_spec("ligand:ATP@3.5 partner:E@3.5"), _records(path)
    )


def test_site_plan_splits_held_from_free_by_role():
    from redaa.config import parse_site_policy

    tmp = tempfile.mkdtemp()
    try:
        site = _plan_fixture(tmp)
        contacts = set(site.indices) - set(site.interface_indices)
        iface = set(site.interface_indices)
        assert contacts and iface, (contacts, iface)

        policies, problems = parse_site_policy("exempt,interface=measure_only")
        assert problems == []
        plan = S.site_plan(site, policies)
        # The pocket is held; the interface is free but still measured.
        assert set(plan.fixed) == contacts
        assert plan.exempt == frozenset(contacts)
        assert plan.forced_types == frozenset()
        assert plan.n_measured == len(iface)
        assert plan.any_fixed

        # in_alphabet on the pocket alone forces only the pocket's letters.
        policies, _ = parse_site_policy(
            "contact=in_alphabet,interface=measure_only"
        )
        plan = S.site_plan(site, policies)
        assert set(plan.fixed) == contacts
        assert plan.exempt == frozenset()
        assert plan.forced_types == frozenset(
            p.native_aa for p in site.positions if p.index in contacts
        )

        # measure_only everywhere holds nothing at all.
        plan = S.site_plan(site, parse_site_policy("measure_only")[0])
        assert plan.fixed == {}
        assert not plan.any_fixed
        assert plan.n_measured == len(site)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_site_plan_of_no_site_is_empty_but_reports_its_policies():
    from redaa.config import parse_site_policy

    plan = S.site_plan(None, parse_site_policy("in_alphabet")[0])
    assert plan.fixed == {} and not plan.any_fixed
    assert plan.by_role["interface"] == "in_alphabet"


def test_identity_is_only_required_where_the_plan_holds_a_position():
    """Under measure_only the design is SUPPOSED to change those residues."""
    tmp = tempfile.mkdtemp()
    try:
        site = _plan_fixture(tmp)
        native = "ACDEFGH"
        # Mutate one interface position and one held position.
        iface_i = site.interface_indices[0]
        held_i = (set(site.indices) - set(site.interface_indices)).pop()
        mutated_iface = list(native)
        mutated_iface[iface_i] = "W"
        mutated_iface = "".join(mutated_iface)

        assert not S.site_identity_kept(site, mutated_iface)
        held = {
            i: native[i] for i in site.indices
            if i not in site.interface_indices
        }
        assert S.site_identity_kept(site, mutated_iface, only=held)

        mutated_held = list(native)
        mutated_held[held_i] = "W"
        assert not S.site_identity_kept(site, "".join(mutated_held), only=held)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_apply_site_leaves_free_positions_alone():
    tmp = tempfile.mkdtemp()
    try:
        site = _plan_fixture(tmp)
        designed = "WWWWWWW"
        held = {
            i: "ACDEFGH"[i] for i in site.indices
            if i not in site.interface_indices
        }
        out = S.apply_site_to_sequence(site, designed, only=held)
        for i in held:
            assert out[i] == held[i]
        for i in site.interface_indices:
            assert out[i] == "W"
        # Without `only` every site position is forced, as before.
        full = S.apply_site_to_sequence(site, designed)
        for i in site.indices:
            assert full[i] == "ACDEFGH"[i]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_alars_cat_is_the_same_case_cut_to_fit_a_smaller_card():
    full = S.preset("alars")
    cut = S.preset("alars_cat")
    # Same site, same ligand, same partner chain, same policies.
    assert cut.site_spec == full.site_spec
    assert cut.keep_ligands == full.keep_ligands
    assert cut.keep_partners == full.keep_partners
    assert dict(cut.settings) == dict(full.settings)
    # It differs only in carrying a residue range.
    assert full.fetch_spec == "3wqy_A"
    assert cut.fetch_spec == "3wqy_A:1-490"
    from redaa.fetch import parse_spec

    assert parse_spec(cut.fetch_spec).resid_range == (1, 490)
    # And it says what the cut gives up, since the answer is not AlaRS.
    assert "catalytic domain" in cut.notes


def test_alars_preset_pairs_chain_a_with_its_trna_and_frees_barred_residues():
    """The Ala-free case: hold the pocket, except where that reinstates Ala."""
    pre = S.preset("alars")
    assert pre.fetch_spec.endswith("_A")
    spec = S.parse_site_spec(pre.site_spec)
    assert [t.chain for t in spec.partners] == ["C"]
    assert [t.comp_id for t in spec.ligands] == ["A5A"]
    settings = dict(pre.settings)
    assert "interface=measure_only" in settings["SITE_RESIDUE_POLICY"]
    assert settings["SITE_GATE_ROLES"] == "catalytic,contact"
    # Without this, EXCLUDE_AAS=A would be contradicted by the held pocket.
    assert settings["SITE_FREE_EXCLUDED"] == "1"
    # Size-stable CB check, as for every site past a few positions.
    assert float(settings["SITE_MAX_CB_DEV"]) == 0.0
    assert float(settings["SITE_MAX_CB_RMS_DEV"]) > 0.0
    # The interface is gated, and more loosely than the pocket.
    assert float(settings["SITE_REST_MAX_LOCAL_RMSD"]) > 1.0


def test_cb_max_dev_grows_with_site_size_but_the_rms_does_not():
    """The reason SITE_MAX_CB_RMS_DEV exists, pinned down as a measurement.

    ``cb_max_dev`` is a worst-of-all-pairs statistic and the pair count grows
    as n^2, so at IDENTICAL backbone accuracy it rises with the size of the
    site. A threshold calibrated on a six-residue triad is therefore a much
    harsher test on a twenty-residue pocket -- which is exactly how a real
    run came to reject every candidate. The RMS over pairs is size-stable.
    """
    rng = np.random.default_rng(4)
    n_res = 120
    # A plausible backbone: a straight-ish chain with some curvature, so the
    # CB-CB distance matrix is not degenerate.
    t = np.arange(n_res)
    nca = np.stack(
        [
            np.stack([3.8 * t + dx, 2.0 * np.sin(t / 6.0), np.cos(t / 9.0)], 1)
            for dx in (-0.6, 0.0, 0.7)
        ],
        axis=1,
    )
    ref = st.frames_with_cb(nca)

    def perturbed():
        """Correlated error: a fold model is not wrong independently per atom."""
        raw = rng.normal(size=(n_res, 3))
        k = np.exp(-0.5 * (np.arange(-30, 31) / 10.0) ** 2)
        k /= k.sum()
        sm = np.stack(
            [np.convolve(raw[:, d], k, mode="same") for d in range(3)], 1
        )
        sm *= 1.0 / sm.std()
        return st.frames_with_cb(nca + sm[:, None, :])

    def average(n):
        maxes, rmss = [], []
        for _ in range(25):
            pos = sorted(rng.choice(n_res, size=n, replace=False).tolist())
            mx, rms = S.st.site_distance_deviation(
                ref, perturbed(), pos, atom="CB"
            )
            maxes.append(mx)
            rmss.append(rms)
        return float(np.mean(maxes)), float(np.mean(rmss))

    small_max, small_rms = average(6)
    large_max, large_rms = average(60)

    # The worst pair gets materially worse with no change in accuracy.
    assert large_max > small_max * 1.25, (small_max, large_max)
    # The RMS does not move: that is what makes it comparable across cases.
    assert abs(large_rms - small_rms) < 0.25 * small_rms, (small_rms, large_rms)


def test_the_rms_cb_check_gates_independently():
    m = S.SiteMetrics(
        bb_rmsd=0.5, local_rmsd=0.4, cb_max_dev=6.0, cb_rms_dev=1.2,
        n_positions=20,
    )
    # Worst-pair check off, RMS check on: this passes, and should.
    assert m.passed(1.5, 1.0, 0.0, require_identity=False, max_cb_rms_dev=2.0)
    # Tighten the RMS check and it fails on that alone.
    assert not m.passed(
        1.5, 1.0, 0.0, require_identity=False, max_cb_rms_dev=1.0
    )
    # Both off: neither CB check can reject it.
    assert m.passed(1.5, 1.0, 0.0, require_identity=False)
    # An unmeasurable RMS is not a pass when the check is asked for.
    blind = S.SiteMetrics(bb_rmsd=0.5, local_rmsd=0.4, n_positions=20)
    assert not blind.passed(
        1.5, 1.0, 0.0, require_identity=False, max_cb_rms_dev=2.0
    )


def test_gate_roles_split_the_site_into_gated_and_recorded():
    from redaa.config import parse_site_policy

    tmp = tempfile.mkdtemp()
    try:
        site = _plan_fixture(tmp)
        iface = set(site.interface_indices)
        pocket = set(site.indices) - iface
        policies, _ = parse_site_policy("exempt,interface=measure_only")

        plan = S.site_plan(site, policies, gate_roles=("catalytic", "contact"))
        assert plan.gated == frozenset(pocket)
        assert plan.recorded == frozenset(iface)
        # The line has to say which it is: a group with its own thresholds is
        # gated, and calling it "recorded only" would misreport the run.
        assert "recorded only" in plan.describe()
        assert "loosely gated" in plan.describe(rest_gated=True)
        assert "recorded only" not in plan.describe(rest_gated=True)

        # No gate_roles means every position is gated, as before.
        plan = S.site_plan(site, policies)
        assert plan.gated == frozenset(site.indices)
        assert plan.recorded == frozenset()
        assert "recorded only" not in plan.describe()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_measure_site_gates_a_subset_and_records_the_rest():
    """Loosening the gate must not mean losing the measurement."""
    tmp = tempfile.mkdtemp()
    try:
        site = _plan_fixture(tmp)
        recs = _records(os.path.join(tmp, "p.pdb"))
        poly = [r for r in recs if not r.is_hetatm]
        frames, _v = st.frames_from_records(poly)
        ref = st.frames_with_cb(frames[:, :3, :])

        pocket = frozenset(set(site.indices) - set(site.interface_indices))
        # Displace the interface badly and leave the pocket alone.
        pred = ref.copy()
        for i in site.interface_indices:
            pred[i] += 6.0

        m = S.measure_site(site, ref, pred, gate_positions=pocket)
        assert m.n_positions == len(pocket)
        assert m.n_recorded == len(site.interface_indices)
        # The gated pocket is untouched, so its local geometry is perfect...
        assert m.local_rmsd is not None and m.local_rmsd < 1e-6
        # ...while the recorded half shows the damage rather than hiding it.
        assert m.recorded is not None
        assert m.recorded.bb_rmsd > 1.0

        # Without gate_positions the two are pooled and nothing is recorded
        # separately -- the old behaviour.
        pooled = S.measure_site(site, ref, pred)
        assert pooled.n_positions == len(site)
        assert pooled.recorded is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_failing_checks_names_each_check_that_bound():
    m = S.SiteMetrics(
        bb_rmsd=0.5, local_rmsd=3.0, cb_rms_dev=2.5, n_positions=20,
        identity_kept=False,
    )
    m.recorded = S.SiteMetrics(bb_rmsd=9.0, local_rmsd=1.0, n_positions=48)
    names = m.failing_checks(
        1.5, 1.0, 0.0, require_identity=True, max_cb_rms_dev=2.0,
        rest=(3.0, 2.5, 0.0, 0.0),
    )
    assert "identity" in names
    assert "local" in names        # 3.0 > 1.0
    assert "cbRMS" in names        # 2.5 > 2.0
    assert "restBB" in names       # 9.0 > 3.0
    assert "bb" not in names       # 0.5 <= 1.5
    assert "cbMax" not in names    # switched off
    assert "restLocal" not in names

    # Identity not required: not named even when lost.
    assert "identity" not in m.failing_checks(
        1.5, 1.0, 0.0, require_identity=False, max_cb_rms_dev=2.0
    )
    # A clean fold names nothing.
    ok = S.SiteMetrics(bb_rmsd=0.3, local_rmsd=0.2, cb_rms_dev=0.4, n_positions=20)
    assert ok.failing_checks(1.5, 1.0, 0.0, require_identity=False,
                             max_cb_rms_dev=2.0) == []


def test_an_unmeasurable_check_that_was_asked_for_counts_as_failing():
    """Consistent with passed(): asked for and not measurable is not a pass."""
    blind = S.SiteMetrics(n_positions=20)
    names = blind.failing_checks(1.5, 1.0, 0.0, require_identity=False,
                                 max_cb_rms_dev=2.0)
    assert set(names) == {"bb", "local", "cbRMS"}


def test_rest_thresholds_gate_the_second_group_independently():
    """The interface has a real requirement, just a looser one."""
    tight = S.SiteMetrics(
        bb_rmsd=0.8, local_rmsd=0.6, cb_rms_dev=1.1, n_positions=20
    )
    tight.recorded = S.SiteMetrics(
        bb_rmsd=2.4, local_rmsd=2.2, cb_rms_dev=1.9, n_positions=48
    )
    tight.n_recorded = 48

    # The pocket clears its tight gate either way.
    assert tight.passed(1.5, 1.0, 0.0, require_identity=False, max_cb_rms_dev=2.0)
    # Loose thresholds on the second group: in the same spot, so it passes.
    assert tight.rest_passed(3.0, 2.5, 0.0, 3.0)
    # Tight thresholds on the second group would reject it -- which is why it
    # gets its own numbers rather than sharing the pocket's.
    assert not tight.rest_passed(1.5, 1.0, 0.0, 2.0)
    # All zero is measurement only: no claim, so nothing to fail.
    assert tight.rest_passed(0.0, 0.0, 0.0, 0.0)


def test_rest_thresholds_reject_a_displaced_second_group():
    m = S.SiteMetrics(bb_rmsd=0.5, local_rmsd=0.4, cb_rms_dev=0.9, n_positions=20)
    m.recorded = S.SiteMetrics(
        bb_rmsd=7.0, local_rmsd=6.2, cb_rms_dev=5.5, n_positions=48
    )
    m.n_recorded = 48
    # The pocket is perfect and the interface has moved across the protein.
    # "In the same spot" is a requirement, so this must not pass.
    assert m.passed(1.5, 1.0, 0.0, require_identity=False, max_cb_rms_dev=2.0)
    assert not m.rest_passed(3.0, 2.5, 0.0, 3.0)


def test_rest_thresholds_on_an_empty_second_group_cannot_fail():
    """No positions outside the gate means no claim being made about them."""
    m = S.SiteMetrics(bb_rmsd=0.5, local_rmsd=0.4, n_positions=20)
    assert m.recorded is None
    assert m.rest_passed(3.0, 2.5, 0.0, 3.0)


def test_glyrs_preset_gates_the_interface_loosely_not_at_all():
    """The pocket is held and judged tightly; the interface is free but placed."""
    settings = dict(S.preset("glyrs").settings)
    assert settings["SITE_GATE_ROLES"] == "catalytic,contact"
    # The interface is gated -- "in the same spot" is a requirement.
    assert float(settings["SITE_REST_MAX_BB_RMSD"]) > 0
    assert float(settings["SITE_REST_MAX_LOCAL_RMSD"]) > 0
    # ...and more loosely than the pocket, which does the chemistry.
    assert float(settings["SITE_REST_MAX_LOCAL_RMSD"]) > 1.0
    assert float(settings["SITE_REST_MAX_BB_RMSD"]) > 1.5


def test_config_roles_match_the_site_roles():
    """Two lists that have to agree, kept out of each other's imports."""
    from redaa.config import SITE_ROLES

    assert set(SITE_ROLES) == {
        S.ROLE_CATALYTIC, S.ROLE_CONTACT, S.ROLE_INTERFACE
    }


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
