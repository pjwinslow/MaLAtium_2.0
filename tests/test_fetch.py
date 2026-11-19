"""Tests for structure fetching and input preparation.

No network: HTTP is stubbed, so the URL ladder, the cache and the mmCIF
fallback are all exercised offline.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redaa import fetch as F  # noqa: E402


def _atom(serial, name, resname, chain, resseq, xyz, altloc=" ", record="ATOM  ",
          icode=" "):
    x, y, z = xyz
    return (
        f"{record}{serial:>5d} {name:^4s}{altloc}{resname:>3s} {chain}"
        f"{resseq:>4d}{icode}   {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00"
    )


def _residue(serial, resname, chain, resseq, base, atoms=("N", "CA", "C", "O")):
    out = []
    for i, atom in enumerate(atoms):
        out.append(
            _atom(serial + i, atom, resname, chain, resseq,
                  (base + i * 0.5, base, base))
        )
    return out


def _two_chain_pdb():
    lines = []
    serial = 1
    for i, resname in enumerate(("ALA", "GLY", "SER", "VAL"), start=1):
        lines += _residue(serial, resname, "A", i, float(i))
        serial += 4
    for i, resname in enumerate(("LEU", "ILE"), start=1):
        lines += _residue(serial, resname, "B", i, float(i))
        serial += 4
    lines.append(_atom(serial, "O", "HOH", "A", 500, (9.0, 9.0, 9.0),
                       record="HETATM"))
    lines.append("END")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------


def test_parses_bare_pdb_id():
    spec = F.parse_spec("1ubq")
    assert (spec.source, spec.ident, spec.chain) == ("rcsb", "1UBQ", None)
    assert spec.name == "1UBQ"


def test_parses_chain_with_each_separator():
    for text in ("1ubq_A", "1ubq:A", "1ubq.A"):
        spec = F.parse_spec(text)
        assert (spec.source, spec.ident, spec.chain) == ("rcsb", "1UBQ", "A"), text
    assert F.parse_spec("1ubq_A").name == "1UBQ_A"


def test_parses_residue_range():
    spec = F.parse_spec("1ubq_A:12-76")
    assert spec.ident == "1UBQ"
    assert spec.chain == "A"
    assert spec.resid_range == (12, 76)
    assert spec.name == "1UBQ_A_12-76"


def test_rejects_inverted_range():
    try:
        F.parse_spec("1ubq_A:76-12")
    except ValueError as exc:
        assert "inverted" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_parses_source_prefixes():
    assert F.parse_spec("cath:1cukA01").source == "cath"
    # A CATH domain ID ends in digits that must not be read as a chain.
    assert F.parse_spec("cath:1cukA01").ident == "1cukA01"
    assert F.parse_spec("cath:1cukA01").chain is None
    assert F.parse_spec("af:P0DTC2").source == "alphafold"
    assert F.parse_spec("af:P0DTC2").name == "AF_P0DTC2"
    assert F.parse_spec("uniprot:p12345").ident == "P12345"
    assert F.parse_spec("pdbe:1ubq").source == "pdbe"


def test_parses_url_and_local_file():
    spec = F.parse_spec("https://example.org/structures/foo.pdb")
    assert spec.source == "url" and spec.ident == "foo"

    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "mydomain.pdb")
        with open(path, "w") as fh:
            fh.write("ATOM\n")
        spec = F.parse_spec(path)
        assert spec.source == "file" and spec.ident == "mydomain"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rejects_bad_identifiers():
    for bad in ("", "notanid", "12345", "/no/such/file.pdb"):
        try:
            F.parse_spec(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_parse_specs_handles_lists_comments_and_duplicates():
    text = """
    1ubq, 2gb1   # two classics
    # a whole comment line
    1ubq
    af:P0DTC2
    """
    specs = F.parse_specs(text)
    assert [s.name for s in specs] == ["1UBQ", "2GB1", "AF_P0DTC2"]


# ---------------------------------------------------------------------------
# Chain inspection
# ---------------------------------------------------------------------------


def test_list_chains_reports_longest_first_and_skips_water():
    chains = F.list_chains(_two_chain_pdb())
    assert [(c.chain, c.n_residues) for c in chains] == [("A", 4), ("B", 2)]
    assert chains[0].sequence == "AGSV"
    assert chains[1].sequence == "LI"


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def test_prepare_defaults_to_the_largest_chain():
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "out.pdb")
        report = F.prepare(_two_chain_pdb(), out)
        assert report["chain"] == "A"
        assert report["n_residues"] == 4
        body = open(out).read()
        assert " B   1" not in body       # the other chain is gone
        assert "HOH" not in body          # water is gone
        assert body.rstrip().endswith("END")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prepare_selects_a_named_chain_case_insensitively():
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "out.pdb")
        report = F.prepare(_two_chain_pdb(), out, chain="b")
        assert report["chain"] == "B"
        assert report["n_residues"] == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prepare_errors_helpfully_on_a_missing_chain():
    tmp = tempfile.mkdtemp()
    try:
        F.prepare(_two_chain_pdb(), os.path.join(tmp, "o.pdb"), chain="Z")
    except F.FetchError as exc:
        assert "A(4)" in str(exc) and "B(2)" in str(exc)
    else:
        raise AssertionError("expected FetchError")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prepare_applies_a_residue_range_and_counts_gaps():
    lines = []
    serial = 1
    # Residues 1,2,3 then 10,11 -- one gap.
    for resseq in (1, 2, 3, 10, 11):
        lines += _residue(serial, "ALA", "A", resseq, float(resseq))
        serial += 4
    lines.append("END")
    text = "\n".join(lines)

    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "o.pdb")
        full = F.prepare(text, out)
        assert full["n_residues"] == 5 and full["gaps"] == 1

        ranged = F.prepare(text, out, resid_range=(1, 3))
        assert ranged["n_residues"] == 3 and ranged["gaps"] == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prepare_drops_residues_without_a_complete_backbone():
    """MPNN needs N, CA and C; a CA-only residue would poison featurisation."""
    lines = []
    lines += _residue(1, "ALA", "A", 1, 1.0)
    lines += _residue(5, "GLY", "A", 2, 2.0, atoms=("CA",))        # CA only
    lines += _residue(6, "SER", "A", 3, 3.0, atoms=("N", "CA"))    # no C
    lines += _residue(8, "VAL", "A", 4, 4.0)
    lines.append("END")

    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "o.pdb")
        report = F.prepare("\n".join(lines), out)
        assert report["n_residues"] == 2, report
        assert report["dropped_incomplete"] == 2, report
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prepare_keeps_only_the_primary_altloc():
    lines = _residue(1, "ALA", "A", 1, 1.0)
    lines += [
        _atom(10, "N", "SER", "A", 2, (2.0, 2.0, 2.0), altloc="A"),
        _atom(11, "CA", "SER", "A", 2, (2.5, 2.0, 2.0), altloc="A"),
        _atom(12, "C", "SER", "A", 2, (3.0, 2.0, 2.0), altloc="A"),
        _atom(13, "CA", "SER", "A", 2, (9.9, 9.9, 9.9), altloc="B"),
    ]
    lines.append("END")
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "o.pdb")
        report = F.prepare("\n".join(lines), out)
        assert report["n_residues"] == 2
        assert "9.900" not in open(out).read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prepare_rejects_a_structure_with_no_protein():
    tmp = tempfile.mkdtemp()
    try:
        F.prepare("HEADER nothing here\nEND\n", os.path.join(tmp, "o.pdb"))
    except F.FetchError as exc:
        assert "no protein chains" in str(exc)
    else:
        raise AssertionError("expected FetchError")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# mmCIF conversion
# ---------------------------------------------------------------------------

_CIF = """data_TEST
#
loop_
_chem_comp.id
ALA
GLY
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . ALA AAA 1 ? 1.000 2.000 3.000 1 A 1
ATOM 2 C CA . ALA AAA 1 ? 1.500 2.000 3.000 1 A 1
ATOM 3 C C . ALA AAA 1 ? 2.000 2.000 3.000 1 A 1
ATOM 4 N N . GLY AAA 2 ? 4.000 2.000 3.000 2 A 1
ATOM 5 C CA . GLY AAA 2 ? 4.500 2.000 3.000 2 A 1
ATOM 6 C C . GLY AAA 2 ? 5.000 2.000 3.000 2 A 1
HETATM 7 O O . HOH BBB 1 ? 9.000 9.000 9.000 99 B 1
ATOM 8 C CA . VAL AAA 3 ? 7.000 7.000 7.000 3 A 2
#
"""


def test_cif_to_pdb_extracts_first_model_protein_atoms():
    text = F.cif_to_pdb(_CIF)
    lines = [line for line in text.splitlines() if line.startswith("ATOM")]
    assert len(lines) == 6, text
    assert all(line[21] == "A" for line in lines)
    assert "HOH" not in text
    assert "VAL" not in text          # model 2 is excluded
    chains = F.list_chains(text)
    assert [(c.chain, c.n_residues) for c in chains] == [("A", 2)]
    assert chains[0].sequence == "AG"
    assert abs(float(lines[0][30:38]) - 1.0) < 1e-6


def test_cif_row_splitter_handles_quotes():
    row = F._split_cif_row("""ATOM 1 C "CA'" . ALA 'long chain' 1""")
    assert row == ["ATOM", "1", "C", "CA'", ".", "ALA", "long chain", "1"]


def test_cif_maps_multicharacter_chains_to_single_characters():
    cif = _CIF.replace(" A 1\n", " AB 1\n")
    text = F.cif_to_pdb(cif)
    chains = F.list_chains(text)
    assert len(chains) == 1
    assert len(chains[0].chain) == 1


def test_read_structure_text_detects_cif_without_an_extension():
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "nosuffix")
        with open(path, "w") as fh:
            fh.write(_CIF)
        text = F.read_structure_text(path)
        assert text.startswith("ATOM")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Download ladder and cache
# ---------------------------------------------------------------------------


class _StubHTTP:
    """Stands in for _http_get; serves a map of URL -> bytes, 404s the rest."""

    def __init__(self, responses):
        self.responses = responses
        self.calls: list[str] = []

    def __call__(self, url, timeout=60.0, retries=3):
        self.calls.append(url)
        if url in self.responses:
            return self.responses[url]
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)


def _with_stub(stub, fn):
    original = F._http_get
    F._http_get = stub
    try:
        return fn()
    finally:
        F._http_get = original


def test_download_falls_back_from_pdb_to_cif():
    spec = F.parse_spec("1abc")
    stub = _StubHTTP({
        "https://files.rcsb.org/download/1ABC.cif": _CIF.encode(),
    })
    tmp = tempfile.mkdtemp()
    try:
        path = _with_stub(stub, lambda: F.download(spec, tmp))
        assert path.endswith(".cif")
        assert stub.calls == [
            "https://files.rcsb.org/download/1ABC.pdb",
            "https://files.rcsb.org/download/1ABC.cif",
        ]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_download_reuses_the_cache_on_a_second_call():
    spec = F.parse_spec("1abc")
    payload = _two_chain_pdb().encode()
    stub = _StubHTTP({"https://files.rcsb.org/download/1ABC.pdb": payload})
    tmp = tempfile.mkdtemp()
    try:
        first = _with_stub(stub, lambda: F.download(spec, tmp))
        second = _with_stub(stub, lambda: F.download(spec, tmp))
        assert first == second
        assert len(stub.calls) == 1, stub.calls
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_download_reports_every_source_it_tried():
    spec = F.parse_spec("1abc")
    stub = _StubHTTP({})
    tmp = tempfile.mkdtemp()
    try:
        _with_stub(stub, lambda: F.download(spec, tmp))
    except F.FetchError as exc:
        assert "HTTP 404" in str(exc)
        assert "1ABC.pdb" in str(exc) and "1ABC.cif" in str(exc)
    else:
        raise AssertionError("expected FetchError")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_alphafold_prefers_the_url_the_api_reports():
    spec = F.parse_spec("af:P12345")
    api = "https://alphafold.ebi.ac.uk/api/prediction/P12345"
    served = "https://alphafold.ebi.ac.uk/files/AF-P12345-F1-model_v9.pdb"
    stub = _StubHTTP({
        api: b'[{"pdbUrl": "' + served.encode() + b'"}]',
        served: _two_chain_pdb().encode(),
    })
    tmp = tempfile.mkdtemp()
    try:
        _with_stub(stub, lambda: F.download(spec, tmp))
        # The API answer is tried before the hardcoded version ladder.
        assert stub.calls[0] == api
        assert stub.calls[1] == served
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_alphafold_falls_back_to_the_version_ladder():
    spec = F.parse_spec("af:P12345")
    v4 = "https://alphafold.ebi.ac.uk/files/AF-P12345-F1-model_v4.pdb"
    stub = _StubHTTP({v4: _two_chain_pdb().encode()})
    tmp = tempfile.mkdtemp()
    try:
        path = _with_stub(stub, lambda: F.download(spec, tmp))
        assert os.path.exists(path)
        assert stub.calls[-1] == v4
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fetch_and_prepare_end_to_end():
    stub = _StubHTTP({
        "https://files.rcsb.org/download/1ABC.pdb": _two_chain_pdb().encode(),
    })
    tmp = tempfile.mkdtemp()
    try:
        report = _with_stub(
            stub, lambda: F.fetch_and_prepare("1abc_B", tmp)
        )
        assert report["name"] == "1ABC_B"
        assert report["chain"] == "B"
        assert report["n_residues"] == 2
        assert report["source"] == "rcsb"
        assert os.path.exists(report["path"])
        assert report["path"].endswith("1ABC_B.pdb")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



# ---------------------------------------------------------------------------
# Ligand retention
# ---------------------------------------------------------------------------
#
# Retention has to be opt-in and precise. A ligand left out means LigandMPNN
# designs against a bare backbone and a pocket cannot be derived at all; a
# ligand wrongly kept -- a cryoprotectant, or a copy bound to a different
# chain of the crystal -- defines a site that is not there.


def _ligand_pdb():
    """Chain A with ATP bound to it, a glycerol, and an ATP on far-off chain B."""
    lines = []
    serial = 1
    for i in range(1, 5):
        for name, dx in ((" N  ", -0.6), (" CA ", 0.0), (" C  ", 0.7)):
            lines.append(
                f"ATOM  {serial:>5d} {name} ALA A{i:>4d}    "
                f"{3.8 * i + dx:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00  0.00"
            )
            serial += 1
    # ATP on chain A, right beside the protein.
    for k in range(3):
        lines.append(
            f"HETATM{serial:>5d}  C{k:<2d} ATP A 900    "
            f"{7.6 + k:>8.3f}{2.0:>8.3f}{0.0:>8.3f}  1.00  0.00"
        )
        serial += 1
    # Glycerol, also close by, but not a ligand anyone asked for.
    lines.append(
        f"HETATM{serial:>5d}  C1  GOL A 901    "
        f"{7.6:>8.3f}{-2.0:>8.3f}{0.0:>8.3f}  1.00  0.00"
    )
    serial += 1
    # A second ATP, on chain B and 200 A away: a crystal neighbour, not part
    # of this chain's site.
    for k in range(3):
        lines.append(
            f"HETATM{serial:>5d}  C{k:<2d} ATP B 902    "
            f"{200.0 + k:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00  0.00"
        )
        serial += 1
    lines.append("HETATM 9999  O   HOH A 950       5.000   5.000   5.000  1.00  0.00")
    lines.append("END")
    return "\n".join(lines) + "\n"


def test_prepare_drops_every_ligand_by_default():
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "o.pdb")
        report = F.prepare(_ligand_pdb(), out)
        body = open(out).read()
        assert "HETATM" not in body
        assert report["ligands_kept"] == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prepare_keeps_a_named_ligand_only():
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "o.pdb")
        report = F.prepare(_ligand_pdb(), out, keep_ligands=("ATP",))
        body = open(out).read()
        assert "ATP" in body
        assert "GOL" not in body          # not asked for
        assert "HOH" not in body          # never kept
        # Only the chain-A copy: 3 atoms, not 6.
        assert report["ligands_kept"] == [("ATP", 3)]
        assert any("ignored, further than" in n for n in report["ligand_notes"])
        # The protein still comes first and the file still ends properly.
        assert body.index("ATOM") < body.index("HETATM")
        assert body.rstrip().endswith("END")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prepare_reports_a_ligand_that_is_not_there():
    tmp = tempfile.mkdtemp()
    try:
        report = F.prepare(
            _ligand_pdb(), os.path.join(tmp, "o.pdb"), keep_ligands=("GTP",)
        )
        assert report["ligands_kept"] == []
        assert any("requested but not present" in n for n in report["ligand_notes"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prepare_keeps_a_ligand_on_another_chain_when_it_touches_this_one():
    tmp = tempfile.mkdtemp()
    try:
        # Same entry, but the chain-B ATP is moved into contact with chain A.
        text = _ligand_pdb().replace("     200.000", "       8.000")
        text = text.replace("     201.000", "       9.000")
        text = text.replace("     202.000", "      10.000")
        out = os.path.join(tmp, "o.pdb")
        report = F.prepare(text, out, chain="A", keep_ligands=("ATP",))
        # Both copies now count: deposition puts a bound ligand on either
        # chain, so proximity, not the chain letter, decides.
        assert report["ligands_kept"] == [("ATP", 6)]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_kept_ligand_is_parseable_as_a_site():
    """The retained ligand has to be usable downstream, not just present."""
    from redaa import sites as S
    from redaa import structure as st

    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "o.pdb")
        F.prepare(_ligand_pdb(), out, keep_ligands=("ATP",))
        recs = st.parse_residues_from_file(
            out, atom_names=None, include_hetatm=True
        )
        site = S.resolve_site(S.parse_site_spec("ligand:ATP@4.5"), recs)
        assert len(site) >= 1, site.warnings
        assert site.ligand_atom_counts == {"ATP": 3}
        assert not any("no atoms found" in w for w in site.warnings)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prepare_records_provenance_that_reads_back():
    """A prepared input has to say what it is.

    An input directory outlives the run that filled it -- in Colab it sits in
    /content across many cell runs -- and a file named for one structure that
    contains another is unidentifiable without this. That is not hypothetical:
    a site spec was checked against the wrong structure and neither the file
    nor the error could say so.
    """
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "o.pdb")
        F.prepare(_two_chain_pdb(), out, chain="A", spec="1abc_A:1-4")
        info = F.read_provenance(out)
        assert info["spec"] == "1abc_A:1-4"
        assert info["chain"] == "A"
        assert info["residues"] == "4"
        assert info["prepared"].endswith("Z")
        described = F.describe_provenance(out)
        assert "1abc_A:1-4" in described and "chain A" in described

        # The remarks must not disturb anything that reads the coordinates.
        from redaa import structure as st

        xyz, seq = st.parse_ca_from_file(out)
        assert len(xyz) == 4 and len(seq) == 4
        recs = st.parse_residues_from_file(out)
        assert len([r for r in recs if not r.is_hetatm]) == 4
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_provenance_of_a_file_without_any_is_reported_not_invented():
    tmp = tempfile.mkdtemp()
    try:
        plain = os.path.join(tmp, "plain.pdb")
        with open(plain, "w") as fh:
            fh.write(_two_chain_pdb())
        assert F.read_provenance(plain) == {}
        assert "no provenance" in F.describe_provenance(plain)
        assert "re-run the input step" in F.describe_provenance(plain).lower()
        # A path that does not exist must not raise.
        assert "no provenance" in F.describe_provenance(os.path.join(tmp, "nope"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prepare_records_kept_ligands_in_the_provenance():
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "o.pdb")
        F.prepare(_ligand_pdb(), out, keep_ligands=("ATP",), spec="5etn_A")
        info = F.read_provenance(out)
        assert info["ligands"] == "ATP:3"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bindable_components_are_reported_even_when_none_are_requested():
    """Without a preset there was no way to learn a structure's ligand code.

    You had to go and read the entry before you could write
    SITE_KEEP_LIGANDS, and a spec naming the wrong code produced an empty
    pocket much later. The candidates are now named by the fetch itself.
    """
    tmp = tempfile.mkdtemp()
    try:
        report = F.prepare(_ligand_pdb(), os.path.join(tmp, "o.pdb"))
        notes = " ".join(report["ligand_notes"])
        assert "ATP" in notes, report["ligand_notes"]
        assert "SITE_KEEP_LIGANDS" in notes
        # Nothing was kept -- this is discovery, not retention.
        assert report["ligands_kept"] == []
        # Crystallisation additives are not advertised as ligands.
        assert "GOL" not in notes
        assert "HOH" not in notes
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_requesting_the_wrong_ligand_code_names_the_right_one():
    tmp = tempfile.mkdtemp()
    try:
        report = F.prepare(
            _ligand_pdb(), os.path.join(tmp, "o.pdb"), keep_ligands=("GTP",)
        )
        notes = " ".join(report["ligand_notes"])
        assert "GTP: requested but not present" in notes
        assert "this entry has ATP" in notes
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_kept_ligand_is_not_also_advertised_as_a_candidate():
    """Reporting what you already asked for would just be noise."""
    tmp = tempfile.mkdtemp()
    try:
        report = F.prepare(
            _ligand_pdb(), os.path.join(tmp, "o.pdb"), keep_ligands=("ATP",)
        )
        notes = " ".join(report["ligand_notes"])
        assert "also contains" not in notes, report["ligand_notes"]
        assert report["ligands_kept"] == [("ATP", 3)]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_ligand_on_another_chain_is_not_advertised_for_this_one():
    tmp = tempfile.mkdtemp()
    try:
        # _ligand_pdb puts a second ATP on chain B, 200 A away.
        report = F.prepare(_ligand_pdb(), os.path.join(tmp, "o.pdb"), chain="A")
        notes = " ".join(report["ligand_notes"])
        # The nearby copy is advertised; its atom count must not include the
        # distant one, or the number implies a ligand that is not there.
        assert "ATP (3 atoms)" in notes, report["ligand_notes"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Polymer partners: a second subunit, or a nucleic acid
# ---------------------------------------------------------------------------


def _partner_pdb():
    """Three chains ~5 A apart in a triangle, plus a modified nucleotide.

    Chain A is the protein to design, chain B a second protein subunit, chain
    D a short RNA.  Every pair is about 5 A apart, so the RNA genuinely
    contacts both subunits -- which is the situation that makes single-chain
    design of an interface only partly possible, and the thing the warning
    has to catch.
    """
    lines = []
    serial = 1

    def put(record, name, resname, chain, resseq, xyz):
        nonlocal serial
        lines.append(_atom(serial, name, resname, chain, resseq, xyz,
                           record=record))
        serial += 1

    for i in range(1, 7):  # chain A: y=0, z=0
        for name, dx in ((" N  ", -0.6), (" CA ", 0.0), (" C  ", 0.7)):
            put("ATOM  ", name, "ALA", "A", i, (3.8 * i + dx, 0.0, 0.0))
    for i in range(1, 5):  # chain B: y=0, z=5
        for name, dx in ((" N  ", -0.6), (" CA ", 0.0), (" C  ", 0.7)):
            put("ATOM  ", name, "LEU", "B", i, (3.8 * i + dx, 0.0, 5.0))
    for i, base in ((2, "  G"), (3, "  A"), (4, "  U")):  # RNA: y=4.33, z=2.5
        for name, dx in ((" P  ", -0.6), (" C1'", 0.0), (" N9 ", 0.7)):
            put("ATOM  ", name, base, "D", i, (3.8 * i + dx, 4.33, 2.5))
    # A modified base, coded HETATM as depositions often do, right beside the
    # protein. Without the nucleic exclusion this is advertised as a ligand.
    put("HETATM", " C1'", "PSU", "D", 5, (3.8 * 5, 4.33, 2.5))
    put("HETATM", " O  ", "HOH", "A", 800, (50.0, 50.0, 50.0))
    lines.append("END")
    return "\n".join(lines) + "\n"


def test_partner_chains_are_reported_when_none_are_kept():
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "o.pdb")
        report = F.prepare(_partner_pdb(), out, chain="A")
        assert report["partners_kept"] == []
        notes = " | ".join(report["partner_notes"])
        assert "chain D (nucleic, 4 residues)" in notes, notes
        assert "chain B (protein, 4 residues)" in notes, notes
        # The note has to say how to act on it, with the chain id filled in.
        assert "SITE_KEEP_PARTNERS=D" in notes, notes
        # A chain-selection list is still protein-only: a tRNA is not a
        # candidate for the chain to design.
        assert [c for c, _n in report["chains_available"]] == ["A", "B"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_nucleic_chain_is_not_advertised_as_a_ligand():
    """75 nucleotides offered as 75 candidate ligands would bury the real one."""
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "o.pdb")
        report = F.prepare(_partner_pdb(), out, chain="A")
        notes = " ".join(report["ligand_notes"])
        assert "PSU" not in notes, notes
        assert "G" not in notes.split() and "U" not in notes.split(), notes
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_kept_partner_is_written_as_hetatm_keeping_its_chain():
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "o.pdb")
        report = F.prepare(_partner_pdb(), out, chain="A", keep_partners=("D",))
        assert len(report["partners_kept"]) == 1
        chain, kind, n_res, n_atoms, n_contacts = report["partners_kept"][0]
        assert (chain, kind, n_res) == ("D", "nucleic", 4)
        assert n_atoms == 10  # 3 nucleotides x 3 atoms, plus the PSU atom
        # All six of chain A: the RNA runs alongside it, and at the 6 A
        # default even the diagonal neighbours are 5.6 A away.
        assert n_contacts == 6

        text = open(out).read()
        partner = [
            l for l in text.splitlines()
            if l.startswith(("ATOM", "HETATM")) and len(l) > 21 and l[21] == "D"
        ]
        assert len(partner) == 10
        # Every one a HETATM, every one still on chain D. Both matter: the
        # record name is what keeps the polymer parsers away from it, and the
        # chain id is what the partner spec selects on.
        assert all(l.startswith("HETATM") for l in partner), partner[:2]
        assert all(l[21] == "D" for l in partner)
        # Columns must not have shifted when the record name was rewritten.
        assert all(l[17:20].strip() in ("G", "A", "U", "PSU") for l in partner)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_kept_partner_is_invisible_to_the_polymer_parser():
    """A partner must never be folded, designed or scored as part of the chain."""
    from redaa import structure as st

    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "o.pdb")
        F.prepare(_partner_pdb(), out, chain="A", keep_partners=("D", "B"))
        text = open(out).read()
        poly = st.parse_residues_from_text(text, atom_names=None)
        assert len(poly) == 6, [r.resname for r in poly]
        assert {r.chain for r in poly} == {"A"}
        with_het = st.parse_residues_from_text(
            text, atom_names=None, include_hetatm=True
        )
        assert len(with_het) == 6 + 4 + 4
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_partner_binding_another_subunit_too_is_flagged():
    """The honest limit of single-chain design, stated rather than implied."""
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "o.pdb")
        report = F.prepare(_partner_pdb(), out, chain="A", keep_partners=("D",))
        notes = " | ".join(report["partner_notes"])
        assert "also contacts" in notes, notes
        assert "chain B" in notes, notes
        assert "cannot be reproduced by a single-chain design" in notes, notes
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_partner_that_is_not_there_is_named_not_ignored():
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "o.pdb")
        report = F.prepare(_partner_pdb(), out, chain="A", keep_partners=("Z",))
        assert report["partners_kept"] == []
        notes = " | ".join(report["partner_notes"])
        assert "'Z' requested but not in the entry" in notes, notes
        assert "A, B, D" in notes, notes
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_designed_chain_cannot_be_its_own_partner():
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "o.pdb")
        report = F.prepare(_partner_pdb(), out, chain="A", keep_partners=("A",))
        assert report["partners_kept"] == []
        assert any(
            "is the chain being designed" in n for n in report["partner_notes"]
        ), report["partner_notes"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_kept_partners_are_recorded_in_the_provenance():
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "o.pdb")
        F.prepare(_partner_pdb(), out, chain="A", keep_partners=("D",),
                  spec="synthetic_A")
        info = F.read_provenance(out)
        assert info["partners"].startswith("D:nucleic:"), info
        line = F.describe_provenance(out)
        # What context a file carries decides what a run can attempt, so it
        # belongs on the one line that identifies the file.
        assert "partners D" in line, line
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chain_classification():
    assert F._classify_chain(["ALA", "GLY", "SER"]) == "protein"
    assert F._classify_chain(["G", "A", "U", "PSU"]) == "nucleic"
    assert F._classify_chain(["ATP", "HEM"]) == "other"
    assert F._classify_chain([]) == "other"


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
