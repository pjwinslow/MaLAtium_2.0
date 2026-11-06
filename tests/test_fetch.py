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
