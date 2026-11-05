"""Fetch and prepare structures to use as search inputs.

The MSI pipeline takes a directory of already-cleaned, single-chain CATH domain
PDBs.  On Colab you usually have an accession rather than a file, so this
module turns an identifier into exactly the kind of file the rest of the
pipeline expects: one chain, standard residues only, primary altloc only,
complete backbone.

Accepted input specs
--------------------
``1UBQ``              RCSB entry, first protein chain
``1ubq_A``            RCSB entry, chain A (``:`` and ``.`` also work as the
                      separator)
``1ubq_A:1-76``       RCSB entry, chain A, residues 1 to 76
``cath:1cukA01``      CATH domain (already domain-bounded)
``af:P0DTC2``         AlphaFold DB model for a UniProt accession
``pdbe:1ubq``         same entry via PDBe rather than RCSB
``https://.../x.pdb`` any direct URL
``/path/to/x.pdb``    a local file, still cleaned and chain-split

Downloads are cached on disk, so re-running a notebook cell does not re-fetch.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .structure import AA3_TO_1, _OK_ALTLOC

USER_AGENT = "redaa-finder/1.0 (structure fetch; python-urllib)"
BACKBONE_ATOMS = ("N", "CA", "C")

# AlphaFold DB bumps its model version periodically (v4 through v6 so far) and
# serves only the current one, so the API is asked for the real URL first and
# these are only a fallback ladder.
_AF_VERSIONS = ("v6", "v5", "v4")


class FetchError(RuntimeError):
    """A structure could not be retrieved or could not be made usable."""


@dataclass
class StructureSpec:
    """A parsed input specification."""

    raw: str
    source: str                   # rcsb | pdbe | cath | alphafold | url | file
    ident: str
    chain: str | None = None
    resid_range: tuple[int, int] | None = None

    @property
    def name(self) -> str:
        """Stable identifier used for output directories and logs."""
        base = self.ident
        if self.source == "alphafold":
            base = f"AF_{self.ident}"
        if self.chain:
            base = f"{base}_{self.chain}"
        if self.resid_range:
            base = f"{base}_{self.resid_range[0]}-{self.resid_range[1]}"
        return re.sub(r"[^A-Za-z0-9_.-]", "_", base)


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------

_PREFIXES = {
    "rcsb": "rcsb", "pdb": "rcsb",
    "pdbe": "pdbe",
    "cath": "cath",
    "af": "alphafold", "alphafold": "alphafold", "uniprot": "alphafold",
}


def parse_spec(spec: str) -> StructureSpec:
    """Parse one input specification. Raises :class:`ValueError` if unusable."""
    text = spec.strip()
    if not text:
        raise ValueError("empty specification")

    if text.lower().startswith(("http://", "https://")):
        ident = os.path.basename(text.split("?")[0]) or "download"
        return StructureSpec(text, "url", os.path.splitext(ident)[0])

    if os.path.sep in text or text.lower().endswith((".pdb", ".ent", ".cif", ".mmcif")):
        if os.path.exists(text):
            stem = os.path.splitext(os.path.basename(text))[0]
            return StructureSpec(text, "file", stem)
        # Fall through: a bare "1ubq.pdb" that is not on disk is still a hint.
        if os.path.sep in text:
            raise ValueError(f"local file not found: {text}")
        text = os.path.splitext(text)[0]

    source = "rcsb"
    if ":" in text:
        head, rest = text.split(":", 1)
        if head.lower() in _PREFIXES:
            source = _PREFIXES[head.lower()]
            text = rest.strip()

    # Residue range, e.g. "A:12-98" or "_A:12-98".
    resid_range = None
    match = re.search(r"[:](\-?\d+)\s*-\s*(\-?\d+)\s*$", text)
    if match:
        resid_range = (int(match.group(1)), int(match.group(2)))
        if resid_range[0] > resid_range[1]:
            raise ValueError(f"inverted residue range in {spec!r}")
        text = text[: match.start()]

    chain = None
    match = re.search(r"[_.:]([A-Za-z0-9]{1,4})\s*$", text)
    if match and source != "cath":
        chain = match.group(1)
        text = text[: match.start()]

    ident = text.strip()
    if not ident:
        raise ValueError(f"no identifier in {spec!r}")

    if source == "rcsb" or source == "pdbe":
        if not re.fullmatch(r"[0-9A-Za-z]{4}", ident):
            raise ValueError(
                f"{ident!r} is not a 4-character PDB ID "
                f"(use cath:/af:/a path for other sources)"
            )
        ident = ident.upper()
    elif source == "cath":
        if not re.fullmatch(r"[0-9A-Za-z]{5,8}", ident):
            raise ValueError(f"{ident!r} does not look like a CATH domain ID")
    elif source == "alphafold":
        ident = ident.upper()

    return StructureSpec(spec.strip(), source, ident, chain, resid_range)


def parse_spec_list(text: str) -> tuple[list[StructureSpec], list[str]]:
    """Parse a list of specs, returning ``(parsed, errors)``.

    Blank entries and ``#`` comments are ignored, so a pasted list with notes
    in it still works, and duplicates are collapsed.  Bad entries are reported
    rather than raised: one typo in a list of twenty accessions should not cost
    you the other nineteen.
    """
    out: list[StructureSpec] = []
    errors: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        for token in re.split(r"[,\s]+", line):
            token = token.strip()
            if not token:
                continue
            try:
                spec = parse_spec(token)
            except ValueError as exc:
                errors.append(f"{token}: {exc}")
                continue
            if spec.name not in seen:
                seen.add(spec.name)
                out.append(spec)
    return out, errors


def parse_specs(text: str) -> list[StructureSpec]:
    """Strict :func:`parse_spec_list`: raises on the first unusable entry."""
    parsed, errors = parse_spec_list(text)
    if errors:
        raise ValueError(errors[0])
    return parsed


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _http_get(url: str, timeout: float = 60.0, retries: int = 3) -> bytes:
    """GET with retries and transparent gzip decoding."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
                encoding = (response.headers.get("Content-Encoding") or "").lower()
            if encoding == "gzip" or (url.endswith(".gz") and data[:2] == b"\x1f\x8b"):
                data = gzip.decompress(data)
            elif data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
            return data
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 400):
                raise  # not transient; let the caller try another URL
            last = exc
        except Exception as exc:  # noqa: BLE001
            last = exc
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    raise FetchError(f"could not download {url}: {last}")


def _candidate_urls(spec: StructureSpec) -> list[str]:
    if spec.source == "url":
        return [spec.raw]
    if spec.source == "rcsb":
        # Entries too large for the PDB format are only published as mmCIF.
        return [
            f"https://files.rcsb.org/download/{spec.ident}.pdb",
            f"https://files.rcsb.org/download/{spec.ident}.cif",
        ]
    if spec.source == "pdbe":
        lower = spec.ident.lower()
        return [
            f"https://www.ebi.ac.uk/pdbe/entry-files/download/pdb{lower}.ent",
            f"https://www.ebi.ac.uk/pdbe/entry-files/download/{lower}.cif",
        ]
    if spec.source == "cath":
        return [
            f"http://www.cathdb.info/version/current/api/rest/id/{spec.ident}.pdb",
            f"http://www.cathdb.info/version/v4_3_0/api/rest/id/{spec.ident}.pdb",
        ]
    if spec.source == "alphafold":
        return [
            f"https://alphafold.ebi.ac.uk/files/AF-{spec.ident}-F1-model_{v}.pdb"
            for v in _AF_VERSIONS
        ]
    raise FetchError(f"no download URL for source {spec.source!r}")


def _alphafold_url_from_api(accession: str) -> str | None:
    """Ask AlphaFold DB for the current model URL.

    The database serves only its latest model version and that version string
    has already moved three times, so asking beats guessing.
    """
    try:
        data = _http_get(
            f"https://alphafold.ebi.ac.uk/api/prediction/{accession}",
            timeout=30.0,
            retries=2,
        )
        entries = json.loads(data.decode("utf-8"))
    except Exception:
        return None
    if isinstance(entries, list) and entries:
        url = entries[0].get("pdbUrl") or entries[0].get("cifUrl")
        if isinstance(url, str) and url:
            return url
    return None


def download(spec: StructureSpec, cache_dir: str) -> str:
    """Retrieve *spec* into *cache_dir*, returning the local path.

    A cached file is reused, so re-running a notebook cell costs nothing.
    """
    if spec.source == "file":
        if not os.path.exists(spec.raw):
            raise FetchError(f"local file not found: {spec.raw}")
        return spec.raw

    os.makedirs(cache_dir, exist_ok=True)
    urls = _candidate_urls(spec)
    if spec.source == "alphafold":
        api_url = _alphafold_url_from_api(spec.ident)
        if api_url:
            urls.insert(0, api_url)

    errors: list[str] = []
    for url in urls:
        suffix = ".cif" if url.split("?")[0].endswith((".cif", ".mmcif")) else ".pdb"
        dest = os.path.join(cache_dir, f"{spec.name}_raw{suffix}")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return dest
        try:
            data = _http_get(url)
        except urllib.error.HTTPError as exc:
            errors.append(f"{url} -> HTTP {exc.code}")
            continue
        except FetchError as exc:
            errors.append(str(exc))
            continue
        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dest)
        return dest

    raise FetchError(
        f"could not fetch {spec.raw!r} from any source:\n  "
        + "\n  ".join(errors)
    )


# ---------------------------------------------------------------------------
# mmCIF -> PDB
# ---------------------------------------------------------------------------


def _split_cif_row(line: str) -> list[str]:
    """Tokenise one mmCIF loop row, honouring single and double quotes."""
    tokens: list[str] = []
    i, n = 0, len(line)
    while i < n:
        while i < n and line[i] in " \t":
            i += 1
        if i >= n:
            break
        if line[i] in "'\"":
            quote = line[i]
            i += 1
            start = i
            while i < n and not (
                line[i] == quote and (i + 1 >= n or line[i + 1] in " \t")
            ):
                i += 1
            tokens.append(line[start:i])
            i += 1
        else:
            start = i
            while i < n and line[i] not in " \t":
                i += 1
            tokens.append(line[start:i])
    return tokens


def cif_to_pdb(text: str) -> str:
    """Convert the ``_atom_site`` loop of an mmCIF file into PDB ATOM records.

    Only what this pipeline needs is carried across: the first model, protein
    ``ATOM`` records, author chain and residue numbering (so chain IDs match
    what a user expects from the PDB entry).  Multi-character mmCIF chain IDs
    are mapped onto single characters, since the PDB format has only one
    column for them.
    """
    lines = text.splitlines()
    fields: list[str] = []
    rows: list[list[str]] = []
    in_loop = False
    reading = False

    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            in_loop, reading, fields = True, False, []
            continue
        if in_loop and stripped.startswith("_atom_site."):
            fields.append(stripped.split(".", 1)[1].split()[0])
            reading = False
            continue
        if fields and not stripped.startswith("_atom_site."):
            if stripped.startswith(("#", "loop_", "_", "data_")) or not stripped:
                if reading:
                    break          # end of the atom_site loop
                if stripped.startswith("_"):
                    fields = []    # a different loop; forget it
                    in_loop = False
                continue
            reading = True
            rows.append(_split_cif_row(line))

    if not fields or not rows:
        return ""

    index = {name: i for i, name in enumerate(fields)}

    def get(row, *names, default=""):
        for name in names:
            i = index.get(name)
            if i is not None and i < len(row):
                return row[i]
        return default

    chain_map: dict[str, str] = {}
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    first_model = None
    out: list[str] = []
    serial = 0

    for row in rows:
        if get(row, "group_PDB", default="ATOM") != "ATOM":
            continue
        model = get(row, "pdbx_PDB_model_num", default="1")
        if first_model is None:
            first_model = model
        if model != first_model:
            continue
        resname = get(row, "label_comp_id", "auth_comp_id").strip()
        if resname not in AA3_TO_1:
            continue

        raw_chain = get(row, "auth_asym_id", "label_asym_id", default="A").strip()
        chain = chain_map.get(raw_chain)
        if chain is None:
            chain = (
                raw_chain
                if len(raw_chain) == 1
                else alphabet[len(chain_map) % len(alphabet)]
            )
            chain_map[raw_chain] = chain

        altloc = get(row, "label_alt_id", default=".").strip()
        altloc = " " if altloc in (".", "?", "") else altloc[0]
        icode = get(row, "pdbx_PDB_ins_code", default=".").strip()
        icode = " " if icode in (".", "?", "") else icode[0]

        try:
            resseq = int(float(get(row, "auth_seq_id", "label_seq_id", default="0")))
            x = float(get(row, "Cartn_x"))
            y = float(get(row, "Cartn_y"))
            z = float(get(row, "Cartn_z"))
        except ValueError:
            continue

        atom = get(row, "label_atom_id", "auth_atom_id").strip().strip('"')
        name = f" {atom:<3s}" if len(atom) < 4 else atom[:4]
        element = get(row, "type_symbol", default=atom[:1]).strip()[:2]
        serial += 1

        out.append(
            f"ATOM  {serial % 100000:>5d} {name}{altloc}{resname:>3s} "
            f"{chain}{resseq:>4d}{icode}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          "
            f"{element:>2s}"
        )

    return "\n".join(out) + ("\n" if out else "")


def read_structure_text(path: str) -> str:
    """Read *path* as PDB text, converting from mmCIF when needed."""
    with open(path, "r", errors="replace") as fh:
        text = fh.read()
    looks_cif = path.lower().endswith((".cif", ".mmcif")) or (
        "_atom_site." in text and not text.lstrip().startswith(("HEADER", "ATOM"))
    )
    if looks_cif:
        converted = cif_to_pdb(text)
        if not converted:
            raise FetchError(f"no usable protein ATOM records in {path}")
        return converted
    return text


# ---------------------------------------------------------------------------
# Chain inspection and preparation
# ---------------------------------------------------------------------------


@dataclass
class ChainInfo:
    chain: str
    n_residues: int
    sequence: str


def list_chains(text: str) -> list[ChainInfo]:
    """Protein chains in PDB *text*, in order of appearance, longest reported first."""
    order: list[str] = []
    seen: dict[str, dict[tuple[int, str], str]] = {}
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM") or len(line) < 54:
            continue
        if line[12:16] != " CA " or line[16] not in _OK_ALTLOC:
            continue
        one = AA3_TO_1.get(line[17:20].strip())
        if one is None:
            continue
        chain = line[21]
        if chain not in seen:
            seen[chain] = {}
            order.append(chain)
        try:
            seen[chain][(int(line[22:26]), line[26])] = one
        except ValueError:
            continue

    infos = [
        ChainInfo(
            chain,
            len(seen[chain]),
            "".join(seen[chain][k] for k in sorted(seen[chain])),
        )
        for chain in order
    ]
    return sorted(infos, key=lambda c: (-c.n_residues, c.chain))



def _atom_xyz(line: str) -> tuple[float, float, float] | None:
    try:
        return (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    except ValueError:
        return None


def _collect_ligands(
    text: str,
    chosen_chain: str,
    kept_protein_lines: list[str],
    keep_ligands: tuple[str, ...] | None,
    cutoff: float,
) -> tuple[list[str], list[tuple[str, int]], list[str]]:
    """Pick out the HETATM records worth keeping.

    Returns ``(lines, [(comp_id, n_atoms), ...], notes)``.  Notes name
    components that were asked for and not found, and copies rejected for
    sitting too far from the chosen chain -- both are the kind of thing that
    otherwise shows up much later as an empty binding site.
    """
    if not keep_ligands:
        return [], [], []

    wanted = {c.strip().upper() for c in keep_ligands if c and c.strip()}
    if not wanted:
        return [], [], []

    prot_xyz = [
        xyz for xyz in (_atom_xyz(l) for l in kept_protein_lines) if xyz is not None
    ]

    groups: dict[tuple[str, str, int, str], list[str]] = {}
    order: list[tuple[str, str, int, str]] = []
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("HETATM") or len(line) < 54:
            continue
        if line[16] not in _OK_ALTLOC:
            continue
        resname = line[17:20].strip().upper()
        if resname not in wanted:
            continue
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        key = (resname, line[21], resseq, line[26])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(line.rstrip("\n"))

    cut2 = cutoff * cutoff
    lines: list[str] = []
    counts: dict[str, int] = {}
    notes: list[str] = []
    rejected: dict[str, int] = {}

    for key in order:
        resname, chain, resseq, icode = key
        group = groups[key]
        if chain != chosen_chain:
            # A copy on another chain counts only if it actually touches this
            # one.
            near = False
            for line in group:
                xyz = _atom_xyz(line)
                if xyz is None:
                    continue
                for p in prot_xyz:
                    dx = xyz[0] - p[0]
                    dy = xyz[1] - p[1]
                    dz = xyz[2] - p[2]
                    if dx * dx + dy * dy + dz * dz <= cut2:
                        near = True
                        break
                if near:
                    break
            if not near:
                rejected[resname] = rejected.get(resname, 0) + 1
                continue
        lines.extend(group)
        counts[resname] = counts.get(resname, 0) + len(group)

    for name, n in sorted(rejected.items()):
        notes.append(
            f"{name}: {n} cop{'y' if n == 1 else 'ies'} ignored, further than "
            f"{cutoff:g} A from chain {chosen_chain}"
        )
    for name in sorted(wanted - set(counts)):
        notes.append(f"{name}: requested but not present in this entry")

    return lines, sorted(counts.items()), notes


def prepare(
    text: str,
    out_path: str,
    chain: str | None = None,
    resid_range: tuple[int, int] | None = None,
    keep_ligands: tuple[str, ...] | None = None,
    ligand_contact_cutoff: float = 6.0,
) -> dict:
    """Write a single-chain, pipeline-ready PDB and report what was dropped.

    A raw RCSB entry is not what the search can consume: it may hold several
    chains, waters and ligands, alternate conformations, and residues whose
    backbone was never resolved.  MPNN needs N, CA and C, so a residue missing
    any of them is dropped rather than left to produce garbage features.  The
    pre-processed CATH files the MSI runs use were already clean, which is why
    V9 could skip all of this.

    ``keep_ligands`` names heteroatom components to retain -- ``("ATP",)`` and
    so on.  Everything else, waters included, is still dropped.  Retention is
    opt-in because a ligand in the file changes what the pipeline does: it is
    what lets LigandMPNN design against the actual bound molecule instead of a
    bare backbone, and what lets a binding site be derived from coordinates
    rather than transcribed by hand.

    A named component is kept if it sits on the chosen chain *or* has any atom
    within ``ligand_contact_cutoff`` of a kept protein atom.  Both tests are
    needed: deposition practice puts a bound ligand sometimes on its
    protein's chain and sometimes on its own, and a copy bound to a different
    chain of the crystal is not part of this chain's site.
    """
    chains = list_chains(text)
    if not chains:
        raise FetchError("no protein chains found")

    if chain is None:
        chosen = chains[0].chain
    else:
        wanted = chain.strip()
        match = [c for c in chains if c.chain == wanted]
        if not match and len(wanted) == 1:
            match = [c for c in chains if c.chain.upper() == wanted.upper()]
        if not match:
            raise FetchError(
                f"chain {chain!r} not found; available: "
                + ", ".join(f"{c.chain}({c.n_residues})" for c in chains)
            )
        chosen = match[0].chain

    # Collect the chosen chain's residues and the atoms belonging to each.
    residues: dict[tuple[int, str], list[str]] = {}
    atoms_by_res: dict[tuple[int, str], set[str]] = {}
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM") or len(line) < 54:
            continue
        if line[21] != chosen or line[16] not in _OK_ALTLOC:
            continue
        if line[17:20].strip() not in AA3_TO_1:
            continue
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        if resid_range and not (resid_range[0] <= resseq <= resid_range[1]):
            continue
        key = (resseq, line[26])
        residues.setdefault(key, []).append(line.rstrip("\n"))
        atoms_by_res.setdefault(key, set()).add(line[12:16].strip())

    if not residues:
        raise FetchError(
            f"chain {chosen} has no standard residues"
            + (f" in range {resid_range[0]}-{resid_range[1]}" if resid_range else "")
        )

    complete = [
        key for key in sorted(residues)
        if all(a in atoms_by_res[key] for a in BACKBONE_ATOMS)
    ]
    dropped = len(residues) - len(complete)
    if not complete:
        raise FetchError(f"chain {chosen}: no residue has a complete N-CA-C backbone")

    lines: list[str] = []
    for key in complete:
        lines.extend(residues[key])

    ligand_lines, ligands_kept, ligand_notes = _collect_ligands(
        text, chosen, lines, keep_ligands, ligand_contact_cutoff
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
        fh.write("\nTER\n")
        if ligand_lines:
            fh.write("\n".join(ligand_lines))
            fh.write("\n")
        fh.write("END\n")

    # A gap means consecutive kept residues are not consecutive in numbering.
    numbers = [key[0] for key in complete]
    gaps = sum(1 for a, b in zip(numbers, numbers[1:]) if b - a > 1)

    return {
        "path": out_path,
        "chain": chosen,
        "n_residues": len(complete),
        "dropped_incomplete": dropped,
        "gaps": gaps,
        "chains_available": [(c.chain, c.n_residues) for c in chains],
        "ligands_kept": ligands_kept,
        "ligand_notes": ligand_notes,
    }


def fetch_and_prepare(
    spec: str | StructureSpec,
    out_dir: str,
    cache_dir: str | None = None,
    keep_ligands: tuple[str, ...] | None = None,
) -> dict:
    """Resolve one spec all the way to a pipeline-ready PDB in *out_dir*.

    Returns the :func:`prepare` report, plus ``name`` and ``source``.
    """
    parsed = parse_spec(spec) if isinstance(spec, str) else spec
    cache_dir = cache_dir or os.path.join(out_dir, "_raw")
    raw_path = download(parsed, cache_dir)
    text = read_structure_text(raw_path)
    out_path = os.path.join(out_dir, f"{parsed.name}.pdb")
    report = prepare(
        text, out_path, parsed.chain, parsed.resid_range,
        keep_ligands=keep_ligands,
    )
    report["name"] = parsed.name
    report["source"] = parsed.source
    report["spec"] = parsed.raw
    return report
