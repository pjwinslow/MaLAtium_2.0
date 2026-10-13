"""Chemical requirements read off a backbone.

Two residues cannot be substituted away, so the search pre-seeds them rather
than wasting rounds rediscovering that fact:

* **CYS** where a structural disulfide exists.  A disulfide is a covalent
  cross-link; no other residue forms one, so a chain without Cys at those
  positions cannot adopt the topology under oxidising conditions.
* **PRO** where a cis-peptide bond exists.  The peptide omega dihedral is
  normally near 180 degrees.  About 5% of Pro residues sit cis, while non-Pro
  cis peptides are vanishingly rare (~0.03%) and usually artefactual.  The cis
  conformation is enforced by proline's pyrrolidine ring, and Gly cannot
  substitute because it has no ring to stabilise omega near zero.

Trans-Pro turns are deliberately *not* required: Gly is often a fine
substitute at those positions, and mandating Pro for every turn inflates the
reported minimal alphabet for domains that do not need it.

Reimplemented on NumPy rather than BioPython so this module -- like
:mod:`redaa.structure` -- runs and can be tested without the heavier
dependency stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .structure import AA3_TO_1, _OK_ALTLOC

SS_DIST_MAX = 2.5     # Angstrom; an SG-SG covalent bond is ~2.05 A
CIS_OMEGA_MAX = 30.0  # degrees; |omega| below this is unambiguously cis


@dataclass
class Requirements:
    required_aas: list[str] = field(default_factory=list)
    n_disulfides: int = 0
    cis_pro_count: int = 0
    notes: list[str] = field(default_factory=list)


def parse_atoms(text: str) -> list[dict]:
    """Minimal PDB atom reader: first model, ``ATOM`` records, primary altloc."""
    atoms: list[dict] = []
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM") or len(line) < 54:
            continue
        if line[16] not in _OK_ALTLOC:
            continue
        resname = line[17:20].strip()
        if resname not in AA3_TO_1:
            continue
        try:
            atoms.append(
                {
                    "name": line[12:16].strip(),
                    "resname": resname,
                    "chain": line[21],
                    "resseq": int(line[22:26]),
                    "icode": line[26],
                    "xyz": np.array(
                        [
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        ]
                    ),
                }
            )
        except ValueError:
            continue
    return atoms


def _residues(atoms: list[dict]) -> list[dict]:
    """Group atoms into residues, ordered by chain then (resseq, icode)."""
    buckets: dict[tuple, dict] = {}
    order: list[tuple] = []
    for atom in atoms:
        key = (atom["chain"], atom["resseq"], atom["icode"])
        res = buckets.get(key)
        if res is None:
            res = {
                "key": key,
                "resname": atom["resname"],
                "chain": atom["chain"],
                "resseq": atom["resseq"],
                "icode": atom["icode"],
                "atoms": {},
            }
            buckets[key] = res
            order.append(key)
        res["atoms"][atom["name"]] = atom["xyz"]

    chain_order: list[str] = []
    for key in order:
        if key[0] not in chain_order:
            chain_order.append(key[0])
    return sorted(
        (buckets[k] for k in order),
        key=lambda r: (chain_order.index(r["chain"]), r["resseq"], r["icode"]),
    )


def dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Signed dihedral angle p0-p1-p2-p3 in degrees."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    norm = np.linalg.norm(b1)
    if norm < 1e-8:
        return float("nan")
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return math.degrees(math.atan2(y, x))


def analyze_requirements(pdb_path: str) -> Requirements:
    """Residues that must be present in the starting alphabet for *pdb_path*."""
    req = Requirements()
    try:
        with open(pdb_path, "r") as fh:
            atoms = parse_atoms(fh.read())
    except OSError as exc:
        req.notes.append(f"WARNING: could not read {pdb_path} ({exc})")
        return req

    residues = _residues(atoms)

    # -- disulfides --------------------------------------------------------
    sg = [
        (r["key"], r["atoms"]["SG"])
        for r in residues
        if r["resname"] == "CYS" and "SG" in r["atoms"]
    ]
    if len(sg) >= 2:
        coords = np.array([x for _, x in sg])
        d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        iu = np.triu_indices(len(sg), k=1)
        req.n_disulfides = int((d[iu] < SS_DIST_MAX).sum())
    if req.n_disulfides > 0:
        req.required_aas.append("C")
        req.notes.append(
            f"CYS required: {req.n_disulfides} disulfide bond(s) detected "
            f"(SG-SG < {SS_DIST_MAX} A)"
        )

    # -- cis-proline -------------------------------------------------------
    for prev, curr in zip(residues, residues[1:]):
        if curr["resname"] != "PRO" or curr["chain"] != prev["chain"]:
            continue
        need_prev = ("CA", "C")
        need_curr = ("N", "CA")
        if not all(a in prev["atoms"] for a in need_prev):
            continue
        if not all(a in curr["atoms"] for a in need_curr):
            continue
        omega = dihedral(
            prev["atoms"]["CA"],
            prev["atoms"]["C"],
            curr["atoms"]["N"],
            curr["atoms"]["CA"],
        )
        if math.isnan(omega):
            continue
        if abs(omega) < CIS_OMEGA_MAX:
            req.cis_pro_count += 1
            req.notes.append(
                f"PRO required: cis-Pro at residue {curr['resseq']} "
                f"(omega = {omega:.1f} deg)"
            )

    if req.cis_pro_count > 0 and "P" not in req.required_aas:
        req.required_aas.append("P")

    return req
