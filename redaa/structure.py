"""Fast structural metrics for the reduced-alphabet search.

Pure NumPy + stdlib: no BioPython, no torch, no disk I/O in the hot path.
That matters because the pipeline computes RMSD and TM-score once per folded
sequence, which is tens of thousands of times per batch job.

The V9 pipeline called BioPython's ``PDBParser`` four times per comparison
(twice inside ``calculate_rmsd`` and twice more inside ``calculate_tm_score``),
re-reading the reference crystal file from shared storage every time.  Here the
reference is parsed once per domain and cached, and the prediction is parsed
straight out of the PDB text that ESMFold2 hands back, so nothing touches the
filesystem unless we explicitly want to keep a structure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

AA3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

# Alternate-location indicators we accept.  A residue modelled in two
# conformations contributes two CA records; taking both would double-count it.
_OK_ALTLOC = (" ", "A", "")


def parse_ca_from_text(text: str) -> tuple[np.ndarray, str]:
    """Extract CA coordinates and the one-letter sequence from PDB text.

    Applies the same filters the V9 pipeline used, so numbers stay comparable:
    first model only, ``ATOM`` records only (``HETATM`` waters, ligands and
    modified residues such as MSE are dropped), standard residues only, and
    blank/``A`` altloc only.  Residues are ordered by chain of first appearance
    and then by (residue number, insertion code).

    Returns ``(coords, seq)`` with ``coords`` of shape ``(N, 3)`` float64.
    """
    chain_order: list[str] = []
    per_chain: dict[str, list[tuple[int, str, float, float, float, str]]] = {}

    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break  # first model only
        if not line.startswith("ATOM"):
            continue
        if len(line) < 54:
            continue
        if line[12:16] != " CA ":
            continue
        if line[16] not in _OK_ALTLOC:
            continue
        resname = line[17:20].strip()
        one = AA3_TO_1.get(resname)
        if one is None:
            continue
        try:
            resseq = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        chain = line[21]
        if chain not in per_chain:
            per_chain[chain] = []
            chain_order.append(chain)
        per_chain[chain].append((resseq, line[26], x, y, z, one))

    coords: list[tuple[float, float, float]] = []
    seq: list[str] = []
    for chain in chain_order:
        for resseq, icode, x, y, z, one in sorted(
            per_chain[chain], key=lambda r: (r[0], r[1])
        ):
            coords.append((x, y, z))
            seq.append(one)

    if not coords:
        return np.zeros((0, 3), dtype=np.float64), ""
    return np.asarray(coords, dtype=np.float64), "".join(seq)


@lru_cache(maxsize=256)
def parse_ca_from_file(path: str) -> tuple[np.ndarray, str]:
    """Cached :func:`parse_ca_from_text` for a file on disk.

    Reference structures are read once per domain and then reused for every
    one of the thousands of comparisons made against them.  The returned array
    is marked read-only so a caller cannot corrupt the cached entry in place.
    """
    with open(path, "r") as fh:
        coords, seq = parse_ca_from_text(fh.read())
    coords.flags.writeable = False
    return coords, seq


def sequence_from_file(path: str) -> str:
    """One-letter sequence of the standard residues in *path*."""
    return parse_ca_from_file(path)[1]


def extract_chain(src: str, chain_id: str, dst: str) -> int:
    """Write one chain of *src* to *dst*, returning the residue count written.

    Used for raw multi-chain RCSB entries given by their four-character ID.
    The design target, the native-sequence source and the RMSD/TM reference
    must all be the *same* single chain, so the chain is materialised once and
    every downstream step points at that file.

    Keeps first-model standard-residue ``ATOM`` records with primary altloc --
    the same filters the metrics use -- so waters, ligands and modified
    residues such as MSE are dropped.
    """
    kept: list[str] = []
    residues: set[tuple[int, str]] = set()
    with open(src, "r") as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if line.startswith("TER") and kept:
                kept.append(line.rstrip("\n"))
                continue
            if not line.startswith("ATOM") or len(line) < 54:
                continue
            if line[21] != chain_id:
                continue
            if line[16] not in _OK_ALTLOC:
                continue
            if line[17:20].strip() not in AA3_TO_1:
                continue
            kept.append(line.rstrip("\n"))
            try:
                residues.add((int(line[22:26]), line[26]))
            except ValueError:
                pass

    if not residues:
        return 0
    with open(dst, "w") as out:
        out.write("\n".join(kept))
        out.write("\nEND\n")
    return len(residues)


# ---------------------------------------------------------------------------
# Superposition
# ---------------------------------------------------------------------------


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal rigid transform taking *mobile* onto *target*.

    Returns ``(rot, trans)`` such that ``mobile @ rot.T + trans`` is the
    least-squares superposition of *mobile* onto *target*.  The reflection
    case is handled, so ``rot`` is always a proper rotation.
    """
    mob_c = mobile.mean(axis=0)
    tgt_c = target.mean(axis=0)
    cov = (mobile - mob_c).T @ (target - tgt_c)
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, tgt_c - rot @ mob_c


def apply_transform(xyz: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return xyz @ rot.T + trans


def kabsch_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    """CA-RMSD in Angstrom after optimal superposition over all given pairs."""
    if len(mobile) == 0 or len(mobile) != len(target):
        return float("nan")
    rot, trans = kabsch(mobile, target)
    diff = apply_transform(mobile, rot, trans) - target
    return float(np.sqrt((diff * diff).sum() / len(target)))


# ---------------------------------------------------------------------------
# TM-score
# ---------------------------------------------------------------------------


def tm_d0(l_ref: int) -> float:
    """Zhang & Skolnick (2004) length-dependent distance scale, in Angstrom."""
    if l_ref <= 15:
        return 0.5
    return max(0.5, 1.24 * (l_ref - 15) ** (1.0 / 3.0) - 1.8)


def _tm_from_seed(
    mobile: np.ndarray,
    target: np.ndarray,
    seed: np.ndarray,
    d0: float,
    d0_search: float,
    l_ref: int,
    max_iter: int = 20,
) -> float:
    """Refine a superposition from one seed fragment and score it.

    This is the inner loop of the TM-score program: superpose on a subset,
    recompute which residues fall inside a distance cutoff, superpose on those
    instead, and repeat.  The score is read off at every iteration because the
    refinement is not monotonic.
    """
    idx = seed
    best = 0.0
    seen: set[bytes] = set()
    for _ in range(max_iter):
        if len(idx) < 3:
            break
        rot, trans = kabsch(mobile[idx], target[idx])
        dist = np.linalg.norm(apply_transform(mobile, rot, trans) - target, axis=1)
        best = max(best, float((1.0 / (1.0 + (dist / d0) ** 2)).sum() / l_ref))

        cutoff = d0_search
        while True:
            nxt = np.flatnonzero(dist < cutoff)
            if len(nxt) >= 3 or cutoff > dist.max():
                break
            cutoff += 0.5
        if len(nxt) < 3:
            break
        key = nxt.tobytes()
        if key in seen:
            break
        seen.add(key)
        idx = nxt
    return best


def tm_score(
    mobile: np.ndarray,
    target: np.ndarray,
    l_ref: int | None = None,
    max_seeds: int = 48,
) -> float:
    """TM-score of *mobile* against *target*, normalised by *l_ref*.

    Assumes a one-to-one residue correspondence between the two arrays, which
    is what this pipeline always has: ProteinMPNN designs a sequence of exactly
    the backbone's length and ESMFold2 predicts every residue of it.  Callers
    with a length mismatch should pair residues with :func:`align_pair` first.

    ``l_ref`` defaults to ``len(target)`` and should be set to the *full*
    reference length when only part of it could be paired, so that a partial
    match to a longer reference is penalised the way TM-align penalises it.

    This searches seed fragments and iteratively refines each superposition,
    which is the TM-score algorithm proper.  A single global superposition --
    what V9 did -- systematically *under*-reports TM-score, because one bad
    terminal segment drags the whole fit off the well-matched core.
    """
    n = len(target)
    if n == 0 or len(mobile) != n:
        return float("nan")
    if l_ref is None:
        l_ref = n
    if l_ref < 15:
        return float("nan")

    d0 = tm_d0(l_ref)
    d0_search = float(np.clip(d0, 4.5, 8.0))

    # Seed fragments at decreasing lengths, the way TM-score does it.  Strides
    # grow with the number of candidate starts so the seed count stays bounded
    # regardless of chain length.
    seeds: list[np.ndarray] = [np.arange(n)]
    frag = n
    while frag > 4:
        frag = max(4, frag // 2)
        n_starts = n - frag + 1
        stride = max(1, math.ceil(n_starts / max(1, max_seeds // 4)))
        for start in range(0, n_starts, stride):
            seeds.append(np.arange(start, start + frag))
        if frag == 4:
            break

    best = 0.0
    for seed in seeds[:max_seeds]:
        best = max(best, _tm_from_seed(mobile, target, seed, d0, d0_search, l_ref))
    return best


# ---------------------------------------------------------------------------
# Residue pairing for the rare length-mismatch case
# ---------------------------------------------------------------------------


def _nw_align(seq1: str, seq2: str, match=2, mismatch=-1, gap=-1) -> tuple[str, str]:
    """Needleman-Wunsch global alignment, returning both gapped strings."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i * gap
    for j in range(n + 1):
        dp[0][j] = j * gap
    for i in range(1, m + 1):
        row, prev = dp[i], dp[i - 1]
        c1 = seq1[i - 1]
        for j in range(1, n + 1):
            s = match if c1 == seq2[j - 1] else mismatch
            row[j] = max(prev[j - 1] + s, prev[j] + gap, row[j - 1] + gap)

    a1: list[str] = []
    a2: list[str] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = match if seq1[i - 1] == seq2[j - 1] else mismatch
            if dp[i][j] == dp[i - 1][j - 1] + s:
                a1.append(seq1[i - 1])
                a2.append(seq2[j - 1])
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + gap:
            a1.append(seq1[i - 1])
            a2.append("-")
            i -= 1
        else:
            a1.append("-")
            a2.append(seq2[j - 1])
            j -= 1
    return "".join(reversed(a1)), "".join(reversed(a2))


def align_pair(
    ref_xyz: np.ndarray,
    ref_seq: str,
    pred_xyz: np.ndarray,
    pred_seq: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pair reference and predicted CA atoms, returning ``(ref, pred, l_ref)``.

    ``l_ref`` is always the *full* reference length, for TM normalisation.
    Equal lengths -- effectively always, here -- pair directly with no
    alignment work at all.  Unequal lengths fall back to a global alignment so
    that an internal gap does not frame-shift every later residue, which is
    what naively taking the first L atoms would do.
    """
    l_ref = len(ref_xyz)
    if l_ref == 0 or len(pred_xyz) == 0:
        return ref_xyz[:0], pred_xyz[:0], l_ref
    if l_ref == len(pred_xyz):
        return ref_xyz, pred_xyz, l_ref

    aln_ref, aln_pred = _nw_align(ref_seq, pred_seq)
    keep_ref: list[int] = []
    keep_pred: list[int] = []
    i = j = 0
    for r, p in zip(aln_ref, aln_pred):
        if r != "-" and p != "-":
            keep_ref.append(i)
            keep_pred.append(j)
        if r != "-":
            i += 1
        if p != "-":
            j += 1
    return ref_xyz[keep_ref], pred_xyz[keep_pred], l_ref


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_HAVE_TMTOOLS: bool | None = None


def have_tmtools() -> bool:
    """Whether the optional ``tmtools`` extension (exact TM-align) is present."""
    global _HAVE_TMTOOLS
    if _HAVE_TMTOOLS is None:
        try:
            import tmtools  # noqa: F401

            _HAVE_TMTOOLS = True
        except Exception:
            _HAVE_TMTOOLS = False
    return _HAVE_TMTOOLS


def compare(
    ref_xyz: np.ndarray,
    ref_seq: str,
    pred_xyz: np.ndarray,
    pred_seq: str,
    use_tmtools: bool = True,
) -> tuple[float | None, float | None]:
    """Return ``(rmsd, tm_score)`` of a prediction against a reference.

    RMSD comes from a single global superposition over the paired CA atoms, as
    in V9, so it stays comparable with earlier runs.  TM-score prefers exact
    TM-align via ``tmtools`` when it is installed and falls back to the
    seed-refined NumPy implementation above otherwise.  Either way the inputs
    are coordinates already in memory -- no file is parsed here.
    """
    ref_p, pred_p, l_ref = align_pair(ref_xyz, ref_seq, pred_xyz, pred_seq)
    if len(ref_p) < 3:
        return None, None

    rmsd = kabsch_rmsd(pred_p, ref_p)

    tm: float | None = None
    if use_tmtools and have_tmtools() and len(ref_p) >= 15 and len(pred_p) >= 15:
        try:
            import tmtools

            # tmtools wants its own sequences; slice them to the paired subset.
            res = tmtools.tm_align(
                np.ascontiguousarray(ref_p, dtype=np.float64),
                np.ascontiguousarray(pred_p, dtype=np.float64),
                ref_seq[: len(ref_p)],
                pred_seq[: len(pred_p)],
            )
            tm = float(res.tm_norm_chain1)
        except Exception:
            tm = None
    if tm is None:
        tm = tm_score(pred_p, ref_p, l_ref=l_ref)

    rmsd_out = None if (rmsd is None or math.isnan(rmsd)) else float(rmsd)
    tm_out = None if (tm is None or math.isnan(tm)) else float(tm)
    return rmsd_out, tm_out


# ---------------------------------------------------------------------------
# Residue-level parsing, for functional-site work
# ---------------------------------------------------------------------------
#
# The CA-only fast path above is what the fold gate needs.  Site preservation
# needs more: to say whether a catalytic constellation survived, you have to
# compare where those residues sit relative to each other, not just whether
# the chain traced the same path.
#
# The extra atoms kept here are N, CA and C only -- the backbone frame.  CB is
# *derived* from them rather than read, deliberately:
#
#   * it exists for glycine, so a site containing one is still measurable;
#   * it does not depend on a predicted rotamer, which is the least reliable
#     part of any folding model's output, nor on which altloc the crystal
#     happened to list first;
#   * reference and prediction are therefore treated identically, so a
#     deviation means the backbone moved, not that two side chains were
#     modelled differently.
#
# This is the same convention motif-scaffolding work uses when it reports
# motif RMSD, and it is the reason the numbers below are comparable between a
# crystal structure and an ESMFold2 model.

FRAME_ATOMS = ("N", "CA", "C")

# Virtual-CB construction from the backbone frame (the standard
# trRosetta/AlphaFold formula).  Reproduces a real CB to well under 0.1 A.
_VCB_A, _VCB_B, _VCB_C = -0.58273431, 0.56802827, -0.54067466


@dataclass
class ResidueRecord:
    """One residue's identity, numbering and backbone frame."""

    chain: str
    resseq: int
    icode: str
    resname: str
    aa: str  # one-letter, "X" for a retained non-standard residue
    atoms: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    is_hetatm: bool = False

    @property
    def author_key(self) -> tuple[str, int, str]:
        return (self.chain, self.resseq, self.icode.strip())

    @property
    def label(self) -> str:
        ic = self.icode.strip()
        return f"{self.chain}/{self.aa}{self.resseq}{ic}"

    def has_frame(self) -> bool:
        return all(name in self.atoms for name in FRAME_ATOMS)


def parse_residues_from_text(
    text: str,
    atom_names: tuple[str, ...] | None = FRAME_ATOMS,
    include_hetatm: bool = False,
    keep_resnames: tuple[str, ...] | None = None,
) -> list[ResidueRecord]:
    """Parse residues with selected atoms, in the same order as the CA parser.

    Ordering is chain of first appearance, then (residue number, insertion
    code) -- identical to :func:`parse_ca_from_text`, so index *i* here is
    index *i* in the sequence that function returns.  That correspondence is
    what lets a site specified in author numbering be turned into a position
    in the sequence MPNN designs.

    ``include_hetatm`` additionally returns ligand/heteroatom groups, which
    :mod:`redaa.sites` uses to find a binding pocket geometrically.  Those
    records are tagged ``is_hetatm`` and are *not* part of the polymer
    ordering; callers filter on the flag.
    """
    # atom_names=None keeps every atom, which is what a ligand contact shell
    # needs; the default keeps the backbone frame, which is what the metrics
    # need and is a quarter of the parsing work.
    keep_all_atoms = atom_names is None
    wanted = set() if keep_all_atoms else {
        f"{n:^4s}" if len(n) < 4 else n for n in atom_names
    }
    wanted_stripped = set() if keep_all_atoms else {n.strip() for n in atom_names}
    keep_set = set(keep_resnames) if keep_resnames else None

    chain_order: list[str] = []
    per_chain: dict[str, dict[tuple[int, str], ResidueRecord]] = {}
    het: list[ResidueRecord] = []
    het_index: dict[tuple[str, int, str, str], ResidueRecord] = {}

    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        is_atom = line.startswith("ATOM")
        is_het = line.startswith("HETATM")
        if not (is_atom or (is_het and include_hetatm)):
            continue
        if len(line) < 54:
            continue
        if line[16] not in _OK_ALTLOC:
            continue

        resname = line[17:20].strip()
        raw_name = line[12:16]
        name = raw_name.strip()
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        icode = line[26]
        chain = line[21]
        try:
            xyz = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        except ValueError:
            continue

        if is_atom:
            one = AA3_TO_1.get(resname)
            if one is None:
                continue
            if not keep_all_atoms and (
                raw_name not in wanted and name not in wanted_stripped
            ):
                continue
            if chain not in per_chain:
                per_chain[chain] = {}
                chain_order.append(chain)
            key = (resseq, icode)
            rec = per_chain[chain].get(key)
            if rec is None:
                rec = ResidueRecord(chain, resseq, icode, resname, one)
                per_chain[chain][key] = rec
            rec.atoms.setdefault(name, xyz)
        else:
            if resname == "HOH" or resname == "DOD":
                continue
            if keep_set is not None and resname not in keep_set:
                continue
            hkey = (chain, resseq, icode, resname)
            rec = het_index.get(hkey)
            if rec is None:
                rec = ResidueRecord(
                    chain, resseq, icode, resname,
                    AA3_TO_1.get(resname, "X"), is_hetatm=True,
                )
                het_index[hkey] = rec
                het.append(rec)
            rec.atoms[name] = xyz

    ordered: list[ResidueRecord] = []
    for chain in chain_order:
        for key in sorted(per_chain[chain]):
            ordered.append(per_chain[chain][key])
    return ordered + het


def parse_residues_from_file(path: str, **kwargs) -> list[ResidueRecord]:
    with open(path) as fh:
        return parse_residues_from_text(fh.read(), **kwargs)


def virtual_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """CB position implied by a backbone frame; works for glycine."""
    b = ca - n
    cc = c - ca
    a = np.cross(b, cc)
    return _VCB_A * a + _VCB_B * b + _VCB_C * cc + ca


def frames_from_records(
    records: list[ResidueRecord],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(frames, valid)`` for the polymer records.

    ``frames`` is ``(L, 4, 3)`` holding N, CA, C and the derived virtual CB.
    ``valid`` is a boolean mask: a residue without a complete N-CA-C frame
    gets NaNs and ``False``, rather than being dropped, so indices still line
    up with the sequence.
    """
    poly = [r for r in records if not r.is_hetatm]
    frames = np.full((len(poly), 4, 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(poly), dtype=bool)
    for i, rec in enumerate(poly):
        if not rec.has_frame():
            continue
        n = np.asarray(rec.atoms["N"], dtype=np.float64)
        ca = np.asarray(rec.atoms["CA"], dtype=np.float64)
        c = np.asarray(rec.atoms["C"], dtype=np.float64)
        frames[i, 0] = n
        frames[i, 1] = ca
        frames[i, 2] = c
        frames[i, 3] = virtual_cb(n, ca, c)
        valid[i] = True
    return frames, valid


def frames_from_text(text: str) -> tuple[np.ndarray, np.ndarray, str]:
    """``(frames, valid, sequence)`` straight from PDB text."""
    records = parse_residues_from_text(text)
    frames, valid = frames_from_records(records)
    seq = "".join(r.aa for r in records if not r.is_hetatm)
    return frames, valid, seq


ATOM_SLOT = {"N": 0, "CA": 1, "C": 2, "CB": 3}


def _gather(frames: np.ndarray, positions, atoms) -> np.ndarray:
    slots = [ATOM_SLOT[a] for a in atoms]
    out = []
    for p in positions:
        for s in slots:
            out.append(frames[p, s])
    return np.asarray(out, dtype=np.float64)


def site_rmsd(
    ref_frames: np.ndarray,
    pred_frames: np.ndarray,
    positions,
    atoms: tuple[str, ...] = ("N", "CA", "C"),
    superpose: str = "global",
) -> float | None:
    """RMSD over the site's atoms.

    ``superpose="global"`` fits on every CA both structures share and then
    measures the site where that fit puts it: it answers "is the site in the
    right place, in the context of the whole fold?".  ``superpose="local"``
    fits on the site atoms alone: it answers "is the site's own internal
    geometry intact?", independently of whether the scaffold around it
    drifted.  A site can pass one and fail the other, and the two together
    say which happened, so the pipeline records both.
    """
    positions = [int(p) for p in positions]
    if not positions:
        return None
    n = min(len(ref_frames), len(pred_frames))
    if any(p < 0 or p >= n for p in positions):
        return None

    ok = np.isfinite(ref_frames[:n, 1, 0]) & np.isfinite(pred_frames[:n, 1, 0])
    if not all(ok[p] for p in positions):
        return None

    try:
        site_ref = _gather(ref_frames, positions, atoms)
        site_pred = _gather(pred_frames, positions, atoms)
    except (IndexError, KeyError):
        return None
    if not (np.isfinite(site_ref).all() and np.isfinite(site_pred).all()):
        return None

    if superpose == "local":
        if len(site_ref) < 3:
            return None
        value = kabsch_rmsd(site_pred, site_ref)
    else:
        anchor_idx = np.nonzero(ok)[0]
        if len(anchor_idx) < 3:
            return None
        rot, trans = kabsch(
            pred_frames[anchor_idx, 1], ref_frames[anchor_idx, 1]
        )
        moved = apply_transform(site_pred, rot, trans)
        diff = moved - site_ref
        value = float(np.sqrt((diff * diff).sum() / len(site_ref)))

    return None if (value is None or math.isnan(value)) else float(value)


def site_distance_deviation(
    ref_frames: np.ndarray,
    pred_frames: np.ndarray,
    positions,
    atom: str = "CB",
) -> tuple[float | None, float | None]:
    """``(max_abs, rms)`` deviation of the site's internal distance matrix.

    Superposition-free, so it cannot be flattered or punished by how the fit
    landed.  For a catalytic constellation this is the number that most
    directly says "are these residues still the right distance apart?".
    """
    positions = [int(p) for p in positions]
    if len(positions) < 2:
        return None, None
    n = min(len(ref_frames), len(pred_frames))
    if any(p < 0 or p >= n for p in positions):
        return None, None
    slot = ATOM_SLOT[atom]
    a = ref_frames[positions, slot]
    b = pred_frames[positions, slot]
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None, None

    devs: list[float] = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d_ref = float(np.linalg.norm(a[i] - a[j]))
            d_pred = float(np.linalg.norm(b[i] - b[j]))
            devs.append(d_pred - d_ref)
    if not devs:
        return None, None
    arr = np.asarray(devs, dtype=np.float64)
    return float(np.abs(arr).max()), float(np.sqrt((arr * arr).mean()))


def frames_with_cb(frames_nca: np.ndarray) -> np.ndarray:
    """Expand a cached ``(L, 3, 3)`` N/CA/C array to ``(L, 4, 3)`` with CB.

    CB is reconstructed rather than stored so the cache holds only measured
    coordinates, and so a row written by any version of this code yields the
    same CB under the same formula.
    """
    arr = np.asarray(frames_nca, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != 3:
        raise ValueError(f"expected (L, 3, 3) N/CA/C frames, got {arr.shape}")
    out = np.empty((arr.shape[0], 4, 3), dtype=np.float64)
    out[:, :3, :] = arr
    out[:, 3, :] = virtual_cb(arr[:, 0], arr[:, 1], arr[:, 2])
    return out
