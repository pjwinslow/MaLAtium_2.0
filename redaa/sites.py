"""Functional sites: catalytic residues and ligand-binding pockets.

The alphabet search on its own asks one question -- what is the smallest set of
amino acids from which MPNN can design a sequence that folds back onto the
reference backbone.  A minimal folding alphabet that answers it is still
useless as a protein: the residues that do the work are exactly the ones a
five-letter alphabet throws away.  A catalytic cysteine is not interchangeable
with a serine, and no amount of fold accuracy compensates for losing it.

This module adds the second question: can the fold be encoded in a small
alphabet *while the functional site is preserved*.  It contributes three
things, and deliberately nothing else:

**A way to say what the site is.**  :func:`parse_site_spec` reads a compact
text spec -- ``"H46 D81 C151"``, ``"ligand:ATP@4.5"``, ``"A/T112 range:110-116"``
-- in *author* (PDB) numbering, because that is the numbering in the papers,
the PDB entry and the user's head.  Nothing else in the pipeline speaks author
numbering, so this is the only place that conversion happens.

**A way to find a pocket without being told.**  Naming a ligand instead of
residues derives the contact shell geometrically from the coordinates, which
is both less error-prone than transcribing a list and specific to the actual
structure in hand.

**Resolution against one prepared structure.**  :func:`resolve_site` turns a
spec into 0-based positions in the sequence MPNN designs, and checks that each
residue is the one the spec expected.  That check matters more than it looks:
PDB entries renumber, constructs carry tags, and a site spec silently applied
to the wrong offset produces a run that looks fine and means nothing.  A
mismatch is reported, not guessed at.

Author numbering is mapped through the *prepared* file, not the raw download,
since preparation drops residues with incomplete backbones and that shifts
every later index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

import numpy as np

from . import structure as st

# Roles.  CATALYTIC residues were named explicitly; CONTACT residues were
# derived from a ligand's contact shell.  The distinction is kept because they
# usually deserve different treatment -- a catalytic residue is
# non-negotiable, while the outer edge of a contact shell is often just
# packing.
ROLE_CATALYTIC = "catalytic"
ROLE_CONTACT = "contact"

# Solvent, cryoprotectants, buffers and crystallisation additives.  Present in
# the coordinates, not part of the protein's function, and a disaster if they
# are taken for the ligand: a pocket "defined" by a glycerol molecule is
# noise.  Kept as a denylist rather than an allowlist because the interesting
# ligand is whatever the user names, and that cannot be enumerated.
SOLVENT_COMPONENTS = frozenset({
    "HOH", "DOD", "D2O", "WAT",
    "GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "2PE", "P6G", "7PE", "XPE",
    "MPD", "DMS", "DMSO", "EOH", "MOH", "IPA", "ACT", "ACY", "ACE",
    "SO4", "PO4", "CL", "BR", "IOD", "NO3", "FMT", "CIT", "FLC", "TRS",
    "EPE", "MES", "BME", "TCE", "AZI", "CAC", "MLI", "TAR", "SCN",
    "NH4", "UNX", "UNL",
})

# Monatomic ions.  Not solvent -- a catalytic zinc or the magnesium that makes
# an ATP site an ATP site is functional -- but too small to define a pocket on
# their own, so they are never auto-selected as "the ligand".  Naming one
# explicitly works.
ION_COMPONENTS = frozenset({
    "ZN", "MG", "MN", "CA", "NA", "K", "FE", "FE2", "CU", "CU1", "CO",
    "NI", "CD", "HG", "LI", "SR", "BA", "CS", "RB", "AL", "W", "MO",
})


class SiteError(ValueError):
    """A site spec that cannot be parsed or cannot be resolved."""


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidueToken:
    """One residue named in a spec, in author numbering."""

    resseq: int
    chain: str | None = None
    icode: str = ""
    expect_aa: str | None = None  # one-letter; validated on resolution

    def matches(self, rec: st.ResidueRecord) -> bool:
        if rec.resseq != self.resseq:
            return False
        if rec.icode.strip() != self.icode.strip():
            return False
        if self.chain is not None and rec.chain != self.chain:
            return False
        return True

    def describe(self) -> str:
        ch = f"{self.chain}/" if self.chain else ""
        return f"{ch}{self.expect_aa or ''}{self.resseq}{self.icode}"


@dataclass(frozen=True)
class LigandToken:
    """A ligand whose contact shell defines part of the site."""

    comp_id: str
    cutoff: float = 4.5

    def describe(self) -> str:
        return f"ligand:{self.comp_id}@{self.cutoff:g}"


@dataclass(frozen=True)
class SiteSpec:
    """A parsed, unresolved site specification."""

    residues: tuple[ResidueToken, ...] = ()
    ligands: tuple[LigandToken, ...] = ()
    name: str = "site"
    default_chain: str | None = None

    def __bool__(self) -> bool:
        return bool(self.residues or self.ligands)

    def describe(self) -> str:
        parts = [t.describe() for t in self.residues]
        parts += [t.describe() for t in self.ligands]
        return " ".join(parts) if parts else "(empty)"


_AA1 = set("ACDEFGHIKLMNPQRSTVWY")

# "A/H46", "H46", "46", "A:46", "A/46A" (trailing insertion code)
_RES_RE = re.compile(
    r"^(?:(?P<chain>[A-Za-z0-9])\s*[/:]\s*)?"
    r"(?P<aa>[A-Za-z])?"
    r"(?P<num>-?\d+)"
    r"(?P<icode>[A-Za-z])?$"
)
_LIG_RE = re.compile(
    r"^ligand\s*[:=]\s*(?P<comp>[A-Za-z0-9]{1,5})(?:\s*@\s*(?P<cut>[0-9.]+))?$",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    r"^(?:(?P<chain>[A-Za-z0-9])\s*[/:]\s*)?"
    r"range\s*[:=]\s*(?P<lo>-?\d+)\s*-\s*(?P<hi>-?\d+)$",
    re.IGNORECASE,
)


def parse_site_spec(
    text: str,
    default_chain: str | None = None,
    name: str = "site",
) -> SiteSpec:
    """Parse a site spec.

    Tokens are separated by commas, whitespace or newlines; ``#`` starts a
    comment.  Recognised forms::

        H46              residue 46, expected to be histidine
        46               residue 46, identity not asserted
        A/H46  A:46      an explicit chain
        46A              an insertion code
        range:110-116    an inclusive span of residue numbers
        ligand:ATP       every residue within 4.5 A of any ATP atom
        ligand:ATP@6.0   the same, at a 6 A cutoff

    Naming the expected residue is optional but worth doing: it is what turns
    a numbering mismatch into an error message instead of a silently wrong
    run.
    """
    residues: list[ResidueToken] = []
    ligands: list[LigandToken] = []

    cleaned = re.sub(r"#[^\n]*", " ", text or "")
    tokens = [t for t in re.split(r"[,\s]+", cleaned) if t]

    for token in tokens:
        lig = _LIG_RE.match(token)
        if lig:
            comp = lig.group("comp").upper()
            cut = float(lig.group("cut")) if lig.group("cut") else 4.5
            if not 1.0 <= cut <= 15.0:
                raise SiteError(
                    f"ligand cutoff {cut} A out of range (1-15) in {token!r}"
                )
            ligands.append(LigandToken(comp, cut))
            continue

        rng = _RANGE_RE.match(token)
        if rng:
            lo, hi = int(rng.group("lo")), int(rng.group("hi"))
            if hi < lo:
                raise SiteError(f"range {token!r} runs backwards")
            if hi - lo > 500:
                raise SiteError(f"range {token!r} is implausibly large")
            chain = rng.group("chain") or default_chain
            for num in range(lo, hi + 1):
                residues.append(ResidueToken(num, chain))
            continue

        res = _RES_RE.match(token)
        if res:
            aa = (res.group("aa") or "").upper() or None
            if aa is not None and aa not in _AA1:
                raise SiteError(
                    f"{token!r}: {aa!r} is not a standard one-letter residue code"
                )
            residues.append(
                ResidueToken(
                    resseq=int(res.group("num")),
                    chain=res.group("chain") or default_chain,
                    icode=(res.group("icode") or "").upper(),
                    expect_aa=aa,
                )
            )
            continue

        raise SiteError(
            f"cannot parse site token {token!r}. Expected forms: H46, 46, "
            "A/H46, 46A, range:10-20, ligand:ATP, ligand:ATP@5.0"
        )

    # Deduplicate residues, keeping the first mention (which carries any
    # asserted identity).
    seen: set[tuple[str | None, int, str]] = set()
    unique: list[ResidueToken] = []
    for tok in residues:
        key = (tok.chain, tok.resseq, tok.icode.strip())
        if key in seen:
            continue
        seen.add(key)
        unique.append(tok)

    return SiteSpec(
        residues=tuple(unique),
        ligands=tuple(dict.fromkeys(ligands)),
        name=name,
        default_chain=default_chain,
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SitePosition:
    """One site residue, resolved against a prepared structure."""

    index: int  # 0-based position in the designed sequence
    chain: str
    resseq: int
    icode: str
    native_aa: str
    role: str
    expect_aa: str | None = None
    min_distance: float | None = None  # to the defining ligand, if derived
    n_contacts: int = 0

    @property
    def label(self) -> str:
        ic = self.icode.strip()
        return f"{self.native_aa}{self.resseq}{ic}"


@dataclass
class ResolvedSite:
    """A site spec bound to one prepared structure."""

    name: str
    positions: tuple[SitePosition, ...] = ()
    ligands: tuple[str, ...] = ()
    ligand_atom_counts: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    spec_text: str = ""

    def __bool__(self) -> bool:
        return bool(self.positions)

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(p.index for p in self.positions)

    @property
    def catalytic_indices(self) -> tuple[int, ...]:
        return tuple(p.index for p in self.positions if p.role == ROLE_CATALYTIC)

    @property
    def native_aas(self) -> str:
        return "".join(p.native_aa for p in self.positions)

    @property
    def residue_types(self) -> frozenset[str]:
        """The distinct amino acids the site needs."""
        return frozenset(p.native_aa for p in self.positions)

    def labels(self) -> list[str]:
        return [p.label for p in self.positions]

    def describe(self) -> str:
        if not self.positions:
            return "no site"
        cat = [p for p in self.positions if p.role == ROLE_CATALYTIC]
        con = [p for p in self.positions if p.role == ROLE_CONTACT]
        bits = []
        if cat:
            bits.append(f"catalytic {'+'.join(p.label for p in cat)}")
        if con:
            bits.append(
                f"{len(con)} contact residues ({''.join(sorted(p.native_aa for p in con))})"
            )
        return "; ".join(bits)


def _author_index(records: list[st.ResidueRecord]) -> list[st.ResidueRecord]:
    return [r for r in records if not r.is_hetatm]


def ligand_groups(
    records: list[st.ResidueRecord],
    comp_ids: tuple[str, ...] | None = None,
) -> list[st.ResidueRecord]:
    """Heteroatom groups that could define a pocket.

    With *comp_ids*, returns exactly those.  Without, returns every retained
    group that is neither solvent nor a lone ion -- the candidates worth
    offering to a user who does not yet know what is in their file.
    """
    out = []
    for rec in records:
        if not rec.is_hetatm:
            continue
        name = rec.resname.upper()
        if comp_ids is not None:
            if name in {c.upper() for c in comp_ids}:
                out.append(rec)
            continue
        if name in SOLVENT_COMPONENTS or name in ION_COMPONENTS:
            continue
        out.append(rec)
    return out


def contact_shell(
    records: list[st.ResidueRecord],
    comp_ids: tuple[str, ...],
    cutoff: float = 4.5,
    max_residues: int = 0,
) -> tuple[list[tuple[int, float, int]], list[str]]:
    """Residues contacting the named ligands.

    Returns ``([(index, min_distance, n_contacting_atoms), ...], warnings)``
    sorted by distance.  Distances are measured from *every parsed protein
    atom* of the residue to every ligand atom, so which atoms were parsed
    changes the answer -- pass records parsed with a wide ``atom_names`` set
    for a true all-atom shell, or the backbone frame for a backbone-contact
    shell.  ``max_residues`` keeps only the closest *n*, which is how a
    sprawling shell is kept from swallowing the protein.
    """
    poly = _author_index(records)
    groups = ligand_groups(records, comp_ids)
    warnings: list[str] = []
    if not groups:
        return [], [
            f"no atoms found for ligand(s) {', '.join(comp_ids)}; "
            "the site has no ligand-derived residues"
        ]

    lig_xyz = np.asarray(
        [xyz for g in groups for xyz in g.atoms.values()], dtype=np.float64
    )
    if lig_xyz.size == 0:
        return [], [f"ligand(s) {', '.join(comp_ids)} have no usable coordinates"]

    found: list[tuple[int, float, int]] = []
    cut2 = cutoff * cutoff
    for i, rec in enumerate(poly):
        if not rec.atoms:
            continue
        prot = np.asarray(list(rec.atoms.values()), dtype=np.float64)
        d2 = ((prot[:, None, :] - lig_xyz[None, :, :]) ** 2).sum(axis=2)
        n_close = int((d2 <= cut2).sum())
        if n_close:
            found.append((i, float(np.sqrt(d2.min())), n_close))

    found.sort(key=lambda t: (t[1], -t[2]))
    if max_residues and len(found) > max_residues:
        warnings.append(
            f"ligand shell for {', '.join(comp_ids)} had {len(found)} residues "
            f"within {cutoff:g} A; kept the {max_residues} closest "
            "(SITE_CONTACT_MAX)"
        )
        found = found[:max_residues]
    return found, warnings


def resolve_site(
    spec: SiteSpec,
    records: list[st.ResidueRecord],
    contact_max: int = 0,
    name: str | None = None,
) -> ResolvedSite:
    """Bind *spec* to one prepared structure.

    *records* must come from the prepared, pipeline-ready PDB -- the same file
    MPNN is given -- parsed with ``include_hetatm=True`` so ligand-derived
    residues can be found.  Author numbering is looked up in that file, which
    is the only way the resulting indices line up with the designed sequence.
    """
    poly = _author_index(records)
    warnings: list[str] = []
    by_role: dict[int, SitePosition] = {}

    if not poly:
        raise SiteError("prepared structure has no polymer residues")

    # -- explicitly named residues ------------------------------------
    for tok in spec.residues:
        hits = [i for i, rec in enumerate(poly) if tok.matches(rec)]
        if not hits:
            chains = sorted({r.chain for r in poly})
            nums = [r.resseq for r in poly]
            warnings.append(
                f"residue {tok.describe()} is not in the prepared structure "
                f"(chains {','.join(chains)}, numbering {min(nums)}..{max(nums)}) "
                "-- it may have been dropped for an incomplete backbone, or "
                "the spec may use different numbering"
            )
            continue
        if len(hits) > 1:
            warnings.append(
                f"residue {tok.describe()} is ambiguous ({len(hits)} matches); "
                "name the chain explicitly"
            )
        i = hits[0]
        rec = poly[i]
        if tok.expect_aa and rec.aa != tok.expect_aa:
            warnings.append(
                f"NUMBERING MISMATCH: spec says {tok.describe()} but the "
                f"structure has {rec.aa}{rec.resseq} at that position"
            )
        by_role[i] = SitePosition(
            index=i, chain=rec.chain, resseq=rec.resseq, icode=rec.icode,
            native_aa=rec.aa, role=ROLE_CATALYTIC, expect_aa=tok.expect_aa,
        )

    # -- ligand-derived contact shells --------------------------------
    atom_counts: dict[str, int] = {}
    for lig in spec.ligands:
        groups = ligand_groups(records, (lig.comp_id,))
        atom_counts[lig.comp_id] = sum(len(g.atoms) for g in groups)
        found, warns = contact_shell(
            records, (lig.comp_id,), lig.cutoff, max_residues=contact_max
        )
        warnings.extend(warns)
        for i, dist, n_atoms in found:
            rec = poly[i]
            existing = by_role.get(i)
            if existing is not None:
                # A named catalytic residue that also contacts the ligand
                # keeps its role, but gains the measured distance.
                by_role[i] = replace(
                    existing, min_distance=dist, n_contacts=n_atoms
                )
                continue
            by_role[i] = SitePosition(
                index=i, chain=rec.chain, resseq=rec.resseq, icode=rec.icode,
                native_aa=rec.aa, role=ROLE_CONTACT,
                min_distance=dist, n_contacts=n_atoms,
            )

    positions = tuple(by_role[i] for i in sorted(by_role))

    # A site that covers most of the protein is not a site, and silently
    # exempting that many positions would make the reported alphabet
    # meaningless.  Warn loudly rather than refuse: a small domain with a
    # large cofactor genuinely can look like this.
    if positions and len(positions) > max(8, 0.25 * len(poly)):
        warnings.append(
            f"site covers {len(positions)} of {len(poly)} residues "
            f"({100.0 * len(positions) / len(poly):.0f}%); with a large "
            "exempt fraction the reported alphabet size describes only the "
            "remaining positions"
        )

    return ResolvedSite(
        name=name or spec.name,
        positions=positions,
        ligands=tuple(l.comp_id for l in spec.ligands),
        ligand_atom_counts=atom_counts,
        warnings=tuple(warnings),
        spec_text=spec.describe(),
    )


# ---------------------------------------------------------------------------
# Site preservation measurement
# ---------------------------------------------------------------------------


@dataclass
class SiteMetrics:
    """How well one predicted structure preserved the site."""

    bb_rmsd: float | None = None  # site N/CA/C, global superposition
    local_rmsd: float | None = None  # site N/CA/C, site-only superposition
    cb_max_dev: float | None = None  # worst CB-CB distance error
    cb_rms_dev: float | None = None  # RMS CB-CB distance error
    identity_kept: bool = True  # site residues still the native ones
    n_positions: int = 0

    def passed(
        self,
        max_bb_rmsd: float,
        max_local_rmsd: float,
        max_cb_dev: float,
        require_identity: bool = True,
    ) -> bool:
        if require_identity and not self.identity_kept:
            return False
        checks = (
            (self.bb_rmsd, max_bb_rmsd),
            (self.local_rmsd, max_local_rmsd),
            (self.cb_max_dev, max_cb_dev),
        )
        for value, limit in checks:
            if limit <= 0:
                continue  # that check is switched off
            if value is None:
                return False  # asked for, not measurable: not a pass
            if value > limit:
                return False
        return True

    def describe(self) -> str:
        def f(v, spec="0.2f"):
            return f"{v:{spec}}" if v is not None else "N/A"

        return (
            f"siteBB={f(self.bb_rmsd)} siteLocal={f(self.local_rmsd)} "
            f"cbDev={f(self.cb_max_dev)}"
            + ("" if self.identity_kept else " IDENTITY_LOST")
        )


def measure_site(
    site: ResolvedSite,
    ref_frames: np.ndarray,
    pred_frames: np.ndarray,
    designed_seq: str | None = None,
) -> SiteMetrics:
    """Compare a prediction's site geometry with the reference's.

    All three numbers are backbone-derived (see :mod:`redaa.structure` on why
    CB is constructed rather than read), so they report whether the scaffold
    puts the functional residues where they belong -- not whether a predicted
    rotamer happened to land well.  They are a necessary condition for a
    working site, not a sufficient one.
    """
    positions = list(site.indices)
    metrics = SiteMetrics(n_positions=len(positions))
    if not positions:
        return metrics

    metrics.bb_rmsd = st.site_rmsd(
        ref_frames, pred_frames, positions, superpose="global"
    )
    metrics.local_rmsd = st.site_rmsd(
        ref_frames, pred_frames, positions, superpose="local"
    )
    metrics.cb_max_dev, metrics.cb_rms_dev = st.site_distance_deviation(
        ref_frames, pred_frames, positions, atom="CB"
    )

    if designed_seq is not None:
        metrics.identity_kept = site_identity_kept(site, designed_seq)
    return metrics


def site_identity_kept(site: ResolvedSite, sequence: str) -> bool:
    """Whether *sequence* still carries the native residue at every site position."""
    for pos in site.positions:
        if pos.index >= len(sequence):
            return False
        if sequence[pos.index] != pos.native_aa:
            return False
    return True


def apply_site_to_sequence(site: ResolvedSite, sequence: str) -> str:
    """Force the native residue at each site position.

    Used when grafting a site onto a sequence designed without it, and as a
    belt-and-braces step after a design run that was supposed to hold those
    positions fixed.
    """
    if not site.positions:
        return sequence
    chars = list(sequence)
    for pos in site.positions:
        if pos.index < len(chars):
            chars[pos.index] = pos.native_aa
    return "".join(chars)


# ---------------------------------------------------------------------------
# Verified test cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SitePreset:
    """A ready-made test case: where to get the structure and what its site is."""

    key: str
    fetch_spec: str
    site_spec: str
    keep_ligands: tuple[str, ...]
    mpnn_model_type: str
    title: str
    notes: str

    def describe(self) -> str:
        return (
            f"{self.key}: {self.title}\n"
            f"    structure : {self.fetch_spec}\n"
            f"    site      : {self.site_spec}\n"
            f"    ligands   : {', '.join(self.keep_ligands) or 'none'}\n"
            f"    mpnn      : {self.mpnn_model_type}\n"
            f"    {self.notes}"
        )


# Both of these were checked against the deposited coordinates rather than
# taken from memory: the residue numbering, the chain, the ligand component
# ids and the contact shells below were all computed from the files.
PRESETS: dict[str, SitePreset] = {
    "atp": SitePreset(
        key="atp",
        # Chain A only, and the whole chain: numbering is 0..158 with no gaps.
        fetch_spec="5etn_A",
        # The pocket is derived from the ATP coordinates rather than listed,
        # so it describes this structure exactly.  At 4.5 A it is 17 residues,
        # among them the arginine cluster that grips the triphosphate.
        site_spec="ligand:ATP@4.5",
        keep_ligands=("ATP",),
        mpnn_model_type="ligand_mpnn",
        title="ATP binding site -- E. coli HPPK (5ETN, 1.4 A)",
        notes=(
            "Monomeric, single chain, 159 modelled residues, genuine ATP "
            "complex. Monomeric matters: the obvious small ATP structures "
            "(PII/SbtB, 6MM2 and 2XBP) are trimers whose pocket sits between "
            "subunits, so a single-chain design can never reconstitute it. "
            "The 4.5 A shell is 17 residues, including the R82/R84/R92/R110/"
            "R121 cluster on the triphosphate. This entry also holds a pterin "
            "analogue (5RX) and Ca/Na ions -- the divalent site is Ca here, "
            "not the physiological Mg -- and none of them are part of the ATP "
            "pocket, so none are kept."
        ),
    ),
    "tev": SitePreset(
        key="tev",
        # Residues 1-221: -8..-1 is a GHHHHHHH purification tag, which would
        # fold as a floppy terminus and drag the native round-trip down for
        # no scientific reason.
        fetch_spec="1lvm_A:1-221",
        # The classic Cys-His-Asp triad of a chymotrypsin-like cysteine
        # protease. Asp148 is included because it sets the P1' specificity
        # that makes the ENLYFQ/S site what it is, and 149-150 complete the
        # oxyanion-forming loop.
        site_spec="H46 D81 C151 D148 G149 Q150",
        keep_ligands=(),
        mpnn_model_type="protein_mpnn",
        title="Catalytic triad -- TEV protease (1LVM, 1.8 A)",
        notes=(
            "Chain A is the protease; C/D/E are product peptide. Verified in "
            "the deposited file: His46 ND1-Asp81 OD2 is 3.09 A and "
            "His46 NE2-Cys151 SG is 4.02 A, so the triad is intact and the "
            "numbering is the one the spec uses. No ligand is kept, so this "
            "case runs on protein_mpnn."
        ),
    ),
}


def preset(key: str) -> SitePreset:
    try:
        return PRESETS[key.strip().lower()]
    except KeyError:
        raise SiteError(
            f"unknown site preset {key!r}; available: {', '.join(sorted(PRESETS))}"
        ) from None


def describe_presets() -> str:
    return "\n".join(PRESETS[k].describe() for k in sorted(PRESETS))
