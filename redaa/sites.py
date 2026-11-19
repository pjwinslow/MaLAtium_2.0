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
# Residues contacting a retained partner *chain* -- another protein subunit,
# or a nucleic acid. Distinguished from a small-molecule contact because the
# failure modes differ: a partner interface is usually larger, and part of it
# may belong to a subunit a single-chain design cannot reconstitute.
ROLE_INTERFACE = "interface"

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
    # Buffers, reductants and crystallisation additives. IMD in particular is
    # everywhere -- it turned up as a "ligand" candidate beside the real one
    # in a GlyRS structure.
    "IMD", "BTB", "B3P", "MRD", "SPD", "SPM", "DTT", "NHE", "LDA", "C8E",
    "PE4", "P33", "PGO", "BU3", "GOL",
})

# Monatomic ions.  Not solvent -- a catalytic zinc or the magnesium that makes
# an ATP site an ATP site is functional -- but too small to define a pocket on
# their own, so they are never auto-selected as "the ligand".  Naming one
# explicitly works.
ION_COMPONENTS = frozenset({
    "ZN", "MG", "MN", "CA", "NA", "K", "FE", "FE2", "CU", "CU1", "CO",
    "NI", "CD", "HG", "LI", "SR", "BA", "CS", "RB", "AL", "W", "MO",
})


# Standard nucleic-acid residues. Excluded from small-molecule ligand
# auto-detection: a tRNA is not a ligand, and advertising its 75 nucleotides
# as 75 candidate "ligands" would bury the real ones. A nucleic acid is kept
# as a partner chain instead, which is what it is.
NUCLEIC_COMPONENTS = frozenset({
    "A", "U", "G", "C", "I",
    "DA", "DT", "DG", "DC", "DU", "DI",
    "PSU", "5MC", "5MU", "1MA", "2MG", "M2G", "7MG", "OMC", "OMG",
    "H2U", "4SU", "YG", "QUO", "UR3", "G7M", "MIA",
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
class PartnerToken:
    """A chain whose interface with the designed chain defines part of a site."""

    chain: str
    cutoff: float = 4.5

    def describe(self) -> str:
        return f"partner:{self.chain}@{self.cutoff:g}"


@dataclass(frozen=True)
class SiteSpec:
    """A parsed, unresolved site specification."""

    residues: tuple[ResidueToken, ...] = ()
    ligands: tuple[LigandToken, ...] = ()
    partners: tuple[PartnerToken, ...] = ()
    name: str = "site"
    default_chain: str | None = None

    def __bool__(self) -> bool:
        return bool(self.residues or self.ligands or self.partners)

    def describe(self) -> str:
        parts = [t.describe() for t in self.residues]
        parts += [t.describe() for t in self.ligands]
        parts += [t.describe() for t in self.partners]
        return " ".join(parts) if parts else "(empty)"


_AA1 = set("ACDEFGHIKLMNPQRSTVWY")

# "A/H46", "H46", "46", "A:46", "A/46A" (trailing insertion code)
_RES_RE = re.compile(
    r"^(?:(?P<chain>[A-Za-z0-9])\s*[/:]\s*)?"
    r"(?P<aa>[A-Za-z])?"
    r"(?P<num>-?\d+)"
    r"(?P<icode>[A-Za-z])?$"
)
_PARTNER_RE = re.compile(
    r"^partner\s*[:=]\s*(?P<chain>[A-Za-z0-9])(?:\s*@\s*(?P<cut>[0-9.]+))?$",
    re.IGNORECASE,
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
        partner:E        every residue within 4.5 A of retained chain E --
                         another subunit, or a nucleic acid such as a tRNA
        partner:E@5.0    the same, at a 5 A cutoff

    Naming the expected residue is optional but worth doing: it is what turns
    a numbering mismatch into an error message instead of a silently wrong
    run.
    """
    residues: list[ResidueToken] = []
    ligands: list[LigandToken] = []
    partners: list[PartnerToken] = []

    cleaned = re.sub(r"#[^\n]*", " ", text or "")
    tokens = [t for t in re.split(r"[,\s]+", cleaned) if t]

    for token in tokens:
        part = _PARTNER_RE.match(token)
        if part:
            cut = float(part.group("cut")) if part.group("cut") else 4.5
            if not 1.0 <= cut <= 15.0:
                raise SiteError(
                    f"partner cutoff {cut} A out of range (1-15) in {token!r}"
                )
            partners.append(PartnerToken(part.group("chain").upper(), cut))
            continue

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
        partners=tuple(dict.fromkeys(partners)),
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
    partners: tuple[str, ...] = ()
    partner_atom_counts: dict[str, int] = field(default_factory=dict)
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
    def interface_indices(self) -> tuple[int, ...]:
        return tuple(p.index for p in self.positions if p.role == ROLE_INTERFACE)

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
        iface = [p for p in self.positions if p.role == ROLE_INTERFACE]
        bits = []
        if cat:
            bits.append(f"catalytic {'+'.join(p.label for p in cat)}")
        if con:
            bits.append(
                f"{len(con)} contact residues ({''.join(sorted(p.native_aa for p in con))})"
            )
        if iface:
            bits.append(
                f"{len(iface)} interface residues to chain "
                f"{'+'.join(self.partners) or '?'} "
                f"({''.join(sorted(p.native_aa for p in iface))})"
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

    found, warns = _shell_from_atoms(
        poly, lig_xyz, cutoff, max_residues, f"ligand shell for {', '.join(comp_ids)}"
    )
    return found, warnings + warns


def _shell_from_atoms(
    poly: list[st.ResidueRecord],
    other_xyz: np.ndarray,
    cutoff: float,
    max_residues: int,
    label: str,
) -> tuple[list[tuple[int, float, int]], list[str]]:
    """Polymer residues within *cutoff* of any row of *other_xyz*."""
    warnings: list[str] = []
    found: list[tuple[int, float, int]] = []
    cut2 = cutoff * cutoff
    for i, rec in enumerate(poly):
        if not rec.atoms:
            continue
        prot = np.asarray(list(rec.atoms.values()), dtype=np.float64)
        d2 = ((prot[:, None, :] - other_xyz[None, :, :]) ** 2).sum(axis=2)
        n_close = int((d2 <= cut2).sum())
        if n_close:
            found.append((i, float(np.sqrt(d2.min())), n_close))

    found.sort(key=lambda t: (t[1], -t[2]))
    if max_residues and len(found) > max_residues:
        warnings.append(
            f"{label} had {len(found)} residues within {cutoff:g} A; kept the "
            f"{max_residues} closest (SITE_CONTACT_MAX)"
        )
        found = found[:max_residues]
    return found, warnings


def partner_atoms(
    records: list[st.ResidueRecord], chain: str
) -> list[st.ResidueRecord]:
    """Retained partner-chain groups: heteroatom records on *chain*.

    ``prepare()`` writes a retained partner chain as HETATM records keeping
    its original chain id, so it is invisible to the polymer parsers (and so
    never folded, designed or scored) while remaining visible here and to
    LigandMPNN's atom context.
    """
    want = chain.upper()
    return [
        rec for rec in records
        if rec.is_hetatm and rec.chain.upper() == want
    ]


def partner_shell(
    records: list[st.ResidueRecord],
    chain: str,
    cutoff: float = 4.5,
    max_residues: int = 0,
) -> tuple[list[tuple[int, float, int]], list[str]]:
    """Residues of the designed chain contacting retained partner *chain*."""
    poly = _author_index(records)
    groups = partner_atoms(records, chain)
    if not groups:
        kept = sorted({r.chain for r in records if r.is_hetatm})
        have = ", ".join(kept) if kept else "none"
        return [], [
            f"partner chain {chain} has no retained atoms (retained "
            f"non-polymer chains: {have}) -- was it kept by SITE_KEEP_PARTNERS "
            "when the input was prepared?"
        ]
    xyz = np.asarray(
        [c for g in groups for c in g.atoms.values()], dtype=np.float64
    )
    if xyz.size == 0:
        return [], [f"partner chain {chain} has no usable coordinates"]
    return _shell_from_atoms(
        poly, xyz, cutoff, max_residues, f"partner shell for chain {chain}"
    )


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
            # Show the neighbourhood. A bare "expected H46, found D46" leaves
            # the obvious question -- what IS this structure? -- unanswered,
            # and answering it has cost a round trip every time.
            lo = max(0, i - 4)
            context = " ".join(
                f"{r.aa}{r.resseq}" for r in poly[lo : i + 5]
            )
            elsewhere = [
                f"{r.aa}{r.resseq}" for r in poly if r.aa == tok.expect_aa
            ]
            hint = ""
            if elsewhere:
                shown = ", ".join(elsewhere[:8])
                more = "" if len(elsewhere) <= 8 else f", +{len(elsewhere) - 8} more"
                hint = f"; {tok.expect_aa} appears at {shown}{more}"
            warnings.append(
                f"NUMBERING MISMATCH: spec says {tok.describe()} but the "
                f"structure has {rec.aa}{rec.resseq} at that position. "
                f"Around it: {context}{hint}"
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

    # -- partner-chain interfaces -------------------------------------
    partner_counts: dict[str, int] = {}
    for part in spec.partners:
        groups = partner_atoms(records, part.chain)
        partner_counts[part.chain] = sum(len(g.atoms) for g in groups)
        found, warns = partner_shell(
            records, part.chain, part.cutoff, max_residues=contact_max
        )
        warnings.extend(warns)
        for i, dist, n_atoms in found:
            rec = poly[i]
            existing = by_role.get(i)
            if existing is not None:
                if existing.min_distance is None or dist < existing.min_distance:
                    by_role[i] = replace(
                        existing, min_distance=dist, n_contacts=n_atoms
                    )
                continue
            by_role[i] = SitePosition(
                index=i, chain=rec.chain, resseq=rec.resseq, icode=rec.icode,
                native_aa=rec.aa, role=ROLE_INTERFACE,
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
        partners=tuple(p.chain for p in spec.partners),
        partner_atom_counts=partner_counts,
        warnings=tuple(warnings),
        spec_text=spec.describe(),
    )


# ---------------------------------------------------------------------------
# Turning a resolved site plus its policies into the three sets that matter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SitePlan:
    """What a resolved site and its per-role policies ask of the search.

    Three sets, derived once, used everywhere.  The pipeline used to compare
    a single policy string at seven different places, which worked only while
    every site position was governed by the same policy.  With a catalytic
    pocket held exactly and a nucleic-acid interface left free, that is no
    longer true.

    ``fixed``   positions MPNN must leave at their native identity.
    ``exempt``  positions not counted against the reported alphabet.
    ``forced_types``  letters the alphabet must contain, because a position
                counted against it is held at that residue.
    ``gated``   positions the geometry thresholds are applied to.
    ``recorded``  the rest of the site: measured and reported, not gated.
    """

    fixed: dict[int, str] = field(default_factory=dict)
    exempt: frozenset[int] = frozenset()
    forced_types: frozenset[str] = frozenset()
    by_role: dict[str, str] = field(default_factory=dict)
    n_measured: int = 0
    gated: frozenset[int] = frozenset()
    recorded: frozenset[int] = frozenset()

    @property
    def any_fixed(self) -> bool:
        return bool(self.fixed)

    def describe(self, rest_gated: bool = False) -> str:
        """One line saying what the policies do to this site.

        *rest_gated* says whether the roles outside the tight gate have
        thresholds of their own, which changes what the line can honestly
        claim about them.
        """
        roles = ", ".join(f"{r}={p}" for r, p in sorted(self.by_role.items()))
        bits = [roles] if roles else []
        bits.append(f"{len(self.fixed)} held")
        if self.exempt:
            bits.append(f"{len(self.exempt)} uncounted")
        if self.forced_types:
            bits.append(f"requires {''.join(sorted(self.forced_types))}")
        if self.n_measured:
            bits.append(f"{self.n_measured} free but measured")
        if self.recorded:
            how = "loosely gated" if rest_gated else "recorded only"
            bits.append(
                f"{len(self.gated)} tightly gated, {len(self.recorded)} {how}"
            )
        return "; ".join(bits)


def site_plan(
    site: ResolvedSite | None,
    policies: dict[str, str],
    gate_roles: "tuple[str, ...] | None" = None,
) -> SitePlan:
    """Resolve *site* against a role -> policy map.

    Policies, per position, by its role:

    ``exempt``        held at native identity, not counted in the alphabet
    ``in_alphabet``   held at native identity, and its residue type counted
    ``measure_only``  designed freely; only the geometry is judged

    *gate_roles*, when given, limits which roles the geometry thresholds are
    applied to.  The rest of the site is still measured and reported.
    """
    if site is None or not site.positions:
        return SitePlan(by_role=dict(policies))

    fixed: dict[int, str] = {}
    exempt: set[int] = set()
    forced: set[str] = set()
    gated: set[int] = set()
    recorded: set[int] = set()
    measured = 0
    for pos in site.positions:
        policy = policies.get(pos.role, "exempt")
        if gate_roles is None or pos.role in gate_roles:
            gated.add(pos.index)
        else:
            recorded.add(pos.index)
        if policy == "measure_only":
            measured += 1
            continue
        fixed[pos.index] = pos.native_aa
        if policy == "in_alphabet":
            forced.add(pos.native_aa)
        else:
            exempt.add(pos.index)
    return SitePlan(
        fixed=fixed,
        exempt=frozenset(exempt),
        forced_types=frozenset(forced),
        by_role=dict(policies),
        n_measured=measured,
        gated=frozenset(gated),
        recorded=frozenset(recorded),
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
    # The same three numbers over the positions the gate does NOT cover.
    # Recorded so a run with a partly ungated site is still informative:
    # without this, loosening the gate would mean losing the measurement.
    recorded: "SiteMetrics | None" = None
    n_recorded: int = 0

    def passed(
        self,
        max_bb_rmsd: float,
        max_local_rmsd: float,
        max_cb_dev: float,
        require_identity: bool = True,
        max_cb_rms_dev: float = 0.0,
    ) -> bool:
        """Whether this fold clears the configured site thresholds.

        A note on ``max_cb_dev`` versus ``max_cb_rms_dev``. The former gates
        the *worst* CB-CB pair, and the number of pairs grows as n^2 with the
        site size, so at identical backbone accuracy it rises with the size
        of the site: measured on a real 519-residue chain under 1 A of
        correlated backbone error, the worst pair averages 2.6 A over 6
        positions and 4.6 A over 68, while the RMS over pairs sits at 1.38 A
        for both. A threshold calibrated on a catalytic triad is therefore not
        the same test on a twenty-residue pocket. Gate the max for a small,
        named site; gate the RMS when the site is large, which is what makes
        the two comparable across cases.
        """
        if require_identity and not self.identity_kept:
            return False
        checks = (
            (self.bb_rmsd, max_bb_rmsd),
            (self.local_rmsd, max_local_rmsd),
            (self.cb_max_dev, max_cb_dev),
            (self.cb_rms_dev, max_cb_rms_dev),
        )
        for value, limit in checks:
            if limit <= 0:
                continue  # that check is switched off
            if value is None:
                return False  # asked for, not measurable: not a pass
            if value > limit:
                return False
        return True

    def rest_passed(
        self,
        max_bb_rmsd: float,
        max_local_rmsd: float,
        max_cb_dev: float = 0.0,
        max_cb_rms_dev: float = 0.0,
    ) -> bool:
        """Whether the roles outside the tight gate clear their own thresholds.

        Two threshold sets exist because the two halves of a site are asked
        for different things, not because one is measured differently. A
        catalytic pocket has to put side chains within bonding distance of a
        substrate, so its geometry is judged tightly. An interface to a
        75-nucleotide RNA has to be in roughly the right place and present
        roughly the right surface; demanding Angstrom precision there tests
        the fold model, not the design.

        With every threshold at 0 this is measurement only, which is the
        default and the old behaviour.
        """
        if self.recorded is None:
            # Nothing outside the gate. Thresholds on an empty group cannot
            # fail: there is no claim being made.
            return True
        return self.recorded.passed(
            max_bb_rmsd,
            max_local_rmsd,
            max_cb_dev,
            require_identity=False,
            max_cb_rms_dev=max_cb_rms_dev,
        )

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
    held: "dict[int, str] | None" = None,
    gate_positions: "frozenset[int] | None" = None,
) -> SiteMetrics:
    """Compare a prediction's site geometry with the reference's.

    All three numbers are backbone-derived (see :mod:`redaa.structure` on why
    CB is constructed rather than read), so they report whether the scaffold
    puts the functional residues where they belong -- not whether a predicted
    rotamer happened to land well.  They are a necessary condition for a
    working site, not a sufficient one.

    With *gate_positions*, the three gated numbers cover only those, and the
    remaining site positions are measured into ``metrics.recorded``.  A site
    with a compact pocket and a chain-spanning interface cannot be judged by
    one set of thresholds, but the ungated half is still worth knowing.
    """
    all_positions = list(site.indices)
    positions = (
        all_positions if gate_positions is None
        else [i for i in all_positions if i in gate_positions]
    )
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

    rest = [i for i in all_positions if i not in positions]
    if rest:
        other = SiteMetrics(n_positions=len(rest))
        other.bb_rmsd = st.site_rmsd(
            ref_frames, pred_frames, rest, superpose="global"
        )
        other.local_rmsd = st.site_rmsd(
            ref_frames, pred_frames, rest, superpose="local"
        )
        other.cb_max_dev, other.cb_rms_dev = st.site_distance_deviation(
            ref_frames, pred_frames, rest, atom="CB"
        )
        metrics.recorded = other
        metrics.n_recorded = len(rest)

    if designed_seq is not None:
        # Only the positions the plan holds. A position left free by policy
        # is *supposed* to change, so counting it as lost identity would
        # condemn every design.
        metrics.identity_kept = site_identity_kept(site, designed_seq, only=held)
    return metrics


def site_identity_kept(
    site: ResolvedSite,
    sequence: str,
    only: "dict[int, str] | None" = None,
) -> bool:
    """Whether *sequence* carries the native residue where it was meant to.

    With *only* -- a ``SitePlan.fixed`` map -- just those positions are
    checked.  Checking the rest would fail by definition: under
    ``measure_only`` the design is supposed to choose its own residues there.
    """
    wanted = (
        only if only is not None
        else {pos.index: pos.native_aa for pos in site.positions}
    )
    for index, aa in wanted.items():
        if index >= len(sequence):
            return False
        if sequence[index] != aa:
            return False
    return True


def apply_site_to_sequence(
    site: ResolvedSite,
    sequence: str,
    only: "dict[int, str] | None" = None,
) -> str:
    """Force the native residue at each held site position.

    Used when grafting a site onto a sequence designed without it, and as a
    belt-and-braces step after a design run that was supposed to hold those
    positions fixed.  With *only*, positions left free by policy are not
    overwritten.
    """
    wanted = (
        only if only is not None
        else {pos.index: pos.native_aa for pos in site.positions}
    )
    if not wanted:
        return sequence
    chars = list(sequence)
    for index, aa in wanted.items():
        if index < len(chars):
            chars[index] = aa
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
    # Whole chains to retain as binding context.
    keep_partners: tuple[str, ...] = ()
    # Settings this case needs that the blanket defaults get wrong, as
    # (env var, value) pairs. One mechanism rather than one field per
    # setting: a case may need a policy, a gate scope and a threshold, and
    # which it needs is a property of the case, not of the preset format.
    # Applied only where the user has not set the variable themselves.
    settings: tuple[tuple[str, str], ...] = ()

    def describe(self) -> str:
        lines = [
            f"{self.key}: {self.title}",
            f"    structure : {self.fetch_spec}",
            f"    site      : {self.site_spec}",
            f"    ligands   : {', '.join(self.keep_ligands) or 'none'}",
        ]
        if self.keep_partners:
            lines.append(f"    partners  : {', '.join(self.keep_partners)}")
        lines.append(f"    mpnn      : {self.mpnn_model_type}")
        for key, value in self.settings:
            lines.append(f"    {key:<24s}: {value}")
        lines.append(f"    {self.notes}")
        return "\n".join(lines)


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
    "glyrs": SitePreset(
        key="glyrs",
        # Chain A, whole. 519 modelled residues numbered 7-678 with 4 gaps.
        # File numbering + 54 = UniProt P41250 numbering, so literature
        # residue numbers do NOT match the spec below: the deposited
        # construct is UNP 55-739 renumbered from 1. The LEHHHHHH tag at
        # 686-693 is unmodelled, so nothing has to be trimmed.
        fetch_spec="5e6m_A",
        # Two sites from one structure, which is the reason for using this
        # entry: the adenylate pocket derived from the bound glycyl-adenylate,
        # and the tRNA interface derived from the tRNA that chain A binds.
        site_spec="ligand:GAP@4.5 partner:E@4.5",
        keep_ligands=("GAP",),
        keep_partners=("E",),
        mpnn_model_type="ligand_mpnn",
        # One blanket policy cannot serve both halves. The 4.5 A shell around
        # the adenylate is 20 residues; the tRNA interface is 48, and between
        # them they use 18 of the 20 amino acids. Held under in_alphabet the
        # "minimal" alphabet would be 18 letters before the search began.
        settings=(
            # The pocket is 20 residues using 9 amino acids; the tRNA
            # interface is 48 using 18. One policy for both would force an
            # 18-letter alphabet before the search started, so the interface
            # is designed freely and judged on geometry alone.
            ("SITE_RESIDUE_POLICY", "exempt,interface=measure_only"),
            # The pocket spans sequence indices 158-369; the interface spans
            # 54-515, essentially the whole chain. A 1.0 A local RMSD over a
            # chain-spanning 48-residue set is a harder test than the global
            # fold gate, so gating it would reject everything. The pocket is
            # gated; the interface is measured and reported.
            ("SITE_GATE_ROLES", "catalytic,contact"),
            # Measured on these coordinates under 1 A of correlated backbone
            # error: the worst CB-CB pair averages 2.6 A over 6 positions and
            # 3.7 A over this pocket's 20, because the pair count grows as
            # n^2. The RMS over pairs is 1.38 A either way. So the worst-pair
            # check is switched off and the size-stable one gated instead.
            ("SITE_MAX_CB_DEV", "0"),
            ("SITE_MAX_CB_RMS_DEV", "2.0"),
            # The interface IS gated, on its own looser numbers. The two
            # halves are asked for different things: the pocket has to put
            # side chains within bonding distance of an adenylate, while the
            # interface has to be in the right place and present the right
            # surface to a 75-nucleotide RNA. Measured on these coordinates,
            # the interface shows BB 2.48 A / local 2.30 A / cbRMS 1.94 A
            # under 1.5 A of correlated backbone error and roughly 4 / 3.9 /
            # 3.4 under 2.5 A, so these thresholds accept a domain that is
            # placed correctly to within about 2 A and reject one that has
            # drifted further.
            ("SITE_REST_MAX_BB_RMSD", "3.0"),
            ("SITE_REST_MAX_LOCAL_RMSD", "2.5"),
            ("SITE_REST_MAX_CB_RMS_DEV", "3.0"),
        ),
        title="Glycyl-adenylate pocket + tRNA interface -- human GlyRS (5E6M, 2.93 A)",
        notes=(
            "Human wild-type GlyRS (GARS, P41250, EC 6.1.1.14) with "
            "tRNA-Gly; chains A and B are the protein, C and E the tRNA. "
            "Computed from the deposited file, not recalled: GAP is "
            "glycyl-adenylate -- the reaction intermediate itself, 27 atoms, "
            "its glycyl N/CA/C/O, the alpha-phosphate and the whole "
            "adenosine -- and all 27 of its atoms lie within 4.5 A of chain "
            "A while none come within 11 A of chain B, so the catalytic "
            "pocket is entirely intra-subunit and a single-chain design can "
            "reconstitute it. The tRNA is not: chain E contacts 75 residues "
            "of chain A and 21 of chain B at 6 A, so about a fifth of that "
            "interface belongs to the other subunit and cannot be reproduced "
            "by a monomer. GlyRS is an obligate dimer in any case (chain B "
            "contacts 98 residues of chain A), so treat a result here as one "
            "subunit of a dimer rather than a free-standing protein. Chain E "
            "is paired with chain A deliberately: chain C binds chain B."
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
