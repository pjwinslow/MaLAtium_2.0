"""Alphabet search: sequence selection, validation, forward greedy, pruning.

The algorithm is V9's -- forward greedy growth with per-round structural
validation, then backward pruning -- with three changes that cut fold count
without changing what the search decides on.

**Nested sequence selection.**  V9's stratified sampler picked a different set
of sequences for a budget of 1 than for a budget of 3, so a cheap first look
could not be reused.  :func:`select_sequences` instead produces one ordered
list whose every prefix is itself stratified across the MPNN score
distribution.  A budget of *k* is the first *k* entries, so a 1-fold screen is
a strict prefix of a 3-fold screen and the first fold is already cached when
the deeper look happens.

**Two-stage candidate screening.**  V9 folded 3 sequences for each of ~17
candidate residues every round: ~51 folds to pick one winner.  Here every
candidate gets 1 fold, then only the plausible winners -- everything that
already passed, plus the top few by TM-score -- get the full budget.  Thanks to
nesting, promotion costs only the *additional* folds. Same evidence for the
candidates that could actually win, at roughly half the cost.

**Downward-closure pruning.**  Exhaustive backward pruning tested every subset
at every size.  If a 5-letter alphabet cannot encode the fold, no 4-letter
subset of it can either, so those subsets need not be tested -- the Apriori
argument.  That collapses most of the exhaustive sweep.  Set
``PRUNING_DOWNWARD_CLOSURE=0`` for an unconditional sweep.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from . import sites as sitemod
from . import structure as st


def in_alphabet(seq: str, allowed: set[str], exempt: frozenset[int]) -> bool:
    """Whether every *designable* position of *seq* is inside *allowed*.

    Exempt functional-site positions are skipped. They hold native residues
    that are usually outside the restricted alphabet -- that is the point of
    exempting them -- so checking them would reject every design the
    site-aware search produces.
    """
    if not exempt:
        return set(seq) <= allowed
    return all(
        c in allowed for i, c in enumerate(seq) if i not in exempt
    )


# ---------------------------------------------------------------------------
# Sequence selection
# ---------------------------------------------------------------------------


def pairwise_identity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(x == y for x, y in zip(a[:n], b[:n])) / n


def diversity_filter(
    pairs: list[tuple[float, str]], max_identity: float = 0.70
) -> list[tuple[float, str]]:
    """Greedily keep sequences no two of which exceed *max_identity*.

    Input must be ascending by score so the best sequence always survives.
    """
    kept: list[tuple[float, str]] = []
    for score, seq in pairs:
        if not any(pairwise_identity(seq, other) > max_identity for _, other in kept):
            kept.append((score, seq))
    return kept


def _spread_order(n: int) -> list[int]:
    """Permutation of ``range(n)`` whose every prefix is spread over the range.

    Farthest-point ordering: start at 0, then repeatedly take the index
    furthest from everything already taken.  The result does not depend on how
    many elements the caller will eventually use, which is the property that
    makes :func:`select_sequences` prefix-stable -- and therefore makes a
    1-fold screen reusable inside a 3-fold screen.
    """
    if n <= 0:
        return []
    order = [0]
    remaining = list(range(1, n))
    while remaining:
        pick = max(remaining, key=lambda i: (min(abs(i - j) for j in order), -i))
        order.append(pick)
        remaining.remove(pick)
    return order


def select_sequences(
    pairs: list[tuple[float, str]],
    max_k: int,
    max_identity: float = 0.70,
) -> list[str]:
    """Order sequences so that every prefix is stratified by MPNN score.

    MPNN's score says how natural a sequence looks under the full 20-residue
    model, not whether it folds under a restricted alphabet.  At small
    alphabets the designs most compatible with the target topology often score
    poorly, because they trade naturalness for the fold.  Taking the top-k by
    score would systematically miss them, so the list round-robins between the
    best, median and worst thirds of the distribution.

    The ordering is independent of *max_k*, so ``select_sequences(pairs, 1)``
    is exactly the first element of ``select_sequences(pairs, 3)``.
    """
    if max_k <= 0 or not pairs:
        return []

    diverse = diversity_filter(sorted(pairs), max_identity=max_identity)
    if not diverse:
        return []

    third = max(1, len(diverse) // 3)
    pools = [diverse[:third], diverse[third : 2 * third], diverse[2 * third :]]
    picks = [
        [pool[i][1] for i in _spread_order(len(pool))] for pool in pools
    ]

    ordered: list[str] = []
    seen: set[str] = set()
    for i in range(max(len(p) for p in picks)):
        for pick in picks:
            if i < len(pick) and pick[i] not in seen:
                seen.add(pick[i])
                ordered.append(pick[i])
        if len(ordered) >= max_k:
            break
    return ordered[:max_k]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    mean_rmsd: float | None = None
    min_rmsd: float | None = None
    mean_tm: float | None = None
    max_tm: float | None = None
    mean_plddt: float | None = None
    max_plddt: float | None = None
    num_pass_all: int = 0
    successful_folds: int = 0
    total_attempted: int = 0
    # Site preservation, taken from the best-preserving fold in the set.
    site_active: bool = False
    site_bb_rmsd: float | None = None
    site_local_rmsd: float | None = None
    site_cb_dev: float | None = None
    num_site_pass: int = 0

    def rank_key(self) -> tuple[float, float, float]:
        """Sort key: site-preserving candidates first, then TM, then RMSD.

        The leading term is 0 for every candidate when no site is configured,
        so the ordering is exactly what it was before sites existed.
        """
        return (
            float(self.num_site_pass > 0) if self.site_active else 0.0,
            self.max_tm if self.max_tm is not None else -999.0,
            -self.min_rmsd if self.min_rmsd is not None else -999.0,
        )


@dataclass
class BestSequence:
    sequence: str | None = None
    alphabet: str | None = None
    rmsd: float | None = None
    tm: float | None = None
    plddt: float | None = None
    site_bb_rmsd: float | None = None
    site_local_rmsd: float | None = None
    site_cb_dev: float | None = None
    site_ok: bool | None = None

    def rank_key(self) -> tuple[float, float, float]:
        return (
            float(bool(self.site_ok)) if self.site_ok is not None else 0.0,
            self.tm if self.tm is not None else 0.0,
            -(self.rmsd if self.rmsd is not None else 999.0),
        )


@dataclass
class Validation:
    passed: bool = False
    metrics: Metrics = field(default_factory=Metrics)
    best: BestSequence = field(default_factory=BestSequence)
    folded: list[tuple[str, float | None, float | None, float | None]] = field(
        default_factory=list
    )


class Validator:
    """Folds candidate sequences and scores them against one reference."""

    def __init__(
        self,
        cfg,
        fold_engine,
        ref_xyz: np.ndarray,
        ref_seq: str,
        site: "sitemod.ResolvedSite | None" = None,
        ref_frames: np.ndarray | None = None,
    ):
        self.cfg = cfg
        self.folder = fold_engine
        self.ref_xyz = ref_xyz
        self.ref_seq = ref_seq
        # When a site is set, every fold additionally gets site geometry
        # measured against ref_frames, and -- if SITE_GATE is on -- has to
        # satisfy the site thresholds as well as the TM gate.
        self.site = site if (site and len(site)) else None
        self.ref_frames = ref_frames
        self.exempt: frozenset[int] = frozenset()
        if self.site is not None and cfg.site_residue_policy == "exempt":
            self.exempt = frozenset(self.site.indices)

    @property
    def site_active(self) -> bool:
        return self.site is not None and self.ref_frames is not None

    def measure(self, seq: str, result) -> "sitemod.SiteMetrics | None":
        """Site metrics for one fold, or None when no site is configured."""
        if not self.site_active:
            return None
        frames = getattr(result, "site_frames", None)
        if frames is None:
            return sitemod.SiteMetrics(n_positions=len(self.site))
        return sitemod.measure_site(self.site, self.ref_frames, frames, seq)

    def validate(
        self,
        allowed_aas,
        pairs: list[tuple[float, str]],
        n_folds: int,
        steps: int | None = None,
        native_seq: str | None = None,
        quiet: bool = False,
    ) -> Validation:
        """Fold up to *n_folds* selected designs and apply the acceptance gate.

        The gate is TM-score only, as in V9: TM is length-normalised so it
        means the same thing across a panel spanning 50 to 250 residues, and
        0.5 is the Zhang & Skolnick threshold for the same fold family.  RMSD
        is computed for every sequence and used to rank candidates that all
        clear the TM gate, but it is not a pass/fail criterion -- a fixed
        Angstrom cutoff across that length range is systematically biased
        against the longer domains.  pLDDT is logged always and gated only if
        ``USE_PLDDT_GATE`` is set, as a disorder filter rather than a quality
        threshold.
        """
        allowed = set(allowed_aas)
        usable = [
            (sc, sq)
            for sc, sq in pairs
            if sq and sq != native_seq and in_alphabet(sq, allowed, self.exempt)
        ]
        if not usable:
            if not quiet:
                print(
                    f"      no in-alphabet designs from MPNN "
                    f"({len(pairs)} raw sequences)"
                )
            return Validation()

        chosen = select_sequences(
            usable, n_folds, max_identity=self.cfg.max_identity_for_fold
        )
        if not chosen:
            return Validation()

        rmsds: list[float] = []
        tms: list[float] = []
        plddts: list[float] = []
        folded: list[tuple[str, float | None, float | None, float | None]] = []
        site_metrics: list["sitemod.SiteMetrics | None"] = []

        results = self.folder.fold_many(chosen, steps=steps)
        for seq, res in zip(chosen, results):
            if res is None or not res.ok:
                continue
            rmsd, tm = st.compare(self.ref_xyz, self.ref_seq, res.ca, seq)
            if rmsd is not None:
                rmsds.append(rmsd)
            if tm is not None:
                tms.append(tm)
            if res.plddt is not None:
                plddts.append(res.plddt)
            folded.append((seq, rmsd, tm, res.plddt))
            site_metrics.append(self.measure(seq, res))

        if not folded:
            return Validation()

        metrics = Metrics(
            mean_rmsd=statistics.mean(rmsds) if rmsds else None,
            min_rmsd=min(rmsds) if rmsds else None,
            mean_tm=statistics.mean(tms) if tms else None,
            max_tm=max(tms) if tms else None,
            mean_plddt=statistics.mean(plddts) if plddts else None,
            max_plddt=max(plddts) if plddts else None,
            successful_folds=len(folded),
            total_attempted=len(chosen),
            site_active=self.site_active,
        )

        if self.site_active:
            measured = [m for m in site_metrics if m is not None]
            bb = [m.bb_rmsd for m in measured if m.bb_rmsd is not None]
            loc = [m.local_rmsd for m in measured if m.local_rmsd is not None]
            cb = [m.cb_max_dev for m in measured if m.cb_max_dev is not None]
            # The best-preserving fold in the set, matching how max_tm and
            # min_rmsd already report the best rather than the mean.
            metrics.site_bb_rmsd = min(bb) if bb else None
            metrics.site_local_rmsd = min(loc) if loc else None
            metrics.site_cb_dev = min(cb) if cb else None

        passed = False
        site_ok_flags: list[bool | None] = []
        for idx, (_seq, _rmsd, tm, plddt) in enumerate(folded):
            ok_tm = tm is not None and tm >= self.cfg.min_tm_score
            ok_plddt = True
            if self.cfg.use_plddt_gate:
                ok_plddt = plddt is None or plddt >= self.cfg.min_plddt_disorder

            sm = site_metrics[idx] if idx < len(site_metrics) else None
            if sm is None:
                ok_site: bool | None = None
            else:
                ok_site = sm.passed(
                    self.cfg.site_max_bb_rmsd,
                    self.cfg.site_max_local_rmsd,
                    self.cfg.site_max_cb_dev,
                    require_identity=self.cfg.site_require_identity,
                )
                if ok_site:
                    metrics.num_site_pass += 1
            site_ok_flags.append(ok_site)

            # The site is a gate only when asked for; otherwise the numbers
            # are recorded and the fold gate alone decides, so a run can be
            # told "what would the site have done" without changing what the
            # search accepts.
            gate_site = True
            if self.site_active and self.cfg.site_gate:
                gate_site = bool(ok_site)

            if ok_tm and ok_plddt and gate_site:
                metrics.num_pass_all += 1
                passed = True

        ranked = sorted(
            range(len(folded)),
            key=lambda i: (
                float(bool(site_ok_flags[i])) if self.site_active else 0.0,
                folded[i][2] if folded[i][2] is not None else -999.0,
                -folded[i][1] if folded[i][1] is not None else -999.0,
            ),
            reverse=True,
        )
        pick = ranked[0]
        top = folded[pick]
        sm = site_metrics[pick] if pick < len(site_metrics) else None
        best = BestSequence(
            sequence=top[0], rmsd=top[1], tm=top[2], plddt=top[3],
            site_bb_rmsd=sm.bb_rmsd if sm else None,
            site_local_rmsd=sm.local_rmsd if sm else None,
            site_cb_dev=sm.cb_max_dev if sm else None,
            site_ok=site_ok_flags[pick],
        )
        return Validation(passed=passed, metrics=metrics, best=best, folded=folded)


# ---------------------------------------------------------------------------
# Backward pruning
# ---------------------------------------------------------------------------


def _fmt(value, spec="0.3f") -> str:
    return f"{value:{spec}}" if value is not None else "N/A"


def backward_prune(
    cfg,
    alphabet: list[str],
    trial_fn,
    protected: set[str] | None = None,
) -> tuple[list[str], int, int]:
    """Shrink *alphabet* as far as the fold allows.

    ``trial_fn(subset, label) -> (passed, Metrics)`` runs one MPNN + fold
    trial.  Returns ``(pruned_alphabet, n_removed, n_trials)``.

    Only chemically non-negotiable residues are protected -- cis-Pro's PRO and
    disulfide CYS.  A and G are not protected: if a fold can be encoded without
    them, the reported minimum should say so.  Protecting the seed, as a naive
    greedy pass does, inflates the answer.
    """
    protected = set(protected or ())
    pruned = list(alphabet)
    n_start = len(pruned)
    removed = 0
    trials = 0
    verdicts: dict[frozenset, tuple[bool, Metrics]] = {}

    def trial(subset: list[str], label: str) -> tuple[bool, Metrics]:
        nonlocal trials
        key = frozenset(subset)
        if key in verdicts:
            return verdicts[key]
        trials += 1
        result = trial_fn(sorted(subset), f"{label}")
        verdicts[key] = result
        return result

    exhaustive = n_start <= cfg.exhaustive_pruning_threshold
    print(
        f"\n  [Backward pruning] start {n_start} AAs ({''.join(sorted(pruned))}), "
        f"protected {sorted(protected) or 'none'}, "
        f"mode {'exhaustive' if exhaustive else 'greedy-best'}"
        + (" + downward closure" if exhaustive and cfg.pruning_downward_closure else "")
    )

    if exhaustive:
        pool = [a for a in pruned if a not in protected]
        best_found = list(pruned)
        # Level-wise descent.  passed_above holds the subsets that passed at
        # the size just above the one being tested.
        passed_above: set[frozenset] = {frozenset(pruned)}
        min_passing = n_start

        for target in range(n_start - 1, 1, -1):
            n_free = target - len(protected)
            if n_free < 0:
                break
            if n_free > len(pool):
                continue

            candidates: list[list[str]] = []
            skipped = 0
            for free in combinations(pool, n_free):
                subset = sorted(set(protected) | set(free))
                if cfg.pruning_downward_closure:
                    # Testable only if every one-larger superset passed; a
                    # failed superset implies this subset fails too.
                    supersets = [
                        frozenset(subset) | {a} for a in pool if a not in subset
                    ]
                    if any(sup not in passed_above for sup in supersets):
                        skipped += 1
                        continue
                candidates.append(subset)

            if not candidates:
                print(
                    f"    size {target}: no candidate survives downward closure "
                    f"({skipped} implied failures) -- minimum is {min_passing}"
                )
                break

            passing: list[tuple[list[str], Metrics]] = []
            for subset in candidates:
                label = f"ex{target}_" + "".join(a for a in subset if a not in protected)
                ok, metrics = trial(subset, label)
                if ok:
                    passing.append((subset, metrics))

            note = f" ({skipped} skipped by closure)" if skipped else ""
            print(
                f"    size {target}: tested {len(candidates)}{note}, "
                f"{len(passing)} passed"
            )

            if not passing:
                print(f"    -> minimum alphabet size is {min_passing}")
                break

            winner, met = max(passing, key=lambda x: x[1].rank_key())
            print(
                f"    -> best at size {target}: {''.join(winner)} "
                f"maxTM={_fmt(met.max_tm)} minRMSD={_fmt(met.min_rmsd, '0.2f')}"
            )
            best_found = winner
            min_passing = target
            passed_above = {frozenset(s) for s, _ in passing}

        pruned = best_found
        removed = n_start - len(pruned)

    else:
        def removable(alpha):
            return [a for a in alpha if a not in protected]

        changed = True
        while changed:
            changed = False

            single = True
            while single:
                single = False
                passing = []
                for aa in removable(pruned):
                    trial_set = [a for a in pruned if a != aa]
                    if len(trial_set) < 2:
                        continue
                    ok, metrics = trial(trial_set, f"rm{aa}")
                    if ok:
                        passing.append((aa, trial_set, metrics))
                if passing:
                    aa, trial_set, met = max(passing, key=lambda x: x[2].rank_key())
                    print(
                        f"  [single] removed {aa} -> {''.join(sorted(trial_set))} "
                        f"maxTM={_fmt(met.max_tm)} "
                        f"minRMSD={_fmt(met.min_rmsd, '0.2f')} "
                        f"({len(passing)} passing)"
                    )
                    pruned = trial_set
                    removed += 1
                    single = changed = True

            cands = removable(pruned)
            if len(cands) < 2:
                break
            passing_pairs = []
            for aa1, aa2 in combinations(cands, 2):
                trial_set = [a for a in pruned if a not in (aa1, aa2)]
                if len(trial_set) < 2:
                    continue
                ok, metrics = trial(trial_set, f"rm{aa1}{aa2}")
                if ok:
                    passing_pairs.append((aa1, aa2, trial_set, metrics))
            if not passing_pairs:
                break
            aa1, aa2, trial_set, met = max(
                passing_pairs, key=lambda x: x[3].rank_key()
            )
            print(
                f"  [pair]   removed {aa1}+{aa2} -> {''.join(sorted(trial_set))} "
                f"maxTM={_fmt(met.max_tm)} minRMSD={_fmt(met.min_rmsd, '0.2f')} "
                f"({len(passing_pairs)} passing)"
            )
            pruned = trial_set
            removed += 2
            changed = True

    print(
        f"  [Backward pruning] final {len(pruned)} AAs "
        f"({''.join(sorted(pruned))}), removed {removed}, {trials} trials"
    )
    return sorted(pruned), removed, trials
