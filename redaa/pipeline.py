"""Batch runner: per-domain and per-replicate orchestration.

Output layout and CSV columns are unchanged from V9 so existing analysis
scripts keep working:

    <output_root>/<run_name>/
        BATCH_SUMMARY.csv
        FLAGGED_DOMAINS.csv
        ALL_BEST_SEQUENCES.fasta
        fold_cache.sqlite
        <domain>/
            domain_info.txt
            native_esmfold.pdb
            <domain>_best_sequence.fasta
            run_<rep>/
                COMPLETED
                alphabet_building_log.csv
                esmfold_validation_log.csv
"""

from __future__ import annotations

import csv
import glob
import os
import signal
import time
from collections import Counter, defaultdict
from datetime import datetime

from . import structure as st
from .chemistry import analyze_requirements
from .config import ALL_AAS
from .foldcache import FoldCache
from .folding import FoldEngine
from .mpnn import MPNNEngine
from .search import Metrics, Validator, backward_prune

DONE_SENTINEL = "COMPLETED"


def _fmt(value, spec="0.3f") -> str:
    return f"{value:{spec}}" if value is not None else "N/A"


class StopRequested(Exception):
    """Raised to unwind cleanly when the job is out of time or preempted."""


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.started = time.time()
        self.stop_requested = False
        self.run_dir = ""
        self.results: list[dict] = []
        self.flagged: list[dict] = []
        self.folder: FoldEngine | None = None
        self.mpnn: MPNNEngine | None = None
        self.cache: FoldCache | None = None

    # -- lifecycle --------------------------------------------------------
    def install_signal_handlers(self) -> None:
        """Turn preemption into a clean stop rather than a lost checkpoint.

        SLURM sends SIGTERM (and optionally SIGUSR1 with ``--signal``) before
        killing a preempted job.  Catching them lets the current replicate
        finish its bookkeeping and the summaries get written.
        """

        def handler(signum, _frame):
            if not self.stop_requested:
                print(
                    f"\n*** signal {signum} received -- finishing the current "
                    "replicate, then checkpointing and exiting ***\n",
                    flush=True,
                )
            self.stop_requested = True

        for sig in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGUSR1", None)):
            if sig is not None:
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass

    def out_of_time(self) -> bool:
        budget = self.cfg.max_wall_seconds
        return bool(budget) and (time.time() - self.started) > budget

    def check_stop(self) -> None:
        if self.stop_requested:
            raise StopRequested("signal received")
        if self.out_of_time():
            raise StopRequested(
                f"wall-clock budget of {self.cfg.max_wall_seconds} s exhausted"
            )

    # -- discovery --------------------------------------------------------
    def discover(self) -> list[tuple[str, str]]:
        paths = sorted(glob.glob(os.path.join(self.cfg.input_pdb_dir, "*.pdb")))
        if not paths:
            raise SystemExit(
                f"ERROR: no .pdb files in {self.cfg.input_pdb_dir}\n"
                "Set INPUT_PDB_DIR to a directory of pre-processed, "
                "single-chain, standard-residue domain PDBs."
            )
        return [(os.path.splitext(os.path.basename(p))[0], p) for p in paths]

    def resolve_run_dir(self) -> str:
        cfg = self.cfg
        stem = cfg.run_name or f"batch_chunk{cfg.chunk_index}"
        if cfg.resume_from and os.path.isdir(cfg.resume_from):
            print(f"Resuming from: {cfg.resume_from}")
            return cfg.resume_from
        if cfg.auto_resume:
            existing = sorted(
                glob.glob(os.path.join(cfg.output_root, f"{stem}_*"))
            )
            existing = [p for p in existing if os.path.isdir(p)]
            if existing:
                print(f"Auto-resuming from: {existing[-1]}")
                return existing[-1]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fresh = os.path.join(cfg.output_root, f"{stem}_{stamp}")
        print(f"Starting fresh: {fresh}")
        return fresh

    # -- engines ----------------------------------------------------------
    def setup(self) -> None:
        cfg = self.cfg
        self.run_dir = self.resolve_run_dir()
        os.makedirs(self.run_dir, exist_ok=True)

        cache_path = cfg.fold_cache_path or os.path.join(
            self.run_dir, "fold_cache.sqlite"
        )
        self.cache = FoldCache(cache_path, enabled=cfg.use_fold_cache)
        if self.cache.enabled:
            print(f"Fold cache: {cache_path}")

        self.folder = FoldEngine(cfg, self.cache)
        self.folder.load()

        self.mpnn = MPNNEngine(cfg)
        self.mpnn.load()

        if not st.have_tmtools():
            print(
                "  NOTE: tmtools not installed -- TM-scores use the built-in "
                "seed-refined implementation.\n"
                "        `pip install tmtools` for exact TM-align and a "
                "faster metrics path."
            )

    def teardown(self) -> None:
        if self.folder is not None:
            print(f"\nESMFold2: {self.folder.stats()}")
        if self.mpnn is not None:
            print(f"LigandMPNN: {self.mpnn.stats()}")
        if self.cache is not None:
            self.cache.close()

    # -- checkpointing ----------------------------------------------------
    @staticmethod
    def rep_done(domain_dir: str, rep: int) -> bool:
        return os.path.exists(
            os.path.join(domain_dir, f"run_{rep}", DONE_SENTINEL)
        )

    @staticmethod
    def mark_rep_done(domain_dir: str, rep: int) -> None:
        path = os.path.join(domain_dir, f"run_{rep}", DONE_SENTINEL)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(datetime.now().isoformat())
        os.replace(tmp, path)  # atomic, so a kill mid-write cannot half-mark it

    # ------------------------------------------------------------------
    # One replicate
    # ------------------------------------------------------------------
    def run_replicate(
        self,
        pdb_path: str,
        domain: str,
        rep: int,
        domain_dir: str,
        start_alphabet: list[str],
        protected: set[str],
        validator: Validator,
        native_seq: str | None,
    ) -> dict:
        cfg = self.cfg
        out_root = os.path.join(domain_dir, f"run_{rep}")
        os.makedirs(out_root, exist_ok=True)
        base_seed = rep * 10_000
        t0 = datetime.now()

        build_path = os.path.join(out_root, "alphabet_building_log.csv")
        val_path = os.path.join(out_root, "esmfold_validation_log.csv")

        with open(build_path, "w", newline="") as bf, open(
            val_path, "w", newline=""
        ) as vf:
            build = csv.writer(bf)
            val = csv.writer(vf)
            build.writerow(
                [
                    "Stage", "Round", "Candidate_AA", "MPNN_Score",
                    "Current_Set", "Improvement_%", "Status",
                ]
            )
            val.writerow(
                [
                    "Stage", "Round", "Alphabet",
                    "Mean_RMSD", "Min_RMSD", "Mean_TM", "Max_TM",
                    "Mean_pLDDT", "Max_pLDDT", "Num_Pass_All",
                    "Successful_Folds", "Total_Attempted", "Validation_Status",
                ]
            )

            def flush():
                bf.flush()
                vf.flush()

            def log_val(stage, rnd, alphabet, metrics: Metrics, passed: bool):
                val.writerow(
                    [
                        stage, rnd, alphabet,
                        _fmt(metrics.mean_rmsd, "0.2f"),
                        _fmt(metrics.min_rmsd, "0.2f"),
                        _fmt(metrics.mean_tm),
                        _fmt(metrics.max_tm),
                        _fmt(metrics.mean_plddt, "0.1f"),
                        _fmt(metrics.max_plddt, "0.1f"),
                        metrics.num_pass_all,
                        metrics.successful_folds,
                        metrics.total_attempted,
                        "PASSED" if passed else "FAILED",
                    ]
                )

            best = {"sequence": None, "alphabet": None, "rmsd": None,
                    "tm": None, "plddt": None}

            def maybe_best(candidate, alphabet: str):
                if candidate.sequence is None:
                    return
                key = candidate.rank_key()
                current = (
                    best["tm"] if best["tm"] is not None else 0.0,
                    -(best["rmsd"] if best["rmsd"] is not None else 999.0),
                )
                if best["sequence"] is None or key > current:
                    best.update(
                        sequence=candidate.sequence,
                        alphabet=alphabet,
                        rmsd=candidate.rmsd,
                        tm=candidate.tm,
                        plddt=candidate.plddt,
                    )

            # -- 1. 20-residue baseline -------------------------------------
            # Always at the coldest temperature, independent of the restricted
            # runs, so it is the strongest possible reference score.
            baseline = self.mpnn.design(
                pdb_path, list(ALL_AAS),
                temperature=cfg.mpnn_baseline_temp, seed=base_seed,
            )
            target_score = baseline.mean_score * cfg.baseline_tolerance
            build.writerow(
                ["BASELINE", 0, "ALL20", baseline.mean_score, "ALL20", 0, "BASELINE"]
            )

            # -- 2. starting alphabet --------------------------------------
            if cfg.dynamic_seeding:
                counts: Counter = Counter()
                for _sc, seq in baseline.pairs:
                    counts.update(c for c in seq if c in set(ALL_AAS))
                top = counts.most_common()
                current = sorted(aa for aa, _ in top[: cfg.dynamic_seed_size])
                comp = " ".join(f"{aa}:{n}" for aa, n in top[:10])
                print(f"  [dynamic seed] composition: {comp}")
                print(f"  [dynamic seed] seeding with {current}")
                designs = self.mpnn.design(pdb_path, current, seed=base_seed + 1)
                last_score = designs.mean_score
                build.writerow(
                    ["DYNAMIC_SEED", 0, "".join(current), last_score,
                     "".join(current), 0, f"SEEDED_FROM_COMPOSITION({comp})"]
                )
            else:
                current = list(start_alphabet)
                designs = self.mpnn.design(pdb_path, current, seed=base_seed + 1)
                last_score = designs.mean_score
                build.writerow(
                    ["INIT", 0, "".join(sorted(current)), last_score,
                     "".join(sorted(current)), 0, "INITIAL"]
                )

                if cfg.use_hydrophobic_seeding:
                    best_aa, best_sc = None, float("inf")
                    for i, aa in enumerate(("V", "I", "L")):
                        if aa in current:
                            continue
                        d = self.mpnn.design(
                            pdb_path, current + [aa], seed=base_seed + 2 + i
                        )
                        build.writerow(
                            ["HYDROPHOBIC", 0, aa, d.mean_score,
                             "".join(sorted(current)), 0, "TRIAL"]
                        )
                        if d.mean_score < best_sc:
                            best_sc, best_aa = d.mean_score, aa
                    if best_aa is not None:
                        imp = (
                            (last_score - best_sc) / last_score
                            if last_score > 0 else 0.0
                        )
                        build.writerow(
                            ["HYDROPHOBIC", 0, best_aa, best_sc,
                             "".join(sorted(current)), imp * 100, "ACCEPTED"]
                        )
                        current.append(best_aa)
                        last_score = best_sc
            flush()

            # -- 3. forward greedy rounds ----------------------------------
            rounds = 0
            while len(current) < 20:
                self.check_stop()
                rounds += 1
                remaining = [aa for aa in ALL_AAS if aa not in current]
                if not remaining:
                    break

                if last_score <= target_score and best["sequence"] is not None:
                    build.writerow(
                        ["OPEN", rounds, "N/A", last_score,
                         "".join(sorted(current)), 0, "STOPPED_TARGET_REACHED"]
                    )
                    flush()
                    break

                # MPNN is now cheap enough to run for every candidate.
                candidates = []
                for idx, aa in enumerate(remaining):
                    d = self.mpnn.design(
                        pdb_path, current + [aa],
                        seed=base_seed + 100 + rounds * 20 + idx,
                    )
                    build.writerow(
                        ["OPEN", rounds, aa, d.mean_score,
                         "".join(sorted(current)), 0, "TRIAL"]
                    )
                    candidates.append((aa, d))
                flush()

                do_validate = (len(current) + 1) >= cfg.esm_start_after_n_aas
                if not do_validate:
                    # Too small for ESMFold to say anything useful: AG-only
                    # compositions are far outside its training distribution,
                    # so take the best MPNN score and grow.
                    aa, d = min(candidates, key=lambda c: c[1].mean_score)
                    imp = (
                        (last_score - d.mean_score) / last_score
                        if last_score > 0 else 0.0
                    )
                    current.append(aa)
                    last_score = d.mean_score
                    build.writerow(
                        ["OPEN", rounds, aa, d.mean_score,
                         "".join(sorted(current)), imp * 100, "ACCEPTED_PRE_ESM"]
                    )
                    flush()
                    print(f"  [round {rounds}] pre-ESM growth: +{aa}")
                    continue

                print(
                    f"\n  [round {rounds}] screening {len(candidates)} candidates "
                    f"({cfg.screen_folds_stage1} fold each, then "
                    f"{cfg.screen_folds_stage2} for the top "
                    f"{cfg.screen_stage2_keep})"
                )

                # Stage 1: one fold per candidate.
                stage1 = []
                for aa, d in candidates:
                    v = validator.validate(
                        current + [aa], d.pairs, cfg.screen_folds_stage1,
                        steps=cfg.fold_steps_screen, native_seq=native_seq,
                        quiet=True,
                    )
                    stage1.append((aa, d, v))

                # Stage 2: everything that already passed, plus the best by TM.
                ranked = sorted(
                    stage1, key=lambda x: x[2].metrics.rank_key(), reverse=True
                )
                promote = {aa for aa, _, v in stage1 if v.passed}
                for aa, _, _ in ranked[: cfg.screen_stage2_keep]:
                    promote.add(aa)

                evaluated = []
                for aa, d, v in stage1:
                    if aa in promote and cfg.screen_folds_stage2 > cfg.screen_folds_stage1:
                        # Nested selection means stage 1's fold is a cache hit,
                        # so this costs only the additional folds.
                        v = validator.validate(
                            current + [aa], d.pairs, cfg.screen_folds_stage2,
                            steps=cfg.fold_steps_screen, native_seq=native_seq,
                            quiet=True,
                        )
                    alphabet = "".join(sorted(current + [aa]))
                    log_val("OPEN", rounds, alphabet, v.metrics, v.passed)
                    print(
                        f"    +{aa} ({alphabet}): "
                        f"minRMSD={_fmt(v.metrics.min_rmsd, '0.2f')} "
                        f"maxTM={_fmt(v.metrics.max_tm)} "
                        f"maxpLDDT={_fmt(v.metrics.max_plddt, '0.1f')} "
                        f"{'PASS' if v.passed else 'fail'}"
                        + ("" if aa in promote else "  [screened out]")
                    )
                    evaluated.append((aa, d, v))
                flush()

                passing = [(aa, d, v) for aa, d, v in evaluated if v.passed]
                if passing:
                    aa, d, v = max(
                        passing, key=lambda x: x[2].metrics.rank_key()
                    )
                    alphabet = "".join(sorted(current + [aa]))
                    maybe_best(v.best, alphabet)
                    imp = (
                        (last_score - d.mean_score) / last_score
                        if last_score > 0 else 0.0
                    )
                    current.append(aa)
                    last_score = d.mean_score
                    build.writerow(
                        ["OPEN", rounds, aa, d.mean_score,
                         "".join(sorted(current)), imp * 100,
                         "ACCEPTED_STOP_VALIDATED"]
                    )
                    flush()
                    print(
                        f"  PASSED at {len(current)} AAs (+{aa}, "
                        f"TM={_fmt(v.metrics.max_tm)}, "
                        f"RMSD={_fmt(v.metrics.min_rmsd, '0.2f')} A)"
                    )
                    break

                structured = [
                    (aa, d, v) for aa, d, v in evaluated
                    if v.metrics.max_tm is not None
                ]
                if structured:
                    aa, d, v = max(
                        structured, key=lambda x: x[2].metrics.rank_key()
                    )
                    reason = "ACCEPTED_BEST_TM"
                else:
                    aa, d, _v = min(evaluated, key=lambda x: x[1].mean_score)
                    reason = "ACCEPTED"
                imp = (
                    (last_score - d.mean_score) / last_score
                    if last_score > 0 else 0.0
                )
                current.append(aa)
                last_score = d.mean_score
                build.writerow(
                    ["OPEN", rounds, aa, d.mean_score,
                     "".join(sorted(current)), imp * 100, reason]
                )
                flush()
                print(f"  no candidate passed -- added +{aa} ({reason})")

            # -- 4. full validation at full sampler quality ----------------
            alphabet = "".join(sorted(current))
            print(
                f"\n  [full validation] {len(current)}-AA alphabet ({alphabet}), "
                f"{cfg.num_seqs_to_fold} folds at {cfg.fold_steps_final} steps"
            )
            full_designs = self.mpnn.design(
                pdb_path, current, seed=base_seed + 50_000
            )
            full = validator.validate(
                current, full_designs.pairs, cfg.num_seqs_to_fold,
                steps=cfg.fold_steps_final, native_seq=native_seq,
            )
            log_val("FULL_VALIDATION", rounds, alphabet, full.metrics, full.passed)
            maybe_best(full.best, alphabet)
            flush()
            print(
                f"  full validation: minRMSD={_fmt(full.metrics.min_rmsd, '0.2f')} A "
                f"maxTM={_fmt(full.metrics.max_tm)} "
                f"maxpLDDT={_fmt(full.metrics.max_plddt, '0.1f')} "
                f"{'PASSED' if full.passed else 'FAILED'}"
            )

            # -- 5. backward pruning ---------------------------------------
            pruned, removed, n_trials = current, 0, 0
            if (
                cfg.do_backward_pruning
                and best["sequence"] is not None
                and len(current) > cfg.pruning_min_size
            ):
                counter = {"n": 0}

                def trial_fn(subset, label):
                    self.check_stop()
                    counter["n"] += 1
                    d = self.mpnn.design(
                        pdb_path, subset, seed=base_seed + 9000 + counter["n"]
                    )
                    v = validator.validate(
                        subset, d.pairs, cfg.prune_folds,
                        steps=cfg.fold_steps_screen, native_seq=native_seq,
                        quiet=True,
                    )
                    log_val(
                        "PRUNING", rounds, "".join(subset), v.metrics, v.passed
                    )
                    if v.passed:
                        maybe_best(v.best, "".join(subset))
                    return v.passed, v.metrics

                pruned, removed, n_trials = backward_prune(
                    cfg, current, trial_fn, protected=protected
                )
                build.writerow(
                    ["PRUNING", rounds, "BACKWARD_PASS", last_score,
                     "".join(sorted(pruned)), removed, f"PRUNED_{n_trials}_TRIALS"]
                )
            flush()

        elapsed = round((datetime.now() - t0).total_seconds())
        return {
            "pdb": domain,
            "replicate": rep,
            "final_alphabet": "".join(sorted(current)),
            "pruned_alphabet": "".join(sorted(pruned)),
            "alphabet_size": len(current),
            "pruned_alphabet_size": len(pruned),
            "aas_removed_pruning": removed,
            "pruning_trials": n_trials,
            "final_mpnn_score": last_score,
            "baseline_mpnn_score": baseline.mean_score,
            "num_rounds": rounds,
            "elapsed_seconds": elapsed,
            "best_sequence": best["sequence"],
            "best_seq_alphabet": best["alphabet"],
            "best_seq_rmsd": best["rmsd"],
            "best_seq_tm": best["tm"],
            "best_seq_plddt": best["plddt"],
        }

    # ------------------------------------------------------------------
    # One domain
    # ------------------------------------------------------------------
    def run_domain(self, domain: str, pdb_path: str, index: str) -> None:
        cfg = self.cfg
        print("\n" + "=" * 70)
        print(f"Domain {index}: {domain}")
        print(f"  file: {pdb_path}")
        print("=" * 70)

        if not os.path.isfile(pdb_path):
            print(f"  ERROR: file not found, skipping")
            self.results.append(
                {"pdb": domain, "replicate": 0, "status": "skipped_file_not_found"}
            )
            return

        domain_dir = os.path.join(self.run_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        source_path = pdb_path

        # Filename length tells us the input type, as in V9: a CATH domain stem
        # such as "1bxoA02" is already single-chain, whereas a bare four-letter
        # RCSB ID needs chain A materialised first.
        if len(domain) > 4:
            input_type = "pre-processed domain (used as-is)"
        else:
            chain_path = os.path.join(domain_dir, f"{domain}_chainA.pdb")
            n_res = st.extract_chain(pdb_path, "A", chain_path)
            if n_res == 0:
                print("  ERROR: chain A absent or has no standard residues, skipping")
                self.results.append(
                    {"pdb": domain, "replicate": 0, "status": "skipped_no_chain_A"}
                )
                return
            pdb_path = chain_path
            input_type = f"RCSB entry, chain A only ({n_res} residues)"
        print(f"  input type: {input_type}")

        ref_xyz, ref_seq = st.parse_ca_from_file(pdb_path)
        if len(ref_xyz) < 15:
            print(f"  ERROR: only {len(ref_xyz)} CA atoms parsed, skipping")
            self.results.append(
                {"pdb": domain, "replicate": 0, "status": "skipped_too_short"}
            )
            return
        native_seq = ref_seq

        info_path = os.path.join(domain_dir, "domain_info.txt")
        if not os.path.exists(info_path):
            with open(info_path, "w") as fh:
                fh.write(f"Domain: {domain}\n")
                fh.write(f"Original source file: {source_path}\n")
                fh.write(f"Input handling: {input_type}\n")
                fh.write(f"PDB used by pipeline: {pdb_path}\n")
                fh.write(f"Sequence length: {len(native_seq)} aa\n")
                fh.write(f"Sequence: {native_seq}\n")

        # -- chemical requirements ----------------------------------------
        req = analyze_requirements(pdb_path)
        protected = set(req.required_aas)
        if cfg.dynamic_seeding:
            start_alphabet: list[str] = []
        else:
            start_alphabet = sorted(set(cfg.starting_alphabet) | protected)

        print("\n  [structure analysis]")
        print(f"    disulfides : {req.n_disulfides}")
        print(f"    cis-Pro    : {req.cis_pro_count}")
        for note in req.notes:
            print(f"    -> {note}")
        print(f"    starting alphabet: {start_alphabet or '(dynamic)'}")
        with open(info_path, "a") as fh:
            fh.write("\nStructural requirements:\n")
            fh.write(f"  Disulfides: {req.n_disulfides}\n")
            fh.write(f"  Cis-Pro residues: {req.cis_pro_count}\n")
            for note in req.notes:
                fh.write(f"  {note}\n")
            fh.write(f"  Starting alphabet: {''.join(start_alphabet)}\n")

        # -- native round-trip sanity check -------------------------------
        print("\n  [sanity check] folding the native sequence...")
        native = self.folder.fold(native_seq, steps=cfg.fold_steps_final)
        if native is None or not native.ok:
            print("  SKIPPING: ESMFold2 produced no model for the native sequence")
            self.results.append(
                {"pdb": domain, "replicate": 0,
                 "status": "skipped_native_fold_failed"}
            )
            return

        nat_rmsd, nat_tm = st.compare(ref_xyz, ref_seq, native.ca, native_seq)
        print(
            f"  native fold: pLDDT={_fmt(native.plddt, '0.1f')} "
            f"RMSD(vs reference)={_fmt(nat_rmsd, '0.2f')} A "
            f"TM(vs reference)={_fmt(nat_tm)}"
        )
        print(
            "  (the crystal/domain PDB is the reference for all folds; the "
            "native fold above is only an ESMFold2 reliability check)"
        )
        # Keep the native prediction on disk for inspection.  Skipped when it
        # already exists, because a resumed run gets the native fold from the
        # cache, which carries coordinates but not the original PDB text --
        # writing it would mean paying for the fold again.
        native_pdb = os.path.join(domain_dir, "native_esmfold.pdb")
        if not os.path.exists(native_pdb):
            self.folder.write_pdb(native, native_pdb)

        domain_flag = ""
        reasons = []
        if cfg.use_plddt_gate and native.plddt is not None:
            if native.plddt < cfg.native_plddt_min:
                reasons.append(f"pLDDT={native.plddt:.1f} < {cfg.native_plddt_min}")
        if nat_tm is not None and nat_tm < cfg.native_tm_min:
            reasons.append(f"TM={nat_tm:.3f} < {cfg.native_tm_min}")

        if reasons:
            # Flag but still run: an IDP, NMR ensemble, multidomain crystal or
            # out-of-distribution fold is still worth searching; the results
            # just carry a warning.
            domain_flag = "; ".join(reasons)
            print(f"  FLAGGED: native round-trip is weak ({domain_flag})")
            print(
                "    likely an IDP, NMR ensemble, multidomain crystal or "
                "out-of-distribution fold -- running anyway, results flagged"
            )
            with open(info_path, "a") as fh:
                fh.write(f"\nFLAGGED: native ESMFold2 round-trip weak -- {domain_flag}\n")
            with open(
                os.path.join(domain_dir, "FLAGGED_native_esmfold_mismatch.txt"), "w"
            ) as fh:
                fh.write(f"Domain: {domain}\n")
                fh.write(f"Native ESMFold2 vs reference: {domain_flag}\n")
                fh.write(f"Native TM: {_fmt(nat_tm)}\n")
                fh.write(
                    "The full alphabet search WAS run. Because ESMFold2 does not "
                    "reproduce the native fold here, pass/fail of any "
                    "reduced-alphabet sequence may be unreliable -- inspect "
                    "manually.\n"
                )
            self.flagged.append(
                {"pdb": domain, "native_tm": nat_tm, "reason": domain_flag}
            )
        else:
            print("  ESMFold2 reproduces the native fold -- reference accepted")

        validator = Validator(cfg, self.folder, ref_xyz, ref_seq)

        # -- replicates ---------------------------------------------------
        try:
            for rep in range(1, cfg.num_replicates + 1):
                if self.rep_done(domain_dir, rep):
                    print(f"\n  --- replicate {rep}/{cfg.num_replicates} "
                          f"[skipped, checkpoint present] ---")
                    self.results.append(
                        {"pdb": domain, "replicate": rep, "status": "resumed",
                         "esmfold_native_flag": domain_flag,
                         "esmfold_native_tm": nat_tm}
                    )
                    continue

                self.check_stop()
                print(f"\n  --- replicate {rep}/{cfg.num_replicates} ---")
                result = self.run_replicate(
                    pdb_path, domain, rep, domain_dir,
                    start_alphabet, protected, validator, native_seq,
                )
                result["esmfold_native_flag"] = domain_flag
                result["esmfold_native_tm"] = nat_tm
                if domain_flag:
                    result["status"] = (
                        f"flagged_esmfold_native_mismatch: {domain_flag}"
                    )
                self.results.append(result)
                self.mark_rep_done(domain_dir, rep)

                summary = (
                    f"  done in {result['elapsed_seconds']} s | forward "
                    f"{result['final_alphabet']} "
                    f"({result['alphabet_size']} AAs)"
                )
                if result["aas_removed_pruning"]:
                    summary += (
                        f" -> pruned {result['pruned_alphabet']} "
                        f"({result['pruned_alphabet_size']} AAs)"
                    )
                print(summary)
        finally:
            # The next domain has a different backbone, so these features are
            # dead weight in VRAM.
            self.mpnn.release(pdb_path)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    SUMMARY_FIELDS = [
        "pdb", "replicate", "status",
        "esmfold_native_flag", "esmfold_native_tm",
        "final_alphabet", "pruned_alphabet",
        "alphabet_size", "pruned_alphabet_size", "aas_removed_pruning",
        "pruning_trials",
        "final_mpnn_score", "baseline_mpnn_score", "num_rounds",
        "elapsed_seconds",
        "best_seq_alphabet", "best_seq_rmsd", "best_seq_tm", "best_seq_plddt",
        "best_sequence",
    ]

    def write_summaries(self) -> tuple[str, str]:
        summary_path = os.path.join(self.run_dir, "BATCH_SUMMARY.csv")
        with open(summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=self.SUMMARY_FIELDS, extrasaction="ignore"
            )
            writer.writeheader()
            for row in self.results:
                writer.writerow({k: row.get(k, "") for k in self.SUMMARY_FIELDS})

        if self.flagged:
            flagged_path = os.path.join(self.run_dir, "FLAGGED_DOMAINS.csv")
            with open(flagged_path, "w", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["pdb", "native_tm", "reason"]
                )
                writer.writeheader()
                writer.writerows(self.flagged)
            print(
                f"\n{len(self.flagged)} domain(s) flagged for a weak native "
                f"round-trip: {[d['pdb'] for d in self.flagged]}"
            )
            print(f"  see {flagged_path}")

        by_domain: dict[str, list[dict]] = defaultdict(list)
        for row in self.results:
            if row.get("best_sequence"):
                by_domain[row["pdb"]].append(row)

        combined: list[str] = []
        for domain, reps in by_domain.items():
            champion = max(
                reps,
                key=lambda r: (
                    r.get("best_seq_tm") or 0.0,
                    -(r.get("best_seq_rmsd") or 999.0),
                    r.get("best_seq_plddt") or 0.0,
                ),
            )
            seq = champion["best_sequence"]
            header = (
                f">{domain}_rep{champion['replicate']} "
                f"alphabet={champion.get('best_seq_alphabet', 'unknown')} "
                f"RMSD={_fmt(champion.get('best_seq_rmsd'), '0.2f')} "
                f"TM={_fmt(champion.get('best_seq_tm'))} "
                f"pLDDT={_fmt(champion.get('best_seq_plddt'), '0.1f')}"
            )
            if champion.get("esmfold_native_flag"):
                header += (
                    f" FLAGGED_native_esmfold_mismatch="
                    f"[{champion['esmfold_native_flag']}]"
                )
            wrapped = [seq[i : i + 80] for i in range(0, len(seq), 80)]
            per_domain = os.path.join(
                self.run_dir, domain, f"{domain}_best_sequence.fasta"
            )
            os.makedirs(os.path.dirname(per_domain), exist_ok=True)
            with open(per_domain, "w") as fh:
                fh.write(header + "\n" + "\n".join(wrapped) + "\n")
            combined += [header, *wrapped, ""]

        fasta_path = os.path.join(self.run_dir, "ALL_BEST_SEQUENCES.fasta")
        with open(fasta_path, "w") as fh:
            fh.write("\n".join(combined))
        return summary_path, fasta_path

    # ------------------------------------------------------------------
    def run(self) -> int:
        cfg = self.cfg
        self.install_signal_handlers()

        all_domains = self.discover()
        print(f"Discovered {len(all_domains)} PDB files in {cfg.input_pdb_dir}")

        if cfg.test_mode:
            all_domains = all_domains[:1]
        start = cfg.chunk_index * cfg.chunk_size
        chunk = all_domains[start : start + cfg.chunk_size]
        if not chunk:
            raise SystemExit(
                f"ERROR: CHUNK_INDEX={cfg.chunk_index} is out of range "
                f"({len(all_domains)} domains, CHUNK_SIZE={cfg.chunk_size})."
            )

        self.setup()

        print("\n" + "=" * 70)
        print("RedAA finder -- ESMFold2-Fast + LigandMPNN")
        print("=" * 70)
        print(f"  chunk {cfg.chunk_index}: domains "
              f"{start + 1}-{start + len(chunk)} of {len(all_domains)}")
        print(f"  this chunk : {[name for name, _ in chunk]}")
        print(f"  replicates : {cfg.num_replicates}")
        print(f"  gate       : TM >= {cfg.min_tm_score}"
              + (f", pLDDT >= {cfg.min_plddt_disorder}" if cfg.use_plddt_gate
                 else " (pLDDT logged, not gated)"))
        print(f"  output     : {self.run_dir}")
        if cfg.max_wall_seconds:
            print(f"  wall budget: {cfg.max_wall_seconds} s")

        stopped = ""
        try:
            for i, (domain, path) in enumerate(chunk):
                self.check_stop()
                self.run_domain(
                    domain, path,
                    f"{i + 1}/{len(chunk)} (global {start + i + 1}/{len(all_domains)})",
                )
        except StopRequested as exc:
            stopped = str(exc)
            print(f"\n*** stopping early: {stopped} ***")
        finally:
            summary_path, fasta_path = self.write_summaries()
            self.teardown()

        completed = len([r for r in self.results if "elapsed_seconds" in r])
        print("\n" + "=" * 70)
        print(f"CHUNK {cfg.chunk_index} "
              + ("STOPPED EARLY" if stopped else "COMPLETE"))
        print("=" * 70)
        print(f"  domains in chunk : {len(chunk)}")
        print(f"  replicates run   : {completed}")
        print(f"  output directory : {self.run_dir}")
        print(f"  summary CSV      : {summary_path}")
        print(f"  combined FASTA   : {fasta_path}")
        print(f"  elapsed          : {(time.time() - self.started) / 60:.1f} min")
        if stopped:
            print(
                "\n  Completed replicates are checkpointed. Re-run the same "
                "command to resume;\n  finished replicates will be skipped."
            )
            return 2
        print(f"\n  next chunk: CHUNK_INDEX={cfg.chunk_index + 1}")
        return 0
