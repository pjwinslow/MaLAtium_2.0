#!/usr/bin/env python3
"""RedAA finder -- minimal amino-acid alphabet search (ESMFold2-Fast + LigandMPNN).

Finds the smallest amino-acid alphabet from which LigandMPNN can design
sequences that ESMFold2-Fast folds back onto the reference crystal structure.

Acceptance is on TM-score alone (>= MIN_TM_SCORE, default 0.51 -- the
Zhang & Skolnick same-fold-family threshold), because TM is length-normalised
and so means the same thing across a panel spanning 50 to 250 residues. RMSD is
computed for every sequence and used to rank candidates that all clear the TM
gate, but a fixed Angstrom cutoff over that length range is systematically
biased against longer domains, so it is not a gate. pLDDT is always logged and
gated only if USE_PLDDT_GATE is set, as a disorder filter rather than a
quality threshold.

Usage
-----
    python run_redaa.py --test            # single-domain smoke test
    python run_redaa.py                   # run the configured chunk
    python run_redaa.py --print-config    # show resolved settings and exit
    python run_redaa.py --self-test       # full unit-test suite, no GPU needed

Everything is configured by environment variable; see README.md, or run
--print-config for the resolved values.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from redaa.config import RedAAConfig  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="smoke test: first domain, 1 replicate, reduced budgets (~5 min)",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the resolved configuration and exit",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the full unit-test suite (no GPU or model weights needed)",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="override INPUT_PDB_DIR",
    )
    parser.add_argument(
        "--fetch",
        metavar="SPECS",
        default=None,
        help="fetch and prepare inputs before running, e.g. "
        "--fetch '1ubq_A, 2gb1, cath:1cukA01, af:P0DTC2'. Accepts PDB IDs "
        "(optionally _CHAIN and :FIRST-LAST), cath:/af:/pdbe: prefixes, URLs "
        "and local paths. Written to --input-dir.",
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="with --fetch, prepare the inputs and exit without searching",
    )
    parser.add_argument(
        "--chain",
        default=None,
        help="with --fetch, chain to use for any spec that does not name one",
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=None,
        help="override CHUNK_INDEX (defaults to SLURM_ARRAY_TASK_ID if set)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None, help="override CHUNK_SIZE"
    )
    parser.add_argument(
        "--replicates", type=int, default=None, help="override NUM_REPLICATES"
    )
    args = parser.parse_args(argv)

    if args.self_test:
        from tests import (
            test_colab,
            test_fetch,
            test_notebook,
            test_pipeline,
            test_search,
            test_structure,
        )

        failed = 0
        for label, module in (
            ("structural metrics", test_structure),
            ("search and chemistry", test_search),
            ("fetching and input prep", test_fetch),
            ("colab helpers", test_colab),
            ("colab notebook", test_notebook),
            ("pipeline orchestration", test_pipeline),
        ):
            print(f"\n=== {label} ===")
            failed |= module.main()
        print("\nALL SUITES PASSED" if not failed else "\nSOME SUITES FAILED")
        return failed

    cfg = RedAAConfig()

    # A SLURM array task id is the natural chunk index; honour it unless the
    # environment or the command line says otherwise.
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if args.chunk_index is not None:
        cfg.chunk_index = args.chunk_index
    elif array_id and "CHUNK_INDEX" not in os.environ:
        cfg.chunk_index = int(array_id)

    if args.input_dir:
        cfg.input_pdb_dir = args.input_dir
    if args.chunk_size is not None:
        cfg.chunk_size = args.chunk_size
    if args.replicates is not None:
        cfg.num_replicates = args.replicates
    if args.test:
        cfg.apply_test_mode()
        print("\n*** TEST MODE: first domain, 1 replicate, reduced budgets ***\n")

    if args.fetch:
        from redaa.colab import describe_inputs, gather_inputs

        print(f"Preparing inputs in {cfg.input_pdb_dir}\n")
        reports = gather_inputs(
            specs=args.fetch, input_dir=cfg.input_pdb_dir, chain=args.chain
        )
        if not reports:
            print("\nNo inputs could be prepared.")
            return 1
        print("\n" + describe_inputs(reports))
        if args.fetch_only:
            return 0
        print()
    elif args.fetch_only:
        parser.error("--fetch-only requires --fetch")

    if args.print_config:
        print("Resolved configuration:")
        print(cfg.describe())
        return 0

    from redaa.pipeline import Pipeline

    return Pipeline(cfg).run()


if __name__ == "__main__":
    raise SystemExit(main())
