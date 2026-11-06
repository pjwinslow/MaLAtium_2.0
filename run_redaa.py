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
        "--list-presets",
        action="store_true",
        help="describe the verified functional-site test cases and exit",
    )
    parser.add_argument(
        "--preset",
        default=None,
        metavar="KEY",
        help="configure a site test case by key (see --list-presets). Sets the "
        "site spec, the ligands to keep and the MPNN model type, and supplies "
        "the structure to --fetch when none is given.",
    )
    parser.add_argument(
        "--preview-site",
        action="store_true",
        help="resolve the configured site against every prepared input, print "
        "it and exit. Worth doing before any long run: a site spec is the one "
        "setting that can be wrong without the run noticing.",
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

    if args.list_presets:
        from redaa.sites import describe_presets

        print("Verified functional-site test cases:\n")
        print(describe_presets())
        print(
            "\nUse one with, for example:\n"
            "  python run_redaa.py --preset tev --fetch-only\n"
            "  SEARCH_MODE=site_aware python run_redaa.py --preset tev\n"
        )
        return 0

    if args.self_test:
        from tests import (
            test_colab,
            test_fetch,
            test_mpnn,
            test_notebook,
            test_pipeline,
            test_search,
            test_sites,
            test_structure,
        )

        failed = 0
        for label, module in (
            ("structural metrics", test_structure),
            ("search and chemistry", test_search),
            ("fetching and input prep", test_fetch),
            ("functional sites", test_sites),
            ("mpnn integration", test_mpnn),
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

    # A preset supplies the site, its ligands and the MPNN model type, and
    # the structure to fetch when none was named. Anything set explicitly on
    # the command line or in the environment still wins.
    keep_ligands = cfg.keep_ligand_tuple
    if args.preset:
        from redaa.sites import preset as _load_preset

        chosen = _load_preset(args.preset)
        print(chosen.describe())
        print()
        cfg.site_preset = chosen.key
        if not cfg.site_spec:
            cfg.site_spec = chosen.site_spec
        if not keep_ligands:
            keep_ligands = chosen.keep_ligands
            cfg.site_keep_ligands = ",".join(chosen.keep_ligands)
        if "MPNN_MODEL_TYPE" not in os.environ:
            cfg.mpnn_model_type = chosen.mpnn_model_type
        if args.fetch is None:
            args.fetch = chosen.fetch_spec
        if cfg.search_mode == "alphabet_only":
            print(
                "NOTE: SEARCH_MODE is alphabet_only, so the site would be "
                "ignored.\n      Set SEARCH_MODE=site_aware or "
                "SEARCH_MODE=scaffold_then_site to use it.\n"
            )

    if args.fetch:
        from redaa.colab import describe_inputs, gather_inputs

        print(f"Preparing inputs in {cfg.input_pdb_dir}\n")
        reports = gather_inputs(
            specs=args.fetch, input_dir=cfg.input_pdb_dir, chain=args.chain,
            keep_ligands=keep_ligands or None,
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

    if args.preview_site:
        from redaa.colab import preview_sites

        if not cfg.site_spec:
            parser.error("--preview-site needs SITE_SPEC or --preset")
        print(f"Resolving {cfg.site_spec!r} against {cfg.input_pdb_dir}\n")
        print(
            preview_sites(
                cfg.input_pdb_dir, cfg.site_spec,
                chain=cfg.site_chain or None,
                contact_max=cfg.site_contact_max,
            )
        )
        return 0

    problems = cfg.validate()
    if problems:
        print("Configuration problems:")
        for line in problems:
            print(f"  - {line}")
        return 1

    if args.print_config:
        print("Resolved configuration:")
        print(cfg.describe())
        return 0

    from redaa.pipeline import Pipeline

    return Pipeline(cfg).run()


if __name__ == "__main__":
    raise SystemExit(main())
