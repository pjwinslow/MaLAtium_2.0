#!/usr/bin/env python3
"""
RedAA_finder — Batch Reduced Alphabet Search (V9)

Finds the minimal amino acid alphabet that allows ProteinMPNN to design
sequences that fold back to the reference crystal structure when predicted
by ESMFold.

V9 CHANGES (from V8):
  Input is now a directory of pre-processed CATH domain PDB files rather
  than an RCSB fetch + monomer extraction pipeline.  Files are expected to
  be single-chain, single-domain, standard-residue PDB files ready for
  ProteinMPNN.  The filename stem (e.g. "3q46A00" from "3q46A00.pdb") is
  used as the domain identifier throughout all output files and logs.

  Removed: PDB_LIST, fetch_pdb_from_rcsb, extract_monomer_chain,
           count_chains_in_pdb, select_longest_chain, requests import.

VALIDATION APPROACH:
  ESMFold round-trip structural testing against the crystal PDB.
  Acceptance gate: TM-score ≥ min_tm_score (Zhang & Skolnick 2004 threshold
  for "same fold family"; length-normalised and therefore consistent across
  proteins of different sizes).  RMSD is computed for every sequence and used
  for candidate ranking and reporting, but is NOT a pass/fail gate.
  pLDDT is a soft disorder filter — only rejects outputs where ESMFold
  produced no structural model at all.

SEARCH ALGORITHM:
  Forward greedy search with per-round ESMFold quick-validation, followed by
  backward pruning (exhaustive for small alphabets, greedy-best for large).
  Starting alphabet: A + G + any chemically required residues (CYS for
  disulfide bonds, PRO for cis-peptide bonds) + best aliphatic (V/I/L).

HARDWARE:
  ESMFold ESM-2 trunk runs in float16; structure module stays in float32.
  ~16 GB VRAM on A100-40GB for proteins up to ~700 residues.
  ESMFold takes ~5–15 s/sequence on GPU.

CAVEATS:
  • The fallback TM-score (when tmtools is unavailable) is computed per
    residue after NW-alignment and BioPython Superimposer superposition
    (Zhang & Skolnick 2004 formula).  Install tmtools for exact TM-align.
  • ESMFold pLDDT is a confidence estimate, not a fold-correctness measure.
    Restricted alphabets can produce correct folds with low pLDDT because
    ESMFold is uncertain about unusual compositions.
  • For proteins < 40 residues, TM-score normalisation is unreliable;
    RMSD and pLDDT are the more informative metrics in that regime.

USAGE:
  python New_V9.py [--test]
  Configure CHUNK_INDEX / CHUNK_SIZE below for multi-session runs.

  Place pre-processed CATH domain PDB files in INPUT_PDB_DIR (default:
  "input_pdbs/" in the same directory as this script).  All *.pdb files
  in that directory will be discovered, sorted alphabetically, and
  processed according to the chunking configuration.
"""

import os
import sys
import glob
import subprocess
import statistics
import csv
import shutil
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# TEST MODE
# Quick single-PDB smoke test:  python New_V9.py --test
#
# Runs 1 replicate of the first PDB file found in INPUT_PDB_DIR with
# reduced MPNN sampling and folds only 2 sequences per validation call.
# Completes in ~5–10 minutes on GPU.
# Use this to verify the full pipeline before submitting a batch job.
# =============================================================================

TEST_MODE = "--test" in sys.argv
if TEST_MODE:
    print("\n*** TEST MODE — first PDB in input dir, 1 replicate ***\n")

# =============================================================================
# CHUNKING  — change CHUNK_INDEX between GPU sessions
# =============================================================================

CHUNK_INDEX = int(os.environ.get("CHUNK_INDEX", 0))   # set by SLURM array task ID

CHUNK_SIZE  = int(os.environ.get("CHUNK_SIZE", 10))   # 1 PDB per job for preempt safety

# Resume an interrupted chunk run by pointing here.  Leave None to start fresh.
RESUME_FROM = None  # e.g. "outputs/batch_run_chunk0_20240601_120000"

# =============================================================================
# INPUT PDB DIRECTORY
# =============================================================================
# Pre-processed CATH domain PDB files.  Each file should be:
#   - Single chain, single domain
#   - Standard amino acids only (no HETATM, no modified residues)
#   - Ready to pass directly to ProteinMPNN
#
# Filename stem is used as the domain identifier (e.g. "3q46A00" from
# "3q46A00.pdb").  Files are sorted alphabetically for reproducibility.

INPUT_PDB_DIR = os.environ.get(
    "INPUT_PDB_DIR",
     "/projects/standard/seeligb/shared/proteinMPNN/ProteinMPNN_latest/All_inputs_skipped_rerun"
   # "/projects/standard/seeligb/shared/proteinMPNN/ProteinMPNN_latest/All_inputs_260612"
   # "/projects/standard/seeligb/shared/proteinMPNN/ProteinMPNN_latest/input_pdbs_plus"
    #luca_domain_pdbs_clean
    #os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_pdbs")
)

# =============================================================================
# DISCOVER INPUT PDBs
# =============================================================================

_all_input_pdbs = sorted(glob.glob(os.path.join(INPUT_PDB_DIR, "*.pdb")))

if not _all_input_pdbs:
    print(f"ERROR: No .pdb files found in {INPUT_PDB_DIR}")
    print(f"  Place pre-processed CATH domain PDB files in this directory.")
    sys.exit(1)

# Build the working list: (domain_name, pdb_path)
PDB_LIST = [(os.path.splitext(os.path.basename(p))[0], p) for p in _all_input_pdbs]

print(f"Discovered {len(PDB_LIST)} PDB files in {INPUT_PDB_DIR}")

# =============================================================================
# GENERAL CONFIGURATION
# =============================================================================

NUM_REPLICATES         = 10
# ── DYNAMIC SEEDING (V10 modification) ───────────────────────────────────────
# Instead of always starting with A+G, run an unconstrained 20-AA MPNN design
# on the backbone, compute per-residue composition, and seed with the N most
# abundant residues. This gives a backbone-specific, data-driven starting point
# and eliminates the systematic A/G path-dependence bias.
DYNAMIC_SEEDING        = False
DYNAMIC_SEED_SIZE      = 2   # number of AAs to seed (top-N by composition)
STARTING_ALPHABET      = ["A","G"]  # fallback when DYNAMIC_SEEDING = False
USE_HYDROPHOBIC_SEEDING = True     # off under dynamic seeding

# ── TM-ONLY VALIDATION GATE (V10 modification) ──────────────────────────────
# pLDDT is a confidence estimate, not a fold-correctness measure. With reduced
# alphabets ESMFold can produce correct folds with low pLDDT because the
# sequences are far from its training distribution. Using pLDDT as a gate
# rejects valid folds and introduces inconsistent behaviour at small alphabet
# sizes. Set False to gate on TM score only; pLDDT is still logged.
USE_PLDDT_GATE = False

mpnn_script   = "protein_mpnn_run.py"
model_weights = "vanilla_model_weights/"
model_name    = "v_48_020"

all_aas = list("ACDEFGHIKLMNPQRSTVWY")

# MPNN sampling
mpnn_num_seq    = 25   # sequences generated per call
mpnn_batch_size = 5    # MPNN internal batch size

# Sampling temperature — fixed at T=0.5 for all reduced-alphabet runs.
#
# WHY CONSTANT rather than adaptive:
#   The earlier adaptive schedule raised T to 0.9 for <=6-AA alphabets to
#   encourage diversity, but at T=0.9 the softmax over 6 allowed residues
#   is nearly flat — MPNN is sampling close to uniformly and largely
#   ignoring its conditioning on the backbone phi/psi angles.  This defeats
#   the purpose of ProteinMPNN, which was designed to generate sequences
#   compatible with a specific backbone, not random sequences from the
#   allowed set.
#
#   T=0.5 is a principled middle ground:
#   - Sufficient diversity: at T=0.5 the softmax spread over a 6-AA
#     alphabet is ~3-4x wider than at T=0.25, generating meaningfully
#     different sequences that the diversity filter can spread further.
#   - Backbone conditioning retained: unlike T=0.9, MPNN's learned
#     backbone preferences still shape the distribution.
#   - Consistent across alphabet sizes: a single value applies the same
#     exploration/exploitation tradeoff everywhere, making cross-alphabet
#     MPNN score comparisons internally consistent.
#
# The 20-AA baseline run always uses T=0.25 (hardcoded in
# run_single_replicate) — this is intentional, giving the coldest
# reference score independent of this setting.
MPNN_SAMPLING_TEMP = 0.5

def mpnn_temp_for_alphabet(n_aas):
    """Return the MPNN sampling temperature. Fixed at MPNN_SAMPLING_TEMP."""
    return MPNN_SAMPLING_TEMP

# Greedy search stopping criterion
baseline_tolerance = 1.50  # stop if score ≤ baseline × this

# ESMFold validation
num_seqs_to_fold        = 9     # sequences folded per full validation call
                                # (stratified across MPNN score distribution)
num_seqs_quick_validate = 3     # sequences folded per candidate AA per round.
                                # Every candidate alphabet is quick-validated
                                # each round so the search picks the AA with
                                # the best structural outcome, not just the
                                # best MPNN score.  3 folds × ~17 candidates
                                # = ~51 ESMFold calls per round (~8 min on GPU).
max_identity_for_fold   = 0.70  # greedy diversity filter before folding

# Start validation only after the alphabet has this many residues.
# ESMFold was trained on natural sequences; AG-only compositions give
# artifactually low pLDDT, not because the fold is wrong.
esm_start_after_n_aas = 3

# Structural acceptance thresholds.
#
# ACCEPTANCE GATE — TM-score only:
#   TM-score is length-normalised and therefore consistent across proteins of
#   different sizes.  ≥ 0.5 is the Zhang & Skolnick (2004) threshold for
#   "same fold family" and is independent of chain length.  It is the single
#   hard gate for alphabet acceptance.
#
# WHY RMSD IS NOT A GATE:
#   A fixed Å cutoff applied across a panel spanning ~50–250 residues creates
#   systematic length-dependent bias.  ESMFold's intrinsic deviation from the
#   crystal for the native sequence alone is already 1–4 Å for beta-rich folds,
#   leaving almost no margin for reduced-alphabet sequences on larger proteins.
#   RMSD is kept as a ranking and reporting metric — used to choose among
#   candidates that all pass TM ≥ min_tm_score, and reported in all output
#   files — but it does not determine pass/fail.
#
# pLDDT is a SOFT disorder filter — only rejects outputs where ESMFold
# produced no meaningful structural model at all (completely disordered).
# It is not a quality threshold.
min_tm_score        = 0.51   # TM-score normalised by reference length — HARD GATE
                            # (≥ 0.5 = same fold family by Zhang & Skolnick 2004)
min_plddt_disorder  = 30.0  # pLDDT floor — soft disorder filter only.
                            # Rejects outputs where ESMFold produced no structure.
                            # pLDDT > 50 = structured prediction worth measuring.
max_rmsd_threshold  = 3.0   # Å CA-RMSD — RANKING / REPORTING ONLY, not a gate.
                            # Used to rank candidates that all pass TM ≥ min_tm_score.

# Backward pruning
DO_BACKWARD_PRUNING    = True
PRUNING_MIN_SIZE              = 3    # only prune alphabets larger than this
EXHAUSTIVE_PRUNING_THRESHOLD  = 8    # alphabets <= this size use exhaustive search;
                                     # larger alphabets use greedy-best single+pair

# Disk management
DELETE_MPNN_SEQS_AFTER_USE = True

# Override settings in test mode
if TEST_MODE:
    PDB_LIST                = [PDB_LIST[0]]   # first file only
    NUM_REPLICATES          = 1
    mpnn_num_seq            = 50
    num_seqs_to_fold        = 2
    num_seqs_quick_validate = 1   # 1 fold per candidate in test mode
    DO_BACKWARD_PRUNING     = True
    CHUNK_INDEX             = 0
    CHUNK_SIZE              = 1

use_gpu = torch.cuda.is_available()

# =============================================================================
# CHUNK SETUP
# =============================================================================

chunk_start = CHUNK_INDEX * CHUNK_SIZE
chunk_end   = min(chunk_start + CHUNK_SIZE, len(PDB_LIST))
PDB_CHUNK   = PDB_LIST[chunk_start:chunk_end]

if not PDB_CHUNK:
    print(f"ERROR: CHUNK_INDEX={CHUNK_INDEX} is out of range "
          f"({len(PDB_LIST)} PDBs, CHUNK_SIZE={CHUNK_SIZE}).")
    sys.exit(1)

if RESUME_FROM and os.path.isdir(RESUME_FROM):
    BATCH_OUTPUT_ROOT = RESUME_FROM
    print(f"Resuming from: {BATCH_OUTPUT_ROOT}")
elif os.environ.get("AUTO_RESUME") == "1":
    import glob as _glob
    existing = sorted(_glob.glob(f"all_outputs_260615_seeded/re_runs/batch_run_esmfold_chunk{CHUNK_INDEX}_*"))
    if existing and os.path.isdir(existing[-1]):
        BATCH_OUTPUT_ROOT = existing[-1]
        print(f"Auto-resuming from: {BATCH_OUTPUT_ROOT}")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        BATCH_OUTPUT_ROOT = f"all_outputs_260615_seeded/re_runs/batch_run_esmfold_chunk{CHUNK_INDEX}_{ts}"
        print(f"Starting fresh: {BATCH_OUTPUT_ROOT}")
else:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    BATCH_OUTPUT_ROOT = f"all_outputs_260615_seeded/re_runs/batch_run_esmfold_chunk{CHUNK_INDEX}_{ts}"
    print(f"Starting fresh: {BATCH_OUTPUT_ROOT}")

# =============================================================================
# ESMFOLD INITIALISATION
# =============================================================================

print("=" * 60)
print("BATCH: RedAA_finder — ESMFold Validation (V9)")
print("=" * 60)
print(f"\nInput directory: {INPUT_PDB_DIR}")
print(f"Chunk {CHUNK_INDEX}: PDBs {chunk_start+1}–{chunk_end} of {len(PDB_LIST)}")
print(f"  This chunk : {[name for name, _ in PDB_CHUNK]}")
print(f"  Replicates : {NUM_REPLICATES}")
print(f"  Seqs folded per validation: {num_seqs_to_fold}")
print(f"  Seeding    : {'dynamic (top-' + str(DYNAMIC_SEED_SIZE) + ' from MPNN composition)' if DYNAMIC_SEEDING else 'fixed ' + str(STARTING_ALPHABET)}")
print(f"  Thresholds : TM ≥ {min_tm_score}"
      + (f", pLDDT disorder filter ≥ {min_plddt_disorder}" if USE_PLDDT_GATE else " (pLDDT gate OFF)")
      + f" (RMSD ≤ {max_rmsd_threshold} Å used for ranking only)")
print(f"  Output dir : {BATCH_OUTPUT_ROOT}")

print("\n" + "=" * 60)
print("ESMFold MODEL SETUP")
print("=" * 60)

try:
    from transformers import EsmForProteinFolding, AutoTokenizer

    print("\nLoading ESMFold tokenizer...")
    esmfold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    print("Loading ESMFold model weights (~2.7 GB)...")
    esmfold_model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        low_cpu_mem_usage=True,
    )
    esmfold_model = esmfold_model.eval()

    if use_gpu:
        esmfold_model = esmfold_model.cuda()
        # Run the ESM-2 language model trunk in float16 to halve VRAM usage.
        # The folding trunk (structure module) stays in float32 for numerical
        # stability.  This is the approach used in the official ESMFold demo.
        esmfold_model.esm = esmfold_model.esm.half()
        print(f"✓ ESMFold on GPU — ESM trunk in fp16, folding trunk in fp32")
    else:
        print("✓ ESMFold on CPU — this will be slow (~60–180 s/sequence)")

except ImportError as e:
    print(f"\nERROR: {e}")
    print("Install: pip install 'transformers>=4.36.0' accelerate")
    sys.exit(1)
except Exception as e:
    print(f"\nERROR loading ESMFold: {e}")
    sys.exit(1)

# Check for tmtools (exact TM-score); fall back to BioPython approximation.
try:
    import tmtools as _tmtools
    _have_tmtools = True
    print("✓ tmtools found — using exact TM-align TM-scores")
except ImportError:
    _have_tmtools = False
    print("  tmtools not found — TM-scores will be approximated from RMSD + length")
    print("  (pip install tmtools for exact values)")

print("=" * 60 + "\n")

# =============================================================================
# ESMFold FOLDING
# =============================================================================

def fold_sequence_with_esmfold(sequence, output_pdb_path):
    """
    Predict the structure of *sequence* with ESMFold and write the result as
    a PDB file.

    Returns (success: bool, mean_plddt: float | None).

    pLDDT is returned on the 0–100 scale (ESMFold internally uses 0–1;
    we multiply by 100 here to match the pLDDT convention used in AlphaFold
    output and in the literature).

    Uses esmfold_model.output_to_pdb() — the built-in conversion method —
    rather than manually invoking atom14_to_atom37 + OpenFold's Protein class.
    The manual route requires accessing keys that are not guaranteed to exist
    in all versions of EsmForProteinFoldingOutput and is fragile.
    """
    try:
        # Tokenise — add_special_tokens=False because ESMFold handles special
        # tokens internally; adding them here causes a length mismatch.
        tokens = esmfold_tokenizer(
            [sequence],
            return_tensors="pt",
            add_special_tokens=False,
        )
        if use_gpu:
            tokens = {k: v.cuda() for k, v in tokens.items()}

        with torch.no_grad():
            outputs = esmfold_model(**tokens)

        # pLDDT: shape (batch, L) on 0–1 scale → convert to 0–100
        plddt_per_res = outputs.plddt[0, :len(sequence)]  # (L,)
        mean_plddt    = plddt_per_res.mean().item() * 100.0

        # output_to_pdb() returns a list of PDB strings, one per sequence in
        # the batch.  We always fold one sequence at a time so index [0].
        pdb_string = esmfold_model.output_to_pdb(outputs)[0]
        with open(output_pdb_path, 'w') as f:
            f.write(pdb_string)

        return True, mean_plddt

    except Exception as e:
        print(f"    ESMFold error: {e}")
        return False, None

# =============================================================================
# STRUCTURAL COMPARISON METRICS
# =============================================================================

# One-letter AA code lookup shared by all metric functions.
_AA_CODES_1LETTER = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F',
    'GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L',
    'MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R',
    'SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y',
}


def _get_ca_atoms(pdb_path):
    """
    Return a list of BioPython Atom objects for CA atoms in *pdb_path*.

    - First model only: avoids duplicate atoms from NMR ensembles.
    - Sorted by residue sequence number: deterministic pairing order.
    - ATOM records only: excludes HETATM (ligands, waters, modified residues).
    - Blank or 'A' altloc only: if a residue has alternate conformations,
      take only the primary one to avoid duplicate CA atoms that would
      inflate the RMSD calculation.
    """
    from Bio.PDB import PDBParser
    parser    = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)
    model     = next(structure.get_models())

    ca_atoms = []
    for chain in model:
        residues = sorted(chain.get_residues(), key=lambda r: r.get_id()[1])
        for res in residues:
            # Skip HETATM residues (water, ligands, modified AAs)
            if res.get_id()[0] != ' ':
                continue
            if 'CA' not in res:
                continue
            ca = res['CA']
            # Skip non-primary alternate conformations
            if ca.get_altloc() not in (' ', 'A', ''):
                continue
            ca_atoms.append(ca)
    return ca_atoms


def _nw_align(seq1, seq2, match=2, mismatch=-1, gap=-1):
    """
    Minimal Needleman-Wunsch global alignment.  Pure-Python, no external
    dependencies beyond the stdlib.  O(mn) time and space — fast enough
    for protein sequences up to several hundred residues.

    Returns (aligned_seq1, aligned_seq2) as strings using '-' for gaps.
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i * gap
    for j in range(n + 1):
        dp[0][j] = j * gap
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = match if seq1[i - 1] == seq2[j - 1] else mismatch
            dp[i][j] = max(
                dp[i - 1][j - 1] + s,
                dp[i - 1][j]     + gap,
                dp[i][j - 1]     + gap,
            )
    a1, a2 = [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = match if seq1[i - 1] == seq2[j - 1] else mismatch
            if dp[i][j] == dp[i - 1][j - 1] + s:
                a1.append(seq1[i - 1]); a2.append(seq2[j - 1])
                i -= 1; j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + gap:
            a1.append(seq1[i - 1]); a2.append('-')
            i -= 1
        else:
            a1.append('-'); a2.append(seq2[j - 1])
            j -= 1
    return ''.join(reversed(a1)), ''.join(reversed(a2))


def _align_and_pair_atoms(ref_pdb, pred_pdb):
    """
    Return matched BioPython CA Atom lists for *ref_pdb* and *pred_pdb*,
    along with the full reference chain length needed for TM normalisation.

    Both structures are parsed via _get_ca_atoms() so extraction is
    consistent: first model only, ATOM records only, blank/A altloc only.
    This fixes the inconsistency that previously existed between the
    different metric calculation paths.

    SAME-LENGTH CASE (overwhelmingly common in this pipeline):
        MPNN designs sequences of exactly the same length as the crystal
        backbone; ESMFold folds all residues.  Atoms are paired directly
        with no alignment overhead.

    DIFFERENT-LENGTH CASE (ESMFold occasionally drops residues):
        A global NW alignment of the one-letter sequences is performed and
        only positions where BOTH structures have a residue are returned.
        This avoids the frame-shift error that "first L atoms" produces
        when missing residues are internal (e.g. a disordered loop) rather
        than terminal.

    Returns
    -------
    atoms_ref_paired  : list of BioPython Atom — reference CA atoms
    atoms_pred_paired : list of BioPython Atom — prediction CA atoms (same length)
    L_ref             : int — FULL reference chain length (for TM normalisation);
                        equals len(atoms_ref_paired) when lengths match.
    """
    atoms_ref  = _get_ca_atoms(ref_pdb)
    atoms_pred = _get_ca_atoms(pred_pdb)

    if not atoms_ref or not atoms_pred:
        return [], [], 0

    L_ref  = len(atoms_ref)
    L_pred = len(atoms_pred)

    # ── Common case: same atom count → direct 1-to-1 pairing ─────────────────
    if L_ref == L_pred:
        return atoms_ref, atoms_pred, L_ref

    # ── Lengths differ: align sequences and pair only matched positions ────────
    mismatch_pct = abs(L_ref - L_pred) / max(L_ref, L_pred) * 100
    print(f"    \u26a0 Length mismatch: ref={L_ref}, pred={L_pred} "
          f"({mismatch_pct:.0f}%); aligning to find correspondence", end="")

    seq_ref  = ''.join(_AA_CODES_1LETTER.get(a.get_parent().get_resname(), 'X')
                        for a in atoms_ref)
    seq_pred = ''.join(_AA_CODES_1LETTER.get(a.get_parent().get_resname(), 'X')
                        for a in atoms_pred)

    aln_ref, aln_pred = _nw_align(seq_ref, seq_pred)

    idx_ref = idx_pred = 0
    paired_ref, paired_pred = [], []
    for r, p in zip(aln_ref, aln_pred):
        if r != '-' and p != '-':
            paired_ref.append(atoms_ref[idx_ref])
            paired_pred.append(atoms_pred[idx_pred])
        if r != '-':
            idx_ref  += 1
        if p != '-':
            idx_pred += 1

    n_paired = len(paired_ref)
    print(f" -> {n_paired}/{L_ref} positions paired", end="")
    return paired_ref, paired_pred, L_ref


def calculate_rmsd(ref_pdb, pred_pdb):
    """
    Compute the CA-RMSD between *ref_pdb* (crystal) and *pred_pdb*
    (ESMFold prediction) after optimal superposition via BioPython's
    Superimposer.

    Uses _align_and_pair_atoms() to build matched Cα pairs consistently:
    - Both structures extracted with identical filters (ATOM, first model,
      blank/A altloc) — fixing the inconsistency that previously existed
      between the two metric paths.
    - When structures have the same length (the common case), atoms are
      paired directly.
    - When lengths differ, a global NW alignment determines which residues
      correspond, avoiding the frame-shift error that "first L atoms"
      produces when ESMFold drops internal rather than terminal residues.

    Returns RMSD in Å, or None on error.
    """
    try:
        from Bio.PDB import Superimposer

        atoms_ref_p, atoms_pred_p, L_ref = _align_and_pair_atoms(ref_pdb, pred_pdb)

        if not atoms_ref_p or not atoms_pred_p:
            return None

        sup = Superimposer()
        sup.set_atoms(atoms_ref_p, atoms_pred_p)
        sup.apply(atoms_pred_p)
        return sup.rms

    except ImportError:
        print("    WARNING: BioPython not available — RMSD cannot be computed")
        return None
    except Exception as e:
        print(f"    RMSD error: {e}")
        return None


def calculate_tm_score(ref_pdb, pred_pdb):
    """
    Compute TM-score normalised by the reference chain length.

    Priority order:
      1. tmtools (pip install tmtools) — runs the exact TM-align algorithm.
         Extracts CA coordinates and sequences using _get_ca_atoms() so
         filtering is identical to the RMSD path (ATOM records, first model,
         blank/A altloc only).  Previously this path used its own parsing
         loop that applied none of those filters.

      2. Per-residue BioPython fallback — uses _align_and_pair_atoms() for
         consistent atom extraction and correct residue correspondence, then
         computes the exact TM-score sum:
           TM = (1/L_ref) * sum_i 1 / (1 + (d_i/d0)^2)
         where d_i is the per-residue distance AFTER superposition and
         d0 = 1.24*(L_ref-15)^(1/3) - 1.8  (Zhang & Skolnick 2004).

         Normalisation uses L_ref (the FULL reference chain length) so that
         partial matches to a longer reference are penalised correctly even
         when some residues were dropped by ESMFold.

    Returns TM-score (float in [0, 1]), or None on error.
    """
    # ── 1. tmtools (exact TM-align) ──────────────────────────────────────────
    if _have_tmtools:
        try:
            import tmtools

            def _get_ca_coords_and_seq(pdb_path):
                """
                Extract CA coordinates and one-letter sequence using
                _get_ca_atoms() so filtering is consistent with every other
                metric function (ATOM records, first model, blank/A altloc).
                """
                atoms = _get_ca_atoms(pdb_path)
                coords = np.array(
                    [a.get_vector().get_array() for a in atoms],
                    dtype=np.float64,
                )
                seq = ''.join(
                    _AA_CODES_1LETTER.get(a.get_parent().get_resname(), 'X')
                    for a in atoms
                )
                return coords, seq

            coords_ref,  seq_ref  = _get_ca_coords_and_seq(ref_pdb)
            coords_pred, seq_pred = _get_ca_coords_and_seq(pred_pdb)

            if len(coords_ref) < 15 or len(coords_pred) < 15:
                raise ValueError("Chain too short for TM-align")

            result = tmtools.tm_align(coords_ref, coords_pred, seq_ref, seq_pred)
            return result.tm_norm_chain1

        except Exception as e:
            print(f"    tmtools error ({e}), falling back to per-residue calc")

    # ── 2. Per-residue BioPython fallback ────────────────────────────────────
    try:
        from Bio.PDB import Superimposer

        atoms_ref_p, atoms_pred_p, L_ref = _align_and_pair_atoms(ref_pdb, pred_pdb)

        if not atoms_ref_p:
            return None
        if L_ref < 15:
            return None

        sup = Superimposer()
        sup.set_atoms(atoms_ref_p, atoms_pred_p)
        sup.apply(atoms_pred_p)

        # d0: length-dependent distance scale (Zhang & Skolnick 2004).
        # Clamped to 0.5 A minimum to avoid d0 going negative for L < 22.
        d0 = max(0.5, 1.24 * (L_ref - 15) ** (1.0 / 3.0) - 1.8)

        tm_sum = 0.0
        for a_ref, a_pred in zip(atoms_ref_p, atoms_pred_p):
            di = a_ref - a_pred   # BioPython Atom subtraction gives distance
            tm_sum += 1.0 / (1.0 + (di / d0) ** 2)

        tm_score = tm_sum / L_ref
        return tm_score

    except ImportError:
        print("    WARNING: BioPython not available — TM-score cannot be computed")
        return None
    except Exception as e:
        print(f"    TM-score error: {e}")
        return None


# =============================================================================
# SEQUENCE UTILITIES
# =============================================================================

def pairwise_identity(a, b):
    L = min(len(a), len(b))
    if L == 0:
        return 0.0
    return sum(x == y for x, y in zip(a[:L], b[:L])) / L


def diversity_filter(seq_score_pairs, max_identity=0.60):
    """
    Greedily retain sequences such that no two share more than max_identity
    fraction of identical positions.  Input must be sorted ascending by score
    (best first) so the highest-quality sequence always survives.
    """
    kept = []
    for score, seq in seq_score_pairs:
        if not any(pairwise_identity(seq, ks) > max_identity
                   for _, ks in kept):
            kept.append((score, seq))
    return kept


def get_native_sequence(pdb_path):
    """
    Extract the single-letter amino acid sequence from *pdb_path*.

    Uses _AA_CODES_1LETTER (the module-level dict) and restricts to:
    - First model only (NMR ensembles would otherwise produce duplicate sequence)
    - ATOM records only (HETATM residues such as selenomethionine MSE share
      three-letter codes with standard AAs and would inflate the sequence
      length if not excluded)

    Mirrors the filtering applied by _get_ca_atoms(), keeping the sequence
    consistent with the Ca atom count used for RMSD/TM calculations.
    """
    try:
        from Bio.PDB import PDBParser
        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure('ref', pdb_path)
        model     = next(structure.get_models())
        for chain in model:
            seq = [
                _AA_CODES_1LETTER[r.get_resname()]
                for r in chain
                if r.get_id()[0] == ' '
                and r.get_resname() in _AA_CODES_1LETTER
            ]
            if seq:
                return ''.join(seq)
    except Exception:
        pass
    return None


def extract_chain_to_file(pdb_path, chain_id, out_path):
    """
    Write a single chain (*chain_id*) from *pdb_path* to *out_path*, keeping
    only the first model and standard-residue ATOM records (HETATM such as
    waters, ligands, and modified residues like MSE are dropped — consistent
    with the filters used everywhere else in this pipeline).

    Used for raw multi-chain RCSB PDBs (4-character IDs): the design target,
    the native-sequence reference, and the RMSD/TM reference must all be the
    SAME single chain, so we materialise chain A to disk once and point the
    rest of the pipeline at it.

    Returns the number of standard residues written (0 if the chain is absent
    or contains no standard residues).
    """
    from Bio.PDB import PDBParser, PDBIO, Select
    parser    = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)
    model     = next(structure.get_models())

    if chain_id not in [c.id for c in model]:
        return 0

    n_std = sum(1 for res in model[chain_id]
                if res.id[0] == ' ' and res.get_resname() in _AA_CODES_1LETTER)
    if n_std == 0:
        return 0

    _first_model_id = model.id

    class _ChainStdSelect(Select):
        def accept_model(self, m):
            return m.id == _first_model_id
        def accept_chain(self, c):
            return c.id == chain_id
        def accept_residue(self, r):
            return r.id[0] == ' ' and r.get_resname() in _AA_CODES_1LETTER

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path, select=_ChainStdSelect())
    return n_std

# =============================================================================
# STRUCTURAL REQUIREMENTS ANALYSIS
# =============================================================================

def analyze_pdb_requirements(pdb_path):
    """
    Inspect a monomeric PDB structure and return amino acids that must be
    present in the starting alphabet based on unambiguous chemical necessity.

    WHAT IS INCLUDED — and why each is chemically non-negotiable:

    1. CYS — structural disulfide bonds
       Detected via SG–SG distance < 2.5 Å (covalent bond length ~2.05 Å,
       threshold allows for coordinate error).  Disulfides are covalent
       cross-links; no other amino acid can form them.  A chain without Cys
       at these positions cannot adopt the correct topology under oxidising
       conditions.  The search would never find a valid alphabet without Cys
       pre-seeded for these proteins.

    2. PRO — cis-peptide bonds
       The peptide ω dihedral is normally ~180° (trans).  Cis-peptide bonds
       (ω near 0°) are rare overall (~0.5% of all peptide bonds) but ~5% of
       Pro residues adopt the cis conformation.  Non-Pro cis peptides are
       vanishingly rare (~0.03%) and typically artefactual.  The cis
       conformation is enforced by the pyrrolidine ring of Pro — Gly cannot
       substitute because Gly lacks the ring that stabilises ω ≈ 0°.
       Detection: |ω| < 30° for any Pro–X bond.

    WHAT IS NOT INCLUDED — and why:

    - Trans-Pro turns (type I/II beta turns, polyproline helices):
      Gly is often a valid substitute for trans-Pro at turn positions in
      designed sequences.  Forcing Pro for all turn positions inflates the
      reported minimal alphabet for proteins that don't need it.  Instead,
      Pro is added to the hydrophobic seeding candidate pool so it is tested
      early without being mandated.

    Returns:
        dict with keys:
          'required_aas'   : list of AA single-letter codes that must be pre-seeded
          'n_disulfides'   : int, number of disulfide bonds detected
          'cis_pro_count'  : int, number of cis-Pro residues detected
          'notes'          : list of human-readable strings explaining each decision
    """
    required  = []
    notes     = []
    n_ss      = 0
    n_cis_pro = 0

    try:
        from Bio.PDB import PDBParser
        import numpy as np

        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure('query', pdb_path)
        model     = next(structure.get_models())

        # ── 1. Disulfide bonds ────────────────────────────────────────────────
        cys_sg = []   # (residue_id, SG_vector)
        for chain in model:
            for res in chain:
                if res.get_resname() == 'CYS' and 'SG' in res:
                    cys_sg.append((res.get_id(), res['SG'].get_vector()))

        SS_DIST_MAX = 2.5   # Å — covalent SG–SG bond
        seen_pairs  = set()
        for i in range(len(cys_sg)):
            for j in range(i + 1, len(cys_sg)):
                dist = (cys_sg[i][1] - cys_sg[j][1]).norm()
                if dist < SS_DIST_MAX:
                    pair = tuple(sorted([cys_sg[i][0], cys_sg[j][0]]))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        n_ss += 1

        if n_ss > 0:
            if 'C' not in required:
                required.append('C')
            notes.append(
                f"CYS required: {n_ss} disulfide bond(s) detected "
                f"(SG–SG < {SS_DIST_MAX} Å)"
            )

        # ── 2. Cis-Pro residues ───────────────────────────────────────────────
        CIS_OMEGA_MAX = 30.0   # degrees — |ω| < 30° is unambiguously cis

        res_list = []
        for chain in model:
            for res in chain:
                if res.id[0] == ' ':   # ATOM records only, not HETATM
                    res_list.append(res)

        for k in range(1, len(res_list)):
            res_prev = res_list[k - 1]
            res_curr = res_list[k]

            if res_curr.get_resname() != 'PRO':
                continue

            if not all(a in res_prev for a in ('CA', 'C')):
                continue
            if not all(a in res_curr for a in ('N', 'CA')):
                continue

            try:
                from Bio.PDB.vectors import calc_dihedral, Vector
                ca_prev = res_prev['CA'].get_vector()
                c_prev  = res_prev['C'].get_vector()
                n_curr  = res_curr['N'].get_vector()
                ca_curr = res_curr['CA'].get_vector()

                omega_rad = calc_dihedral(ca_prev, c_prev, n_curr, ca_curr)
                omega_deg = abs(np.degrees(omega_rad))

                if omega_deg > 180:
                    omega_deg = 360 - omega_deg

                if omega_deg < CIS_OMEGA_MAX:
                    n_cis_pro += 1
                    notes.append(
                        f"PRO required: cis-Pro at residue "
                        f"{res_curr.get_id()[1]} (ω = {omega_deg:.1f}°)"
                    )
            except Exception:
                continue

        if n_cis_pro > 0 and 'P' not in required:
            required.append('P')

    except ImportError:
        notes.append("WARNING: BioPython not available — skipping structural analysis")
    except Exception as e:
        notes.append(f"WARNING: Structural analysis failed ({e})")

    return {
        'required_aas':  required,
        'n_disulfides':  n_ss,
        'cis_pro_count': n_cis_pro,
        'notes':         notes,
    }


# =============================================================================
# PROTEINMPNN
# =============================================================================

def run_mpnn(pdb_path, allowed_aas, run_dir, seed=None, temp=None):
    """
    Run ProteinMPNN with per-AA logit bias of -1e9 for disallowed residues.

    Alphabet enforcement uses --bias_AA_jsonl: a per-residue bias dict written
    to a temporary JSON file in run_dir.  Disallowed AAs receive -1e9
    (effectively -infinity in float32), steering MPNN to sample only from
    the allowed set.

    temp: if None, MPNN_SAMPLING_TEMP is used via mpnn_temp_for_alphabet().

    Returns (mean_score, seq_score_pairs) where seq_score_pairs is a list of
    (mpnn_score, sequence) tuples sorted ascending by score.
    """
    if temp is None:
        temp = mpnn_temp_for_alphabet(len(allowed_aas))

    os.makedirs(run_dir, exist_ok=True)

    import json
    bias_path = os.path.join(run_dir, "bias.jsonl")
    bias_dict = {aa: (0.0 if aa in allowed_aas else -1e9) for aa in all_aas}
    with open(bias_path, 'w') as f:
        json.dump(bias_dict, f)

    cmd = (
        f"python {mpnn_script} "
        f"--path_to_model_weights {model_weights} "
        f"--model_name {model_name} "
        f"--pdb_path {pdb_path} "
        f"--out_folder {run_dir} "
        f"--bias_AA_jsonl {bias_path} "
        f"--num_seq_per_target {mpnn_num_seq} "
        f"--sampling_temp {temp:.4f} "
        f"--batch_size {mpnn_batch_size}"
    )
    if seed is not None:
        cmd += f" --seed {seed}"

    subprocess.run(cmd, shell=True, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    scores, pairs = [], []
    seqs_dir = os.path.join(run_dir, "seqs")
    if os.path.exists(seqs_dir):
        for fa in [f for f in os.listdir(seqs_dir) if f.endswith(".fa")]:
            with open(os.path.join(seqs_dir, fa)) as fh:
                cur_score = None
                for line in fh:
                    if line.startswith(">") and "score=" in line:
                        for part in line.split(","):
                            if "score=" in part:
                                try:
                                    cur_score = float(part.split("=")[1])
                                except Exception:
                                    pass
                    elif not line.startswith(">") and cur_score is not None:
                        seq = line.strip()
                        if seq:
                            scores.append(cur_score)
                            pairs.append((cur_score, seq))
                        cur_score = None

        if DELETE_MPNN_SEQS_AFTER_USE:
            shutil.rmtree(seqs_dir, ignore_errors=True)

    pairs.sort()
    return (statistics.mean(scores) if scores else 999.0), pairs

# =============================================================================
# ESMFOLD VALIDATION
# =============================================================================

def create_empty_metrics():
    return {
        'mean_rmsd':  None, 'min_rmsd':   None,
        'mean_tm':    None, 'max_tm':     None,
        'mean_plddt': None, 'max_plddt':  None,
        'num_pass_all': 0,
        'successful_folds': 0, 'total_attempted': 0,
    }

def create_empty_best():
    return {'sequence': None, 'rmsd': None, 'tm': None, 'plddt': None}


def validate_alphabet_with_esmfold(allowed_aas, seq_score_pairs,
                                    reference_pdb, fold_dir,
                                    num_seqs=None, crystal_pdb=None):
    """
    Fold a stratified sample of diverse MPNN sequences with ESMFold and
    compare the resulting structures to *reference_pdb* using RMSD and TM-score.

    SEQUENCE SELECTION — WHY NOT TOP-K BY MPNN SCORE:
    MPNN's score measures how natural a sequence looks under the full 20-AA
    model, not whether it folds correctly under a restricted alphabet.  At
    small alphabets the sequences most compatible with the target fold may
    have mediocre or poor MPNN scores — they sacrifice naturalness to adopt
    the topology.  Selecting only the top-k by MPNN score systematically
    under-samples the fold-compatible region of restricted sequence space.

    STRATIFIED SAMPLING:
    After diversity filtering, sequences are sorted by ascending MPNN score
    (best first).  We then select num_seqs_to_fold sequences spanning the
    full distribution:
      - bottom third  : best MPNN scores  (most natural-looking)
      - middle third  : median MPNN scores
      - top third     : worst MPNN scores (may be most fold-compatible
                        under the restricted alphabet)

    Slots are divided as evenly as possible across thirds.  If the diverse
    pool has fewer sequences than num_seqs_to_fold, all are used.

    Acceptance: at least one folded sequence must satisfy
      TM    ≥ min_tm_score         (hard gate — length-normalised fold identity)
      pLDDT ≥ min_plddt_disorder   (soft disorder filter only)
    RMSD is computed for all sequences and used for ranking among passing
    candidates, but does NOT determine pass/fail.

    Returns (passed: bool, metrics: dict, best_info: dict).
    """
    os.makedirs(fold_dir, exist_ok=True)

    native_seq  = get_native_sequence(crystal_pdb if crystal_pdb else reference_pdb)
    allowed_set = set(allowed_aas) | {'X'}

    # Filter: remove native sequence and out-of-alphabet sequences
    filtered = [(sc, sq) for sc, sq in seq_score_pairs
                if sq != native_seq
                and sq
                and set(sq).issubset(allowed_set)]
    filtered.sort()   # ascending MPNN score — best first

    if not filtered:
        if seq_score_pairs:
            sample_seqs = [sq for _, sq in seq_score_pairs[:3] if sq]
            for i, sq in enumerate(sample_seqs):
                forbidden = sorted(set(sq) - allowed_set)
                print(f" (seq{i+1} AAs: {''.join(sorted(set(sq)))} "
                      f"forbidden: {forbidden or 'none'})", end=" ")
        print(f" (no valid sequences from MPNN — {len(seq_score_pairs)} raw)", end=" ")
        return False, create_empty_metrics(), create_empty_best()

    # Diversity filter
    diverse = diversity_filter(filtered, max_identity=max_identity_for_fold)

    if not diverse:
        print(f" (diversity filter eliminated all {len(filtered)} sequences)", end=" ")
        return False, create_empty_metrics(), create_empty_best()

    # Stratified selection across the score distribution
    n_pool = len(diverse)
    k      = min(num_seqs if num_seqs is not None else num_seqs_to_fold, n_pool)

    if n_pool <= k or n_pool < 3:
        selected = [sq for _, sq in diverse]
    else:
        per_third   = k // 3
        remainder   = k - per_third * 3

        n_best   = per_third + remainder
        n_mid    = per_third
        n_worst  = per_third

        third     = n_pool // 3
        two_third = 2 * third

        best_pool  = diverse[:third]
        mid_pool   = diverse[third:two_third]
        worst_pool = diverse[two_third:]

        def spaced_picks(pool, n):
            if not pool or n == 0:
                return []
            if len(pool) <= n:
                return [sq for _, sq in pool]
            step = len(pool) / n
            indices = [int(i * step) for i in range(n)]
            return [pool[idx][1] for idx in indices]

        selected = (spaced_picks(best_pool,  n_best) +
                    spaced_picks(mid_pool,   n_mid)  +
                    spaced_picks(worst_pool, n_worst))

        seen, unique = set(), []
        for sq in selected:
            if sq not in seen:
                seen.add(sq)
                unique.append(sq)
        selected = unique

    to_fold = selected
    if not to_fold:
        return False, create_empty_metrics(), create_empty_best()

    rmsds, tms, plddts = [], [], []
    seq_metrics = []

    for i, seq in enumerate(to_fold):
        pred_path = os.path.join(fold_dir, f"fold_seq{i}.pdb")
        print(f"    Folding seq {i+1}/{len(to_fold)} "
              f"(len={len(seq)})...", end=" ", flush=True)

        success, plddt = fold_sequence_with_esmfold(seq, pred_path)

        if not success:
            print("FAILED")
            continue

        rmsd = calculate_rmsd(reference_pdb, pred_path)
        tm   = calculate_tm_score(reference_pdb, pred_path)

        r_str = f"{rmsd:.2f}" if rmsd is not None else "N/A"
        t_str = f"{tm:.3f}"   if tm   is not None else "N/A"
        p_str = f"{plddt:.1f}" if plddt is not None else "N/A"
        print(f"RMSD={r_str} Å  TM={t_str}  pLDDT={p_str}")

        if rmsd is not None:
            rmsds.append(rmsd)
        if tm is not None:
            tms.append(tm)
        if plddt is not None:
            plddts.append(plddt)
        seq_metrics.append((seq, rmsd, tm, plddt))

    if not seq_metrics:
        return False, create_empty_metrics(), create_empty_best()

    metrics = {
        'mean_rmsd':       statistics.mean(rmsds)  if rmsds  else None,
        'min_rmsd':        min(rmsds)               if rmsds  else None,
        'mean_tm':         statistics.mean(tms)     if tms    else None,
        'max_tm':          max(tms)                 if tms    else None,
        'mean_plddt':      statistics.mean(plddts)  if plddts else None,
        'max_plddt':       max(plddts)              if plddts else None,
        'successful_folds': len(seq_metrics),
        'total_attempted':  len(to_fold),
        'num_pass_all':    0,
    }

    passed  = False
    num_all = 0
    for seq, rmsd, tm, plddt in seq_metrics:
        ok_tm    = (tm    is not None and tm    >= min_tm_score)
        ok_plddt = (plddt is None     or plddt >= min_plddt_disorder) if USE_PLDDT_GATE else True

        if ok_tm and ok_plddt:
            num_all += 1
            passed = True
    metrics['num_pass_all'] = num_all

    def _quality(item):
        seq, rmsd, tm, plddt = item
        return (tm    if tm    is not None else -999.0,
                -rmsd if rmsd  is not None else -999.0)

    best = max(seq_metrics, key=_quality)
    best_info = {'sequence': best[0], 'rmsd': best[1],
                 'tm': best[2], 'plddt': best[3]}

    return passed, metrics, best_info

# =============================================================================
# BACKWARD PRUNING
# =============================================================================

def _prune_trial(pdb_path, trial_set, trial_dir, seed, reference_pdb=None):
    """Run MPNN + ESMFold validation for a candidate pruned alphabet.

    Uses num_seqs_quick_validate (not the full num_seqs_to_fold budget) because
    pruning involves many trials — O(n) in the greedy path, and up to
    sum_k C(n,k) in the exhaustive path.  3 folds per trial is the same
    budget used per candidate in the forward greedy rounds and is sufficient
    to establish whether a pruned alphabet retains fold-encoding capacity.

    Returns (passed, metrics, best_info).
    """
    fold_dir = os.path.join(trial_dir, "esmfold")
    ref = reference_pdb if reference_pdb else pdb_path
    _, sp = run_mpnn(pdb_path, trial_set, trial_dir, seed=seed)
    return validate_alphabet_with_esmfold(
        trial_set, sp, ref, fold_dir,
        num_seqs=num_seqs_quick_validate,
        crystal_pdb=pdb_path,
    )


def backward_pruning_pass(pdb_path, current_set, pdb_output_dir,
                           replicate_num, base_seed, protected_aas=None,
                           reference_pdb=None):
    """
    Find the globally minimal amino acid alphabet that still encodes the
    target fold, starting from *current_set*.

    PROTECTION:
      Only chemically non-negotiable AAs (cis-Pro PRO, disulfide CYS) are
      protected from removal — passed in via *protected_aas*.  A and G are
      NOT protected: if a fold can be encoded without them, they should be
      removed.  Greedy approaches that hard-protect the starting alphabet
      inflate the reported minimum.

    ALGORITHM — two modes depending on alphabet size:

    EXHAUSTIVE (len(current_set) <= EXHAUSTIVE_PRUNING_THRESHOLD, default 8):
      Try all subsets of decreasing size, one size level at a time.
      Guaranteed to find the globally minimal alphabet.

    GREEDY-BEST FALLBACK (len(current_set) > EXHAUSTIVE_PRUNING_THRESHOLD):
      Greedy-best single removal until convergence, then greedy-best pair
      removal, cycling until neither phase removes anything.
    """
    from itertools import combinations

    pruned  = current_set.copy()
    removed = 0
    trial_counter = 0

    _protected = set(protected_aas) if protected_aas else set()

    n_start = len(pruned)
    print(f"\n  [Backward pruning] Start: {n_start} AAs "
          f"({''.join(sorted(pruned))})  "
          f"protected: {sorted(_protected) if _protected else 'none'}  "
          f"mode: {'exhaustive' if n_start <= EXHAUSTIVE_PRUNING_THRESHOLD else 'greedy-best'}")

    def run_trial(trial_list, label):
        nonlocal trial_counter
        trial_counter += 1
        trial_dir = os.path.join(
            pdb_output_dir, f"run_{replicate_num}",
            f"prune_{trial_counter:04d}_{label}"
        )
        seed = base_seed + 9000 + trial_counter
        passed, metrics, _ = _prune_trial(pdb_path, trial_list, trial_dir, seed,
                                           reference_pdb=reference_pdb)
        return passed, metrics

    def _best_metrics_key(metrics):
        return (
             metrics['max_tm']   if metrics['max_tm']   is not None else -999.0,
            -metrics['min_rmsd'] if metrics['min_rmsd'] is not None else -999.0,
        )

    # ── EXHAUSTIVE PATH ───────────────────────────────────────────────────────
    if len(pruned) <= EXHAUSTIVE_PRUNING_THRESHOLD:

        best_found       = pruned
        best_found_score = None
        min_passing_size = len(pruned)

        for target_size in range(len(pruned) - 1, 1, -1):
            candidates_pool = [a for a in pruned if a not in _protected]
            n_free          = len(candidates_pool)
            n_keep_free     = target_size - len(_protected)

            if n_keep_free < 0:
                break
            if n_keep_free > n_free:
                continue

            passing_at_size = []
            n_tested = 0

            for free_subset in combinations(candidates_pool, n_keep_free):
                subset = sorted(list(_protected) + list(free_subset))
                label  = f"ex{target_size}_" + "".join(sorted(free_subset))
                passed, metrics = run_trial(subset, label)
                n_tested += 1
                if passed:
                    passing_at_size.append((subset, metrics))

            print(f"    Size {target_size}: tested {n_tested} subsets, "
                  f"{len(passing_at_size)} passed")

            if passing_at_size:
                best_subset, best_met = max(
                    passing_at_size, key=lambda x: _best_metrics_key(x[1])
                )
                tm_str   = f"{best_met['max_tm']:.3f}"   if best_met['max_tm']   is not None else "N/A"
                rmsd_str = f"{best_met['min_rmsd']:.2f}" if best_met['min_rmsd'] is not None else "N/A"
                print(f"    -> Best at size {target_size}: "
                      f"{''.join(sorted(best_subset))}  "
                      f"maxTM={tm_str} minRMSD={rmsd_str}")
                best_found       = best_subset
                best_found_score = _best_metrics_key(best_met)
                min_passing_size = target_size
            else:
                print(f"    -> No subset of size {target_size} passed — "
                      f"minimum is size {min_passing_size}")
                break

        removed = n_start - len(best_found)
        pruned  = best_found

    # ── GREEDY-BEST FALLBACK ──────────────────────────────────────────────────
    else:
        def removable(alphabet):
            return [a for a in alphabet if a not in _protected]

        outer_changed = True
        while outer_changed:
            outer_changed = False

            single_changed = True
            while single_changed:
                single_changed = False
                passing_singles = []

                for aa in removable(pruned):
                    trial = [a for a in pruned if a != aa]
                    if len(trial) < 2:
                        continue
                    passed, metrics = run_trial(trial, f"rm{aa}")
                    if passed:
                        passing_singles.append((aa, trial, metrics))

                if passing_singles:
                    best_aa, best_trial, best_met = max(
                        passing_singles, key=lambda x: _best_metrics_key(x[2])
                    )
                    tm_str   = f"{best_met['max_tm']:.3f}"   if best_met['max_tm']   is not None else "N/A"
                    rmsd_str = f"{best_met['min_rmsd']:.2f}" if best_met['min_rmsd'] is not None else "N/A"
                    print(f"  [Single] Removed {best_aa} -> "
                          f"{''.join(sorted(best_trial))}  "
                          f"(maxTM={tm_str} minRMSD={rmsd_str} "
                          f"from {len(passing_singles)} passing candidate(s))")
                    pruned         = best_trial
                    removed       += 1
                    single_changed = True
                    outer_changed  = True

            cands = removable(pruned)
            if len(cands) < 2:
                break

            passing_pairs = []
            for aa1, aa2 in combinations(cands, 2):
                trial = [a for a in pruned if a not in (aa1, aa2)]
                if len(trial) < 2:
                    continue
                passed, metrics = run_trial(trial, f"rm{aa1}{aa2}")
                if passed:
                    passing_pairs.append((aa1, aa2, trial, metrics))

            if passing_pairs:
                best_aa1, best_aa2, best_trial, best_met = max(
                    passing_pairs, key=lambda x: _best_metrics_key(x[3])
                )
                tm_str   = f"{best_met['max_tm']:.3f}"   if best_met['max_tm']   is not None else "N/A"
                rmsd_str = f"{best_met['min_rmsd']:.2f}" if best_met['min_rmsd'] is not None else "N/A"
                print(f"  [Pair]   Removed {best_aa1}+{best_aa2} -> "
                      f"{''.join(sorted(best_trial))}  "
                      f"(maxTM={tm_str} minRMSD={rmsd_str} "
                      f"from {len(passing_pairs)} passing pair(s))")
                pruned        = best_trial
                removed      += 2
                outer_changed = True
            else:
                break

    print(f"  [Backward pruning] Final: {len(pruned)} AAs "
          f"({''.join(sorted(pruned))}), removed {removed} total")
    return pruned, removed


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

DONE_SENTINEL = "COMPLETED"

def rep_is_done(pdb_out_dir, rep):
    return os.path.exists(os.path.join(pdb_out_dir, f"run_{rep}", DONE_SENTINEL))

def mark_rep_done(pdb_out_dir, rep):
    path = os.path.join(pdb_out_dir, f"run_{rep}", DONE_SENTINEL)
    with open(path, 'w') as f:
        f.write(datetime.now().isoformat())

# =============================================================================
# SINGLE REPLICATE
# =============================================================================

def run_single_replicate(pdb_path, pdb_name, rep, pdb_out_dir,
                          starting_alphabet=None, reference_pdb=None):
    """
    Run one complete greedy alphabet search + backward pruning for a single
    PDB and replicate number.

    reference_pdb: crystal (monomeric) PDB used as the RMSD/TM reference for
                   all fold comparisons.  Designed sequences are folded with
                   ESMFold and compared directly to the experimental structure,
                   measuring fold recapitulation in absolute structural terms.
                   Falls back to pdb_path if not provided.
    """
    _ref_pdb = reference_pdb if reference_pdb else pdb_path
    out_root = os.path.join(pdb_out_dir, f"run_{rep}")
    os.makedirs(out_root, exist_ok=True)

    csv_path = os.path.join(out_root, "alphabet_building_log.csv")
    esm_path = os.path.join(out_root, "esmfold_validation_log.csv")

    current_set = ([] if DYNAMIC_SEEDING else
                   (starting_alphabet if starting_alphabet is not None
                    else STARTING_ALPHABET).copy())
    base_seed   = rep * 10_000
    t_start     = datetime.now()

    with open(csv_path, 'w', newline='') as csv_f, \
         open(esm_path, 'w', newline='') as esm_f:

        w   = csv.writer(csv_f)
        ew  = csv.writer(esm_f)

        w.writerow(["Stage", "Round", "Candidate_AA", "MPNN_Score",
                    "Current_Set", "Improvement_%", "Status"])
        ew.writerow(["Stage", "Round", "Alphabet",
                     "Mean_RMSD", "Min_RMSD",
                     "Mean_TM",   "Max_TM",
                     "Mean_pLDDT","Max_pLDDT",
                     "Num_Pass_All", "Successful_Folds", "Total_Attempted",
                     "Validation_Status"])

        def flush():
            csv_f.flush(); esm_f.flush()

        best_validated = {'sequence': None, 'alphabet': None,
                          'rmsd': None, 'tm': None, 'plddt': None}

        def maybe_update_best(info, alph_str):
            def _key(d):
                return (  d['tm']    if d['tm']    is not None else  0.0,
                        -(d['rmsd']  if d['rmsd']  is not None else 999.0))
            if best_validated['sequence'] is None or _key(info) > _key(best_validated):
                best_validated.update({
                    'sequence': info['sequence'],
                    'alphabet': alph_str,
                    'rmsd':     info['rmsd'],
                    'tm':       info['tm'],
                    'plddt':    info['plddt'],
                })

        # ── 1. Baseline (all 20 AAs) ─────────────────────────────────────────
        baseline_score, baseline_pairs = run_mpnn(
            pdb_path, all_aas,
            os.path.join(out_root, "baseline_20"),
            seed=base_seed,
            temp=0.25,
        )
        target_score = baseline_score * baseline_tolerance
        w.writerow(["BASELINE", 0, "ALL20", baseline_score, "ALL20", 0, "BASELINE"])
        flush()

        # ── 2. Determine starting seed ───────────────────────────────────────
        if DYNAMIC_SEEDING:
            # Compute per-AA composition from baseline designed sequences.
            # Pick the N most abundant as the backbone-specific seed.
            from collections import Counter as _Counter
            _aa_counts = _Counter()
            _all_set = set(all_aas)
            for _sc, _seq in baseline_pairs:
                for _ch in _seq:
                    if _ch in _all_set:
                        _aa_counts[_ch] += 1
            _top = _aa_counts.most_common()
            _seed_aas = [aa for aa, _ in _top[:DYNAMIC_SEED_SIZE]]
            current_set = sorted(_seed_aas)

            # Log the full composition ranking for reproducibility
            _comp_str = " ".join(f"{aa}:{ct}" for aa, ct in _top[:10])
            print(f"  [Dynamic seed] MPNN composition (top 10): {_comp_str}")
            print(f"  [Dynamic seed] Seeding with: {current_set}")

            seed_str = "".join(sorted(current_set))
            last_score, _ = run_mpnn(
                pdb_path, current_set,
                os.path.join(out_root, f"init_{seed_str}"),
                seed=base_seed + 1,
            )
            w.writerow(["DYNAMIC_SEED", 0, seed_str, last_score, seed_str, 0,
                         f"SEEDED_FROM_COMPOSITION({_comp_str})"])
            flush()

        else:
            # Original fixed-seed path
            seed_str = "".join(sorted(current_set))
            last_score, _ = run_mpnn(
                pdb_path, current_set,
                os.path.join(out_root, f"init_{seed_str}"),
                seed=base_seed + 1,
            )
            w.writerow(["INIT", 0, seed_str, last_score, seed_str, 0, "INITIAL"])
            flush()

            # ── 3. Optional hydrophobic pre-seeding ──────────────────────────
            if USE_HYDROPHOBIC_SEEDING:
                bh_aa, bh_score = None, 999.0
                for aa in ["V", "I", "L"]:
                    if aa in current_set:
                        continue
                    td = os.path.join(out_root, f"hydro_test_{aa}")
                    trial_set = current_set + [aa]
                    sc, sp = run_mpnn(pdb_path, trial_set, td,
                                      seed=base_seed + 2 + ["V","I","L"].index(aa))
                    w.writerow(["HYDROPHOBIC", 0, aa, sc,
                                "".join(sorted(current_set)), 0, "TRIAL"])
                    flush()
                    if sc < bh_score:
                        bh_score, bh_aa = sc, aa
                if bh_aa:
                    imp = (last_score - bh_score) / last_score if last_score > 0 else 0
                    w.writerow(["HYDROPHOBIC", 0, bh_aa, bh_score,
                                "".join(sorted(current_set)), imp * 100, "ACCEPTED"])
                    flush()
                    current_set.append(bh_aa)
                last_score = bh_score

        # ── 4. Greedy forward rounds ──────────────────────────────────────────
        round_counter = 0

        while len(current_set) < 20:
            round_counter += 1
            remaining = [aa for aa in all_aas if aa not in current_set]

            if last_score <= target_score and best_validated['sequence'] is not None:
                w.writerow(["OPEN", round_counter, "N/A", last_score,
                            "".join(sorted(current_set)), 0,
                            "STOPPED_TARGET_REACHED"])
                flush()
                break

            # ── Step 1: MPNN for all candidates ──────────────────────────────
            candidate_results = []

            for idx, aa in enumerate(remaining):
                trial_set = current_set + [aa]
                td = os.path.join(out_root, f"size{len(trial_set)}_test{aa}")
                sc, sp = run_mpnn(pdb_path, trial_set, td,
                                  seed=base_seed + 100 + round_counter * 20 + idx)
                w.writerow(["OPEN", round_counter, aa, sc,
                            "".join(sorted(current_set)), 0, "TRIAL"])
                flush()
                candidate_results.append((aa, sc, td, sp))

            if not candidate_results:
                break

            # ── Step 2: Quick ESMFold validation for every candidate ──────────
            print(f"\n  [Round {round_counter}] Quick-validating "
                  f"{len(candidate_results)} candidate alphabets "
                  f"({num_seqs_quick_validate} folds each)...")

            do_validate = (len(current_set) + 1) >= esm_start_after_n_aas

            validated = []
            for aa, sc, td, sp in candidate_results:
                if do_validate:
                    fold_dir  = os.path.join(td, "esmfold")
                    alph_str  = "".join(sorted(current_set + [aa]))
                    passed, metrics, best_info = validate_alphabet_with_esmfold(
                        current_set + [aa], sp, _ref_pdb, fold_dir,
                        num_seqs=num_seqs_quick_validate,
                        crystal_pdb=pdb_path,
                    )
                    def fmt(v, spec):
                        return f"{v:{spec}}" if v is not None else "N/A"
                    ew.writerow([
                        "OPEN", round_counter, alph_str,
                        fmt(metrics['mean_rmsd'],  '.2f'),
                        fmt(metrics['min_rmsd'],   '.2f'),
                        fmt(metrics['mean_tm'],    '.3f'),
                        fmt(metrics['max_tm'],     '.3f'),
                        fmt(metrics['mean_plddt'], '.1f'),
                        fmt(metrics['max_plddt'],  '.1f'),
                        metrics['num_pass_all'],
                        metrics['successful_folds'],
                        metrics['total_attempted'],
                        "PASSED" if passed else "FAILED",
                    ])
                    flush()
                    r_str = fmt(metrics['min_rmsd'],  '.2f')
                    t_str = fmt(metrics['max_tm'],    '.3f')
                    p_str = fmt(metrics['max_plddt'], '.1f')
                    print(f"    +{aa} ({alph_str}): "
                          f"minRMSD={r_str} maxTM={t_str} maxpLDDT={p_str} "
                          f"{'✓' if passed else '✗'}")
                else:
                    passed, metrics, best_info = False, create_empty_metrics(), create_empty_best()

                validated.append((aa, sc, td, passed, metrics, best_info))

            # ── Step 3: Select the best AA to add ────────────────────────────
            passing = [(aa, sc, td, m, bi)
                       for aa, sc, td, passed, m, bi in validated if passed]

            if passing:
                best_aa, best_score, best_trial_dir, best_metrics, best_info = max(
                    passing,
                    key=lambda x: (
                         x[3]['max_tm']   if x[3]['max_tm']   is not None else -999.0,
                        -x[3]['min_rmsd'] if x[3]['min_rmsd'] is not None else -999.0,
                    )
                )
                maybe_update_best(best_info, "".join(sorted(current_set + [best_aa])))

                alph_str = "".join(sorted(current_set + [best_aa]))
                if best_info['sequence']:
                    seq_file = os.path.join(best_trial_dir, "best_validated_sequence.txt")
                    with open(seq_file, 'w') as sf:
                        sf.write(f"Alphabet : {alph_str}\n")
                        if best_info['rmsd']  is not None:
                            sf.write(f"RMSD     : {best_info['rmsd']:.2f} Å\n")
                        if best_info['tm']    is not None:
                            sf.write(f"TM-score : {best_info['tm']:.3f}\n")
                        if best_info['plddt'] is not None:
                            sf.write(f"pLDDT    : {best_info['plddt']:.1f}\n")
                        sf.write(f"Sequence :\n{best_info['sequence']}\n")

                imp = ((last_score - best_score) / last_score
                       if last_score > 0 else 0)
                current_set.append(best_aa)
                last_score = best_score
                w.writerow(["OPEN", round_counter, best_aa, best_score,
                            "".join(sorted(current_set)), imp * 100,
                            "ACCEPTED_STOP_VALIDATED"])
                flush()
                tm_str = f"{best_metrics['max_tm']:.3f}" if best_metrics['max_tm'] is not None else "N/A"
                rm_str = f"{best_metrics['min_rmsd']:.2f}" if best_metrics['min_rmsd'] is not None else "N/A"
                print(f"  ✓ PASSED — stopping at {len(current_set)} AAs "
                      f"(+{best_aa}, TM={tm_str}, RMSD={rm_str} Å)")
                break

            else:
                # No candidate passed the TM gate. Pick the one with best TM
                # among those that produced any successful fold.
                structured = [
                    (aa, sc, td, m, bi)
                    for aa, sc, td, passed, m, bi in validated
                    if m['max_tm'] is not None
                ]

                if structured and do_validate:
                    best_aa, best_score, best_trial_dir, _, _ = max(
                        structured,
                        key=lambda x: (
                             x[3]['max_tm']   if x[3]['max_tm']   is not None else -999.0,
                            -x[3]['min_rmsd'] if x[3]['min_rmsd'] is not None else -999.0,
                        )
                    )
                    reason = "ACCEPTED_BEST_TM"
                else:
                    best_aa, best_score, best_trial_dir = min(
                        [(aa, sc, td) for aa, sc, td, _p, _m, _b in validated],
                        key=lambda x: x[1]
                    )
                    reason = "ACCEPTED"

                imp = ((last_score - best_score) / last_score
                       if last_score > 0 else 0)
                current_set.append(best_aa)
                last_score = best_score
                w.writerow(["OPEN", round_counter, best_aa, best_score,
                            "".join(sorted(current_set)), imp * 100, reason])
                flush()
                print(f"  ✗ No candidate passed — added +{best_aa} ({reason})")

        # ── 5. Full validation on the accepted alphabet ───────────────────────
        print(f"\n  [Full validation] {len(current_set)}-AA alphabet "
              f"({''.join(sorted(current_set))}) — {num_seqs_to_fold} folds...")
        _full_val_dir  = os.path.join(out_root, "full_validation")
        _full_fold_dir = os.path.join(_full_val_dir, "esmfold")
        _, _full_sp = run_mpnn(pdb_path, current_set, _full_val_dir,
                               seed=base_seed + 50_000)
        _full_passed, _full_metrics, _full_best = validate_alphabet_with_esmfold(
            current_set, _full_sp, _ref_pdb, _full_fold_dir,
            num_seqs=num_seqs_to_fold,
            crystal_pdb=pdb_path,
        )
        def _fmt(v, spec):
            return f"{v:{spec}}" if v is not None else "N/A"
        _alph_str = "".join(sorted(current_set))
        ew.writerow([
            "FULL_VALIDATION", round_counter, _alph_str,
            _fmt(_full_metrics['mean_rmsd'],  '.2f'),
            _fmt(_full_metrics['min_rmsd'],   '.2f'),
            _fmt(_full_metrics['mean_tm'],    '.3f'),
            _fmt(_full_metrics['max_tm'],     '.3f'),
            _fmt(_full_metrics['mean_plddt'], '.1f'),
            _fmt(_full_metrics['max_plddt'],  '.1f'),
            _full_metrics['num_pass_all'],
            _full_metrics['successful_folds'],
            _full_metrics['total_attempted'],
            "PASSED" if _full_passed else "FAILED",
        ])
        flush()
        if _full_best['sequence']:
            maybe_update_best(_full_best, _alph_str)
        print(f"  Full validation: "
              f"minRMSD={_fmt(_full_metrics['min_rmsd'], '.2f')} Å  "
              f"maxTM={_fmt(_full_metrics['max_tm'], '.3f')}  "
              f"maxpLDDT={_fmt(_full_metrics['max_plddt'], '.1f')}  "
              f"{'✓ PASSED' if _full_passed else '✗ FAILED'}")

        # ── 6. Backward pruning ───────────────────────────────────────────────
        pruned_set, n_removed = current_set, 0
        if (DO_BACKWARD_PRUNING
                and best_validated['sequence'] is not None
                and len(current_set) > PRUNING_MIN_SIZE):
            pruned_set, n_removed = backward_pruning_pass(
                pdb_path, current_set, pdb_out_dir, rep, base_seed,
                protected_aas=[a for a in (starting_alphabet or [])
                               if a not in STARTING_ALPHABET],
                reference_pdb=_ref_pdb,
            )
            w.writerow(["PRUNING", round_counter, "BACKWARD_PASS",
                        last_score, "".join(sorted(pruned_set)),
                        n_removed, "PRUNED"])
            flush()

    elapsed = round((datetime.now() - t_start).total_seconds())

    return {
        'pdb':                   pdb_name,
        'replicate':             rep,
        'final_alphabet':        "".join(sorted(current_set)),
        'pruned_alphabet':       "".join(sorted(pruned_set)),
        'alphabet_size':         len(current_set),
        'pruned_alphabet_size':  len(pruned_set),
        'aas_removed_pruning':   n_removed,
        'final_mpnn_score':      last_score,
        'baseline_mpnn_score':   baseline_score,
        'num_rounds':            round_counter,
        'elapsed_seconds':       elapsed,
        'best_sequence':         best_validated['sequence'],
        'best_seq_alphabet':     best_validated['alphabet'],
        'best_seq_rmsd':         best_validated['rmsd'],
        'best_seq_tm':           best_validated['tm'],
        'best_seq_plddt':        best_validated['plddt'],
    }

# =============================================================================
# BATCH RUNNER
# =============================================================================

os.makedirs(BATCH_OUTPUT_ROOT, exist_ok=True)
all_results = []
flagged_domains = []   # domains whose native sequence fails the ESMFold round-trip

print(f"\nStarting batch processing...")
print("=" * 60)

for pdb_idx, (domain_name, pdb_path) in enumerate(PDB_CHUNK):

    print(f"\n{'='*60}")
    print(f"Domain {pdb_idx+1}/{len(PDB_CHUNK)}: {domain_name}  "
          f"(global {chunk_start+pdb_idx+1}/{len(PDB_LIST)})")
    print(f"  File: {pdb_path}")
    print("=" * 60)

    if not os.path.isfile(pdb_path):
        print(f"  ERROR: file not found — {pdb_path}, skipping.")
        all_results.append({'pdb': domain_name, 'replicate': 0,
                            'status': 'skipped_file_not_found'})
        continue

    pdb_name = domain_name

    pdb_out_dir = os.path.join(BATCH_OUTPUT_ROOT, pdb_name)
    os.makedirs(pdb_out_dir, exist_ok=True)

    # ── Input-type handling by filename length ────────────────────────────────
    # > 4 chars (e.g. "1bxoA02")  -> pre-processed CATH domain: use the file
    #                                 as-is (already single-chain/single-domain).
    # == 4 chars (e.g. "6WVS")    -> standard RCSB PDB ID: keep ONLY chain A,
    #                                 materialised to a clean single-chain file
    #                                 that every downstream step then uses.
    orig_input_path = pdb_path
    if len(domain_name) > 4:
        input_type = "processed CATH domain (used as-is)"
        print(f"  Input type: {input_type}")
    else:
        chainA_path = os.path.join(pdb_out_dir, f"{pdb_name}_chainA.pdb")
        n_chainA = extract_chain_to_file(pdb_path, "A", chainA_path)
        if n_chainA == 0:
            print(f"  ERROR: chain A not found / no standard residues in "
                  f"{pdb_path}, skipping.")
            all_results.append({'pdb': domain_name, 'replicate': 0,
                                'status': 'skipped_no_chain_A'})
            continue
        pdb_path   = chainA_path   # reassign so all downstream code uses chain A
        input_type = f"standard PDB ID — chain A only ({n_chainA} residues)"
        print(f"  Input type: {input_type}")

    # ── Record input file info ────────────────────────────────────────────────
    domain_info_path = os.path.join(pdb_out_dir, "domain_info.txt")
    if not os.path.exists(domain_info_path):
        native_seq_for_info = get_native_sequence(pdb_path)
        with open(domain_info_path, 'w') as f:
            f.write(f"Domain: {domain_name}\n")
            f.write(f"Original source file: {orig_input_path}\n")
            f.write(f"Input handling: {input_type}\n")
            f.write(f"PDB used by pipeline: {pdb_path}\n")
            if native_seq_for_info:
                f.write(f"Sequence length: {len(native_seq_for_info)} aa\n")
                f.write(f"Sequence: {native_seq_for_info}\n")

    # ── Structural requirements analysis ─────────────────────────────────────
    req      = analyze_pdb_requirements(pdb_path)
    if DYNAMIC_SEEDING:
        pdb_start = []   # determined per-replicate from MPNN composition
    else:
        pdb_start = sorted(set(STARTING_ALPHABET) | set(req['required_aas']))

    print(f"\n  [Structure analysis]")
    print(f"    Disulfides : {req['n_disulfides']}")
    print(f"    Cis-Pro    : {req['cis_pro_count']}")
    for note in req['notes']:
        print(f"    → {note}")
    if req['required_aas']:
        print(f"    Required AAs pre-seeded: {req['required_aas']}")
        print(f"    Starting alphabet for this domain: {pdb_start}")
    else:
        print(f"    No required AAs beyond default — starting with {pdb_start}")

    # Persist to domain_info.txt
    with open(domain_info_path, 'a') as f:
        f.write(f"\nStructural requirements:\n")
        f.write(f"  Disulfides: {req['n_disulfides']}\n")
        f.write(f"  Cis-Pro residues: {req['cis_pro_count']}\n")
        for note in req['notes']:
            f.write(f"  {note}\n")
        f.write(f"  Starting alphabet: {''.join(pdb_start)}\n")

    # ── Verify native sequence folds reasonably with ESMFold ─────────────────
    NATIVE_PLDDT_MIN = 50.0
    NATIVE_TM_MIN    = 0.5

    print(f"\n  [Sanity check] Folding native sequence with ESMFold...")
    native_seq = get_native_sequence(pdb_path)
    esmfold_reliable  = True    # False only on a HARD technical failure (no model)
    native_fold_path  = None
    domain_flag       = ""      # "" = clean; non-empty = ESMFold-mismatch reason
    native_tm_value   = None    # native ESMFold TM vs crystal, for the summary

    if native_seq:
        native_fold_path = os.path.join(pdb_out_dir, "native_esmfold.pdb")
        success, nat_plddt = fold_sequence_with_esmfold(native_seq, native_fold_path)
        if success:
            nat_rmsd_crystal = calculate_rmsd(pdb_path, native_fold_path)
            nat_tm_crystal   = calculate_tm_score(pdb_path, native_fold_path)
            native_tm_value  = nat_tm_crystal
            print(f"  Native fold: pLDDT={nat_plddt:.1f}  "
                  f"RMSD(vs crystal)={f'{nat_rmsd_crystal:.2f} Å' if nat_rmsd_crystal else 'N/A'}  "
                  f"TM(vs crystal)={f'{nat_tm_crystal:.3f}' if nat_tm_crystal else 'N/A'}")
            print(f"  (crystal/domain PDB is the RMSD/TM reference for all subsequent folds; "
                  f"native ESMFold fold reported above as ESMFold reliability check only)")

            fail_reasons = []
            if USE_PLDDT_GATE and nat_plddt < NATIVE_PLDDT_MIN:
                fail_reasons.append(f"pLDDT={nat_plddt:.1f} < {NATIVE_PLDDT_MIN}")
            if nat_tm_crystal is not None and nat_tm_crystal < NATIVE_TM_MIN:
                fail_reasons.append(f"TM={nat_tm_crystal:.3f} < {NATIVE_TM_MIN}")

            if fail_reasons:
                # CHANGED: do NOT discard the run. Flag it and continue so the
                # full alphabet search still runs; results are marked for review.
                domain_flag = "; ".join(fail_reasons)
                print(f"  ⚠ FLAGGED {pdb_name}: ESMFold native round-trip is weak "
                      f"({domain_flag}).")
                print(f"    Likely an IDP, NMR ensemble, multidomain crystal, or "
                      f"out-of-distribution fold — running anyway, results flagged.")
                with open(domain_info_path, 'a') as f:
                    f.write(f"\nFLAGGED: native ESMFold round-trip weak — {domain_flag}\n")
                    f.write(f"  (pipeline run anyway; treat structural validation "
                            f"results with caution)\n")
                with open(os.path.join(pdb_out_dir,
                                       "FLAGGED_native_esmfold_mismatch.txt"), 'w') as f:
                    f.write(f"Domain: {pdb_name}\n")
                    f.write(f"Native ESMFold vs crystal: {domain_flag}\n")
                    f.write(f"Native TM(vs crystal): "
                            f"{f'{nat_tm_crystal:.3f}' if nat_tm_crystal is not None else 'NA'}\n")
                    f.write("The full alphabet search WAS run. Because ESMFold does "
                            "not reproduce the native fold here, pass/fail of any "
                            "reduced-alphabet sequence may be unreliable — inspect "
                            "manually.\n")
                flagged_domains.append({
                    'pdb':       pdb_name,
                    'native_tm': nat_tm_crystal,
                    'reason':    domain_flag,
                })
            else:
                print(f"  ✓ ESMFold reliable for {pdb_name} — using crystal/domain PDB as RMSD/TM reference.")
        else:
            esmfold_reliable = False
            native_fold_path = None
            print(f"  ✗ SKIPPING {pdb_name}: Native fold failed entirely (no model produced).")
            all_results.append({
                'pdb': pdb_name, 'replicate': 0,
                'status': 'skipped_native_fold_failed',
            })
    else:
        print(f"  WARNING: Could not extract native sequence from {pdb_name}.")
        print(f"  Falling back to crystal/domain PDB as RMSD reference — validate results manually.")

    if not esmfold_reliable:
        continue

    # ── Replicates ────────────────────────────────────────────────────────────
    for rep in range(1, NUM_REPLICATES + 1):

        if rep_is_done(pdb_out_dir, rep):
            print(f"\n  --- Replicate {rep}/{NUM_REPLICATES} [SKIPPED — checkpoint found] ---")
            all_results.append({'pdb': pdb_name, 'replicate': rep,
                                 'status': 'resumed',
                                 'esmfold_native_flag': domain_flag,
                                 'esmfold_native_tm': native_tm_value})
            continue

        print(f"\n  --- Replicate {rep}/{NUM_REPLICATES} ---")
        result = run_single_replicate(pdb_path, pdb_name, rep, pdb_out_dir,
                                       starting_alphabet=pdb_start,
                                       reference_pdb=pdb_path)
        result['esmfold_native_flag'] = domain_flag
        result['esmfold_native_tm']   = native_tm_value
        if domain_flag and 'status' not in result:
            result['status'] = f'flagged_esmfold_native_mismatch: {domain_flag}'
        all_results.append(result)
        mark_rep_done(pdb_out_dir, rep)

        print(f"  Done ({result['elapsed_seconds']} s)  |  "
              f"Forward: {result['final_alphabet']} ({result['alphabet_size']} AAs)"
              + (f"  → Pruned: {result['pruned_alphabet']} "
                 f"({result['pruned_alphabet_size']} AAs)"
                 if result.get('aas_removed_pruning', 0) > 0 else ""))

# =============================================================================
# SUMMARIES
# =============================================================================

summary_fields = [
    'pdb', 'replicate', 'status',
    'esmfold_native_flag', 'esmfold_native_tm',
    'final_alphabet', 'pruned_alphabet',
    'alphabet_size', 'pruned_alphabet_size', 'aas_removed_pruning',
    'final_mpnn_score', 'baseline_mpnn_score', 'num_rounds', 'elapsed_seconds',
    'best_seq_alphabet', 'best_seq_rmsd', 'best_seq_tm', 'best_seq_plddt',
    'best_sequence',
]
summary_path = os.path.join(BATCH_OUTPUT_ROOT, "BATCH_SUMMARY.csv")
with open(summary_path, 'w', newline='') as f:
    wtr = csv.DictWriter(f, fieldnames=summary_fields, extrasaction='ignore')
    wtr.writeheader()
    for row in all_results:
        wtr.writerow({k: row.get(k, '') for k in summary_fields})

# Separate roll-up of domains whose native sequence failed the ESMFold
# round-trip. These were processed in full but their structural validation
# should be treated with caution.
flagged_path = os.path.join(BATCH_OUTPUT_ROOT, "FLAGGED_DOMAINS.csv")
if flagged_domains:
    with open(flagged_path, 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=['pdb', 'native_tm', 'reason'])
        wtr.writeheader()
        for row in flagged_domains:
            wtr.writerow(row)
    print(f"\n⚠ {len(flagged_domains)} domain(s) flagged (native ESMFold "
          f"round-trip weak): {[d['pdb'] for d in flagged_domains]}")
    print(f"  See {flagged_path}")

# Per-domain champion FASTA
pdb_reps      = defaultdict(list)
combined_lines = []

for r in all_results:
    if r.get('best_sequence'):
        pdb_reps[r['pdb']].append(r)

for pdb_name, reps in pdb_reps.items():
    def _champ_key(r):
        rmsd  = r.get('best_seq_rmsd')
        tm    = r.get('best_seq_tm')
        plddt = r.get('best_seq_plddt')
        return (
              tm    if tm    is not None else  0.0,
            -(rmsd  if rmsd  is not None else 999.0),
              plddt if plddt is not None else  0.0,
        )
    best = max(reps, key=_champ_key)
    seq  = best['best_sequence']
    alph = best.get('best_seq_alphabet', 'unknown')
    rmsd = best.get('best_seq_rmsd')
    tm   = best.get('best_seq_tm')
    pl   = best.get('best_seq_plddt')

    header = (f">{pdb_name}_rep{best['replicate']} "
              f"alphabet={alph} "
              f"RMSD={f'{rmsd:.2f}' if rmsd is not None else 'NA'} "
              f"TM={f'{tm:.3f}' if tm is not None else 'NA'} "
              f"pLDDT={f'{pl:.1f}' if pl is not None else 'NA'}")
    if best.get('esmfold_native_flag'):
        header += f" FLAGGED_native_esmfold_mismatch=[{best['esmfold_native_flag']}]"

    per_pdb_fasta = os.path.join(BATCH_OUTPUT_ROOT, pdb_name,
                                 f"{pdb_name}_best_sequence.fasta")
    with open(per_pdb_fasta, 'w') as f:
        f.write(header + "\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")

    combined_lines += [header] + [seq[i:i+80] for i in range(0, len(seq), 80)] + [""]

combined_fasta = os.path.join(BATCH_OUTPUT_ROOT, "ALL_BEST_SEQUENCES.fasta")
with open(combined_fasta, 'w') as f:
    f.write("\n".join(combined_lines))

print(f"\n{'='*60}")
print(f"CHUNK {CHUNK_INDEX} COMPLETE")
print(f"{'='*60}")
print(f"  Domains processed : {len(PDB_CHUNK)}")
print(f"  Runs completed    : {len([r for r in all_results if 'elapsed_seconds' in r])}")
print(f"  Output directory  : {BATCH_OUTPUT_ROOT}")
print(f"  Summary CSV       : {summary_path}")
print(f"  Combined FASTA    : {combined_fasta}")
print(f"{'='*60}")
print(f"\nTo run the next chunk, set:  CHUNK_INDEX = {CHUNK_INDEX + 1}")
print(f"To resume this chunk:        RESUME_FROM = \"{BATCH_OUTPUT_ROOT}\"")
