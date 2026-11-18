#!/bin/bash
#SBATCH --job-name=redaa
#SBATCH --array=0-9                      # one task per chunk; see CHUNK_SIZE below
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48g
#SBATCH --output=logs/redaa_%A_%a.out
#SBATCH --error=logs/redaa_%A_%a.err
# Ask Slurm to warn us 5 minutes before the wall clock runs out, and 'B:' sends
# it to the batch step rather than the job steps.  run_redaa.py catches
# SIGUSR1/SIGTERM, finishes the replicate in flight, checkpoints and exits, so
# a re-submitted job resumes instead of redoing the work.
#SBATCH --signal=B:USR1@300
#
# EDIT THESE for your MSI allocation. Partition names, GPU type strings and
# account names are site-specific and change -- check `sinfo` and your group's
# allocation rather than trusting the values below.
#SBATCH --partition=a100-4
##SBATCH --account=seeligb
#
# Run slurm/prepare_msi.sh on a login node first, and let --preflight pass once
# on a GPU node. Both are explained in the README's MSI section. This script
# assumes everything is already staged and runs with no network.

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load cuda/12.4 || true           # must match what your torch was built against

source "${CONDA_PREFIX_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${REDAA_ENV:-redaa}"

# Keep model downloads and compile caches off /home (small quota) and on
# shared scratch, so array tasks share one copy instead of each fetching it.
export HF_HOME="${HF_HOME:-/scratch.global/$USER/hf}"
# Offline by default. The weights are ~25 GB: ten array tasks discovering that
# for themselves is a bad day, and on most clusters the compute nodes cannot
# reach the internet anyway. prepare_msi.sh caches them; set this to 0 only if
# you know your nodes have outbound network and you want the model to update.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch.global/$USER/inductor}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR"

# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------
export INPUT_PDB_DIR="${INPUT_PDB_DIR:?set INPUT_PDB_DIR to your domain PDB directory}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch.global/$USER/redaa_outputs}"

# LigandMPNN checkout. MPNN_CHECKPOINT is deliberately NOT set: the engine
# resolves the checkpoint from MPNN_MODEL_TYPE, and the files are not
# interchangeable -- a ligand_mpnn run given the ProteinMPNN weights fails at
# load with a tensor-shape error that never mentions the checkpoint. Set it
# only to use weights of your own.
export MPNN_REPO="${MPNN_REPO:?set MPNN_REPO to your LigandMPNN checkout}"
export MPNN_MODEL_TYPE="${MPNN_MODEL_TYPE:-protein_mpnn}"

# ESMFold2-Fast. FOLD_PRECISION=auto asks the GPU: bf16 where supported,
# fp16 otherwise.
export FOLD_MODEL_ID="${FOLD_MODEL_ID:-Synthyra/ESMFold2-Fast}"
export FOLD_PRECISION="${FOLD_PRECISION:-auto}"
export FOLD_STEPS_SCREEN="${FOLD_STEPS_SCREEN:-50}"
export FOLD_STEPS_FINAL="${FOLD_STEPS_FINAL:-200}"
# Sequences longer than this are skipped rather than run until they OOM. 1024
# suits an 80 GB A100; lower it on a smaller card.
export FOLD_MAX_RESIDUES="${FOLD_MAX_RESIDUES:-1024}"

# ---------------------------------------------------------------------------
# Functional site (leave SEARCH_MODE=alphabet_only for a fold-only search)
# ---------------------------------------------------------------------------
#   alphabet_only       smallest alphabet that folds; no site
#   site_aware          site fixed and gated from round zero
#   scaffold_then_site  search blind, then graft the site back on
export SEARCH_MODE="${SEARCH_MODE:-alphabet_only}"
# A preset ("tev", "atp") fills in the spec, the ligands to keep and the MPNN
# model type. SITE_SPEC overrides it; see `python run_redaa.py --list-presets`.
export SITE_PRESET="${SITE_PRESET:-}"
export SITE_SPEC="${SITE_SPEC:-}"
export SITE_CHAIN="${SITE_CHAIN:-}"
export SITE_KEEP_LIGANDS="${SITE_KEEP_LIGANDS:-}"
export SITE_CONTACT_MAX="${SITE_CONTACT_MAX:-0}"
# exempt: site positions are held at native identity and are not counted as
# alphabet letters. in_alphabet: their residue types join the alphabet and are
# protected from pruning. These answer different questions -- see the README.
export SITE_RESIDUE_POLICY="${SITE_RESIDUE_POLICY:-exempt}"
export SITE_GATE="${SITE_GATE:-1}"
export SITE_MAX_BB_RMSD="${SITE_MAX_BB_RMSD:-1.5}"
export SITE_MAX_LOCAL_RMSD="${SITE_MAX_LOCAL_RMSD:-1.0}"
export SITE_MAX_CB_DEV="${SITE_MAX_CB_DEV:-1.5}"
export SITE_REQUIRE_IDENTITY="${SITE_REQUIRE_IDENTITY:-1}"

# ---------------------------------------------------------------------------
# Chunking and budgets
# ---------------------------------------------------------------------------
# CHUNK_INDEX defaults to SLURM_ARRAY_TASK_ID.
export CHUNK_SIZE="${CHUNK_SIZE:-10}"
export NUM_REPLICATES="${NUM_REPLICATES:-10}"

# Fold cache. Per-task by default: SQLite in WAL mode expects POSIX locking,
# and a parallel filesystem may not provide it faithfully across nodes -- a
# shared cache can then hang or corrupt instead of merely missing. The cache
# disables itself rather than failing a run, but a per-task file on scratch is
# the safe default. Set FOLD_CACHE_PATH to one shared path if you have
# confirmed locking works on that filesystem; the saving is real, because ten
# replicates of one domain fold many of the same sequences.
export FOLD_CACHE_PATH="${FOLD_CACHE_PATH:-$OUTPUT_ROOT/fold_cache_task${SLURM_ARRAY_TASK_ID:-0}.sqlite}"
mkdir -p "$OUTPUT_ROOT"

# Stop ~10 minutes before the wall clock so the run checkpoints itself even if
# the SIGUSR1 above is missed.
export MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-42600}"

# Resume into the newest matching output directory rather than starting fresh.
export AUTO_RESUME="${AUTO_RESUME:-1}"

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Settings are easy to get wrong and a bad combination wastes the whole
# allocation, so check them before the models load.
python run_redaa.py --print-config | sed 's/^/  /'

# `set -e` would abort before we could inspect the status, so capture it here.
status=0
srun python run_redaa.py || status=$?

# Exit code 2 means "stopped early but checkpointed"; requeue to carry on.
if [ "$status" -eq 2 ]; then
  echo "checkpointed early; resubmit the same array task to resume"
fi
exit "$status"
