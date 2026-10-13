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
# EDIT THESE for your MSI allocation:
#SBATCH --partition=a100-4
##SBATCH --account=seeligb

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load cuda/12.4 || true           # match whatever your conda env was built against

# Activate the environment that already has ESMFold2-Fast and LigandMPNN working.
source "${CONDA_PREFIX_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${REDAA_ENV:-redaa}"

# Keep model downloads and compile caches off /home (small quota) and on
# shared scratch, so array tasks share one copy instead of each fetching it.
export HF_HOME="${HF_HOME:-/scratch.global/$USER/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # set 1 once weights are cached
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch.global/$USER/inductor}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR"

# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------
export INPUT_PDB_DIR="${INPUT_PDB_DIR:?set INPUT_PDB_DIR to your domain PDB directory}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch.global/$USER/redaa_outputs}"

# LigandMPNN checkout and weights.
export MPNN_REPO="${MPNN_REPO:?set MPNN_REPO to your LigandMPNN checkout}"
export MPNN_CHECKPOINT="${MPNN_CHECKPOINT:-$MPNN_REPO/model_params/proteinmpnn_v_48_020.pt}"

# ESMFold2-Fast.
export FOLD_MODEL_ID="${FOLD_MODEL_ID:-Synthyra/ESMFold2-Fast}"
export FOLD_STEPS_SCREEN="${FOLD_STEPS_SCREEN:-50}"
export FOLD_STEPS_FINAL="${FOLD_STEPS_FINAL:-200}"

# Chunking: CHUNK_INDEX defaults to SLURM_ARRAY_TASK_ID.
export CHUNK_SIZE="${CHUNK_SIZE:-10}"
export NUM_REPLICATES="${NUM_REPLICATES:-10}"

# One shared fold cache across the whole array, so a sequence folded by task 3
# is free for task 7.  SQLite in WAL mode handles the concurrency; point this
# at a path all tasks can reach.
export FOLD_CACHE_PATH="${FOLD_CACHE_PATH:-$OUTPUT_ROOT/fold_cache_shared.sqlite}"
mkdir -p "$OUTPUT_ROOT"

# Stop ~10 minutes before the wall clock so the run checkpoints itself even if
# the SIGUSR1 above is missed.
export MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-42600}"

# Resume into the newest matching output directory rather than starting fresh.
export AUTO_RESUME="${AUTO_RESUME:-1}"

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "${SLURM_SUBMIT_DIR:-$PWD}"
# `set -e` would abort before we could inspect the status, so capture it here.
status=0
srun python run_redaa.py || status=$?

# Exit code 2 means "stopped early but checkpointed"; requeue to carry on.
if [ "$status" -eq 2 ]; then
  echo "checkpointed early; resubmit the same array task to resume"
fi
exit "$status"
