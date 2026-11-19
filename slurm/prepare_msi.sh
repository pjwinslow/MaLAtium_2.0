#!/bin/bash
# Stage everything a compute node cannot fetch for itself.
#
# Run this ON A LOGIN NODE (or a data-transfer node), once. Compute nodes on
# most clusters have no outbound internet, and even where they do, ten array
# tasks each pulling 25 GB of ESMFold2 weights is not a plan. Everything that
# needs the network happens here; the array job then runs offline.
#
#   bash slurm/prepare_msi.sh
#
# Idempotent: re-running skips what is already in place. Nothing here needs a
# GPU -- the one step that does (a first fold, which warms the model's own
# data files) is an interactive srun, printed at the end.

set -euo pipefail

say() { printf '\n== %s\n' "$*"; }

# ---------------------------------------------------------------------------
# Where things go. Override any of these before running.
# ---------------------------------------------------------------------------
# /home quotas on MSI are small and the model cache is tens of gigabytes, so
# everything bulky defaults to global scratch. Check your allocation's own
# recommended path -- scratch is periodically purged, so project space is the
# better home for inputs and outputs you want to keep.
SCRATCH="${REDAA_SCRATCH:-/scratch.global/$USER}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf}"
MPNN_REPO="${MPNN_REPO:-$SCRATCH/LigandMPNN}"
INPUT_PDB_DIR="${INPUT_PDB_DIR:-$SCRATCH/redaa_inputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/redaa_outputs}"
ENV_NAME="${REDAA_ENV:-redaa}"
FOLD_MODEL_ID="${FOLD_MODEL_ID:-Synthyra/ESMFold2-Fast}"
# What to fetch. A preset name ("tev", "atp") or any spec list --fetch accepts.
FETCH="${REDAA_FETCH:-}"
PRESET="${REDAA_PRESET:-}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p "$HF_HOME" "$INPUT_PDB_DIR" "$OUTPUT_ROOT" logs

say "paths"
cat <<EOF
  repo            $REPO_DIR
  conda env       $ENV_NAME
  HF_HOME         $HF_HOME
  MPNN_REPO       $MPNN_REPO
  INPUT_PDB_DIR   $INPUT_PDB_DIR
  OUTPUT_ROOT     $OUTPUT_ROOT
EOF

# ---------------------------------------------------------------------------
# 1. Python environment
# ---------------------------------------------------------------------------
# Deliberately not created here. torch has to match the CUDA module you will
# load in the batch script, and only you know which that is -- installing the
# wrong build is the slowest possible way to find out. See the README.
say "checking the python environment"
if ! python -c "import torch" 2>/dev/null; then
  cat <<'EOF'
  torch is not importable in the active environment.

  Create the environment first, with a torch build matching the CUDA module
  you will load in redaa_array.slurm:

    module load conda                     # or your site's equivalent
    conda create -y -n redaa python=3.11
    conda activate redaa
    # pick the index-url for your CUDA module from pytorch.org:
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements.txt

  Then re-run this script with that environment active.
EOF
  exit 1
fi
python - <<'EOF'
import torch
print("  torch %s (cuda build %s)" % (torch.__version__, torch.version.cuda))
EOF

say "checking this pipeline's own dependencies"
python run_redaa.py --self-test >/dev/null && echo "  self-test passed (no GPU needed)"

# ---------------------------------------------------------------------------
# 2. The fold model's own dependencies
# ---------------------------------------------------------------------------
# ESMFold2 loads with trust_remote_code=True, so its implementation arrives
# with the weights and imports packages this repo does not depend on. A
# missing one surfaces as ModuleNotFoundError from inside the model, on the
# first fold -- i.e. inside the array job, after the queue wait.
say "checking the fold model's declared dependencies"
python - "$FOLD_MODEL_ID" <<'EOF'
import subprocess
import sys

from redaa.colab import describe_dependency_plan, model_dependency_plan

plan = model_dependency_plan(sys.argv[1])
print(describe_dependency_plan(plan))
if plan["install"]:
    print("  installing: %s" % " ".join(plan["install"]))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *plan["install"]],
        check=True,
    )
    print("  done")
EOF

# ---------------------------------------------------------------------------
# 3. LigandMPNN checkout and weights
# ---------------------------------------------------------------------------
say "LigandMPNN"
if [ ! -d "$MPNN_REPO/.git" ]; then
  git clone --depth 1 https://github.com/dauparas/LigandMPNN "$MPNN_REPO"
else
  echo "  already cloned at $MPNN_REPO"
fi

# Only the checkpoints the three usable model types resolve to. get_model_params.sh
# pulls about fifteen; a ligand_mpnn run needs different weights from a
# protein_mpnn run and they are not interchangeable.
mkdir -p "$MPNN_REPO/model_params"
python - "$MPNN_REPO" <<'EOF'
import os
import sys
import urllib.request

from redaa.mpnn import CHECKPOINT_BASE_URL, DEFAULT_CHECKPOINTS

params = os.path.join(sys.argv[1], "model_params")
for name in sorted(set(DEFAULT_CHECKPOINTS.values())):
    path = os.path.join(params, name)
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        print("  have %s" % name)
        continue
    url = "%s/%s" % (CHECKPOINT_BASE_URL, name)
    print("  downloading %s" % name)
    urllib.request.urlretrieve(url, path)
    print("    %.0f MB" % (os.path.getsize(path) / 1e6))
EOF

# ---------------------------------------------------------------------------
# 4. Model weights
# ---------------------------------------------------------------------------
# ESMFold2-Fast is ~25 GB. Pull it once, here, into the shared HF_HOME so the
# array tasks can run with HF_HUB_OFFLINE=1.
# A token is optional -- the model is public -- but anonymous traffic is rate
# limited, which is felt at 25 GB. Export HF_TOKEN before running this script
# to use one. Never put it in the script or in a job file: those get committed
# and shared.
if [ -n "${HF_TOKEN:-}" ]; then
  echo "  using the HF_TOKEN from the environment"
else
  echo "  no HF_TOKEN set; downloading anonymously (slower, rate limited)"
fi

say "pre-caching $FOLD_MODEL_ID into $HF_HOME (this is the slow part)"
python - "$FOLD_MODEL_ID" <<'EOF'
import sys

from huggingface_hub import snapshot_download

model_id = sys.argv[1]
path = snapshot_download(model_id)
print("  cached at %s" % path)
EOF

# ---------------------------------------------------------------------------
# 5. Inputs
# ---------------------------------------------------------------------------
say "inputs"
if [ -n "$PRESET" ]; then
  python run_redaa.py --preset "$PRESET" --fetch-only --input-dir "$INPUT_PDB_DIR"
elif [ -n "$FETCH" ]; then
  python run_redaa.py --fetch "$FETCH" --fetch-only --input-dir "$INPUT_PDB_DIR"
else
  n=$(find "$INPUT_PDB_DIR" -maxdepth 1 -name '*.pdb' 2>/dev/null | wc -l)
  echo "  no REDAA_PRESET or REDAA_FETCH given; $INPUT_PDB_DIR holds $n .pdb file(s)"
  echo "  (copy your own domain PDBs there, or re-run with REDAA_PRESET=tev)"
fi

# ---------------------------------------------------------------------------
# What is left, and why it cannot happen here
# ---------------------------------------------------------------------------
cat <<EOF

== staged. One step remains, and it needs a GPU.

The model downloads a chemical-component data file (~400 MB) on its FIRST
FOLD, not at load, so caching the weights above does not cover it. Warm it --
and confirm the whole path works -- on a GPU node, interactively, while the
network is still reachable:

  srun --partition=\${PARTITION:-a100-4} --gres=gpu:a100:1 --time=0:40:00 \\
       --cpus-per-task=4 --mem=48g --pty bash
  conda activate $ENV_NAME
  export HF_HOME=$HF_HOME MPNN_REPO=$MPNN_REPO
  export INPUT_PDB_DIR=$INPUT_PDB_DIR OUTPUT_ROOT=$OUTPUT_ROOT
  export SEARCH_MODE=site_aware SITE_PRESET=tev
  python run_redaa.py --preflight

Preflight does one of everything -- load both models, featurise, design with
the site held fixed, fold, measure, and fold the native sequence -- in a few
minutes. If it passes there, the array job will run.

If it fails on a download, your compute nodes have no outbound network: note
what it tried to fetch and we will stage that too. Do not submit the array
until preflight has passed once.

Then:
  sbatch --export=ALL slurm/redaa_array.slurm
EOF
