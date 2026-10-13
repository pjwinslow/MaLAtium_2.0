#!/usr/bin/env python3
"""Generate colab/RedAA_Colab.ipynb.

The notebook is built from here rather than hand-edited so the cell sources
stay reviewable as plain Python and the JSON cannot drift into something Colab
refuses to open.  Regenerate with:

    python colab/build_notebook.py
"""

from __future__ import annotations

import json
import os

REPO_URL = "https://github.com/pjwinslow/MaLAtium_2.0"
BRANCH = "claude/esm-fold2-mpnn-optimization-brhedz"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    text = text.replace("__REPO_URL__", REPO_URL).replace("__BRANCH__", BRANCH)
    CELLS.append(("code", text.strip("\n")))


# ---------------------------------------------------------------------------

md(
    """
# RedAA finder — minimal amino-acid alphabet search

Finds the smallest amino-acid alphabet from which **LigandMPNN** can design
sequences that **ESMFold2-Fast** folds back onto a reference structure.

Inputs can be **fetched** (PDB / CATH / AlphaFold accessions) or **uploaded**
(your own PDB or mmCIF files) — or both.

### Before you start

1. **Runtime → Change runtime type → GPU.** An L4 or A100 is strongly
   preferred: ESMFold2's ESM-C backbone wants bfloat16, which a free-tier T4
   does not have. Step 2 checks this and tells you what to do about it.
2. **Mount Drive (step 3) if you care about the results.** Colab sessions end
   without warning. The pipeline checkpoints every replicate and caches every
   fold, so a session writing to Drive resumes where it left off — one that
   isn't starts over from nothing.

Run the cells in order. Steps 1–3 are once per session.
"""
)

code(
    """
#@title Step 1 — Install (once per session, ~3-5 min)
REPO_URL = "__REPO_URL__" #@param {type:"string"}
BRANCH   = "__BRANCH__" #@param {type:"string"}

import os, subprocess, sys

REPO_DIR = "/content/MaLAtium_2.0"
MPNN_DIR = "/content/LigandMPNN"
MPNN_CKPT_URL = "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt"

def sh(cmd):
    print("$ " + cmd)
    subprocess.run(cmd, shell=True, check=True)

# --- this repository -------------------------------------------------------
if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    sh("git -C " + REPO_DIR + " fetch -q --depth 1 origin " + BRANCH)
    sh("git -C " + REPO_DIR + " checkout -q FETCH_HEAD")
else:
    sh("git clone -q --depth 1 --branch " + BRANCH + " " + REPO_URL + " " + REPO_DIR)

# --- python dependencies ---------------------------------------------------
# Colab already ships torch and numpy. ESMFold2-Fast loads through
# transformers with trust_remote_code=True, which pulls the FastPLMs
# implementation out of the model repository itself.
sh("pip install -q --upgrade 'transformers>=4.44' accelerate huggingface_hub")
sh("pip install -q tmtools ml-collections prody")   # tmtools = exact TM-align

# --- LigandMPNN ------------------------------------------------------------
if not os.path.isdir(os.path.join(MPNN_DIR, ".git")):
    sh("git clone -q --depth 1 https://github.com/dauparas/LigandMPNN " + MPNN_DIR)

# LigandMPNN's get_model_params.sh pulls ~15 checkpoints; we need exactly one.
CKPT = os.path.join(MPNN_DIR, "model_params", "proteinmpnn_v_48_020.pt")
os.makedirs(os.path.dirname(CKPT), exist_ok=True)
if not os.path.exists(CKPT):
    sh("wget -q -O " + CKPT + " " + MPNN_CKPT_URL)

os.environ["MPNN_REPO"] = MPNN_DIR
os.environ["MPNN_CHECKPOINT"] = CKPT
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

print("\\nInstall complete.")
print("  repo        " + REPO_DIR)
print("  LigandMPNN  " + MPNN_DIR)
print("  checkpoint  %s (%.0f MB)" % (CKPT, os.path.getsize(CKPT) / 1e6))
"""
)

code(
    """
#@title Step 2 — Check the GPU, apply matching defaults, self-test
import importlib, subprocess, sys
sys.path.insert(0, "/content/MaLAtium_2.0")

import redaa.colab as rc
importlib.reload(rc)

gpu = rc.detect_gpu()
rec = rc.recommend(gpu)

for note in rec.notes:
    print("  " + note)
for warning in rec.warnings:
    print("\\n  !! " + warning)

applied = rc.apply_env(rec.env)
if applied:
    print("\\nDefaults applied for this GPU:")
    for key, value in applied.items():
        print("    %s=%s" % (key, value))
print("\\nFree disk: %.1f GB" % rc.disk_free_gb("/content"))

# Exercise everything that needs no GPU, so a broken install surfaces here
# rather than twenty minutes into a run.
print("\\nSelf-test (metrics, search, fetching, orchestration):")
result = subprocess.run(
    [sys.executable, "run_redaa.py", "--self-test"],
    capture_output=True, text=True, cwd="/content/MaLAtium_2.0",
)
import re
tail = [ln.strip() for ln in result.stdout.splitlines()
        if re.match(r"^\\d+/\\d+ passed", ln.strip()) or "SUITES" in ln]
for line in tail:
    print("  " + line)
if result.returncode != 0:
    print("\\n  !! self-test failed — rerun step 1 before going further")
    print(result.stdout[-3000:])
"""
)

code(
    """
#@title Step 3 — Mount Google Drive (optional, recommended)
#@markdown Keeps outputs, the fold cache and model weights across disconnects,
#@markdown so a reconnected session resumes instead of starting over.
USE_DRIVE = True #@param {type:"boolean"}
DRIVE_FOLDER = "redaa" #@param {type:"string"}

import os

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    env = rc.use_drive("/content/drive/MyDrive/" + DRIVE_FOLDER)
    print("\\nUsing Drive for:")
    for key, value in env.items():
        print("  %s=%s" % (key, value))
else:
    os.environ["OUTPUT_ROOT"] = "/content/redaa_outputs"
    print("Drive off — results live in /content and are lost on disconnect.")

print("\\nOUTPUT_ROOT = " + os.environ["OUTPUT_ROOT"])
"""
)

md(
    """
## Step 4 — Choose your inputs

**Fetch by accession.** One per line, or comma-separated:

| spec | meaning |
|---|---|
| `1UBQ` | RCSB entry, largest protein chain |
| `1ubq_A` | chain A (`:` and `.` also work as the separator) |
| `1ubq_A:1-76` | chain A, residues 1–76 |
| `cath:1cukA01` | CATH domain, already domain-bounded |
| `af:P0DTC2` | AlphaFold DB model for a UniProt accession |
| `pdbe:1ubq` | the same entry via PDBe instead of RCSB |
| `https://…/x.pdb` | any direct URL |

`#` starts a comment. Unusable entries are reported and skipped, so one typo
does not cost you the rest of the list. Entries too large for the PDB format
fall back to mmCIF automatically.

**Upload your own.** Set `INPUT_MODE` to `upload` and a file picker appears;
PDB and mmCIF both work. `drive_folder` reads every structure file in a folder
on your Drive. Already-clean single-domain files — like the CATH domains the
MSI runs use — pass straight through.

Either way, each input is reduced to **one chain, standard residues, primary
altloc, and only residues with a complete N–CA–C backbone**. That last filter
matters: MPNN needs those three atoms, and a raw RCSB entry routinely contains
residues that lack them.
"""
)

code(
    """
#@title Step 4 — Fetch and/or upload inputs
INPUT_MODE = "fetch" #@param ["fetch", "upload", "drive_folder", "both"]
PDB_SPECS = "1ubq_A\\n2gb1_A" #@param {type:"string"}
CHAIN_OVERRIDE = "" #@param {type:"string"}
DRIVE_INPUT_FOLDER = "redaa/inputs" #@param {type:"string"}

import os

INPUT_DIR = "/content/redaa_inputs"
files = []

if INPUT_MODE in ("upload", "both"):
    from google.colab import files as colab_files
    print("Choose one or more .pdb / .cif files:")
    uploaded = colab_files.upload()
    os.makedirs("/content/redaa_uploads", exist_ok=True)
    for name, blob in uploaded.items():
        path = os.path.join("/content/redaa_uploads", name)
        with open(path, "wb") as fh:
            fh.write(blob)
        files.append(path)

if INPUT_MODE == "drive_folder":
    folder = "/content/drive/MyDrive/" + DRIVE_INPUT_FOLDER
    if not os.path.isdir(folder):
        raise SystemExit(folder + " not found — mount Drive in step 3 first.")
    files.append(folder)

specs = PDB_SPECS if INPUT_MODE in ("fetch", "both") else ""

reports = rc.gather_inputs(
    specs=specs,
    files=files,
    input_dir=INPUT_DIR,
    chain=CHAIN_OVERRIDE.strip() or None,
)

print()
print(rc.describe_inputs(reports))
os.environ["INPUT_PDB_DIR"] = INPUT_DIR
"""
)

code(
    """
#@title Step 5 — Search settings
#@markdown Defaults are tuned for a Colab session (hours, not days). Raise
#@markdown `NUM_REPLICATES` toward 10 for MSI-comparable statistics if you
#@markdown have a long session.
NUM_REPLICATES = 3 #@param {type:"integer"}
MIN_TM_SCORE = 0.51 #@param {type:"number"}
NUM_SEQS_TO_FOLD = 5 #@param {type:"integer"}
FOLD_STEPS_SCREEN = 50 #@param {type:"integer"}
FOLD_STEPS_FINAL = 200 #@param {type:"integer"}
FOLD_MODEL_ID = "Synthyra/ESMFold2-Fast" #@param ["Synthyra/ESMFold2-Fast", "Synthyra/ESMFold2", "Synthyra/ESMFold2-600", "Synthyra/ESMFold2-300"]
FOLD_PRECISION = "auto" #@param ["auto", "bf16", "fp16", "fp32"]
DO_BACKWARD_PRUNING = True #@param {type:"boolean"}
MAX_WALL_MINUTES = 660 #@param {type:"integer"}
TEST_MODE = False #@param {type:"boolean"}
#@markdown `TEST_MODE` runs one domain, one replicate, reduced budgets. Use it
#@markdown once to confirm the whole path works before committing hours to it.

import os

_settings = {
    "NUM_REPLICATES": NUM_REPLICATES,
    "MIN_TM_SCORE": MIN_TM_SCORE,
    "NUM_SEQS_TO_FOLD": NUM_SEQS_TO_FOLD,
    "FOLD_STEPS_SCREEN": FOLD_STEPS_SCREEN,
    "FOLD_STEPS_FINAL": FOLD_STEPS_FINAL,
    "FOLD_MODEL_ID": FOLD_MODEL_ID,
    "FOLD_PRECISION": FOLD_PRECISION,
    "DO_BACKWARD_PRUNING": int(DO_BACKWARD_PRUNING),
    "MAX_WALL_SECONDS": MAX_WALL_MINUTES * 60,
    "CHUNK_SIZE": 1000,          # one Colab session is one chunk
    "AUTO_RESUME": 1,
}
for key, value in _settings.items():
    os.environ[key] = str(value)

# Print only what the pipeline actually reads, rather than prefix-matching
# os.environ and picking up unrelated Colab variables.
from redaa.config import ENV_VARS

print("Settings in effect:")
for key in ENV_VARS:
    if key in os.environ:
        print("  %s=%s" % (key, os.environ[key]))
"""
)

code(
    """
#@title Step 6 — Run the search
#@markdown Safe to re-run: finished replicates are skipped, cached folds are
#@markdown free, and the loaded models stay in memory between runs.
import importlib, sys, time

import redaa.config, redaa.pipeline
importlib.reload(redaa.config)
importlib.reload(redaa.pipeline)

cfg = redaa.config.RedAAConfig()
if TEST_MODE:
    cfg.apply_test_mode()
    print("*** TEST MODE: one domain, one replicate, reduced budgets ***\\n")

# Hand back the engines from an earlier run in this session so ESMFold2 is not
# reloaded. Pipeline reloads anyway if the model id changed.
_cached = globals().get("_REDAA_ENGINES") or {}
pipe = redaa.pipeline.Pipeline(
    cfg, folder=_cached.get("folder"), mpnn=_cached.get("mpnn")
)

_started = time.time()
exit_code = 1
try:
    exit_code = pipe.run()
finally:
    _REDAA_ENGINES = {"folder": pipe.folder, "mpnn": pipe.mpnn}
    RUN_DIR = pipe.run_dir

print("\\nExit code %d (%s)" % (
    exit_code,
    "complete" if exit_code == 0 else "stopped early but checkpointed",
))
print("Wall time %.1f min" % ((time.time() - _started) / 60))
print("Run directory: " + str(RUN_DIR))
"""
)

code(
    """
#@title Step 7 — Results
DOWNLOAD_ZIP = True #@param {type:"boolean"}

import os

run_dir = globals().get("RUN_DIR") or rc.latest_run_dir()
if not run_dir:
    raise SystemExit("No run directory found — run step 6 first.")
print("Run directory: " + run_dir + "\\n")

rows = rc.summary_rows(run_dir)
if rows:
    keep = ["pdb", "replicate", "final_alphabet", "alphabet_size",
            "pruned_alphabet", "pruned_alphabet_size",
            "best_seq_tm", "best_seq_rmsd", "best_seq_plddt", "status"]
    columns = [c for c in keep if c in rows[0]]
    try:
        import pandas as pd
        display(pd.DataFrame(rows)[columns])
    except Exception:
        for row in rows:
            print({k: row.get(k) for k in columns})
else:
    print("BATCH_SUMMARY.csv has no rows — did any replicate finish?")

fasta = os.path.join(run_dir, "ALL_BEST_SEQUENCES.fasta")
if os.path.exists(fasta) and os.path.getsize(fasta):
    print("\\n=== best sequence per domain ===")
    print(open(fasta).read())

flagged = os.path.join(run_dir, "FLAGGED_DOMAINS.csv")
if os.path.exists(flagged):
    print("=== flagged: ESMFold2 does not reproduce the native fold ===")
    print("Treat these domains' pass/fail calls with caution.")
    print(open(flagged).read())

if DOWNLOAD_ZIP:
    path = rc.zip_results(run_dir)
    print("\\nZipped to %s (%.1f MB)" % (path, os.path.getsize(path) / 1e6))
    try:
        from google.colab import files as colab_files
        colab_files.download(path)
    except ImportError:
        print("(not in Colab — the zip is on disk at the path above)")
"""
)

md(
    """
## Troubleshooting

**bf16 / `esmc_precision` errors on a T4.** The free-tier T4 is pre-Ampere and
has no bfloat16. Try `FOLD_PRECISION="fp16"` in step 5, or one of the smaller
`ESMFold2-600` / `ESMFold2-300` variants, or switch to an L4 or A100 runtime.

**CUDA out of memory.** Lower `FOLD_MAX_RESIDUES` — step 2 sets a default from
your GPU's VRAM — or use smaller domains. Over-long sequences are skipped
rather than run until they die, and MPNN halves its batch and retries on OOM.

**`Could not import LigandMPNN's data_utils`.** Step 1 did not finish. Check
that `/content/LigandMPNN/data_utils.py` and the checkpoint under
`model_params/` both exist.

**Session died mid-run.** If step 3 put things on Drive: re-run steps 1–3, then
step 6. Finished replicates are skipped and cached folds cost nothing, so it
picks up where it stopped. Without Drive the work is gone — that is the whole
reason to mount it.

**Nothing passes the gate.** Look at the flagged-domains table first. If
ESMFold2 cannot reproduce a domain's *native* fold, no reduced-alphabet result
for it means much; that is usually an IDP, an NMR ensemble, a multidomain
crystal, or a fold outside the model's training distribution.

**Slower than expected.** The first fold of a session includes the model
download and warm-up. After that, watch the `s/fold` figure and the cache hit
rate printed at the end of a run.

## Relationship to the MSI version

Same code, same gate, same output format — `redaa/` is imported directly, not
reimplemented, so a Colab run and a batch run are comparable. The
Colab-specific parts are input fetching (`redaa/fetch.py`), the GPU-aware
defaults and Drive wiring (`redaa/colab.py`), and the smaller budgets in
step 5. For the batch job and the full configuration reference see `README.md`
and `slurm/redaa_array.slurm` in the repository.

The same fetching works on the command line, if you would rather not use the
notebook at all:

```bash
python run_redaa.py --fetch "1ubq_A, 2gb1_A, cath:1cukA01" --input-dir ./inputs
python run_redaa.py --input-dir ./inputs
```
"""
)


# ---------------------------------------------------------------------------


def build() -> dict:
    cells = []
    for kind, source in CELLS:
        lines = source.splitlines(keepends=True)
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            # No cellView: "form" -- the form widgets still render from the
            # #@param comments, but the code stays visible, which is what you
            # want in a research tool you may need to tweak mid-session.
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": lines,
                }
            )
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "name": "RedAA_Colab.ipynb",
                "provenance": [],
                "toc_visible": True,
            },
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


if __name__ == "__main__":
    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "RedAA_Colab.ipynb"
    )
    notebook = build()
    with open(out, "w") as fh:
        json.dump(notebook, fh, indent=1)
        fh.write("\n")
    n_code = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
    print(f"wrote {out}: {len(notebook['cells'])} cells ({n_code} code)")
