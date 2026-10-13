# RedAA finder — minimal amino-acid alphabet search

Finds the smallest amino-acid alphabet from which MPNN can design sequences
that fold back onto a reference crystal structure.

This is a rewrite of `New_V9_619.py` that swaps both model components and
restructures the search so it runs well on MSI GPUs:

| | V9 | this version |
|---|---|---|
| Folding | ESMFold v1 (`facebook/esmfold_v1`) | **ESMFold2-Fast** (`Synthyra/ESMFold2-Fast`) |
| Design | ProteinMPNN via `protein_mpnn_run.py` subprocess | **LigandMPNN**, in-process |
| Metrics | BioPython, re-parsing PDBs from disk | NumPy, coordinates in memory |
| Fold reuse | none | SQLite cache, shared across replicates and array tasks |

The science is unchanged: forward greedy alphabet growth with per-round
structural validation, then backward pruning, gated on TM-score ≥ 0.51.
Output directory layout and all CSV column headers are identical to V9, so
existing analysis scripts keep working.

---

## Two ways to run it

**Google Colab** — `colab/RedAA_Colab.ipynb`. Fetches structures by accession or
takes your own uploads, picks defaults to match whichever GPU the session got,
and keeps results on Drive so a disconnect resumes instead of restarting.
See [Running on Colab](#running-on-colab).

**MSI batch** — `slurm/redaa_array.slurm`. An array job over chunks of a domain
panel, with a shared fold cache and preemption-safe checkpointing.

Both import the same `redaa/` package, so results are comparable.

---

## Quick start (MSI)

```bash
# 1. Point at your LigandMPNN checkout and your domain PDBs
export MPNN_REPO=/projects/standard/seeligb/shared/LigandMPNN
export INPUT_PDB_DIR=/projects/standard/seeligb/shared/.../All_inputs_260612

# 2. Check the metrics layer (no GPU, no model weights, runs anywhere)
python run_redaa.py --self-test

# 3. Confirm the resolved settings
python run_redaa.py --print-config

# 4. Smoke test on one domain (~5 min on an A100)
python run_redaa.py --test

# 5. Submit the real thing
sbatch slurm/redaa_array.slurm
```

## Inputs: fetch or supply

The MSI runs take a directory of pre-cleaned, single-chain CATH domain PDBs.
When you have an accession instead, `--fetch` turns one into the other:

```bash
python run_redaa.py --fetch "1ubq_A, 2gb1_A, cath:1cukA01, af:P0DTC2" \
                    --input-dir ./inputs --fetch-only
python run_redaa.py --input-dir ./inputs
```

| spec | meaning |
|---|---|
| `1UBQ` | RCSB entry, largest protein chain |
| `1ubq_A` | chain A (`:` and `.` also work as the separator) |
| `1ubq_A:1-76` | chain A, residues 1–76 |
| `cath:1cukA01` | CATH domain, already domain-bounded |
| `af:P0DTC2` | AlphaFold DB model for a UniProt accession |
| `pdbe:1ubq` | the same entry via PDBe instead of RCSB |
| `https://…/x.pdb` | any direct URL |
| `./my/domain.pdb` | a local file — still cleaned and chain-split |

Lists may be comma-, space- or newline-separated, with `#` comments. Unusable
entries are reported and skipped rather than aborting the batch. Entries too
large for the PDB format fall back to mmCIF, which is parsed directly. Downloads
are cached, so re-running costs nothing.

Whatever the source, each input is reduced to **one chain, standard residues,
primary altloc, and only residues with a complete N–CA–C backbone**. That last
filter matters: MPNN needs those three atoms and a raw RCSB entry routinely has
residues that lack them. `--chain` sets the chain for every spec that does not
name one. The prepared-input table reports dropped residues and chain gaps so
you can see what the cleaning did.

## Running on Colab

Open `colab/RedAA_Colab.ipynb` (Runtime → Change runtime type → GPU). Seven
cells: install, GPU check, Drive, inputs, settings, run, results.

What it handles that the batch job does not need to:

- **GPU roulette.** A free session is usually a T4, which is pre-Ampere and has
  **no bfloat16** — and ESMFold2's ESM-C backbone loads in bf16 by default. The
  GPU-check cell reports this up front and suggests `FOLD_PRECISION="fp16"`, a
  smaller ESMFold2 variant, or an L4/A100 runtime, rather than letting you
  discover it twenty minutes in. It also caps `FOLD_MAX_RESIDUES` from the
  VRAM it actually finds (320 at T4 scale, 512 at L4, 1024 at A100).
- **Sessions that end without warning.** The Drive cell points `OUTPUT_ROOT`,
  `FOLD_CACHE_PATH` and `HF_HOME` at Drive. Completed replicates are
  checkpointed and every fold is cached, so re-running after a disconnect
  resumes; without Drive it starts over.
- **Hours, not days.** Defaults drop to 3 replicates and 5 final folds. The
  gate is unchanged, so the answers mean the same thing — there are just fewer
  replicates behind them. `TEST_MODE` runs one domain first.
- **Re-running the run cell is cheap.** The loaded ESMFold2 and LigandMPNN
  models are handed back into the next run, so changing a threshold does not
  pay for a model load again. Changing the model id does reload, deliberately.

The notebook is generated from `colab/build_notebook.py` — edit that and
re-run it rather than hand-editing the JSON.

## Install

```bash
pip install -r requirements.txt
```

`tmtools` is optional but recommended: it provides exact TM-align. Without it
the pipeline falls back to its own seed-refined TM-score, which agrees closely
but is slower per comparison. LigandMPNN is used as a source checkout, not a
package — clone it and run its `get_model_params.sh`, then set `MPNN_REPO`.

ESMFold2-Fast needs a recent `transformers` and loads with
`trust_remote_code=True`, which pulls the FastPLMs implementation from the
model repository. Pre-cache the weights on a login node (`HF_HOME` pointing at
scratch) before submitting an array job, so twenty tasks do not all try to
download them at once.

---

## What was done for speed

Ordered by how much they actually save.

### 1. MPNN runs in-process, and the backbone is featurised once

V9 shelled out to `protein_mpnn_run.py` for **every candidate alphabet**:

```python
subprocess.run(f"python {mpnn_script} --pdb_path {pdb} --bias_AA_jsonl {bias} ...")
```

Each call paid for a fresh interpreter, `import torch`, a CUDA context, a
checkpoint load, a PDB parse and a featurisation — several seconds of fixed
cost for roughly 50 ms of GPU work — then wrote a FASTA file that the parent
read back and deleted. A single replicate makes 50–300 of those calls.

Now the checkpoint loads once per process and, more importantly, the backbone
is parsed and featurised **once per domain**. The backbone never changes during
a search; only the amino-acid bias vector does. Every candidate alphabet is
therefore a bias-tensor rebuild plus one `model.sample()` call on features
already resident on the GPU.

### 2. Nothing is folded twice

Every prediction goes through a SQLite cache keyed by sequence hash plus model
and sampler settings. Repeated sequences — common under restricted alphabets,
across the forward and backward passes, and across ten replicates of the same
domain — are free. Point `FOLD_CACHE_PATH` at a shared location and the whole
SLURM array shares one cache.

One file rather than one-per-prediction also matters on shared storage: tens of
thousands of small files is what makes a Lustre scratch directory crawl.

### 3. Screening folds are cheap folds

ESMFold2's structure module is a diffusion sampler, so `num_sampling_steps`
trades wall time against precision directly. Telling "right fold" from "wrong
fold" does not need the step count that a reported number needs, so screening
runs at `FOLD_STEPS_SCREEN=50` and the folds that end up in `BATCH_SUMMARY.csv`
run at `FOLD_STEPS_FINAL=200`.

### 4. Two-stage candidate screening

V9 folded 3 sequences for each of ~17 candidate residues every round: ~51 folds
to pick one winner. Here every candidate gets 1 fold, then only the plausible
winners — everything that already passed, plus the top `SCREEN_STAGE2_KEEP` by
TM-score — get the full budget.

The saving is only real because sequence selection was made **prefix-stable**:
`select_sequences(pairs, 1)` is exactly the first element of
`select_sequences(pairs, 3)`, so promoting a candidate costs only the
*additional* folds, the first one being a cache hit. V9's stratified sampler
picked a different set for a budget of 1 than for 3, so a cheap first look
could not have been reused.

### 5. Downward-closure pruning

If a 5-letter alphabet cannot encode the fold, no 4-letter subset of it can
either. Exhaustive backward pruning therefore only needs to test a subset whose
every one-larger superset passed — the Apriori argument. On the test case this
cuts 56 trials to 10 for an identical answer. Set
`PRUNING_DOWNWARD_CLOSURE=0` for an unconditional sweep.

### 6. Metrics without disk or BioPython

V9 called BioPython's `PDBParser` four times per comparison (twice in
`calculate_rmsd`, twice more in `calculate_tm_score`), re-reading the reference
crystal file from shared storage every time, and wrote every prediction to a
PDB file first. Now the reference is parsed once per domain and cached, and
predicted coordinates are read straight out of the PDB text ESMFold2 returns,
in memory. Files are written only for structures worth keeping.

The TM-score fallback was also fixed rather than just moved. V9 computed it
from a single global superposition, which systematically *under*-reports
TM-score because one badly-placed terminal segment drags the fit off the
matched core. The replacement searches seed fragments and iteratively refines,
which is the TM-score algorithm proper. On a synthetic hinge case the old
approach reports 0.18 where the correct answer is 0.50.

### 7. Survives preemption

`SIGTERM`/`SIGUSR1` and a `MAX_WALL_SECONDS` budget both trigger a clean stop:
the replicate in flight finishes its bookkeeping, summaries are written, and
the process exits with status 2. Completed replicates are marked with an
atomically-written `COMPLETED` sentinel, so re-running the same command resumes
rather than redoing work.

### 8. Smaller things

- TF32 matmul and cuDNN autotuning on; `torch.inference_mode()` throughout.
- MPNN samples all sequences for an alphabet in one batched call (`MPNN_BATCH_SIZE`,
  default 25) instead of five batches of five, and halves the batch and retries
  on OOM rather than dying.
- Duplicate designs are collapsed before folding — at a six-letter alphabet
  MPNN repeats itself often, and each duplicate would otherwise cost a fold.
- `X` is biased out of sampling entirely. V9's `bias.jsonl` covered only the 20
  standard residues, leaving `X` samplable.
- CSVs flush per round, not per row.
- One domain's cached features are released before the next domain loads.

### Measured effect

On the synthetic end-to-end case in `tests/test_pipeline.py`, which has a known
3-letter answer, all three configurations agree on `GLV`:

| configuration | folds | pruning trials |
|---|---|---|
| V9-equivalent (3 folds × every candidate, no closure) | 91 | 10 |
| two-stage screening | 72 | 10 |
| two-stage + downward closure (default) | 54 | 4 |

That is a 41% fold reduction on a case that only reaches round 1 — the
two-stage saving compounds with every additional forward round, and none of
this counts the in-process MPNN change or the fold cache, which are the two
largest wins and cannot be measured without the real models.

---

## Configuration

Everything is environment-driven; `python run_redaa.py --print-config` shows
resolved values. The ones you are most likely to touch:

| Variable | Default | Meaning |
|---|---|---|
| `INPUT_PDB_DIR` | `input_pdbs` | Directory of single-chain, standard-residue domain PDBs |
| `OUTPUT_ROOT` | `outputs` | Where run directories are created |
| `MPNN_REPO` | — | LigandMPNN checkout (needs `data_utils.py`, `model_utils.py`) |
| `MPNN_CHECKPOINT` | `$MPNN_REPO/model_params/proteinmpnn_v_48_020.pt` | Weights |
| `MPNN_MODEL_TYPE` | `protein_mpnn` | Also `soluble_mpnn`, `ligand_mpnn` |
| `FOLD_MODEL_ID` | `Synthyra/ESMFold2-Fast` | Any ESMFold2 variant |
| `FOLD_STEPS_SCREEN` | `50` | Diffusion steps for screening folds |
| `FOLD_STEPS_FINAL` | `200` | Diffusion steps for reported folds |
| `CHUNK_INDEX` | `$SLURM_ARRAY_TASK_ID` | Which chunk of domains to run |
| `CHUNK_SIZE` | `10` | Domains per chunk |
| `NUM_REPLICATES` | `10` | Independent searches per domain |
| `MIN_TM_SCORE` | `0.51` | The acceptance gate |
| `USE_PLDDT_GATE` | `0` | pLDDT is logged either way |
| `SCREEN_FOLDS_STAGE1` | `1` | Folds per candidate in the cheap pass |
| `SCREEN_FOLDS_STAGE2` | `3` | Folds per promoted candidate |
| `SCREEN_STAGE2_KEEP` | `5` | How many candidates get promoted |
| `PRUNING_DOWNWARD_CLOSURE` | `1` | Apriori pruning of the exhaustive sweep |
| `FOLD_CACHE_PATH` | `<run_dir>/fold_cache.sqlite` | Set to a shared path across an array |
| `MAX_WALL_SECONDS` | `0` (off) | Stop and checkpoint before the wall clock |
| `AUTO_RESUME` | `1` | Resume the newest matching run directory |

To reproduce V9's screening behaviour exactly, set
`SCREEN_FOLDS_STAGE1=3 SCREEN_FOLDS_STAGE2=3 PRUNING_DOWNWARD_CLOSURE=0
FOLD_STEPS_SCREEN=200`.

## Tests

```bash
python run_redaa.py --self-test  # all five suites, 81 tests

python tests/test_structure.py   # metrics: parsing, Kabsch, RMSD, TM-score
python tests/test_search.py      # selection, pruning, chemical requirements
python tests/test_fetch.py       # spec parsing, download ladder, mmCIF, cleaning
python tests/test_colab.py       # GPU recommendations, input gathering, results
python tests/test_pipeline.py    # full orchestration against stub engines
```

All of them run without torch, a GPU, model weights or network access — HTTP is
stubbed in the fetch tests.

## Layout

```
run_redaa.py                 entry point (--test, --fetch, --self-test)
redaa/config.py              environment-driven configuration
redaa/structure.py           PDB parsing, Kabsch, RMSD, TM-score
redaa/chemistry.py           disulfide and cis-Pro detection
redaa/fetch.py               RCSB/CATH/AlphaFold fetching, mmCIF, input cleaning
redaa/folding.py             ESMFold2-Fast engine
redaa/mpnn.py                in-process LigandMPNN engine
redaa/foldcache.py           SQLite fold cache
redaa/search.py              selection, validation, backward pruning
redaa/pipeline.py            batch and replicate orchestration
redaa/colab.py               GPU-aware defaults, Drive wiring, input gathering
colab/RedAA_Colab.ipynb      the notebook
colab/build_notebook.py      generates the notebook; edit this, not the JSON
slurm/redaa_array.slurm      MSI array job template
reference/New_V9_619.py      the original script, for diffing
```

## Notes and caveats

- **ESMFold2-Fast's exact keyword arguments vary between releases.** The call is
  built by introspecting the bound method, so only supported keywords are
  passed. If your installed revision names things differently, `redaa/folding.py`
  is the only file to touch — `FoldEngine.load` resolves the fold and
  PDB-writing methods, and `_supported_kwargs` maps settings onto them.
- **pLDDT scale is detected, not assumed.** ESMFold v1 reported 0–1 and V9
  multiplied by 100; ESMFold2 releases have used both conventions, so
  `normalise_plddt` rescales on magnitude. The thresholds therefore keep meaning
  the same thing.
- **MPNN scores stay comparable with V9.** The in-process path reports mean
  per-residue negative log-likelihood from `get_score`, which is the same
  quantity ProteinMPNN wrote as `score=` in its FASTA headers. Lower is better.
  Note this is *not* LigandMPNN's `overall_confidence`, which is `exp(-loss)`
  and runs the other way.
- **Downward-closure pruning assumes monotonicity** — that removing a residue
  from an alphabet that fails cannot make it succeed. That is nearly always
  true and is the standard assumption, but it is an assumption; set
  `PRUNING_DOWNWARD_CLOSURE=0` if a particular result needs the unconditional
  sweep.
- **Fetched inputs are not the same as curated inputs.** `--fetch` picks the
  largest protein chain when you do not name one, and keeps the whole chain
  unless you give a residue range. A multidomain crystal chain fetched whole
  will usually fail the native round-trip check and be flagged — which is the
  system working, but it means a fetched panel needs a look at
  `FLAGGED_DOMAINS.csv` before you trust it. Domain-bounded sources
  (`cath:…`) avoid this.
- **TM-score is still the only gate.** RMSD is computed for every sequence and
  ranks candidates that all clear the gate, but a fixed Ångström cutoff across
  a 50–250 residue panel is biased against the longer domains, so it does not
  decide pass/fail. For domains under ~40 residues TM normalisation is
  unreliable regardless, and RMSD and pLDDT are the more informative numbers.
