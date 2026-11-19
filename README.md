# RedAA finder — minimal amino-acid alphabet search

Finds the smallest amino-acid alphabet from which MPNN can design sequences
that fold back onto a reference crystal structure — optionally while keeping a
binding site or catalytic residues intact.

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

**MSI batch** — `slurm/prepare_msi.sh` once on a login node, then
`slurm/redaa_array.slurm`: an array job over chunks of a domain panel, with
preemption-safe checkpointing. See [Running on MSI](#running-on-msi).

Both import the same `redaa/` package, so results are comparable.

---

## Running on MSI

The order matters more than the commands do: **everything that needs the
network happens on a login node**, because compute nodes usually cannot reach
the internet, and because ten array tasks each discovering a 25 GB model
download is not a plan.

### 1. Build the environment (once)

`torch` has to match the CUDA module the batch script loads, so this one step
is yours — installing the wrong build is the slowest possible way to find out.

```bash
module load conda                  # or your site's equivalent
conda create -y -n redaa python=3.11
conda activate redaa
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA module
pip install -r requirements.txt
```

### 2. Stage everything offline-unfriendly (login node, once)

```bash
REDAA_PRESET=tev bash slurm/prepare_msi.sh
```

That script is idempotent and does the four things a compute node cannot:
installs the **fold model's own** dependencies (ESMFold2 loads with
`trust_remote_code=True`, so it imports `biotite`, `rdkit`, `omegaconf` and
others that are not this package's dependencies), clones LigandMPNN and
downloads the checkpoint for **each** model type, pre-caches the ~25 GB of
ESMFold2 weights into a shared `HF_HOME`, and fetches your input structures.
Use `REDAA_FETCH='1ubq_A, cath:1cukA01'` instead of `REDAA_PRESET` for your own
accessions, or copy prepared PDBs into `INPUT_PDB_DIR` yourself.

### 3. Prove it works on one GPU node, interactively

```bash
srun --partition=a100-4 --gres=gpu:a100:1 --time=0:40:00 --cpus-per-task=4 --mem=48g --pty bash
```

```bash
conda activate redaa
export HF_HOME=/scratch.global/$USER/hf MPNN_REPO=/scratch.global/$USER/LigandMPNN
export INPUT_PDB_DIR=/scratch.global/$USER/redaa_inputs
export SEARCH_MODE=site_aware SITE_PRESET=tev
python run_redaa.py --preflight
```

**Do not submit the array until this passes once.** Preflight does one of
everything in a few minutes — loads both models, featurises, designs with the
site held fixed, folds, measures, and folds the native sequence — and names
the first thing that breaks. Finding the same problem through the queue costs
a wait per attempt.

One thing to watch for here: the model downloads a chemical-component data
file (~400 MB) on its **first fold**, not at load, so step 2 does not cover
it. Running preflight interactively warms it. If preflight fails on a download
instead, your compute nodes have no outbound network — note what it tried to
fetch and that can be staged too.

### 4. Submit

```bash
sbatch --export=ALL slurm/redaa_array.slurm
```

`--export=ALL` matters: the site settings you exported above are what make it
a site-aware run. The script prints `--print-config` before loading any model,
so the first lines of the log tell you what it is actually about to do.

Edit the `#SBATCH` header for your allocation — **partition names, GPU type
strings and account names are site-specific and change**, so check `sinfo` and
your group's allocation rather than trusting the values committed here.

### What the batch script assumes

- **`HF_HUB_OFFLINE=1` by default.** Step 2 cached the weights; set it to `0`
  only if you know your nodes have network.
- **`MPNN_CHECKPOINT` is deliberately unset**, so the engine picks the file
  matching `MPNN_MODEL_TYPE`. The checkpoints are not interchangeable.
- **A per-task fold cache.** SQLite in WAL mode expects POSIX locking that a
  parallel filesystem may not provide faithfully across nodes, so a shared
  cache can hang or corrupt rather than merely miss. Sharing saves real work —
  ten replicates of one domain fold many of the same sequences — so set
  `FOLD_CACHE_PATH` to one path *if* you have confirmed locking works there.
- **Preemption is survivable.** `--signal=B:USR1@300` and `MAX_WALL_SECONDS`
  both trigger a clean stop: the replicate in flight finishes its bookkeeping,
  summaries are written, and the task exits 2. Resubmit the same array task
  and it resumes — completed replicates are skipped.

### A smaller first run

```bash
python run_redaa.py --test --preset tev
```

One domain, one replicate, reduced budgets. Note that `--test` rewrites the
budgets *after* the environment is read, so the run records what it actually
used in `RESOLVED_CONFIG.txt` in its output directory — trust that over the
variables you exported.

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

Open `colab/RedAA_Colab.ipynb` (Runtime → Change runtime type → GPU). Nine
steps: install, GPU check, Drive, inputs, **mode and site**, settings, run,
results, diagnose.

What it handles that the batch job does not need to:

- **GPU roulette.** A free session is usually a T4, which is pre-Ampere and has
  **no bfloat16** — and ESMFold2's ESM-C backbone wants bf16 by default. The
  default `FOLD_PRECISION="auto"` now asks the device and picks fp16 on such a
  GPU, printing which it chose and why, so this is handled rather than
  diagnosed. The GPU-check cell still reports it up front and suggests a
  smaller ESMFold2 variant or an L4/A100 runtime if you want the accuracy
  back. It also caps `FOLD_MAX_RESIDUES` from the
  VRAM it actually finds (320 at T4 scale, 512 at L4, 1024 at A100).
- **Sessions that end without warning.** The Drive cell points `OUTPUT_ROOT`,
  `FOLD_CACHE_PATH` and `HF_HOME` at Drive. Completed replicates are
  checkpointed and every fold is cached, so re-running after a disconnect
  resumes; without Drive it starts over.
- **Every run leaves one attachable report.** Step 7 tees its output to
  `<outputs>/../logs/run_*.log` and writes `report_*.txt` beside it — whether
  the run finishes or crashes, with the traceback included. That file carries
  the environment, the resolved settings, the results so far and the console
  log, so it is the one thing worth attaching when asking about a failure.
  Colab truncates long cell output and loses it on a disconnect; the file does
  not. `rc.capture_run_log()` and `rc.write_report()` are the API.
- **A Hugging Face token is optional but worth having.** ESMFold2 is public
  and downloads anonymously, which is what the
  `You are sending unauthenticated requests` warning means — it is a notice,
  not an error. Anonymous traffic is rate limited, though, and the weights are
  25 GB, which a fresh runtime re-fetches. Make a **read-only** token at
  `huggingface.co/settings/tokens`, store it as the Colab secret `HF_TOKEN`
  (key icon in the left sidebar), and every cell picks it up. It is read from
  the secret store, never written into the notebook, never printed, and
  deliberately not among the variables the run report dumps — that file is
  built to be attached to bug reports. On MSI, export `HF_TOKEN` before
  running `prepare_msi.sh`; never put it in a job script.
- **Model weights are kept off Drive on purpose.** safetensors reads weights
  by memory-mapping them, and Drive's FUSE mount serves neither mmap nor
  reliable multi-gigabyte writes, so a model cached there fails to load with
  `SafetensorError: Error while deserializing header: header too large`.
  Only outputs and the fold cache go to Drive; re-downloading weights after a
  restart is the cheaper problem. Step 9's `REPAIR_HF_CACHE` validates every
  cached weight file's header and deletes the broken ones.
- **Changing the GPU type restarts the runtime**, so switching T4 → A100 wipes
  `/content` like any other restart and Step 1 has to run again. Step 9 of the
  notebook (`rc.print_diagnosis()`) reports the environment, every package
  version, the resolved settings and a pass/fail list naming the fix for each
  problem — start there when a run fails.
- **Runtime restarts, which wipe `/content`.** Every cell after the installer
  starts by re-establishing the checkout, `sys.path` and the environment, and
  re-clones the repo if it has vanished — so a restart produces a cell that
  says what happened and keeps going, not a bare
  `ModuleNotFoundError: No module named 'redaa'`. pip packages and the MPNN
  checkpoint go with `/content` too, so the message points back at Step 1;
  Step 1 itself verifies imports, the checkpoint size and `transformers`
  before reporting success, so a partial install cannot look like a good one.
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

**Checkpoints are per model type and are not interchangeable.** `protein_mpnn`
loads `proteinmpnn_v_48_020.pt`, `ligand_mpnn` loads
`ligandmpnn_v_32_010_25.pt` and `soluble_mpnn` loads `solublempnn_v_48_020.pt`
(no underscore before `mpnn` — that is upstream's spelling). Leave
`MPNN_CHECKPOINT` unset and the engine picks the right one from
`MPNN_MODEL_TYPE`; pointing a `ligand_mpnn` run at the ProteinMPNN file fails
at load with a tensor-shape error that never mentions the checkpoint, so that
case is detected and named explicitly.

`tmtools` is optional but recommended: it provides exact TM-align. Without it
the pipeline falls back to its own seed-refined TM-score, which agrees closely
but is slower per comparison. LigandMPNN is used as a source checkout, not a
package — clone it and run its `get_model_params.sh`, then set `MPNN_REPO`.

**The fold model brings its own dependencies.** ESMFold2 loads with
`trust_remote_code=True`, so its implementation is downloaded with the weights
and imports packages this repository cannot know about — `biotite`, `rdkit`,
`omegaconf` and others. A missing one surfaces as `ModuleNotFoundError` from
inside the model, on the first fold, after the weights have downloaded.
`redaa.colab.model_dependency_plan()` reads the model card's own
`requirements.txt` and reports what is absent; Step 1 installs it, preflight
checks it before folding, and a fold that fails this way names the `pip
install` line. Only *missing* packages are installed — a card's floors for
torch, numpy or transformers are reported and left alone, because replacing
those under a kernel that has already imported them is the worse failure.
`requirements.txt` carries the same list for an offline or batch install.

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

## Preflight: fail in a minute, not in an hour

```bash
python run_redaa.py --preflight
```

Loads both models and does **one of everything** on the shortest input --
featurise the backbone, one MPNN design, one design with the site held fixed,
one fold, the metrics, the site geometry -- then stops with a pass/fail table.
In Colab it is Step 6b.

This exists because of where the failures actually were. The three bugs that
reached a real GPU run were all contract mismatches at the LigandMPNN or
ESMFold2 boundary: a key `featurize` reads that `parse_PDB` never writes, a
checkpoint belonging to a different model type, a ligand model given no
ligand. None of them could be caught by a stub -- a stub agrees with whatever
this code believes -- and all of them surfaced either on the first design call
or forty minutes into a search.

Preflight hits every one of those points once, in the order the real run hits
them, and names the first thing that broke.

It then folds the **native** sequence and reports TM, RMSD and pLDDT against
the reference, plus the site geometry that fold achieves. That is a different
question from the ones above — not "does the machinery work" but "would the
answers mean anything":

- if ESMFold2 cannot reproduce the reference from its own native sequence, no
  reduced-alphabet design's pass or fail on that domain can be trusted;
- the native fold's site geometry is the **ceiling** for any design, so a
  `SITE_MAX_*` threshold tighter than it is unreachable for reasons that have
  nothing to do with the alphabet.

Both are reported as warnings rather than failures — the pipeline is not
broken — and both are repeated in the summary so they cannot scroll away. The
native fold is not wasted work: the cache is keyed on sequence plus model and
sampler settings, so the real run's own native check is a cache hit.

## Preserving binding sites and catalytic residues

A minimal folding alphabet is not a protein. The residues that do the work —
a catalytic cysteine, the arginines that grip a triphosphate — are exactly the
ones a five-letter alphabet discards, and a catalytic cysteine has no
substitute. `SEARCH_MODE` adds the second question: the smallest alphabet that
folds **and** keeps the site.

### The two modes answer different questions

| `SEARCH_MODE` | question |
|---|---|
| `alphabet_only` | What is the smallest alphabet that folds? (the original search) |
| `site_aware` | What is the smallest alphabet for a protein that *still has this site*? |
| `scaffold_then_site` | Is the minimal folding alphabet compatible with this site — and if not, what did the site cost? |

`site_aware` fixes the site's positions from round zero, so every candidate
alphabet is judged on fold and site together. `scaffold_then_site` runs the
original site-blind search, then grafts the site back on; if that breaks the
fold it grows the alphabet until it holds, and reports how many letters the
site cost.

They can disagree, and **the disagreement is the result worth having**: a
site-blind search is free to spend the site's positions on folding, so the
graft can fail where `site_aware` would have routed around the problem. Run
both on the same domain and the difference is a direct measure of how much the
function constrains the alphabet.

### Writing a site spec

In author (PDB) numbering — the numbering in the paper and in the entry.
`redaa/sites.py` is the only code that speaks it, and converts once, against
the *prepared* file, since preparation drops residues with incomplete
backbones and that shifts every later index.

| token | meaning |
|---|---|
| `H46 D81 C151` | those residues, asserted to be His/Asp/Cys |
| `46` | residue 46, identity not asserted |
| `A/H46`, `A:46` | an explicit chain |
| `46A` | an insertion code |
| `range:148-151` | an inclusive span |
| `ligand:ATP` | every residue within 4.5 Å of any ATP atom |
| `ligand:ATP@6.0` | the same at a 6 Å cutoff |

Naming the expected residue is optional and worth doing every time: it turns a
numbering mismatch into an error instead of a silently wrong run. Naming a
*ligand* derives the pocket from coordinates, which beats transcribing a list
and is specific to the structure in hand. A ligand has to survive cleaning
first — `SITE_KEEP_LIGANDS=ATP`, or `--preset atp`, since everything
heteroatom is stripped by default.

**You do not have to know the ligand's code in advance.** Fetching a structure
reports the bindable components it contains, so the code you need is in the
output of the command you were already running:

```
  ok    4BS0_A: chain A, 299 residues  [1 incomplete dropped]
        ligand: this entry also contains 6NT (16 atoms) -- keep one with
                SITE_KEEP_LIGANDS to design against it or derive a pocket from it
```

Solvent, cryoprotectants and lone ions are left out of that list, and a copy
bound to a different chain of the crystal is not advertised as this chain's.
Asking for a code the entry does not have says which ones it does.

```bash
# See the two verified test cases
python run_redaa.py --list-presets

# Fetch one and look at the site before spending any GPU time
python run_redaa.py --preset tev --fetch-only --input-dir ./inputs
python run_redaa.py --preset tev --preview-site --input-dir ./inputs

# Then run it
SEARCH_MODE=site_aware python run_redaa.py --preset tev --input-dir ./inputs
SEARCH_MODE=scaffold_then_site python run_redaa.py --preset atp --input-dir ./inputs
```

`--preview-site` is worth making a habit. A site spec is the one setting that
can be wrong in a way the run does not notice — right numbering for a
different entry, a ligand stripped during cleaning, a contact shell that
swallowed half the protein — and all of it is visible in a second against the
actual file.

### Do the site's residues count as alphabet letters?

`SITE_RESIDUE_POLICY` decides, and the two answers are both publishable
framings of different claims:

- **`exempt`** (default) — site positions are held at native identity and are
  *not* alphabet letters. A result reads "5 letters plus 6 exempt positions".
  Keeps the folding question clean.
- **`in_alphabet`** — the site's residue types are forced into the alphabet
  and protected from pruning, so the reported alphabet is one a real protein
  could be written in end to end.

- **`measure_only`** — the site is neither fixed nor exempt. MPNN puts
  whatever the alphabet allows there, and the site is gated on **geometry
  alone**; the identity check is off, because the design chose its own
  residues.

`exempt` can flatter a result — a "3-letter alphabet" with twenty exempt
positions is not a 3-letter protein — so the exempt count and residue types
are always reported beside the alphabet, and a site covering more than a
quarter of the chain warns.

### Ligand-shaped design without preserving any identity

`MPNN_MODEL_TYPE=ligand_mpnn` and the site machinery are independent, so the
ligand can shape a design through atom context with nothing held native:

```bash
SEARCH_MODE=alphabet_only SITE_KEEP_LIGANDS=6NT MPNN_MODEL_TYPE=ligand_mpnn   python run_redaa.py --input-dir ./inputs
```

That folds only — the ligand is in the prepared PDB and LigandMPNN conditions
on it, but **nothing checks the pocket**, so there is no evidence the ligand
context changed anything. For the version with evidence, keep the site and set
the policy to `measure_only`:

```bash
SEARCH_MODE=site_aware SITE_RESIDUE_POLICY=measure_only   SITE_SPEC="ligand:6NT@4.0" SITE_KEEP_LIGANDS=6NT   MPNN_MODEL_TYPE=ligand_mpnn python run_redaa.py --input-dir ./inputs
```

Now the design picks its own residues in the pocket, and the pocket still has
to be in the right place and the right shape. That is a different and arguably
purer question than fixing the native residues: not "can this alphabet carry
*these* residues" but "can it build a pocket for this ligand at all".
`measure_only` cannot be combined with `scaffold_then_site`, which has nothing
to graft.

### How preservation is measured

On the **backbone frame**, with CB *derived* from N–CA–C rather than read.
That is deliberate: it works for glycine, it does not depend on a predicted
rotamer (the least reliable part of any folding model's output) or on which
altloc a crystal listed first, and it treats reference and prediction
identically — so a deviation means the backbone moved, not that two side
chains were modelled differently. It is the convention motif-scaffolding work
uses, and it is why these numbers are comparable between a crystal structure
and an ESMFold2 model.

| column | superposition | answers |
|---|---|---|
| `Site_BB_RMSD` | whole structure | is the site in the right *place*? |
| `Site_Local_RMSD` | site only | is the site's internal *geometry* intact? |
| `Site_CB_Dev` | none | worst error in the site's internal CB–CB distance matrix |

A site can pass one and fail another, and which one fails tells you what
happened — a correctly-shaped site in the wrong place is a different failure
from a site that has been pulled apart. All three are rigid-body invariant and
all three are recorded whether or not they gate (`SITE_GATE=0` records without
gating).

**The fold gate and the site gate are separate, and a verdict says which
bound.** A candidate line reads

```
+R (ACDGLPRT): minRMSD=1.40 maxTM=0.960 maxpLDDT=84.8 siteBB=2.41 siteLocal=0.84 cbDev=1.90 fail[site]
```

so a design that folds beautifully while losing the site is distinguishable at
a glance from one that never folded. `fail[TM]`, `fail[site]`, `fail[TM and
site]` and `fail[no single design passed both]` are the four cases. The last
one is worth knowing about: both gates are applied **per fold**, while the
reported numbers are each the best across folds, so a candidate can show a
passing TM from one design and a passing site from another and still fail
because no single design managed both.

**These are a necessary condition, not a sufficient one.** They say the
scaffold puts the functional residues where they belong. Whether the chemistry
works is a wet-lab question, and nothing here predicts catalysis.

### What LigandMPNN is given

With `MPNN_MODEL_TYPE=ligand_mpnn` and a retained ligand, design is conditioned
on the actual bound molecule's atoms. Atom context is switched on exactly when
the checkpoint is a ligand model: enabling it for `protein_mpnn` would ask the
network for a tensor it was never trained to read, and leaving it off for
`ligand_mpnn` would load ligand-aware weights and then hide the ligand from
them — the more insidious mistake, because it still runs. A `ligand_mpnn` run
on an input with no ligand atoms says so.

Site positions are held fixed three ways over: the chain mask tells the model
not to redesign them, the bias vector forbids anything else there, and the
sampled sequence is checked and corrected afterwards. The belt-and-braces is
deliberate — `chain_mask` semantics have moved between LigandMPNN revisions
and the installed one cannot be checked until it runs. If the correction ever
has work to do it is reported rather than passing silently, because it means
MPNN scores in that run are means over a different position set than intended.

### The verified test cases

Both were checked by computing them from the deposited coordinates, not taken
from memory.

| preset | structure | site | notes |
|---|---|---|---|
| `atp` | `5etn_A` — *E. coli* HPPK, 1.4 Å, 159 residues | `ligand:ATP@4.5` → 17 residues | **Monomeric**, which is the whole reason for this choice |
| `tev` | `1lvm_A:1-221` — TEV protease, 1.8 Å | `H46 D81 C151 D148 G149 Q150` | The Cys–His–Asp triad plus the specificity and oxyanion loop |

**Why 5ETN and not an obvious small ATP protein.** The small, high-resolution
single-chain ATP structures that a search turns up first — PII/SbtB, `6MM2`
and `2XBP` — are **trimers whose ATP pocket sits between subunits**. A
single-chain design can never reconstitute that site, so they are the wrong
test case however convenient they look. 5ETN is monomeric, so its pocket is
complete within the one chain being designed. Its 4.5 Å shell computes to 17
residues, including the R82/R84/R92/R110/R121 cluster on the triphosphate. The
entry also holds a pterin analogue (5RX) and Ca/Na — the divalent site is Ca
here, not the physiological Mg — and none of them are part of the ATP pocket,
so none are kept.

**TEV numbering.** Chain A of 1LVM runs −8..221; −8..−1 is a `GHHHHHHH`
purification tag, which would fold as a floppy terminus and drag the native
round-trip down for no scientific reason, so the preset takes 1–221 (221
residues, no gaps). In the deposited file His46 ND1–Asp81 OD2 is 3.09 Å and
His46 NE2–Cys151 SG is 4.02 Å, so the triad is intact and the numbering is the
one the spec is written against. Chains C/D/E are product peptide, not
protease.

### What the run refuses to do

Three things are checked before any search starts, because each would
otherwise produce a run that completes normally and means nothing:

- a spec whose asserted residue identities do not match the structure is
  **fatal**, not a warning — it means the numbering is not what the spec was
  written against;
- MPNN's residue order is compared against the site layer's, since a
  disagreement would fix the wrong residues at every position;
- a site position whose reference backbone is incomplete is **unmeasurable**
  rather than silently passing.

---

## Choosing and constraining the alphabet

`START_ALPHABET` sets where the forward search begins — `AG` by default, the
two smallest residues, which is what V9 used so results stay comparable. The
search only ever grows, so a larger seed asks a different and weaker question:
what that set needs *added*, not what the minimum is.

`EXCLUDE_AAS` bars residues outright. They are never seeded, never offered as
a candidate, and left out of the 20-residue baseline — that last part matters,
because the baseline is the reference every restricted alphabet is scored
against, and a 20-letter baseline would be a target no candidate could reach.

```bash
EXCLUDE_AAS=C python run_redaa.py                 # no cysteine, so no disulfides
START_ALPHABET=GS EXCLUDE_AAS=CWM python run_redaa.py
```

Three conflicts are reported rather than silently resolved:

- **Excluded and in the starting alphabet** — refused by `--print-config`
  before anything runs.
- **Excluded but chemically needed** — a structure with a disulfide asks for
  `C` to be protected. Config validation cannot know that, so the run says so
  per domain and **the exclusion wins**: you asked for it, and quietly
  reinstating the residue would make the reported alphabet a lie. The result
  is then a design under that constraint, not a reproduction of the native.
- **Excluded but needed by an `in_alphabet` site** — a contradiction, so that
  domain is skipped with the two ways to resolve it.
- **Excluded but held by an `exempt` site** — consistent as far as the
  *alphabet* goes, since an exempt residue is not a letter, but it puts the
  residue straight back into the **sequence**, at the positions that matter
  most. If the point is to remove a residue from the protein entirely, that is
  the opposite of what you wanted, so the run says which positions and how
  many, and suggests `measure_only`.

Whichever happens, the outcome is reported rather than inferred from the
settings: `BATCH_SUMMARY.csv` and `INVERSE_FOLD.csv` both carry
`excluded_present` and `excluded_count` for the sequence in hand. If you barred
a residue in order to eliminate it, that column is the thing to read.

## Inverse folding: turning an alphabet into candidates

The search answers *how small can the alphabet be* and reports one best
sequence per replicate. That is the wrong output for anything downstream — you
cannot order a point estimate.

```bash
python run_redaa.py --inverse-fold                # the alphabet the run found
python run_redaa.py --inverse-fold GLVAP          # or one you name
python run_redaa.py --inverse-fold --inverse-seqs 200 --inverse-folds 20
```

Designs many sequences from a fixed alphabet, folds a spread of them at
`FOLD_STEPS_FINAL`, and ranks them. With no argument it reads the alphabet the
run settled on — the pruned one if pruning removed anything, and the smallest
across replicates. Naming one explicitly is how you ask "what would one more
letter buy me?".

Everything the run was configured with still applies: the site is held exactly
as `SEARCH_MODE` and `SITE_RESIDUE_POLICY` say, and the fold cache means
sequences the search already folded cost nothing. Output is
`INVERSE_FOLD.csv` — every folded sequence with its own TM, RMSD, pLDDT and
site metrics — and `INVERSE_FOLD.fasta`. In Colab it is Step 8b.

These are candidates, not results. A passing gate means the backbone and the
site geometry look right, which is necessary and nowhere near sufficient.

## Configuration

Everything is environment-driven; `python run_redaa.py --print-config` shows
resolved values. The ones you are most likely to touch:

| Variable | Default | Meaning |
|---|---|---|
| `INPUT_PDB_DIR` | `input_pdbs` | Directory of single-chain, standard-residue domain PDBs |
| `OUTPUT_ROOT` | `outputs` | Where run directories are created |
| `MPNN_REPO` | — | LigandMPNN checkout (needs `data_utils.py`, `model_utils.py`) |
| `MPNN_CHECKPOINT` | `$MPNN_REPO/model_params/proteinmpnn_v_48_020.pt` | Weights |
| `MPNN_MODEL_TYPE` | `protein_mpnn` | Also `soluble_mpnn`, `ligand_mpnn`; selects the checkpoint |
| `MPNN_CHECKPOINT` | by model type | Override only to use your own weights |
| `SEARCH_MODE` | `alphabet_only` | Also `site_aware`, `scaffold_then_site` |
| `SITE_PRESET` | — | `atp` or `tev`; fills in spec, ligands and model type |
| `SITE_SPEC` | — | Site in author numbering, e.g. `H46 D81 C151` or `ligand:ATP@4.5` |
| `SITE_CHAIN` | — | Default chain for spec tokens that do not name one |
| `SITE_KEEP_LIGANDS` | — | Components to retain through cleaning, e.g. `ATP` |
| `SITE_CONTACT_MAX` | `0` (all) | Cap a ligand contact shell, closest first |
| `SITE_RESIDUE_POLICY` | `exempt` | Also `in_alphabet`, `measure_only` |
| `SITE_GATE` | `1` | `0` records site metrics without gating on them |
| `SITE_MAX_BB_RMSD` | `1.5` | Å, site backbone after global superposition; `0` disables |
| `SITE_MAX_LOCAL_RMSD` | `1.0` | Å, site backbone after site-only superposition |
| `SITE_MAX_CB_DEV` | `1.5` | Å, worst site CB–CB distance error |
| `SITE_REQUIRE_IDENTITY` | `1` | Fail if a site position lost its native residue |
| `FOLD_MODEL_ID` | `Synthyra/ESMFold2-Fast` | Any ESMFold2 variant |
| `FOLD_PRECISION` | `auto` | `auto` asks the GPU: bf16 where supported, else fp16; fp32 on CPU. Also `bf16`, `fp16`, `fp32` |
| `FOLD_MAX_RESIDUES` | `1024` | Longer sequences are skipped, not run until they die |
| `FOLD_STEPS_SCREEN` | `50` | Diffusion steps for screening folds |
| `FOLD_STEPS_FINAL` | `200` | Diffusion steps for reported folds |
| `CHUNK_INDEX` | `$SLURM_ARRAY_TASK_ID` | Which chunk of domains to run |
| `CHUNK_SIZE` | `10` | Domains per chunk |
| `NUM_REPLICATES` | `10` | Independent searches per domain |
| `START_ALPHABET` | `AG` | Where the forward search begins |
| `EXCLUDE_AAS` | — | Residues barred entirely, e.g. `C` or `CM` |
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
python run_redaa.py --self-test  # all nine suites, 269 tests

python tests/test_structure.py   # metrics: parsing, Kabsch, RMSD, TM-score
python tests/test_search.py      # selection, pruning, chemical requirements
python tests/test_fetch.py       # spec parsing, download ladder, mmCIF, cleaning
python tests/test_colab.py       # GPU recommendations, input gathering, results
python tests/test_notebook.py    # notebook structure, forms, restart bootstrap
python tests/test_sites.py       # site specs, resolution, metrics, presets
python tests/test_mpnn.py        # LigandMPNN's featurise contract, fixed positions
python tests/test_slurm.py       # the batch scripts, against the settings they drive
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
redaa/sites.py               functional sites: specs, pockets, metrics, presets
redaa/fetch.py               RCSB/CATH/AlphaFold fetching, mmCIF, input cleaning
redaa/folding.py             ESMFold2-Fast engine
redaa/mpnn.py                in-process LigandMPNN engine
redaa/foldcache.py           SQLite fold cache
redaa/search.py              selection, validation, backward pruning
redaa/pipeline.py            batch and replicate orchestration
redaa/colab.py               GPU-aware defaults, Drive wiring, input gathering
colab/RedAA_Colab.ipynb      the notebook
colab/build_notebook.py      generates the notebook; edit this, not the JSON
slurm/prepare_msi.sh         login-node staging: deps, weights, checkpoints, inputs
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
- **`ligand_mpnn` needs an actual ligand.** It cannot run without one --
  LigandMPNN looks for nearest neighbours among zero ligand atoms and fails on
  an empty dimension -- so this is refused up front, at site preview and again
  before featurising. A catalytic-site case such as `tev` wants
  `protein_mpnn`.
- **A preserved site geometry is not a working site.** Every site number here
  is backbone geometry: it says the scaffold puts the functional residues
  where they belong, which is necessary and nowhere near sufficient. Nothing
  in this pipeline predicts catalysis or binding affinity, and a design that
  passes every gate is a candidate to test, not a result.
- **Exempt site positions are not alphabet letters.** Under the default
  `SITE_RESIDUE_POLICY=exempt` a reported alphabet describes the *designable*
  positions only, and the real protein still needs the exempt residues'
  types. Compare against an `in_alphabet` run before quoting a number.
- **Scores are not comparable across modes.** With fixed positions, MPNN
  scores are means over the designed positions only — the right denominator
  within a site-aware run, but not the same quantity an `alphabet_only` run
  reports.
- **TM-score is still the only fold gate.** RMSD is computed for every sequence and
  ranks candidates that all clear the gate, but a fixed Ångström cutoff across
  a 50–250 residue panel is biased against the longer domains, so it does not
  decide pass/fail. For domains under ~40 residues TM normalisation is
  unreliable regardless, and RMSD and pLDDT are the more informative numbers.
