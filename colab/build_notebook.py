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
import subprocess

REPO_URL = "https://github.com/pjwinslow/MaLAtium_2.0"


def _current_branch() -> str:
    """The branch this checkout is on, which is the branch to clone in Colab.

    This used to be a hardcoded constant, and it went stale exactly the way a
    hardcoded constant does: the notebook kept pointing at the branch it was
    first built on, so Step 1 cloned code that did not match the cells around
    it -- a notebook showing a functional-site step while importing a `redaa`
    that had never heard of one. Deriving it means the generated notebook
    always clones the branch it was generated from.

    ``REDAA_BRANCH`` overrides, for building a notebook aimed at a branch
    other than the current one (a release notebook pinned to ``main``).
    """
    override = os.environ.get("REDAA_BRANCH", "").strip()
    if override:
        return override
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=10,
        )
        name = out.stdout.strip()
    except Exception:  # noqa: BLE001
        name = ""
    if not name or name == "HEAD":
        # Detached HEAD, or no git at all. "main" is wrong often enough that
        # guessing silently would just recreate the original bug.
        raise SystemExit(
            "Cannot tell which branch to bake into the notebook: this is not a "
            "git checkout, or HEAD is detached.\n"
            "Set REDAA_BRANCH to the branch Colab should clone, e.g.\n"
            "  REDAA_BRANCH=main python colab/build_notebook.py"
        )
    return name


BRANCH = _current_branch()

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    text = text.replace("__REPO_URL__", REPO_URL).replace("__BRANCH__", BRANCH)
    CELLS.append(("code", text.strip("\n")))


# A Colab runtime restart wipes /content, which leaves every later cell
# importing a package that is no longer on disk -- the symptom is a bare
# "No module named 'redaa'" from a cell that worked ten minutes earlier.
# Each cell below therefore re-establishes its own footing before doing
# anything, so cells are runnable in any order and after any restart.
BOOTSTRAP = """
# --- bootstrap: re-establish the repo, sys.path and env (safe to re-run) ---
import os, subprocess, sys

_REPO = "/content/MaLAtium_2.0"
_MPNN = "/content/LigandMPNN"

if not os.path.exists(os.path.join(_REPO, "redaa", "__init__.py")):
    print("Repository is not on disk -- the Colab runtime was restarted,")
    print("which wipes /content. Re-cloning it now.")
    subprocess.run(["rm", "-rf", _REPO], check=False)
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", "--branch",
         globals().get("BRANCH", "__BRANCH__"),
         globals().get("REPO_URL", "__REPO_URL__"), _REPO],
        check=True,
    )
    print("Re-cloned. If a later cell fails on a missing package or model,")
    print("re-run Step 1 -- pip installs and the checkpoint were wiped too.")

if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)
os.environ.setdefault("MPNN_REPO", _MPNN)
# MPNN_CHECKPOINT is deliberately left unset: the engine picks the checkpoint
# that matches MPNN_MODEL_TYPE, and pinning one here would silently hand a
# ligand_mpnn run the ProteinMPNN weights.

import redaa.colab as rc
# -------------------------------------------------------------------------
"""


def code_with_bootstrap(text: str) -> None:
    """Emit a code cell whose #@title and #@param block stays on top.

    Colab only renders form widgets for #@param lines that precede other
    statements, so the bootstrap is spliced in after the parameter block
    rather than at the very top of the cell.
    """
    text = text.strip("\n")
    lines = text.splitlines()
    cut = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith(("#@", "#"))
            or "#@param" in line
            or not stripped
        ):
            cut = i + 1
        else:
            break
    head, tail = lines[:cut], lines[cut:]
    code("\n".join(head + BOOTSTRAP.strip("\n").splitlines() + [""] + tail))


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
MPNN_CKPT_BASE = "https://files.ipd.uw.edu/pub/ligandmpnn"

def sh(cmd, quiet_cmd=None):
    # quiet_cmd is what gets printed when the real command carries a token.
    print("$ " + (quiet_cmd or cmd))
    subprocess.run(cmd, shell=True, check=True)

# --- credentials, if this repository is private -----------------------------
# A private repo cannot be cloned anonymously. Colab's secret store (the key
# icon in the left sidebar) is the only safe place to put a token in a
# notebook: it is not saved in the .ipynb and not shared when the notebook is.
# Add a secret named GITHUB_TOKEN, enable it for this notebook, and use a
# fine-grained token scoped to this repository with Contents: Read.
GH_TOKEN = None
try:
    from google.colab import userdata
    GH_TOKEN = userdata.get("GITHUB_TOKEN")
except Exception:
    pass

def auth_url():
    # REPO_URL with the token spliced in, or unchanged if there is none.
    # (A docstring here would end the generator's own triple-quoted cell.)
    if GH_TOKEN and REPO_URL.startswith("https://"):
        return REPO_URL.replace(
            "https://", "https://x-access-token:" + GH_TOKEN + "@", 1
        )
    return REPO_URL

def scrub(text):
    return text.replace(GH_TOKEN, "***") if GH_TOKEN else text

# --- this repository -------------------------------------------------------
def clone_fresh():
    sh("rm -rf " + REPO_DIR)
    sh(
        "git clone -q --depth 1 --branch " + BRANCH + " " + auth_url()
        + " " + REPO_DIR,
        quiet_cmd="git clone -q --depth 1 --branch " + BRANCH + " "
        + REPO_URL + " " + REPO_DIR,
    )

def update_or_clone():
    if os.path.isdir(os.path.join(REPO_DIR, ".git")):
        # Updating a shallow clone can fail in several uninteresting ways; a
        # re-clone is seconds, so do not make the user debug it.
        try:
            sh(
                "git -C " + REPO_DIR + " fetch -q --depth 1 " + auth_url()
                + " " + BRANCH,
                quiet_cmd="git -C " + REPO_DIR + " fetch -q --depth 1 origin "
                + BRANCH,
            )
            sh("git -C " + REPO_DIR + " reset -q --hard FETCH_HEAD")
            sh("git -C " + REPO_DIR + " clean -qfd")
            return
        except subprocess.CalledProcessError:
            print("In-place update failed -- re-cloning instead.")
    clone_fresh()

try:
    update_or_clone()
except subprocess.CalledProcessError as exc:
    # The overwhelmingly likely cause is a private repository with no token,
    # and the raw git error ("could not read Username", or a bare 128) does
    # not say so. Name the actual fix.
    print()
    print("=" * 70)
    print("Could not get " + REPO_URL + " (branch " + BRANCH + ")")
    print("=" * 70)
    if not GH_TOKEN:
        print(
            "No GITHUB_TOKEN secret is set. If this repository is private,\\n"
            "that is why: an anonymous clone of a private repo fails.\\n"
            "\\n"
            "  1. GitHub > Settings > Developer settings > Personal access\\n"
            "     tokens > Fine-grained. Scope it to this one repository,\\n"
            "     Contents: Read. Give it a short expiry.\\n"
            "  2. In Colab, click the key icon in the left sidebar, add a\\n"
            "     secret named GITHUB_TOKEN, paste the value, and switch on\\n"
            "     'Notebook access'.\\n"
            "  3. Re-run this cell.\\n"
            "\\n"
            "The secret is not stored in the .ipynb and is not shared with\\n"
            "anyone you share the notebook with."
        )
    else:
        print(
            "A GITHUB_TOKEN secret is set, so the token itself is probably\\n"
            "the problem: check it has not expired, that it is scoped to\\n"
            "this repository, and that it grants Contents: Read.\\n"
            "Also check BRANCH above names a branch that exists."
        )
    raise SystemExit(scrub(str(exc)))

# --- python dependencies ---------------------------------------------------
# Colab already ships torch, numpy and usually transformers. Only install what
# is actually missing or too old: a blanket `pip install --upgrade` can swap
# out numpy, tokenizers or torch underneath a kernel that has already imported
# them, and the resulting failures surface far from the cause. Which packages
# that hits depends on the runtime image, so it bites on one GPU type and not
# another.
import importlib.metadata as _md


def installed_version(dist):
    try:
        return _md.version(dist)
    except Exception:
        return None


def at_least(have, want):
    def parts(v):
        out = []
        for chunk in str(v).split(".")[:3]:
            digits = "".join(c for c in chunk if c.isdigit())
            out.append(int(digits) if digits else 0)
        return out
    return bool(have) and parts(have) >= parts(want)

needed = []
if not at_least(installed_version("transformers"), "4.44"):
    needed.append("'transformers>=4.44'")
for dist in ("accelerate", "huggingface_hub", "tmtools", "ml-collections", "prody"):
    if installed_version(dist) is None:
        needed.append(dist)

if needed:
    sh("pip install -q " + " ".join(needed))
else:
    print("$ (python dependencies already satisfied -- nothing to install)")

# --- LigandMPNN ------------------------------------------------------------
if not os.path.isdir(os.path.join(MPNN_DIR, ".git")):
    sh("git clone -q --depth 1 https://github.com/dauparas/LigandMPNN " + MPNN_DIR)

# LigandMPNN's get_model_params.sh pulls ~15 checkpoints. We need the ones
# the three usable model types resolve to -- and we need more than one: a
# ligand-binding search runs MPNN_MODEL_TYPE=ligand_mpnn, whose weights are a
# different file with a different atom_context_num. Handing it the ProteinMPNN
# checkpoint fails at load with a tensor-shape error that says nothing about
# the real mistake.
from redaa.mpnn import DEFAULT_CHECKPOINTS

CKPT_DIR = os.path.join(MPNN_DIR, "model_params")
os.makedirs(CKPT_DIR, exist_ok=True)
for _name in sorted(set(DEFAULT_CHECKPOINTS.values())):
    _path = os.path.join(CKPT_DIR, _name)
    if os.path.exists(_path) and os.path.getsize(_path) > 1_000_000:
        print("  have " + _name)
        continue
    sh("wget -q -O " + _path + " " + MPNN_CKPT_BASE + "/" + _name)

os.environ["MPNN_REPO"] = MPNN_DIR
# Left unset on purpose, so the engine resolves the checkpoint from
# MPNN_MODEL_TYPE. Set it only to use a checkpoint of your own.
os.environ.pop("MPNN_CHECKPOINT", None)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

# --- did pip move something out from under this kernel? -------------------
# Colab's classic failure: a package is replaced on disk while the session
# still holds the old module object. Nothing reports a version mismatch; you
# get an ABI error or a nonsense AttributeError somewhere unrelated. The only
# fix is a kernel restart, so stop here and say so rather than letting later
# cells fail obscurely.
import redaa.colab as _rc

_stale = _rc.stale_imports()
if _stale:
    print("\\n" + "=" * 70)
    print("RESTART REQUIRED")
    print("=" * 70)
    for _name, _imported, _now in _stale:
        print("  %s: this session imported %s, but %s is now installed"
              % (_name, _imported, _now))
    print("\\n  Runtime > Restart session, then re-run from Step 2.")
    print("  Do NOT re-run this cell -- the install already succeeded.")
    print("=" * 70)
    raise SystemExit("restart the runtime, then continue from Step 2")

# --- verify, so a partial install cannot look like a successful one -------
import importlib

problems = []
for module in ("redaa.config", "redaa.colab", "redaa.fetch", "redaa.structure"):
    try:
        importlib.import_module(module)
    except Exception as exc:
        problems.append("cannot import %s: %s" % (module, exc))
if not os.path.exists(os.path.join(MPNN_DIR, "data_utils.py")):
    problems.append("LigandMPNN data_utils.py missing from " + MPNN_DIR)
for _name in sorted(set(DEFAULT_CHECKPOINTS.values())):
    _path = os.path.join(CKPT_DIR, _name)
    if not os.path.exists(_path) or os.path.getsize(_path) < 1_000_000:
        problems.append("MPNN checkpoint missing or truncated: " + _path)
try:
    import transformers
except Exception as exc:
    problems.append("transformers not importable: %s" % exc)

if problems:
    print("\\nINSTALL INCOMPLETE:")
    for line in problems:
        print("  - " + line)
    raise SystemExit("Fix the above and re-run Step 1.")

print("\\nInstall complete and verified.")
print("  repo        " + REPO_DIR)
print("  LigandMPNN  " + MPNN_DIR)
for _type, _name in sorted(DEFAULT_CHECKPOINTS.items()):
    _path = os.path.join(CKPT_DIR, _name)
    print("  checkpoint  %-13s %s (%.0f MB)"
          % (_type, _name, os.path.getsize(_path) / 1e6))
print("  transformers %s" % transformers.__version__)
print("\\nNote: a Colab runtime restart wipes /content. Later cells re-clone the")
print("repo automatically, but pip packages and the checkpoint need this cell again.")
"""
)

code_with_bootstrap(
    """
#@title Step 2 — Check the GPU, apply matching defaults, self-test
import importlib, subprocess, sys

importlib.reload(rc)   # pick up any change the bootstrap just pulled down

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

code_with_bootstrap(
    """
#@title Step 3 — Mount Google Drive (optional, recommended)
#@markdown Keeps outputs and the fold cache across disconnects, so a
#@markdown reconnected session resumes instead of starting over.
#@markdown
#@markdown Model weights stay on local disk on purpose: safetensors reads them
#@markdown by memory-mapping, and Drive's FUSE mount cannot serve that, so a
#@markdown model cached on Drive fails with "SafetensorError: header too
#@markdown large". Re-downloading after a restart is the cheaper problem.
USE_DRIVE = True #@param {type:"boolean"}
DRIVE_FOLDER = "redaa" #@param {type:"string"}
WEIGHTS_ON_DRIVE = False #@param {type:"boolean"}

import os

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    env = rc.use_drive(
        "/content/drive/MyDrive/" + DRIVE_FOLDER, hf_cache=WEIGHTS_ON_DRIVE
    )
    print("\\nUsing Drive for:")
    for key, value in env.items():
        print("  %s=%s" % (key, value))
    print("\\nModel weight cache: " + rc.hf_cache_dir())
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

code_with_bootstrap(
    """
#@title Step 4 — Fetch and/or upload inputs
#@markdown Pick a `SITE_PRESET` to load a verified test case and leave
#@markdown `PDB_SPECS` alone — it fills in the structure and the ligands to
#@markdown keep. `none` uses whatever you type.
SITE_PRESET = "none" #@param ["none", "atp", "tev"]
INPUT_MODE = "fetch" #@param ["fetch", "upload", "drive_folder", "both"]
PDB_SPECS = "1ubq_A\\n2gb1_A" #@param {type:"string"}
CHAIN_OVERRIDE = "" #@param {type:"string"}
KEEP_LIGANDS = "" #@param {type:"string"}
DRIVE_INPUT_FOLDER = "redaa/inputs" #@param {type:"string"}
#@markdown `KEEP_LIGANDS` is a comma-separated list of PDB component ids
#@markdown (`ATP`), kept through cleaning. Everything else, water included, is
#@markdown still stripped. A ligand-binding search needs this: without it
#@markdown LigandMPNN designs against a bare backbone and no pocket can be
#@markdown derived from coordinates.

import os

INPUT_DIR = "/content/redaa_inputs"
files = []

_preset = None
if SITE_PRESET != "none":
    _preset = rc.preset(SITE_PRESET)
    print(_preset.describe())
    print()
    # Only fill in what was left at its default, so editing a field still wins.
    if PDB_SPECS.strip() in ("", "1ubq_A\\n2gb1_A"):
        PDB_SPECS = _preset.fetch_spec
        print("using the preset's structure: %s" % PDB_SPECS)
    if not KEEP_LIGANDS.strip() and _preset.keep_ligands:
        KEEP_LIGANDS = ",".join(_preset.keep_ligands)
        print("keeping the preset's ligand(s): %s" % KEEP_LIGANDS)
    print()

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

_keep = tuple(
    x.strip().upper() for x in KEEP_LIGANDS.replace(",", " ").split() if x.strip()
)

reports = rc.gather_inputs(
    specs=specs,
    files=files,
    input_dir=INPUT_DIR,
    chain=CHAIN_OVERRIDE.strip() or None,
    keep_ligands=_keep or None,
)

print()
print(rc.describe_inputs(reports))
os.environ["INPUT_PDB_DIR"] = INPUT_DIR
os.environ["SITE_KEEP_LIGANDS"] = ",".join(_keep)
if _preset is not None:
    os.environ["SITE_PRESET"] = SITE_PRESET
"""
)

md(
    """
## Step 5 — Mode and functional site

A minimal folding alphabet is not yet a protein. The residues that do the
work — a catalytic cysteine, the arginines that grip a triphosphate — are
exactly the ones a five-letter alphabet throws away, and a catalytic cysteine
has no substitute. This step says which residues have to survive, and how the
search should treat them.

**The two site modes answer different questions, and can disagree.**

| mode | question |
|---|---|
| `alphabet_only` | What is the smallest alphabet that folds? (no site) |
| `site_aware` | What is the smallest alphabet for a protein that *still has this site*? |
| `scaffold_then_site` | Is the minimal folding alphabet compatible with this site — and if not, what did the site cost? |

`site_aware` fixes the site from the first round, so every candidate alphabet
is judged on fold and site together. `scaffold_then_site` runs the original
site-blind search, then grafts the site back on and grows the alphabet if that
broke the fold. A blind search is free to spend the site's positions on
folding, so the graft can fail where `site_aware` would have routed around the
problem. **That difference is the interesting result**, which is why both exist
rather than only the better one.

**Writing a site spec.** Author (PDB) numbering, the numbering in the paper and
the entry:

| token | meaning |
|---|---|
| `H46 D81 C151` | those residues, asserted to be His/Asp/Cys |
| `46` | residue 46, identity not asserted |
| `A/H46` | an explicit chain |
| `range:148-151` | an inclusive span |
| `ligand:ATP` | every residue within 4.5 Å of any ATP atom |
| `ligand:ATP@6.0` | the same at a 6 Å cutoff |

Naming the expected residue is optional and worth doing every time: it is what
turns a numbering mismatch into an error instead of a silently wrong run.
Naming a *ligand* instead of residues derives the pocket from the coordinates,
which beats transcribing a list and is specific to the structure in hand.

**`SITE_RESIDUE_POLICY`** decides whether the site's residue types count as
alphabet letters:

- `exempt` — site positions are held at native identity and are *not* alphabet
  letters. A result reads "5 letters plus 6 exempt positions". Keeps the
  folding question clean.
- `in_alphabet` — the site's types are forced into the alphabet and protected
  from pruning, so the reported alphabet is one a real protein could be written
  in end to end.

`exempt` is the default, but it can flatter a result — the exempt count and
residue types are always reported next to the alphabet, and a site covering
more than a quarter of the chain warns.

**Site preservation is measured on the backbone**, with CB derived from N–CA–C
rather than read, so it works for glycine and does not depend on a predicted
rotamer — the least reliable part of any folding model's output. Three numbers,
and which one fails tells you what went wrong:

- `Site_BB_RMSD` — site backbone after superposing the *whole* structure: is
  the site in the right place?
- `Site_Local_RMSD` — after superposing the *site only*: is its internal
  geometry intact?
- `Site_CB_Dev` — worst error in the site's internal CB–CB distance matrix,
  superposition-free.

These are a necessary condition for a working site, not a sufficient one. They
say the scaffold puts the functional residues where they belong; whether the
chemistry works is a wet-lab question.
"""
)

code_with_bootstrap(
    """
#@title Step 5 — Mode and functional site
SEARCH_MODE = "alphabet_only" #@param ["alphabet_only", "site_aware", "scaffold_then_site"]
SITE_PRESET = "none" #@param ["none", "atp", "tev"]
SITE_SPEC = "" #@param {type:"string"}
SITE_CHAIN = "" #@param {type:"string"}
SITE_RESIDUE_POLICY = "exempt" #@param ["exempt", "in_alphabet"]
SITE_CONTACT_MAX = 0 #@param {type:"integer"}
MPNN_MODEL_TYPE = "" #@param ["", "protein_mpnn", "soluble_mpnn", "ligand_mpnn"]
#@markdown Gate thresholds in Angstroms. Set any to 0 to record the number
#@markdown without gating on it; untick `SITE_GATE` to record all three and let
#@markdown the TM gate alone decide.
SITE_GATE = True #@param {type:"boolean"}
SITE_MAX_BB_RMSD = 1.5 #@param {type:"number"}
SITE_MAX_LOCAL_RMSD = 1.0 #@param {type:"number"}
SITE_MAX_CB_DEV = 1.5 #@param {type:"number"}
SITE_REQUIRE_IDENTITY = True #@param {type:"boolean"}

import os

# A preset fills in whatever is left blank -- its spec, ligands and MPNN model
# type -- so picking "atp" and typing nothing else is complete. Leaving
# SITE_PRESET at "none" uses exactly what you typed.
_applied = rc.configure_site(
    mode=SEARCH_MODE,
    spec=SITE_SPEC,
    preset="" if SITE_PRESET == "none" else SITE_PRESET,
    chain=SITE_CHAIN,
    keep_ligands=os.environ.get("SITE_KEEP_LIGANDS", ""),
    contact_max=SITE_CONTACT_MAX,
    policy=SITE_RESIDUE_POLICY,
    gate=SITE_GATE,
    max_bb_rmsd=SITE_MAX_BB_RMSD,
    max_local_rmsd=SITE_MAX_LOCAL_RMSD,
    max_cb_dev=SITE_MAX_CB_DEV,
    require_identity=SITE_REQUIRE_IDENTITY,
    mpnn_model_type=MPNN_MODEL_TYPE,
)

print()
print("Mode: %s" % _applied["SEARCH_MODE"])
for key in sorted(_applied):
    if key != "SEARCH_MODE":
        print("  %s=%s" % (key, _applied[key]))

# Resolve the site against the files actually prepared in Step 4, and show it.
# This is the one setting that can be wrong in a way the run does not notice --
# right numbering for a different entry, a ligand stripped during cleaning, a
# contact shell that swallowed half the protein -- and all of it is visible
# here in a second instead of after an hour of GPU time.
if _applied["SEARCH_MODE"] != "alphabet_only" and _applied["SITE_SPEC"]:
    print()
    print("=" * 70)
    print("Resolving the site against the prepared inputs")
    print("=" * 70)
    print(rc.preview_sites(
        os.environ.get("INPUT_PDB_DIR", "/content/redaa_inputs"),
        _applied["SITE_SPEC"],
        chain=_applied["SITE_CHAIN"] or None,
        contact_max=SITE_CONTACT_MAX,
    ))
"""
)

code_with_bootstrap(
    """
#@title Step 6 — Search settings
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

code_with_bootstrap(
    """
#@title Step 7 — Run the search
#@markdown Safe to re-run: finished replicates are skipped, cached folds are
#@markdown free, and the loaded models stay in memory between runs.
import importlib, os, sys, time

import redaa.config, redaa.pipeline
importlib.reload(redaa.config)
importlib.reload(redaa.pipeline)

if "INPUT_PDB_DIR" not in os.environ:
    raise SystemExit(
        "INPUT_PDB_DIR is not set -- run Step 4 (inputs) and Step 6 (settings) "
        "first. A runtime restart clears them."
    )

cfg = redaa.config.RedAAConfig()
if bool(globals().get("TEST_MODE", False)):
    cfg.apply_test_mode()
    print("*** TEST MODE: one domain, one replicate, reduced budgets ***\\n")

# Hand back the engines from an earlier run in this session so ESMFold2 is not
# reloaded. Pipeline reloads anyway if the model id changed.
_cached = globals().get("_REDAA_ENGINES") or {}
pipe = redaa.pipeline.Pipeline(
    cfg, folder=_cached.get("folder"), mpnn=_cached.get("mpnn")
)

# Everything the run prints is teed to a file as well as to this cell, and a
# single attachable report is written whether the run finishes or crashes.
# Colab truncates long output and loses it entirely on a disconnect, and the
# output you most need is from a run that died.
_started = time.time()
exit_code = 1
with rc.capture_run_log() as runlog:
    try:
        exit_code = pipe.run()
    finally:
        _REDAA_ENGINES = {"folder": pipe.folder, "mpnn": pipe.mpnn}
        RUN_DIR = pipe.run_dir

REPORT_PATH = runlog.report_path

print("\\nExit code %d (%s)" % (
    exit_code,
    "complete" if exit_code == 0 else "stopped early but checkpointed",
))
print("Wall time %.1f min" % ((time.time() - _started) / 60))
print("Run directory: " + str(RUN_DIR))
"""
)

code_with_bootstrap(
    """
#@title Step 8 — Results
DOWNLOAD_ZIP = True #@param {type:"boolean"}
DOWNLOAD_REPORT = True #@param {type:"boolean"}
#@markdown `DOWNLOAD_REPORT` grabs the single text file carrying the
#@markdown environment, the settings, the results and the console log -- the
#@markdown file to attach when asking about a failure.

import os

run_dir = globals().get("RUN_DIR") or rc.latest_run_dir()
if not run_dir:
    raise SystemExit("No run directory found — run step 7 first.")
print("Run directory: " + run_dir + "\\n")

rows = rc.summary_rows(run_dir)
if rows:
    keep = ["pdb", "replicate", "final_alphabet", "alphabet_size",
            "pruned_alphabet", "pruned_alphabet_size",
            "best_seq_tm", "best_seq_rmsd", "best_seq_plddt", "status"]
    # The site and graft columns only carry anything in a site mode, so they
    # are shown only when they do rather than padding every table with blanks.
    _mode = (rows[0].get("search_mode") or "alphabet_only")
    if _mode != "alphabet_only":
        keep[-1:-1] = ["site_n_positions", "site_residue_types",
                       "best_seq_site_bb_rmsd", "best_seq_site_local_rmsd",
                       "best_seq_site_cb_dev", "best_seq_site_ok"]
    if _mode == "scaffold_then_site":
        keep[-1:-1] = ["graft_direct_pass", "graft_added_aas",
                       "graft_final_alphabet", "graft_passed"]
    columns = [c for c in keep if c in rows[0]]
    try:
        import pandas as pd
        display(pd.DataFrame(rows)[columns])
    except Exception:
        for row in rows:
            print({k: row.get(k) for k in columns})
    print()
    print(rc.describe_site_results(rows))
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

def _download(path):
    try:
        from google.colab import files as colab_files
        colab_files.download(path)
    except ImportError:
        print("  (not in Colab — the file is on disk at the path above)")

if DOWNLOAD_ZIP:
    path = rc.zip_results(run_dir)
    print("\\nZipped to %s (%.1f MB)" % (path, os.path.getsize(path) / 1e6))
    _download(path)

if DOWNLOAD_REPORT:
    report = globals().get("REPORT_PATH") or rc.latest_report()
    if not (report and os.path.exists(report)):
        report = rc.write_report(run_dir=run_dir)
        print("\\nNo run report in this session; wrote one from current state.")
    print("\\nRun report: %s (%.0f KB)"
          % (report, os.path.getsize(report) / 1024))
    _download(report)
"""
)

code_with_bootstrap(
    """
#@title Step 9 — Diagnose a failure (run this if anything above broke)
#@markdown Prints the environment, every relevant package version, the
#@markdown resolved settings and an explicit pass/fail list. Paste the whole
#@markdown output into a bug report. `TRY_MODEL_LOAD` additionally tries to
#@markdown load both models -- slow, but it separates "the environment is
#@markdown wrong" from "this model will not load on this GPU".
TRY_MODEL_LOAD = False #@param {type:"boolean"}
REPAIR_HF_CACHE = False #@param {type:"boolean"}
#@markdown `REPAIR_HF_CACHE` deletes cached model files whose headers do not
#@markdown parse, so the next load re-downloads them. Use it after a
#@markdown "SafetensorError: header too large", which means a weight file was
#@markdown truncated or was cached somewhere that cannot serve it.

import importlib

importlib.reload(rc)

if REPAIR_HF_CACHE:
    removed = rc.repair_hf_cache()
    if removed:
        print("Deleted %d cached entr%s:" % (
            len(removed), "y" if len(removed) == 1 else "ies"))
        for path in removed:
            print("  " + path)
        print("\\nRe-run Step 7; the weights will be downloaded again.\\n")
    else:
        print("Nothing to repair -- no corrupt weight files found.\\n")

report = rc.print_diagnosis(try_model_load=TRY_MODEL_LOAD)

# Leave an attachable copy behind, including the newest run's results.
_path = rc.write_report(run_dir=rc.latest_run_dir())
print("\\nWritten to %s -- attach this file when asking about a failure."
      % _path)
"""
)

md(
    """
## Step 10 — Hand a run back for debugging (optional)

There is no way for anything outside your browser to attach to a running Colab
kernel: the kernel has no public inbound address, and Colab exposes no API for
it. So a session cannot be driven from elsewhere, and this step is the next
best thing — it makes the round trip cheap instead of impossible.

**Without any setup.** Step 8 already downloads one text file carrying the
environment, every package version, the resolved settings, the results so far
and the full console log — including the traceback if the run crashed. Attach
that file to the conversation. It is the single most useful thing to send,
because Colab truncates long cell output and loses it entirely on a
disconnect, while the file does not.

**With about two minutes of setup**, the cell below removes the manual step by
pushing that same report into this repository, where it can be read directly:

1. Make a **fine-grained** GitHub personal access token
   (Settings → Developer settings → Personal access tokens → Fine-grained).
   Scope it to **this repository only**, with **Contents: Read and write**.
   Nothing else. Give it a short expiry.
2. In Colab, open the key icon in the left sidebar (Secrets), add
   `GITHUB_TOKEN` with that value, and enable it for this notebook.
3. Run the cell. It commits only the report and the summary CSVs — never the
   fold cache, the weights or the structures — to a `colab-reports/` path on a
   separate branch, so it cannot touch code.

The token is read from Colab's secret store and never printed, never written
to the notebook, and never committed. If the secret is absent the cell says so
and does nothing.

Then the loop is: you run, the cell pushes, a fix is pushed back, and you
re-run **Step 1** (which re-clones) and **Step 7**. Finished replicates are
skipped and cached folds cost nothing, so resuming is cheap.
"""
)

code_with_bootstrap(
    """
#@title Step 10 — Push this run's report to GitHub (optional)
REPORTS_BRANCH = "colab-reports" #@param {type:"string"}
NOTE = "" #@param {type:"string"}
#@markdown `NOTE` is a one-line description of what you were trying — it goes
#@markdown into the commit message, which is usually the fastest way to say
#@markdown what you expected to happen.

import os, shutil, subprocess, time

def _run(cmd, **kw):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)

token = None
try:
    from google.colab import userdata
    token = userdata.get("GITHUB_TOKEN")
except Exception as exc:
    print("No Colab secret available (%s)." % type(exc).__name__)

if not token:
    print(
        "GITHUB_TOKEN is not set as a Colab secret, so nothing was pushed.\\n"
        "Use Step 8's DOWNLOAD_REPORT and attach the file instead, or add the\\n"
        "secret as described above and re-run this cell."
    )
else:
    run_dir = globals().get("RUN_DIR") or rc.latest_run_dir()
    if not run_dir:
        raise SystemExit("No run directory found — run step 7 first.")

    report = globals().get("REPORT_PATH") or rc.latest_report()
    if not (report and os.path.exists(report)):
        report = rc.write_report(run_dir=run_dir)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    work = "/content/_report_push"
    shutil.rmtree(work, ignore_errors=True)

    # A fresh shallow clone, so nothing here can disturb the working checkout
    # in /content/MaLAtium_2.0 that the rest of the notebook imports from.
    url = REPO_URL.replace("https://", "https://x-access-token:%s@" % token)
    out = _run("git clone --depth 1 --branch %s %s %s"
               % (REPORTS_BRANCH, url, work))
    if out.returncode != 0:
        # The branch does not exist yet: start it with no history, so the
        # reports branch never carries a copy of the code.
        out = _run("git clone --depth 1 %s %s" % (url, work))
        if out.returncode != 0:
            raise SystemExit("clone failed: " + (out.stderr or "")[-800:])
        _run("git checkout --orphan %s" % REPORTS_BRANCH, cwd=work)
        _run("git rm -rf --quiet . || true", cwd=work)

    dest = os.path.join(work, "colab-reports", stamp)
    os.makedirs(dest, exist_ok=True)
    shutil.copy2(report, os.path.join(dest, os.path.basename(report)))

    # Only the small text artefacts. Never the fold cache (large, binary, and
    # pointless outside its own session), never weights, never structures.
    copied = [os.path.basename(report)]
    for name in ("BATCH_SUMMARY.csv", "FLAGGED_DOMAINS.csv"):
        src = os.path.join(run_dir, name)
        if os.path.exists(src) and os.path.getsize(src) < 2_000_000:
            shutil.copy2(src, os.path.join(dest, name))
            copied.append(name)

    message = "Colab run report %s" % stamp
    if NOTE.strip():
        message += "\\n\\n" + NOTE.strip()

    _run("git config user.email colab@example.invalid", cwd=work)
    _run("git config user.name 'Colab run'", cwd=work)
    _run("git add colab-reports", cwd=work)
    commit = _run("git commit -m %s" % repr(message), cwd=work)
    if commit.returncode != 0 and "nothing to commit" not in (commit.stdout or ""):
        raise SystemExit("commit failed: " + (commit.stderr or commit.stdout)[-800:])

    push = _run("git push -u origin %s" % REPORTS_BRANCH, cwd=work)
    if push.returncode != 0:
        # Never echo the command: it carries the token.
        raise SystemExit(
            "push failed. Check the token is fine-grained for this repo with "
            "Contents: Read and write.\\n"
            + (push.stderr or "").replace(token, "***")[-800:]
        )

    print("Pushed %s to %s:colab-reports/%s"
          % (", ".join(copied), REPORTS_BRANCH, stamp))
    print("Share that branch and path; nothing else was uploaded.")
    shutil.rmtree(work, ignore_errors=True)
"""
)

md(
    """
## Troubleshooting

**Start with Step 9.** It prints the environment, every package version, the
resolved settings and a pass/fail list, and names the fix for each problem it
finds. Almost everything below is something Step 9 will tell you directly.

**Every run leaves a report behind.** Step 7 tees its output to
`<outputs>/../logs/run_*.log` and writes `report_*.txt` beside it — whether the
run finishes or crashes, with the traceback included. That one text file
carries the environment, the resolved settings, the results so far and the
console log, so it is the thing worth attaching when asking about a failure.
Step 8 downloads it and Step 9 writes a fresh one. Colab truncates long cell
output and loses it on a disconnect; the file does not.

**Changing the GPU type restarts the runtime**, which wipes `/content` exactly
like any other restart — so switching from T4 to A100 means Step 1 has to run
again. Steps 2–7 re-clone the repo themselves, but the pip packages and the
MPNN checkpoint go with `/content` and only Step 1 restores them.

**A nonsense error right after Step 1** (an ABI complaint, an `AttributeError`
on something that obviously exists, a segfault) usually means `pip` replaced a
package this kernel had already imported. Step 1 now detects that and stops
with `RESTART REQUIRED`; if you see it, do Runtime > Restart session and
continue from Step 2 — do not re-run the install.

**`SafetensorError: Error while deserializing header: header too large`.** A
cached weight file is unreadable. Either the download was truncated, or the
cache was on Google Drive, whose FUSE mount cannot serve the memory-mapped
reads safetensors uses. Re-run Step 3 (which keeps weights on local disk) and
then Step 9 with `REPAIR_HF_CACHE` to clear the bad entries.

**bf16 / `esmc_precision` errors on a T4.** The free-tier T4 is pre-Ampere and
has no bfloat16. Try `FOLD_PRECISION="fp16"` in step 6, or one of the smaller
`ESMFold2-600` / `ESMFold2-300` variants, or switch to an L4 or A100 runtime.

**CUDA out of memory.** Lower `FOLD_MAX_RESIDUES` — step 2 sets a default from
your GPU's VRAM — or use smaller domains. Over-long sequences are skipped
rather than run until they die, and MPNN halves its batch and retries on OOM.

**`Could not import LigandMPNN's data_utils`.** Step 1 did not finish. Check
that `/content/LigandMPNN/data_utils.py` and the checkpoint under
`model_params/` both exist.

**Session died mid-run.** If step 3 put things on Drive: re-run steps 1–3, then
step 7. Finished replicates are skipped and cached folds cost nothing, so it
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
step 6. For the batch job and the full configuration reference see `README.md`
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
