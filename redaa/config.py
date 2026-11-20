"""Configuration for the reduced-alphabet search.

Every knob is overridable from the environment so a SLURM array job can be
retargeted without editing the source.  ``RedAAConfig.describe()`` prints the
resolved values into the job log, which is what you want when you come back to
a six-month-old output directory and need to know what produced it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields

ALL_AAS = tuple("ACDEFGHIKLMNPQRSTVWY")

# How the search treats a functional site.
#
#   ALPHABET_ONLY       the original search: fold only, no site.
#   SCAFFOLD_THEN_SITE  find the minimal folding alphabet first, blind to the
#                       site, then graft the site in and ask what it cost.
#                       Answers "is the minimal scaffold alphabet compatible
#                       with this site at all?"
#   SITE_AWARE          hold the site fixed from the first round, so every
#                       alphabet is judged on fold *and* site together.
#                       Answers "what is the minimal alphabet for a protein
#                       that still has this site?"
#
# They can disagree, and that disagreement is the interesting result: a
# scaffold alphabet found without the site is free to use the positions the
# site needs for folding, and grafting the site back then breaks the fold.
MODE_ALPHABET_ONLY = "alphabet_only"
MODE_SCAFFOLD_THEN_SITE = "scaffold_then_site"
MODE_SITE_AWARE = "site_aware"
SEARCH_MODES = (MODE_ALPHABET_ONLY, MODE_SCAFFOLD_THEN_SITE, MODE_SITE_AWARE)

# Whether the site's residue types have to be inside the reported alphabet.
#
#   EXEMPT       site positions are held at native identity and are not
#                counted as alphabet letters. A result reads "5 letters plus
#                6 exempt positions". This is the right framing when the
#                question is about the folding alphabet and the site is
#                treated as machinery bolted onto it.
#   IN_ALPHABET  the site's residue types are forced into the alphabet and
#                protected from pruning, so the reported alphabet is one a
#                real protein could be written in end to end.
#
# EXEMPT is the default because it keeps the folding question clean, but it
# can flatter a result, so the number of exempt positions and their residue
# types are always reported alongside the alphabet.
#   MEASURE_ONLY site positions are neither fixed nor exempt: MPNN is free to
#                put whatever the alphabet allows there, and the site is
#                measured and gated on *geometry alone*. Identity is not
#                required, because the whole point is that the design chose
#                its own residues.
#
# MEASURE_ONLY is the honest companion to MPNN_MODEL_TYPE=ligand_mpnn without
# fixed residues: the ligand shapes the design through atom context, nothing
# is held at its native identity, and the pocket's backbone geometry is still
# measured -- otherwise there is no evidence the ligand context did anything.
POLICY_EXEMPT = "exempt"
POLICY_IN_ALPHABET = "in_alphabet"
POLICY_MEASURE_ONLY = "measure_only"
SITE_POLICIES = (POLICY_EXEMPT, POLICY_IN_ALPHABET, POLICY_MEASURE_ONLY)

# The site roles a policy can be set for. Mirrors redaa.sites.ROLE_*, which
# this module deliberately does not import (sites pulls in numpy and the
# structure parser; config is meant to be importable on its own). A test keeps
# the two lists in step.
SITE_ROLES = ("catalytic", "contact", "interface")


def parse_site_policy(text: str) -> tuple[dict[str, str], list[str]]:
    """Resolve ``SITE_RESIDUE_POLICY`` to one policy per role.

    Accepts a bare policy for every role::

        exempt

    or per-role overrides, with or without a bare default in front::

        exempt,interface=measure_only
        catalytic=exempt,contact=in_alphabet,interface=measure_only

    Per-role control is not a nicety.  A ligand pocket is a handful of
    residues whose side chains do the chemistry; an interface to a nucleic
    acid is fifty residues spanning nearly every amino acid.  Holding both
    under one policy either freezes a sixth of the protein or forces an
    eighteen-letter alphabet, and neither answers the question the search is
    asking.

    Returns ``(role -> policy, problems)``.
    """
    default = POLICY_EXEMPT
    overrides: dict[str, str] = {}
    problems: list[str] = []
    seen_bare = False

    for tok in (t for t in re.split(r"[,\s]+", text or "") if t):
        if "=" in tok:
            role, _, policy = tok.partition("=")
            role = role.strip().lower()
            policy = policy.strip()
            if role not in SITE_ROLES:
                problems.append(
                    f"SITE_RESIDUE_POLICY names role {role!r}, which is not "
                    f"one of {', '.join(SITE_ROLES)}"
                )
                continue
            if policy not in SITE_POLICIES:
                problems.append(
                    f"SITE_RESIDUE_POLICY sets {role}={policy!r}, which is "
                    f"not one of {', '.join(SITE_POLICIES)}"
                )
                continue
            overrides[role] = policy
            continue
        if tok not in SITE_POLICIES:
            problems.append(
                f"SITE_RESIDUE_POLICY={tok!r} is not one of "
                f"{', '.join(SITE_POLICIES)}, and is not a role=policy pair"
            )
            continue
        if seen_bare:
            problems.append(
                f"SITE_RESIDUE_POLICY gives two blanket policies ({default}, "
                f"{tok}); give one, then per-role overrides such as "
                "'exempt,interface=measure_only'"
            )
            continue
        default = tok
        seen_bare = True

    resolved = {role: overrides.get(role, default) for role in SITE_ROLES}
    return resolved, problems

# Every environment variable this module reads. Exported so a caller can
# report the settings actually in effect without pattern-matching os.environ
# and picking up unrelated variables.
ENV_VARS = (
    "AUTO_RESUME", "BASELINE_TOLERANCE", "CHUNK_INDEX", "CHUNK_SIZE",
    "DO_BACKWARD_PRUNING", "DYNAMIC_SEEDING", "DYNAMIC_SEED_SIZE",
    "ESM_START_AFTER_N_AAS", "EXHAUSTIVE_PRUNING_THRESHOLD",
    "FOLD_BATCH_SIZE", "FOLD_CACHE_PATH", "FOLD_LOOPS", "FOLD_MAX_RESIDUES",
    "FOLD_MODEL_ID", "FOLD_PRECISION", "FOLD_SEED", "FOLD_STEPS_FINAL",
    "FOLD_STEPS_SCREEN", "INPUT_PDB_DIR", "MAX_IDENTITY_FOR_FOLD",
    "MAX_RMSD_THRESHOLD", "MAX_WALL_SECONDS", "MIN_PLDDT_DISORDER",
    "MIN_TM_SCORE", "MPNN_BASELINE_TEMP", "MPNN_BATCH_SIZE",
    "MPNN_CHECKPOINT", "MPNN_MODEL_TYPE", "MPNN_NUM_SEQ", "MPNN_REPO",
    "MPNN_TEMP", "NATIVE_PLDDT_MIN", "NATIVE_TM_MIN", "NUM_REPLICATES",
    "NUM_SEQS_TO_FOLD", "OUTPUT_ROOT", "PRUNE_FOLDS",
    "PRUNING_DOWNWARD_CLOSURE", "PRUNING_MIN_SIZE", "RESUME_FROM",
    "RUN_NAME", "SAVE_FOLD_PDBS", "SCREEN_FOLDS_STAGE1",
    "SCREEN_FOLDS_STAGE2", "SCREEN_STAGE2_KEEP", "SEARCH_MODE",
    "EXCLUDE_AAS", "START_ALPHABET",
    "SITE_CHAIN", "SITE_CONTACT_MAX", "SITE_GATE", "SITE_KEEP_LIGANDS",
    "SITE_MAX_BB_RMSD", "SITE_MAX_CB_DEV", "SITE_MAX_LOCAL_RMSD",
    "SITE_PRESET", "SITE_REQUIRE_IDENTITY", "SITE_RESIDUE_POLICY",
    "SITE_SPEC", "USE_FOLD_CACHE",
    "USE_HYDROPHOBIC_SEEDING", "USE_PLDDT_GATE", "USE_TORCH_COMPILE",
)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _env_aas(name: str, default: str) -> tuple[str, ...]:
    """Parse a set of one-letter residue codes from the environment.

    Accepts "AG", "A,G", "A G" and lower case. Unknown letters raise rather
    than being dropped: a typo in an alphabet is not something to discover
    from a result that looks plausible.
    """
    raw = os.environ.get(name)
    text = default if raw is None or raw.strip() == "" else raw
    letters = [c for c in text.upper() if not c.isspace() and c != ","]
    bad = sorted({c for c in letters if c not in ALL_AAS})
    if bad:
        raise ValueError(
            f"{name}={text!r} contains {', '.join(bad)}, which "
            f"{'is' if len(bad) == 1 else 'are'} not standard one-letter "
            "residue code(s). Use letters from " + "".join(ALL_AAS)
        )
    return tuple(dict.fromkeys(letters))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class RedAAConfig:
    """Resolved run configuration."""

    # ---------------- input / output ----------------
    input_pdb_dir: str = field(
        default_factory=lambda: _env_str("INPUT_PDB_DIR", "input_pdbs")
    )
    output_root: str = field(default_factory=lambda: _env_str("OUTPUT_ROOT", "outputs"))
    run_name: str = field(default_factory=lambda: _env_str("RUN_NAME", ""))
    chunk_index: int = field(default_factory=lambda: _env_int("CHUNK_INDEX", 0))
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 10))
    resume_from: str = field(default_factory=lambda: _env_str("RESUME_FROM", ""))
    auto_resume: bool = field(default_factory=lambda: _env_bool("AUTO_RESUME", True))

    # ---------------- ESMFold2-Fast ----------------
    fold_model_id: str = field(
        default_factory=lambda: _env_str("FOLD_MODEL_ID", "Synthyra/ESMFold2-Fast")
    )
    fold_precision: str = field(
        default_factory=lambda: _env_str("FOLD_PRECISION", "auto")
    )
    # ESMFold2's structure module is a diffusion sampler, so the step count is
    # a direct speed/quality dial.  Screening folds -- the overwhelming
    # majority -- only ever need to tell "right fold" from "wrong fold", so
    # they run short; the folds that end up in the published numbers run long.
    fold_steps_screen: int = field(
        default_factory=lambda: _env_int("FOLD_STEPS_SCREEN", 50)
    )
    fold_steps_final: int = field(
        default_factory=lambda: _env_int("FOLD_STEPS_FINAL", 200)
    )
    fold_loops: int = field(default_factory=lambda: _env_int("FOLD_LOOPS", 1))
    fold_seed: int = field(default_factory=lambda: _env_int("FOLD_SEED", 7))
    fold_batch_size: int = field(default_factory=lambda: _env_int("FOLD_BATCH_SIZE", 4))
    fold_max_residues: int = field(
        default_factory=lambda: _env_int("FOLD_MAX_RESIDUES", 1024)
    )
    use_torch_compile: bool = field(
        default_factory=lambda: _env_bool("USE_TORCH_COMPILE", False)
    )

    # ---------------- LigandMPNN ----------------
    # Path to a LigandMPNN checkout (contains data_utils.py / model_utils.py).
    mpnn_repo: str = field(default_factory=lambda: _env_str("MPNN_REPO", ""))
    mpnn_checkpoint: str = field(
        default_factory=lambda: _env_str("MPNN_CHECKPOINT", "")
    )
    mpnn_model_type: str = field(
        default_factory=lambda: _env_str("MPNN_MODEL_TYPE", "protein_mpnn")
    )
    mpnn_num_seq: int = field(default_factory=lambda: _env_int("MPNN_NUM_SEQ", 25))
    mpnn_batch_size: int = field(
        default_factory=lambda: _env_int("MPNN_BATCH_SIZE", 25)
    )
    mpnn_temp: float = field(default_factory=lambda: _env_float("MPNN_TEMP", 0.5))
    mpnn_baseline_temp: float = field(
        default_factory=lambda: _env_float("MPNN_BASELINE_TEMP", 0.25)
    )

    # ---------------- search ----------------
    num_replicates: int = field(default_factory=lambda: _env_int("NUM_REPLICATES", 10))
    # Where the forward search starts. A and G by default: the two smallest
    # residues, and the pair V9 used, so results stay comparable. Any set of
    # one-letter codes works -- "AGV", "GS", a single letter. The search can
    # only grow, so a larger seed is a different (and weaker) question: it
    # asks what that set needs *added*, not what the minimum is.
    starting_alphabet: tuple[str, ...] = field(
        default_factory=lambda: _env_aas("START_ALPHABET", "AG")
    )
    # Residues the search may never use: never seeded, never offered as a
    # candidate, and left out of the 20-letter baseline so that reference
    # score is the achievable ceiling rather than an unreachable one.
    #
    # The obvious use is C, to keep disulfides out of a design, or W and M for
    # a target where oxidation matters. Note this is a statement about the
    # *alphabet*, not about the structure: a site residue held at its native
    # identity under SITE_RESIDUE_POLICY=exempt is not an alphabet letter, so
    # excluding its type is consistent. Under in_alphabet it is a
    # contradiction, and validate() says so.
    exclude_aas: tuple[str, ...] = field(
        default_factory=lambda: _env_aas("EXCLUDE_AAS", "")
    )
    use_hydrophobic_seeding: bool = field(
        default_factory=lambda: _env_bool("USE_HYDROPHOBIC_SEEDING", True)
    )
    dynamic_seeding: bool = field(
        default_factory=lambda: _env_bool("DYNAMIC_SEEDING", False)
    )
    dynamic_seed_size: int = field(
        default_factory=lambda: _env_int("DYNAMIC_SEED_SIZE", 2)
    )
    baseline_tolerance: float = field(
        default_factory=lambda: _env_float("BASELINE_TOLERANCE", 1.50)
    )
    esm_start_after_n_aas: int = field(
        default_factory=lambda: _env_int("ESM_START_AFTER_N_AAS", 3)
    )

    # Two-stage candidate screening.  V9 folded `num_seqs_quick_validate`
    # sequences for all ~17 candidates every round.  Here every candidate gets
    # a cheap first look, then only the most promising few get the full
    # per-candidate budget -- the decision is made on the same evidence for
    # the candidates that could actually win, at roughly half the folds.
    screen_folds_stage1: int = field(
        default_factory=lambda: _env_int("SCREEN_FOLDS_STAGE1", 1)
    )
    screen_folds_stage2: int = field(
        default_factory=lambda: _env_int("SCREEN_FOLDS_STAGE2", 3)
    )
    screen_stage2_keep: int = field(
        default_factory=lambda: _env_int("SCREEN_STAGE2_KEEP", 5)
    )
    num_seqs_to_fold: int = field(
        default_factory=lambda: _env_int("NUM_SEQS_TO_FOLD", 9)
    )
    max_identity_for_fold: float = field(
        default_factory=lambda: _env_float("MAX_IDENTITY_FOR_FOLD", 0.70)
    )

    # ---------------- functional site ----------------
    search_mode: str = field(
        default_factory=lambda: _env_str("SEARCH_MODE", MODE_ALPHABET_ONLY)
    )
    # A named test case (see redaa.sites.PRESETS): fills in the structure, the
    # site spec, which ligands to keep and the MPNN model type in one go.
    site_preset: str = field(default_factory=lambda: _env_str("SITE_PRESET", ""))
    # Author-numbering site spec, e.g. "H46 D81 C151" or "ligand:ATP@4.5".
    site_spec: str = field(default_factory=lambda: _env_str("SITE_SPEC", ""))
    site_chain: str = field(default_factory=lambda: _env_str("SITE_CHAIN", ""))
    site_keep_ligands: str = field(
        default_factory=lambda: _env_str("SITE_KEEP_LIGANDS", "")
    )
    # Whole chains to retain as binding context -- a second subunit, or a
    # nucleic acid such as a tRNA. Retained as heteroatom records, so they are
    # never folded or designed; they define an interface and give LigandMPNN
    # something to design against.
    site_keep_partners: str = field(
        default_factory=lambda: _env_str("SITE_KEEP_PARTNERS", "")
    )
    # Cap on a ligand-derived contact shell, closest residues first. 0 keeps
    # the whole shell.
    site_contact_max: int = field(
        default_factory=lambda: _env_int("SITE_CONTACT_MAX", 0)
    )
    # One policy for every site role, or per-role overrides. See
    # parse_site_policy for the accepted forms.
    site_residue_policy: str = field(
        default_factory=lambda: _env_str("SITE_RESIDUE_POLICY", POLICY_EXEMPT)
    )
    # Site-preservation thresholds, in Angstroms. Set any of them to 0 to
    # record the number without gating on it.
    site_gate: bool = field(default_factory=lambda: _env_bool("SITE_GATE", True))
    # Which site roles the geometry thresholds are applied to. Empty means
    # all of them, which is the only sensible default for a compact site.
    # It stops being sensible the moment a site has a compact half and a
    # sprawling half: thresholds that are right for a 20-residue pocket are
    # unreachable over a 48-residue interface spanning the whole chain, so
    # one set of numbers applied to both rejects everything. Roles left out
    # are still measured and reported, just not gated.
    site_gate_roles: str = field(
        default_factory=lambda: _env_str("SITE_GATE_ROLES", "")
    )
    site_max_bb_rmsd: float = field(
        default_factory=lambda: _env_float("SITE_MAX_BB_RMSD", 1.5)
    )
    site_max_local_rmsd: float = field(
        default_factory=lambda: _env_float("SITE_MAX_LOCAL_RMSD", 1.0)
    )
    site_max_cb_dev: float = field(
        default_factory=lambda: _env_float("SITE_MAX_CB_DEV", 1.5)
    )
    # RMS rather than worst-case CB-CB distance error. Off by default so
    # nothing changes for an existing run. Prefer it over SITE_MAX_CB_DEV
    # whenever the gated site is more than a handful of residues: the worst
    # pair grows with site size at fixed accuracy, the RMS does not.
    site_max_cb_rms_dev: float = field(
        default_factory=lambda: _env_float("SITE_MAX_CB_RMS_DEV", 0.0)
    )
    # Thresholds for the roles NOT named in SITE_GATE_ROLES. All 0 by
    # default, which is measurement without gating -- the old behaviour. Set
    # them when a site's second half has a real but looser requirement: a
    # tRNA interface that has to be in the right place, with no constraint on
    # how precisely, is gated here while the pocket is gated above.
    site_rest_max_bb_rmsd: float = field(
        default_factory=lambda: _env_float("SITE_REST_MAX_BB_RMSD", 0.0)
    )
    site_rest_max_local_rmsd: float = field(
        default_factory=lambda: _env_float("SITE_REST_MAX_LOCAL_RMSD", 0.0)
    )
    site_rest_max_cb_dev: float = field(
        default_factory=lambda: _env_float("SITE_REST_MAX_CB_DEV", 0.0)
    )
    site_rest_max_cb_rms_dev: float = field(
        default_factory=lambda: _env_float("SITE_REST_MAX_CB_RMS_DEV", 0.0)
    )
    # Leave a site position free rather than holding a residue EXCLUDE_AAS
    # bars. Off by default: holding the site is usually the point, and
    # quietly not holding part of it would be worse than the contradiction.
    # The run names every position this frees.
    site_free_excluded: bool = field(
        default_factory=lambda: _env_bool("SITE_FREE_EXCLUDED", False)
    )
    site_require_identity: bool = field(
        default_factory=lambda: _env_bool("SITE_REQUIRE_IDENTITY", True)
    )

    # ---------------- acceptance ----------------
    min_tm_score: float = field(default_factory=lambda: _env_float("MIN_TM_SCORE", 0.51))
    use_plddt_gate: bool = field(
        default_factory=lambda: _env_bool("USE_PLDDT_GATE", False)
    )
    min_plddt_disorder: float = field(
        default_factory=lambda: _env_float("MIN_PLDDT_DISORDER", 30.0)
    )
    max_rmsd_threshold: float = field(
        default_factory=lambda: _env_float("MAX_RMSD_THRESHOLD", 3.0)
    )
    native_plddt_min: float = field(
        default_factory=lambda: _env_float("NATIVE_PLDDT_MIN", 50.0)
    )
    native_tm_min: float = field(
        default_factory=lambda: _env_float("NATIVE_TM_MIN", 0.5)
    )

    # ---------------- backward pruning ----------------
    do_backward_pruning: bool = field(
        default_factory=lambda: _env_bool("DO_BACKWARD_PRUNING", True)
    )
    pruning_min_size: int = field(
        default_factory=lambda: _env_int("PRUNING_MIN_SIZE", 3)
    )
    exhaustive_pruning_threshold: int = field(
        default_factory=lambda: _env_int("EXHAUSTIVE_PRUNING_THRESHOLD", 8)
    )
    # Downward closure: if alphabet S cannot encode the fold, assume no subset
    # of S can either, and skip those subsets.  This is the Apriori argument
    # and it collapses most of the exhaustive search.  Set False for a fully
    # unconditional sweep.
    pruning_downward_closure: bool = field(
        default_factory=lambda: _env_bool("PRUNING_DOWNWARD_CLOSURE", True)
    )
    prune_folds: int = field(default_factory=lambda: _env_int("PRUNE_FOLDS", 3))

    # ---------------- caching / housekeeping ----------------
    fold_cache_path: str = field(
        default_factory=lambda: _env_str("FOLD_CACHE_PATH", "")
    )
    use_fold_cache: bool = field(
        default_factory=lambda: _env_bool("USE_FOLD_CACHE", True)
    )
    save_fold_pdbs: bool = field(
        default_factory=lambda: _env_bool("SAVE_FOLD_PDBS", False)
    )
    # Wall-clock budget in seconds.  The run stops between replicates once it
    # is exceeded, so a preempted or time-limited job exits with a usable
    # checkpoint instead of being killed mid-write.
    max_wall_seconds: int = field(
        default_factory=lambda: _env_int("MAX_WALL_SECONDS", 0)
    )
    test_mode: bool = False

    # -- derived -------------------------------------------------------
    @property
    def site_enabled(self) -> bool:
        return self.search_mode in (MODE_SCAFFOLD_THEN_SITE, MODE_SITE_AWARE)

    @property
    def usable_aas(self) -> tuple[str, ...]:
        """The residues the search is allowed to consider, in a fixed order."""
        barred = set(self.exclude_aas)
        return tuple(aa for aa in ALL_AAS if aa not in barred)

    @property
    def keep_ligand_tuple(self) -> tuple[str, ...]:
        raw = (self.site_keep_ligands or "").replace(",", " ").split()
        return tuple(dict.fromkeys(x.strip().upper() for x in raw if x.strip()))

    @property
    def site_rest_gated(self) -> bool:
        """Whether the roles outside SITE_GATE_ROLES are gated at all."""
        return any((
            self.site_rest_max_bb_rmsd > 0,
            self.site_rest_max_local_rmsd > 0,
            self.site_rest_max_cb_dev > 0,
            self.site_rest_max_cb_rms_dev > 0,
        ))

    @property
    def gate_role_tuple(self) -> tuple[str, ...]:
        raw = (self.site_gate_roles or "").replace(",", " ").split()
        named = tuple(dict.fromkeys(x.strip().lower() for x in raw if x.strip()))
        return named or SITE_ROLES

    @property
    def keep_partner_tuple(self) -> tuple[str, ...]:
        raw = (self.site_keep_partners or "").replace(",", " ").split()
        return tuple(dict.fromkeys(x.strip() for x in raw if x.strip()))

    # Env var -> config attribute, for the settings a preset may carry.
    ENV_TO_FIELD = {
        "SITE_RESIDUE_POLICY": "site_residue_policy",
        "SITE_GATE_ROLES": "site_gate_roles",
        "SITE_GATE": "site_gate",
        "SITE_MAX_BB_RMSD": "site_max_bb_rmsd",
        "SITE_MAX_LOCAL_RMSD": "site_max_local_rmsd",
        "SITE_MAX_CB_DEV": "site_max_cb_dev",
        "SITE_MAX_CB_RMS_DEV": "site_max_cb_rms_dev",
        "SITE_FREE_EXCLUDED": "site_free_excluded",
        "SITE_REST_MAX_BB_RMSD": "site_rest_max_bb_rmsd",
        "SITE_REST_MAX_LOCAL_RMSD": "site_rest_max_local_rmsd",
        "SITE_REST_MAX_CB_DEV": "site_rest_max_cb_dev",
        "SITE_REST_MAX_CB_RMS_DEV": "site_rest_max_cb_rms_dev",
        "SITE_REQUIRE_IDENTITY": "site_require_identity",
        "SITE_CONTACT_MAX": "site_contact_max",
        "MPNN_MODEL_TYPE": "mpnn_model_type",
    }

    def apply_env_setting(self, key: str, value: str) -> None:
        """Set one field from its environment-variable name and string value.

        Used to apply a preset's recommended settings without going through
        the environment, so the reported configuration is what actually ran.
        """
        name = self.ENV_TO_FIELD.get(key)
        if name is None:
            raise ValueError(f"no configuration field for {key}")
        current = getattr(self, name)
        if isinstance(current, bool):
            setattr(self, name, value.strip().lower() in ("1", "true", "yes", "on"))
        elif isinstance(current, int):
            setattr(self, name, int(value))
        elif isinstance(current, float):
            setattr(self, name, float(value))
        else:
            setattr(self, name, value)

    def validate(self) -> list[str]:
        """Return a list of configuration problems, empty if all is well."""
        problems: list[str] = []
        if self.search_mode not in SEARCH_MODES:
            problems.append(
                f"SEARCH_MODE={self.search_mode!r} is not one of "
                f"{', '.join(SEARCH_MODES)}"
            )
        _policies, policy_problems = parse_site_policy(self.site_residue_policy)
        problems.extend(policy_problems)
        if self.site_rest_gated and not self.site_gate_roles.strip():
            problems.append(
                "SITE_REST_MAX_* thresholds apply to the roles left out of "
                "SITE_GATE_ROLES, but SITE_GATE_ROLES is empty, so every role "
                "is in the tight gate and these would never be used. Name the "
                "tightly-gated roles, e.g. SITE_GATE_ROLES=catalytic,contact"
            )
        bad_roles = [
            r for r in (self.site_gate_roles or "").replace(",", " ").split()
            if r.strip().lower() not in SITE_ROLES
        ]
        if bad_roles:
            problems.append(
                f"SITE_GATE_ROLES names {', '.join(bad_roles)}, which is not "
                f"among {', '.join(SITE_ROLES)}"
            )
        if self.site_enabled and not (self.site_spec or self.site_preset):
            problems.append(
                f"SEARCH_MODE={self.search_mode} needs a site: set SITE_SPEC "
                "or SITE_PRESET"
            )
        if (
            self.search_mode == MODE_SCAFFOLD_THEN_SITE
            and not policy_problems
            and all(p == POLICY_MEASURE_ONLY for p in _policies.values())
        ):
            problems.append(
                "SITE_RESIDUE_POLICY=measure_only has nothing to graft, so it "
                "cannot be combined with SEARCH_MODE=scaffold_then_site; use "
                "SEARCH_MODE=site_aware, or hold one role (for example "
                "'catalytic=exempt,interface=measure_only')"
            )
        barred = set(self.exclude_aas)
        if barred:
            overlap = sorted(barred & set(self.starting_alphabet))
            if overlap:
                problems.append(
                    f"EXCLUDE_AAS and START_ALPHABET both name "
                    f"{', '.join(overlap)}: the search cannot start from a "
                    "residue it may not use"
                )
            if len(barred) >= len(ALL_AAS) - 1:
                problems.append(
                    f"EXCLUDE_AAS bars {len(barred)} of {len(ALL_AAS)} "
                    "residues, leaving nothing to search with"
                )
        if (self.site_spec or self.site_preset) and not self.site_enabled:
            problems.append(
                "a site was given but SEARCH_MODE=alphabet_only, so it would "
                "be ignored; set SEARCH_MODE=site_aware or scaffold_then_site"
            )
        return problems

    def apply_test_mode(self) -> None:
        """Shrink every budget for a ~5 minute single-domain smoke test."""
        self.test_mode = True
        self.num_replicates = 1
        self.chunk_index = 0
        self.chunk_size = 1
        self.mpnn_num_seq = 50
        self.num_seqs_to_fold = 2
        self.screen_folds_stage1 = 1
        self.screen_folds_stage2 = 1
        self.screen_stage2_keep = 2
        self.prune_folds = 1
        self.fold_steps_screen = min(self.fold_steps_screen, 25)
        self.fold_steps_final = min(self.fold_steps_final, 50)

    def describe(self) -> str:
        lines = []
        for f in fields(self):
            lines.append(f"  {f.name:<30s} {getattr(self, f.name)!r}")
        return "\n".join(lines)
