"""Contract tests for the in-process LigandMPNN integration.

Every bug this layer has had was in the same blind spot: the stub engines used
elsewhere in the suite imitate what *this* code expects, so they agree with it
even when it is wrong, and nothing else in the repo runs LigandMPNN. The first
real GPU run then failed with ``KeyError: 'chain_mask'`` -- raised inside
``data_utils.featurize``, on a key ``parse_PDB`` never writes, because
upstream's ``run.py`` builds it between the two calls and an in-process caller
has to do the same.

So the fakes here imitate **LigandMPNN**, not this pipeline. Each one encodes
a requirement read out of the real source:

* ``parse_PDB`` returns exactly the keys the real one returns -- and no
  ``chain_mask``.
* ``featurize`` raises ``KeyError`` for every input it reads, so a missing one
  fails here the way it fails there.
* the ligand branch raises ``IndexError`` when asked for nearest neighbours
  among zero ligand atoms, which is what really happens in
  ``get_nearest_neighbours`` at ``L2_AB_nn[:, 0]``.

Torch is faked over numpy, so this runs with no GPU, no torch and no
checkpoint, like the rest of the suite.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redaa.config import RedAAConfig  # noqa: E402

# Keys data_utils.parse_PDB actually populates (read from its source). The
# absence of "chain_mask" is the point.
PARSE_PDB_KEYS = (
    "X", "S", "R_idx", "chain_labels", "chain_letters", "chain_list",
    "mask", "mask_c", "Y", "Y_t", "Y_m",
)

# Keys data_utils.featurize reads from its input dict.
FEATURIZE_REQUIRES = (
    "X", "S", "R_idx", "chain_labels", "chain_mask", "mask",
)
FEATURIZE_REQUIRES_LIGAND = ("Y", "Y_t", "Y_m")

AA = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# A numpy-backed stand-in for torch
# ---------------------------------------------------------------------------


class T:
    """Just enough tensor for this module: shape, clone, indexing, repeat."""

    def __init__(self, array):
        self.a = np.asarray(array)

    @property
    def shape(self):
        return self.a.shape

    def clone(self):
        return T(self.a.copy())

    def detach(self):
        return self

    def to(self, *_args, **_kwargs):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.a

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = tuple(k.a if isinstance(k, T) else k for k in key)
        out = self.a[key]
        return T(out) if isinstance(out, np.ndarray) else out

    def __setitem__(self, key, value):
        self.a[key] = value.a if isinstance(value, T) else value

    def __mul__(self, other):
        return T(self.a * (other.a if isinstance(other, T) else other))

    __rmul__ = __mul__

    def repeat(self, *sizes):
        return T(np.tile(self.a, sizes))

    def __len__(self):
        return len(self.a)


def _fake_torch() -> types.ModuleType:
    mod = types.ModuleType("torch")
    mod.float32 = np.float32
    mod.int32 = np.int32

    def _unwrap(x):
        return x.a if isinstance(x, T) else x

    mod.tensor = lambda data, **kw: T(_unwrap(data))
    mod.ones_like = lambda x, **kw: T(np.ones_like(_unwrap(x)))
    mod.zeros_like = lambda x, **kw: T(np.zeros_like(_unwrap(x)))
    mod.ones = lambda size, **kw: T(np.ones(size))
    mod.full = lambda size, value, **kw: T(np.full(size, value))
    mod.randn = lambda size, **kw: T(np.zeros(size))
    mod.device = lambda spec: spec
    mod.manual_seed = lambda *_a: None
    mod.inference_mode = lambda: _NullContext()

    cuda = types.SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda *_a: None,
        empty_cache=lambda: None,
        OutOfMemoryError=RuntimeError,
    )
    mod.cuda = cuda
    mod.OutOfMemoryError = RuntimeError
    return mod


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


# ---------------------------------------------------------------------------
# A stand-in for LigandMPNN's data_utils
# ---------------------------------------------------------------------------


def _fake_data_utils(n_res: int, n_ligand_atoms: int, chains="A") -> types.ModuleType:
    mod = types.ModuleType("data_utils")
    mod.restype_str_to_int = {c: i for i, c in enumerate(AA)}
    mod.restype_str_to_int["X"] = 20
    mod.restype_int_to_str = {i: c for c, i in mod.restype_str_to_int.items()}
    mod.calls = {"parse": 0, "featurize": [], "nearest_neighbours": 0}

    native = [mod.restype_str_to_int[AA[i % 20]] for i in range(n_res)]

    def parse_PDB(path, device="cpu", parse_all_atoms=False,
                  parse_atoms_with_zero_occupancy=False):
        mod.calls["parse"] += 1
        letters = [chains[i % len(chains)] for i in range(n_res)]
        out = {
            "X": T(np.zeros((n_res, 4, 3))),
            "S": T(np.array(native, dtype=np.int64)),
            "R_idx": T(np.arange(1, n_res + 1)),
            "chain_labels": T(np.zeros(n_res)),
            "chain_letters": letters,
            "chain_list": sorted(set(letters)),
            "mask": T(np.ones(n_res)),
            "mask_c": T(np.ones(n_res)),
            "Y": T(np.zeros((n_ligand_atoms, 3))),
            "Y_t": T(np.zeros(n_ligand_atoms)),
            "Y_m": T(np.ones(n_ligand_atoms)),
        }
        assert set(out) == set(PARSE_PDB_KEYS), "fake drifted from the real keys"
        # The real parse_PDB returns (protein_dict, backbone, other_atoms,
        # icodes, _).
        return out, None, None, [""] * n_res, None

    def featurize(input_dict, cutoff_for_score=8.0, use_atom_context=True,
                  number_of_ligand_atoms=16, model_type="protein_mpnn"):
        mod.calls["featurize"].append(
            {
                "use_atom_context": use_atom_context,
                "number_of_ligand_atoms": number_of_ligand_atoms,
                "model_type": model_type,
                "cutoff_for_score": cutoff_for_score,
            }
        )
        required = list(FEATURIZE_REQUIRES)
        if model_type == "ligand_mpnn":
            required += list(FEATURIZE_REQUIRES_LIGAND)
        for key in required:
            if key not in input_dict:
                # Exactly what the real one does, and the failure that
                # reached Colab.
                raise KeyError(key)

        if model_type == "ligand_mpnn":
            # get_nearest_neighbours() takes L2_AB_nn[:, 0], which indexes an
            # empty dimension when there are no ligand atoms.
            mod.calls["nearest_neighbours"] += 1
            if input_dict["Y"].shape[0] == 0:
                raise IndexError(
                    "index 0 is out of bounds for dimension 1 with size 0"
                )

        out = {
            "S": T(np.asarray(input_dict["S"].a)[None, :]),
            "X": T(np.asarray(input_dict["X"].a)[None, ...]),
            "mask": T(np.asarray(input_dict["mask"].a)[None, :]),
            "chain_mask": T(np.asarray(input_dict["chain_mask"].a)[None, :]),
            "R_idx": T(np.asarray(input_dict["R_idx"].a)[None, :]),
        }
        return out

    mod.parse_PDB = parse_PDB
    mod.featurize = featurize
    mod.get_score = lambda S, log_probs, mask: (T(np.zeros(len(S.a))), None)
    return mod


def _engine(n_res=12, n_ligand_atoms=0, model_type="protein_mpnn", chains="A"):
    """An MPNNEngine wired to the fakes, with load() bypassed."""
    sys.modules["torch"] = _fake_torch()
    from redaa.mpnn import MPNNEngine

    cfg = RedAAConfig()
    cfg.mpnn_model_type = model_type
    eng = MPNNEngine(cfg, verbose=False)
    eng.model = object()          # stands in for a loaded checkpoint
    eng.device = "cpu"
    eng._atom_context_num = 25
    eng._mods = {
        "data_utils": _fake_data_utils(n_res, n_ligand_atoms, chains),
        "model_utils": types.ModuleType("model_utils"),
    }
    return eng


def _cleanup():
    sys.modules.pop("torch", None)


# ---------------------------------------------------------------------------
# The contract that broke
# ---------------------------------------------------------------------------


def test_parse_pdb_output_alone_is_not_featurisable():
    """Documents the trap: parse_PDB does not produce what featurize needs.

    If this ever stops raising, the upstream contract changed and the
    workaround in features_for can be reconsidered.
    """
    du = _fake_data_utils(8, 0)
    protein_dict, *_ = du.parse_PDB("x.pdb")
    assert "chain_mask" not in protein_dict
    try:
        du.featurize(protein_dict)
    except KeyError as exc:
        assert "chain_mask" in str(exc)
    else:
        raise AssertionError("expected KeyError: 'chain_mask'")


def test_features_for_supplies_the_chain_mask_featurize_requires():
    """The actual regression: KeyError: 'chain_mask' on the first real run."""
    try:
        eng = _engine(n_res=10)
        feats = eng.features_for("dom.pdb")
        assert "chain_mask" in feats
        assert feats["chain_mask"].shape == (1, 10)
        # Everything designable by default; fixed positions come later.
        assert feats["chain_mask"].a.sum() == 10
    finally:
        _cleanup()


def test_chain_mask_covers_every_chain_in_the_file():
    try:
        eng = _engine(n_res=9, chains="AB")
        feats = eng.features_for("dom.pdb")
        # run.py designs every chain in chains_to_design, which defaults to
        # all of them; a chain silently left out would be held native.
        assert feats["chain_mask"].a.sum() == 9
    finally:
        _cleanup()


def test_features_are_computed_once_per_domain():
    try:
        eng = _engine()
        eng.features_for("dom.pdb")
        eng.features_for("dom.pdb")
        eng.features_for("dom.pdb")
        assert eng._mods["data_utils"].calls["parse"] == 1
    finally:
        _cleanup()


def test_release_forgets_the_domain_and_its_native_sequence():
    try:
        eng = _engine()
        eng.features_for("dom.pdb")
        assert eng.native_sequence("dom.pdb")
        eng.release("dom.pdb")
        assert "dom.pdb" not in eng._features
        assert "dom.pdb" not in eng._native
        assert "dom.pdb" not in eng._base_chain_mask
    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# Ligand context
# ---------------------------------------------------------------------------


def test_ligand_model_without_a_ligand_says_what_to_do():
    """Better than the IndexError this would otherwise become."""
    try:
        eng = _engine(n_res=10, n_ligand_atoms=0, model_type="ligand_mpnn")
        try:
            eng.features_for("dom.pdb")
        except RuntimeError as exc:
            message = str(exc)
            assert "no ligand atoms" in message
            assert "SITE_KEEP_LIGANDS" in message
            assert "protein_mpnn" in message
            assert "dom.pdb" in message
        except IndexError:
            raise AssertionError(
                "fell through to LigandMPNN's IndexError instead of "
                "explaining the problem"
            )
        else:
            raise AssertionError("expected RuntimeError")
        # It failed before featurize, so nothing half-built is cached.
        assert "dom.pdb" not in eng._features
    finally:
        _cleanup()


def test_ligand_model_with_a_ligand_turns_atom_context_on():
    try:
        eng = _engine(n_res=10, n_ligand_atoms=31, model_type="ligand_mpnn")
        eng.features_for("dom.pdb")
        call = eng._mods["data_utils"].calls["featurize"][0]
        assert call["model_type"] == "ligand_mpnn"
        assert call["use_atom_context"] is True
        # From the checkpoint, not a guess: the ligand model is _25.
        assert call["number_of_ligand_atoms"] == 25
        assert eng.ligand_atom_count("dom.pdb") == 31
    finally:
        _cleanup()


def test_protein_model_keeps_atom_context_off_even_with_a_ligand_present():
    """protein_mpnn was never trained to read the ligand tensor."""
    try:
        eng = _engine(n_res=10, n_ligand_atoms=31, model_type="protein_mpnn")
        eng.features_for("dom.pdb")
        call = eng._mods["data_utils"].calls["featurize"][0]
        assert call["model_type"] == "protein_mpnn"
        assert call["use_atom_context"] is False
        # And the ligand branch was never entered.
        assert eng._mods["data_utils"].calls["nearest_neighbours"] == 0
    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# Fixed positions
# ---------------------------------------------------------------------------


def test_native_sequence_comes_from_the_featurised_tensor():
    try:
        eng = _engine(n_res=10)
        native = eng.native_sequence("dom.pdb")
        assert native == AA[:10]
    finally:
        _cleanup()


def test_fixed_positions_zero_the_chain_mask():
    try:
        eng = _engine(n_res=10)
        eng.features_for("dom.pdb")
        mask = eng._chain_mask_for("dom.pdb", 10, {2: "C", 7: "H"})
        assert mask.shape == (1, 10)
        assert mask.a[0, 2] == 0.0 and mask.a[0, 7] == 0.0
        # LigandMPNN reads 0 as "keep the native residue here" and 1 as
        # "design this position"; everything else must stay designable.
        assert mask.a.sum() == 8
    finally:
        _cleanup()


def test_fixed_positions_are_also_pinned_in_the_bias():
    try:
        eng = _engine(n_res=6)
        eng.features_for("dom.pdb")
        bias = eng._bias_tensor(["G", "A"], 6, fixed={1: "C"})
        assert bias.shape == (1, 6, 21)
        idx = eng._mods["data_utils"].restype_str_to_int

        # A designable position allows exactly the alphabet.
        row = bias.a[0, 0]
        assert row[idx["G"]] == 0.0 and row[idx["A"]] == 0.0
        assert row[idx["C"]] < -1e6
        assert row[idx["X"]] < -1e6, "X must not be samplable"

        # The fixed position allows only its native residue -- which is
        # outside the alphabet, and that is the whole point.
        row = bias.a[0, 1]
        assert row[idx["C"]] == 0.0
        assert row[idx["G"]] < -1e6 and row[idx["A"]] < -1e6
    finally:
        _cleanup()


def test_enforce_fixed_reports_when_it_had_to_correct_something():
    from redaa.mpnn import _enforce_fixed

    seq, moved = _enforce_fixed("GGGGGG", {1: "C", 4: "H"})
    assert seq == "GCGGHG" and moved == 2
    seq, moved = _enforce_fixed(seq, {1: "C", 4: "H"})
    assert moved == 0, "already-correct positions must not be counted"


def main():
    failures = []
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc!r}")
            import traceback

            traceback.print_exc()
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
