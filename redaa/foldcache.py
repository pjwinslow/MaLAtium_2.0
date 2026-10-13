"""Persistent cache of ESMFold2 predictions, keyed by sequence.

Folding is the pipeline's whole cost, and the search folds the same sequence
more than once far more often than you would guess: ProteinMPNN under a
six-letter alphabet produces duplicates within a single call, the forward
search re-tests alphabets that the backward pass then re-tests again, and ten
replicates of the same domain explore heavily overlapping sequence space.
A hit here is a fold that costs nothing.

One SQLite file holds the lot.  SQLite is in the standard library, gives
atomic writes, survives preemption mid-run, and -- unlike one small file per
prediction -- does not create tens of thousands of inodes on shared storage,
which is what makes a Lustre or NFS scratch directory crawl.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS folds (
    key     TEXT PRIMARY KEY,
    n_res   INTEGER NOT NULL,
    plddt   REAL,
    ptm     REAL,
    coords  BLOB NOT NULL
);
"""


def fold_key(sequence: str, model_id: str, steps: int, loops: int, seed: int) -> str:
    """Stable cache key.

    The sequence is hashed rather than stored so the key stays short and the
    index stays small; the model and sampler settings are part of the key
    because they change the prediction.
    """
    digest = hashlib.sha1(sequence.encode("ascii")).hexdigest()
    return f"{model_id}|{steps}|{loops}|{seed}|{digest}"


class FoldCache:
    """Thread-safe, process-safe cache of ``(plddt, ptm, CA coords)``.

    Coordinates are stored as float32, which is ~4 KB for a 300-residue
    domain and well inside the precision that an RMSD or TM-score needs.
    """

    def __init__(self, path: str | None, enabled: bool = True):
        self.enabled = bool(enabled and path)
        self.path = path
        self._lock = threading.Lock()
        self._mem: dict[str, tuple[float | None, float | None, np.ndarray]] = {}
        self._conn: sqlite3.Connection | None = None
        self.hits = 0
        self.misses = 0

        if self.enabled:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            try:
                self._conn = sqlite3.connect(path, timeout=60.0, check_same_thread=False)
                # WAL lets a reader proceed while another array task writes;
                # the busy timeout covers the brief exclusive locks.
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.execute("PRAGMA busy_timeout=60000")
                self._conn.executescript(_SCHEMA)
                self._conn.commit()
            except Exception as exc:  # noqa: BLE001
                print(f"  [fold cache] disabled ({exc})")
                self.enabled = False
                self._conn = None

    # -- read -------------------------------------------------------------
    def get(self, key: str):
        """Return ``(plddt, ptm, coords)`` or ``None``."""
        hit = self._mem.get(key)
        if hit is None and self._conn is not None:
            try:
                with self._lock:
                    row = self._conn.execute(
                        "SELECT plddt, ptm, coords FROM folds WHERE key = ?", (key,)
                    ).fetchone()
            except Exception:
                row = None
            if row is not None:
                coords = np.frombuffer(row[2], dtype=np.float32).reshape(-1, 3)
                hit = (row[0], row[1], coords.astype(np.float64))
                self._mem[key] = hit
        if hit is None:
            self.misses += 1
            return None
        self.hits += 1
        return hit

    # -- write ------------------------------------------------------------
    def put(
        self,
        key: str,
        plddt: float | None,
        ptm: float | None,
        coords: np.ndarray,
    ) -> None:
        self._mem[key] = (plddt, ptm, coords)
        if self._conn is None:
            return
        blob = np.ascontiguousarray(coords, dtype=np.float32).tobytes()
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO folds (key, n_res, plddt, ptm, coords) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (key, len(coords), plddt, ptm, blob),
                )
                self._conn.commit()
        except Exception as exc:  # noqa: BLE001
            # A cache write failure must never take the run down; we just lose
            # the reuse.
            print(f"  [fold cache] write failed ({exc})")

    def stats(self) -> str:
        total = self.hits + self.misses
        pct = (100.0 * self.hits / total) if total else 0.0
        return f"{self.hits}/{total} lookups served from cache ({pct:.1f}%)"

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None
