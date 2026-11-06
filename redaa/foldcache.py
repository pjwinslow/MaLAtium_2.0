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

# Added after the first release, so it arrives by migration rather than being
# part of the table definition.  Holds the N/CA/C backbone frame per residue,
# which is what the functional-site metrics need and the CA-only `coords`
# column cannot provide.  Nullable on purpose: a row written by an older run
# has no frames, and a site-aware run treats such a row as a miss rather than
# silently reporting nothing for the site.
_MIGRATIONS = (
    ("frames", "ALTER TABLE folds ADD COLUMN frames BLOB"),
)


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
                self._migrate()
                self._conn.commit()
            except Exception as exc:  # noqa: BLE001
                print(f"  [fold cache] disabled ({exc})")
                self.enabled = False
                self._conn = None

    def _migrate(self) -> None:
        """Add columns introduced after a cache file was first written."""
        if self._conn is None:
            return
        have = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(folds)").fetchall()
        }
        for column, ddl in _MIGRATIONS:
            if column not in have:
                self._conn.execute(ddl)

    # -- read -------------------------------------------------------------
    def get(self, key: str, need_frames: bool = False):
        """Return ``(plddt, ptm, coords, frames)`` or ``None``.

        ``frames`` is ``(L, 3, 3)`` of N/CA/C, or ``None`` for a row written
        before that column existed.  With ``need_frames`` such a row is
        reported as a miss: a site-aware run would otherwise inherit a hit it
        cannot measure the site from, and quietly produce blank site columns
        for every sequence an earlier site-blind run had already folded.
        """
        hit = self._mem.get(key)
        if hit is None and self._conn is not None:
            try:
                with self._lock:
                    row = self._conn.execute(
                        "SELECT plddt, ptm, coords, frames FROM folds WHERE key = ?",
                        (key,),
                    ).fetchone()
            except Exception:
                row = None
            if row is not None:
                coords = np.frombuffer(row[2], dtype=np.float32).reshape(-1, 3)
                frames = None
                if row[3] is not None:
                    frames = (
                        np.frombuffer(row[3], dtype=np.float32)
                        .reshape(-1, 3, 3)
                        .astype(np.float64)
                    )
                hit = (row[0], row[1], coords.astype(np.float64), frames)
                self._mem[key] = hit
        if hit is None:
            self.misses += 1
            return None
        if need_frames and hit[3] is None:
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
        frames: np.ndarray | None = None,
    ) -> None:
        self._mem[key] = (plddt, ptm, coords, frames)
        if self._conn is None:
            return
        blob = np.ascontiguousarray(coords, dtype=np.float32).tobytes()
        fblob = (
            None
            if frames is None
            else np.ascontiguousarray(frames, dtype=np.float32).tobytes()
        )
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO folds "
                    "(key, n_res, plddt, ptm, coords, frames) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (key, len(coords), plddt, ptm, blob, fblob),
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
