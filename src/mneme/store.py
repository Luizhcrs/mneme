"""Storage layer: SQLite + sqlite-vec for vectors, JSONL for procedural/reflections."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Generic, TypeVar

import numpy as np
import sqlite_vec
from numpy.typing import NDArray
from pydantic import BaseModel

from mneme.embedder import EMBED_DIM
from mneme.schema import CapabilityCard

T = TypeVar("T", bound=BaseModel)


class SqliteStore:
    """SQLite + sqlite-vec store for capability cards and their embeddings.

    Not thread-safe: one instance per process or one per thread. The connection
    uses ``check_same_thread=True`` (sqlite3 default) and will raise
    ``ProgrammingError`` if shared across threads. For Phase 1 the hook runs
    one process per UserPromptSubmit invocation, so this is fine.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn = sqlite3.connect(path)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def __enter__(self) -> SqliteStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS capabilities (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                json TEXT NOT NULL
            )
            """
        )
        cur.execute("PRAGMA table_info(capabilities)")
        cols = {row[1] for row in cur.fetchall()}
        if "active" not in cols:
            cur.execute("ALTER TABLE capabilities ADD COLUMN active INTEGER NOT NULL DEFAULT 1")
        cur.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS capability_vec USING vec0(
                embedding float[{EMBED_DIM}]
            )
            """
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS rowid_to_id (rowid INTEGER PRIMARY KEY, id TEXT NOT NULL)"
        )
        # FTS5 lexical index for hybrid retrieval. The text column packs
        # name + description + triggers + action_verb + example so BM25
        # ranking surfaces cards whose words appear verbatim in the query —
        # complements the cosine layer which surfaces semantically related
        # cards even when no word overlaps.
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS capability_fts USING fts5(
                text, tokenize = 'unicode61 remove_diacritics 2'
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def _fts_text_for(card: CapabilityCard) -> str:
        """Pack searchable text for the FTS5 lexical index.

        Order matters loosely: BM25 weights all tokens equally but having
        the name and triggers first front-loads them in the document for
        readability when debugging.
        """
        return " ".join(
            [
                card.id,
                card.name,
                card.action_verb,
                " ".join(card.triggers),
                card.description,
                card.example,
            ]
        )

    def upsert_capability(self, card: CapabilityCard, embedding: NDArray[np.float32]) -> None:
        if embedding.shape != (EMBED_DIM,):
            raise ValueError(f"embedding shape {embedding.shape} != ({EMBED_DIM},)")
        cur = self._conn.cursor()
        cur.execute("SELECT rowid FROM rowid_to_id WHERE id = ?", (card.id,))
        row = cur.fetchone()
        if row is not None:
            rowid = row[0]
            cur.execute("DELETE FROM capability_vec WHERE rowid = ?", (rowid,))
            cur.execute("DELETE FROM capability_fts WHERE rowid = ?", (rowid,))
            cur.execute("DELETE FROM capabilities WHERE id = ?", (card.id,))
            cur.execute("DELETE FROM rowid_to_id WHERE rowid = ?", (rowid,))

        cur.execute(
            "INSERT INTO capabilities(id, category, active, json) VALUES (?, ?, ?, ?)",
            (card.id, card.category, 1 if card.active else 0, card.model_dump_json()),
        )
        cur.execute("INSERT INTO rowid_to_id(id) VALUES (?)", (card.id,))
        new_rowid = cur.lastrowid
        if new_rowid is None:
            raise RuntimeError(
                "INSERT into rowid_to_id did not return a rowid; database may be corrupt"
            )
        cur.execute(
            "INSERT INTO capability_vec(rowid, embedding) VALUES (?, ?)",
            (new_rowid, sqlite_vec.serialize_float32(embedding.astype(np.float32).tolist())),
        )
        cur.execute(
            "INSERT INTO capability_fts(rowid, text) VALUES (?, ?)",
            (new_rowid, self._fts_text_for(card)),
        )
        self._conn.commit()

    def get_capability(self, card_id: str) -> CapabilityCard | None:
        cur = self._conn.cursor()
        cur.execute("SELECT json FROM capabilities WHERE id = ?", (card_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return CapabilityCard.model_validate_json(row[0])

    def search_capabilities_lexical(
        self,
        query_text: str,
        k: int = 5,
        active_only: bool = True,
    ) -> list[tuple[CapabilityCard, float]]:
        """BM25 lexical search via FTS5. Returns (card, similarity in [0,1])."""
        if not query_text.strip():
            return []
        # Sanitize: keep only word characters and spaces, collapse the rest to
        # avoid FTS5 syntax errors. Wrap each token in OR so partial matches
        # surface (FTS5 default is implicit AND, which is too strict).
        import re as _re
        tokens = [t for t in _re.findall(r"[\w]+", query_text.lower()) if len(t) > 1]
        if not tokens:
            return []
        match_expr = " OR ".join(tokens)
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                SELECT rowid, bm25(capability_fts) AS score
                FROM capability_fts
                WHERE capability_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (match_expr, k * 4),
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            return []
        if not rows:
            return []

        # bm25() returns lower=better with values typically in [-50, 0].
        # Normalize to [0, 1] by min-max within the result set.
        scores = [float(s) for _, s in rows]
        s_min, s_max = min(scores), max(scores)
        spread = s_max - s_min if s_max > s_min else 1.0

        out: list[tuple[CapabilityCard, float]] = []
        for rowid, raw_score in rows:
            cur.execute("SELECT id FROM rowid_to_id WHERE rowid = ?", (rowid,))
            id_row = cur.fetchone()
            if id_row is None:
                continue
            card = self.get_capability(id_row[0])
            if card is None:
                continue
            if active_only and not card.active:
                continue
            similarity = 1.0 - ((float(raw_score) - s_min) / spread)
            out.append((card, similarity))
            if len(out) >= k:
                break
        return out

    def search_capabilities(
        self,
        query: NDArray[np.float32],
        k: int = 5,
        categories: list[str] | None = None,
        threshold: float = 0.0,
        active_only: bool = True,
    ) -> list[tuple[CapabilityCard, float]]:
        if query.shape != (EMBED_DIM,):
            raise ValueError(f"query shape {query.shape} != ({EMBED_DIM},)")

        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT rowid, distance
            FROM capability_vec
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (sqlite_vec.serialize_float32(query.astype(np.float32).tolist()), k * 4),
        )
        candidates = cur.fetchall()

        out: list[tuple[CapabilityCard, float]] = []
        for rowid, distance in candidates:
            cur.execute("SELECT id FROM rowid_to_id WHERE rowid = ?", (rowid,))
            row = cur.fetchone()
            if row is None:
                continue
            card = self.get_capability(row[0])
            if card is None:
                continue
            if active_only and not card.active:
                continue
            if categories is not None and card.category not in categories:
                continue
            # sqlite-vec returns Euclidean (L2) distance. For unit-normalized
            # vectors, the relation to cosine similarity is:
            #   ||u - v||^2 = 2 - 2 * cos(u, v)
            # so cos = 1 - L2^2 / 2. Both index and query vectors are L2
            # normalized by OllamaEmbedder, so this conversion is exact.
            l2 = float(distance)
            similarity = 1.0 - (l2 * l2) / 2.0
            if similarity < threshold:
                continue
            out.append((card, similarity))
            if len(out) >= k:
                break
        return out

    def close(self) -> None:
        self._conn.close()


class JsonlStore(Generic[T]):
    """Append-only JSONL store for workflows or reflections."""

    def __init__(self, path: Path, model: type[T]) -> None:
        self._path = path
        self._model = model
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    def append(self, item: T) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(item.model_dump_json() + "\n")

    def iter_all(self) -> Iterator[T]:
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield self._model.model_validate_json(line)
