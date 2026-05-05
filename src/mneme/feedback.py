"""Active feedback loop.

When a retrieval misses (the agent picked the wrong tool, or mneme
ranked the wrong card top-1), the user runs ``mneme correct <query>
<tool_id>`` to record what the answer should have been. Each correction
is stored as a JSONL entry with the query embedding pre-computed so the
retrieval hot path does not have to re-embed the historical corrections
on every turn.

At retrieval time the Retriever scans the feedback log and boosts the
corrected tool when the current query embeds close to a recorded one
(cosine >= match_threshold). The boost is implemented by inserting the
corrected card at rank 1 of the lexical list before RRF fusion, which
keeps the structure honest: corrections are evidence, not overrides.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mneme.embedder import EMBED_DIM


@dataclass
class FeedbackEntry:
    query: str
    tool_id: str
    embedding: NDArray[np.float32]
    timestamp: str


class FeedbackStore:
    """Append-only JSONL store of user corrections.

    Each line is ``{"ts": ..., "query": ..., "tool_id": ..., "embedding": [...]}``.
    The embedding is L2-normalized float32 so cosine reduces to dot product.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    def append(self, query: str, tool_id: str, embedding: NDArray[np.float32]) -> None:
        if embedding.shape != (EMBED_DIM,):
            raise ValueError(f"embedding shape {embedding.shape} != ({EMBED_DIM},)")
        with self._path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "query": query,
                        "tool_id": tool_id,
                        "embedding": embedding.astype(np.float32).tolist(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def iter_all(self) -> Iterator[FeedbackEntry]:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                vec_raw = data.get("embedding") or []
                if len(vec_raw) != EMBED_DIM:
                    continue
                yield FeedbackEntry(
                    query=str(data.get("query", "")),
                    tool_id=str(data.get("tool_id", "")),
                    embedding=np.asarray(vec_raw, dtype=np.float32),
                    timestamp=str(data.get("ts", "")),
                )
