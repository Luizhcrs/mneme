"""Reflexion: turn failure logs into searchable lessons.

Phase 2 minimum-viable: rule-based templating. When a tool exit code is
non-zero the PostToolUse hook writes a structured entry to failures.log;
``mneme reflect`` reads those entries and produces templated reflections
into reflections.jsonl. Each reflection carries the situation embedding
so the Retriever can surface it on similar future queries.

Phase 3 will replace the templating with an LLM-generated reflection
(Reflexion paper, Shinn et al. 2023). The interface is intentionally
identical so swapping is a one-line change.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mneme.embedder import EMBED_DIM
from mneme.types import EmbedderProto


@dataclass
class Reflection:
    situation: str
    tool: str
    lesson: str
    embedding: NDArray[np.float32]
    timestamp: str


def _read_failure_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _templated_lesson(entry: dict[str, Any]) -> str:
    tool = str(entry.get("tool", "unknown"))
    exit_code = entry.get("exit_code", "?")
    error = str(entry.get("error", "")).strip()
    if error:
        snippet = error[:140]
        return (
            f"Last time {tool} ran on a similar task it failed with "
            f"exit_code={exit_code}: {snippet}. Consider an alternative or "
            f"address the underlying error before retrying."
        )
    return (
        f"Last time {tool} ran on a similar task it failed with "
        f"exit_code={exit_code}. Investigate before retrying."
    )


class ReflectionStore:
    """Append-only JSONL log of reflections with embeddings."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    def append(self, reflection: Reflection) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": reflection.timestamp,
                        "situation": reflection.situation,
                        "tool": reflection.tool,
                        "lesson": reflection.lesson,
                        "embedding": reflection.embedding.astype(np.float32).tolist(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def iter_all(self) -> Iterator[Reflection]:
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
                vec = data.get("embedding") or []
                if len(vec) != EMBED_DIM:
                    continue
                yield Reflection(
                    situation=str(data.get("situation", "")),
                    tool=str(data.get("tool", "")),
                    lesson=str(data.get("lesson", "")),
                    embedding=np.asarray(vec, dtype=np.float32),
                    timestamp=str(data.get("ts", "")),
                )


def consolidate(
    failure_log_path: Path,
    reflection_store_path: Path,
    embedder: EmbedderProto,
) -> int:
    """Walk the failure log, write a reflection for any entry not yet covered.

    Idempotent: each reflection is keyed by (situation, tool) so re-runs
    do not produce duplicates. Returns the number of new reflections
    written this run.
    """
    failures = _read_failure_log(failure_log_path)
    if not failures:
        return 0

    store = ReflectionStore(reflection_store_path)
    seen: set[tuple[str, str]] = set()
    for r in store.iter_all():
        seen.add((r.situation, r.tool))

    written = 0
    for entry in failures:
        situation = str(entry.get("prompt", ""))[:500]
        tool = str(entry.get("tool", ""))
        if not situation or not tool:
            continue
        key = (situation, tool)
        if key in seen:
            continue
        seen.add(key)
        lesson = _templated_lesson(entry)
        embedding = embedder.embed(situation)
        store.append(
            Reflection(
                situation=situation,
                tool=tool,
                lesson=lesson,
                embedding=embedding,
                timestamp=str(entry.get("ts", datetime.now(UTC).isoformat())),
            )
        )
        written += 1
    return written
