"""Tests for the reflexion consolidation and retrieval flow."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mneme.embedder import EMBED_DIM
from mneme.reflexion import ReflectionStore, consolidate
from mneme.retrieve import Retriever
from mneme.schema import CapabilityCard
from mneme.store import SqliteStore


def _unit(seed: int) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class _FakeEmbedder:
    def __init__(self, vec: NDArray[np.float32]) -> None:
        self._vec = vec

    def embed(self, text: str) -> NDArray[np.float32]:
        return self._vec


def _make_card(card_id: str) -> CapabilityCard:
    return CapabilityCard(
        id=card_id,
        name=card_id,
        category="web_browser",
        action_verb="x",
        triggers=["x"],
        description="x",
        params_required=[],
        params_optional=[],
        example="x",
        schema_version="1",
        source="mcp",
    )


def test_consolidate_writes_one_reflection_per_failure(tmp_mneme_dir: Path) -> None:
    log_path = tmp_mneme_dir / "failures.log"
    log_path.write_text(
        '{"ts": "2026-05-05T10:00:00Z", "tool": "playwright", '
        '"prompt": "screenshot a SPA", "exit_code": 1, '
        '"error": "selector timeout"}\n'
        '{"ts": "2026-05-05T11:00:00Z", "tool": "telegram", '
        '"prompt": "send alert", "exit_code": 2, '
        '"error": "auth failed"}\n',
        encoding="utf-8",
    )
    refl_path = tmp_mneme_dir / "reflections.jsonl"
    embedder = _FakeEmbedder(_unit(1))
    written = consolidate(log_path, refl_path, embedder)
    assert written == 2

    store = ReflectionStore(refl_path)
    items = list(store.iter_all())
    assert len(items) == 2
    assert any("playwright" in r.lesson and "selector timeout" in r.lesson for r in items)


def test_consolidate_is_idempotent(tmp_mneme_dir: Path) -> None:
    log_path = tmp_mneme_dir / "failures.log"
    log_path.write_text(
        '{"ts": "x", "tool": "playwright", "prompt": "screenshot", '
        '"exit_code": 1, "error": "boom"}\n',
        encoding="utf-8",
    )
    refl_path = tmp_mneme_dir / "reflections.jsonl"
    embedder = _FakeEmbedder(_unit(1))
    consolidate(log_path, refl_path, embedder)
    second = consolidate(log_path, refl_path, embedder)
    assert second == 0


def test_retriever_surfaces_reflection_for_similar_prompt(tmp_mneme_dir: Path) -> None:
    log_path = tmp_mneme_dir / "failures.log"
    log_path.write_text(
        '{"ts": "x", "tool": "playwright", '
        '"prompt": "screenshot the homepage", '
        '"exit_code": 1, "error": "selector timeout"}\n',
        encoding="utf-8",
    )
    refl_path = tmp_mneme_dir / "reflections.jsonl"
    consolidate(log_path, refl_path, _FakeEmbedder(_unit(42)))

    store = SqliteStore(tmp_mneme_dir / "semantic.sqlite")
    store.upsert_capability(_make_card("a"), _unit(1))

    embedder = _FakeEmbedder(_unit(42))
    refl_store = ReflectionStore(refl_path)
    retriever = Retriever(
        store,
        embedder,
        top_categories=35,
        threshold=-1.0,
        reflection_store=refl_store,
        reflection_threshold=0.5,
    )
    result = retriever.retrieve("any prompt")
    assert any("playwright" in lesson for lesson in result.reflections)


def test_render_includes_reflection_block(tmp_mneme_dir: Path) -> None:
    refl_path = tmp_mneme_dir / "reflections.jsonl"
    log_path = tmp_mneme_dir / "failures.log"
    log_path.write_text(
        '{"ts": "x", "tool": "telegram", "prompt": "send alert", '
        '"exit_code": 2, "error": "auth"}\n',
        encoding="utf-8",
    )
    consolidate(log_path, refl_path, _FakeEmbedder(_unit(7)))

    store = SqliteStore(tmp_mneme_dir / "semantic.sqlite")
    store.upsert_capability(_make_card("a"), _unit(1))

    retriever = Retriever(
        store,
        _FakeEmbedder(_unit(7)),
        top_categories=35,
        threshold=-1.0,
        reflection_store=ReflectionStore(refl_path),
        reflection_threshold=0.5,
    )
    rendered = retriever.retrieve("any").render()
    assert "reflection from past failure" in rendered
    assert "telegram" in rendered
