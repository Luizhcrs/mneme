"""Tests for the active feedback loop."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mneme.embedder import EMBED_DIM
from mneme.feedback import FeedbackStore
from mneme.retrieve import Retriever
from mneme.schema import CapabilityCard
from mneme.store import SqliteStore


def _unit(seed: int) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_card(card_id: str, category: str = "web_browser") -> CapabilityCard:
    return CapabilityCard(
        id=card_id,
        name=card_id,
        category=category,
        action_verb="x",
        triggers=["x"],
        description="x",
        params_required=[],
        params_optional=[],
        example="x",
        schema_version="1",
        source="mcp",
    )


class _FakeEmbedder:
    """Constant-vector embedder so feedback similarity is deterministic."""

    def __init__(self, vec: NDArray[np.float32]) -> None:
        self._vec = vec

    def embed(self, text: str) -> NDArray[np.float32]:
        return self._vec


def test_feedback_store_roundtrip(tmp_mneme_dir: Path) -> None:
    store = FeedbackStore(tmp_mneme_dir / "feedback.jsonl")
    store.append(query="screenshot a site", tool_id="playwright_screenshot", embedding=_unit(1))
    entries = list(store.iter_all())
    assert len(entries) == 1
    assert entries[0].query == "screenshot a site"
    assert entries[0].tool_id == "playwright_screenshot"
    assert entries[0].embedding.shape == (EMBED_DIM,)


def test_feedback_store_skips_malformed_lines(tmp_mneme_dir: Path) -> None:
    path = tmp_mneme_dir / "feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"ts": "x", "query": "ok", "tool_id": "t", "embedding": [0.1]}\n'
        "garbage line\n"
        '{"ts": "y", "query": "bad", "tool_id": "u"}\n',
        encoding="utf-8",
    )
    store = FeedbackStore(path)
    entries = list(store.iter_all())
    # First entry has wrong-dim embedding; third is missing embedding.
    # All three lines are skipped, so iter returns nothing.
    assert entries == []


def test_retriever_promotes_corrected_tool(tmp_mneme_dir: Path) -> None:
    store = SqliteStore(tmp_mneme_dir / "semantic.sqlite")
    store.upsert_capability(_make_card("a"), _unit(1))
    store.upsert_capability(_make_card("b", "filesystem"), _unit(2))
    store.upsert_capability(_make_card("c", "comms"), _unit(3))

    fb = FeedbackStore(tmp_mneme_dir / "feedback.jsonl")
    fb.append(query="screenshot the homepage", tool_id="b", embedding=_unit(42))

    embedder = _FakeEmbedder(_unit(42))
    retriever = Retriever(
        store,
        embedder,
        top_categories=35,
        top_capabilities=5,
        threshold=-1.0,
        feedback_store=fb,
        feedback_threshold=0.5,
    )
    result = retriever.retrieve("screenshot the homepage please")
    ids = [card.id for card, _ in result.capabilities]
    assert ids[0] == "b"


def test_retriever_ignores_feedback_below_threshold(tmp_mneme_dir: Path) -> None:
    store = SqliteStore(tmp_mneme_dir / "semantic.sqlite")
    store.upsert_capability(_make_card("a"), _unit(1))
    store.upsert_capability(_make_card("b"), _unit(2))

    fb = FeedbackStore(tmp_mneme_dir / "feedback.jsonl")
    fb.append(query="totally unrelated thing", tool_id="b", embedding=_unit(99))

    embedder = _FakeEmbedder(_unit(1))
    retriever = Retriever(
        store,
        embedder,
        top_categories=35,
        top_capabilities=5,
        threshold=-1.0,
        feedback_store=fb,
        feedback_threshold=0.95,
    )
    result = retriever.retrieve("query")
    ids = [card.id for card, _ in result.capabilities]
    assert ids[0] == "a"
