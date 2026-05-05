"""Tests for two-stage retrieval."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from mneme.embedder import EMBED_DIM
from mneme.retrieve import Retriever
from mneme.schema import CapabilityCard
from mneme.store import SqliteStore


class _FakeEmbedder:
    """Deterministic stand-in: maps text to fixed vectors per substring keyword."""

    def __init__(self, mapping: dict[str, NDArray[np.float32]]) -> None:
        self._mapping = mapping

    def embed(self, text: str) -> NDArray[np.float32]:
        for key, vec in self._mapping.items():
            if key in text:
                return vec
        return np.zeros(EMBED_DIM, dtype=np.float32)


def _unit(seed: int) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_card(card_id: str, category: str) -> CapabilityCard:
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


@pytest.fixture
def seeded_store(tmp_mneme_dir: Path) -> SqliteStore:
    store = SqliteStore(tmp_mneme_dir / "semantic.sqlite")
    store.upsert_capability(_make_card("playwright", "web_browser"), _unit(1))
    store.upsert_capability(_make_card("read_file", "filesystem"), _unit(2))
    store.upsert_capability(_make_card("telegram", "comms"), _unit(3))
    return store


def test_retrieves_top_capabilities_for_matching_query(seeded_store: SqliteStore) -> None:
    web_vec = _unit(1)
    embedder = _FakeEmbedder({"web_browser": web_vec, "screenshot": web_vec})
    retriever = Retriever(
        seeded_store, embedder, top_categories=3, top_capabilities=2, threshold=0.5
    )
    result = retriever.retrieve("screenshot the homepage")
    ids = [c.id for c, _ in result.capabilities]
    assert "playwright" in ids


def test_returns_empty_below_threshold(seeded_store: SqliteStore) -> None:
    embedder = _FakeEmbedder({})  # all zeros, dissimilar to everything
    retriever = Retriever(seeded_store, embedder, threshold=0.65)
    result = retriever.retrieve("totally unrelated nonsense")
    assert result.capabilities == []


def test_format_injection_places_capabilities_block(seeded_store: SqliteStore) -> None:
    web_vec = _unit(1)
    embedder = _FakeEmbedder({"web_browser": web_vec, "screenshot": web_vec})
    retriever = Retriever(seeded_store, embedder, threshold=0.5)
    result = retriever.retrieve("screenshot please")
    rendered = result.render()
    assert rendered.startswith("<capabilities-available>")
    assert rendered.endswith("</capabilities-available>")
    assert "playwright" in rendered


def test_render_empty_when_no_results(seeded_store: SqliteStore) -> None:
    embedder = _FakeEmbedder({})
    retriever = Retriever(seeded_store, embedder, threshold=0.65)
    result = retriever.retrieve("nope")
    assert result.render() == ""


def test_retriever_includes_top_workflows(tmp_mneme_dir: Path, seeded_store: SqliteStore) -> None:
    from mneme.schema import Workflow
    from mneme.store import JsonlStore

    wf_store = JsonlStore[Workflow](tmp_mneme_dir / "procedural.jsonl", Workflow)
    wf_store.append(
        Workflow(
            situation="screenshot example.com",
            sequence=["playwright_goto", "playwright_screenshot"],
            outcome="success",
        )
    )

    web_vec = _unit(1)
    embedder = _FakeEmbedder({"screenshot": web_vec, "headless": web_vec})
    retriever = Retriever(
        seeded_store,
        embedder,
        threshold=0.0,
        workflow_store=wf_store,
        top_workflows=3,
    )
    result = retriever.retrieve("screenshot the homepage")
    assert any(seq[0] == "playwright_goto" for seq in result.workflows)
    rendered = result.render()
    assert "workflow that worked before" in rendered


def test_category_filter_excludes_capabilities_outside_top_categories(
    seeded_store: SqliteStore,
) -> None:
    """With top_categories=1, only capabilities in the highest-scoring category surface.

    The seeded store has 3 capabilities in 3 distinct categories
    (web_browser, filesystem, comms). The fake embedder uses "headless" as a
    discriminator that appears only in the web_browser category description.
    With top_categories=1 the filter must pick web_browser exclusively, so
    the filesystem and comms capabilities are excluded even though they exist.
    """
    web_vec = _unit(1)
    embedder = _FakeEmbedder({"headless": web_vec})
    retriever = Retriever(
        seeded_store,
        embedder,
        top_categories=1,
        top_capabilities=10,
        threshold=0.0,
    )
    result = retriever.retrieve("need a headless browser session")
    ids = [c.id for c, _ in result.capabilities]
    assert "playwright" in ids
    assert "read_file" not in ids
    assert "telegram" not in ids
