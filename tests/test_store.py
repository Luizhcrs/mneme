"""Tests for SqliteStore and JsonlStore."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mneme.embedder import EMBED_DIM
from mneme.schema import CapabilityCard, Reflection, Workflow
from mneme.store import JsonlStore, SqliteStore


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


def _vec(seed: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed * 1000))
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def test_sqlite_store_insert_and_search(tmp_mneme_dir: Path) -> None:
    store = SqliteStore(tmp_mneme_dir / "semantic.sqlite")
    store.upsert_capability(_make_card("a"), _vec(0.1))
    store.upsert_capability(_make_card("b", "filesystem"), _vec(0.2))

    results = store.search_capabilities(_vec(0.1), k=2)
    assert len(results) == 2
    assert results[0][0].id == "a"
    assert 0.0 <= results[0][1] <= 1.0


def test_sqlite_store_filters_by_category(tmp_mneme_dir: Path) -> None:
    store = SqliteStore(tmp_mneme_dir / "semantic.sqlite")
    store.upsert_capability(_make_card("a", "web_browser"), _vec(0.1))
    store.upsert_capability(_make_card("b", "filesystem"), _vec(0.2))

    results = store.search_capabilities(_vec(0.1), k=5, categories=["filesystem"])
    assert len(results) == 1
    assert results[0][0].id == "b"


def test_sqlite_store_threshold_filter(tmp_mneme_dir: Path) -> None:
    store = SqliteStore(tmp_mneme_dir / "semantic.sqlite")
    store.upsert_capability(_make_card("a"), _vec(0.1))

    far_query = -1 * _vec(0.1)
    results = store.search_capabilities(far_query, k=5, threshold=0.65)
    assert results == []


def test_sqlite_store_upsert_replaces(tmp_mneme_dir: Path) -> None:
    store = SqliteStore(tmp_mneme_dir / "semantic.sqlite")
    card = _make_card("a")
    store.upsert_capability(card, _vec(0.1))
    card.success_count = 5
    store.upsert_capability(card, _vec(0.1))

    fetched = store.get_capability("a")
    assert fetched is not None
    assert fetched.success_count == 5


def test_jsonl_store_workflow_roundtrip(tmp_mneme_dir: Path) -> None:
    store = JsonlStore[Workflow](tmp_mneme_dir / "procedural.jsonl", Workflow)
    wf = Workflow(situation="x", sequence=["a", "b"], outcome="success")
    store.append(wf)
    items = list(store.iter_all())
    assert len(items) == 1
    assert items[0].sequence == ["a", "b"]


def test_jsonl_store_reflection_roundtrip(tmp_mneme_dir: Path) -> None:
    store = JsonlStore[Reflection](tmp_mneme_dir / "reflections.jsonl", Reflection)
    r = Reflection(capability_id="a", situation="x", error="boom")
    store.append(r)
    items = list(store.iter_all())
    assert items[0].error == "boom"


def test_sqlite_store_context_manager_closes(tmp_mneme_dir: Path) -> None:
    path = tmp_mneme_dir / "semantic.sqlite"
    with SqliteStore(path) as store:
        store.upsert_capability(_make_card("a"), _vec(0.1))
        assert store.get_capability("a") is not None
    reopened = SqliteStore(path)
    assert reopened.get_capability("a") is not None
    reopened.close()


def test_sqlite_store_rejects_wrong_shape_on_upsert(tmp_mneme_dir: Path) -> None:
    store = SqliteStore(tmp_mneme_dir / "semantic.sqlite")
    bad = np.zeros(100, dtype=np.float32)
    with pytest.raises(ValueError, match="embedding shape"):
        store.upsert_capability(_make_card("a"), bad)


def test_sqlite_store_rejects_wrong_shape_on_search(tmp_mneme_dir: Path) -> None:
    store = SqliteStore(tmp_mneme_dir / "semantic.sqlite")
    bad = np.zeros(100, dtype=np.float32)
    with pytest.raises(ValueError, match="query shape"):
        store.search_capabilities(bad)
