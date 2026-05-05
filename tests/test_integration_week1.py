"""End-to-end integration test for Week 1: seed YAML -> store -> retrieve -> render."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mneme.embedder import EMBED_DIM
from mneme.loader import seed_store
from mneme.retrieve import Retriever
from mneme.store import SqliteStore


class _DeterministicEmbedder:
    """Hash-based fake: maps text to a stable unit vector per lowercase content."""

    def __init__(self) -> None:
        self._cache: dict[str, NDArray[np.float32]] = {}

    def embed(self, text: str) -> NDArray[np.float32]:
        key = text.lower()
        if key not in self._cache:
            rng = np.random.default_rng(abs(hash(key)) % (2**32))
            v = rng.standard_normal(EMBED_DIM).astype(np.float32)
            self._cache[key] = v / np.linalg.norm(v)
        return self._cache[key]


def test_seed_then_retrieve_end_to_end(tmp_mneme_dir: Path) -> None:
    repo_root = Path(__file__).parent.parent
    yml_src = repo_root / "src" / "mneme" / "seed" / "capabilities.example.yaml"
    yml = tmp_mneme_dir / "capabilities.yaml"
    yml.write_text(yml_src.read_text(encoding="utf-8"), encoding="utf-8")

    embedder = _DeterministicEmbedder()
    store = SqliteStore(tmp_mneme_dir / "semantic.sqlite")
    n = seed_store(yml, store, embedder=embedder)
    assert n == 10

    retriever = Retriever(store, embedder, threshold=0.0)
    result = retriever.retrieve("playwright_screenshot")
    rendered = result.render()
    assert rendered, "retriever should produce non-empty output for a seeded id query"
    assert rendered.startswith("<capabilities-available>")
    assert rendered.endswith("</capabilities-available>")
    store.close()
