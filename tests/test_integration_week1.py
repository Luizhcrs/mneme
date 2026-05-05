"""End-to-end integration test for Week 1: seed YAML -> store -> retrieve -> render."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mneme.embedder import EMBED_DIM
from mneme.loader import seed_store
from mneme.retrieve import Retriever
from mneme.store import SqliteStore


def _stable_seed(text: str) -> int:
    """Cross-platform deterministic seed. Python's `hash()` randomizes per process."""
    digest = hashlib.md5(text.lower().encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


class _DeterministicEmbedder:
    """Hash-based fake: maps text to a stable unit vector per lowercase content."""

    def __init__(self) -> None:
        self._cache: dict[str, NDArray[np.float32]] = {}

    def embed(self, text: str) -> NDArray[np.float32]:
        key = text.lower()
        if key not in self._cache:
            rng = np.random.default_rng(_stable_seed(key))
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

    # Disable filtering so the deterministic-fake embedder (random unit vectors,
    # no semantic structure) does not get rejected by the category or threshold
    # gates. Goal of this test is to validate end-to-end plumbing, not retrieval
    # quality — that is gated separately by the benchmark with real Ollama.
    retriever = Retriever(
        store,
        embedder,
        top_categories=35,
        top_capabilities=10,
        threshold=0.0,
    )
    seeded_text_for_telegram = (
        "Telegram Send Message. sends a message to a Telegram chat. "
        "MCP Telegram plugin. Sends text to a chat_id. "
        "Triggers: telegram, manda mensagem, notifica, alerta."
    )
    result = retriever.retrieve(seeded_text_for_telegram)
    rendered = result.render()
    assert rendered, "retriever should produce non-empty output for a seeded text"
    assert rendered.startswith("<capabilities-available>")
    assert rendered.endswith("</capabilities-available>")
    assert "telegram_send" in rendered
    store.close()
