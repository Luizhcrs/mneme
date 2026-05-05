"""Tests for the YAML capability loader."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mneme.embedder import EMBED_DIM
from mneme.loader import load_capabilities, seed_store
from mneme.store import SqliteStore


def test_load_capabilities_parses_yaml(tmp_path: Path) -> None:
    yml = tmp_path / "caps.yaml"
    yml.write_text(
        """
- id: a
  name: A
  category: web_browser
  action_verb: x
  triggers: [t]
  description: d
  params_required: []
  params_optional: []
  example: e
  schema_version: "1"
  source: mcp
""",
        encoding="utf-8",
    )
    cards = load_capabilities(yml)
    assert len(cards) == 1
    assert cards[0].id == "a"


def test_load_capabilities_rejects_duplicate_ids(tmp_path: Path) -> None:
    yml = tmp_path / "caps.yaml"
    body = "\n".join(
        "- {id: dup, name: x, category: web_browser, action_verb: x, "
        "triggers: [t], description: d, params_required: [], params_optional: [], "
        "example: e, schema_version: '1', source: mcp}"
        for _ in range(2)
    )
    yml.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_capabilities(yml)


def test_load_capabilities_handles_empty_file(tmp_path: Path) -> None:
    yml = tmp_path / "empty.yaml"
    yml.write_text("", encoding="utf-8")
    cards = load_capabilities(yml)
    assert cards == []


def test_load_capabilities_loads_seed_example() -> None:
    seed_path = (
        Path(__file__).parent.parent
        / "src"
        / "mneme"
        / "seed"
        / "capabilities.example.yaml"
    )
    cards = load_capabilities(seed_path)
    assert len(cards) == 10
    assert {c.id for c in cards} == {
        "playwright_screenshot",
        "playwright_goto",
        "filesystem_read",
        "telegram_send",
        "github_create_issue",
        "ollama_generate",
        "docker_compose_up",
        "postgres_query",
        "obsidian_search_vault",
        "pyautogui_screenshot",
    }


def test_seed_store_inserts_all(tmp_mneme_dir: Path) -> None:
    yml = tmp_mneme_dir / "caps.yaml"
    yml.write_text(
        """
- id: a
  name: A
  category: web_browser
  action_verb: x
  triggers: [t]
  description: d
  params_required: []
  params_optional: []
  example: e
  schema_version: "1"
  source: mcp
""",
        encoding="utf-8",
    )

    # Use a fake embedder so the test does not require Ollama running.

    class _FakeEmbedder:
        def embed(self, text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            v = rng.standard_normal(EMBED_DIM).astype(np.float32)
            return v / np.linalg.norm(v)

    store = SqliteStore(tmp_mneme_dir / "s.sqlite")
    n = seed_store(yml, store, embedder=_FakeEmbedder())  # type: ignore[arg-type]
    assert n == 1
    assert store.get_capability("a") is not None
    store.close()
