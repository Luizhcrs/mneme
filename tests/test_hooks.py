"""Tests for hook entry points."""
from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import pytest
import respx

from mneme.embedder import EMBED_DIM
from mneme.hooks.user_prompt_submit import run_hook
from mneme.loader import seed_store
from mneme.store import SqliteStore


@pytest.fixture
def seeded_env(tmp_mneme_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set MNEME_HOME, seed a single-card YAML, build the semantic DB."""
    monkeypatch.setenv("MNEME_HOME", str(tmp_mneme_dir))

    yml = tmp_mneme_dir / "capabilities.yaml"
    yml.write_text(
        """
- id: telegram_send
  name: Telegram Send
  category: comms
  action_verb: send a Telegram message
  triggers: [telegram, manda mensagem]
  description: send a message via Telegram MCP
  params_required: [chat_id, text]
  params_optional: []
  example: 'telegram.send(chat_id=X, text=Y)'
  schema_version: "1"
  source: mcp
""",
        encoding="utf-8",
    )

    # Seed with a constant embedder so we can predict embeddings at test time.
    import numpy as np
    from numpy.typing import NDArray

    class _ConstantEmbedder:
        def embed(self, text: str) -> NDArray[np.float32]:
            v = np.full(EMBED_DIM, 0.1, dtype=np.float32)
            return v / np.linalg.norm(v)

    store = SqliteStore(tmp_mneme_dir / "semantic.sqlite")
    seed_store(yml, store, embedder=_ConstantEmbedder())
    store.close()
    return tmp_mneme_dir


def test_hook_injects_capabilities_when_match(seeded_env: Path) -> None:
    # Mock Ollama to return embeddings that make comms a top category.
    # For prompts and descriptions containing "telegram" or "Telegram",
    # return the same embedding (high similarity).
    def embedding_response(request: httpx.Request) -> httpx.Response:
        import hashlib

        import numpy as np

        payload = json.loads(request.content)
        prompt = payload.get("prompt", "")

        # Return a constant embedding for "telegram" prompts and comms category.
        # All other categories get unique but consistent embeddings.
        if "telegram" in prompt.lower() or "email SMS Telegram" in prompt:
            embedding = [0.1] * EMBED_DIM
        else:
            # Use the prompt hash to seed a deterministic but different embedding
            h = hashlib.md5(prompt.encode()).digest()
            seed = int.from_bytes(h[:4], "little")
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(EMBED_DIM).astype(np.float32)
            embedding = (v / np.linalg.norm(v)).tolist()
        return httpx.Response(200, json={"embedding": embedding})

    with respx.mock(base_url="http://localhost:11434", assert_all_called=False) as r:
        r.post("/api/embeddings").mock(side_effect=embedding_response)
        stdin = io.StringIO(json.dumps({"prompt": "manda mensagem no telegram avisando deploy"}))
        stdout = io.StringIO()
        rc = run_hook(stdin=stdin, stdout=stdout)
    assert rc == 0
    out = stdout.getvalue()
    assert "<capabilities-available>" in out
    assert "telegram_send" in out


def test_hook_passes_through_when_db_missing(
    tmp_mneme_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MNEME_HOME", str(tmp_mneme_dir))
    stdin = io.StringIO(json.dumps({"prompt": "anything"}))
    stdout = io.StringIO()
    rc = run_hook(stdin=stdin, stdout=stdout)
    assert rc == 0
    assert stdout.getvalue() == ""


def test_hook_fails_safe_when_ollama_down(seeded_env: Path) -> None:
    with respx.mock(base_url="http://localhost:11434", assert_all_called=False) as r:
        r.post("/api/embeddings").mock(side_effect=httpx.ConnectError("down"))
        stdin = io.StringIO(json.dumps({"prompt": "manda mensagem no telegram"}))
        stdout = io.StringIO()
        rc = run_hook(stdin=stdin, stdout=stdout)
    # Phase 1 fail-safe: hook does not block; exits 0 with empty output.
    assert rc == 0
    assert stdout.getvalue() == ""


def test_hook_passes_through_on_invalid_json(seeded_env: Path) -> None:
    stdin = io.StringIO("not valid json {{{")
    stdout = io.StringIO()
    rc = run_hook(stdin=stdin, stdout=stdout)
    assert rc == 0
    assert stdout.getvalue() == ""


def test_hook_passes_through_on_empty_prompt(seeded_env: Path) -> None:
    stdin = io.StringIO(json.dumps({"prompt": ""}))
    stdout = io.StringIO()
    rc = run_hook(stdin=stdin, stdout=stdout)
    assert rc == 0
    assert stdout.getvalue() == ""
