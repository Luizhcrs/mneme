"""Tests for OllamaEmbedder."""
from __future__ import annotations

from collections.abc import Iterator

import httpx
import numpy as np
import pytest
import respx

from mneme.embedder import EMBED_DIM, INSTRUCTION_PREFIX, OllamaEmbedder


@pytest.fixture
def mock_ollama() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url="http://localhost:11434") as r:
        r.post("/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": [0.1] * EMBED_DIM})
        )
        yield r


def test_embedder_uses_instruction_prefix(mock_ollama: respx.MockRouter) -> None:
    embedder = OllamaEmbedder()
    vec = embedder.embed("screenshot the homepage")
    assert vec.shape == (EMBED_DIM,)
    assert vec.dtype == np.float32
    request = mock_ollama.calls.last.request
    body = request.read().decode()
    assert INSTRUCTION_PREFIX in body
    assert "screenshot the homepage" in body


def test_embedder_returns_l2_normalized(mock_ollama: respx.MockRouter) -> None:
    embedder = OllamaEmbedder()
    vec = embedder.embed("anything")
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5


def test_embedder_raises_on_dimension_mismatch() -> None:
    with respx.mock(base_url="http://localhost:11434") as r:
        r.post("/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": [0.1] * 100})
        )
        embedder = OllamaEmbedder()
        with pytest.raises(ValueError, match="expected 768"):
            embedder.embed("x")


def test_embedder_raises_on_ollama_unreachable() -> None:
    with respx.mock(base_url="http://localhost:11434") as r:
        r.post("/api/embeddings").mock(side_effect=httpx.ConnectError("connection refused"))
        embedder = OllamaEmbedder()
        with pytest.raises(ConnectionError, match="Ollama"):
            embedder.embed("x")


def test_embedder_raises_runtime_error_on_http_status_error() -> None:
    with respx.mock(base_url="http://localhost:11434") as r:
        r.post("/api/embeddings").mock(
            return_value=httpx.Response(404, json={"error": "model 'unknown' not found"})
        )
        embedder = OllamaEmbedder(model="unknown")
        with pytest.raises(RuntimeError, match="returned HTTP 404"):
            embedder.embed("x")


def test_embedder_raises_on_zero_norm_embedding() -> None:
    with respx.mock(base_url="http://localhost:11434") as r:
        r.post("/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": [0.0] * EMBED_DIM})
        )
        embedder = OllamaEmbedder()
        with pytest.raises(ValueError, match="zero-norm"):
            embedder.embed("x")
