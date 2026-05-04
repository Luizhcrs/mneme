"""Local embedding via Ollama nomic-embed-text with instruction prefix."""
from __future__ import annotations

import httpx
import numpy as np
from numpy.typing import NDArray

EMBED_DIM = 768
DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_HOST = "http://localhost:11434"
INSTRUCTION_PREFIX = "Given this task intent, retrieve tools that enable it: "
DEFAULT_TIMEOUT = 30.0


class OllamaEmbedder:
    """Embeds text via a local Ollama HTTP server.

    Always prepends the instruction prefix at index time and query time, per
    "Retrieval Models Aren't Tool-Savvy" (ACL 2025): generic IR embeddings
    score nDCG@10 = 33.83 on tool retrieval; instruction-tuning is the
    cheapest mitigation.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout

    def embed(self, text: str) -> NDArray[np.float32]:
        prompt = f"{INSTRUCTION_PREFIX}{text}"
        try:
            response = httpx.post(
                f"{self._host}/api/embeddings",
                json={"model": self._model, "prompt": prompt},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama at {self._host} returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}. "
                f"Confirm the model `{self._model}` is pulled (`ollama pull {self._model}`)."
            ) from exc
        except httpx.HTTPError as exc:
            raise ConnectionError(
                f"Ollama at {self._host} unreachable: {exc}. "
                f"Run `ollama serve` and `ollama pull {self._model}`."
            ) from exc

        raw = response.json()["embedding"]
        if len(raw) != EMBED_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected {EMBED_DIM}, got {len(raw)}"
            )

        vec = np.asarray(raw, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm == 0:
            raise ValueError("Ollama returned a zero-norm embedding; cannot normalize.")
        return vec / norm
