"""Local embedding via Ollama nomic-embed-text with instruction prefix."""
from __future__ import annotations

import httpx
import numpy as np

EMBED_DIM = 768
DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_HOST = "http://localhost:11434"
INSTRUCTION_PREFIX = "Given this task intent, retrieve tools that enable it: "


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
        timeout: float = 5.0,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout

    def embed(self, text: str) -> np.ndarray:
        prompt = f"{INSTRUCTION_PREFIX}{text}"
        try:
            response = httpx.post(
                f"{self._host}/api/embeddings",
                json={"model": self._model, "prompt": prompt},
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise ConnectionError(
                f"Ollama at {self._host} unreachable: {exc}. "
                f"Run `ollama serve` and `ollama pull {self._model}`."
            ) from exc

        response.raise_for_status()
        raw = response.json()["embedding"]
        if len(raw) != EMBED_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected {EMBED_DIM}, got {len(raw)}"
            )

        vec = np.asarray(raw, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec
