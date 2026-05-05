"""Two-stage retrieval: category top-3 then capability top-k filtered by category."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from mneme.categories import CATEGORY_DESCRIPTIONS
from mneme.schema import CapabilityCard
from mneme.store import SqliteStore


@runtime_checkable
class _EmbedderProto(Protocol):
    def embed(self, text: str) -> NDArray[np.float32]: ...


@dataclass
class RetrievalResult:
    capabilities: list[tuple[CapabilityCard, float]] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    def render(self) -> str:
        if not self.capabilities:
            return ""
        lines = ["<capabilities-available>"]
        for card, score in self.capabilities:
            lines.append(
                f"- [{card.id}] {card.name} ({card.category}, score={score:.2f})\n"
                f"  verb: {card.action_verb}\n"
                f"  params_required: {card.params_required}\n"
                f"  example: {card.example.strip()}"
            )
        lines.append("</capabilities-available>")
        return "\n".join(lines)


class Retriever:
    """Two-stage retrieval per AnyTool: category filter then capability search."""

    def __init__(
        self,
        store: SqliteStore,
        embedder: _EmbedderProto,
        top_categories: int = 3,
        top_capabilities: int = 5,
        threshold: float = 0.65,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._top_categories = top_categories
        self._top_capabilities = top_capabilities
        self._threshold = threshold
        self._category_vectors = self._build_category_index()

    def _build_category_index(self) -> dict[str, NDArray[np.float32]]:
        return {
            name: self._embedder.embed(desc)
            for name, desc in CATEGORY_DESCRIPTIONS.items()
        }

    def _top_categories_for(self, query_vec: NDArray[np.float32]) -> list[str]:
        scored = [
            (name, float(np.dot(query_vec, vec)))
            for name, vec in self._category_vectors.items()
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [name for name, _ in scored[: self._top_categories]]

    def retrieve(self, prompt: str) -> RetrievalResult:
        query_vec = self._embedder.embed(prompt)
        cats = self._top_categories_for(query_vec)
        caps = self._store.search_capabilities(
            query_vec,
            k=self._top_capabilities,
            categories=cats,
            threshold=self._threshold,
        )
        return RetrievalResult(capabilities=caps, categories=cats)
