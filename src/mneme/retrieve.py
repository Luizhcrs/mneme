"""Two-stage retrieval: category top-3 then capability top-k filtered by category."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mneme.categories import CATEGORY_DESCRIPTIONS
from mneme.feedback import FeedbackStore
from mneme.reflexion import ReflectionStore
from mneme.schema import CapabilityCard, Workflow
from mneme.store import JsonlStore, SqliteStore
from mneme.types import EmbedderProto


@dataclass
class RetrievalResult:
    capabilities: list[tuple[CapabilityCard, float]] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    workflows: list[list[str]] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Render injection block. Caller MUST place this at the START of the prompt
        (Lost in the Middle, TACL 2024 — middle-of-context content gets ignored).

        The block opens with an explicit behavioral directive: do not claim a
        limitation when one of the listed capabilities matches the task. The
        agent is biased toward fallbacks (curl, regex, manual parsing) when
        unsure; a passive list of names is not enough — the directive raises
        the bar to "use one of these or justify why none fit".
        """
        if not self.capabilities and not self.workflows and not self.reflections:
            return ""
        lines = [
            "<capabilities-available>",
            "INSTRUCTION: Before responding that you cannot do something, scan",
            "this list. If any entry matches the user's task, you HAVE that tool",
            "and MUST attempt it before falling back to manual workarounds (curl,",
            "regex, screenshots-by-hand, etc.). Stating a limitation that the list",
            "contradicts is a hallucination — do not do it.",
            "",
        ]
        for card, score in self.capabilities:
            lines.append(
                f"- [{card.id}] {card.name} ({card.category}, score={score:.2f})\n"
                f"  verb: {card.action_verb}\n"
                f"  description: {card.description.strip()[:200]}\n"
                f"  triggers: {card.triggers[:8]}\n"
                f"  example: {card.example.strip()[:140]}"
            )
        for seq in self.workflows:
            lines.append(f"- workflow that worked before: {' -> '.join(seq)}")
        for lesson in self.reflections:
            lines.append(f"- reflection from past failure: {lesson}")
        lines.append("</capabilities-available>")
        return "\n".join(lines)


class Retriever:
    """Two-stage retrieval per AnyTool: category filter then capability search.

    Stage 2 is a hybrid retriever: it merges semantic (cosine) ranking with
    lexical (BM25) ranking via Reciprocal Rank Fusion. RRF (Cormack et al.,
    2009) is robust to score-scale differences across rankers; each result's
    fused score is sum(1 / (k_rrf + rank)) over the lists where it appears.
    The hybrid surfaces cards that match the query semantically AND/OR
    lexically, which fixes failure modes where a strong PT-BR/EN keyword
    overlap dominates pure cosine (e.g. 'tira print do site' was beating
    web-browser cards because pyautogui's seed had stronger PT-BR triggers
    than the discovered Playwright manifest).
    """

    def __init__(
        self,
        store: SqliteStore,
        embedder: EmbedderProto,
        top_categories: int = 10,
        top_capabilities: int = 5,
        threshold: float = 0.65,
        workflow_store: JsonlStore[Workflow] | None = None,
        top_workflows: int = 3,
        rrf_k: int = 60,
        lexical_weight: float = 1.0,
        feedback_store: FeedbackStore | None = None,
        feedback_threshold: float = 0.80,
        reflection_store: ReflectionStore | None = None,
        reflection_threshold: float = 0.75,
        top_reflections: int = 2,
    ) -> None:
        """Default top_categories=10 — empirically validated on the 50-task
        benchmark with real Ollama nomic-embed-text. Lower values (3) over-
        filter at small registry sizes (10 capabilities, 35 categories) and
        binary-eject correct results that lost the category sort to noise.
        Larger registries (100+ capabilities) can tighten this back to 3-5.

        ``rrf_k`` is the RRF constant; 60 is the value reported by Cormack et
        al. ``lexical_weight`` multiplies the BM25 ranker's contribution
        before fusion (1.0 = equal weight with cosine; 0 = pure semantic).
        """
        self._store = store
        self._embedder = embedder
        self._top_categories = top_categories
        self._top_capabilities = top_capabilities
        self._threshold = threshold
        self._workflow_store = workflow_store
        self._top_workflows = top_workflows
        self._rrf_k = rrf_k
        self._lexical_weight = lexical_weight
        self._feedback_store = feedback_store
        self._feedback_threshold = feedback_threshold
        self._reflection_store = reflection_store
        self._reflection_threshold = reflection_threshold
        self._top_reflections = top_reflections
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

    def _retrieve_workflows(self, query_vec: NDArray[np.float32]) -> list[list[str]]:
        if self._workflow_store is None:
            return []
        scored: list[tuple[float, list[str]]] = []
        for wf in self._workflow_store.iter_all():
            wf_vec = self._embedder.embed(wf.situation)
            sim = float(np.dot(query_vec, wf_vec))
            if sim >= self._threshold:
                scored.append((sim, list(wf.sequence)))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [seq for _, seq in scored[: self._top_workflows]]

    def _rrf_fuse(
        self,
        semantic: list[tuple[CapabilityCard, float]],
        lexical: list[tuple[CapabilityCard, float]],
    ) -> list[tuple[CapabilityCard, float]]:
        """Reciprocal Rank Fusion: combine two ranked lists into one.

        Each card's RRF score is ``sum(weight_i / (k + rank_i))`` where
        weight_i is 1.0 for the semantic list and ``self._lexical_weight``
        for the lexical list. Cards that appear in both lists get a strong
        boost; cards that only appear in one still surface if their rank
        is high.
        """
        scores: dict[str, float] = {}
        cards: dict[str, CapabilityCard] = {}
        for rank, (card, _) in enumerate(semantic):
            scores[card.id] = scores.get(card.id, 0.0) + 1.0 / (self._rrf_k + rank + 1)
            cards[card.id] = card
        for rank, (card, _) in enumerate(lexical):
            scores[card.id] = scores.get(card.id, 0.0) + self._lexical_weight / (
                self._rrf_k + rank + 1
            )
            cards[card.id] = card
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(cards[cid], score) for cid, score in ranked[: self._top_capabilities]]

    def _feedback_boost(
        self,
        query_vec: NDArray[np.float32],
        merged: list[tuple[CapabilityCard, float]],
    ) -> list[tuple[CapabilityCard, float]]:
        """Promote a corrected tool when the current query matches a logged one.

        For each user correction in the feedback log, compute cosine between
        the current query and the recorded query. If any recorded query is
        within ``feedback_threshold`` (default 0.80, treated as a near-
        duplicate intent), find the corresponding card in the merged result
        list and lift it to rank 1 with a score equal to the best-fused
        score plus a small epsilon. Cards not in the merged list are
        ignored — feedback boosts existing candidates, it does not invent
        new retrievals (which would require a fresh embed of the corrected
        card on every turn).
        """
        if self._feedback_store is None or not merged:
            return merged
        best_match: tuple[float, str] | None = None
        for entry in self._feedback_store.iter_all():
            sim = float(np.dot(query_vec, entry.embedding))
            if sim < self._feedback_threshold:
                continue
            if best_match is None or sim > best_match[0]:
                best_match = (sim, entry.tool_id)
        if best_match is None:
            return merged

        target_id = best_match[1]
        promoted_index: int | None = None
        for i, (card, _) in enumerate(merged):
            if card.id == target_id:
                promoted_index = i
                break
        if promoted_index is None or promoted_index == 0:
            return merged
        promoted_card, _ = merged[promoted_index]
        new_top_score = merged[0][1] + 1e-3
        rest = [pair for i, pair in enumerate(merged) if i != promoted_index]
        return [(promoted_card, new_top_score), *rest][: self._top_capabilities]

    def _retrieve_reflections(self, query_vec: NDArray[np.float32]) -> list[str]:
        if self._reflection_store is None:
            return []
        scored: list[tuple[float, str]] = []
        for r in self._reflection_store.iter_all():
            sim = float(np.dot(query_vec, r.embedding))
            if sim >= self._reflection_threshold:
                scored.append((sim, r.lesson))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [lesson for _, lesson in scored[: self._top_reflections]]

    def retrieve(self, prompt: str) -> RetrievalResult:
        query_vec = self._embedder.embed(prompt)
        cats = self._top_categories_for(query_vec)
        semantic = self._store.search_capabilities(
            query_vec,
            k=self._top_capabilities * 3,
            categories=cats,
            threshold=self._threshold,
        )
        lexical = self._store.search_capabilities_lexical(
            prompt,
            k=self._top_capabilities * 3,
        )
        if cats:
            lexical = [(c, s) for c, s in lexical if c.category in cats]
        caps = self._rrf_fuse(semantic, lexical)
        caps = self._feedback_boost(query_vec, caps)
        workflows = self._retrieve_workflows(query_vec)
        reflections = self._retrieve_reflections(query_vec)
        return RetrievalResult(
            capabilities=caps,
            categories=cats,
            workflows=workflows,
            reflections=reflections,
        )
