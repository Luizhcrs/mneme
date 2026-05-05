"""YAML capability loader and seeding helper."""
from __future__ import annotations

from pathlib import Path

import yaml

from mneme.embedder import OllamaEmbedder
from mneme.schema import CapabilityCard
from mneme.store import SqliteStore
from mneme.types import EmbedderProto


def load_capabilities(path: Path) -> list[CapabilityCard]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(
            f"YAML root must be a list of capability cards, got {type(raw).__name__} in {path}"
        )
    cards = [CapabilityCard.model_validate(item) for item in raw]
    seen: set[str] = set()
    for card in cards:
        if card.id in seen:
            raise ValueError(f"duplicate capability id '{card.id}' in {path}")
        seen.add(card.id)
    return cards


def seed_store(
    yaml_path: Path,
    store: SqliteStore,
    embedder: EmbedderProto | None = None,
) -> int:
    """Load YAML and embed-and-upsert each card into the store. Returns count."""
    embedder = embedder or OllamaEmbedder()
    cards = load_capabilities(yaml_path)
    for card in cards:
        text_for_embed = (
            f"{card.name}. {card.action_verb}. {card.description} "
            f"Triggers: {', '.join(card.triggers)}."
        )
        vec = embedder.embed(text_for_embed)
        store.upsert_capability(card, vec)
    return len(cards)
