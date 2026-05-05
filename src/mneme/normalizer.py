"""Normalize verbose descriptions into EasyTool canonical form (Phase 1: rule-based)."""
from __future__ import annotations

import re

_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_MD_ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_MAX_LEN = 200
_VERBS = (
    "sends",
    "creates",
    "reads",
    "writes",
    "runs",
    "queries",
    "captures",
    "navigates",
    "lists",
    "deletes",
    "fetches",
    "scrapes",
    "renders",
    "executes",
    "analyzes",
)


def _strip_markdown(text: str) -> str:
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_ITALIC.sub(r"\1", text)
    return text


def normalize_description(text: str, name: str) -> str:
    """Compress a verbose description into <=200 chars while keeping the salient verb.

    Rule-based for Phase 1: strips markdown noise, then if the result is short enough
    returns it as-is. Otherwise, ranks sentences by name-mention and verb presence,
    keeps as many as fit under the limit, and re-orders them by original position.
    """
    text = _strip_markdown(text).strip()
    if len(text) <= _MAX_LEN:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    name_lower = name.lower()

    scored: list[tuple[float, int, str]] = []
    for idx, raw_sentence in enumerate(sentences):
        s = raw_sentence.strip()
        if not s:
            continue
        score = 0.0
        if name_lower and name_lower in s.lower():
            score += 2.0
        if any(verb in s.lower() for verb in _VERBS):
            score += 1.5
        score -= idx * 0.1
        scored.append((score, idx, s))

    scored.sort(key=lambda t: (-t[0], t[1]))
    chosen: list[tuple[int, str]] = []
    total = 0
    for _, idx, s in scored:
        added = len(s) + (1 if chosen else 0)
        if total + added > _MAX_LEN:
            continue
        chosen.append((idx, s))
        total += added

    chosen.sort(key=lambda t: t[0])
    return " ".join(s for _, s in chosen)
