"""Standard mneme directory layout under ~/.claude/mneme."""
from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    override = os.environ.get("MNEME_HOME")
    if override:
        return Path(override)
    return Path.home() / ".claude" / "mneme"


def capabilities_yaml() -> Path:
    return home() / "capabilities.yaml"


def semantic_db() -> Path:
    return home() / "semantic.sqlite"


def procedural_jsonl() -> Path:
    return home() / "procedural.jsonl"


def reflections_jsonl() -> Path:
    return home() / "reflections.jsonl"


def failure_log() -> Path:
    return home() / "failures.log"


def feedback_jsonl() -> Path:
    return home() / "feedback.jsonl"
