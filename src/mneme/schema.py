"""Pydantic schemas for capability cards, workflows, and reflections."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

CATEGORIES: frozenset[str] = frozenset(
    {
        "web_browser",
        "web_api",
        "filesystem",
        "comms",
        "code_search",
        "vcs",
        "data_pipeline",
        "db",
        "ml_inference",
        "ml_training",
        "nlp",
        "vision_image",
        "voice_audio",
        "audio_media",
        "desktop_automation",
        "mobile_dev",
        "vault_kb",
        "documentation",
        "deploy",
        "cloud_infra",
        "container",
        "kubernetes",
        "monitoring",
        "testing",
        "security",
        "crypto",
        "agent_orchestration",
        "automation_rpa",
        "math_scientific",
        "finance_trading",
        "geo_maps",
        "3d_graphics",
        "game_engine",
        "hardware_io",
        "embedded_iot",
    }
)

ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

Source = Literal["mcp", "skill", "plugin", "command", "project", "hermes", "os", "manual"]
Outcome = Literal["success", "partial", "failure"]


class CapabilityCard(BaseModel):
    """A single affordance the agent has access to."""

    id: str
    name: str
    category: str
    action_verb: str
    triggers: list[str] = Field(min_length=1)
    description: str
    params_required: list[str]
    params_optional: list[str]
    example: str
    schema_version: str
    source: Source
    namespace: str = "global"
    last_used: datetime | None = None
    success_count: int = 0
    failure_count: int = 0
    decay_score: float = 1.0

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        if not ID_PATTERN.match(v):
            raise ValueError(f"id must match {ID_PATTERN.pattern}, got {v!r}")
        return v

    @field_validator("category")
    @classmethod
    def _validate_category(cls, v: str) -> str:
        if v not in CATEGORIES:
            raise ValueError(f"category must be one of {sorted(CATEGORIES)}, got {v!r}")
        return v


class Workflow(BaseModel):
    """A successful tool sequence persisted as procedural memory."""

    situation: str
    sequence: list[str] = Field(min_length=1)
    outcome: Outcome
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Reflection(BaseModel):
    """A note about a failure, optionally with an LLM-generated reflection."""

    capability_id: str
    situation: str
    error: str
    reflection: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
