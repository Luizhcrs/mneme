"""Pydantic schemas for capability cards, workflows, and reflections."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, Field

Category = Literal[
    "agent_orchestration",
    "audio_media",
    "automation_rpa",
    "cloud_infra",
    "code_search",
    "comms",
    "container",
    "crypto",
    "data_pipeline",
    "db",
    "deploy",
    "desktop_automation",
    "documentation",
    "embedded_iot",
    "filesystem",
    "finance_trading",
    "game_engine",
    "geo_maps",
    "graphics_3d",
    "hardware_io",
    "kubernetes",
    "math_scientific",
    "ml_inference",
    "ml_training",
    "mobile_dev",
    "monitoring",
    "nlp",
    "security",
    "testing",
    "vault_kb",
    "vcs",
    "vision_image",
    "voice_audio",
    "web_api",
    "web_browser",
]

CATEGORIES: frozenset[str] = frozenset(get_args(Category))

ID_PATTERN_STR = r"^[a-z][a-z0-9_]*$"
CardId = Annotated[str, Field(pattern=ID_PATTERN_STR)]

Source = Literal["mcp", "skill", "plugin", "command", "project", "service", "os", "manual"]
Outcome = Literal["success", "partial", "failure"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class CapabilityCard(BaseModel):
    """A single affordance the agent has access to."""

    id: CardId
    name: str
    category: Category
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
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    decay_score: float = Field(default=1.0, ge=0.0, le=1.0)
    active: bool = True


class Workflow(BaseModel):
    """A successful tool sequence persisted as procedural memory."""

    situation: str
    sequence: list[str] = Field(min_length=1)
    outcome: Outcome
    timestamp: datetime = Field(default_factory=_utcnow)


class Reflection(BaseModel):
    """A note about a failure, optionally with an LLM-generated reflection."""

    capability_id: CardId
    situation: str
    error: str
    reflection: str | None = None
    timestamp: datetime = Field(default_factory=_utcnow)
