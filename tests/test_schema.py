"""Tests for capability/workflow/reflection schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mneme.schema import CapabilityCard, Reflection, Workflow


def test_capability_card_minimal() -> None:
    card = CapabilityCard(
        id="playwright_screenshot",
        name="Playwright Screenshot",
        category="web_browser",
        action_verb="captures image of web page",
        triggers=["screenshot", "scrape visual"],
        description="MCP Playwright headless browser screenshots.",
        params_required=["url"],
        params_optional=["selector"],
        example='Call: playwright.screenshot(url="...")',
        schema_version="1.45",
        source="mcp",
    )
    assert card.id == "playwright_screenshot"
    assert card.success_count == 0
    assert card.decay_score == 1.0
    assert card.namespace == "global"


def test_capability_card_rejects_invalid_id() -> None:
    with pytest.raises(ValidationError):
        CapabilityCard(
            id="Has Spaces And Caps",
            name="x",
            category="filesystem",
            action_verb="x",
            triggers=["x"],
            description="x",
            params_required=[],
            params_optional=[],
            example="x",
            schema_version="1",
            source="mcp",
        )


def test_capability_card_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError):
        CapabilityCard(
            id="x",
            name="x",
            category="not_a_real_category",
            action_verb="x",
            triggers=["x"],
            description="x",
            params_required=[],
            params_optional=[],
            example="x",
            schema_version="1",
            source="mcp",
        )


def test_workflow_minimal() -> None:
    wf = Workflow(
        situation="screenshot a SPA homepage",
        sequence=["playwright_goto", "playwright_wait_for", "playwright_screenshot"],
        outcome="success",
    )
    assert len(wf.sequence) == 3
    assert wf.outcome == "success"


def test_reflection_minimal() -> None:
    r = Reflection(
        capability_id="playwright_screenshot",
        situation="screenshot SPA before DOM ready",
        error="timeout waiting for selector",
    )
    assert r.capability_id == "playwright_screenshot"
