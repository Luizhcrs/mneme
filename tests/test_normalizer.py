"""Tests for the verbose-to-EasyTool normalizer."""
from __future__ import annotations

from mneme.normalizer import normalize_description


def test_truncates_long_description_to_canonical_form() -> None:
    verbose = (
        "This is a very long description with many sentences. "
        "It rambles about installation. It talks about edge cases. "
        "It includes credits and changelogs. The Telegram tool sends a message to a chat. "
        "There are also examples about timezones and rate limits. "
        "The library was first released years ago and has many contributors."
    )
    out = normalize_description(verbose, name="Telegram Send")
    assert len(out) <= 200
    assert "telegram" in out.lower() or "sends a message" in out.lower()


def test_strips_markdown_artifacts() -> None:
    raw = "**Telegram** sends a message via [API](https://example.com)."
    out = normalize_description(raw, name="Telegram")
    assert "**" not in out
    assert "[" not in out and "]" not in out


def test_preserves_short_input() -> None:
    short = "Sends a message to a Telegram chat."
    out = normalize_description(short, name="Telegram")
    assert out == short


def test_handles_empty_input() -> None:
    out = normalize_description("", name="X")
    assert out == ""
