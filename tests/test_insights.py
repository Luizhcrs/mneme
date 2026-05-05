"""Tests for the insights aggregator and Insights.render output."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mneme.insights import Insights, aggregate


def _write_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def _ts(days_ago: float = 0.0) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()


def test_aggregate_empty_when_no_telemetry(
    tmp_mneme_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MNEME_HOME", str(tmp_mneme_dir))
    insights = aggregate(window_days=7)
    assert insights.total_retrievals == 0
    assert insights.total_outcomes == 0
    assert insights.avg_latency_ms == 0.0
    assert insights.lang_distribution == {}


def test_aggregate_counts_retrievals_and_outcomes(
    tmp_mneme_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MNEME_HOME", str(tmp_mneme_dir))
    _write_events(
        tmp_mneme_dir / "telemetry.jsonl",
        [
            {
                "ts": _ts(0.1),
                "event": "retrieval",
                "prompt_len": 30,
                "lang_hint": "pt-br",
                "top_capability": "telegram_send",
                "top_score": 0.78,
                "n_returned": 2,
                "latency_ms": 150.0,
                "fallback": False,
            },
            {
                "ts": _ts(0.2),
                "event": "retrieval",
                "prompt_len": 50,
                "lang_hint": "en",
                "top_capability": "telegram_send",
                "top_score": 0.81,
                "n_returned": 1,
                "latency_ms": 120.0,
                "fallback": False,
            },
            {
                "ts": _ts(0.3),
                "event": "tool_outcome",
                "tool": "telegram_send",
                "outcome": "success",
                "exit_code": 0,
            },
            {
                "ts": _ts(0.4),
                "event": "tool_outcome",
                "tool": "playwright_screenshot",
                "outcome": "failure",
                "exit_code": 1,
            },
        ],
    )

    insights = aggregate(window_days=7)
    assert insights.total_retrievals == 2
    assert insights.total_outcomes == 2
    assert insights.avg_latency_ms == pytest.approx(135.0)
    assert insights.lang_distribution == {"pt-br": 1, "en": 1}
    assert insights.top_capabilities[0] == ("telegram_send", 2)


def test_aggregate_excludes_events_outside_window(
    tmp_mneme_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MNEME_HOME", str(tmp_mneme_dir))
    _write_events(
        tmp_mneme_dir / "telemetry.jsonl",
        [
            {
                "ts": _ts(0.1),
                "event": "retrieval",
                "prompt_len": 10,
                "lang_hint": "en",
                "top_capability": "x",
                "top_score": 0.5,
                "n_returned": 1,
                "latency_ms": 10.0,
                "fallback": False,
            },
            {
                "ts": _ts(30.0),  # out of 7-day window
                "event": "retrieval",
                "prompt_len": 10,
                "lang_hint": "en",
                "top_capability": "y",
                "top_score": 0.5,
                "n_returned": 1,
                "latency_ms": 10.0,
                "fallback": False,
            },
        ],
    )
    insights = aggregate(window_days=7)
    assert insights.total_retrievals == 1


def test_aggregate_flags_no_match_rate(
    tmp_mneme_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MNEME_HOME", str(tmp_mneme_dir))
    _write_events(
        tmp_mneme_dir / "telemetry.jsonl",
        [
            {"ts": _ts(0.1), "event": "retrieval", "prompt_len": 5, "lang_hint": "en",
             "top_capability": "x", "top_score": 0.5, "n_returned": 1,
             "latency_ms": 10.0, "fallback": False},
            {"ts": _ts(0.2), "event": "retrieval", "prompt_len": 5, "lang_hint": "en",
             "top_capability": None, "top_score": None, "n_returned": 0,
             "latency_ms": 8.0, "fallback": False},
            {"ts": _ts(0.3), "event": "retrieval", "prompt_len": 5, "lang_hint": "en",
             "top_capability": None, "top_score": None, "n_returned": 0,
             "latency_ms": 8.0, "fallback": False},
        ],
    )
    insights = aggregate(window_days=7)
    assert insights.no_match_rate == pytest.approx(2 / 3)


def test_aggregate_computes_failure_rate(
    tmp_mneme_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MNEME_HOME", str(tmp_mneme_dir))
    outcomes = [
        {"ts": _ts(0.1), "event": "tool_outcome", "tool": "buggy", "outcome": "failure",
         "exit_code": 1},
        {"ts": _ts(0.2), "event": "tool_outcome", "tool": "buggy", "outcome": "failure",
         "exit_code": 1},
        {"ts": _ts(0.3), "event": "tool_outcome", "tool": "buggy", "outcome": "success",
         "exit_code": 0},
        {"ts": _ts(0.4), "event": "tool_outcome", "tool": "ok_tool", "outcome": "success",
         "exit_code": 0},
    ]
    _write_events(tmp_mneme_dir / "telemetry.jsonl", outcomes)
    insights = aggregate(window_days=7)
    by_tool = {name: rate for name, rate, _ in insights.failure_rate_per_tool}
    assert by_tool["buggy"] == pytest.approx(2 / 3)
    assert "ok_tool" not in by_tool  # 0% failures filtered out


def test_render_handles_empty_insights() -> None:
    out = Insights(window_days=7).render()
    assert "mneme insights" in out
    assert "retrievals:     0" in out


def test_render_lists_top_capabilities() -> None:
    insights = Insights(window_days=7)
    insights.total_retrievals = 5
    insights.top_capabilities = [("telegram_send", 3), ("playwright_screenshot", 2)]
    out = insights.render()
    assert "telegram_send" in out
    assert "playwright_screenshot" in out


def test_render_lists_dead_capabilities() -> None:
    insights = Insights(window_days=7)
    insights.dead_capabilities = ["unused_card_a", "unused_card_b"]
    out = insights.render()
    assert "unused_card_a" in out
    assert "never retrieved" in out


def test_render_lists_suspicious_cards() -> None:
    insights = Insights(window_days=7)
    insights.suspicious_cards = [("retrieved_but_unused", 12, 0)]
    out = insights.render()
    assert "retrieved_but_unused" in out
    assert "suspicious cards" in out
