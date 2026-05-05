"""Aggregate local telemetry into actionable signals.

Surfaces patterns the user would otherwise have to observe manually:
  - capabilities with high retrieval but low success (card mismatch)
  - capabilities never retrieved in N days (dead cards — candidate for removal)
  - latency outliers
  - language distribution (does PT-BR retrieval work as well as EN?)
  - failure clusters per capability

Read-only. Operates on `~/.claude/mneme/telemetry.jsonl` produced by the hooks.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from mneme import paths


@dataclass
class Insights:
    window_days: int
    total_retrievals: int = 0
    total_outcomes: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    fallback_rate: float = 0.0
    no_match_rate: float = 0.0
    lang_distribution: dict[str, int] = field(default_factory=dict)
    top_capabilities: list[tuple[str, int]] = field(default_factory=list)
    dead_capabilities: list[str] = field(default_factory=list)
    failure_rate_per_tool: list[tuple[str, float, int]] = field(default_factory=list)
    suspicious_cards: list[tuple[str, int, int]] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"mneme insights (last {self.window_days} days)", "=" * 50, ""]
        lines.append(f"retrievals:     {self.total_retrievals}")
        lines.append(f"tool outcomes:  {self.total_outcomes}")
        lines.append(f"avg latency:    {self.avg_latency_ms:.0f} ms")
        lines.append(f"p95 latency:    {self.p95_latency_ms:.0f} ms")
        lines.append(f"fallback rate:  {self.fallback_rate:.1%}")
        lines.append(f"no-match rate:  {self.no_match_rate:.1%}")
        lines.append(f"languages:      {dict(self.lang_distribution)}")
        lines.append("")
        if self.top_capabilities:
            lines.append("top capabilities by retrieval count:")
            for name, count in self.top_capabilities[:10]:
                lines.append(f"  {count:4d}  {name}")
            lines.append("")
        if self.dead_capabilities:
            lines.append("never retrieved (consider removing from registry):")
            for name in self.dead_capabilities:
                lines.append(f"  - {name}")
            lines.append("")
        if self.failure_rate_per_tool:
            lines.append("failure rate per tool (>0):")
            for name, rate, n in self.failure_rate_per_tool:
                lines.append(f"  {rate:5.1%}  ({n:3d} runs)  {name}")
            lines.append("")
        if self.suspicious_cards:
            lines.append("suspicious cards — retrieved often, never used by agent:")
            lines.append("(card description likely does not match the real task)")
            for name, retrievals, outcomes in self.suspicious_cards:
                lines.append(f"  retrieved={retrievals:3d}  outcomes={outcomes:3d}  {name}")
            lines.append("")
        return "\n".join(lines)


def _read_events(path: Path, window_days: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    cutoff = datetime.now(UTC) - timedelta(days=window_days)
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                ts = datetime.fromisoformat(ev["ts"])
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            if ts >= cutoff:
                out.append(ev)
    return out


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * pct)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def _registry_capability_ids() -> set[str]:
    db = paths.semantic_db()
    if not db.exists():
        return set()
    import sqlite3

    try:
        conn = sqlite3.connect(db)
        rows = conn.execute("SELECT id FROM capabilities").fetchall()
        conn.close()
    except sqlite3.Error:
        return set()
    return {row[0] for row in rows}


def aggregate(window_days: int = 7) -> Insights:
    events = _read_events(paths.home() / "telemetry.jsonl", window_days)

    retrievals = [ev for ev in events if ev.get("event") == "retrieval"]
    outcomes = [ev for ev in events if ev.get("event") == "tool_outcome"]

    insights = Insights(window_days=window_days)
    insights.total_retrievals = len(retrievals)
    insights.total_outcomes = len(outcomes)

    latencies = [ev.get("latency_ms", 0.0) for ev in retrievals]
    if latencies:
        insights.avg_latency_ms = sum(latencies) / len(latencies)
        insights.p95_latency_ms = _percentile(latencies, 0.95)

    fallbacks = sum(1 for ev in retrievals if ev.get("fallback"))
    insights.fallback_rate = fallbacks / len(retrievals) if retrievals else 0.0

    no_match = sum(1 for ev in retrievals if not ev.get("top_capability"))
    insights.no_match_rate = no_match / len(retrievals) if retrievals else 0.0

    lang_counter = Counter(ev.get("lang_hint", "unknown") for ev in retrievals)
    insights.lang_distribution = dict(lang_counter)

    cap_counter = Counter(
        ev["top_capability"] for ev in retrievals if ev.get("top_capability")
    )
    insights.top_capabilities = cap_counter.most_common()

    registry = _registry_capability_ids()
    retrieved_ever = set(cap_counter)
    insights.dead_capabilities = sorted(registry - retrieved_ever)

    outcomes_by_tool: defaultdict[str, list[str]] = defaultdict(list)
    for ev in outcomes:
        tool = ev.get("tool", "")
        outcome = ev.get("outcome", "")
        if tool and outcome:
            outcomes_by_tool[tool].append(outcome)

    failure_rates: list[tuple[str, float, int]] = []
    for tool, outs in outcomes_by_tool.items():
        n = len(outs)
        rate = sum(1 for o in outs if o == "failure") / n
        if rate > 0:
            failure_rates.append((tool, rate, n))
    failure_rates.sort(key=lambda t: -t[1])
    insights.failure_rate_per_tool = failure_rates

    outcome_count_by_tool: dict[str, int] = {
        tool: len(outs) for tool, outs in outcomes_by_tool.items()
    }
    suspicious: list[tuple[str, int, int]] = []
    for cap, retrieved_count in cap_counter.items():
        outcome_count = outcome_count_by_tool.get(cap, 0)
        if retrieved_count >= 5 and outcome_count == 0:
            suspicious.append((cap, retrieved_count, outcome_count))
    suspicious.sort(key=lambda t: -t[1])
    insights.suspicious_cards = suspicious

    return insights
