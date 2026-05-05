"""Local-only structural telemetry.

Records *metadata* about retrievals and tool outcomes — never prompt content,
URLs, IDs, or any field that could leak personal usage. Lives at
``~/.claude/mneme/telemetry.jsonl``. Never transmitted off the local machine.

Each line is a JSON event with one of two shapes:

    {"ts": "...", "event": "retrieval", "prompt_len": int, "lang_hint": "en|pt-br|mixed",
     "top_capability": str, "top_score": float, "n_returned": int,
     "latency_ms": float, "fallback": bool}

    {"ts": "...", "event": "tool_outcome", "tool": str, "outcome": "success|failure",
     "exit_code": int}

Disable entirely with ``MNEME_TELEMETRY=0``. Default is enabled.
"""
from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from typing import Any

from mneme import paths

_PT_BR_HINTS = re.compile(
    r"\b(de|do|da|e|para|que|tu|com|ja|nao|nao|seu|sua|um|uma|os|as|"
    r"executa|consulta|busca|le|abre|sobe|tira|manda|notifica|alerta|"
    r"avisa|cria|roda|faz|usuario)\b",
    re.IGNORECASE,
)
_EN_HINTS = re.compile(
    r"\b(the|and|for|that|with|from|this|your|are|have|will|"
    r"send|read|create|open|run|use|search|find|navigate|fetch|"
    r"query|notify|alert|capture|render)\b",
    re.IGNORECASE,
)


def _disabled() -> bool:
    return os.environ.get("MNEME_TELEMETRY", "1") == "0"


def _classify_language(text: str) -> str:
    pt = len(_PT_BR_HINTS.findall(text))
    en = len(_EN_HINTS.findall(text))
    if pt > en + 1:
        return "pt-br"
    if en > pt + 1:
        return "en"
    return "mixed"


def _append(event: dict[str, Any]) -> None:
    if _disabled():
        return
    try:
        path = paths.home() / "telemetry.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")
    except OSError:
        pass  # telemetry must never break the hook


def record_retrieval(
    prompt: str,
    top_capability: str | None,
    top_score: float | None,
    n_returned: int,
    latency_ms: float,
    fallback: bool = False,
) -> None:
    _append(
        {
            "ts": datetime.now(UTC).isoformat(),
            "event": "retrieval",
            "prompt_len": len(prompt),
            "lang_hint": _classify_language(prompt),
            "top_capability": top_capability,
            "top_score": round(top_score, 3) if top_score is not None else None,
            "n_returned": n_returned,
            "latency_ms": round(latency_ms, 2),
            "fallback": fallback,
        }
    )


def record_tool_outcome(tool: str, outcome: str, exit_code: int) -> None:
    if outcome not in {"success", "failure"}:
        return
    _append(
        {
            "ts": datetime.now(UTC).isoformat(),
            "event": "tool_outcome",
            "tool": tool,
            "outcome": outcome,
            "exit_code": exit_code,
        }
    )
