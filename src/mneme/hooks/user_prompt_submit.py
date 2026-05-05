"""UserPromptSubmit hook: inject relevant capabilities at the start of the prompt."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import IO

from mneme import paths, telemetry
from mneme.embedder import OllamaEmbedder
from mneme.retrieve import Retriever
from mneme.schema import CapabilityCard, Workflow
from mneme.store import JsonlStore, SqliteStore


def _regex_fallback(prompt: str, db_path: Path, stdout: IO[str]) -> int:
    """Opt-in degraded path when Ollama is unreachable.

    Triggered only by `MNEME_FALLBACK=1`. Reads capabilities directly from
    SQLite (no extension load needed), and matches their `triggers` field
    against the prompt as case-insensitive substrings. Better than nothing
    when Ollama is down.
    """
    if os.environ.get("MNEME_FALLBACK") != "1":
        return 0
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.execute("SELECT json FROM capabilities")
        rows = cur.fetchall()
        conn.close()
    except sqlite3.Error:
        return 0

    matches: list[CapabilityCard] = []
    prompt_l = prompt.lower()
    for (raw_json,) in rows:
        card = CapabilityCard.model_validate_json(raw_json)
        for trigger in card.triggers:
            if trigger.lower() in prompt_l:
                matches.append(card)
                break

    if not matches:
        return 0

    lines = ["<capabilities-available>"]
    for card in matches[:5]:
        lines.append(
            f"- [{card.id}] {card.name} ({card.category}, fallback)\n"
            f"  verb: {card.action_verb}"
        )
    lines.append("</capabilities-available>")
    stdout.write("\n".join(lines))
    return 0


def run_hook(
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
) -> int:
    """Read prompt from stdin (Claude Code JSON), write capability injection to stdout.

    Always returns 0. Never blocks Claude Code: any failure (missing DB,
    Ollama down, malformed input) results in empty stdout and exit 0.
    """
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout

    try:
        payload = json.load(stdin)
    except json.JSONDecodeError:
        return 0

    prompt = payload.get("prompt", "") if isinstance(payload, dict) else ""
    if not isinstance(prompt, str) or not prompt.strip():
        return 0

    db_path = paths.semantic_db()
    if not db_path.exists():
        return 0

    t0 = time.perf_counter()
    try:
        store = SqliteStore(db_path)
        embedder = OllamaEmbedder()
        wf_path = paths.procedural_jsonl()
        wf_store = (
            JsonlStore[Workflow](wf_path, Workflow) if wf_path.exists() else None
        )
        retriever = Retriever(store, embedder, workflow_store=wf_store)
        result = retriever.retrieve(prompt)
    except ConnectionError:
        rc = _regex_fallback(prompt, db_path, stdout)
        latency_ms = (time.perf_counter() - t0) * 1000
        telemetry.record_retrieval(
            prompt=prompt,
            top_capability=None,
            top_score=None,
            n_returned=0,
            latency_ms=latency_ms,
            fallback=True,
        )
        return rc
    except (ValueError, RuntimeError, OSError):
        return 0

    latency_ms = (time.perf_counter() - t0) * 1000
    top_cap = result.capabilities[0][0].id if result.capabilities else None
    top_score = result.capabilities[0][1] if result.capabilities else None
    telemetry.record_retrieval(
        prompt=prompt,
        top_capability=top_cap,
        top_score=top_score,
        n_returned=len(result.capabilities),
        latency_ms=latency_ms,
        fallback=False,
    )

    rendered = result.render()
    if rendered:
        stdout.write(rendered)
    return 0


def main() -> None:  # pragma: no cover
    sys.exit(run_hook())


if __name__ == "__main__":  # pragma: no cover
    main()
