"""PostToolUse hook: success -> procedural workflow; failure -> failures.log."""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import IO

from mneme import paths, telemetry
from mneme.schema import Workflow
from mneme.store import JsonlStore


def run_hook(stdin: IO[str] | None = None) -> int:
    """Read tool-execution result from stdin, persist outcome.

    Always exits 0. Success path appends a Workflow to procedural.jsonl.
    Failure path appends a structured entry to failures.log (Phase 1 simple
    log; Phase 2 will trigger an LLM-driven Reflexion subagent).
    """
    stdin = stdin or sys.stdin
    try:
        payload = json.load(stdin)
    except json.JSONDecodeError:
        return 0

    if not isinstance(payload, dict):
        return 0

    prompt = payload.get("prompt", "")
    tool = payload.get("tool_name", "")
    try:
        exit_code = int(payload.get("exit_code", 0))
    except (TypeError, ValueError):
        return 0

    if not isinstance(prompt, str) or not isinstance(tool, str) or not tool:
        return 0

    home = paths.home()
    home.mkdir(parents=True, exist_ok=True)

    if exit_code == 0:
        store = JsonlStore[Workflow](paths.procedural_jsonl(), Workflow)
        store.append(
            Workflow(
                situation=prompt[:500],
                sequence=[tool],
                outcome="success",
            )
        )
        telemetry.record_tool_outcome(tool, "success", exit_code)
    else:
        telemetry.record_tool_outcome(tool, "failure", exit_code)
        error = str(payload.get("error", ""))[:500]
        with paths.failure_log().open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "tool": tool,
                        "prompt": prompt[:500],
                        "exit_code": exit_code,
                        "error": error,
                    }
                )
                + "\n"
            )
    return 0


def main() -> None:  # pragma: no cover
    sys.exit(run_hook())


if __name__ == "__main__":  # pragma: no cover
    main()
