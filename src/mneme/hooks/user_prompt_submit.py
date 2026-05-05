"""UserPromptSubmit hook: inject relevant capabilities at the start of the prompt."""
from __future__ import annotations

import json
import sys
from typing import IO

from mneme import paths
from mneme.embedder import OllamaEmbedder
from mneme.retrieve import Retriever
from mneme.store import SqliteStore


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

    try:
        store = SqliteStore(db_path)
        embedder = OllamaEmbedder()
        retriever = Retriever(store, embedder)
        result = retriever.retrieve(prompt)
    except (ConnectionError, ValueError, RuntimeError, OSError):
        return 0  # fail-safe: any failure exits silently

    rendered = result.render()
    if rendered:
        stdout.write(rendered)
    return 0


def main() -> None:  # pragma: no cover
    sys.exit(run_hook())


if __name__ == "__main__":  # pragma: no cover
    main()
