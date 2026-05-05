"""A/B test: same agent, same task, mneme injection ON vs OFF.

Methodology
-----------
For each scenario, build a prompt that asks an LLM to plan how it would
handle the user's request given the constraints of a Claude-Code-style
agent. The LLM stands in for the agent and produces a textual plan.

  Variant A (control): only the user's request.
  Variant B (mneme):   mneme's <capabilities-available> block prepended,
                        then the user's request.

Both variants reach the same LLM (Ollama qwen2.5:3b for offline runs)
with the same parameters. The response is scored deterministically:

  +1  for mentioning the expected tool id or one of its triggers
  -1  for proposing a known wrong fallback (curl, regex, manual parsing)
  -1  for explicitly stating an inability the listed capability contradicts
       (e.g. "I cannot take screenshots", "I do not have browser access")

The aggregate score and per-scenario verdicts are printed. A real impact
shows up as variant B materially outscoring variant A on adversarial
scenarios where the right tool is non-obvious without explicit prompting.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from mneme.embedder import OllamaEmbedder
from mneme.loader import seed_store
from mneme.retrieve import Retriever
from mneme.store import SqliteStore

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"


@dataclass
class Scenario:
    prompt: str
    expected_tool_id: str
    expected_keywords: list[str]
    wrong_fallbacks: list[str]
    cant_phrases: list[str]


SCENARIOS: list[Scenario] = [
    Scenario(
        prompt=(
            "I need you to take a screenshot of the homepage of "
            "https://example.com so I can attach it to a report. "
            "Walk me through what you would do."
        ),
        expected_tool_id="playwright_screenshot",
        expected_keywords=["playwright", "headless", "screenshot"],
        wrong_fallbacks=["curl", "wget", "regex", "manual"],
        cant_phrases=[
            "i cannot take screenshots",
            "no browser",
            "i don't have access",
            "not able to render",
        ],
    ),
    Scenario(
        prompt=(
            "Tira print da home da example.com pra eu colar num "
            "documento. Como tu vai fazer isso?"
        ),
        expected_tool_id="playwright_screenshot",
        expected_keywords=["playwright", "navegador", "headless"],
        wrong_fallbacks=["curl", "wget", "regex"],
        cant_phrases=[
            "nao tenho como",
            "nao consigo tirar print",
            "sem navegador",
        ],
    ),
    Scenario(
        prompt=(
            "Send a Telegram alert to chat 12345678 saying the deploy "
            "finished. Walk me through how you would send it."
        ),
        expected_tool_id="telegram_send",
        expected_keywords=["telegram.send", "telegram", "chat_id"],
        wrong_fallbacks=["copy and paste", "manually open telegram"],
        cant_phrases=[
            "i cannot send",
            "i do not have telegram",
            "not able to message",
        ],
    ),
    Scenario(
        prompt=(
            "Open a GitHub issue on repo myorg/myapp titled 'login broken' "
            "with a body explaining the auth flow is failing on Safari. "
            "Walk me through your approach."
        ),
        expected_tool_id="github_create_issue",
        expected_keywords=["gh.issue.create", "gh issue", "github_create_issue"],
        wrong_fallbacks=["copy paste in browser", "manually open github"],
        cant_phrases=[
            "i cannot create issues",
            "i don't have github access",
        ],
    ),
    Scenario(
        prompt=(
            "Run the SQL query SELECT id, email FROM users LIMIT 10 "
            "against my postgres database. Walk me through how you would "
            "execute it."
        ),
        expected_tool_id="postgres_query",
        expected_keywords=["pg.query", "postgres", "psycopg"],
        wrong_fallbacks=["pgadmin gui", "manually run"],
        cant_phrases=[
            "i cannot run sql",
            "i don't have database access",
        ],
    ),
    Scenario(
        prompt=(
            "Sobe os containers do projeto que ta em /path/to/project com "
            "docker compose. Mostra como tu faria."
        ),
        expected_tool_id="docker_compose_up",
        expected_keywords=["compose.up", "docker compose up"],
        wrong_fallbacks=["sudo systemctl"],
        cant_phrases=[
            "nao consigo subir",
            "sem docker",
        ],
    ),
    Scenario(
        prompt=(
            "Read the file /var/log/app.log and tell me the last 20 lines. "
            "How would you do it?"
        ),
        expected_tool_id="filesystem_read",
        expected_keywords=["filesystem.read", "open file", "tail"],
        wrong_fallbacks=["copy paste from terminal"],
        cant_phrases=[
            "i cannot read files",
            "no file access",
        ],
    ),
    Scenario(
        prompt=(
            "Use a local AI model to summarize this paragraph: "
            "'The discovery of CRISPR enabled targeted gene editing.' "
            "How do you proceed?"
        ),
        expected_tool_id="ollama_generate",
        expected_keywords=["ollama.generate", "ollama", "llama"],
        wrong_fallbacks=["call openai", "use cloud api"],
        cant_phrases=[
            "i cannot run models",
            "i do not have llm",
        ],
    ),
    Scenario(
        prompt=(
            "Search my Obsidian vault for notes about the memory system "
            "I have been working on. Walk me through the steps."
        ),
        expected_tool_id="obsidian_search_vault",
        expected_keywords=["vault.search", "obsidian"],
        wrong_fallbacks=["grep manually", "copy paste folder"],
        cant_phrases=[
            "i cannot search obsidian",
            "no vault access",
        ],
    ),
    Scenario(
        prompt=(
            "Tira um print da tela do meu computador agora pra eu colar "
            "num documento. Como tu faz?"
        ),
        expected_tool_id="pyautogui_screenshot",
        expected_keywords=["pyautogui.screenshot", "pyautogui"],
        wrong_fallbacks=["use windows snipping tool manually"],
        cant_phrases=[
            "nao consigo capturar",
            "sem acesso ao desktop",
        ],
    ),
]


def call_llm_ollama(prompt: str, model: str, *, timeout: float = 90.0) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 220},
    }
    r = httpx.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return str(r.json().get("response", ""))


def _resolve_claude_cli() -> str | None:
    """Find the Claude Code binary, honouring Windows PATHEXT (.cmd)."""
    return (
        shutil.which("claude")
        or shutil.which("claude.cmd")
        or shutil.which("claude.exe")
    )


_CLAUDE_CLEAN_SYSTEM_PROMPT = (
    "You are a coding assistant. Answer the user's request directly with a "
    "concrete approach (3-5 sentences) including specific tool calls or "
    "commands. Do not ask clarifying questions. Do not refer to memory, "
    "auto-pilot, or any project-specific context. Treat the prompt as a "
    "self-contained task."
)


def call_llm_claude(prompt: str, *, timeout: float = 180.0) -> str:
    """Run Claude Code CLI cleanly via stdin + system-prompt override.

    The combination ``--system-prompt <override> --tools ""`` cuts the user's
    CLAUDE.md auto-memory and tool surface out of the call, leaving the
    underlying Anthropic model to respond to the prompt directly. This is
    important for the A/B: without these flags, Claude Code reads the user's
    global CLAUDE.md, which on this developer's machine triggers an
    'auto-pilot session init' interpretation that ignores the actual task.

    Authentication still flows through the user's OAuth login (no
    ANTHROPIC_API_KEY required) — only the bare flag forces an API-key
    handshake, and we are not using bare.

    The prompt is delivered via stdin to avoid Windows command-line length
    limits on long compound prompts.
    """
    import os
    import tempfile

    binary = _resolve_claude_cli()
    if binary is None:
        raise RuntimeError(
            "claude CLI not found in PATH; install Claude Code or use --judge ollama"
        )
    with tempfile.TemporaryDirectory() as empty_home:
        env = {**os.environ, "MNEME_HOME": empty_home, "MNEME_TELEMETRY": "0"}
        proc = subprocess.run(
            [
                binary,
                "--print",
                "--system-prompt",
                _CLAUDE_CLEAN_SYSTEM_PROMPT,
                "--tools",
                "",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            env=env,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI exit {proc.returncode}: {proc.stderr.strip()[:200]}"
        )
    # Strip session-end hook noise that may follow the model output.
    out = proc.stdout
    cutoff_markers = ("SessionEnd hook", "[claude-mem]", "Stop hook")
    for marker in cutoff_markers:
        idx = out.find(marker)
        if idx > 0:
            out = out[:idx]
    return out.rstrip()


def score(
    response: str,
    scenario: Scenario,
) -> dict[str, int]:
    text = response.lower()
    result = {
        "expected_hit": 0,
        "wrong_fallback": 0,
        "cant_phrase": 0,
        "total": 0,
    }
    if scenario.expected_tool_id.lower() in text or any(
        kw.lower() in text for kw in scenario.expected_keywords
    ):
        result["expected_hit"] = 1
    if any(re.search(rf"\b{re.escape(fb.lower())}\b", text) for fb in scenario.wrong_fallbacks):
        result["wrong_fallback"] = 1
    if any(phrase in text for phrase in scenario.cant_phrases):
        result["cant_phrase"] = 1
    result["total"] = (
        result["expected_hit"]
        - result["wrong_fallback"]
        - result["cant_phrase"]
    )
    return result


def build_with_mneme_prompt(injection: str, user_request: str) -> str:
    return (
        f"You are a coding agent that has access to local tools. The system\n"
        f"has provided you with the following capabilities and reflections:\n\n"
        f"{injection}\n\n"
        f"User request: {user_request}\n\n"
        f"Briefly describe (3-5 sentences) the concrete approach you would take, "
        f"including any specific tool calls."
    )


def build_control_prompt(user_request: str) -> str:
    return (
        f"You are a coding agent. Briefly describe (3-5 sentences) the concrete "
        f"approach you would take to handle this user request, including any "
        f"specific tool calls.\n\n"
        f"User request: {user_request}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=len(SCENARIOS))
    parser.add_argument(
        "--judge",
        choices=("ollama", "claude"),
        default="ollama",
        help="Which LLM evaluates each prompt. 'claude' uses the Claude Code CLI "
        "in --bare --print mode (no hooks, no plugins) and is much stronger in "
        "PT-BR than the small local Ollama model.",
    )
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model name when --judge=ollama (default: qwen2.5:3b).",
    )
    args = parser.parse_args()

    def call(prompt: str) -> str:
        if args.judge == "claude":
            return call_llm_claude(prompt)
        return call_llm_ollama(prompt, args.ollama_model)

    here = Path(__file__).parent
    seed = here.parent.parent / "src" / "mneme" / "seed" / "capabilities.example.yaml"
    db = here / "_ab.sqlite"
    if db.exists():
        db.unlink()
    embedder = OllamaEmbedder()
    store = SqliteStore(db)
    seed_store(seed, store, embedder=embedder)
    retriever = Retriever(store, embedder, top_categories=10, top_capabilities=5, threshold=0.0)

    totals_a = {"expected_hit": 0, "wrong_fallback": 0, "cant_phrase": 0, "total": 0}
    totals_b = {"expected_hit": 0, "wrong_fallback": 0, "cant_phrase": 0, "total": 0}

    rows: list[tuple[Scenario, dict[str, int], dict[str, int]]] = []

    for sc in SCENARIOS[: args.limit]:
        result = retriever.retrieve(sc.prompt)
        injection = result.render()
        prompt_a = build_control_prompt(sc.prompt)
        prompt_b = build_with_mneme_prompt(injection or "(no relevant capabilities)", sc.prompt)

        t0 = time.perf_counter()
        resp_a = call(prompt_a)
        resp_b = call(prompt_b)
        elapsed_s = time.perf_counter() - t0
        del elapsed_s  # measured for future logging; not reported per-row

        sa = score(resp_a, sc)
        sb = score(resp_b, sc)
        for k in totals_a:
            totals_a[k] += sa[k]
            totals_b[k] += sb[k]
        rows.append((sc, sa, sb))

    print("=" * 80)
    print(f"A/B TEST RESULTS — control vs mneme (judge: {args.judge})")
    print("=" * 80)
    print()
    for sc, sa, sb in rows:
        prompt_short = sc.prompt[:60].replace("\n", " ")
        delta = sb["total"] - sa["total"]
        sign = "+" if delta > 0 else ("=" if delta == 0 else "-")
        print(f"[{sign}{abs(delta)}] {prompt_short}...")
        print(
            f"    control: hit={sa['expected_hit']} fallback={sa['wrong_fallback']} "
            f"cant={sa['cant_phrase']} total={sa['total']}"
        )
        print(
            f"    mneme:   hit={sb['expected_hit']} fallback={sb['wrong_fallback']} "
            f"cant={sb['cant_phrase']} total={sb['total']}"
        )
    print()
    print("-" * 80)
    n = len(rows)
    print(
        f"control total ({n} scenarios): "
        f"expected_hit={totals_a['expected_hit']}/{n}  "
        f"wrong_fallback={totals_a['wrong_fallback']}  "
        f"cant_phrase={totals_a['cant_phrase']}  "
        f"score={totals_a['total']}"
    )
    print(
        f"mneme   total ({n} scenarios): "
        f"expected_hit={totals_b['expected_hit']}/{n}  "
        f"wrong_fallback={totals_b['wrong_fallback']}  "
        f"cant_phrase={totals_b['cant_phrase']}  "
        f"score={totals_b['total']}"
    )
    print(
        f"\ndelta: hit={totals_b['expected_hit']-totals_a['expected_hit']:+d}  "
        f"score={totals_b['total']-totals_a['total']:+d}"
    )
    store.close()


if __name__ == "__main__":
    main()
