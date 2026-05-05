"""Hard A/B test — adversarial scenarios that exercise mneme's actual value.

The first A/B harness (ab_test.py) uses well-known tools (Playwright, Telegram,
GitHub, Docker, Postgres, Ollama, Obsidian, PyAutoGUI). Claude Opus knows all
of these by name from its training data, so the control already names them
correctly and the mneme injection adds no measurable lift — both variants
score 3/3 in our 3-scenario pilot.

This harness is designed to surface mneme's real lift: scenarios where the
right answer is a card the model would NOT name without injection. Three
categories of hardness:

  CATEGORY 1 — User-installed Claude Code plugins not in LLM training
    plugin:chrome-devtools-mcp, plugin:claude-mem, plugin:caveman,
    plugin:voltagent-qa-sec, etc. These are installed on the user's machine
    but the LLM has no prior knowledge of them. mneme's scanner discovers
    them and surfaces them in the registry.

  CATEGORY 2 — Exact MCP tool path required for an actual call
    The model may say 'use Playwright' (vague) but mneme's card carries the
    exact MCP path 'plugin:playwright:playwright' that Claude Code calls.
    Scoring rewards naming the exact id, not a generic noun.

  CATEGORY 3 — Multi-tool composition with disambiguation
    'Audit page accessibility' could mean Lighthouse (chrome-devtools-mcp)
    or a generic checklist. Without mneme, the model defaults to a generic
    written checklist; with mneme injection, it should reach for the actual
    audit tool.

Run:
    python tests/benchmark/ab_test_hard.py --judge claude
    python tests/benchmark/ab_test_hard.py --judge ollama
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

import httpx

from mneme import paths
from mneme.embedder import OllamaEmbedder
from mneme.feedback import FeedbackStore
from mneme.retrieve import Retriever
from mneme.schema import Workflow
from mneme.store import JsonlStore, SqliteStore

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"


@dataclass
class Scenario:
    name: str
    category: str
    prompt: str
    expected_keywords: list[str]
    wrong_fallbacks: list[str]


# Adversarial scenarios. Each picks a card that the model is unlikely to
# name without injection (custom plugin) and rewards exact mention.
SCENARIOS: list[Scenario] = [
    # CATEGORY 1 — user-installed plugins
    Scenario(
        name="cdt-a11y",
        category="custom-plugin",
        prompt=(
            "I need to audit the accessibility of the page at "
            "https://example.com per WCAG. What tool on my machine should "
            "I run? Name it explicitly."
        ),
        expected_keywords=[
            "chrome-devtools",
            "lighthouse",
            "plugin_chrome_devtools",
            "cdt",
        ],
        wrong_fallbacks=["axe-core manually", "wave online", "manually inspect"],
    ),
    Scenario(
        name="claude-mem-search",
        category="custom-plugin",
        prompt=(
            "I want to search across my past Claude Code sessions for "
            "anything related to retrieval tuning. What installed tool "
            "should I reach for? Name it."
        ),
        expected_keywords=["claude-mem", "claude_mem", "mcp-search", "memory"],
        wrong_fallbacks=["grep manually", "scroll terminal history"],
    ),
    Scenario(
        name="voltagent-qa-sec",
        category="custom-plugin",
        prompt=(
            "I want to run a security review subagent on the diff in my "
            "current branch. What tool installed on my machine handles "
            "that? Name it."
        ),
        expected_keywords=["voltagent", "qa-sec", "qa_sec", "voltagent_qa"],
        wrong_fallbacks=["manual code review", "sonarqube only"],
    ),
    Scenario(
        name="caveman",
        category="custom-plugin",
        prompt=(
            "I want to compress my markdown memory file to save tokens. "
            "What installed plugin does that? Name it."
        ),
        expected_keywords=["caveman", "compress"],
        wrong_fallbacks=["manually rewrite", "use sed"],
    ),
    Scenario(
        name="cicd-automation",
        category="custom-plugin",
        prompt=(
            "I need to set up a release pipeline with semantic versioning. "
            "What installed plugin focuses on that? Name it."
        ),
        expected_keywords=["cicd", "cicd_automation", "ci/cd", "pipeline"],
        wrong_fallbacks=["github actions only", "write yaml by hand"],
    ),
    # CATEGORY 2 — exact MCP path required
    Scenario(
        name="playwright-mcp-path",
        category="exact-id",
        prompt=(
            "I want Claude Code to navigate a headless browser. The MCP "
            "is installed. What's the EXACT MCP server name to call?"
        ),
        expected_keywords=[
            "plugin:playwright:playwright",
            "plugin_playwright_playwright",
            "plugin_playwright",
        ],
        wrong_fallbacks=["just use playwright cli"],
    ),
    Scenario(
        name="telegram-mcp-path",
        category="exact-id",
        prompt=(
            "What's the exact MCP id for the Telegram tool installed on "
            "this machine?"
        ),
        expected_keywords=[
            "plugin:telegram:telegram",
            "plugin_telegram_telegram",
            "plugin_telegram",
        ],
        wrong_fallbacks=["telegram bot api directly"],
    ),
    # CATEGORY 3 — multi-tool with disambiguation
    Scenario(
        name="trace-perf",
        category="disambiguation",
        prompt=(
            "I want to profile the performance of a webpage and analyze "
            "the LCP. What installed tool on my machine should run that?"
        ),
        expected_keywords=[
            "chrome-devtools",
            "performance_start_trace",
            "lighthouse",
            "plugin_chrome_devtools",
        ],
        wrong_fallbacks=["webpagetest online", "manually use devtools"],
    ),
    Scenario(
        name="rust-lsp",
        category="custom-plugin",
        prompt=(
            "I'm refactoring a Rust project and want IDE-grade type info "
            "from Claude. What installed plugin gives that? Name it."
        ),
        expected_keywords=["rust-analyzer", "rust_analyzer", "rust-analyzer-lsp"],
        wrong_fallbacks=["rustc by hand", "cargo check only"],
    ),
    Scenario(
        name="db-design-plugin",
        category="custom-plugin",
        prompt=(
            "I need a subagent that specializes in database schema design "
            "review. What installed plugin do I have for that? Name it."
        ),
        expected_keywords=["database-design", "database_design"],
        wrong_fallbacks=["draw a diagram manually"],
    ),
]


def call_llm_ollama(prompt: str, model: str, *, timeout: float = 120.0) -> str:
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
    return (
        shutil.which("claude")
        or shutil.which("claude.cmd")
        or shutil.which("claude.exe")
    )


def call_llm_claude(prompt: str, *, timeout: float = 180.0) -> str:
    """Run Claude Code CLI in clean print mode.

    --system-prompt overrides the default (skips CLAUDE.md auto-discovery).
    --tools "" disables tool access (we only want the model's first response).
    Empty MNEME_HOME defangs the user's UserPromptSubmit hook so it cannot
    re-inject mneme into the control variant via path-based fail-safe.
    """
    binary = _resolve_claude_cli()
    if binary is None:
        raise RuntimeError(
            "claude CLI not found in PATH; install Claude Code or use --judge ollama"
        )
    sysprompt = (
        "You are a coding assistant. Answer directly. "
        "Name specific tools by their exact installed name when they exist on "
        "this machine. Be concise (3-6 sentences)."
    )
    with tempfile.TemporaryDirectory() as empty_home:
        env = {
            **os.environ,
            "MNEME_HOME": empty_home,
            "MNEME_TELEMETRY": "0",
        }
        proc = subprocess.run(
            [
                binary,
                "--print",
                "--system-prompt",
                sysprompt,
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
    return proc.stdout


def score(response: str, scenario: Scenario) -> dict[str, int]:
    text = response.lower()
    hit = any(kw.lower() in text for kw in scenario.expected_keywords)
    fallback = any(fb.lower() in text for fb in scenario.wrong_fallbacks)
    return {
        "expected_hit": int(hit),
        "wrong_fallback": int(fallback),
        "total": int(hit) - int(fallback),
    }


def build_with_mneme_prompt(prompt: str) -> str:
    """Augment with the live mneme injection block from the user's REAL registry.

    Uses ~/.claude/mneme/semantic.sqlite (the index built from the user's
    capabilities.yaml after `mneme rescan && mneme reindex`) so the test
    measures mneme as it actually runs in production, not the 10-card seed.
    """
    db = paths.semantic_db()
    if not db.exists():
        return prompt
    embedder = OllamaEmbedder()
    store = SqliteStore(db)
    wf_path = paths.procedural_jsonl()
    wf_store = (
        JsonlStore[Workflow](wf_path, Workflow) if wf_path.exists() else None
    )
    fb_path = paths.feedback_jsonl()
    fb_store = FeedbackStore(fb_path) if fb_path.exists() else None
    retriever = Retriever(
        store,
        embedder,
        workflow_store=wf_store,
        feedback_store=fb_store,
    )
    block = retriever.retrieve(prompt).render()
    store.close()
    if not block:
        return prompt
    return f"{block}\n\n{prompt}"


def build_control_prompt(prompt: str) -> str:
    return prompt


def run(judge: str, ollama_model: str, limit: int | None = None) -> None:
    scenarios = SCENARIOS[:limit] if limit else SCENARIOS
    if judge == "claude":
        call = lambda p: call_llm_claude(p)  # noqa: E731
    else:
        call = lambda p: call_llm_ollama(p, ollama_model)  # noqa: E731

    rows: list[tuple[Scenario, dict[str, int], dict[str, int]]] = []
    for sc in scenarios:
        t0 = time.perf_counter()
        try:
            ctrl_response = call(build_control_prompt(sc.prompt))
        except Exception as exc:
            ctrl_response = f"<error: {exc}>"
        ctrl_score = score(ctrl_response, sc)
        try:
            mneme_response = call(build_with_mneme_prompt(sc.prompt))
        except Exception as exc:
            mneme_response = f"<error: {exc}>"
        mneme_score = score(mneme_response, sc)
        rows.append((sc, ctrl_score, mneme_score))
        print(
            f"[{sc.name:24s}] ctrl_hit={ctrl_score['expected_hit']} "
            f"mneme_hit={mneme_score['expected_hit']} "
            f"({time.perf_counter() - t0:.1f}s)"
        )

    print()
    print("=" * 80)
    print(f"HARD A/B RESULTS (judge={judge})")
    print("=" * 80)
    ctrl_total = sum(s[1]["expected_hit"] for s in rows)
    mneme_total = sum(s[2]["expected_hit"] for s in rows)
    n = len(rows)
    print(f"control: hit {ctrl_total}/{n} ({ctrl_total/n:.0%})")
    print(f"mneme:   hit {mneme_total}/{n} ({mneme_total/n:.0%})")
    print(f"delta:   +{mneme_total - ctrl_total} (absolute)")
    print()
    flips = [
        (sc.name, sc.category)
        for (sc, c, m) in rows
        if c["expected_hit"] == 0 and m["expected_hit"] == 1
    ]
    if flips:
        print("scenarios where mneme flipped a miss into a hit:")
        for name, cat in flips:
            print(f"  + [{cat}] {name}")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=["claude", "ollama"], default="claude")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.judge, args.model, args.limit)


if __name__ == "__main__":
    _main()
