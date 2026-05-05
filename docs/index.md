---
title: mneme — your AI never forgets what it can do
---

<style>
.md-typeset h1, .md-content__button { display: none; }
</style>

<div align="center" markdown>

![mneme — Mnemosyne](mnemosine.png){ width=280 }

# mneme

*Your AI never forgets what it can do.*

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Luizhcrs/mneme/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/badge/release-v0.1.0--rc1-orange.svg)](https://github.com/Luizhcrs/mneme/releases/tag/v0.1.0-rc1)

[Quickstart](quickstart.md){ .md-button .md-button--primary }
[GitHub](https://github.com/Luizhcrs/mneme){ .md-button }

</div>

---

## The pain

You install Playwright as an MCP server. You ask Claude Code to scrape a website.

!!! failure "Without mneme"
    Claude tries `curl https://...`, gets HTML it cannot parse, falls back to fragile regex, and **never once considers** that Playwright is sitting right there waiting.

This is **affordance blindness**. Your agent forgets the surface of capabilities it actually has and picks the first plausible execution path it can imagine. The same agent that has Telegram, GitHub, Postgres, Obsidian, PyAutoGUI, Docker installed — and ignores them under pressure.

Documented in the literature:

- ["Lost in the Middle" (Liu et al., TACL 2024)](https://arxiv.org/abs/2307.03172) — content placed in the middle of a long context is ignored by the LLM's attention.
- ["RAG-MCP" (Gan & Sun, 2025)](https://arxiv.org/abs/2505.03275) — empirical: tool-selection accuracy collapses from **43.1% with retrieval to 13.6% without**.

Existing memory libraries (Mem0, Zep, Letta, Memori, Anthropic Memory tool) all solve a *different* problem. They remember **facts about you**. None remember **what your agent itself can do**.

mneme fills that gap.

---

## The solution

```bash
pip install git+https://github.com/Luizhcrs/mneme.git
ollama pull nomic-embed-text
mneme init && mneme reindex
```

Add two lines to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [{ "type": "command", "command": "python -m mneme.hooks.user_prompt_submit" }] }
    ],
    "PostToolUse": [
      { "hooks": [{ "type": "command", "command": "python -m mneme.hooks.post_tool_use" }] }
    ]
  }
}
```

Done. Every prompt your agent receives now starts with a `<capabilities-available>` block listing the tools that match the task — by semantic similarity, in EN or PT-BR, with confidence scores.

---

## Does it actually work?

Measured on a 50-task benchmark (real `nomic-embed-text` via Ollama, mixed EN+PT-BR queries, 10 capability registry):

| Metric | Result |
|--------|-------:|
| top-1 affordance recall | **90%** |
| top-3 affordance recall | **94%** |
| top-5 affordance recall | **94%** |
| RAG-MCP minimum bar | 43% |
| **Margin over the published baseline** | **+51 pp** |

**9 out of 10 times the agent gets the right tool on the first try**, with no fine-tuning, no cloud account, and a generic embedding model running on your laptop. The acceptance gate is reproducible — `MNEME_REAL_OLLAMA=1 pytest tests/test_benchmark_acceptance.py`.

---

## In practice

```text
QUERY: "manda mensagem no telegram avisando que deploy terminou"
  → telegram_send (comms, score=0.76)

QUERY: "abre uma issue no github sobre esse bug"
  → github_create_issue (vcs, score=0.74)

QUERY: "sobe os containers do projeto com docker"
  → docker_compose_up (container, score=0.73)

QUERY: "roda um modelo local de ia"
  → ollama_generate (ml_inference, score=0.72)

QUERY: "busca minhas notas sobre memoria no obsidian"
  → obsidian_search_vault (vault_kb, score=0.69)
```

---

## What is in the box

<div class="grid cards" markdown>

- :material-cloud-off-outline:{ .lg .middle } **Local-only**

    ---

    Ollama for embeddings, SQLite + sqlite-vec for the vector store, JSONL for procedural memory. Zero cloud calls. Zero API cost.

- :material-translate:{ .lg .middle } **Bilingual out of the box**

    ---

    Category descriptions and seed cards in EN + PT-BR. Both languages embed into the same vector space.

- :material-stairs-up:{ .lg .middle } **Two-stage retrieval**

    ---

    Top-K categories → top-N capabilities ([AnyTool](https://arxiv.org/abs/2402.04253)). Tunable to your registry size.

- :material-brain:{ .lg .middle } **Procedural memory**

    ---

    Successful tool sequences persist as workflows ([Voyager](https://arxiv.org/abs/2305.16291)) and re-surface on similar future tasks.

- :material-magnify-scan:{ .lg .middle } **Auto-discovery**

    ---

    Scanner reads `claude mcp list`, plugins, and slash commands automatically.

- :material-shield-check:{ .lg .middle } **Fail-safe**

    ---

    Hook crashes never block Claude Code. Optional regex fallback when Ollama is unavailable.

- :material-chart-line:{ .lg .middle } **Auto-telemetry**

    ---

    Local-only metadata (no prompt content). `mneme insights` surfaces dead cards, latency outliers, suspicious matches automatically.

- :material-file-document-multiple:{ .lg .middle } **Open standard**

    ---

    Capability cards are plain YAML; submit a PR to publish cards for popular MCPs.

</div>

---

## Compared to other memory libraries

| Project | What it does | Resolves affordance blindness |
|---------|--------------|:---:|
| [Mem0](https://github.com/mem0ai/mem0) | User-facts memory layer | :material-close: |
| [Letta / MemGPT](https://github.com/letta-ai/letta) | Tiered memory with self-edit | Partial |
| [Zep + Graphiti](https://github.com/getzep/graphiti) | Temporal knowledge graph | Partial |
| [Anthropic Memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool) | First-party `/memory` directory | :material-close: |
| [Sourcegraph Cody](https://sourcegraph.com/docs/cody/capabilities/agentic-context-fetching) | Agentic MCP fetching | Partial (closed) |
| [Smithery MCP marketplace](https://smithery.ai/) | MCP discovery | :material-close: |
| **mneme** | Capability index + hook + retrieval | :material-check-bold: |

mneme is **complementary**, not competitive. Run it alongside Mem0: Mem0 remembers your preferences, mneme remembers your tools.

---

## Next steps

- [Quickstart](quickstart.md) — install + wire hooks in 2 minutes
- [Architecture](architecture.md) — how retrieval works under the hood
- [Capability cards](capability-card-format.md) — contribute to the open registry
- [Roadmap](roadmap.md) — Phase 2 + Phase 3 ahead
