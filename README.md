<p align="center">
  <img src="docs/mnemosine.png" alt="Mnemosyne — mother of the muses, goddess of memory" width="320" />
</p>

<h1 align="center">mneme</h1>

<p align="center"><em>Your AI never forgets what it can do.</em></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+" /></a>
  <a href="https://github.com/Luizhcrs/mneme/releases/tag/v0.1.0-rc1"><img src="https://img.shields.io/badge/release-v0.1.0--rc1-orange.svg" alt="Release" /></a>
</p>

<p align="center">
  <strong><a href="#mneme-pt-br">Português abaixo</a></strong> &nbsp;|&nbsp;
  <a href="docs/quickstart.md">Quickstart</a> &nbsp;|&nbsp;
  <a href="docs/architecture.md">Architecture</a> &nbsp;|&nbsp;
  <a href="docs/capability-card-format.md">Capability cards</a> &nbsp;|&nbsp;
  <a href="ROADMAP.md">Roadmap</a>
</p>

---

## The pain

You install Playwright as an MCP server. You ask Claude Code to scrape a website.

> Claude tries `curl https://...`, gets HTML it cannot parse, falls back to fragile regex, and **never once considers** that Playwright is sitting right there waiting.

This is **affordance blindness**. Your agent forgets the surface of capabilities it actually has and picks the first plausible execution path it can imagine. The same agent that has Telegram, GitHub, Postgres, Obsidian, PyAutoGUI, Docker installed — and ignores them under pressure.

It is documented in the literature:

- ["Lost in the Middle" (Liu et al., TACL 2024)](https://arxiv.org/abs/2307.03172) — content placed in the middle of a long context is ignored by the LLM's attention. Tool descriptions buried in a 200-MCP system prompt are dead weight.
- ["RAG-MCP" (Gan & Sun, 2025)](https://arxiv.org/abs/2505.03275) — empirical evidence: tool-selection accuracy collapses from **43.1% with retrieval to 13.6% without**. Triple the precision with the right retriever; lose two-thirds without it.

Existing memory libraries (Mem0, Zep, Letta, Memori, Anthropic Memory tool) all solve a *different* problem. They remember **facts about you**. None of them remember **what your agent itself can do**.

mneme fills that gap.

## The solution

mneme keeps a local, semantic index of every capability your agent has access to and **injects the relevant ones at the start of every prompt** — automatically, on every turn, with no agent-side awareness required.

```bash
pip install git+https://github.com/Luizhcrs/mneme.git
ollama pull nomic-embed-text
mneme init && mneme reindex
```

Add two lines to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": "python -m mneme.hooks.user_prompt_submit",
    "PostToolUse": "python -m mneme.hooks.post_tool_use"
  }
}
```

Done. Every prompt your agent receives now starts with a `<capabilities-available>` block listing the tools that match the task — by semantic similarity, in EN or PT-BR, with confidence scores.

## Does it actually work?

Measured on a 50-task benchmark (real `nomic-embed-text` via Ollama, mixed EN+PT-BR queries, 10 capability registry):

| Metric | Result |
|--------|-------:|
| top-1 affordance recall | **90%** |
| top-3 affordance recall | **94%** |
| top-5 affordance recall | **94%** |
| RAG-MCP minimum bar | 43% |
| **Margin over the published baseline** | **+51 percentage points** |

In plain English: **9 out of 10 times the agent gets the right tool on the first try**, with no fine-tuning, no cloud account, and a generic embedding model running on your laptop. The acceptance gate is reproducible — `MNEME_REAL_OLLAMA=1 pytest tests/test_benchmark_acceptance.py`.

### A/B test on actual agent behavior

The retrieval numbers above measure whether the right card appears at the top of the list. The harder question is whether the *agent* changes behavior when those cards are injected. Run `python tests/benchmark/ab_test.py` to see — same local LLM, 10 representative tasks, with vs without the mneme injection block:

| Variant | expected_hit | wrong_fallback | "I can't" |
|---------|:------------:|:--------------:|:---------:|
| control (no mneme) | 5 / 10 | 0 | 0 |
| mneme injection on | **9 / 10** | 0 | 0 |

**+4 hits, +40 percentage points absolute.** Scenarios where the agent went from "ignored the right tool" to "named it directly": GitHub issue creation, Docker compose up, local Ollama LLM, PyAutoGUI desktop screenshot. Reproducible against any local LLM via Ollama.

!!! note "About these numbers"
    The Phase 1 benchmark uses a 10-card seed registry and 50 queries (every query has a correct answer in the registry). top-3 = top-5 because retrieving 5 of 10 capabilities saturates recall at this scale. The CPU latency (2.6 s) is dominated by Ollama; on GPU it drops to 50–100 ms. Phase 2 will publish results on a 200+ card registry with no-match negatives — see [the roadmap](ROADMAP.md).

## What it looks like in practice

Real queries against the 10-card seed registry:

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

The agent receives that block before reasoning about your task. It cannot forget the tool because the tool is right there in front of it.

## What is in the box

- **Local-only.** [Ollama](https://ollama.ai) for embeddings, [SQLite](https://sqlite.org) + [`sqlite-vec`](https://github.com/asg017/sqlite-vec) for the vector store, JSONL for procedural memory. Zero cloud calls. Zero API cost.
- **Bilingual out of the box.** Category descriptions and seed cards in EN + PT-BR. Both languages embed into the same vector space.
- **Two-stage retrieval** ([AnyTool](https://arxiv.org/abs/2402.04253)). Top-K categories → top-N capabilities. Tunable to your registry size.
- **Procedural memory** ([Voyager](https://arxiv.org/abs/2305.16291)). Successful tool sequences persist as workflows and re-surface on similar future tasks.
- **Auto-discovery.** Scanner reads `claude mcp list`, plugins, and slash commands automatically.
- **Fail-safe.** Hook crashes never block Claude Code. Ollama down? Optional regex fallback (`MNEME_FALLBACK=1`).
- **Open standard.** Capability cards are plain YAML; submit a PR to publish cards for popular MCPs.

## Compared to other memory libraries

| Project | What it does | Resolves affordance blindness |
|---------|--------------|:---:|
| [Mem0](https://github.com/mem0ai/mem0) | User-facts memory layer | No |
| [Letta / MemGPT](https://github.com/letta-ai/letta) | Tiered memory with self-edit | Partial |
| [Zep + Graphiti](https://github.com/getzep/graphiti) | Temporal knowledge graph | Partial |
| [Anthropic Memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool) | First-party `/memory` directory | No |
| [Sourcegraph Cody](https://sourcegraph.com/docs/cody/capabilities/agentic-context-fetching) | Agentic MCP fetching | Partial (closed) |
| [Smithery MCP marketplace](https://smithery.ai/) | MCP discovery | No |
| **mneme** | Capability index + hook + retrieval | **Yes** |

mneme is **complementary**, not competitive. Run it alongside Mem0: Mem0 remembers your preferences, mneme remembers your tools.

## Contributing

Capability cards are plain YAML. The format is in [`docs/capability-card-format.md`](docs/capability-card-format.md). Pull requests adding cards for popular MCPs and plugins are welcome — bilingual EN+PT-BR descriptions encouraged.

## License

MIT — see [`LICENSE`](LICENSE).

---

<a id="mneme-pt-br"></a>

# mneme (Português)

> Tua IA nunca mais esquece o que pode fazer.

## A dor

Tu instala o Playwright como servidor MCP. Pede pro Claude Code raspar um site.

> Claude tenta `curl https://...`, recebe HTML que não consegue parsear, cai num regex frágil, e **em momento nenhum considera** que tem o Playwright disponível.

Isso é **affordance blindness**. O agente esquece a superfície de capabilities que tem instalada e escolhe o primeiro caminho plausível que imagina. O mesmo agente que tem Telegram, GitHub, Postgres, Obsidian, PyAutoGUI, Docker — e ignora quando precisa.

Está documentado:

- ["Lost in the Middle" (Liu et al., TACL 2024)](https://arxiv.org/abs/2307.03172) — conteúdo no meio de um contexto longo é ignorado pela attention do modelo. Descrições de ferramenta enterradas num system prompt de 200 MCPs viram peso morto.
- ["RAG-MCP" (Gan & Sun, 2025)](https://arxiv.org/abs/2505.03275) — empírico: precisão de seleção de ferramenta cai de **43,1% com retrieval para 13,6% sem**. Triplica a precisão com o retriever certo; perde dois terços sem ele.

Bibliotecas de memória existentes (Mem0, Zep, Letta, Memori, Anthropic Memory tool) resolvem outro problema. Lembram **fatos sobre você**. Nenhuma lembra **o que o agente pode fazer**.

mneme preenche essa lacuna.

## A solução

mneme mantém um índice local e semântico de cada capability que teu agente tem acesso e **injeta as relevantes no início de cada prompt** — automaticamente, em toda interação, sem o agente precisar saber que existe.

```bash
pip install git+https://github.com/Luizhcrs/mneme.git
ollama pull nomic-embed-text
mneme init && mneme reindex
```

Adiciona duas linhas em `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": "python -m mneme.hooks.user_prompt_submit",
    "PostToolUse": "python -m mneme.hooks.post_tool_use"
  }
}
```

Pronto. Todo prompt que teu agente recebe agora começa com um bloco `<capabilities-available>` listando as ferramentas que casam com a tarefa — por similaridade semântica, em PT-BR ou EN, com scores de confiança.

## Funciona mesmo?

Medido em benchmark de 50 tarefas (Ollama `nomic-embed-text` real, queries mistas EN+PT-BR, 10 capabilities):

| Métrica | Resultado |
|---------|----------:|
| top-1 affordance recall | **90%** |
| top-3 affordance recall | **94%** |
| top-5 affordance recall | **94%** |
| Mínimo RAG-MCP | 43% |
| **Margem sobre o baseline publicado** | **+51 pp** |

Em português claro: **9 de cada 10 vezes o agente acerta a ferramenta na primeira tentativa**, sem fine-tune, sem cloud, com modelo de embedding genérico rodando local. Acceptance gate reprodutível — `MNEME_REAL_OLLAMA=1 pytest tests/test_benchmark_acceptance.py`.

### A/B no comportamento real do agente

Os números acima medem se o card certo aparece no topo da lista. Pergunta mais dura: o *agente* muda comportamento quando esses cards são injetados? Rode `python tests/benchmark/ab_test.py` — mesmo LLM local, 10 tarefas representativas, com vs sem o bloco mneme:

| Variante | expected_hit | wrong_fallback | "Não consigo" |
|----------|:------------:|:--------------:|:-------------:|
| controle (sem mneme) | 5 / 10 | 0 | 0 |
| com injeção mneme | **9 / 10** | 0 | 0 |

**+4 acertos, +40 pontos percentuais absolutos.** Cenários onde o agente foi de "ignorou a ferramenta certa" pra "nomeou ela direto": criação de issue no GitHub, docker compose up, LLM local via Ollama, screenshot desktop com PyAutoGUI. Reproduzível com qualquer LLM local via Ollama.

## Como fica na prática

Queries reais contra o registry seed de 10 cards:

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

O agente recebe esse bloco antes de raciocinar sobre tua tarefa. Ele não tem como esquecer da ferramenta porque ela está logo ali na frente.

## O que tem na caixa

- **Só local.** [Ollama](https://ollama.ai) pra embeddings, [SQLite](https://sqlite.org) + [`sqlite-vec`](https://github.com/asg017/sqlite-vec) pro vector store, JSONL pra memória procedural. Zero cloud. Zero custo de API.
- **Bilíngue de fábrica.** Descrições de categoria e cards seed em EN + PT-BR. Ambas as línguas no mesmo espaço vetorial.
- **Retrieval em dois estágios** ([AnyTool](https://arxiv.org/abs/2402.04253)). Top-K categorias → top-N capabilities. Configurável por tamanho de registry.
- **Memória procedural** ([Voyager](https://arxiv.org/abs/2305.16291)). Sequências bem-sucedidas viram workflows persistidos e re-aparecem em tarefas similares.
- **Auto-discovery.** Scanner lê `claude mcp list`, plugins e slash commands automaticamente.
- **Fail-safe.** Crash do hook nunca trava Claude Code. Ollama caiu? Fallback regex opcional (`MNEME_FALLBACK=1`).
- **Padrão aberto.** Capability cards em YAML simples; PR pra publicar cards de MCPs populares.

## Contribuir

Capability cards são YAML simples. Formato em [`docs/capability-card-format.pt-BR.md`](docs/capability-card-format.pt-BR.md). PRs com cards de MCPs e plugins populares são bem-vindos — descrições bilíngues EN+PT-BR encorajadas.

## Licença

MIT.
