<p align="center">
  <img src="docs/mnemosine.png" alt="Mnemosyne — mother of the muses, goddess of memory" width="320" />
</p>

<h1 align="center">mneme</h1>

<p align="center"><em>Capability-recall layer for Claude Code. Your AI never forgets what it can do.</em></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+" /></a>
  <a href="docs/superpowers/plans/2026-05-04-mneme-phase1.md"><img src="https://img.shields.io/badge/status-phase%201%20MVP-orange.svg" alt="Status: Phase 1" /></a>
</p>

<p align="center">
  <strong><a href="#mneme-pt-br">Português abaixo</a></strong> &nbsp;|&nbsp;
  <a href="docs/superpowers/specs/2026-05-04-mneme-design.md">Architecture</a> &nbsp;|&nbsp;
  <a href="docs/superpowers/plans/2026-05-04-mneme-phase1.md">Implementation Plan</a>
</p>

---

## The problem: affordance blindness

You install Playwright as an MCP server. You ask Claude Code to scrape a website. It tries `curl`, gets HTML it cannot parse, falls back to fragile regex, and never once considers that Playwright is sitting right there waiting.

This is **affordance blindness**: the agent forgets the surface of capabilities it actually has and picks the first plausible execution path it can imagine. It is documented in the literature as a context-window phenomenon (["Lost in the Middle", Liu et al., TACL 2024](https://arxiv.org/abs/2307.03172)) and as an empirical retrieval failure (["RAG-MCP", Gan & Sun, 2025](https://arxiv.org/abs/2505.03275) — accuracy drops from **43.13% with retrieval to 13.62% without**).

Existing memory layers (Mem0, Zep, Letta, Memori) solve a different problem: they remember **facts about the user**. None of them store **what the agent itself can do**. mneme fills that gap.

## The solution

mneme keeps a local index of every capability your agent has access to — MCPs, plugins, slash commands, custom scripts, project-local utilities — and injects the relevant ones at the start of every prompt the agent receives. Three published, peer-reviewed techniques in one library:

- **RAG-MCP** ([arXiv:2505.03275](https://arxiv.org/abs/2505.03275)) — retrieve a small, relevant subset of capabilities per turn instead of stuffing all 200 into the system prompt.
- **AnyTool** ([arXiv:2402.04253](https://arxiv.org/abs/2402.04253)) — two-stage hierarchy (top-3 categories → top-5 capabilities filtered by category) reports +35.4% pass rate over flat retrieval.
- **Voyager** ([arXiv:2305.16291](https://arxiv.org/abs/2305.16291)) — successful tool sequences are persisted as procedural memory and re-surfaced on similar tasks (3-15× improvement vs ReAct/Reflexion in the original benchmark).

Plus an opinionated stack:

- **Local-only**: [Ollama](https://ollama.ai) `nomic-embed-text` for embeddings, [SQLite](https://sqlite.org) + [`sqlite-vec`](https://github.com/asg017/sqlite-vec) for the vector store, JSONL for procedural memory. Zero cloud calls. Zero API cost.
- **Plug-and-play**: three hooks in `~/.claude/settings.json`. No fork of Claude Code required.
- **Auto-discovery**: scanner finds installed MCPs, plugins, and slash commands — no manual registration.
- **Improves with use**: every successful workflow accumulates in the skill library; future similar tasks benefit automatically.

## Architecture sketch

```
                          UserPromptSubmit hook
                                 |
          (1) embed prompt with instruction prefix
                                 |
          (2) top-3 categories (cosine)
                                 |
          (3) top-5 capabilities, filtered by category, threshold 0.65
                                 |
          (4) inject <capabilities-available>...</> at START of prompt
                          (Lost in the Middle: never the middle)
                                 |
                       Claude Code answers
                                 |
                          PostToolUse hook
                                 |
   on success: persist successful tool sequence to procedural.jsonl
   on failure: log to failures.log (Phase 2: LLM reflection)
```

Source of truth lives at `~/.claude/mneme/`:

```
~/.claude/mneme/
├── capabilities.yaml      # hand-edited or scanner-discovered
├── semantic.sqlite        # vector index (sqlite-vec)
├── procedural.jsonl       # successful tool sequences (Voyager pattern)
├── reflections.jsonl      # failure notes (Phase 2)
└── failures.log           # raw failure log (Phase 1)
```

## Why this is worth your time

If you are running Claude Code (or Cursor, or Continue.dev, or any agent harness with prompt hooks) with more than ~10 MCPs, plugins, or skills installed, your agent is forgetting them. Not all of them, not all the time — but often enough that you have caught yourself thinking "wait, why didn't it use X?" That is affordance blindness, and it costs you tokens, attempts, and trust.

mneme is a small, local, MIT-licensed library that fixes this without any cloud account, model fine-tuning, or harness modification. It is built on three published, replicable papers, not vibes.

## Status

Phase 1 MVP **release candidate** tagged at [`v0.1.0-rc1`](https://github.com/Luizhcrs/mneme/releases/tag/v0.1.0-rc1). Tracker: [`docs/superpowers/plans/2026-05-04-mneme-phase1.md`](docs/superpowers/plans/2026-05-04-mneme-phase1.md).

| Layer | Status | Tests |
|-------|--------|-------|
| Schema (35 categories, pydantic v2) | done | 11 |
| Embedder (Ollama + instruction prefix, fail-loud) | done | 6 |
| Store (sqlite-vec + JSONL, context manager) | done | 9 |
| Loader (YAML + 10 seed cards) | done | 6 |
| Retrieval (two-stage AnyTool, procedural workflows) | done | 6 |
| Hooks (UserPromptSubmit, PostToolUse, regex fallback) | done | 10 |
| Scanner (claude mcp list, plugins, commands) | done | 3 |
| Normalizer (rule-based EasyTool format) | done | 4 |
| CLI (init, reindex, list, search, stats) | done | 8 |
| Benchmark (50-task dataset, acceptance gates) | done | 3 |
| Bilingual docs (EN + PT-BR) | done | — |
| GitHub Actions CI (Python 3.11 + 3.12) | done | — |

**66 tests passing, 94% coverage. ruff and mypy strict clean.**

Phase 1 acceptance gates (validated by the 50-task benchmark): top-3 affordance recall ≥ 43% (RAG-MCP minimum reproduced, gated behind `MNEME_REAL_OLLAMA=1`), avg latency < 100 ms with the deterministic fake embedder, token overhead per turn < 5%.

## Quickstart (preview — final command set lands in Task 14)

```bash
pip install mneme
ollama pull nomic-embed-text
mneme init
mneme rescan --apply
```

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": "python -m mneme.hooks.user_prompt_submit",
    "PostToolUse": "python -m mneme.hooks.post_tool_use"
  }
}
```

## Contributing capability cards

mneme uses an open YAML schema for capability cards. The format is defined in [`docs/capability-card-format.md`](docs/capability-card-format.md) (lands in Task 17). Pull requests adding cards for popular MCPs and plugins are welcome.

## Related work

| Project | What it does | Resolves affordance blindness? |
|---------|--------------|--------------------------------|
| [Mem0](https://github.com/mem0ai/mem0) | User-facts memory layer | No (fact-centric, not affordance-centric) |
| [Letta / MemGPT](https://github.com/letta-ai/letta) | Tiered memory with self-edit | Partial (registry hardcoded at boot) |
| [Zep + Graphiti](https://github.com/getzep/graphiti) | Temporal knowledge graph | Partial (adaptable with custom Tool ontology) |
| [Anthropic Memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool) | First-party `/memory` directory | No (barebones, you implement retrieval) |
| [Sourcegraph Cody](https://sourcegraph.com/docs/cody/capabilities/agentic-context-fetching) | Agentic MCP fetching | Partial (closed, manual registry) |
| [Smithery MCP marketplace](https://smithery.ai/) | MCP discovery | No (catalog, not memory) |
| **mneme** | Capability index + hook + retrieval | Yes |

mneme is **complementary**, not competitive. You can run mneme alongside Mem0 — Mem0 remembers your preferences, mneme remembers your tools.

## License

MIT — see [`LICENSE`](LICENSE).

Citing the papers used in the design is encouraged but not required.

---

<a id="mneme-pt-br"></a>
## mneme (Português)

> Camada de recall de capabilities pro Claude Code. Tua IA nunca mais esquece o que pode fazer.

### O problema: affordance blindness

Você instala o Playwright como servidor MCP. Pede pro Claude Code raspar um site. Ele tenta `curl`, recebe HTML que não consegue parsear, cai num regex frágil, e em momento nenhum considera que tem o Playwright disponível.

Isso é **affordance blindness**: o agente esquece a superfície de capabilities que tem disponível e escolhe o primeiro caminho plausível que imagina. Está documentado na literatura como fenômeno de context window (["Lost in the Middle", Liu et al., TACL 2024](https://arxiv.org/abs/2307.03172)) e como falha empírica de retrieval (["RAG-MCP", Gan & Sun, 2025](https://arxiv.org/abs/2505.03275) — precisão cai de **43,13% com retrieval pra 13,62% sem**).

Camadas de memória existentes (Mem0, Zep, Letta, Memori) resolvem outro problema: lembram **fatos sobre o usuário**. Nenhuma delas armazena **o que o próprio agente pode fazer**. mneme preenche essa lacuna.

### A solução

mneme mantém um índice local de cada capability que teu agente tem acesso — MCPs, plugins, slash commands, scripts custom, utilitários do projeto — e injeta as relevantes no início de cada prompt. Três técnicas peer-reviewed publicadas, em uma só biblioteca:

- **RAG-MCP** ([arXiv:2505.03275](https://arxiv.org/abs/2505.03275)) — recupera um subconjunto pequeno e relevante por turno em vez de empilhar 200 descrições no system prompt.
- **AnyTool** ([arXiv:2402.04253](https://arxiv.org/abs/2402.04253)) — hierarquia em dois estágios (top-3 categorias → top-5 capabilities filtradas) reporta +35,4% de pass rate sobre retrieval flat.
- **Voyager** ([arXiv:2305.16291](https://arxiv.org/abs/2305.16291)) — sequências bem-sucedidas viram memória procedural e são re-apresentadas em tarefas similares (3-15× sobre ReAct/Reflexion no benchmark original).

Stack opinativa:

- **Só local**: [Ollama](https://ollama.ai) `nomic-embed-text` pra embeddings, [SQLite](https://sqlite.org) + [`sqlite-vec`](https://github.com/asg017/sqlite-vec) pro vector store, JSONL pra memória procedural. Zero chamadas cloud. Zero custo de API.
- **Plug-and-play**: três hooks no `~/.claude/settings.json`. Sem fork do Claude Code.
- **Auto-discovery**: scanner detecta MCPs, plugins e slash commands instalados — sem cadastro manual.
- **Cresce com uso**: cada workflow bem-sucedido acumula na skill library; tarefas similares futuras se beneficiam automaticamente.

### Vale teu tempo?

Se tu roda Claude Code (ou Cursor, ou Continue.dev, ou qualquer harness com hooks de prompt) com mais de ~10 MCPs/plugins/skills instalados, teu agente está esquecendo deles. Não todos, não sempre — mas com frequência suficiente pra tu já ter pensado "espera, por que ele não usou o X?". Isso é affordance blindness, e custa tokens, tentativas e confiança.

mneme é uma biblioteca pequena, local e MIT que corrige isso sem conta cloud, fine-tune ou modificação do harness. Construída em três papers replicáveis, não em achismo.

### Status

Phase 1 MVP **em desenvolvimento**. Tracker: [`docs/superpowers/plans/2026-05-04-mneme-phase1.md`](docs/superpowers/plans/2026-05-04-mneme-phase1.md).

### Quickstart (preview)

```bash
pip install mneme
ollama pull nomic-embed-text
mneme init
mneme rescan --apply
```

Adiciona em `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": "python -m mneme.hooks.user_prompt_submit",
    "PostToolUse": "python -m mneme.hooks.post_tool_use"
  }
}
```

### Contribuir capability cards

mneme usa um schema YAML aberto pra capability cards. Formato em [`docs/capability-card-format.pt-BR.md`](docs/capability-card-format.pt-BR.md) (chega na Task 17). PRs adicionando cards pra MCPs e plugins populares são bem-vindos.

### Licença

MIT.
