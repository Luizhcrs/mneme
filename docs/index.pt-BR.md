---
title: mneme — tua IA nunca mais esquece o que pode fazer
---

<style>
.md-typeset h1, .md-content__button { display: none; }
</style>

<div align="center" markdown>

![mneme — Mnemosyne](mnemosine.png){ width=280 }

# mneme

*Tua IA nunca mais esquece o que pode fazer.*

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Luizhcrs/mneme/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/badge/release-v0.1.0--rc1-orange.svg)](https://github.com/Luizhcrs/mneme/releases/tag/v0.1.0-rc1)

[Início rápido](quickstart.md){ .md-button .md-button--primary }
[GitHub](https://github.com/Luizhcrs/mneme){ .md-button }

</div>

---

## A dor

Tu instala o Playwright como servidor MCP. Pede pro Claude Code raspar um site.

!!! failure "Sem mneme"
    Claude tenta `curl https://...`, recebe HTML que não consegue parsear, cai num regex frágil, e **em momento nenhum considera** que tem o Playwright disponível.

Isso é **affordance blindness**. O agente esquece a superfície de capabilities que tem instalada e escolhe o primeiro caminho plausível. O mesmo agente que tem Telegram, GitHub, Postgres, Obsidian, PyAutoGUI, Docker — e ignora quando precisa.

Documentado:

- ["Lost in the Middle" (Liu et al., TACL 2024)](https://arxiv.org/abs/2307.03172) — conteúdo no meio do contexto longo é ignorado pela attention do modelo.
- ["RAG-MCP" (Gan & Sun, 2025)](https://arxiv.org/abs/2505.03275) — empírico: precisão de seleção de ferramenta cai de **43,1% com retrieval para 13,6% sem**.

Bibliotecas de memória (Mem0, Zep, Letta, Memori, Anthropic Memory tool) resolvem outro problema. Lembram **fatos sobre você**. Nenhuma lembra **o que o agente pode fazer**.

mneme preenche essa lacuna.

---

## A solução

```bash
pip install git+https://github.com/Luizhcrs/mneme.git
ollama pull nomic-embed-text
mneme init && mneme reindex
```

Adiciona em `~/.claude/settings.json`:

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

Pronto. Todo prompt que teu agente recebe agora começa com um bloco `<capabilities-available>` listando as ferramentas que casam com a tarefa — por similaridade semântica, em PT-BR ou EN, com scores.

---

## Funciona mesmo?

Benchmark de 50 tarefas (Ollama `nomic-embed-text` real, queries mistas EN+PT-BR, 10 capabilities):

| Métrica | Resultado |
|---------|----------:|
| top-1 affordance recall | **90%** |
| top-3 affordance recall | **94%** |
| top-5 affordance recall | **94%** |
| Mínimo RAG-MCP | 43% |
| **Margem sobre o baseline publicado** | **+51 pp** |

**9 de cada 10 vezes o agente acerta a ferramenta na primeira tentativa**, sem fine-tune, sem cloud, modelo de embedding genérico rodando local. Acceptance gate reprodutível — `MNEME_REAL_OLLAMA=1 pytest tests/test_benchmark_acceptance.py`.

---

## Na prática

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

## O que tem na caixa

<div class="grid cards" markdown>

- :material-cloud-off-outline:{ .lg .middle } **Só local**

    ---

    Ollama pra embeddings, SQLite + sqlite-vec pro vector store, JSONL pra memória procedural. Zero cloud. Zero custo de API.

- :material-translate:{ .lg .middle } **Bilíngue de fábrica**

    ---

    Descrições de categoria e cards seed em EN + PT-BR. Ambas as línguas no mesmo espaço vetorial.

- :material-stairs-up:{ .lg .middle } **Retrieval em dois estágios**

    ---

    Top-K categorias → top-N capabilities ([AnyTool](https://arxiv.org/abs/2402.04253)). Configurável por tamanho de registry.

- :material-brain:{ .lg .middle } **Memória procedural**

    ---

    Sequências bem-sucedidas viram workflows persistidos ([Voyager](https://arxiv.org/abs/2305.16291)) e re-aparecem em tarefas similares.

- :material-magnify-scan:{ .lg .middle } **Auto-discovery**

    ---

    Scanner lê `claude mcp list`, plugins e slash commands automaticamente.

- :material-shield-check:{ .lg .middle } **Fail-safe**

    ---

    Crash do hook nunca trava Claude Code. Fallback regex opcional quando Ollama indisponível.

- :material-chart-line:{ .lg .middle } **Auto-telemetria**

    ---

    Metadata local (sem conteúdo de prompt). `mneme insights` surfa cards mortos, outliers de latência, matches suspeitos automaticamente.

- :material-file-document-multiple:{ .lg .middle } **Padrão aberto**

    ---

    Capability cards em YAML simples; PR pra publicar cards de MCPs populares.

</div>

---

## Próximos passos

- [Início rápido](quickstart.md) — instalar + conectar hooks em 2 minutos
- [Arquitetura](architecture.md) — como o retrieval funciona por baixo
- [Capability cards](capability-card-format.md) — contribuir pra registry aberta
- [Roadmap](roadmap.md) — Phase 2 + Phase 3 pela frente
