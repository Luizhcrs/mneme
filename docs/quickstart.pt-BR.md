# mneme - Quickstart (PT-BR)

## Pré-requisitos

- Python 3.11+
- [Ollama](https://ollama.ai) rodando local
- Claude Code instalado (ou qualquer harness com hooks de prompt)

## Instalar

```bash
pip install mneme
ollama pull nomic-embed-text
mneme init
mneme reindex
```

`mneme init` cria `~/.claude/mneme/capabilities.yaml` com 10 capability cards iniciais.
`mneme reindex` embeda no SQLite + sqlite-vec local em `~/.claude/mneme/semantic.sqlite`.

## Conectar os hooks

Adicione em `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": "python -m mneme.hooks.user_prompt_submit",
    "PostToolUse": "python -m mneme.hooks.post_tool_use"
  }
}
```

O hook `UserPromptSubmit` injeta capabilities relevantes em cada prompt.
O hook `PostToolUse` registra sequências bem-sucedidas como memória procedural
pra que tarefas similares reutilizem o caminho que funcionou.

## Verificar

```bash
mneme search "tira print do site"
mneme stats
```

Deve retornar capabilities Playwright. Abre o Claude Code, manda um prompt
mencionando uma ferramenta instalada e o agente deve escolher certo de primeira.

## Adicionar capability cards próprios

Edita `~/.claude/mneme/capabilities.yaml`. Cada entrada segue o schema em
[`capability-card-format.pt-BR.md`](capability-card-format.pt-BR.md). Depois:

```bash
mneme reindex
```

## Fallback offline

Se Ollama não está disponível mas você ainda quer recall degradado:

```bash
export MNEME_FALLBACK=1
```

O hook então casa o prompt contra os `triggers` por substring (sem embeddings).
