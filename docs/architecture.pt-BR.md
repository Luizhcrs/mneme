# Arquitetura mneme (PT-BR)

## Componentes

```
~/.claude/mneme/
├── capabilities.yaml        # fonte da verdade, editada ou descoberta por scanner
├── semantic.sqlite          # vetores das capabilities (sqlite-vec)
├── procedural.jsonl         # sequências bem-sucedidas (padrão Voyager)
├── reflections.jsonl        # notas de falhas (Fase 2)
└── failures.log             # log bruto de falhas (Fase 1)
```

Hooks declarados em `~/.claude/settings.json`:

- `UserPromptSubmit` -> `python -m mneme.hooks.user_prompt_submit` injeta
  capabilities relevantes no início do prompt.
- `PostToolUse` -> `python -m mneme.hooks.post_tool_use` registra sucesso e
  falha de cada uso de ferramenta.

## Retrieval

Hierarquia em dois estágios (AnyTool [arXiv:2402.04253](https://arxiv.org/abs/2402.04253)):

1. Embeda o prompt com prefixo de instrução: `"Given this task intent, retrieve tools that enable it: <prompt>"`.
2. Top-3 categorias por cosine (35 categorias indexadas em memória na construção do Retriever).
3. Top-5 capabilities filtradas pelas categorias, threshold >= 0.65.
4. Opcionalmente recupera top-3 workflows procedurais cujo `situation` também passa do threshold.
5. Formata o bloco de injeção (schema EasyTool conciso) e coloca no INÍCIO do prompt aumentado.

Se nada passa do threshold, nada é injetado (fail-safe). O prompt nunca é poluído com matches de baixa confiança.

## Por que essas escolhas

- **Só local.** Zero dependência cloud; roda em laptop sem conta.
- **Embeddings instruction-tuned.** Embeddings IR genéricos pontuam nDCG@10 = 33.83 em tool retrieval (["Retrieval Models Aren't Tool-Savvy", ACL 2025](https://arxiv.org/abs/2503.01763)). Instruction-tuning é a mitigação mais barata.
- **Hierarquia em dois estágios.** Retrieval flat em escala degrada; AnyTool reporta +35.4% sobre baseline flat.
- **Memória procedural separada da semântica.** [Voyager (NeurIPS 2023)](https://arxiv.org/abs/2305.16291) mostra 3-15x quando sequências bem-sucedidas são armazenadas como código, não como fatos.
- **Lost in the Middle.** Injeção no INÍCIO do prompt (nunca no meio) por Liu et al., [TACL 2024](https://arxiv.org/abs/2307.03172).
- **Fail-safe sempre.** Crash do hook nunca trava Claude Code; Ollama indisponível cai pra regex sobre triggers se habilitado (`MNEME_FALLBACK=1`).
