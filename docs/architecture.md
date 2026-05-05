# mneme Architecture

## Components

```
~/.claude/mneme/
├── capabilities.yaml        # source of truth, hand-edited or scanner-discovered
├── semantic.sqlite          # capability vectors (sqlite-vec)
├── procedural.jsonl         # successful tool sequences (Voyager pattern)
├── reflections.jsonl        # failure notes (Phase 2)
└── failures.log             # raw failure log (Phase 1)
```

Hooks declared in `~/.claude/settings.json`:

- `UserPromptSubmit` -> `python -m mneme.hooks.user_prompt_submit` injects
  relevant capabilities at the start of the prompt.
- `PostToolUse` -> `python -m mneme.hooks.post_tool_use` records workflow
  success and failure outcomes.

## Retrieval

Two-stage hierarchy per AnyTool ([arXiv:2402.04253](https://arxiv.org/abs/2402.04253)):

1. Embed prompt with instruction prefix: `"Given this task intent, retrieve tools that enable it: <prompt>"`.
2. Cosine top-3 categories (35 categories indexed in memory at Retriever construction).
3. Cosine top-5 capabilities filtered by those categories, threshold >= 0.65.
4. Optionally retrieve top-3 procedural workflows whose `situation` cosine to the prompt also clears the threshold.
5. Format the injection block (EasyTool concise schema) and place it at the START of the augmented prompt.

If no result clears the threshold, no injection occurs (fail-safe). The prompt is never polluted with low-confidence matches.

## Why these design choices

- **Local-only.** Zero cloud dependencies; runs on a developer laptop without an account.
- **Instruction-tuned embeddings.** Generic IR embeddings score nDCG@10 = 33.83 on tool retrieval (["Retrieval Models Aren't Tool-Savvy", ACL 2025](https://arxiv.org/abs/2503.01763)). Instruction-tuning is the cheapest mitigation that preserves Phase 1 simplicity.
- **Two-stage hierarchy.** Flat retrieval at scale degrades; AnyTool reports +35.4% pass-rate over flat baselines.
- **Procedural memory separate from semantic.** [Voyager (NeurIPS 2023)](https://arxiv.org/abs/2305.16291) shows 3-15x improvement when successful sequences are stored as code, not as facts.
- **Lost in the Middle placement.** Injection at the START of the prompt (never the middle) per Liu et al., [TACL 2024](https://arxiv.org/abs/2307.03172).
- **Fail-safe everywhere.** A hook crash never blocks Claude Code; missing Ollama falls back to regex matching on triggers when explicitly enabled (`MNEME_FALLBACK=1`).
