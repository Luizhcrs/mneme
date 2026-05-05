# mneme Quickstart

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) running locally
- Claude Code installed (or any agent harness with prompt hooks)

## Install

```bash
pip install mneme
ollama pull nomic-embed-text
mneme init
mneme reindex
```

`mneme init` creates `~/.claude/mneme/capabilities.yaml` with 10 starter capability cards.
`mneme reindex` embeds them into a local SQLite + sqlite-vec store at `~/.claude/mneme/semantic.sqlite`.

## Wire the hooks

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": "python -m mneme.hooks.user_prompt_submit",
    "PostToolUse": "python -m mneme.hooks.post_tool_use"
  }
}
```

The `UserPromptSubmit` hook injects relevant capabilities into every prompt.
The `PostToolUse` hook records successful tool sequences as procedural memory
so similar future tasks reuse them.

## Verify

```bash
mneme search "screenshot the homepage"
mneme stats
```

You should see Playwright capabilities surface for the screenshot query.
Open Claude Code, type a prompt that mentions a tool you have installed,
and the agent should now consistently pick the correct capability.

## Add your own capability cards

Edit `~/.claude/mneme/capabilities.yaml`. Each entry follows the schema in
[`capability-card-format.md`](capability-card-format.md). After editing:

```bash
mneme reindex
```

## Offline fallback

If Ollama is unavailable but you still want degraded recall, set:

```bash
export MNEME_FALLBACK=1
```

The hook then matches the prompt against `triggers` substrings (no embeddings).
