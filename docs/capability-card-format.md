# Capability Card Format (Open Spec v1)

A capability card is one entry in `capabilities.yaml`. Each card describes one tool, MCP, plugin, command, or skill the agent can use.

## Required fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string (snake_case) | Globally unique slug. Pattern: `^[a-z][a-z0-9_]*$` |
| `name` | string | Human-readable name |
| `category` | string | One of the 35 canonical categories (see `categories.md`) |
| `action_verb` | string | One short phrase describing the action: "captures image of web page", "sends a Telegram message" |
| `triggers` | list[string] | Keywords or intents that should surface this capability (multilingual welcome) |
| `description` | string | One paragraph; concise per EasyTool format |
| `params_required` | list[string] | Required parameters by name |
| `params_optional` | list[string] | Optional parameters by name |
| `example` | string | One concrete usage example |
| `schema_version` | string | Tool/MCP version this card describes |
| `source` | string | One of: `mcp`, `skill`, `plugin`, `command`, `project`, `service`, `os`, `manual` |

## Optional fields

| Field | Default | Notes |
|-------|---------|-------|
| `namespace` | `"global"` | Multi-tenant scope (Phase 2 enforcement) |
| `success_count` / `failure_count` | 0 | Maintained by the `PostToolUse` hook (>= 0) |
| `last_used` | null | ISO-8601 timestamp |
| `decay_score` | 1.0 | Reduced by nightly consolidation if unused (Phase 2; in `[0.0, 1.0]`) |

## Example

```yaml
- id: telegram_send
  name: Telegram Send Message
  category: comms
  action_verb: "sends a message to a Telegram chat"
  triggers: [telegram, manda mensagem, notifica, alerta]
  description: MCP Telegram plugin. Sends text to a chat_id.
  params_required: [chat_id, text]
  params_optional: [parse_mode]
  example: 'Call: telegram.send(chat_id="12345678", text="deploy ok")'
  schema_version: "1"
  source: mcp
```

## Validation

The schema is enforced at load time by `mneme.schema.CapabilityCard`. Constraints
appear in the JSON Schema (`CapabilityCard.model_json_schema()`) so YAML editors
and IDE plugins can validate cards inline.
