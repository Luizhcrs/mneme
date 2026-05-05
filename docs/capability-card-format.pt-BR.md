# Formato Capability Card (Spec Aberto v1) - PT-BR

Cada capability card é uma entrada em `capabilities.yaml`. Descreve uma ferramenta, MCP, plugin, command ou skill que o agente pode usar.

## Campos obrigatórios

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `id` | string (snake_case) | Slug único global. Padrão: `^[a-z][a-z0-9_]*$` |
| `name` | string | Nome legível |
| `category` | string | Uma das 35 categorias canônicas (`categories.md`) |
| `action_verb` | string | Frase curta da ação: "captura imagem de página web", "manda mensagem Telegram" |
| `triggers` | list[string] | Palavras ou intents que devem surfa esta capability (multilíngue OK) |
| `description` | string | Um parágrafo conciso (formato EasyTool) |
| `params_required` | list[string] | Parâmetros obrigatórios |
| `params_optional` | list[string] | Parâmetros opcionais |
| `example` | string | Exemplo concreto de uso |
| `schema_version` | string | Versão da ferramenta/MCP descrita |
| `source` | string | Um de: `mcp`, `skill`, `plugin`, `command`, `project`, `service`, `os`, `manual` |

## Campos opcionais

| Campo | Default | Notas |
|-------|---------|-------|
| `namespace` | `"global"` | Escopo multi-tenant (Fase 2) |
| `success_count` / `failure_count` | 0 | Atualizado pelo hook `PostToolUse` (>= 0) |
| `last_used` | null | Timestamp ISO-8601 |
| `decay_score` | 1.0 | Reduzido pela consolidação noturna se não usado (Fase 2; em `[0.0, 1.0]`) |

## Exemplo

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

## Validação

O schema é validado no load por `mneme.schema.CapabilityCard`. As restrições
aparecem no JSON Schema (`CapabilityCard.model_json_schema()`), então editores
YAML e IDE plugins conseguem validar cards inline.
