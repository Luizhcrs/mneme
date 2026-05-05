# mneme Roadmap

> Cuidado: roadmap é vivo. Datas são teto, não compromisso. Phase 2/3 dependem do que o dogfood de Phase 1 expor.

## Phase 1 — DONE (v0.1.0-rc1, 2026-05-05)

Capability index + retrieval + hooks + bilingual EN+PT-BR + benchmark.

- 32 commits, 67 tests, 94% coverage, ruff/mypy strict clean.
- Acceptance gate: top-3 = 94% (RAG-MCP minimum 43%, +51pp margin).
- CI verde Python 3.11 + 3.12.
- Hooks wired no `~/.claude/settings.json` do dev — em uso diário.

---

## Phase 1.5 — Dogfood (week 1 pós-wire, 2026-05-05 → 2026-05-12)

**Goal:** colher signals reais antes de planejar Phase 2.

| Item | Critério de saída |
|------|-------------------|
| Usar Claude Code com hooks ligados em todo trabalho | 7 dias contínuos |
| Anotar quando mneme acerta vs erra (logs em `~/.claude/mneme/failures.log` + observações em [[Insight]]) | ≥ 30 prompts reais analisados |
| Adicionar capability cards próprios do user (MCPs, plugins, scripts locais) | ≥ 5 cards custom |
| Medir overhead real (latência adicionada por turno) | < 200 ms p95 em PC com Ollama na CPU |

**Decisões esperadas no fim:** quais Phase 2 items são dor REAL vs vontade abstrata.

### Quick wins na semana (paralelos ao dogfood)

- [ ] **Persist category vectors** entre processos. Hoje `Retriever.__init__` re-embeda 35 categorias (~1.5s no Ollama CPU) toda invocação do hook. Persistir em `category_vec` table: -90% latência em runs subsequentes.
- [ ] **`mneme add <yaml>`** comando CLI pra adicionar card sem editar YAML manualmente.
- [ ] **Shared `mneme.types.EmbedderProto`** módulo público (Important do reviewer Task 6, ainda não corrigido).
- [ ] **Hook timeout configurável** via `MNEME_TIMEOUT=8`.

---

## Phase 2 — Validation & Polish (weeks 2-5, 2026-05-12 → 2026-06-09)

**Goal:** transformar Phase 1 RC em v0.2.0 release público com confiança.

### Funcional

- [ ] **Reflexion subagent** — falha de tool dispara reflexão verbal via Haiku, persistida em `reflections.jsonl`. Injection no próximo prompt similar inclui reflexões relevantes ("já falhei aqui: ...").
- [ ] **Sleep-time consolidation cron** (LightMem pattern) — cron 03:00 noite faz: decay de capabilities sem uso 30+ dias, merge de workflows similares (cosine ≥ 0.9), synthesize meta-skills das top-10 mais usadas.
- [ ] **Filesystem watcher** — detect MCP install/remove em `~/.claude/plugins/` e propor diff sem precisar rodar `mneme rescan`.
- [ ] **Multi-tenant namespacing real** — `MNEME_NAMESPACE=cliente_X` agora apenas atributo dummy. Phase 2 enforça isolamento via partition no SQLite.
- [ ] **Capability versioning** — quando MCP atualiza, schema_version bump dispara re-embed.

### Qualidade

- [ ] **Real benchmark dataset 200+ tasks** — escala dos atuais 50, mais domains, mais variação de phrasing. Validate top-3 ≥ 80% nesse tamanho.
- [ ] **Fine-tune embedding** com synthetic ToolRet-train (200k exemplos) usando model leve (bge-m3 ou e5-mistral). Target: top-3 ≥ 90% no benchmark expandido.
- [ ] **Top-k dinâmico** — `top_categories` adaptativo por tamanho do registry: 3 (200+ caps), 7 (50-200), 10 (10-50), 35 (<10).
- [ ] **Embedding cache** — query repetida não re-embeda. LRU 512 entries.

### Adoção

- [ ] **`mneme registry` comando** — busca cards publicados na comunidade via repo central (proposta: `mneme.dev/registry` ou `github.com/mneme-cards`).
- [ ] **20+ capability cards oficiais** pra MCPs populares: Anthropic, Smithery top-20, MCPs do user.
- [ ] **Adapter Cursor / Continue.dev / Aider** — formato de hook varia por harness. Documentar + shim Python.
- [ ] **Pacote PyPI publicado** — `pip install mneme` funcionando (hoje só git clone + pip install -e .).

### Observabilidade

- [ ] **`mneme stats --detailed`** — taxa de retrieval, latência p50/p95, capabilities mais retornadas, prompts que não tiveram match.
- [ ] **Optional Prometheus exporter** — stats expostos em `:9100/metrics` se ligar `MNEME_METRICS=1`.

**Critério v0.2.0:** dogfood 30 dias + 200-task benchmark top-3 ≥ 80% + 20 cards oficiais + PyPI publicado.

---

## Phase 3 — Ecosystem (months 2-3, 2026-06-09 → 2026-08-09)

**Goal:** mneme vira padrão emergente no espaço de capability awareness.

- [ ] **Capability Card Spec v1.1** — schema YAML formalizado, validado por JSON Schema. Versão semântica, mecanismo de upgrade.
- [ ] **`mneme.dev/registry`** — site web pra browse + download de capability cards verificados. Lighthouse 100, mobile-first, EN+PT-BR.
- [ ] **GitHub Action `setup-mneme`** — `uses: Luizhcrs/mneme/setup@v1` no CI de outros projetos.
- [ ] **VS Code extension** — IntelliSense pra YAML de capability cards (autocomplete categorias, validação inline).
- [ ] **Anthropic Memory tool integration** — bridge: mneme escreve no `/memory` directory tool, Claude API consome nativamente.
- [ ] **Tool/MCP author SDK** — pacote `mneme-card` que publishers de MCP rodam pra gerar capability card automaticamente do schema MCP.
- [ ] **Open Telemetry tracing** — spans no hook pra debug de latência em prod.

**Critério v1.0.0:** 1000+ stars no GitHub OU 100+ cards na registry OU adoção em outro harness mainstream (Cursor/Continue/Aider).

---

## Phase 4 — Strategic plays (months 4+)

Long-term, alinhado com [[Supermente]]:

### Tese 4 — Vault que pensa

- mneme como dependência do agente de insight diário em vaults locais. Cron noturno gera insight do vault USANDO mneme pra saber que tem endpoints HTTP locais, Telegram MCP, embedder local disponíveis.

### Tese 1 — Orb agente universal

- mneme registry compartilhado entre Claude Code, Orb, alpha. Orb sabe que tem PyAutoGUI + visionQwen + Playwright sem hardcode.

### Tese 2 — Plataforma de agentes verticais

- mneme multi-tenant em produção. Cada cliente da plataforma tem namespace próprio (`tenant/restaurant_X/`). Cards específicos do domínio (restaurante: WhatsApp + Stripe + reservation_logic).
- Pode virar produto SaaS: "mneme cloud" — registry hosted, capability cards verificados, billing per-tenant.

### Tese 3 — Modelo próprio

- Fine-tuned embedding pra capability retrieval (synthetic ToolRet-train). Modelo leve, ~50M params, roda CPU. Substitui nomic-embed-text com model mneme-specific.
- Publicável como paper.

---

## Não-objetivos (anti-roadmap)

Coisas que mneme **não vai virar** pra evitar scope creep:

- Memory-of-facts (Mem0/Zep já fazem). mneme fica em affordances.
- Replacement de MCP marketplace (Smithery já faz). mneme consome marketplace, não compete.
- LLM agent harness próprio. mneme é layer ortogonal pra harness existentes.
- Embedding service em cloud. Local-only é diferencial; manter.
- Não-Python. Bindings TypeScript/Rust são Phase 4+ se demanda existir.

---

## Como contribuir

- **Capability cards** pra MCPs populares: PR com YAML em `src/mneme/community/`.
- **Bug reports**: GitHub issues. Inclua `mneme stats`, OS, Python version, output de `mneme search "<query>"`.
- **Feature ideas**: GitHub Discussions. Feature deve mapear pra alguma fase do roadmap acima OU justificar por que merece nova fase.

---

## Cadência de release

- **v0.1.x** — bugfixes Phase 1 RC. Patch a cada bug crítico encontrado em dogfood.
- **v0.2.0** — Phase 2 done. Target: 2026-06-09.
- **v0.3.x** — incrementais Phase 3.
- **v1.0.0** — quando tiver adoção fora do desenvolvedor (ver critérios Phase 3).

Versioning: SemVer estrito. Breaking changes em capability card format = major bump.
