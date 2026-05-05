"""Category metadata: bilingual EN+PT-BR descriptions for stage-1 retrieval.

Each description packs EN keywords first, PT-BR equivalents second. Both
languages are flattened into one embedding so queries in either language
align with the same category vector. This was a measured fix: PT-BR-only
queries were missing categories whose descriptions were EN-only, dropping
real-world recall against the benchmark numbers.
"""
from __future__ import annotations

from mneme.schema import CATEGORIES

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "agent_orchestration": (
        "multi-agent task graphs agent runtimes "
        "agentes orquestracao tarefas fluxo execucao"
    ),
    "audio_media": (
        "music podcasts DSP format conversion "
        "musica audio podcast som conversao formato"
    ),
    "automation_rpa": (
        "cron scripts robotic process automation scheduled jobs "
        "automacao agendado script tarefa rotina cron"
    ),
    "cloud_infra": (
        "AWS GCP Azure infrastructure as code Terraform Pulumi "
        "nuvem infraestrutura provedor cloud servidor"
    ),
    "code_search": (
        "codebase navigation AST tools symbol lookup grep find references "
        "buscar codigo simbolo funcao referencia navegar codigo"
    ),
    "comms": (
        "email SMS Telegram WhatsApp Slack Discord push notifications chat message "
        "mensagem notificacao alerta avisar email telegram whatsapp comunicacao"
    ),
    "container": (
        "Docker Podman Compose image registries container "
        "docker container imagem subir servico compose"
    ),
    "crypto": (
        "encryption hashing signing blockchain wallets cryptography "
        "criptografia hash assinatura blockchain carteira seguranca"
    ),
    "data_pipeline": (
        "ETL streaming batch jobs schedulers data pipeline "
        "pipeline dados ETL processamento lote streaming"
    ),
    "db": (
        "relational NoSQL vector time-series graph databases SQL query "
        "banco dados consulta query postgres mysql sqlite mongo redis tabela"
    ),
    "deploy": (
        "Coolify Vercel Netlify deployment platforms publish release "
        "deploy publicar lancar producao subir aplicacao coolify vercel"
    ),
    "desktop_automation": (
        "PyAutoGUI AppleScript Win32 accessibility GUI control desktop screen mouse keyboard "
        "automacao desktop tela mouse teclado clicar janela aplicativo"
    ),
    "documentation": (
        "docs generation markdown tools diagram rendering documentation "
        "documentacao docs markdown gerar diagrama"
    ),
    "embedded_iot": (
        "microcontrollers firmware edge devices Arduino ESP32 Raspberry "
        "microcontrolador firmware iot dispositivo arduino raspberry"
    ),
    "filesystem": (
        "file read write search archive sync filesystem path directory "
        "arquivo ler escrever pasta diretorio caminho ler arquivo abrir arquivo"
    ),
    "finance_trading": (
        "accounting trading portfolio market data finance investing "
        "financas contabilidade trading investimento mercado portfolio"
    ),
    "game_engine": (
        "Unity Unreal Godot game development tooling "
        "jogo game engine desenvolvimento jogo unity unreal godot"
    ),
    "geo_maps": (
        "geographic information geocoding routing satellite GIS map location "
        "geografia mapa localizacao gps satelite rota cep"
    ),
    "graphics_3d": (
        "Blender mesh rendering shaders Three.js 3D graphics "
        "3d grafico renderizar blender mesh shader modelagem"
    ),
    "hardware_io": (
        "USB serial GPIO sensors robotics hardware peripheral "
        "hardware sensor robotica usb serial gpio porta"
    ),
    "kubernetes": (
        "k8s Helm operators kubernetes pod deployment "
        "kubernetes k8s pod deployment helm operador"
    ),
    "math_scientific": (
        "sympy numpy advanced scientific computing math statistics "
        "matematica calculo cientifico estatistica numero"
    ),
    "ml_inference": (
        "local LLM vision embeddings ASR TTS runtime inference model "
        "inferencia modelo ia llm local rodar modelo ollama embedding"
    ),
    "ml_training": (
        "fine-tuning LoRA distributed training machine learning "
        "treinamento fine-tune lora ml aprendizado modelo treinar"
    ),
    "mobile_dev": (
        "iOS Android tooling emulators app packaging mobile "
        "mobile celular app android ios aplicativo emulador"
    ),
    "monitoring": (
        "logs metrics traces alerting dashboards observability "
        "logs metricas observabilidade alerta dashboard monitoramento"
    ),
    "nlp": (
        "tokenization parsing classification named entity recognition NLP text processing "
        "processamento texto nlp tokenizar classificar entidade"
    ),
    "security": (
        "vulnerability scan secrets signing pentest helpers security audit "
        "seguranca vulnerabilidade pentest auditoria segredo senha"
    ),
    "testing": (
        "unit integration end-to-end fuzzing benchmarking tests "
        "teste unitario integracao e2e fuzzing benchmark rodar testes"
    ),
    "vault_kb": (
        "Obsidian Notion knowledge bases semantic search over notes wiki "
        "obsidian notion vault notas conhecimento buscar nota wiki anotacao"
    ),
    "vcs": (
        "git GitHub GitLab pull request tooling commit branch repository "
        "git github gitlab commit branch pull request issue repositorio"
    ),
    "vision_image": (
        "image processing OCR object detection screenshot analysis computer vision "
        "imagem visao computacional ocr deteccao print captura tela analise"
    ),
    "voice_audio": (
        "text-to-speech speech-to-text voice cloning TTS STT "
        "voz fala tts stt audio narracao transcricao"
    ),
    "web_api": (
        "REST GraphQL clients webhooks HTTP utilities API call request "
        "api rest http requisicao endpoint webhook chamada"
    ),
    "web_browser": (
        "browser automation scraping headless rendering screenshots web page site "
        "navegador site web pagina playwright browser print site captura site scraping"
    ),
}

_described = set(CATEGORY_DESCRIPTIONS)
if _described != CATEGORIES:
    raise RuntimeError(
        "categories.py coverage mismatch: "
        f"missing={sorted(CATEGORIES - _described)}, "
        f"extra={sorted(_described - CATEGORIES)}"
    )
del _described
