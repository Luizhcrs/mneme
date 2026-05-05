"""Discovery: scan installed MCPs, plugins, slash commands into capability cards."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import cast

from mneme.categories import CATEGORY_DESCRIPTIONS
from mneme.schema import CATEGORIES, CapabilityCard, Category


def _resolve_executable(name: str) -> str | None:
    """Resolve a CLI name to a full path so subprocess.run finds it on Windows.

    On Windows, ``claude`` is typically installed as ``claude.cmd``; bare
    ``subprocess.run(["claude", ...])`` fails because Python does not append
    PATHEXT extensions automatically. We use ``shutil.which`` which honours
    PATHEXT.
    """
    import shutil

    return shutil.which(name) or shutil.which(f"{name}.cmd") or shutil.which(f"{name}.exe")


def _run_command(cmd: list[str]) -> str:
    if cmd and not Path(cmd[0]).is_absolute():
        resolved = _resolve_executable(cmd[0])
        if resolved is not None:
            cmd = [resolved, *cmd[1:]]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    return proc.stdout


_MCP_LINE_LEGACY = re.compile(r"^(?P<name>[a-z][a-z0-9_-]*)\s*\(mcp\):\s*(?P<desc>.+)$")
_MCP_LINE_MODERN = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9_:.\- ]*?):\s+(?P<rest>.+?)$"
)
_MCP_STATUS_CONNECTED = re.compile(r"(?:✓\s*Connected|Connected)")
_MCP_STATUS_NEEDS_AUTH = re.compile(r"Needs authentication")
_WORD = re.compile(r"[a-z0-9]+")


def _safe_id(s: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", s.lower()).strip("_")


def _read_plugin_manifest(
    entries: list[dict[str, object]], simple_name: str
) -> tuple[str | None, list[str]]:
    """Return (description, extra_triggers) by reading the plugin's own manifest.

    Claude Code installs each plugin under
    ``<installPath>/.claude-plugin/plugin.json`` with a rich `description`
    written by the plugin author (Microsoft, Anthropic, etc.). Reading it
    promotes the discovered card from a hollow stub to something the
    embedding model can actually match against a user query.

    Skill names found alongside the manifest are also harvested as triggers
    so queries like 'screenshot the homepage' surface the plugin even when
    the manifest description does not say 'screenshot' verbatim.
    """
    for entry in entries:
        install_path = entry.get("installPath")
        if not isinstance(install_path, str):
            continue
        plugin_dir = Path(install_path)
        plugin_json = plugin_dir / ".claude-plugin" / "plugin.json"
        description: str | None = None
        if plugin_json.exists():
            try:
                data = json.loads(plugin_json.read_text(encoding="utf-8"))
                desc = data.get("description")
                if isinstance(desc, str) and desc.strip():
                    description = desc.strip()
            except (json.JSONDecodeError, OSError):
                description = None

        triggers: list[str] = []
        skills_dir = plugin_dir / "skills"
        if skills_dir.exists():
            for child in skills_dir.iterdir():
                if child.is_dir():
                    triggers.append(child.name)

        if description is not None or triggers:
            return description, triggers

    return None, []


def _guess_category(name: str, description: str) -> Category:
    """Best-effort category for a discovered card via keyword overlap.

    Counts how many distinct keywords from each category description appear
    in the name+description text. Falls back to ``agent_orchestration`` when
    no category beats the threshold (intentionally generic, signals "manual
    review needed").
    """
    text = f"{name} {description}".lower()
    text_words = set(_WORD.findall(text))
    if not text_words:
        return "agent_orchestration"

    best_cat = "agent_orchestration"
    best_score = 0
    for cat, desc in CATEGORY_DESCRIPTIONS.items():
        cat_words = {w for w in _WORD.findall(desc.lower()) if len(w) > 3}
        score = len(text_words & cat_words)
        if score > best_score:
            best_score = score
            best_cat = cat
    if best_cat not in CATEGORIES:
        return "agent_orchestration"
    return cast(Category, best_cat)


def scan_claude_mcp_list() -> list[CapabilityCard]:
    """Parse `claude mcp list` output across legacy and modern formats.

    Modern format:  ``<name>: <command-or-url> - <status>``
    Legacy format:  ``<name> (mcp): <description>``

    Cards are emitted with active=True when status is 'Connected', False
    when 'Needs authentication', and True otherwise (best effort).
    """
    try:
        out = _run_command(["claude", "mcp", "list"])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    cards: list[CapabilityCard] = []
    for line in out.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("Checking"):
            continue

        legacy = _MCP_LINE_LEGACY.match(stripped)
        if legacy:
            name = legacy.group("name")
            desc = legacy.group("desc").strip()
            active = True
        else:
            modern = _MCP_LINE_MODERN.match(stripped)
            if not modern:
                continue
            name = modern.group("name").strip()
            rest = modern.group("rest").strip()
            connected = bool(_MCP_STATUS_CONNECTED.search(rest))
            needs_auth = bool(_MCP_STATUS_NEEDS_AUTH.search(rest))
            active = connected
            if needs_auth:
                active = False
            desc = rest.split(" - ")[0].strip()

        triggers = [name.split(":")[-1].strip(), name]
        cards.append(
            CapabilityCard(
                id=_safe_id(name),
                name=name,
                category=_guess_category(name, desc),
                action_verb=desc[:80],
                triggers=triggers,
                description=desc,
                params_required=[],
                params_optional=[],
                example=f"MCP: {name}",
                schema_version="discovered",
                source="mcp",
                active=active,
            )
        )
    return cards


def scan_plugins(root: Path) -> list[CapabilityCard]:
    """Discover Claude Code plugins from the installed_plugins.json manifest.

    Newer Claude Code versions write a structured manifest at
    ``~/.claude/plugins/installed_plugins.json`` listing every installed
    plugin keyed by ``name@marketplace``. We prefer that manifest when it
    exists. As a fallback, we scan for legacy plugin.json files directly
    inside ``~/.claude/plugins/<name>/``.
    """
    cards: list[CapabilityCard] = []
    if not root.exists():
        return cards

    manifest = root / "installed_plugins.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        plugins = data.get("plugins", {})
        for full_name, entries in plugins.items():
            simple_name = full_name.split("@", 1)[0]
            real_description, real_triggers = _read_plugin_manifest(entries, simple_name)
            description = real_description or f"Installed Claude Code plugin '{full_name}'."
            triggers = list(dict.fromkeys([simple_name, full_name, *real_triggers]))
            action_verb = (
                description.split(".", 1)[0][:80]
                if real_description
                else f"Claude Code plugin: {simple_name}"
            )
            cards.append(
                CapabilityCard(
                    id=_safe_id(f"plugin_{simple_name}"),
                    name=simple_name,
                    category=_guess_category(simple_name, description),
                    action_verb=action_verb,
                    triggers=triggers,
                    description=description,
                    params_required=[],
                    params_optional=[],
                    example=f"Claude Code plugin: {simple_name}",
                    schema_version="discovered",
                    source="plugin",
                    active=True,
                )
            )
        return cards

    for plugin_json in root.glob("*/plugin.json"):
        try:
            data = json.loads(plugin_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        name = data.get("name") or plugin_json.parent.name
        desc = str(data.get("description") or "Claude Code plugin")
        cards.append(
            CapabilityCard(
                id=_safe_id(f"plugin_{name}"),
                name=name,
                category=_guess_category(name, desc),
                action_verb=desc[:80],
                triggers=[name],
                description=desc,
                params_required=[],
                params_optional=[],
                example=f"Claude Code plugin: {name}",
                schema_version="discovered",
                source="plugin",
                active=True,
            )
        )
    return cards


def scan_commands(root: Path) -> list[CapabilityCard]:
    cards: list[CapabilityCard] = []
    if not root.exists():
        return cards
    for cmd_md in root.glob("*.md"):
        name = cmd_md.stem
        body = cmd_md.read_text(encoding="utf-8").strip().splitlines()
        first_line = next((line for line in body if line and not line.startswith("#")), name)
        cards.append(
            CapabilityCard(
                id=_safe_id(f"command_{name}"),
                name=f"/{name}",
                category=_guess_category(name, first_line),
                action_verb=first_line[:80],
                triggers=[name, f"/{name}"],
                description=first_line,
                params_required=[],
                params_optional=[],
                example=f"Slash command: /{name}",
                schema_version="discovered",
                source="command",
                active=True,
            )
        )
    return cards


def scan_all(
    plugins_root: Path | None = None,
    commands_root: Path | None = None,
) -> list[CapabilityCard]:
    plugins_root = plugins_root or Path.home() / ".claude" / "plugins"
    commands_root = commands_root or Path.home() / ".claude" / "commands"
    out: list[CapabilityCard] = []
    out.extend(scan_claude_mcp_list())
    out.extend(scan_plugins(plugins_root))
    out.extend(scan_commands(commands_root))
    return out
