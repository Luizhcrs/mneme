"""Discovery: scan installed MCPs, plugins, slash commands into capability cards."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from mneme.schema import CapabilityCard


def _run_command(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    return proc.stdout


_MCP_LINE = re.compile(r"^(?P<name>[a-z][a-z0-9_-]*)\s*\(mcp\):\s*(?P<desc>.+)$")


def _safe_id(s: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", s.lower()).strip("_")


def scan_claude_mcp_list() -> list[CapabilityCard]:
    try:
        out = _run_command(["claude", "mcp", "list"])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    cards: list[CapabilityCard] = []
    for line in out.splitlines():
        m = _MCP_LINE.match(line.strip())
        if not m:
            continue
        name = m.group("name")
        desc = m.group("desc").strip()
        cards.append(
            CapabilityCard(
                id=_safe_id(name),
                name=name,
                category="agent_orchestration",
                action_verb=desc[:80],
                triggers=[name],
                description=desc,
                params_required=[],
                params_optional=[],
                example=f"MCP: {name}",
                schema_version="discovered",
                source="mcp",
            )
        )
    return cards


def scan_plugins(root: Path) -> list[CapabilityCard]:
    cards: list[CapabilityCard] = []
    if not root.exists():
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
                category="agent_orchestration",
                action_verb=desc[:80],
                triggers=[name],
                description=desc,
                params_required=[],
                params_optional=[],
                example=f"Claude Code plugin: {name}",
                schema_version="discovered",
                source="plugin",
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
                category="agent_orchestration",
                action_verb=first_line[:80],
                triggers=[name, f"/{name}"],
                description=first_line,
                params_required=[],
                params_optional=[],
                example=f"Slash command: /{name}",
                schema_version="discovered",
                source="command",
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
