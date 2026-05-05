"""Validate that each capability card describes something actually installed.

A capability card is *theoretical* by default until something on the local
machine confirms it. The retrieval layer must never inject theoretical cards
into a prompt — that misleads the agent into believing it has tools it does
not have. This module audits the registry and updates the ``active`` flag
on each card based on:

  - source=mcp     → present in ``claude mcp list``?
  - source=plugin  → directory under ``~/.claude/plugins/<name>/``?
  - source=command → file at ``~/.claude/commands/<id>.md``?
  - source=os      → executable in PATH (heuristic: first word of example)?
  - source=skill | project | service | manual → trust (active=True)

The audit is conservative: if it cannot prove a card is active, it does NOT
flip it inactive. The flip only happens when a check is well-defined for the
source and the check fails.
"""
from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from mneme.schema import CapabilityCard


@dataclass
class VerifyResult:
    total: int
    activated: list[str]
    deactivated: list[str]
    skipped: list[tuple[str, str]]

    def render(self) -> str:
        lines = [f"verified {self.total} capability cards", "=" * 50, ""]
        if self.activated:
            lines.append(f"[active] {len(self.activated)} cards confirmed installed:")
            for cid in self.activated:
                lines.append(f"  + {cid}")
            lines.append("")
        if self.deactivated:
            lines.append(f"[inactive] {len(self.deactivated)} cards not detected on this machine:")
            for cid in self.deactivated:
                lines.append(f"  - {cid}")
            lines.append("")
        if self.skipped:
            lines.append(f"[skipped] {len(self.skipped)} cards (source not auto-verifiable):")
            for cid, src in self.skipped:
                lines.append(f"  ? {cid} (source={src})")
            lines.append("")
        return "\n".join(lines)


def _list_installed_mcps() -> set[str]:
    try:
        proc = subprocess.run(
            ["claude", "mcp", "list"], capture_output=True, text=True, timeout=10, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()
    names: set[str] = set()
    for line in proc.stdout.splitlines():
        token = line.strip().split()[:1]
        if not token:
            continue
        first = token[0].rstrip(":")
        if first.replace("_", "").replace("-", "").isalnum():
            names.add(first.lower())
    return names


def _plugin_present(card_id: str, plugins_root: Path) -> bool:
    if not plugins_root.exists():
        return False
    expected = card_id.removeprefix("plugin_")
    return (plugins_root / expected).exists() or (plugins_root / card_id).exists()


def _command_present(card_id: str, commands_root: Path) -> bool:
    if not commands_root.exists():
        return False
    name = card_id.removeprefix("command_")
    return (commands_root / f"{name}.md").exists() or (commands_root / f"{card_id}.md").exists()


def _os_executable_present(card: CapabilityCard) -> bool | None:
    """Heuristic: if the example starts with a known shell command, check PATH.

    Returns True/False when the heuristic applies, None when it does not.
    """
    example = card.example.strip().lower()
    for prefix in ("call: ", "run: ", "$ ", "> "):
        if example.startswith(prefix):
            example = example[len(prefix):].strip()
            break
    first_word = example.split(".", 1)[0].split("(", 1)[0].strip()
    if not first_word or len(first_word) < 2:
        return None
    binary_aliases = {
        "playwright": "playwright",
        "pyautogui": "python",
        "filesystem": None,
        "ollama": "ollama",
        "compose": "docker",
        "pg": "psql",
        "vault": None,
        "gh": "gh",
        "telegram": None,
    }
    binary = binary_aliases.get(first_word, first_word)
    if binary is None:
        return None
    return shutil.which(binary) is not None


def verify_cards(
    cards: Iterable[CapabilityCard],
    plugins_root: Path | None = None,
    commands_root: Path | None = None,
) -> tuple[list[CapabilityCard], VerifyResult]:
    """Return cards with `active` updated and a summary of the audit."""
    plugins_root = plugins_root or Path.home() / ".claude" / "plugins"
    commands_root = commands_root or Path.home() / ".claude" / "commands"
    installed_mcps = _list_installed_mcps()

    out: list[CapabilityCard] = []
    activated: list[str] = []
    deactivated: list[str] = []
    skipped: list[tuple[str, str]] = []

    for card in cards:
        verdict: bool | None
        if card.source == "mcp":
            verdict = card.id.lower() in installed_mcps or any(
                trigger.lower() in installed_mcps for trigger in card.triggers
            )
        elif card.source == "plugin":
            verdict = _plugin_present(card.id, plugins_root)
        elif card.source == "command":
            verdict = _command_present(card.id, commands_root)
        elif card.source == "os":
            verdict = _os_executable_present(card)
        else:
            verdict = None

        if verdict is True:
            new = card.model_copy(update={"active": True})
            activated.append(card.id)
            out.append(new)
        elif verdict is False:
            new = card.model_copy(update={"active": False})
            deactivated.append(card.id)
            out.append(new)
        else:
            new = card.model_copy(update={"active": True})
            skipped.append((card.id, card.source))
            out.append(new)

    total = len(out)
    return out, VerifyResult(
        total=total,
        activated=activated,
        deactivated=deactivated,
        skipped=skipped,
    )
