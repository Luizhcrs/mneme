"""Tests for the verify auditor."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mneme.schema import CapabilityCard
from mneme.verify import verify_cards


def _card(
    card_id: str,
    source: str,
    example: str = "x",
    triggers: list[str] | None = None,
) -> CapabilityCard:
    return CapabilityCard(
        id=card_id,
        name=card_id,
        category="filesystem",
        action_verb="x",
        triggers=triggers or [card_id],
        description="x",
        params_required=[],
        params_optional=[],
        example=example,
        schema_version="1",
        source=source,  # type: ignore[arg-type]
    )


def test_mcp_card_active_when_listed(tmp_path: Path) -> None:
    cards = [_card("playwright", "mcp", triggers=["playwright"])]
    with patch("mneme.verify.subprocess.run") as run:
        run.return_value.stdout = "playwright (mcp): browser automation\n"
        updated, result = verify_cards(
            cards,
            plugins_root=tmp_path / "p",
            commands_root=tmp_path / "c",
        )
    assert updated[0].active is True
    assert "playwright" in result.activated


def test_mcp_card_inactive_when_not_listed(tmp_path: Path) -> None:
    cards = [_card("telegram_send", "mcp", triggers=["telegram"])]
    with patch("mneme.verify.subprocess.run") as run:
        run.return_value.stdout = ""
        updated, result = verify_cards(
            cards,
            plugins_root=tmp_path / "p",
            commands_root=tmp_path / "c",
        )
    assert updated[0].active is False
    assert "telegram_send" in result.deactivated


def test_plugin_card_active_when_directory_present(tmp_path: Path) -> None:
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "my_plugin").mkdir()
    cards = [_card("plugin_my_plugin", "plugin")]
    with patch("mneme.verify.subprocess.run") as run:
        run.return_value.stdout = ""
        updated, result = verify_cards(cards, plugins_root=plugins, commands_root=tmp_path / "c")
    assert updated[0].active is True
    assert "plugin_my_plugin" in result.activated


def test_plugin_card_inactive_when_directory_missing(tmp_path: Path) -> None:
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    cards = [_card("plugin_absent", "plugin")]
    with patch("mneme.verify.subprocess.run") as run:
        run.return_value.stdout = ""
        updated, _ = verify_cards(cards, plugins_root=plugins, commands_root=tmp_path / "c")
    assert updated[0].active is False


def test_command_card_active_when_md_present(tmp_path: Path) -> None:
    commands = tmp_path / "commands"
    commands.mkdir()
    (commands / "deploy.md").write_text("Deploy stuff", encoding="utf-8")
    cards = [_card("command_deploy", "command")]
    with patch("mneme.verify.subprocess.run") as run:
        run.return_value.stdout = ""
        updated, _ = verify_cards(cards, plugins_root=tmp_path / "p", commands_root=commands)
    assert updated[0].active is True


def test_skill_card_left_active(tmp_path: Path) -> None:
    cards = [_card("user_skill", "skill")]
    with patch("mneme.verify.subprocess.run") as run:
        run.return_value.stdout = ""
        updated, result = verify_cards(
            cards,
            plugins_root=tmp_path / "p",
            commands_root=tmp_path / "c",
        )
    assert updated[0].active is True
    assert ("user_skill", "skill") in result.skipped


def test_manual_card_left_active(tmp_path: Path) -> None:
    cards = [_card("custom", "manual")]
    with patch("mneme.verify.subprocess.run") as run:
        run.return_value.stdout = ""
        updated, _ = verify_cards(
            cards,
            plugins_root=tmp_path / "p",
            commands_root=tmp_path / "c",
        )
    assert updated[0].active is True


def test_missing_claude_cli_does_not_crash(tmp_path: Path) -> None:
    cards = [_card("anything", "mcp")]
    with patch("mneme.verify.subprocess.run", side_effect=FileNotFoundError):
        updated, _ = verify_cards(
            cards,
            plugins_root=tmp_path / "p",
            commands_root=tmp_path / "c",
        )
    assert updated[0].active is False
