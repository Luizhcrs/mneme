"""Tests for capability discovery scanner."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from mneme.scanner import scan_all, scan_claude_mcp_list


def test_scan_claude_mcp_list_parses_output() -> None:
    fake_stdout = "playwright (mcp): Browser automation\ntelegram (mcp): Messaging\n"
    with patch("mneme.scanner._run_command", return_value=fake_stdout):
        ids = [c.id for c in scan_claude_mcp_list()]
    assert "playwright" in ids
    assert "telegram" in ids


def test_scan_claude_mcp_list_handles_missing_cli() -> None:
    with patch("mneme.scanner._run_command", side_effect=FileNotFoundError):
        results = scan_claude_mcp_list()
    assert results == []


def test_scan_all_combines_sources(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "my_plugin").mkdir()
    (plugins_dir / "my_plugin" / "plugin.json").write_text(
        json.dumps({"name": "my_plugin", "description": "a plugin"}),
        encoding="utf-8",
    )

    commands_dir = tmp_path / "commands"
    commands_dir.mkdir()
    (commands_dir / "deploy.md").write_text("# Deploy\nDeploys the app.", encoding="utf-8")

    with patch("mneme.scanner._run_command", return_value=""):
        results = scan_all(plugins_root=plugins_dir, commands_root=commands_dir)

    ids = {c.id for c in results}
    assert "plugin_my_plugin" in ids
    assert "command_deploy" in ids
