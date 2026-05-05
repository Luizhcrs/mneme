"""Tests for the mneme CLI."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from mneme.cli import app
from mneme.embedder import EMBED_DIM


@pytest.fixture
def cli_env(tmp_mneme_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MNEME_HOME", str(tmp_mneme_dir))
    return tmp_mneme_dir


@pytest.fixture
def mock_ollama() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url="http://localhost:11434", assert_all_called=False) as r:
        r.post("/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": [0.1] * EMBED_DIM})
        )
        yield r


def test_init_creates_directory_and_seed(cli_env: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert (cli_env / "capabilities.yaml").exists()


def test_init_does_not_overwrite(cli_env: Path) -> None:
    runner = CliRunner()
    runner.invoke(app, ["init"])
    yml = cli_env / "capabilities.yaml"
    yml.write_text("# user edited\n", encoding="utf-8")
    runner.invoke(app, ["init"])
    assert yml.read_text(encoding="utf-8") == "# user edited\n"


def test_reindex_requires_init(cli_env: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["reindex"])
    assert result.exit_code == 1
    assert "init" in result.output


def test_reindex_after_init(cli_env: Path, mock_ollama: respx.MockRouter) -> None:
    runner = CliRunner()
    runner.invoke(app, ["init"])
    result = runner.invoke(app, ["reindex"])
    assert result.exit_code == 0
    assert "indexed 10 capabilities" in result.output


def test_list_outputs_capability_ids(cli_env: Path, mock_ollama: respx.MockRouter) -> None:
    runner = CliRunner()
    runner.invoke(app, ["init"])
    runner.invoke(app, ["reindex"])
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "playwright_screenshot" in result.output


def test_list_filters_by_category(cli_env: Path, mock_ollama: respx.MockRouter) -> None:
    runner = CliRunner()
    runner.invoke(app, ["init"])
    runner.invoke(app, ["reindex"])
    result = runner.invoke(app, ["list", "--category", "comms"])
    assert result.exit_code == 0
    assert "telegram_send" in result.output
    assert "playwright_screenshot" not in result.output


def test_search_runs(cli_env: Path, mock_ollama: respx.MockRouter) -> None:
    runner = CliRunner()
    runner.invoke(app, ["init"])
    runner.invoke(app, ["reindex"])
    result = runner.invoke(app, ["search", "telegram message"])
    assert result.exit_code == 0


def test_stats_runs(cli_env: Path, mock_ollama: respx.MockRouter) -> None:
    runner = CliRunner()
    runner.invoke(app, ["init"])
    runner.invoke(app, ["reindex"])
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "capabilities: 10" in result.output
