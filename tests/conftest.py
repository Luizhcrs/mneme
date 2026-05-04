"""Shared pytest fixtures."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def tmp_mneme_dir(tmp_path: Path) -> Iterator[Path]:
    """Provide an isolated temp directory acting as ~/.claude/mneme for a test."""
    target = tmp_path / "mneme"
    target.mkdir()
    yield target
