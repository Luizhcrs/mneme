"""Acceptance gates for the 50-task benchmark.

Phase 1 minimum (RAG-MCP, arXiv:2505.03275): top-3 affordance recall >= 43%.
Latency target: avg < 100 ms with the deterministic fake embedder.

The deterministic fake stands in for the real instruction-tuned Ollama model
on CI runs that do not have an Ollama daemon. With random unit vectors the
fake produces near-uniform retrieval (~12% top-3 across 10 capabilities) and
is therefore unsuitable for the recall gate. Recall is checked only when
MNEME_REAL_OLLAMA=1 is set; otherwise the test is skipped (not failed) so
unrelated CI runs stay green while still flagging the gate explicitly.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.benchmark.run_benchmark import run

ROOT = Path(__file__).parent.parent
SEED = ROOT / "src" / "mneme" / "seed" / "capabilities.example.yaml"
TASKS = ROOT / "tests" / "benchmark" / "tasks.yaml"
USE_REAL = os.environ.get("MNEME_REAL_OLLAMA") == "1"


def test_dataset_has_50_tasks() -> None:
    result = run(TASKS, SEED, use_real_ollama=False)
    assert result.total == 50


def test_avg_latency_under_target_with_fake_embedder() -> None:
    result = run(TASKS, SEED, use_real_ollama=False)
    assert result.avg_latency_ms < 100, (
        f"avg latency {result.avg_latency_ms:.2f}ms exceeds <100ms target"
    )


@pytest.mark.skipif(
    not USE_REAL,
    reason="recall gate requires MNEME_REAL_OLLAMA=1 with a running Ollama daemon",
)
def test_top3_recall_meets_phase1_minimum() -> None:
    result = run(TASKS, SEED, use_real_ollama=True)
    assert result.top3_rate >= 0.43, (
        f"top3 recall {result.top3_rate:.2%} < 43% Phase 1 minimum "
        f"(RAG-MCP arxiv:2505.03275)"
    )
