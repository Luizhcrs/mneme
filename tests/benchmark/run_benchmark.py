"""Benchmark runner: measure affordance recall on the 50-task dataset."""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

from mneme.embedder import EMBED_DIM
from mneme.loader import seed_store
from mneme.retrieve import Retriever
from mneme.store import SqliteStore


def _stable_seed(text: str) -> int:
    digest = hashlib.md5(text.lower().encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


@dataclass
class BenchmarkResult:
    total: int
    top1: int
    top3: int
    top5: int
    avg_latency_ms: float

    @property
    def top1_rate(self) -> float:
        return self.top1 / self.total if self.total else 0.0

    @property
    def top3_rate(self) -> float:
        return self.top3 / self.total if self.total else 0.0

    @property
    def top5_rate(self) -> float:
        return self.top5 / self.total if self.total else 0.0


class _DeterministicEmbedder:
    """Hash-based fake used when no real Ollama is available."""

    def __init__(self) -> None:
        self._cache: dict[str, NDArray[np.float32]] = {}

    def embed(self, text: str) -> NDArray[np.float32]:
        key = text.lower()
        if key not in self._cache:
            rng = np.random.default_rng(_stable_seed(key))
            v = rng.standard_normal(EMBED_DIM).astype(np.float32)
            self._cache[key] = v / np.linalg.norm(v)
        return self._cache[key]


def run(
    tasks_path: Path,
    seed_yaml: Path,
    use_real_ollama: bool = False,
) -> BenchmarkResult:
    tasks = yaml.safe_load(tasks_path.read_text(encoding="utf-8"))
    db = tasks_path.parent / "_bench.sqlite"
    if db.exists():
        db.unlink()

    if use_real_ollama:
        from mneme.embedder import OllamaEmbedder

        embedder: object = OllamaEmbedder()
    else:
        embedder = _DeterministicEmbedder()

    store = SqliteStore(db)
    seed_store(seed_yaml, store, embedder=embedder)  # type: ignore[arg-type]
    retriever = Retriever(store, embedder, threshold=0.0, top_capabilities=5)  # type: ignore[arg-type]

    top1 = top3 = top5 = 0
    latencies: list[float] = []
    for task in tasks:
        prompt = task["prompt"]
        expected = task["expected_capability"]
        t0 = time.perf_counter()
        result = retriever.retrieve(prompt)
        latencies.append((time.perf_counter() - t0) * 1000)
        ids = [card.id for card, _ in result.capabilities]
        if ids and ids[0] == expected:
            top1 += 1
        if expected in ids[:3]:
            top3 += 1
        if expected in ids[:5]:
            top5 += 1

    store.close()
    return BenchmarkResult(
        total=len(tasks),
        top1=top1,
        top3=top3,
        top5=top5,
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
    )


def _main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-ollama", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).parent
    seed = here.parent.parent / "src" / "mneme" / "seed" / "capabilities.example.yaml"
    use_real = args.real_ollama or os.environ.get("MNEME_REAL_OLLAMA") == "1"
    result = run(here / "tasks.yaml", seed, use_real_ollama=use_real)
    print(
        f"top1: {result.top1_rate:.2%}  top3: {result.top3_rate:.2%}  "
        f"top5: {result.top5_rate:.2%}\n"
        f"avg_latency_ms: {result.avg_latency_ms:.2f}"
    )


if __name__ == "__main__":  # pragma: no cover
    _main()
