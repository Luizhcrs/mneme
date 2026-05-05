"""mneme CLI."""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import typer

from mneme import paths
from mneme.embedder import OllamaEmbedder
from mneme.loader import seed_store
from mneme.retrieve import Retriever
from mneme.store import SqliteStore

app = typer.Typer(help="mneme - capability-recall layer for Claude Code", no_args_is_help=True)

SEED_SOURCE = Path(__file__).parent / "seed" / "capabilities.example.yaml"


@app.command()
def init() -> None:
    """Create ~/.claude/mneme directory and copy seed capabilities."""
    home = paths.home()
    home.mkdir(parents=True, exist_ok=True)
    target = paths.capabilities_yaml()
    if not target.exists():
        shutil.copy(SEED_SOURCE, target)
        typer.echo(f"created {target}")
    else:
        typer.echo(f"already exists: {target}")


@app.command()
def reindex() -> None:
    """Rebuild semantic.sqlite from capabilities.yaml using Ollama embeddings."""
    yml = paths.capabilities_yaml()
    if not yml.exists():
        typer.echo("run `mneme init` first", err=True)
        raise typer.Exit(1)
    db = paths.semantic_db()
    if db.exists():
        db.unlink()
    store = SqliteStore(db)
    n = seed_store(yml, store)
    store.close()
    typer.echo(f"indexed {n} capabilities")


@app.command(name="list")
def list_caps(category: str | None = typer.Option(None, "--category", "-c")) -> None:
    """List known capabilities, optionally filtered by category."""
    db = paths.semantic_db()
    if not db.exists():
        typer.echo("run `mneme reindex` first", err=True)
        raise typer.Exit(1)

    conn = sqlite3.connect(db)
    if category is not None:
        cur = conn.execute(
            "SELECT id, category FROM capabilities WHERE category = ? ORDER BY id",
            (category,),
        )
    else:
        cur = conn.execute("SELECT id, category FROM capabilities ORDER BY id")
    for cap_id, cat in cur.fetchall():
        typer.echo(f"{cap_id}\t{cat}")
    conn.close()


@app.command()
def search(query: str) -> None:
    """Show retrieval result for an ad-hoc query (debug)."""
    db = paths.semantic_db()
    if not db.exists():
        typer.echo("run `mneme reindex` first", err=True)
        raise typer.Exit(1)
    store = SqliteStore(db)
    embedder = OllamaEmbedder()
    retriever = Retriever(store, embedder, threshold=0.0)
    result = retriever.retrieve(query)
    rendered = result.render()
    typer.echo(rendered if rendered else "<no match>")
    store.close()


@app.command()
def stats() -> None:
    """Show counts of capabilities and procedural workflows."""
    db = paths.semantic_db()
    n_caps = 0
    if db.exists():
        conn = sqlite3.connect(db)
        n_caps = conn.execute("SELECT COUNT(*) FROM capabilities").fetchone()[0]
        conn.close()

    proc = paths.procedural_jsonl()
    n_workflows = 0
    if proc.exists():
        with proc.open("r", encoding="utf-8") as f:
            n_workflows = sum(1 for line in f if line.strip())

    typer.echo(f"home:         {paths.home()}")
    typer.echo(f"capabilities: {n_caps}")
    typer.echo(f"workflows:    {n_workflows}")
