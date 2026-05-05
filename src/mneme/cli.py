"""mneme CLI."""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import typer
import yaml

from mneme import insights as _insights_mod
from mneme import paths
from mneme.embedder import OllamaEmbedder
from mneme.feedback import FeedbackStore
from mneme.loader import load_capabilities, seed_store
from mneme.reflexion import consolidate as reflect_consolidate
from mneme.retrieve import Retriever
from mneme.scanner import scan_all
from mneme.store import SqliteStore
from mneme.verify import verify_cards

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
    fb_path = paths.feedback_jsonl()
    fb_store = FeedbackStore(fb_path) if fb_path.exists() else None
    retriever = Retriever(store, embedder, threshold=0.0, feedback_store=fb_store)
    result = retriever.retrieve(query)
    rendered = result.render()
    typer.echo(rendered if rendered else "<no match>")
    store.close()


@app.command()
def correct(query: str, tool_id: str) -> None:
    """Record a correction: 'for query X, the right tool was Y'.

    Future queries semantically similar to X will surface tool Y at the
    top of the retrieval list. This is the active feedback loop — mneme
    learns from your corrections without retraining or re-embedding the
    whole registry.
    """
    db = paths.semantic_db()
    if not db.exists():
        typer.echo("run `mneme reindex` first", err=True)
        raise typer.Exit(1)
    store = SqliteStore(db)
    if store.get_capability(tool_id) is None:
        typer.echo(f"unknown tool_id: {tool_id!r}. Run `mneme list` to see ids.", err=True)
        store.close()
        raise typer.Exit(1)
    store.close()

    embedder = OllamaEmbedder()
    vec = embedder.embed(query)
    fb_store = FeedbackStore(paths.feedback_jsonl())
    fb_store.append(query=query, tool_id=tool_id, embedding=vec)
    typer.echo(f"recorded: {query!r} -> {tool_id}")


@app.command()
def rescan(
    apply: bool = typer.Option(
        False, "--apply", help="Write the discovered cards into capabilities.yaml"
    ),
) -> None:
    """Discover MCPs, plugins, slash commands installed locally and surface them.

    Without ``--apply`` this is a dry run: prints the diff against the current
    capabilities.yaml so you can review before persisting. With ``--apply`` the
    new cards are merged into capabilities.yaml (existing card ids are NOT
    overwritten — manual edits win).
    """
    yml = paths.capabilities_yaml()
    if not yml.exists():
        typer.echo("run `mneme init` first", err=True)
        raise typer.Exit(1)

    existing = load_capabilities(yml)
    existing_ids = {c.id for c in existing}
    discovered = scan_all()

    new_cards = [c for c in discovered if c.id not in existing_ids]
    typer.echo(f"discovered {len(discovered)} cards, {len(new_cards)} new")
    if new_cards:
        typer.echo("")
        typer.echo("new cards:")
        for c in new_cards:
            flag = "active" if c.active else "inactive"
            typer.echo(f"  + [{c.id}] {c.name} ({c.source}, {c.category}, {flag})")
    skipped_ids = sorted(existing_ids & {c.id for c in discovered})
    if skipped_ids:
        typer.echo("")
        typer.echo(f"already in registry (left untouched): {len(skipped_ids)}")

    if not apply:
        typer.echo("")
        typer.echo("dry run. re-run with --apply to write capabilities.yaml.")
        return

    merged = list(existing) + new_cards
    yml.write_text(
        yaml.safe_dump(
            [c.model_dump(mode="json") for c in merged],
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    typer.echo("")
    typer.echo(f"wrote {len(merged)} cards to {yml}")
    typer.echo("run `mneme reindex` to apply to the search index.")


@app.command()
def verify() -> None:
    """Audit capability cards against the local machine.

    For each card, check if the thing it describes is actually installed
    here (MCP registered, plugin present, command file present, OS binary
    in PATH). Updates the ``active`` flag in capabilities.yaml. Cards that
    fail the check stop appearing in retrieval injection — preventing the
    agent from claiming tools it does not actually have.

    Sources that cannot be auto-verified (skill, project, service, manual)
    are left alone with ``active=True``.
    """
    yml = paths.capabilities_yaml()
    if not yml.exists():
        typer.echo("run `mneme init` first", err=True)
        raise typer.Exit(1)

    cards = load_capabilities(yml)
    updated, result = verify_cards(cards)
    yml.write_text(
        yaml.safe_dump(
            [c.model_dump(mode="json") for c in updated],
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    typer.echo(result.render())
    typer.echo("capabilities.yaml updated. Run `mneme reindex` to apply to the search index.")


@app.command()
def reflect() -> None:
    """Consolidate failures.log into reflections that surface on similar prompts.

    Walks every failure entry produced by the PostToolUse hook and writes
    a templated lesson into reflections.jsonl with the prompt embedding so
    the retrieval layer can match it against future queries. Idempotent —
    re-runs produce no duplicates.
    """
    embedder = OllamaEmbedder()
    written = reflect_consolidate(
        failure_log_path=paths.failure_log(),
        reflection_store_path=paths.reflections_jsonl(),
        embedder=embedder,
    )
    typer.echo(f"wrote {written} new reflections")


@app.command()
def insights(window: int = typer.Option(7, "--window", "-w", help="Days to aggregate")) -> None:
    """Surface patterns from local telemetry: top tools, dead cards, failures, latency."""
    result = _insights_mod.aggregate(window_days=window)
    typer.echo(result.render())


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
