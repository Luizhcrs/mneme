"""MCP stdio server — exposes mneme retrieval to any MCP-compatible client.

Hooks (UserPromptSubmit, PostToolUse) only work in Claude Code. To reach the
rest of the agent ecosystem (Codex, Cursor, Continue.dev, Cline, custom
agents) we expose the same retrieval surface as an MCP server. The agent
chooses when to call ``recall`` instead of mneme injecting unconditionally —
reactive instead of proactive — but the registry, hybrid scoring, feedback
boost, reflections, and procedural workflows are all the same.

Run via:
    mneme serve

Wire into a client by adding to its MCP config (Claude Code example):
    {
      "mcpServers": {
        "mneme": { "command": "mneme", "args": ["serve"] }
      }
    }
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from mneme import paths
from mneme.embedder import OllamaEmbedder
from mneme.feedback import FeedbackStore
from mneme.retrieve import Retriever
from mneme.schema import Workflow
from mneme.store import JsonlStore, SqliteStore

logger = logging.getLogger("mneme.mcp")

server: Server[None] = Server("mneme")


def _open_retriever() -> tuple[SqliteStore, Retriever] | None:
    db = paths.semantic_db()
    if not db.exists():
        return None
    try:
        store = SqliteStore(db)
        embedder = OllamaEmbedder()
        wf_path = paths.procedural_jsonl()
        wf_store = (
            JsonlStore[Workflow](wf_path, Workflow) if wf_path.exists() else None
        )
        fb_path = paths.feedback_jsonl()
        fb_store = FeedbackStore(fb_path) if fb_path.exists() else None
        retriever = Retriever(
            store, embedder, workflow_store=wf_store, feedback_store=fb_store
        )
    except (OSError, RuntimeError):
        return None
    return store, retriever


@server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="recall",
            description=(
                "Retrieve the most relevant capability cards for a task description. "
                "Use this BEFORE responding that you cannot do something — if the "
                "registry contains a tool that matches the user's intent, you have "
                "that tool and should attempt it before falling back to manual "
                "workarounds (curl, regex, screenshots-by-hand, etc.). "
                "Returns up to k cards with id, name, category, score, action_verb, "
                "params, example, and any matching past workflows or failure reflections."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's task or intent in natural language.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "How many capabilities to return (default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="record_correction",
            description=(
                "Record a correction: 'for this query, the right tool was that id'. "
                "Future similar queries will surface that tool at the top of the "
                "retrieval list. Use after a miss is observed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query that was wrongly resolved.",
                    },
                    "tool_id": {
                        "type": "string",
                        "description": "The capability id that should have been retrieved.",
                    },
                },
                "required": ["query", "tool_id"],
            },
        ),
        Tool(
            name="list_capabilities",
            description=(
                "List capability ids in the registry, optionally filtered by category. "
                "Useful for debugging which tools mneme knows about."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional category filter (e.g. web_browser, vcs).",
                    }
                },
            },
        ),
    ]


@server.call_tool()  # type: ignore[untyped-decorator]
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "recall":
        return await _recall(arguments)
    if name == "record_correction":
        return await _record_correction(arguments)
    if name == "list_capabilities":
        return await _list_capabilities(arguments)
    return [TextContent(type="text", text=json.dumps({"error": f"unknown tool {name}"}))]


async def _recall(arguments: dict[str, Any]) -> list[TextContent]:
    query = str(arguments.get("query", "")).strip()
    k = int(arguments.get("k", 5))
    if not query:
        return [
            TextContent(
                type="text",
                text=json.dumps({"capabilities": [], "note": "empty query"}),
            )
        ]
    opened = _open_retriever()
    if opened is None:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "capabilities": [],
                        "note": "mneme registry empty; run `mneme init && mneme reindex` first",
                    }
                ),
            )
        ]
    store, retriever = opened
    retriever._top_capabilities = max(1, k)
    try:
        result = retriever.retrieve(query)
    finally:
        store.close()
    payload = {
        "capabilities": [
            {
                "id": card.id,
                "name": card.name,
                "category": card.category,
                "score": round(score, 3),
                "action_verb": card.action_verb,
                "params_required": card.params_required,
                "params_optional": card.params_optional,
                "example": card.example.strip(),
            }
            for card, score in result.capabilities
        ],
        "categories_top": result.categories,
        "workflows": result.workflows,
        "reflections": result.reflections,
    }
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]


async def _record_correction(arguments: dict[str, Any]) -> list[TextContent]:
    query = str(arguments.get("query", "")).strip()
    tool_id = str(arguments.get("tool_id", "")).strip()
    if not query or not tool_id:
        err = json.dumps({"ok": False, "error": "query and tool_id required"})
        return [TextContent(type="text", text=err)]

    db = paths.semantic_db()
    if not db.exists():
        err = json.dumps({"ok": False, "error": "registry not initialized"})
        return [TextContent(type="text", text=err)]
    store = SqliteStore(db)
    if store.get_capability(tool_id) is None:
        store.close()
        err = json.dumps({"ok": False, "error": f"unknown tool_id {tool_id}"})
        return [TextContent(type="text", text=err)]
    store.close()

    embedder = OllamaEmbedder()
    vec = embedder.embed(query)
    fb = FeedbackStore(paths.feedback_jsonl())
    fb.append(query=query, tool_id=tool_id, embedding=vec)
    return [TextContent(type="text", text=json.dumps({"ok": True, "recorded": tool_id}))]


async def _list_capabilities(arguments: dict[str, Any]) -> list[TextContent]:
    import sqlite3

    category = arguments.get("category")
    db = paths.semantic_db()
    if not db.exists():
        empty = json.dumps({"capabilities": [], "note": "registry empty"})
        return [TextContent(type="text", text=empty)]
    conn = sqlite3.connect(db)
    if category:
        rows = conn.execute(
            "SELECT id, category FROM capabilities WHERE category = ? AND active = 1 ORDER BY id",
            (category,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, category FROM capabilities WHERE active = 1 ORDER BY id"
        ).fetchall()
    conn.close()
    return [
        TextContent(
            type="text",
            text=json.dumps([{"id": cid, "category": cat} for cid, cat in rows]),
        )
    ]


async def _run() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    asyncio.run(_run())
