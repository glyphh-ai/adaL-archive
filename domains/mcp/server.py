"""
MCP Server — Ada's brain exposed as three tools.

  think(input)    — pure associative recall, returns what activates
  ask(question)   — targeted retrieval, returns one fact + confidence
  write(text)     — absorb a fact into the substrate (optional versioning)

The original single `think` tool is preserved as the catch-all entry that
runs Ada's full cognitive loop (firewall + classify + route + absorb).
The new `think` here is the surface variant: pure associative wave, no
synthesis. Either is fine to call; they target different cognitive needs.
"""

import json
import logging
from typing import Any

from mcp.server.lowlevel.server import Server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)

from domains.auth.service import AuthService

logger = logging.getLogger(__name__)


def create_mcp_server(brain: Any, auth_service: AuthService) -> Server:
    """Create the MCP server with three cognitive tools.

    Args:
        brain: The Brain instance (domains.brain.think.Brain)
        auth_service: AuthService for token validation
    """
    app = Server("ada")

    @app.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="think",
                description=(
                    "Pure associative recall over Ada's substrate. Surfaces the "
                    "constellation of thoughts that resonate with the input — "
                    "no answer, no refusal, no synthesis. Call this when you "
                    "want CONTEXT to bring into a response."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"},
                        "top_k": {"type": "integer", "default": 8},
                    },
                    "required": ["input"],
                },
            ),
            Tool(
                name="ask",
                description=(
                    "Targeted retrieval — find me the fact that answers this "
                    "question. Returns a single fact + confidence, or refuses "
                    "with 'I don't know.' Call this when you NEED a specific "
                    "answer and would rather be told 'no' than guess."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                    },
                    "required": ["question"],
                },
            ),
            Tool(
                name="write",
                description=(
                    "Absorb a fact into Ada's substrate. Optional `key` makes "
                    "this a new version of that key (versioned chain, drift "
                    "queryable). Call this on every decision or observation "
                    "you'd want to recall later."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "key": {"type": "string"},
                    },
                    "required": ["text"],
                },
            ),
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> CallToolResult:
        args = arguments or {}

        if name == "think":
            return _handle_think(brain, args)
        if name == "ask":
            return _handle_ask(brain, args)
        if name == "write":
            return _handle_write(brain, args)
        return _err(f"Unknown tool: {name}")

    return app


# ── handlers ────────────────────────────────────────────────────────

def _handle_think(brain, args: dict) -> CallToolResult:
    input_text = args.get("input", "")
    top_k = int(args.get("top_k", 8))
    if not input_text:
        return _err("Missing 'input' parameter")
    try:
        surface = _get_surface(brain)
        act = surface.think(input_text, top_k=top_k)
        return _ok({
            "input": input_text,
            "activated": [
                {"content": t.content, "similarity": round(s, 3)}
                for t, s in zip(act.thoughts, act.similarities)
            ],
            "pathway": act.pathway,
            "pattern_boost": round(act.pattern_boost, 3),
        })
    except Exception as e:
        logger.error("think failed: %s", e, exc_info=True)
        return _err(str(e))


def _handle_ask(brain, args: dict) -> CallToolResult:
    question = args.get("question", "")
    if not question:
        return _err("Missing 'question' parameter")
    try:
        surface = _get_surface(brain)
        a = surface.ask(question)
        return _ok({
            "question": question,
            "refused": a.refused,
            "confidence": round(a.confidence, 3),
            "fact": a.fact.content if a.fact else None,
            "answer": a.rendered,
        })
    except Exception as e:
        logger.error("ask failed: %s", e, exc_info=True)
        return _err(str(e))


def _handle_write(brain, args: dict) -> CallToolResult:
    text = args.get("text", "")
    key = args.get("key")
    if not text:
        return _err("Missing 'text' parameter")
    try:
        space = brain._cognitive.thought_space
        stored = space.absorb(text, key=key)
        if stored is None:
            return _ok({"absorbed": False, "reason": "duplicate"})
        return _ok({
            "absorbed": True,
            "thought_id": stored.thought_id,
            "version": stored.metadata.get("_version"),
            "key": stored.metadata.get("_key"),
        })
    except Exception as e:
        logger.error("write failed: %s", e, exc_info=True)
        return _err(str(e))


# ── plumbing ────────────────────────────────────────────────────────

_surface_cache: dict = {}


def _get_surface(brain):
    """Build a CognitiveSurface over the Brain's existing thought-space."""
    if id(brain) in _surface_cache:
        return _surface_cache[id(brain)]
    from ada.cognitive.generate import build_llm_renderer
    from ada.cognitive.surface import CognitiveSurface
    space = brain._cognitive.thought_space
    cognitive = brain._cognitive._glyph_loop
    surface = CognitiveSurface(
        space=space,
        cognitive=cognitive,
        renderer=build_llm_renderer(),
    )
    _surface_cache[id(brain)] = surface
    return surface


def _ok(payload: dict) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
    )


def _err(msg: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps({"error": msg}))],
        isError=True,
    )
