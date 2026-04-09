"""
MCP Server — Ada's brain exposed as a single tool.

One tool: think. Natural language in, structured response out.
Ada routes internally, processes, and returns.
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
    """Create the MCP server with a single `think` tool.

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
                    "Send a natural language request to Ada's brain. "
                    "Ada routes internally to the right capability "
                    "(firewall, voice, memory, etc.) and returns a "
                    "structured response. Use this for ALL interactions."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Natural language input to process",
                        },
                    },
                    "required": ["input"],
                },
            ),
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> CallToolResult:
        if name != "think":
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"}),
                )],
                isError=True,
            )

        input_text = (arguments or {}).get("input", "")
        if not input_text:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({"error": "Missing 'input' parameter"}),
                )],
                isError=True,
            )

        try:
            result = await brain.think(input_text)

            response = {
                "response": result.response,
                "capability": result.capability,
                "confidence": round(result.confidence, 3),
                "cognitive_state": result.cognitive_state,
                "gate": result.gate,
                "llm_fallback": result.llm_fallback,
                "elapsed_ms": round(result.elapsed_ms, 1),
            }

            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(response, ensure_ascii=False),
                )],
            )

        except Exception as e:
            logger.error(f"think() failed: {e}", exc_info=True)
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)}),
                )],
                isError=True,
            )

    return app
