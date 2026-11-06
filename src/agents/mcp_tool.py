"""MCP tool agent: web search through the Tavily MCP client."""

import json
import logging

from src.graph.state import ChatState
from src.mcp.client import MCPToolError
from src.mcp.tavily import TavilySearch

logger = logging.getLogger(__name__)

MAX_CONTENT_CHARS = 1200


def parse_web_results(text: str) -> list[dict]:
    try:
        results = json.loads(text).get("results", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    return [
        {
            "title": r.get("title", "") or r.get("url", ""),
            "url": r.get("url", ""),
            "content": (r.get("content") or "")[:MAX_CONTENT_CHARS],
        }
        for r in results
        if r.get("url") and (r.get("content") or r.get("title"))
    ]


class MCPToolAgent:
    def __init__(self, tavily: TavilySearch) -> None:
        self._tavily = tavily

    async def run(self, state: ChatState) -> dict:
        try:
            text = await self._tavily.web_search(state["question"])
        except Exception as exc:
            logger.warning("Web search failed: %s: %s", type(exc).__name__, exc)
            return {"web_results": [], "errors": ["Web search is unavailable right now."]}
        return {"web_results": parse_web_results(text)}
