"""Tavily web search through its hosted MCP server."""

import json

from src.core.config import Settings
from src.mcp.client import MCPToolClient, MCPToolError

TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/"

NEWS_KEYWORDS = ("news", "latest", "today", "market", "stocks", "headline", "current")


class TavilySearch:
    def __init__(self, client: MCPToolClient) -> None:
        self._client = client

    async def web_search(
        self, query: str, max_results: int = 5, topic: str | None = None
    ) -> str:
        """Search the web. Returns the results as text. Raises MCPToolError on failure."""
        args: dict = {
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
        }
        if topic:
            args["topic"] = topic
        elif any(k in query.lower() for k in NEWS_KEYWORDS):
            args["topic"] = "news"
        else:
            args["topic"] = "general"

        text = await self._client.call_tool("tavily_search", args)
        self._raise_if_error_payload(text)
        return text

    @staticmethod
    def _raise_if_error_payload(text: str) -> None:
        # Tavily reports some failures (e.g. bad API key) as a normal reply
        # shaped like {"error": "...", "status": 401}, not as an MCP error.
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict) and "error" in payload and "results" not in payload:
            raise MCPToolError(f"Tavily error (status {payload.get('status', 'unknown')})")


def create_tavily_search(settings: Settings) -> TavilySearch:
    client = MCPToolClient(
        TAVILY_MCP_URL,
        headers={"Authorization": f"Bearer {settings.tavily_api_key.get_secret_value()}"},
    )
    return TavilySearch(client)
