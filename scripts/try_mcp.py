"""Quick manual check of the MCP client against Tavily's hosted MCP server.

Run:  uv run python -X utf8 -m scripts.try_mcp
"""

import asyncio
import warnings

from src.core.config import get_settings
from src.mcp.client import MCPToolClient, MCPToolError
from src.mcp.tavily import TAVILY_MCP_URL, TavilySearch, create_tavily_search

warnings.filterwarnings("ignore")


async def main() -> None:
    settings = get_settings()
    tavily = create_tavily_search(settings)

    print("1. tools   :", await tavily._client.list_tools())

    result = await tavily.web_search("What is Azure AI Search?", max_results=2)
    print("2. search  :", len(result), "chars returned")
    print("   preview :", result[:200].replace("\n", " "), "...")

    bad = TavilySearch(MCPToolClient(TAVILY_MCP_URL, headers={"Authorization": "Bearer tvly-wrong-key"}))
    try:
        await bad.web_search("test")
        print("3. bad key : unexpectedly worked")
    except MCPToolError as exc:
        print("3. bad key : clean error ->", exc)


if __name__ == "__main__":
    asyncio.run(main())
