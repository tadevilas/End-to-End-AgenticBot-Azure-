"""Generic MCP client. Connects to any remote MCP server over streamable HTTP.

Client side only - the MCP servers are run by someone else (e.g. Tavily).
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class MCPToolError(Exception):
    """Raised when an MCP tool call fails after all retries."""


class MCPToolClient:
    def __init__(self, url: str, headers: dict[str, str] | None = None, timeout: float = 30.0) -> None:
        self._url = url
        self._headers = headers or {}
        self._timeout = timeout

    async def list_tools(self) -> list[str]:
        async with self._session() as session:
            result = await session.list_tools()
            return [tool.name for tool in result.tools]

    @retry(
        retry=retry_if_exception_type((httpx2.TransportError, TimeoutError, OSError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _call_tool_with_retry(self, name: str, arguments: dict[str, Any]) -> str:
        async with self._session() as session:
            result = await session.call_tool(name, arguments)
        text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
        if result.is_error:
            raise MCPToolError(text or f"Tool '{name}' returned an error")
        return text

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool and return its text output. Raises MCPToolError on failure."""
        try:
            return await self._call_tool_with_retry(name, arguments)
        except MCPToolError:
            raise
        except Exception as exc:
            logger.warning("MCP tool '%s' failed: %s", name, type(exc).__name__)
            raise MCPToolError(f"Tool '{name}' failed: {type(exc).__name__}") from exc

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[ClientSession]:
        """Open an HTTP connection + MCP session, and close both on exit."""
        async with httpx2.AsyncClient(headers=self._headers, timeout=self._timeout) as http:
            async with streamable_http_client(self._url, http_client=http) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    yield session