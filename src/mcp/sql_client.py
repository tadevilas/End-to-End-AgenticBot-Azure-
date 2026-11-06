"""SQL MCP Client for structured enterprise database queries (Databricks / Postgres / ANSI SQL).

Connects via the Model Context Protocol (MCP) to an external SQL MCP server.
Client side only — does not run an internal database server.
"""

import logging
import os
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.core.config import Settings
from src.mcp.client import MCPToolError

logger = logging.getLogger(__name__)


class SQLMCPClient:
    """Async MCP Client connecting to external SQL / Databricks MCP server."""

    def __init__(self, command: str = "uvx", args: list[str] | None = None, env: dict[str, str] | None = None) -> None:
        self._server_params = StdioServerParameters(
            command=command,
            args=args or ["mcp-server-sqlite", "--db-path", ":memory:"],
            env=env or {},
        )

    @classmethod
    def from_databricks_settings(cls, settings: Settings) -> "SQLMCPClient":
        """Instantiate client preconfigured for Azure Databricks SQL MCP Server."""
        env = {
            "DATABRICKS_HOST": settings.databricks_server_hostname,
            "DATABRICKS_HTTP_PATH": settings.databricks_http_path,
            "DATABRICKS_TOKEN": settings.databricks_token.get_secret_value() if settings.databricks_token else "",
        }
        return cls(command="uvx", args=["mcp-server-databricks"], env=env)

    @retry(
        retry=retry_if_exception_type((TimeoutError, OSError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        reraise=True,
    )
    async def execute_query(self, query: str) -> str:
        """Executes SQL query against external database warehouse through MCP."""
        try:
            async with stdio_client(self._server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("execute_sql", arguments={"query": query})
                    text = "\n".join(
                        c.text for c in result.content if getattr(c, "type", "") == "text"
                    )
                    if result.is_error:
                        raise MCPToolError(text or "SQL tool execution failed")
                    return text
        except MCPToolError:
            raise
        except Exception as exc:
            logger.warning("SQL MCP tool failed: %s", type(exc).__name__)
            raise MCPToolError(f"SQL execution error: {type(exc).__name__}") from exc
