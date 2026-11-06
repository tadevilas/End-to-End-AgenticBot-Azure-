"""What the planner decides, and what a citation looks like."""

from typing import Literal

from pydantic import BaseModel


class Plan(BaseModel):
    use_documents: bool = True
    use_web: bool = False
    use_sql: bool = False
    sql_query: str | None = None


class Citation(BaseModel):
    index: int
    title: str
    source: str  # file name for documents, URL for web results, table/query for SQL
    kind: Literal["document", "web", "sql"]
    snippet: str = ""
