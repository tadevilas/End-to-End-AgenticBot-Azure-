"""The shared state that flows through the LangGraph workflow."""

import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from src.domain.plan import Citation, Plan


class ChatState(TypedDict, total=False):
    question: str
    history: list[BaseMessage]  # earlier messages of this conversation
    plan: Plan  # set by the planner
    doc_chunks: list[Document]  # set by retrieval
    web_results: list[dict]  # set by the MCP tool agent: title, url, content
    sql_results: list[dict]  # set by the MCP tool agent: title, query, result
    sources: list[Citation]  # numbered sources shown to the LLM
    answer: str  # set by response, validated by citation
    citations: list[Citation]  # sources the answer actually cites
    errors: Annotated[list[str], operator.add]  # failures collected from any node
