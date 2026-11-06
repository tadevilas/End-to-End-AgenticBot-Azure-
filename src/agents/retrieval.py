"""Retrieval agent: fetches relevant chunks from Azure AI Search."""

import logging

from langchain_core.retrievers import BaseRetriever
from tenacity import retry, stop_after_attempt, wait_exponential

from src.graph.state import ChatState

logger = logging.getLogger(__name__)


class RetrievalAgent:
    def __init__(self, retriever: BaseRetriever) -> None:
        self._retriever = retriever

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4), reraise=True)
    async def _search(self, question: str):
        return await self._retriever.ainvoke(question)

    async def run(self, state: ChatState) -> dict:
        try:
            return {"doc_chunks": await self._search(state["question"])}
        except Exception as exc:
            logger.warning("Document search failed: %s", type(exc).__name__)
            return {"doc_chunks": [], "errors": ["Document search is unavailable right now."]}
