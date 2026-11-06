"""Hybrid retriever (BM25 + vector + semantic re-ranking) as a LangChain BaseRetriever."""

from typing import Any

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType, VectorizedQuery
from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from src.rag.index import SEMANTIC_CONFIG


class HybridRetriever(BaseRetriever):
    search_client: Any  # azure.search.documents.aio.SearchClient
    embeddings: Embeddings
    top_k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        raise NotImplementedError("HybridRetriever is async-only; use ainvoke().")

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        client: SearchClient = self.search_client
        vector = await self.embeddings.aembed_query(query)
        results = await client.search(
            search_text=query,  # BM25 keyword search
            vector_queries=[
                VectorizedQuery(vector=vector, k_nearest_neighbors=self.top_k * 4, fields="content_vector")
            ],  # vector search
            query_type=QueryType.SEMANTIC,  # semantic re-ranking
            semantic_configuration_name=SEMANTIC_CONFIG,
            select=["content", "title", "source", "page"],
            top=self.top_k,
        )
        documents: list[Document] = []
        async for r in results:
            documents.append(
                Document(
                    page_content=r["content"],
                    metadata={
                        "title": r.get("title", ""),
                        "source": r.get("source", ""),
                        "page": r.get("page"),
                        "score": r.get("@search.reranker_score") or r["@search.score"],
                    },
                )
            )
        return documents
