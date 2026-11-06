"""Local text embeddings (fastembed) exposed as a LangChain Embeddings object."""

import asyncio

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings


class LocalEmbedder(Embeddings):
    def __init__(self, model_name: str, dimension: int) -> None:
        self._model = TextEmbedding(model_name=model_name)
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    # fastembed is CPU-bound and blocking, so run it off the event loop.
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)
