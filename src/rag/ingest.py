"""Splits LangChain Documents into chunks and uploads them to Azure AI Search."""

import hashlib

from azure.search.documents.aio import SearchClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document], chunk_size: int = 400, chunk_overlap: int = 50
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def _chunk_id(source: str, page: int, index: int) -> str:
    # Stable ID: re-ingesting the same document overwrites instead of duplicating.
    return hashlib.sha1(f"{source}|{page}|{index}".encode()).hexdigest()


async def ingest_documents(
    search_client: SearchClient,
    embeddings: Embeddings,
    documents: list[Document],
    batch_size: int = 32,
) -> int:
    """Chunk, embed, and upload documents. Returns the number of chunks stored.

    Each Document's metadata should have: title, source, and optionally page.
    """
    chunks = split_documents(documents)
    counters: dict[tuple[str, int], int] = {}
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset : offset + batch_size]
        vectors = await embeddings.aembed_documents([c.page_content for c in batch])
        payload = []
        for chunk, vector in zip(batch, vectors):
            source = chunk.metadata.get("source", "unknown")
            page = int(chunk.metadata.get("page", 1))
            position = counters.get((source, page), 0)
            counters[(source, page)] = position + 1
            payload.append(
                {
                    "id": _chunk_id(source, page, position),
                    "content": chunk.page_content,
                    "title": chunk.metadata.get("title", source),
                    "source": source,
                    "page": page,
                    "content_vector": vector,
                }
            )
        await search_client.upload_documents(payload)
    return len(chunks)
