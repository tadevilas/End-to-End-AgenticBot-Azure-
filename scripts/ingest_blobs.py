"""Ingest every supported file from the Blob Storage container into Azure AI Search.

Flow:  Blob Storage -> loader (file to text) -> chunks -> local embeddings -> Azure AI Search

Run:  uv run python -X utf8 -m scripts.ingest_blobs
"""

import asyncio
import warnings

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient

from src.core.config import get_settings
from src.rag.blob_source import create_blob_source
from src.rag.embedder import LocalEmbedder
from src.rag.index import ensure_index
from src.rag.ingest import ingest_documents, split_documents
from src.rag.loaders import is_supported, load_file

warnings.filterwarnings("ignore")


async def main() -> None:
    s = get_settings()
    credential = AzureKeyCredential(s.azure_search_api_key.get_secret_value())
    embeddings = LocalEmbedder(s.embedding_model_name, s.embedding_dimension)

    async with SearchIndexClient(s.azure_search_endpoint, credential) as index_client:
        await ensure_index(index_client, s.azure_search_index_name, s.embedding_dimension)

    total_chunks = 0
    async with SearchClient(s.azure_search_endpoint, s.azure_search_index_name, credential) as search:
        async for name, data in create_blob_source(s).iter_files():
            if not is_supported(name):
                print(f"skip   {name} (unsupported type)")
                continue
            try:
                documents = load_file(data, name)
                stored = await ingest_documents(search, embeddings, documents)
            except Exception as exc:
                print(f"FAILED {name}: {type(exc).__name__}: {exc}")
                continue
            pages = len(documents)
            chunks_preview = len(split_documents(documents))
            print(f"ok     {name}: {len(data) // 1024} KB -> {pages} page(s) -> {chunks_preview} chunk(s) stored={stored}")
            total_chunks += stored

    print(f"\nDone. {total_chunks} chunk(s) uploaded to index '{s.azure_search_index_name}'.")


if __name__ == "__main__":
    asyncio.run(main())
