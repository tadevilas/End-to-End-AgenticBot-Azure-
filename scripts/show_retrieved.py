"""Show which chunks the retriever returns for a question (no LLM involved).

Usage:
    uv run python -X utf8 -m scripts.show_retrieved "What does the policy say about leave?"
"""

import asyncio
import sys
import warnings

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient

from src.core.config import get_settings
from src.rag.embedder import LocalEmbedder
from src.rag.retriever import HybridRetriever

warnings.filterwarnings("ignore")


async def main(question: str) -> None:
    settings = get_settings()
    embeddings = await asyncio.to_thread(
        LocalEmbedder, settings.embedding_model_name, settings.embedding_dimension
    )
    async with SearchClient(
        settings.azure_search_endpoint,
        settings.azure_search_index_name,
        AzureKeyCredential(settings.azure_search_api_key.get_secret_value()),
    ) as search:
        retriever = HybridRetriever(search_client=search, embeddings=embeddings)
        docs = await asyncio.wait_for(retriever.ainvoke(question), timeout=60)

    print(f"\nQuestion: {question}\nRetrieved {len(docs)} chunk(s)\n")
    for i, doc in enumerate(docs, start=1):
        m = doc.metadata
        print(f"--- [{i}] {m['source']} | page {m['page']} | score {m['score']:.3f}")
        print(doc.page_content)
        print()


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "What is attendance criteria?"))
