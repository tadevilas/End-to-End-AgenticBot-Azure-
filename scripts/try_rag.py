"""Quick manual check of the RAG pipeline: ingest -> hybrid retrieve -> LangChain answer.

Run:  uv run python -X utf8 scripts/try_rag.py
"""

import asyncio
import warnings

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.core.config import get_settings
from src.infrastructure.llm import create_chat_model
from src.rag.embedder import LocalEmbedder
from src.rag.ingest import ingest_documents
from src.rag.retriever import HybridRetriever

warnings.filterwarnings("ignore")

DOCS = [
    Document(
        page_content="Azure Cosmos DB free tier gives 1000 RU/s and 25 GB of storage at no cost. "
        "Each subscription can have only one free tier account.",
        metadata={"title": "Cosmos DB Guide", "source": "cosmos-guide.txt"},
    ),
    Document(
        page_content="Azure AI Search supports hybrid search which combines BM25 keyword matching "
        "with vector similarity. The free tier allows 3 indexes and 50 MB of storage.",
        metadata={"title": "Search Guide", "source": "search-guide.txt"},
    ),
    Document(
        page_content="Docker multi-stage builds keep production images small. Build dependencies "
        "in one stage, then copy only the runtime files into a slim final image.",
        metadata={"title": "Docker Guide", "source": "docker-guide.txt"},
    ),
]

QUESTION = "how many indexes are free in search?"


async def main() -> None:
    s = get_settings()
    credential = AzureKeyCredential(s.azure_search_api_key.get_secret_value())
    embeddings = LocalEmbedder(s.embedding_model_name, s.embedding_dimension)

    async with SearchClient(s.azure_search_endpoint, s.azure_search_index_name, credential) as client:
        print("1. ingested chunks :", await ingest_documents(client, embeddings, DOCS))
        await asyncio.sleep(3)  # let the index refresh

        retriever = HybridRetriever(search_client=client, embeddings=embeddings, top_k=2)
        docs = await retriever.ainvoke(QUESTION)
        print("2. retriever.ainvoke:")
        for d in docs:
            print(f"     {d.metadata['score']:.3f} | {d.metadata['title']} | {d.page_content[:50]}...")

        prompt = ChatPromptTemplate.from_template(
            "Answer using only this context:\n{context}\n\nQuestion: {question}\nAnswer in one sentence."
        )
        chain = prompt | create_chat_model(s) | StrOutputParser()
        context = "\n".join(d.page_content for d in docs)
        print("3. LLM answer      :", await chain.ainvoke({"context": context, "question": QUESTION}))


if __name__ == "__main__":
    asyncio.run(main())
