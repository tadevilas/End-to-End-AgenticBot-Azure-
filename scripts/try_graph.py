"""Manual check of the full LangGraph workflow against real Azure services.

Run:  uv run python -X utf8 -m scripts.try_graph
"""

import asyncio
import warnings

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from langchain_core.messages import AIMessage, HumanMessage

from src.core.config import get_settings
from src.graph.builder import build_graph
from src.infrastructure.llm import create_chat_model, create_planner_model
from src.mcp.tavily import create_tavily_search
from src.rag.embedder import LocalEmbedder
from src.rag.retriever import HybridRetriever

warnings.filterwarnings("ignore")

CASES = [
    ("document question", "How many indexes does the free tier of Azure AI Search allow?", []),
    ("web question", "What is the latest stable version of Python released this year?", []),
    ("small talk", "Hi, thanks for your help!", []),
    (
        "follow-up using history",
        "And how much storage does it include?",
        [
            HumanMessage("How many indexes does the free tier of Azure AI Search allow?"),
            AIMessage("The free tier of Azure AI Search allows 3 indexes [1]."),
        ],
    ),
]


async def main() -> None:
    s = get_settings()
    credential = AzureKeyCredential(s.azure_search_api_key.get_secret_value())
    embeddings = LocalEmbedder(s.embedding_model_name, s.embedding_dimension)

    async with SearchClient(s.azure_search_endpoint, s.azure_search_index_name, credential) as client:
        retriever = HybridRetriever(search_client=client, embeddings=embeddings, top_k=3)
        graph = build_graph(create_chat_model(s), create_planner_model(s), retriever, create_tavily_search(s))

        for label, question, history in CASES:
            result = await asyncio.wait_for(graph.ainvoke({"question": question, "history": history, "errors": []}), timeout=120)
            plan = result["plan"]
            print(f"\n=== {label}: {question!r}")
            print(f"plan     : documents={plan.use_documents} web={plan.use_web}")
            print(f"answer   : {result['answer']}")
            for c in result["citations"]:
                print(f"  cite [{c.index}] {c.kind}: {c.title} ({c.source[:60]})")
            if result.get("errors"):
                print("errors   :", result["errors"])


if __name__ == "__main__":
    asyncio.run(main())
