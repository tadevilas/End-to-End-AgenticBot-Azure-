"""FastAPI application. Start it with:

    uv run uvicorn src.api.app:app --port 8000
"""

import asyncio
import logging
import warnings
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.core.config import get_settings
from src.graph.builder import build_graph
from src.infrastructure.cosmos import create_cosmos_client, get_conversations_container
from src.infrastructure.llm import create_chat_model, create_planner_model
from src.mcp.tavily import create_tavily_search
from src.memory.repository import CosmosConversationRepository
from src.rag.embedder import LocalEmbedder
from src.rag.retriever import HybridRetriever
from src.services.chat_service import ChatService

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the shared clients once at startup and close them cleanly at shutdown."""
    settings = get_settings()
    async with AsyncExitStack() as stack:
        cosmos = await stack.enter_async_context(create_cosmos_client(settings))
        search = await stack.enter_async_context(
            SearchClient(
                settings.azure_search_endpoint,
                settings.azure_search_index_name,
                AzureKeyCredential(settings.azure_search_api_key.get_secret_value()),
            )
        )
        # Loading the embedding model is slow and blocking, so keep it off the event loop.
        embeddings = await asyncio.to_thread(
            LocalEmbedder, settings.embedding_model_name, settings.embedding_dimension
        )
        graph = build_graph(
            create_chat_model(settings),
            create_planner_model(settings),
            HybridRetriever(search_client=search, embeddings=embeddings),
            create_tavily_search(settings),
        )
        repository = CosmosConversationRepository(get_conversations_container(cosmos, settings))
        app.state.chat_service = ChatService(repository, graph)
        logger.info("Chat service ready")
        yield


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    app = FastAPI(title="Enterprise Agentic RAG Chat API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-User-Id"],
    )
    app.include_router(router)
    return app


app = create_app()
