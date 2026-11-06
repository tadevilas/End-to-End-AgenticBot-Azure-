"""Creates the async Cosmos DB client and container handle."""

from azure.cosmos.aio import CosmosClient
from azure.cosmos.aio._container import ContainerProxy

from src.core.config import Settings


def create_cosmos_client(settings: Settings) -> CosmosClient:
    return CosmosClient(
        settings.azure_cosmos_endpoint,
        credential=settings.azure_cosmos_key.get_secret_value(),
    )


def get_conversations_container(client: CosmosClient, settings: Settings) -> ContainerProxy:
    database = client.get_database_client(settings.azure_cosmos_database_name)
    return database.get_container_client(settings.azure_cosmos_container_name)
