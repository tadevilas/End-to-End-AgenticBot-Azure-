"""Reads documents from an Azure Blob Storage container.

Uses Azure AD login (your az login locally, a managed identity in Azure) - no storage keys.
"""

from collections.abc import AsyncIterator

from azure.identity.aio import DefaultAzureCredential
from azure.storage.blob.aio import BlobServiceClient

from src.core.config import Settings


class BlobDocumentSource:
    def __init__(self, account_name: str, container_name: str) -> None:
        self._account_url = f"https://{account_name}.blob.core.windows.net"
        self._container_name = container_name

    async def iter_files(self) -> AsyncIterator[tuple[str, bytes]]:
        """Yield (blob name, file bytes) for every blob in the container."""
        async with DefaultAzureCredential() as credential:
            async with BlobServiceClient(self._account_url, credential=credential) as service:
                container = service.get_container_client(self._container_name)
                async for blob in container.list_blobs():
                    downloader = await container.download_blob(blob.name)
                    yield blob.name, await downloader.readall()


def create_blob_source(settings: Settings) -> BlobDocumentSource:
    return BlobDocumentSource(settings.azure_storage_account_name, settings.azure_storage_container_name)
