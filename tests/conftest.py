"""Pytest configuration and common fixtures."""

import os
import pytest

# Ensure required environment variables for test runs without live Azure credentials
os.environ.setdefault("AZURE_AI_FOUNDRY_CHAT_ENDPOINT", "https://mock-foundry.openai.azure.com")
os.environ.setdefault("AZURE_AI_FOUNDRY_CHAT_KEY", "mock-key")
os.environ.setdefault("AZURE_COSMOS_ENDPOINT", "https://mock-cosmos.documents.azure.com:443/")
os.environ.setdefault("AZURE_COSMOS_KEY", "mock-key==")
os.environ.setdefault("AZURE_SEARCH_ENDPOINT", "https://mock-search.search.windows.net")
os.environ.setdefault("AZURE_SEARCH_API_KEY", "mock-search-key")
os.environ.setdefault("TAVILY_API_KEY", "mock-tavily-key")
os.environ.setdefault("AZURE_BLOB_ACCOUNT_URL", "https://mockstorage.blob.core.windows.net")
