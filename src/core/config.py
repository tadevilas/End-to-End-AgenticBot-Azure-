"""Application settings, loaded once from environment variables / .env."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # General
    environment: str = "development"
    log_level: str = "INFO"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:8501"]

    # Azure AI Foundry - chat model
    azure_ai_foundry_chat_endpoint: str
    azure_ai_foundry_chat_key: SecretStr
    azure_ai_foundry_chat_model_name: str = "Phi-4-mini-instruct"
    # Stronger model used only by the planner (small models can't follow its JSON format).
    azure_ai_foundry_planner_model_name: str = "Llama-3.3-70B-Instruct"

    # Local embeddings (fastembed)
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384

    # Azure AI Search
    azure_search_endpoint: str
    azure_search_index_name: str = "kb-foundry-documents"
    azure_search_api_key: SecretStr

    # Azure Cosmos DB
    azure_cosmos_endpoint: str
    azure_cosmos_database_name: str = "enterprise_rag_db"
    azure_cosmos_container_name: str = "conversations"
    azure_cosmos_key: SecretStr

    # Azure Blob Storage (source documents; accessed with Azure AD login, no key)
    azure_storage_account_name: str = "stragchatbot2112"
    azure_storage_container_name: str = "documents"

    # Azure Key Vault
    azure_keyvault_url: str = ""

    # MCP tools
    tavily_api_key: SecretStr
    databricks_server_hostname: str = ""
    databricks_http_path: str = ""
    databricks_token: SecretStr = SecretStr("")

    # Streamlit frontend
    backend_api_url: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
