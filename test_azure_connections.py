"""
Script to validate all Azure & Azure AI Foundry connections and credentials.
Run this script after filling in your .env file:
    python test_azure_connections.py
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

async def test_foundry_chat():
    print("\n[1/4] Testing Azure AI Foundry Chat Model...")
    endpoint = os.getenv("AZURE_AI_FOUNDRY_CHAT_ENDPOINT")
    key = os.getenv("AZURE_AI_FOUNDRY_CHAT_KEY")
    model = os.getenv("AZURE_AI_FOUNDRY_CHAT_MODEL_NAME", "Phi-3.5-mini-instruct")

    if not endpoint or not key or "your_" in key:
        print("  ⚠️ Skipped: AZURE_AI_FOUNDRY_CHAT_ENDPOINT or KEY not configured in .env")
        return False

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=endpoint, api_key=key)
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Respond with 'OK' if you can read this."}],
            max_tokens=10
        )
        print(f"  ✅ SUCCESS: Chat model '{model}' responded: {response.choices[0].message.content.strip()}")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {str(e)}")
        return False

async def test_local_embeddings():
    print("\n[2/4] Testing Local Embeddings (fastembed)...")
    model = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    expected_dim = int(os.getenv("EMBEDDING_DIMENSION", "384"))

    try:
        from fastembed import TextEmbedding
        embedder = TextEmbedding(model_name=model)
        vector_len = len(next(iter(embedder.embed(["Azure Chatbot Test"]))))
        if vector_len != expected_dim:
            print(f"  ❌ FAILED: '{model}' produced {vector_len} dimensions, but EMBEDDING_DIMENSION={expected_dim}")
            return False
        print(f"  ✅ SUCCESS: Local model '{model}' generated embeddings! Dimension: {vector_len}")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {str(e)}")
        return False

async def test_cosmos_db():
    print("\n[3/4] Testing Azure Cosmos DB...")
    endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")
    key = os.getenv("AZURE_COSMOS_KEY")
    db_name = os.getenv("AZURE_COSMOS_DATABASE_NAME", "enterprise_rag_db")

    if not endpoint or not key or "your_" in key:
        print("  ⚠️ Skipped: AZURE_COSMOS_ENDPOINT or KEY not configured in .env")
        return False

    try:
        from azure.cosmos.aio import CosmosClient
        async with CosmosClient(endpoint, credential=key) as client:
            database = client.get_database_client(db_name)
            await database.read()
            print(f"  ✅ SUCCESS: Connected to Cosmos DB database '{db_name}'")
            return True
    except Exception as e:
        print(f"  ❌ FAILED: {str(e)}")
        return False

async def test_azure_search():
    print("\n[4/4] Testing Azure AI Search...")
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    key = os.getenv("AZURE_SEARCH_API_KEY")

    if not endpoint or not key or "your_" in key:
        print("  ⚠️ Skipped: AZURE_SEARCH_ENDPOINT or KEY not configured in .env")
        return False

    try:
        from azure.search.documents.indexes.aio import SearchIndexClient
        from azure.core.credentials import AzureKeyCredential
        async with SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key)) as client:
            indexes = [index.name async for index in client.list_indexes()]
            print(f"  ✅ SUCCESS: Connected to Azure AI Search! Available indexes: {indexes}")
            return True
    except Exception as e:
        print(f"  ❌ FAILED: {str(e)}")
        return False

async def main():
    print("=" * 60)
    print(" Azure Chatbot Project — Connection & Credential Validation")
    print("=" * 60)
    await test_foundry_chat()
    await test_local_embeddings()
    await test_cosmos_db()
    await test_azure_search()
    print("\n" + "=" * 60)
    print(" Validation Finished")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
