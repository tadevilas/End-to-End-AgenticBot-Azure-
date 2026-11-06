"""Creates the Azure AI Search index (text + vector + semantic ranking)."""

from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

VECTOR_PROFILE = "vector-profile"
VECTOR_ALGORITHM = "hnsw-config"
SEMANTIC_CONFIG = "semantic-config"


def build_index(index_name: str, dimension: int) -> SearchIndex:
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchableField(name="title", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dimension,
            vector_search_profile_name=VECTOR_PROFILE,
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name=VECTOR_ALGORITHM)],
        profiles=[VectorSearchProfile(name=VECTOR_PROFILE, algorithm_configuration_name=VECTOR_ALGORITHM)],
    )
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIG,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="content")],
                ),
            )
        ]
    )
    return SearchIndex(
        name=index_name, fields=fields, vector_search=vector_search, semantic_search=semantic_search
    )


async def ensure_index(client: SearchIndexClient, index_name: str, dimension: int) -> None:
    """Create the index if missing; leave it alone if it already exists."""
    existing = [name async for name in client.list_index_names()]
    if index_name not in existing:
        await client.create_index(build_index(index_name, dimension))
