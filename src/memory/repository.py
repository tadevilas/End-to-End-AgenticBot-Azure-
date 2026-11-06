"""Saves and loads conversations in Azure Cosmos DB.

One conversation = one document, partitioned by userId, so every call below
stays inside a single user's partition (fast and cheap).
"""

from datetime import datetime, timezone
from typing import Literal

from azure.cosmos.aio._container import ContainerProxy
from azure.cosmos.exceptions import CosmosResourceNotFoundError

from src.domain.conversation import Conversation, ConversationSummary, Message


class CosmosConversationRepository:
    def __init__(self, container: ContainerProxy) -> None:
        self._container = container

    async def create(self, user_id: str, title: str = "New chat") -> Conversation:
        conversation = Conversation(user_id=user_id, title=title)
        await self._save(conversation)
        return conversation

    async def get(self, conversation_id: str, user_id: str) -> Conversation | None:
        try:
            item = await self._container.read_item(conversation_id, partition_key=user_id)
        except CosmosResourceNotFoundError:
            return None
        return Conversation.model_validate(item)

    async def add_message(
        self,
        conversation_id: str,
        user_id: str,
        role: Literal["user", "assistant"],
        content: str,
    ) -> Conversation:
        conversation = await self.get(conversation_id, user_id)
        if conversation is None:
            raise LookupError(f"Conversation {conversation_id} not found for user {user_id}")
        conversation.messages.append(Message(role=role, content=content))
        conversation.updated_at = datetime.now(timezone.utc)
        await self._save(conversation)
        return conversation

    async def list_for_user(self, user_id: str, limit: int = 50) -> list[ConversationSummary]:
        query = (
            "SELECT c.id, c.title, c.updatedAt FROM c "
            "WHERE c.userId = @userId ORDER BY c.updatedAt DESC OFFSET 0 LIMIT @limit"
        )
        parameters = [{"name": "@userId", "value": user_id}, {"name": "@limit", "value": limit}]
        return await self._query_summaries(query, parameters, user_id)

    async def search(self, user_id: str, text: str, limit: int = 20) -> list[ConversationSummary]:
        query = (
            "SELECT c.id, c.title, c.updatedAt FROM c WHERE c.userId = @userId AND ("
            "CONTAINS(LOWER(c.title), @text) OR "
            "EXISTS(SELECT VALUE m FROM m IN c.messages WHERE CONTAINS(LOWER(m.content), @text))"
            ") ORDER BY c.updatedAt DESC OFFSET 0 LIMIT @limit"
        )
        parameters = [
            {"name": "@userId", "value": user_id},
            {"name": "@text", "value": text.lower()},
            {"name": "@limit", "value": limit},
        ]
        return await self._query_summaries(query, parameters, user_id)

    async def delete(self, conversation_id: str, user_id: str) -> None:
        try:
            await self._container.delete_item(conversation_id, partition_key=user_id)
        except CosmosResourceNotFoundError:
            pass

    async def save(self, conversation: Conversation) -> None:
        """Insert or update a whole conversation, refreshing its updated-at time."""
        conversation.updated_at = datetime.now(timezone.utc)
        await self._save(conversation)

    async def _save(self, conversation: Conversation) -> None:
        await self._container.upsert_item(conversation.model_dump(mode="json", by_alias=True))

    async def _query_summaries(
        self, query: str, parameters: list[dict], user_id: str
    ) -> list[ConversationSummary]:
        items = self._container.query_items(query, parameters=parameters, partition_key=user_id)
        return [ConversationSummary.model_validate(item) async for item in items]
