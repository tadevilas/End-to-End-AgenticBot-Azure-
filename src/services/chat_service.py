"""Chat use-cases: run the agent graph and keep the conversation history in Cosmos DB."""

import asyncio
import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from src.agents.citation import FALLBACK_ANSWER
from src.domain.conversation import Conversation, ConversationSummary, Message
from src.memory.repository import CosmosConversationRepository
from src.models.api import ChatResponse, FeedbackRequest

logger = logging.getLogger(__name__)

GRAPH_TIMEOUT_SECONDS = 90
TITLE_MAX_CHARS = 50


class ConversationNotFoundError(LookupError):
    """The conversation or message does not exist for this user."""


class ChatUnavailableError(RuntimeError):
    """The assistant could not produce an answer."""


def make_title(first_message: str) -> str:
    text = " ".join(first_message.split())
    return text if len(text) <= TITLE_MAX_CHARS else text[: TITLE_MAX_CHARS - 1].rstrip() + "…"


def to_langchain_history(messages: list[Message]) -> list[BaseMessage]:
    return [HumanMessage(m.content) if m.role == "user" else AIMessage(m.content) for m in messages]


class ChatService:
    def __init__(self, repository: CosmosConversationRepository, graph: CompiledStateGraph) -> None:
        self._repository = repository
        self._graph = graph

    async def chat(self, user_id: str, message: str, conversation_id: str | None) -> ChatResponse:
        conversation = await self._load_or_start(user_id, conversation_id)

        try:
            result = await asyncio.wait_for(
                self._graph.ainvoke(
                    {
                        "question": message,
                        "history": to_langchain_history(conversation.messages),
                        "errors": [],
                    }
                ),
                timeout=GRAPH_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.error("Agent graph failed: %s", type(exc).__name__)
            raise ChatUnavailableError("The assistant is temporarily unavailable.") from exc

        answer = result.get("answer", "")
        if not answer or answer == FALLBACK_ANSWER:
            # Don't store a failed turn in the user's history.
            raise ChatUnavailableError("The assistant could not answer right now.")

        if not conversation.messages:
            conversation.title = make_title(message)
        conversation.messages.append(Message(role="user", content=message))
        reply = Message(role="assistant", content=answer, citations=result.get("citations", []))
        conversation.messages.append(reply)
        await self._repository.save(conversation)

        return ChatResponse(
            conversation_id=conversation.id,
            message_id=reply.id,
            title=conversation.title,
            answer=answer,
            citations=reply.citations,
        )

    async def list_conversations(self, user_id: str, search: str | None) -> list[ConversationSummary]:
        if search:
            return await self._repository.search(user_id, search)
        return await self._repository.list_for_user(user_id)

    async def get_conversation(self, user_id: str, conversation_id: str) -> Conversation:
        conversation = await self._repository.get(conversation_id, user_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)
        return conversation

    async def add_feedback(self, user_id: str, feedback: FeedbackRequest) -> None:
        conversation = await self.get_conversation(user_id, feedback.conversation_id)
        target = next(
            (m for m in conversation.messages if m.id == feedback.message_id and m.role == "assistant"),
            None,
        )
        if target is None:
            raise ConversationNotFoundError(feedback.message_id)
        target.rating = feedback.rating
        target.feedback_comment = feedback.comment
        await self._repository.save(conversation)

    async def _load_or_start(self, user_id: str, conversation_id: str | None) -> Conversation:
        if not conversation_id or not conversation_id.strip():
            return Conversation(user_id=user_id)  # saved only after a successful answer
        return await self.get_conversation(user_id, conversation_id.strip())
