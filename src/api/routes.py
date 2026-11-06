"""HTTP endpoints. All handlers are async and delegate to ChatService."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response

from src.api.dependencies import ChatServiceDep, UserIdDep
from src.domain.conversation import Conversation, ConversationSummary
from src.models.api import ChatRequest, ChatResponse, FeedbackRequest, HealthResponse
from src.services.chat_service import ChatUnavailableError, ConversationNotFoundError

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, service: ChatServiceDep, user_id: UserIdDep) -> ChatResponse:
    try:
        return await service.chat(user_id, body.message, body.conversation_id)
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    except ChatUnavailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get(
    "/conversations", response_model=list[ConversationSummary], response_model_by_alias=False
)
async def list_conversations(
    service: ChatServiceDep,
    user_id: UserIdDep,
    q: Annotated[str | None, Query(max_length=100, description="Search titles and messages")] = None,
) -> list[ConversationSummary]:
    return await service.list_conversations(user_id, q)


@router.get(
    "/conversation/{conversation_id}", response_model=Conversation, response_model_by_alias=False
)
async def get_conversation(
    conversation_id: str, service: ChatServiceDep, user_id: UserIdDep
) -> Conversation:
    try:
        return await service.get_conversation(user_id, conversation_id)
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail="Conversation not found.")


@router.post("/feedback", status_code=204)
async def feedback(body: FeedbackRequest, service: ChatServiceDep, user_id: UserIdDep) -> Response:
    try:
        await service.add_feedback(user_id, body)
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail="Conversation or message not found.")
    return Response(status_code=204)
