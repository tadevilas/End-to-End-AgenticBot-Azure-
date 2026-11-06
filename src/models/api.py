"""Request and response shapes of the HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field

from src.domain.plan import Citation


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None  # omit to start a new conversation


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    title: str
    answer: str
    citations: list[Citation]


class FeedbackRequest(BaseModel):
    conversation_id: str
    message_id: str
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
