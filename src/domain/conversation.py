"""Core chat entities. Pure data - no Azure or database code lives here."""

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.domain.plan import Citation


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid4())


class Message(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=_new_id)
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=_now, alias="createdAt")
    citations: list[Citation] = Field(default_factory=list)  # assistant messages only
    rating: Literal["up", "down"] | None = None  # user feedback on an assistant message
    feedback_comment: str | None = Field(default=None, alias="feedbackComment")


class Conversation(BaseModel):
    """One chat thread. Stored as a single Cosmos document."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=_new_id)
    user_id: str = Field(alias="userId")
    title: str = "New chat"
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now, alias="createdAt")
    updated_at: datetime = Field(default_factory=_now, alias="updatedAt")


class ConversationSummary(BaseModel):
    """Lightweight row for the sidebar list (no messages)."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    title: str
    updated_at: datetime = Field(alias="updatedAt")
