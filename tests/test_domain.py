"""Tests for domain models."""

from src.domain.conversation import Conversation, Message, ConversationSummary
from src.domain.plan import Citation, Plan


def test_plan_defaults():
    plan = Plan()
    assert plan.use_documents is True
    assert plan.use_web is False


def test_citation_model():
    citation = Citation(
        index=1,
        title="Azure Docs",
        source="https://docs.microsoft.com",
        kind="web",
        snippet="Azure Search provides hybrid querying.",
    )
    assert citation.index == 1
    assert citation.title == "Azure Docs"
    assert citation.kind == "web"


def test_message_creation():
    msg = Message(role="user", content="Hello Copilot")
    assert msg.role == "user"
    assert msg.content == "Hello Copilot"
    assert msg.id is not None
    assert msg.created_at is not None


def test_conversation_summary_serialization():
    conv = Conversation(user_id="user_123", title="Test Thread")
    summary = ConversationSummary(
        id=conv.id,
        title=conv.title,
        updatedAt=conv.updated_at,
    )
    assert summary.id == conv.id
    assert summary.title == "Test Thread"
