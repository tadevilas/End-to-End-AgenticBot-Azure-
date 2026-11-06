"""Tests for FastAPI endpoints."""

from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from src.api.routes import router
from src.api.dependencies import get_chat_service
from src.domain.conversation import Conversation, ConversationSummary
from src.models.api import ChatResponse
from src.services.chat_service import ChatService, ConversationNotFoundError


@pytest.fixture
def mock_chat_service():
    service = MagicMock(spec=ChatService)
    return service


@pytest.fixture
def app(mock_chat_service):
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[get_chat_service] = lambda: mock_chat_service
    return test_app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_conversations_empty(client, mock_chat_service):
    mock_chat_service.list_conversations = AsyncMock(return_value=[])
    response = client.get("/conversations", headers={"X-User-Id": "test-user"})
    assert response.status_code == 200
    assert response.json() == []


def test_list_conversations_success(client, mock_chat_service):
    conv = Conversation(user_id="test-user", title="Test Session")
    summary = ConversationSummary(id=conv.id, title=conv.title, updatedAt=conv.updated_at)
    mock_chat_service.list_conversations = AsyncMock(return_value=[summary])

    response = client.get("/conversations", headers={"X-User-Id": "test-user"})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == conv.id
    assert data[0]["title"] == "Test Session"


def test_get_conversation_not_found(client, mock_chat_service):
    mock_chat_service.get_conversation = AsyncMock(side_effect=ConversationNotFoundError("Not found"))
    response = client.get("/conversation/non-existent-id", headers={"X-User-Id": "test-user"})
    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."


def test_chat_endpoint_success(client, mock_chat_service):
    mock_chat_service.chat = AsyncMock(
        return_value=ChatResponse(
            conversation_id="conv_123",
            message_id="msg_456",
            title="Chat Title",
            answer="This is the answer [1].",
            citations=[],
        )
    )

    response = client.post(
        "/chat",
        headers={"X-User-Id": "test-user"},
        json={"message": "What is the policy?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == "conv_123"
    assert data["answer"] == "This is the answer [1]."


def test_feedback_endpoint_success(client, mock_chat_service):
    mock_chat_service.add_feedback = AsyncMock(return_value=None)
    response = client.post(
        "/feedback",
        headers={"X-User-Id": "test-user"},
        json={
            "conversation_id": "conv_123",
            "message_id": "msg_456",
            "rating": "up",
            "comment": "Great answer!",
        },
    )
    assert response.status_code == 204
