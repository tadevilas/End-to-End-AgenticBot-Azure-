"""Tests for PlannerAgent."""

from unittest.mock import AsyncMock, MagicMock
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from src.agents.planner import PlannerAgent
from src.domain.plan import Plan


@pytest.mark.asyncio
async def test_planner_agent_run():
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.ainvoke = AsyncMock(return_value=Plan(use_documents=True, use_web=False))
    mock_llm.with_structured_output.return_value = mock_structured_llm

    planner = PlannerAgent(mock_llm)
    state = {
        "question": "What is our enterprise PTO policy?",
        "history": [HumanMessage("Hello"), AIMessage("Hi! How can I help you?")],
    }

    result = await planner.run(state)
    assert "plan" in result
    assert result["plan"].use_documents is True
    assert result["plan"].use_web is False
