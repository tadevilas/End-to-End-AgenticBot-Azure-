"""Tests for CitationAgent."""

import pytest
from src.agents.citation import CitationAgent
from src.domain.plan import Citation


@pytest.mark.asyncio
async def test_citation_agent_filters_unused_and_invalid_citations():
    agent = CitationAgent()

    sources = [
        Citation(index=1, title="Doc 1", source="doc1.pdf", kind="document", snippet="Content 1"),
        Citation(index=2, title="Doc 2", source="doc2.pdf", kind="document", snippet="Content 2"),
        Citation(index=3, title="Web 3", source="https://example.com", kind="web", snippet="Web snippet"),
    ]

    # Answer cites [1] and non-existent [99], but does not cite [2] or [3]
    state = {
        "answer": "According to the official docs [1], and missing ref [99], this is correct.",
        "sources": sources,
    }

    result = await agent.run(state)

    # [99] should be stripped, [1] preserved
    assert "[1]" in result["answer"]
    assert "[99]" not in result["answer"]
    assert len(result["citations"]) == 1
    assert result["citations"][0].index == 1
    assert result["citations"][0].title == "Doc 1"


@pytest.mark.asyncio
async def test_citation_agent_fallback_on_empty_answer():
    agent = CitationAgent()
    state = {"answer": "", "sources": []}
    result = await agent.run(state)

    assert "Sorry, I couldn't generate an answer" in result["answer"]
    assert result["citations"] == []
