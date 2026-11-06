"""Planner agent: decides whether to search documents, the web, both, or neither."""

import json
import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.domain.plan import Plan
from src.graph.state import ChatState

logger = logging.getLogger(__name__)

MAX_HISTORY = 6

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You route questions for an enterprise assistant. Decide which sources are needed.\n"
            "- use_documents: true for any factual or how-to question that the company's knowledge "
            "base (internal documents, guides, policies, technical docs) could answer. "
            "When unsure, choose true.\n"
            "- use_web: true ONLY if the question needs current events, recent releases, live data, "
            "or public information that internal documents would not have.\n"
            "- Both false for greetings, thanks, or small talk.\n"
            'Reply with ONLY a JSON object, e.g. {{"use_documents": true, "use_web": false}}',
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


def parse_plan(raw: str) -> Plan:
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in planner output")
    data = json.loads(match.group(0))
    # Plan has defaults, so unrelated JSON would validate silently. Require the real keys.
    if "use_documents" not in data or "use_web" not in data:
        raise ValueError("planner output is missing use_documents / use_web")
    return Plan.model_validate(data)


class PlannerAgent:
    def __init__(self, llm: BaseChatModel) -> None:
        self._chain = PROMPT | llm | StrOutputParser()

    async def run(self, state: ChatState) -> dict:
        try:
            raw = await self._chain.ainvoke(
                {"question": state["question"], "history": state.get("history", [])[-MAX_HISTORY:]}
            )
            return {"plan": parse_plan(raw)}
        except Exception as exc:
            # Small models sometimes return bad JSON. Safe default: search the documents.
            logger.warning("Planner fell back to document search: %s", type(exc).__name__)
            return {"plan": Plan(use_documents=True, use_web=False)}
