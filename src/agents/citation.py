"""Citation agent: validates the answer and keeps only the sources it really cites."""

import re

from src.graph.state import ChatState

CITATION_MARK = re.compile(r"\[(\d+)\]")
FALLBACK_ANSWER = "Sorry, I couldn't generate an answer right now. Please try again in a moment."


class CitationAgent:
    async def run(self, state: ChatState) -> dict:
        answer = state.get("answer", "").strip()
        if not answer:
            return {"answer": FALLBACK_ANSWER, "citations": []}

        sources = {s.index: s for s in state.get("sources", [])}

        # Output validation: drop [n] markers that point to a source that doesn't exist.
        def keep_valid(match: re.Match) -> str:
            return match.group(0) if int(match.group(1)) in sources else ""

        answer = CITATION_MARK.sub(keep_valid, answer).strip()
        used = sorted({int(n) for n in CITATION_MARK.findall(answer)})
        return {"answer": answer, "citations": [sources[n] for n in used]}
