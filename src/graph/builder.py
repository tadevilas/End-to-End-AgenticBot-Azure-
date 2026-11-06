"""Builds and compiles the LangGraph workflow.

START -> planner -> [retrieval] -> [mcp_tool] -> response -> citation -> END
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.citation import CitationAgent
from src.agents.mcp_tool import MCPToolAgent
from src.agents.planner import PlannerAgent
from src.agents.response import ResponseAgent
from src.agents.retrieval import RetrievalAgent
from src.graph.state import ChatState
from src.mcp.tavily import TavilySearch


def _after_planner(state: ChatState) -> str:
    plan = state["plan"]
    if plan.use_documents:
        return "retrieval"
    if plan.use_web:
        return "mcp_tool"
    return "response"


def _after_retrieval(state: ChatState) -> str:
    return "mcp_tool" if state["plan"].use_web else "response"


def build_graph(
    llm: BaseChatModel,
    planner_llm: BaseChatModel,
    retriever: BaseRetriever,
    tavily: TavilySearch,
) -> CompiledStateGraph:
    graph = StateGraph(ChatState)

    graph.add_node("planner", PlannerAgent(planner_llm).run)
    graph.add_node("retrieval", RetrievalAgent(retriever).run)
    graph.add_node("mcp_tool", MCPToolAgent(tavily).run)
    graph.add_node("response", ResponseAgent(llm).run)
    graph.add_node("citation", CitationAgent().run)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        _after_planner,
        {"retrieval": "retrieval", "mcp_tool": "mcp_tool", "response": "response"},
    )
    graph.add_conditional_edges(
        "retrieval", _after_retrieval, {"mcp_tool": "mcp_tool", "response": "response"}
    )
    graph.add_edge("mcp_tool", "response")
    graph.add_edge("response", "citation")
    graph.add_edge("citation", END)

    return graph.compile()
