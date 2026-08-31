from __future__ import annotations
from typing import TypedDict, Any, Callable
from langgraph.graph import StateGraph, END


class AgentState(TypedDict, total=False):
    run_id: str
    request: dict
    validation: dict
    results: dict
    errors: list[str]


def _validate(s: AgentState):
    return {"validation": {"valid": True}}


def _orchestrate(s: AgentState):
    # Model execution is delegated to explicit pipeline functions. No LLM node exists.
    return {"results": {"orchestrated": True}}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("validate", _validate)
    g.add_node("orchestrate", _orchestrate)
    g.set_entry_point("validate")
    g.add_edge("validate", "orchestrate")
    g.add_edge("orchestrate", END)
    return g.compile()


def run_graph(request: dict[str, Any], handler: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """Deterministic two-node Web state machine with the real handler at orchestration.

    Batch entrypoints do not call this helper; it exists only to keep the README's
    Streamlit -> FastAPI -> LangGraph boundary explicit for one-case Web inference.
    """
    result_holder: dict[str, Any] = {}

    def validate(state: AgentState):
        payload = state.get("request") or {}
        if not isinstance(payload, dict):
            raise ValueError("request must be a JSON object")
        return {"validation": {"valid": True}}

    def orchestrate(state: AgentState):
        result_holder["value"] = handler(dict(state.get("request") or {}))
        return {"results": {"orchestrated": True}}

    g = StateGraph(AgentState)
    g.add_node("validate", validate)
    g.add_node("orchestrate", orchestrate)
    g.set_entry_point("validate")
    g.add_edge("validate", "orchestrate")
    g.add_edge("orchestrate", END)
    g.compile().invoke({"request": dict(request)})
    return result_holder["value"]
