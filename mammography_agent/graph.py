from __future__ import annotations
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END

class AgentState(TypedDict, total=False):
    run_id: str
    request: dict
    validation: dict
    results: dict
    errors: list[str]

def _validate(s: AgentState):
    return {"validation":{"valid":True}}

def _orchestrate(s: AgentState):
    # Model execution is delegated to explicit pipeline functions. No LLM node exists.
    return {"results":{"orchestrated":True}}

def build_graph():
    g=StateGraph(AgentState); g.add_node("validate",_validate); g.add_node("orchestrate",_orchestrate)
    g.set_entry_point("validate"); g.add_edge("validate","orchestrate"); g.add_edge("orchestrate",END)
    return g.compile()
