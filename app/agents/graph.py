from langgraph.graph import END, StateGraph

from app.agents.nodes import (
    analyze_style,
    check_dataset,
    extract_claims,
    gather_evidence,
    synthesize_verdict,
)
from app.agents.state import GraphState


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("check_dataset", check_dataset)
    graph.add_node("extract_claims", extract_claims)
    graph.add_node("gather_evidence", gather_evidence)
    graph.add_node("analyze_style", analyze_style)
    graph.add_node("synthesize_verdict", synthesize_verdict)

    graph.set_entry_point("check_dataset")
    graph.add_conditional_edges(
        "check_dataset",
        lambda state: "matched" if state["dataset_match"] else "not_matched",
        {"matched": END, "not_matched": "extract_claims"},
    )
    graph.add_edge("extract_claims", "gather_evidence")
    graph.add_edge("gather_evidence", "analyze_style")
    graph.add_edge("analyze_style", "synthesize_verdict")
    graph.add_edge("synthesize_verdict", END)

    return graph.compile()


compiled_graph = build_graph()
