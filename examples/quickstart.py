"""CausalRAG2 quickstart: natural text -> causal graph -> question answering.

1. Put your key in a .env file (see .env.example):  OPENAI_API_KEY=sk-...
2. Run:  python examples/quickstart.py

This builds a small causal knowledge graph from the passage below and answers a
question over it. Swap in your own text and questions freely.
"""

from __future__ import annotations

from causalrag2 import build_graph, run_single

# --- 1. Some natural text to turn into a graph ------------------------------- #
TEXT = """
Sleep deprivation has a wide range of downstream effects on the human body.
When a person does not sleep enough, the body raises levels of the stress
hormone cortisol. Elevated cortisol increases blood pressure and promotes
inflammation throughout the body. Chronic inflammation, in turn, damages the
lining of blood vessels and accelerates the buildup of arterial plaque.

Poor sleep also impairs the brain's ability to clear metabolic waste. During
deep sleep the glymphatic system flushes out beta-amyloid, a protein linked to
Alzheimer's disease. Without enough deep sleep, beta-amyloid accumulates, which
contributes to cognitive decline over time.

Finally, insufficient sleep disrupts the hormones that regulate appetite. It
lowers leptin, which signals fullness, and raises ghrelin, which signals hunger.
The result is increased calorie intake, weight gain, and a higher risk of
type 2 diabetes.
"""

# --- 2. Model + question ----------------------------------------------------- #
MODEL = "gpt-5-nano"
QUESTION = "How can poor sleep lead to cardiovascular damage?"


def main() -> None:
    # Build the causal graph (entities -> communities -> topics + causal edges).
    graph_root = build_graph(TEXT, out_root="runs/quickstart", model=MODEL)

    # Answer the question over the graph.
    result = run_single({"question": QUESTION}, str(graph_root), model=MODEL)

    print("\n" + "=" * 70)
    print("Q:", QUESTION)
    print("A:", result["answer"])
    print("=" * 70)
    print("seed entities :", [e["title"] for e in result["meta"]["seed_entities"]])
    print("subgraph size :", result["meta"]["subgraph_nodes"], "nodes /",
          result["meta"]["subgraph_edges"], "edges")


if __name__ == "__main__":
    main()
