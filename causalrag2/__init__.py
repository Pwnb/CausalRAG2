"""CausalRAG2: causal, hierarchical graph retrieval-augmented generation.

Public API
----------
build_graph(source, out_root, ...)   Build graph artifacts from raw text (indexer).
run_single(example, graph_root, ...)  Answer a question over a built graph.
load_graph(graph_root)                Load a built graph as a GraphData object.
"""

from .core import GraphData, load_graph, run_single
from .indexer import LLMClient, build_graph

__all__ = ["build_graph", "run_single", "load_graph", "GraphData", "LLMClient"]
__version__ = "0.1.0"
