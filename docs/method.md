# Method overview

CausalRAG2 answers a question over a knowledge graph in three stages
(implemented in `causalrag2/core.py::run_single`):

1. **Seed retrieval.** The query is matched against entities, relationships,
   text units, and community summaries using a blend of dense (embedding) and
   lexical similarity. This yields a small set of seed nodes.

2. **Causal subgraph expansion.** Starting from the seeds, the retriever walks
   the graph for up to `max_hops`, following four edge types with different
   weights: `causal` (community-to-community causal gates), `structural`
   (community hierarchy), `membership` (entity-in-community), and
   `relationship` (entity-to-entity). Expansion is budgeted by fan-out and a
   hop decay so the subgraph stays focused.

3. **Causal rerank then answer.** An LLM reranks the retrieved context items and
   drafts an answer (`causal` or counterfactual `ct` mode), then a second LLM
   call writes the final answer grounded in the selected evidence.

The distinctive ingredient is the **causal gate** layer: directed edges between
communities that were accepted by an explicit causal judgment at index time (see
[graph_construction.md](graph_construction.md)). These gates let retrieval jump
between topically distant but causally related parts of the corpus.

## Key parameters

`run_single` exposes many knobs; the defaults follow the paper (Appendix F.3 / B.2):

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `model` | answer / rerank LLM | `gpt-5-nano` |
| `temperature` | generation temperature | `0.0` |
| `embed_model` | sentence-transformers model for retrieval | `all-MiniLM-L6-v2` |
| `seed_k_entities` / `seed_k_communities` | seed budget K0 / KL | `3` / `3` |
| `seed_lexical_alpha` | hybrid scoring weight α (semantic vs lexical) | `0.7` |
| `hop_decay` | traversal decay γ | `0.7` |
| `edge_weight_causal` / `_structural`+`_membership` / `_relationship` | causal gate / hierarchical / structural edge weights | `1.2` / `1.0` / `0.8` |
| `max_hops` | subgraph expansion depth | `5` |
| `causal_llm_mode` | `causal` or `ct` (spurious-aware, main setting) | auto |

The full return value is a dict with `answer`, `retrieved_context`,
`usage`, and a `meta` block recording the seeds, subgraph, and prompts for
inspection.
