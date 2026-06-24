# Graph construction

`causalrag2/indexer.py` builds the graph artifacts the method consumes, following
the paper's offline pipeline (Section 4.1, Appendix B.1):

```
raw text
  -> chunk into text units
  -> LLM entity + relationship extraction        (Figure 8)
  -> two-stage entity canonicalization            (B.1: fuzzy string + embedding)
  -> recursive multi-level Leiden hierarchy        (B.1: H0 entities .. HL modules)
  -> LLM community reports
  -> causal gates via Top-Down Hierarchical Pruning (Algorithm 2 + Figure 9)
  -> parquet output
```

## Steps in detail

1. **Extraction (Figure 8).** Each chunk is sent to the GraphRAG-style IE prompt
   (`IE_EXTRACTION_PROMPT` in `prompts.py`), producing tuple-delimited entity and
   relationship records. Extraction is concurrent and cached by content hash.

2. **Entity canonicalization (Appendix B.1).** Mentions are merged in two stages:
   surface-level normalization plus fuzzy string matching, then embedding-similarity
   merging of the remaining canonical names. Descriptions and supporting text units
   are pooled per canonical entity.

3. **Hierarchical partitioning (Appendix B.1).** The entity graph is recursively
   partitioned with the Leiden algorithm: a module larger than `max_cluster_size`
   is subdivided into a finer level, up to `max_levels`. Levels are indexed
   bottom-up, so the hierarchy reads H0 (entity base) .. HL (coarsest modules),
   with parent/child links between adjacent levels.

4. **Community reports.** Each module gets an LLM-written report (title + summary),
   generated bottom-up so a module can name its sub-modules. Pass
   `llm_reports=False` for a cheaper extractive variant.

5. **Causal gates (Algorithm 2, Figure 9).** Gates are built with **Top-Down
   Hierarchical Pruning**: iterating from the coarsest level to the finest, each
   module is checked against its same-level peers (intra-layer), then against the
   next finer level while pruning its own children and the children of peers it is
   already gate-connected to (inter-layer look-ahead). Each check is the binary
   yes/no causal verification prompt (`CAUSAL_GATE_PROMPT`). Accepted gates are
   undirected.

## Output layout

```
<out_root>/output/          base index
    entities.parquet            id, title, description, text_unit_ids
    relationships.parquet       id, source, target, description, text_unit_ids
    text_units.parquet          id, text
    communities.parquet         community, level, title, children, entity_ids, text_unit_ids
    community_reports.parquet   community, title, summary
<out_root>/output_causal/   output/ + causal gates (this is what run_single reads)
    community_causal.parquet    + causal_children (undirected gates)
```

## Building a graph

```bash
# from a text file, a directory of .txt files, or a .jsonl corpus
python -m causalrag2.indexer path/to/corpus.txt runs/my_graph --model gpt-5-nano
```

or from Python:

```python
from causalrag2 import build_graph
build_graph("path/to/corpus.txt", out_root="runs/my_graph",
            max_cluster_size=10, max_levels=4, llm_reports=True, build_causal=True)
```

Optional libraries (scikit-learn, sentence-transformers, igraph/leidenalg) are used
when present; on a minimal install the partitioning uses NetworkX community
detection and embeddings use a hashed/TF-IDF backend. Install the `[full]` extras
for the highest-quality build.
