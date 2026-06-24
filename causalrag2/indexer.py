"""Knowledge-graph indexer for CausalRAG2.

This builds the graph artifacts that :func:`causalrag2.core.run_single` consumes,
implementing the paper's offline pipeline (Section 4.1, Appendix B.1):

    raw text
      -> chunk into text units
      -> LLM entity/relation extraction          (Figure 8)
      -> two-stage entity canonicalization        (B.1: fuzzy string + embedding)
      -> recursive multi-level Leiden hierarchy    (B.1: H0 entities .. HL modules)
      -> LLM community reports
      -> causal gates via Top-Down Hierarchical Pruning (Algorithm 2, Figure 9)
      -> parquet output

On-disk schema:

    <out_root>/output/        entities|relationships|text_units|communities|community_reports.parquet
    <out_root>/output_causal/ (copy of output/) + community_causal.parquet

The LLM client is pluggable (``client=`` / the ``LLMClient`` protocol) so the
pipeline is easy to test or point at a different provider. Optional libraries
(scikit-learn, sentence-transformers, igraph/leidenalg) are used when present,
with lightweight fallbacks otherwise.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .graph_utils import (
    EMBEDDINGS,
    _HAVE_LEIDEN,
    _community_detection_with_leiden,
    _community_detection_with_networkx,
)
from .prompts import (
    CAUSAL_GATE_PROMPT,
    COMMUNITY_REPORT_PROMPT,
    COMPLETION_DELIMITER,
    DEFAULT_ENTITY_TYPES,
    IE_EXTRACTION_PROMPT,
    RECORD_DELIMITER,
    TUPLE_DELIMITER,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


# --------------------------------------------------------------------------- #
# Environment / IO helpers
# --------------------------------------------------------------------------- #
def _load_env() -> None:
    """Load KEY=VALUE pairs from a local .env into os.environ (no overwrite)."""
    for candidate in (Path(".env"), Path(__file__).resolve().parents[1] / ".env"):
        if not candidate.exists():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))
        break


def _read_documents(source) -> List[Tuple[str, str]]:
    """Return a list of (doc_id, text).

    ``source`` may be: a list/tuple of strings, raw text, a path to a .txt file,
    a path to a .jsonl file (uses the ``text``/``context_text``/``abstract``
    field), or a directory of .txt files.
    """
    if isinstance(source, (list, tuple)):
        return [(f"doc{i}", str(t)) for i, t in enumerate(source) if str(t).strip()]

    path = Path(source)
    if isinstance(source, str) and not path.exists():
        return [("doc0", source)] if source.strip() else []

    if path.is_dir():
        docs = []
        for fp in sorted(path.glob("*.txt")):
            text = fp.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                docs.append((fp.stem, text))
        return docs

    if path.suffix == ".jsonl":
        docs = []
        for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            text = obj.get("text") or obj.get("context_text") or obj.get("abstract") or ""
            doc_id = str(obj.get("id") or obj.get("qa_id") or f"doc{i}")
            if str(text).strip():
                docs.append((doc_id, str(text)))
        return docs

    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [(path.stem, text)] if text else []


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into ~chunk_size character chunks on whitespace, with overlap."""
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for word in words:
        current.append(word)
        length += len(word) + 1
        if length >= chunk_size:
            chunks.append(" ".join(current))
            keep, kept = [], 0
            for w in reversed(current):
                if kept >= overlap:
                    break
                keep.insert(0, w)
                kept += len(w) + 1
            current, length = keep, kept
    if current:
        chunks.append(" ".join(current))
    return chunks


def _normalize_name(name: str) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", str(name or "").lower())
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# LLM client
# --------------------------------------------------------------------------- #
class LLMClient:
    """Thin OpenAI chat wrapper with retries.

    Any object exposing ``chat(messages, json_mode=False) -> str`` can be used
    instead (e.g. a mock in tests, or a different provider).
    """

    def __init__(
        self,
        model: str = "gpt-5-nano",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        max_retries: int = 5,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, messages: List[Dict], json_mode: bool = False) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or pass api_key=.")
        from openai import OpenAI

        kwargs: Dict = {"model": self.model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - transient API errors
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(2.0 * (2 ** attempt) + 0.25)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_error}")


def _parse_json(text: str, default: Dict) -> Dict:
    cleaned = (text or "").strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        cleaned = cleaned.removeprefix("json").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:
                return default
    return default


# --------------------------------------------------------------------------- #
# Extraction (Figure 8: GraphRAG-style tuple-delimited records)
# --------------------------------------------------------------------------- #
def _clean_field(text: str) -> str:
    return text.strip().strip('"').strip()


def _parse_graphrag_records(content: str) -> Dict:
    """Parse the delimited entity/relationship records produced by the IE prompt."""
    entities: List[Dict] = []
    relationships: List[Dict] = []
    text = (content or "").replace(COMPLETION_DELIMITER, "")
    for raw in text.split(RECORD_DELIMITER):
        record = raw.strip()
        if not record:
            continue
        start, end = record.find("("), record.rfind(")")
        if start != -1 and end != -1 and end > start:
            record = record[start + 1 : end]
        parts = [_clean_field(p) for p in record.split(TUPLE_DELIMITER)]
        if not parts:
            continue
        kind = parts[0].strip().strip('"').lower()
        if kind == "entity" and len(parts) >= 4:
            entities.append({"name": parts[1], "type": parts[2], "description": parts[3]})
        elif kind == "relationship" and len(parts) >= 4:
            relationships.append({"source": parts[1], "target": parts[2], "description": parts[3]})
    return {"entities": entities, "relationships": relationships}


def _extract_chunk(client: LLMClient, passage: str, entity_types: str) -> Dict:
    prompt = IE_EXTRACTION_PROMPT.format(
        entity_types=entity_types,
        tuple_delimiter=TUPLE_DELIMITER,
        record_delimiter=RECORD_DELIMITER,
        completion_delimiter=COMPLETION_DELIMITER,
        input_text=passage,
    )
    content = client.chat([{"role": "user", "content": prompt}])
    return _parse_graphrag_records(content)


class _ExtractCache:
    """Append-only JSONL cache of chunk extractions keyed by (model, text) hash."""

    def __init__(self, path: Path, enabled: bool) -> None:
        self.path = path
        self.enabled = enabled
        self.data: Dict[str, Dict] = {}
        if enabled and path.exists():
            for raw in path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(raw)
                except Exception:
                    continue
                if rec.get("key"):
                    self.data[rec["key"]] = rec["value"]

    @staticmethod
    def key(model: str, text: str) -> str:
        return f"{model}|{hashlib.sha1(text.encode('utf-8')).hexdigest()}"

    def get(self, key: str) -> Optional[Dict]:
        return self.data.get(key) if self.enabled else None

    def put(self, key: str, value: Dict) -> None:
        if not self.enabled:
            return
        self.data[key] = value
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "value": value}, ensure_ascii=True) + "\n")


# --------------------------------------------------------------------------- #
# Two-stage entity canonicalization (Appendix B.1)
# --------------------------------------------------------------------------- #
@dataclass
class _Entity:
    id: str
    title: str
    description: str = ""
    text_unit_ids: List[str] = field(default_factory=list)


def _union_find(items: List[str]):
    parent = {x: x for x in items}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    return find, union


def _canonicalize_entities(
    mentions: List[Tuple[str, str, str]],
    fuzzy_threshold: float,
    embedding_threshold: float,
) -> Tuple[Dict[str, _Entity], Dict[str, str]]:
    """Two-stage dedup of entity mentions -> canonical entities.

    ``mentions`` is a list of (surface_name, description, text_unit_id).
    Returns (entities_by_id, norm_name -> entity_id).
    Stage 1: surface canonicalization (normalization + fuzzy string matching).
    Stage 2: embedding-similarity merging of the remaining canonical names.
    """
    # group identical-after-normalization surface forms
    groups: Dict[str, Dict] = {}
    for name, desc, tu in mentions:
        norm = _normalize_name(name)
        if not norm:
            continue
        g = groups.get(norm)
        if g is None:
            g = {"surface": str(name).strip(), "desc": str(desc or ""), "tus": []}
            groups[norm] = g
        if desc and len(str(desc)) > len(g["desc"]):
            g["desc"] = str(desc)
        if tu not in g["tus"]:
            g["tus"].append(tu)

    norms = list(groups)
    find, union = _union_find(norms)

    # Stage 1: fuzzy string matching (bucketed by leading characters for speed)
    buckets: Dict[str, List[str]] = defaultdict(list)
    for n in norms:
        buckets[n[:2]].append(n)
    for bucket in buckets.values():
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a, b = bucket[i], bucket[j]
                if abs(len(a) - len(b)) > 8:
                    continue
                if difflib.SequenceMatcher(None, a, b).ratio() >= fuzzy_threshold:
                    union(a, b)

    # collapse stage-1 clusters
    cluster_members: Dict[str, List[str]] = defaultdict(list)
    for n in norms:
        cluster_members[find(n)].append(n)

    cluster_keys = list(cluster_members)

    def _merge_group(norm_list: List[str]) -> Dict:
        surface = max((groups[n]["surface"] for n in norm_list), key=len)
        desc = max((groups[n]["desc"] for n in norm_list), key=len, default="")
        tus: List[str] = []
        for n in norm_list:
            for tu in groups[n]["tus"]:
                if tu not in tus:
                    tus.append(tu)
        return {"surface": surface, "desc": desc, "tus": tus}

    cluster_rec = {key: _merge_group(members) for key, members in cluster_members.items()}

    # Stage 2: embedding-similarity merge across stage-1 clusters
    if embedding_threshold < 1.0 and len(cluster_keys) > 1:
        titles = [cluster_rec[k]["surface"] for k in cluster_keys]
        try:
            embs = np.asarray(EMBEDDINGS.encode(titles), dtype="float32")
            norms_vec = np.linalg.norm(embs, axis=1, keepdims=True)
            norms_vec[norms_vec == 0] = 1.0
            unit = embs / norms_vec
            sims = unit @ unit.T
            find2, union2 = _union_find(cluster_keys)
            for i in range(len(cluster_keys)):
                for j in range(i + 1, len(cluster_keys)):
                    if sims[i, j] >= embedding_threshold:
                        union2(cluster_keys[i], cluster_keys[j])
            merged: Dict[str, List[str]] = defaultdict(list)
            for k in cluster_keys:
                merged[find2(k)].extend(cluster_members[k])
            cluster_members = merged
            cluster_rec = {key: _merge_group(members) for key, members in cluster_members.items()}
        except Exception:
            pass  # embedding backend unavailable; keep stage-1 result

    # assign final ids
    entities: Dict[str, _Entity] = {}
    norm_to_id: Dict[str, str] = {}
    for idx, (key, members) in enumerate(cluster_members.items()):
        eid = str(idx)
        rec = cluster_rec[key]
        entities[eid] = _Entity(id=eid, title=rec["surface"], description=rec["desc"],
                                text_unit_ids=list(rec["tus"]))
        for n in members:
            norm_to_id[n] = eid
    return entities, norm_to_id


# --------------------------------------------------------------------------- #
# Recursive multi-level Leiden hierarchy (Appendix B.1)
# --------------------------------------------------------------------------- #
@dataclass
class _Community:
    id: int
    level: int
    parent: Optional[int]
    entity_ids: List[str] = field(default_factory=list)
    children: List[int] = field(default_factory=list)
    text_unit_ids: List[str] = field(default_factory=list)
    title: str = ""
    summary: str = ""


def _partition(subgraph: nx.Graph) -> Dict[str, int]:
    if subgraph.number_of_nodes() == 0:
        return {}
    if _HAVE_LEIDEN and subgraph.number_of_edges() > 0:
        try:
            return _community_detection_with_leiden(subgraph)
        except Exception:
            pass
    return _community_detection_with_networkx(subgraph)


def _hierarchical_communities(
    graph: nx.Graph, max_cluster_size: int, max_levels: int
) -> Dict[int, _Community]:
    """Partition the entity graph into a multi-level module hierarchy.

    The graph is partitioned recursively: a coarse module is subdivided until it
    is small enough or the depth limit is hit. Levels are then indexed bottom-up
    to match the paper's convention, where the entity base is H0, the finest
    modules sit just above it, and the coarsest modules occupy the top level HL.
    """
    communities: Dict[int, _Community] = {}
    counter = [0]

    def new_id() -> int:
        cid = counter[0]
        counter[0] += 1
        return cid

    def recurse(nodes: List[str], level: int, parent: Optional[int]) -> List[int]:
        part = _partition(graph.subgraph(nodes))
        groups: Dict[int, List[str]] = defaultdict(list)
        for n in nodes:
            groups[part.get(n, 0)].append(n)
        group_lists = list(groups.values())

        if len(group_lists) <= 1:  # cannot split further -> single leaf module
            cid = new_id()
            communities[cid] = _Community(id=cid, level=level, parent=parent, entity_ids=list(nodes))
            return [cid]

        ids: List[int] = []
        for grp in group_lists:
            cid = new_id()
            if len(grp) > max_cluster_size and level < max_levels:
                child_ids = recurse(grp, level + 1, cid)
                communities[cid] = _Community(id=cid, level=level, parent=parent, children=child_ids)
            else:
                communities[cid] = _Community(id=cid, level=level, parent=parent, entity_ids=list(grp))
            ids.append(cid)
        return ids

    if graph.number_of_nodes():
        recurse(list(graph.nodes()), 0, None)
    # Re-index levels bottom-up: finest modules low, coarsest at the top, so the
    # stored hierarchy reads H0 (entity base) .. HL (coarsest modules).
    if communities:
        max_level = max(c.level for c in communities.values())
        for c in communities.values():
            c.level = max_level - c.level
    return communities


def _propagate_text_units(communities: Dict[int, _Community], entities: Dict[str, _Entity]) -> None:
    """Fill each community's text_unit_ids bottom-up (post-order over the tree)."""

    def collect(cid: int) -> List[str]:
        comm = communities[cid]
        seen: List[str] = []
        for eid in comm.entity_ids:
            for tu in entities.get(eid, _Entity(id=eid, title="")).text_unit_ids:
                if tu not in seen:
                    seen.append(tu)
        for child in comm.children:
            for tu in collect(child):
                if tu not in seen:
                    seen.append(tu)
        comm.text_unit_ids = seen
        return seen

    roots = [cid for cid, c in communities.items() if c.parent is None]
    for cid in roots:
        collect(cid)


# --------------------------------------------------------------------------- #
# Community reports
# --------------------------------------------------------------------------- #
def _community_records(comm: _Community, entities: Dict[str, _Entity],
                       communities: Dict[int, _Community], rel_by_entity: Dict[str, List[str]]) -> str:
    lines: List[str] = []
    if comm.entity_ids:
        for eid in comm.entity_ids[:30]:
            ent = entities.get(eid)
            if ent:
                lines.append(f"- entity: {ent.title} ({ent.description[:160]})")
                for rel in rel_by_entity.get(eid, [])[:3]:
                    lines.append(f"  - {rel}")
    for child in comm.children[:30]:
        child_c = communities.get(child)
        if child_c and child_c.title:
            lines.append(f"- subtopic: {child_c.title}")
    return "\n".join(lines) if lines else "(empty community)"


def _extractive_report(records: str) -> Dict[str, str]:
    titles = [ln.split(":", 1)[-1].strip() for ln in records.splitlines() if ln.startswith("- ")]
    title = (titles[0][:60] if titles else "Community").split("(")[0].strip()
    summary = " ".join(t for t in titles[:6])[:480]
    return {"title": title or "Community", "summary": summary}


# --------------------------------------------------------------------------- #
# Causal gates: Algorithm 2 (Top-Down Hierarchical Pruning), Figure 9
# --------------------------------------------------------------------------- #
def _verify_gate(client: LLMClient, a_text: str, b_text: str) -> bool:
    prompt = CAUSAL_GATE_PROMPT.format(a_text=a_text, b_text=b_text)
    answer = (client.chat([{"role": "user", "content": prompt}]) or "").strip().lower()
    return answer.startswith("yes")


def _children_of(communities: Dict[int, _Community]) -> Dict[int, set]:
    children: Dict[int, set] = {cid: set(c.children) for cid, c in communities.items()}
    return children


def _build_causal_gates_top_down(
    communities: Dict[int, _Community],
    client: LLMClient,
    max_workers: int,
) -> Dict[int, set]:
    """Algorithm 2: construct undirected causal gates with top-down pruning.

    Iterates modules from the coarsest level (top) to the finest. At each level
    it verifies causal links between same-level peers, then verifies links to the
    next finer level while pruning a module's own children and the children of
    peers it is already gate-connected to.
    """
    text = {cid: (c.summary or c.title or f"Community {cid}") for cid, c in communities.items()}
    children = _children_of(communities)
    by_level: Dict[int, List[int]] = defaultdict(list)
    for cid, c in communities.items():
        by_level[c.level].append(cid)
    levels = sorted(by_level, reverse=True)  # coarsest (top) -> finest

    gates: set = set()  # frozenset({u, v})

    def verify_pairs(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        accepted: List[Tuple[int, int]] = []
        if not pairs:
            return accepted

        def work(pair):
            u, v = pair
            return pair, _verify_gate(client, text[u], text[v])

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for fut in as_completed([pool.submit(work, p) for p in pairs]):
                pair, ok = fut.result()
                if ok:
                    accepted.append(pair)
        return accepted

    for li, level in enumerate(levels):
        modules = by_level[level]

        # 1. Intra-layer verification (all unordered peer pairs at this level)
        intra_pairs = [(modules[i], modules[j])
                       for i in range(len(modules)) for j in range(i + 1, len(modules))]
        connected_peers: Dict[int, set] = defaultdict(set)
        for u, v in verify_pairs(intra_pairs):
            gates.add(frozenset((u, v)))
            connected_peers[u].add(v)
            connected_peers[v].add(u)

        # 2. Inter-layer look-ahead pruning to the next finer level
        if li + 1 < len(levels):
            lower = set(by_level[levels[li + 1]])
            inter_pairs: List[Tuple[int, int]] = []
            for u in modules:
                candidates = set(lower) - children[u]
                for v in connected_peers[u]:
                    candidates -= children.get(v, set())
                inter_pairs.extend((u, w) for w in candidates)
            for u, w in verify_pairs(inter_pairs):
                gates.add(frozenset((u, w)))

    causal_children: Dict[int, set] = defaultdict(set)
    for gate in gates:
        a, b = tuple(gate)
        causal_children[a].add(b)
        causal_children[b].add(a)  # undirected gate
    return causal_children


# --------------------------------------------------------------------------- #
# Main build
# --------------------------------------------------------------------------- #
def build_graph(
    source,
    out_root,
    *,
    model: str = "gpt-5-nano",
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    entity_types: str = DEFAULT_ENTITY_TYPES,
    max_cluster_size: int = 10,
    max_levels: int = 4,
    fuzzy_threshold: float = 0.92,
    embedding_threshold: float = 0.93,
    llm_reports: bool = True,
    build_causal: bool = True,
    max_workers: int = 8,
    cache: bool = True,
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
    client: Optional[LLMClient] = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Build the CausalRAG2 graph artifacts under ``out_root`` and return that path.

    Set ``llm_reports=False`` for cheaper extractive community summaries, or
    ``build_causal=False`` to skip causal-gate construction.
    """
    _load_env()
    out_root = Path(out_root)
    output_dir = out_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if client is None:
        client = LLMClient(model=model, api_key=api_key, base_url=base_url)

    # 1) chunk -------------------------------------------------------------- #
    documents = _read_documents(source)
    if not documents:
        raise ValueError(f"No usable text found in source: {source!r}")
    text_units: List[Dict] = []
    for doc_id, text in documents:
        for chunk in _chunk_text(text, chunk_size, chunk_overlap):
            text_units.append({"id": str(len(text_units)), "doc_id": doc_id, "text": chunk})
    log(f"[index] {len(documents)} document(s) -> {len(text_units)} chunk(s)")

    # 2) extract (Figure 8) ------------------------------------------------- #
    extract_cache = _ExtractCache(out_root / "extract_cache.jsonl", enabled=cache)

    def run_extract(tu: Dict) -> Tuple[str, Dict]:
        ckey = _ExtractCache.key(model, tu["text"])
        cached = extract_cache.get(ckey)
        if cached is not None:
            return tu["id"], cached
        parsed = _extract_chunk(client, tu["text"], entity_types)
        extract_cache.put(ckey, parsed)
        return tu["id"], parsed

    extractions: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for fut in as_completed([pool.submit(run_extract, tu) for tu in text_units]):
            tu_id, parsed = fut.result()
            extractions[tu_id] = parsed
    log(f"[index] extracted entities/relations for {len(extractions)} chunk(s)")

    # 3) two-stage entity canonicalization (B.1) ---------------------------- #
    mentions: List[Tuple[str, str, str]] = []
    raw_rels: List[Tuple[str, str, str, str]] = []  # (src_name, tgt_name, desc, tu_id)
    for tu in text_units:
        parsed = extractions.get(tu["id"], {})
        for ent in parsed.get("entities", []) or []:
            if isinstance(ent, dict) and ent.get("name"):
                mentions.append((ent["name"], ent.get("description", ""), tu["id"]))
        for rel in parsed.get("relationships", []) or []:
            if isinstance(rel, dict) and rel.get("source") and rel.get("target"):
                mentions.append((rel["source"], "", tu["id"]))
                mentions.append((rel["target"], "", tu["id"]))
                raw_rels.append((rel["source"], rel["target"],
                                 str(rel.get("description") or "related"), tu["id"]))

    if not mentions:
        raise RuntimeError("Extraction produced no entities; cannot build a graph.")
    entities, norm_to_id = _canonicalize_entities(mentions, fuzzy_threshold, embedding_threshold)
    log(f"[index] {len(mentions)} mentions -> {len(entities)} canonical entities")

    # 4) relationships (remapped to canonical entities) --------------------- #
    relationships: List[Dict] = []
    rel_by_entity: Dict[str, List[str]] = defaultdict(list)
    rel_seen: set = set()
    for src_name, tgt_name, desc, tu_id in raw_rels:
        sid = norm_to_id.get(_normalize_name(src_name))
        tid = norm_to_id.get(_normalize_name(tgt_name))
        if not sid or not tid or sid == tid:
            continue
        src_title, tgt_title = entities[sid].title, entities[tid].title
        relationships.append({"id": str(len(relationships)), "source": src_title,
                              "target": tgt_title, "description": desc, "text_unit_ids": [tu_id]})
        rel_by_entity[sid].append(f"{src_title} -> {tgt_title}: {desc}"[:200])
        rel_seen.add((sid, tid))
    log(f"[index] {len(relationships)} relationships")

    # 5) hierarchical communities (B.1) ------------------------------------- #
    graph = nx.Graph()
    graph.add_nodes_from(entities.keys())
    for sid, tid in rel_seen:
        w = graph.get_edge_data(sid, tid, default={}).get("weight", 0.0) + 1.0
        graph.add_edge(sid, tid, weight=w)
    communities = _hierarchical_communities(graph, max_cluster_size, max_levels)
    _propagate_text_units(communities, entities)
    n_levels = (max(c.level for c in communities.values()) + 1) if communities else 0
    log(f"[index] {len(communities)} modules across {n_levels} level(s)")

    # 6) community reports -------------------------------------------------- #
    def make_report(cid: int) -> Tuple[int, Dict[str, str]]:
        records = _community_records(communities[cid], entities, communities, rel_by_entity)
        if llm_reports:
            try:
                parsed = _parse_json(
                    client.chat([{"role": "user", "content": COMMUNITY_REPORT_PROMPT.format(records=records)}]),
                    default={})
                title = str(parsed.get("title") or "").strip()
                summary = str(parsed.get("summary") or "").strip()
                if title or summary:
                    return cid, {"title": title or f"Community {cid}", "summary": summary}
            except Exception:
                pass
        return cid, _extractive_report(records)

    # reports bottom-up (finest level first) so internal modules can name children
    report_levels: Dict[int, List[int]] = defaultdict(list)
    for cid, c in communities.items():
        report_levels[c.level].append(cid)
    for level in sorted(report_levels, reverse=True):
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for cid, rep in pool.map(make_report, report_levels[level]):
                communities[cid].title = rep["title"]
                communities[cid].summary = rep["summary"]
    log(f"[index] wrote {len(communities)} community reports")

    # 7) base parquet ------------------------------------------------------- #
    entities_df = pd.DataFrame([{"id": e.id, "title": e.title, "description": e.description,
                                 "text_unit_ids": e.text_unit_ids} for e in entities.values()])
    relationships_df = pd.DataFrame(relationships) if relationships else pd.DataFrame(
        columns=["id", "source", "target", "description", "text_unit_ids"])
    text_units_df = pd.DataFrame([{"id": tu["id"], "text": tu["text"]} for tu in text_units])
    communities_df = pd.DataFrame([{
        "community": c.id, "level": c.level, "title": c.title,
        "children": list(c.children), "entity_ids": list(c.entity_ids),
        "text_unit_ids": list(c.text_unit_ids),
    } for c in communities.values()]) if communities else pd.DataFrame(
        columns=["community", "level", "title", "children", "entity_ids", "text_unit_ids"])
    reports_df = pd.DataFrame([{"community": c.id, "title": c.title, "summary": c.summary}
                               for c in communities.values()]) if communities else pd.DataFrame(
        columns=["community", "title", "summary"])

    entities_df.to_parquet(output_dir / "entities.parquet", index=False)
    relationships_df.to_parquet(output_dir / "relationships.parquet", index=False)
    text_units_df.to_parquet(output_dir / "text_units.parquet", index=False)
    communities_df.to_parquet(output_dir / "communities.parquet", index=False)
    reports_df.to_parquet(output_dir / "community_reports.parquet", index=False)
    log(f"[index] wrote base parquet to {output_dir}")

    # 8) causal gates (Algorithm 2) -> output_causal/ ----------------------- #
    out_causal = out_root / "output_causal"
    if out_causal.exists():
        shutil.rmtree(out_causal)
    shutil.copytree(output_dir, out_causal)

    causal_children: Dict[int, set] = {}
    if build_causal and communities:
        log("[index] building causal gates (Algorithm 2: top-down pruning) ...")
        causal_children = _build_causal_gates_top_down(communities, client, max_workers)
    causal_df = communities_df.copy()
    causal_df["causal_children"] = causal_df["community"].apply(
        lambda c: sorted(causal_children.get(int(c), set())))
    causal_df["children"] = causal_df.apply(
        lambda row: list(dict.fromkeys(list(row.get("children") or []) + list(row["causal_children"]))),
        axis=1)
    causal_df.to_parquet(out_causal / "community_causal.parquet", index=False)
    n_gates = sum(len(v) for v in causal_children.values()) // 2
    log(f"[index] wrote {out_causal} (causal gates: {n_gates})")
    return out_root


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build a CausalRAG2 graph from raw text.")
    parser.add_argument("source", help="Path to a .txt/.jsonl file, a directory of .txt, or text.")
    parser.add_argument("out_root", help="Output graph directory (the graph root).")
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--max-cluster-size", type=int, default=10,
                        help="Modules larger than this are subdivided into a finer level.")
    parser.add_argument("--max-levels", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--extractive-reports", action="store_true",
                        help="Use extractive community summaries instead of LLM reports.")
    parser.add_argument("--no-causal", action="store_true", help="Skip causal gate construction.")
    args = parser.parse_args()

    build_graph(
        args.source, args.out_root, model=args.model, chunk_size=args.chunk_size,
        max_cluster_size=args.max_cluster_size, max_levels=args.max_levels,
        max_workers=args.max_workers, llm_reports=not args.extractive_reports,
        build_causal=not args.no_causal,
    )


if __name__ == "__main__":
    _main()
