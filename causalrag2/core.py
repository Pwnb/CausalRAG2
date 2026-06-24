from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .token_utils import coerce_usage, estimate_tokens
from .prompts import CAUSAL_RERANK_PROMPT, CT_CAUSAL_RERANK_PROMPT, FINAL_ANSWER_PROMPT

_EMBEDDERS: Dict[str, object] = {}
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOP_TOKENS = {
    "definition",
    "define",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "why",
    "how",
    "can",
    "could",
    "would",
    "should",
    "certain",
    "affect",
    "effect",
    "with",
    "about",
    "from",
    "into",
    "that",
    "this",
    "these",
    "those",
}
_RELATION_KEYWORDS = ("spouse", "husband", "wife", "married", "partner")


@dataclass
class Node:
    key: str
    node_type: str
    node_id: str
    level: Optional[int]
    title: str
    summary: str
    text_unit_ids: List[str]


@dataclass
class Edge:
    edge_id: str
    source: str
    target: str
    edge_type: str
    description: str
    direction: str
    text_unit_ids: List[str]


@dataclass
class GraphData:
    nodes: List[Node]
    node_index: Dict[str, int]
    edges: List[Edge]
    adjacency: Dict[str, List[int]]
    text_units: Dict[str, str]


@dataclass
class ContextItem:
    item_type: str
    original_id: str
    content: str
    score: float
    short_id: str = ""
    edge_ids: Optional[List[str]] = None


def _load_env() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip("'").strip('"')
        if key == "OPENAI_API_KEY":
            os.environ["OPENAI_API_KEY"] = value


def _to_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except Exception:
        return []


def _trim(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _coerce_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower()


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(_normalize_text(text))


def _lexical_scores(query: str, nodes: List[Node]) -> np.ndarray:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return np.zeros(len(nodes), dtype="float32")
    q_set = set(q_tokens)
    q_norm = _normalize_text(query)
    scores = np.zeros(len(nodes), dtype="float32")
    for idx, node in enumerate(nodes):
        text = f"{node.title} {node.summary}".strip()
        if not text:
            continue
        tokens = set(_tokenize(text))
        overlap = len(q_set.intersection(tokens))
        score = overlap / max(len(q_set), 1)
        title_norm = _normalize_text(node.title)
        if title_norm and title_norm in q_norm:
            score += 0.5
        scores[idx] = score
    return scores


def _tokenize_filtered(text: str, min_len: int = 3) -> List[str]:
    return [
        tok
        for tok in _tokenize(text)
        if len(tok) >= min_len and tok not in _STOP_TOKENS
    ]


def _score_text_units(
    query: str,
    text_units: Dict[str, str],
    max_items: int,
    min_score: float = 0.0,
    min_token_len: int = 3,
) -> List[Tuple[float, str]]:
    q_tokens = _tokenize_filtered(query, min_token_len)
    if not q_tokens:
        return []
    q_set = set(q_tokens)
    scored: List[Tuple[float, str]] = []
    for tid, text in text_units.items():
        if not text:
            continue
        t_tokens = set(_tokenize_filtered(text, min_token_len))
        if not t_tokens:
            continue
        overlap = len(q_set.intersection(t_tokens))
        if overlap <= 0:
            continue
        score = overlap / max(len(q_set), 1)
        if score < min_score:
            continue
        scored.append((score, str(tid)))
    scored.sort(key=lambda x: (-x[0], x[1]))
    if max_items > 0:
        return scored[:max_items]
    return scored


def _build_text_unit_to_nodes(nodes: List[Node], edges: List[Edge]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}

    def _add(tid: str, node_key: str) -> None:
        if not tid or not node_key:
            return
        bucket = mapping.setdefault(tid, [])
        if node_key not in bucket:
            bucket.append(node_key)

    for node in nodes:
        for tid in node.text_unit_ids:
            _add(str(tid), node.key)
    for edge in edges:
        for tid in edge.text_unit_ids:
            _add(str(tid), edge.source)
            _add(str(tid), edge.target)
    return mapping


def _seed_nodes_from_text_units(
    query: str,
    graph: GraphData,
    node_sims: Dict[str, float],
    max_text_units: int,
    max_nodes: int,
    min_score: float = 0.0,
    min_token_len: int = 3,
) -> Tuple[List[str], List[str]]:
    scored_units = _score_text_units(
        query,
        graph.text_units,
        max_text_units,
        min_score=min_score,
        min_token_len=min_token_len,
    )
    if not scored_units:
        return [], []
    tu_to_nodes = _build_text_unit_to_nodes(graph.nodes, graph.edges)
    candidates: List[Tuple[float, float, str, str]] = []
    for score, tid in scored_units:
        for node_key in tu_to_nodes.get(tid, []):
            candidates.append((score, node_sims.get(node_key, 0.0), node_key, tid))
    candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
    seeds: List[str] = []
    used = set()
    used_text_units: List[str] = []
    for _, _, node_key, tid in candidates:
        if node_key in used:
            continue
        seeds.append(node_key)
        used.add(node_key)
        if tid not in used_text_units:
            used_text_units.append(tid)
        if max_nodes > 0 and len(seeds) >= max_nodes:
            break
    return seeds, used_text_units


def _seed_communities_from_summary(
    nodes: List[Node],
    sims: np.ndarray,
    top_k: int,
    min_score: float,
) -> List[str]:
    candidates: List[Tuple[float, str]] = []
    for idx, node in enumerate(nodes):
        if node.node_type != "community":
            continue
        if not node.summary:
            continue
        score = float(sims[idx])
        if score < min_score:
            continue
        candidates.append((score, node.key))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    if top_k > 0:
        candidates = candidates[:top_k]
    return [key for _, key in candidates]


def _merge_seed_keys(*lists: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for items in lists:
        for key in items:
            if key in seen:
                continue
            merged.append(key)
            seen.add(key)
    return merged


def _seed_topk_by_type(
    query: str,
    nodes: List[Node],
    embeddings: np.ndarray,
    embed_model: str,
    seed_top_k: int,
    seed_mix_nodes: bool,
    seed_k_entities: int,
    seed_k_communities: int,
    seed_lexical_alpha: float,
    seed_mmr_lambda: float,
) -> Tuple[
    List[str],
    Dict[str, float],
    np.ndarray,
    List[int],
    List[int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    embedder = _load_embedder(embed_model)
    query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = np.dot(embeddings, query_vec)
    lex = _lexical_scores(query, nodes)
    alpha = min(max(float(seed_lexical_alpha), 0.0), 1.0)
    combined = alpha * sims + (1.0 - alpha) * lex
    node_sims = {nodes[i].key: float(combined[i]) for i in range(len(nodes))}

    entity_indices = [i for i, n in enumerate(nodes) if n.node_type == "entity"]
    community_indices = [i for i, n in enumerate(nodes) if n.node_type == "community"]

    if seed_mix_nodes:
        selected_entities = _mmr_select(
            entity_indices,
            max(0, seed_k_entities),
            combined,
            embeddings,
            seed_mmr_lambda,
        )
        selected_communities = _mmr_select(
            community_indices,
            max(0, seed_k_communities),
            combined,
            embeddings,
            seed_mmr_lambda,
        )
        selected = selected_entities + selected_communities
    else:
        selected = _mmr_select(
            list(range(len(nodes))),
            max(1, seed_top_k),
            combined,
            embeddings,
            seed_mmr_lambda,
        )
        selected_entities = [i for i in selected if nodes[i].node_type == "entity"]
        selected_communities = [i for i in selected if nodes[i].node_type == "community"]

    if not selected:
        ranked = sorted(range(len(nodes)), key=lambda i: combined[i], reverse=True)
        selected = ranked[: max(1, seed_top_k)]
        selected_entities = [i for i in selected if nodes[i].node_type == "entity"]
        selected_communities = [i for i in selected if nodes[i].node_type == "community"]

    seed_keys = [nodes[i].key for i in selected]
    return (
        seed_keys,
        node_sims,
        query_vec,
        selected_entities,
        selected_communities,
        sims,
        lex,
        combined,
    )


def _edge_base_id(edge_id: str) -> str:
    if edge_id.endswith(":rev"):
        return edge_id[: -len(":rev")]
    if edge_id.endswith(":up"):
        return edge_id[: -len(":up")]
    return edge_id


def _edge_content(edge: Edge, title_map: Dict[str, str], max_chars: int) -> str:
    source = title_map.get(edge.source, edge.source)
    target = title_map.get(edge.target, edge.target)
    desc = _trim(edge.description, max_chars)
    if desc:
        return f"{source} -> {target} ({edge.edge_type}): {desc}"
    return f"{source} -> {target} ({edge.edge_type})"


def _node_content(node: Node, max_chars: int, include_summary: bool) -> str:
    title = node.title or node.key
    summary = _trim(node.summary, max_chars)
    if include_summary and summary:
        return f"{title}. {summary}".strip()
    return title


def _edge_lexical_score(query_tokens: set, text: str, min_token_len: int) -> float:
    if not query_tokens:
        return 0.0
    tokens = set(_tokenize_filtered(text, min_token_len))
    if not tokens:
        return 0.0
    overlap = len(query_tokens.intersection(tokens))
    return overlap / max(len(query_tokens), 1)


def _score_edges(
    query: str,
    edges: List[Edge],
    node_sims: Dict[str, float],
    title_map: Dict[str, str],
    edge_lexical_alpha: float,
    min_token_len: int,
) -> Tuple[Dict[str, float], Dict[str, Edge], Dict[str, List[str]]]:
    alpha = min(max(float(edge_lexical_alpha), 0.0), 1.0)
    query_tokens = set(_tokenize_filtered(query, min_token_len))
    base_to_edge: Dict[str, Edge] = {}
    base_to_ids: Dict[str, List[str]] = {}
    for edge in edges:
        base_id = _edge_base_id(edge.edge_id)
        base_to_ids.setdefault(base_id, []).append(edge.edge_id)
        if base_id not in base_to_edge or edge.edge_id == base_id:
            base_to_edge[base_id] = edge

    scores: Dict[str, float] = {}
    for base_id, edge in base_to_edge.items():
        source_score = node_sims.get(edge.source, 0.0)
        target_score = node_sims.get(edge.target, 0.0)
        node_score = (source_score + target_score) / 2.0
        text = f"{title_map.get(edge.source, edge.source)} {title_map.get(edge.target, edge.target)} {edge.description}"
        lex_score = _edge_lexical_score(query_tokens, text, min_token_len)
        scores[base_id] = alpha * node_score + (1.0 - alpha) * lex_score
    return scores, base_to_edge, base_to_ids


def _select_seed_edges(
    edge_scores: Dict[str, float],
    base_to_edge: Dict[str, Edge],
    top_k: int,
    min_score: float,
) -> Tuple[List[str], List[str]]:
    if top_k <= 0:
        return [], []
    scored = [
        (score, base_id)
        for base_id, score in edge_scores.items()
        if score >= min_score and base_id in base_to_edge
    ]
    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = scored[:top_k]
    seed_edge_ids = [base_id for _, base_id in selected]
    seed_node_keys: List[str] = []
    seen = set()
    for _, base_id in selected:
        edge = base_to_edge[base_id]
        for node_key in (edge.source, edge.target):
            if node_key not in seen:
                seen.add(node_key)
                seed_node_keys.append(node_key)
    return seed_edge_ids, seed_node_keys


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _select_nodes_for_context(
    nodes: List[Node],
    node_sims: Dict[str, float],
    max_nodes: int,
    max_chars: int,
    include_community_summary: bool,
    force_node_keys: Iterable[str],
) -> List[ContextItem]:
    items: List[ContextItem] = []
    for node in nodes:
        item_type = "community" if node.node_type == "community" else "node"
        content = _node_content(node, max_chars, include_community_summary or item_type != "community")
        items.append(
            ContextItem(
                item_type=item_type,
                original_id=node.key,
                content=content,
                score=float(node_sims.get(node.key, 0.0)),
            )
        )
    items.sort(key=lambda x: x.score, reverse=True)
    forced = set(force_node_keys)
    selected: List[ContextItem] = []
    if forced:
        for item in items:
            if item.original_id in forced:
                selected.append(item)
    for item in items:
        if item in selected:
            continue
        selected.append(item)
        if max_nodes > 0 and len(selected) >= max_nodes:
            break
    if max_nodes > 0:
        return selected[:max_nodes]
    return selected


def _select_edges_for_context(
    edges: List[Edge],
    edge_scores: Dict[str, float],
    base_to_edge: Dict[str, Edge],
    base_to_ids: Dict[str, List[str]],
    title_map: Dict[str, str],
    max_edges: int,
    max_chars: int,
) -> List[ContextItem]:
    items: List[ContextItem] = []
    for base_id, edge in base_to_edge.items():
        score = float(edge_scores.get(base_id, 0.0))
        content = _edge_content(edge, title_map, max_chars)
        edge_ids = base_to_ids.get(base_id, [edge.edge_id])
        items.append(
            ContextItem(
                item_type="edge",
                original_id=base_id,
                content=content,
                score=score,
                edge_ids=edge_ids,
            )
        )
    items.sort(key=lambda x: x.score, reverse=True)
    if max_edges > 0:
        return items[:max_edges]
    return items


def _select_text_units_for_context(
    query: str,
    text_units: Dict[str, str],
    candidate_ids: Iterable[str],
    max_units: int,
    max_chars: int,
    min_score: float,
    min_token_len: int,
) -> List[ContextItem]:
    query_norm = _normalize_text(query)
    q_tokens = _tokenize_filtered(query, min_token_len)
    if not q_tokens:
        return []
    q_set = set(q_tokens)
    candidates: Dict[str, str] = {}
    for tid in candidate_ids:
        text = text_units.get(str(tid), "")
        if text:
            candidates[str(tid)] = text

    scored: List[Tuple[float, str]] = []
    token_hits: Dict[str, set] = {}
    for tid, text in candidates.items():
        t_tokens = set(_tokenize_filtered(text, min_token_len))
        if not t_tokens:
            continue
        hits = q_set.intersection(t_tokens)
        if not hits:
            continue
        score = len(hits) / max(len(q_set), 1)
        if score < min_score:
            continue
        scored.append((score, str(tid)))
        token_hits[str(tid)] = hits
    scored.sort(key=lambda x: (-x[0], x[1]))
    if not scored:
        return []

    if max_units > 0:
        selected: List[Tuple[float, str]] = []
        seen = set()
        for tok in sorted(q_tokens, key=len, reverse=True):
            if len(selected) >= max_units:
                break
            for score, tid in scored:
                if tid in seen:
                    continue
                if tok in token_hits.get(tid, set()):
                    selected.append((score, tid))
                    seen.add(tid)
                    break
        for score, tid in scored:
            if len(selected) >= max_units:
                break
            if tid in seen:
                continue
            selected.append((score, tid))
            seen.add(tid)
        scored = selected

    items: List[ContextItem] = []
    for score, tid in scored:
        raw = candidates.get(tid, "")
        content = _snippet_text(raw, query_norm, max_chars)
        if content:
            items.append(
                ContextItem(
                    item_type="text_unit",
                    original_id=str(tid),
                    content=content,
                    score=float(score),
                )
            )
    return items


def _assign_short_ids(items: List[ContextItem]) -> Dict[str, ContextItem]:
    counters = {"community": 0, "node": 0, "edge": 0, "text_unit": 0}
    mapping: Dict[str, ContextItem] = {}
    for item in items:
        counters[item.item_type] = counters.get(item.item_type, 0) + 1
        prefix = {"community": "C", "node": "N", "edge": "E", "text_unit": "T"}.get(
            item.item_type, "X"
        )
        item.short_id = f"{prefix}{counters[item.item_type]}"
        mapping[item.short_id] = item
    return mapping


def _build_context_table(items: List[ContextItem]) -> str:
    lines = []
    for item in items:
        content = item.content.strip()
        if content:
            lines.append(f"{item.short_id}: {content}")
    return "\\n".join(lines) if lines else "(none)"


def _trim_context_by_tokens(
    items: List[ContextItem],
    build_prompt,
    max_tokens: int,
    protected_ids: Optional[set] = None,
) -> List[ContextItem]:
    if max_tokens is None or max_tokens <= 0:
        return items
    protected_ids = protected_ids or set()
    trimmed = list(items)
    while trimmed:
        prompt = build_prompt(trimmed)
        if estimate_tokens(prompt) <= max_tokens:
            break
        # remove the lowest-score item that is not protected
        removable = [i for i in trimmed if i.short_id not in protected_ids]
        if not removable:
            break
        worst = min(removable, key=lambda x: x.score)
        trimmed.remove(worst)
    return trimmed


def _resolve_short_ids(ids: Iterable[str], mapping: Dict[str, ContextItem], max_items: int) -> List[ContextItem]:
    resolved: List[ContextItem] = []
    for raw in ids:
        key = str(raw).strip()
        candidates: List[str] = []
        if key:
            candidates.append(key)
        candidates.extend(re.findall(r"[CNTE]\\d+", key))
        picked = None
        for candidate in candidates:
            if candidate in mapping:
                picked = candidate
                break
        if not picked:
            continue
        if mapping[picked] in resolved:
            continue
        resolved.append(mapping[picked])
        if max_items > 0 and len(resolved) >= max_items:
            break
    return resolved


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_title(value: str) -> str:
    return (value or "").strip().lower()


def _resolve_output_dir(graph_root: Path) -> Path:
    if (graph_root / "output_causal").is_dir():
        return graph_root / "output_causal"
    if (graph_root / "output").is_dir():
        return graph_root / "output"
    if graph_root.name == "output_causal" or graph_root.name == "output":
        return graph_root
    raise RuntimeError(f"Graph output not found under {graph_root}")


def load_graph(graph_root: Path) -> GraphData:
    output_dir = _resolve_output_dir(graph_root)
    communities = pd.read_parquet(output_dir / "communities.parquet")
    community_reports = pd.read_parquet(output_dir / "community_reports.parquet")
    entities = pd.read_parquet(output_dir / "entities.parquet")
    relationships = pd.read_parquet(output_dir / "relationships.parquet")
    text_units = pd.read_parquet(output_dir / "text_units.parquet")
    causal_path = output_dir / "community_causal.parquet"
    if causal_path.exists():
        community_causal = pd.read_parquet(causal_path)
    else:
        community_causal = communities.copy()
        community_causal["causal_children"] = [[] for _ in range(len(community_causal))]

    report_map: Dict[int, Dict[str, str]] = {}
    for _, row in community_reports.iterrows():
        cid = _safe_int(row.get("community"))
        title = str(row.get("title") or "")
        summary = str(row.get("summary") or row.get("full_content") or "")
        report_map[cid] = {"title": title, "summary": summary}

    nodes: List[Node] = []
    node_index: Dict[str, int] = {}

    for _, row in communities.iterrows():
        cid = _safe_int(row.get("community"))
        level = _safe_int(row.get("level"), None)
        report = report_map.get(cid, {})
        title = str(report.get("title") or row.get("title") or f"Community {cid}")
        summary = str(report.get("summary") or "")
        key = f"C:{cid}"
        node = Node(
            key=key,
            node_type="community",
            node_id=str(cid),
            level=level,
            title=title,
            summary=summary,
            text_unit_ids=[str(x) for x in _to_list(row.get("text_unit_ids"))],
        )
        node_index[key] = len(nodes)
        nodes.append(node)

    for _, row in entities.iterrows():
        eid = str(row.get("id"))
        title = str(row.get("title") or "")
        summary = str(row.get("description") or "")
        key = f"E:{eid}"
        node = Node(
            key=key,
            node_type="entity",
            node_id=eid,
            level=None,
            title=title,
            summary=summary,
            text_unit_ids=[str(x) for x in _to_list(row.get("text_unit_ids"))],
        )
        node_index[key] = len(nodes)
        nodes.append(node)

    edges: List[Edge] = []
    adjacency: Dict[str, List[int]] = {n.key: [] for n in nodes}

    causal_pairs = set()
    for _, row in community_causal.iterrows():
        src = _safe_int(row.get("community"))
        src_key = f"C:{src}"
        if src_key not in node_index:
            continue
        for child in _to_list(row.get("causal_children")):
            tgt = _safe_int(child)
            tgt_key = f"C:{tgt}"
            if tgt_key not in node_index:
                continue
            edge_id = f"causal:{src}->{tgt}"
            edge = Edge(
                edge_id=edge_id,
                source=src_key,
                target=tgt_key,
                edge_type="causal",
                description="causal gate",
                direction="forward",
                text_unit_ids=[],
            )
            adjacency[src_key].append(len(edges))
            edges.append(edge)
            causal_pairs.add((src, tgt))

    for _, row in communities.iterrows():
        src = _safe_int(row.get("community"))
        src_key = f"C:{src}"
        if src_key not in node_index:
            continue
        for child in _to_list(row.get("children")):
            tgt = _safe_int(child)
            if (src, tgt) in causal_pairs:
                continue
            tgt_key = f"C:{tgt}"
            if tgt_key not in node_index:
                continue
            edge_id = f"struct:{src}->{tgt}"
            edge = Edge(
                edge_id=edge_id,
                source=src_key,
                target=tgt_key,
                edge_type="structural",
                description="community child",
                direction="down",
                text_unit_ids=[],
            )
            adjacency[src_key].append(len(edges))
            edges.append(edge)
            reverse_edge = Edge(
                edge_id=edge_id + ":up",
                source=tgt_key,
                target=src_key,
                edge_type="structural",
                description="community parent",
                direction="up",
                text_unit_ids=[],
            )
            adjacency[tgt_key].append(len(edges))
            edges.append(reverse_edge)

    entity_title_to_key: Dict[str, str] = {}
    for _, row in entities.iterrows():
        title = str(row.get("title") or "")
        if not title:
            continue
        entity_title_to_key[_normalize_title(title)] = f"E:{row.get('id')}"

    for _, row in communities.iterrows():
        src = _safe_int(row.get("community"))
        src_key = f"C:{src}"
        if src_key not in node_index:
            continue
        for entity_id in _to_list(row.get("entity_ids")):
            tgt_key = f"E:{entity_id}"
            if tgt_key not in node_index:
                continue
            edge_id = f"member:{src}->{entity_id}"
            edge = Edge(
                edge_id=edge_id,
                source=src_key,
                target=tgt_key,
                edge_type="membership",
                description="community membership",
                direction="down",
                text_unit_ids=[],
            )
            adjacency[src_key].append(len(edges))
            edges.append(edge)
            reverse_edge = Edge(
                edge_id=edge_id + ":up",
                source=tgt_key,
                target=src_key,
                edge_type="membership",
                description="community membership",
                direction="up",
                text_unit_ids=[],
            )
            adjacency[tgt_key].append(len(edges))
            edges.append(reverse_edge)

    for _, row in relationships.iterrows():
        source_title = _normalize_title(str(row.get("source") or ""))
        target_title = _normalize_title(str(row.get("target") or ""))
        src_key = entity_title_to_key.get(source_title)
        tgt_key = entity_title_to_key.get(target_title)
        if not src_key or not tgt_key:
            continue
        edge_id = f"rel:{row.get('id')}"
        text_unit_ids = [str(x) for x in _to_list(row.get("text_unit_ids"))]
        edge = Edge(
            edge_id=edge_id,
            source=src_key,
            target=tgt_key,
            edge_type="relationship",
            description=str(row.get("description") or ""),
            direction="both",
            text_unit_ids=text_unit_ids,
        )
        adjacency[src_key].append(len(edges))
        edges.append(edge)
        reverse_edge = Edge(
            edge_id=edge_id + ":rev",
            source=tgt_key,
            target=src_key,
            edge_type="relationship",
            description=str(row.get("description") or ""),
            direction="both",
            text_unit_ids=text_unit_ids,
        )
        adjacency[tgt_key].append(len(edges))
        edges.append(reverse_edge)

    text_unit_map = {str(row.get("id")): str(row.get("text") or "") for _, row in text_units.iterrows()}

    return GraphData(nodes=nodes, node_index=node_index, edges=edges, adjacency=adjacency, text_units=text_unit_map)


class _HashingEmbedder:
    """Deterministic, dependency-free fallback text embedder.

    Used when ``sentence-transformers`` is not installed so the pipeline still
    runs out of the box. It produces L2-normalized hashed bag-of-words vectors.
    Quality is lower than a real sentence encoder; install
    ``sentence-transformers`` for best retrieval results.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True,
               batch_size=64, show_progress_bar=False):
        import hashlib

        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for tok in _TOKEN_RE.findall((text or "").lower()):
                col = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self.dim
                vectors[row, col] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        return vectors


def _load_embedder(embed_model: str):
    if embed_model in _EMBEDDERS:
        return _EMBEDDERS[embed_model]
    try:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(embed_model)
    except Exception:
        # Dependency-free fallback so the pipeline runs without heavy extras.
        embedder = _HashingEmbedder()
    _EMBEDDERS[embed_model] = embedder
    return embedder


def _node_text(node: Node) -> str:
    if node.node_type == "community":
        if node.summary:
            return f"{node.title}. {node.summary}".strip()
        return node.title
    if node.summary:
        return f"{node.title}. {node.summary}".strip()
    return node.title


def _embed_cache_paths(output_dir: Path, embed_model: str) -> Tuple[Path, Path]:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", embed_model)
    cache_dir = output_dir / "causalrag2_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"embeddings_{safe_name}.npz", cache_dir / f"embeddings_{safe_name}.keys.json"


def load_or_build_embeddings(
    graph_root: Path,
    nodes: List[Node],
    embed_model: str,
    batch_size: int = 64,
) -> Tuple[np.ndarray, List[str]]:
    output_dir = _resolve_output_dir(graph_root)
    cache_path, keys_path = _embed_cache_paths(output_dir, embed_model)
    node_keys = [node.key for node in nodes]

    if cache_path.exists() and keys_path.exists():
        try:
            cached_keys = json.loads(keys_path.read_text(encoding="utf-8"))
            if cached_keys == node_keys:
                data = np.load(cache_path)
                embeddings = data["embeddings"]
                return embeddings, node_keys
        except Exception:
            pass

    embedder = _load_embedder(embed_model)
    texts = [_node_text(node) for node in nodes]
    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    ).astype("float32")
    np.savez(cache_path, embeddings=embeddings)
    keys_path.write_text(json.dumps(node_keys, ensure_ascii=True), encoding="utf-8")
    return embeddings, node_keys


def _mmr_select(
    candidates: List[int],
    k: int,
    sims: np.ndarray,
    embeddings: np.ndarray,
    lambda_val: float,
) -> List[int]:
    if k <= 0 or not candidates:
        return []
    selected: List[int] = []
    remaining = set(candidates)
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda idx: sims[idx])
            selected.append(best)
            remaining.remove(best)
            continue
        best_idx = None
        best_score = None
        for idx in remaining:
            max_sim = max(float(np.dot(embeddings[idx], embeddings[s])) for s in selected)
            score = lambda_val * sims[idx] - (1.0 - lambda_val) * max_sim
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def _edge_allowed(edge: Edge, enable_causal: bool, enable_upward: bool, enable_downward: bool) -> bool:
    if edge.edge_type == "causal" and not enable_causal:
        return False
    if edge.direction == "up" and not enable_upward:
        return False
    if edge.direction == "down" and not enable_downward:
        return False
    return True


def expand_subgraph(
    seed_keys: List[str],
    graph: GraphData,
    node_sims: Dict[str, float],
    max_hops: int,
    hop_mode: str,
    gain_min_delta: float,
    gain_min_score: float,
    gain_patience: int,
    enable_causal: bool,
    enable_upward: bool,
    enable_downward: bool,
    max_fanout_per_node: int,
    max_nodes: int,
    hop_decay: float,
    edge_weights: Dict[str, float],
) -> Tuple[set, set]:
    sub_nodes = set(seed_keys)
    frontier = list(seed_keys)
    low_gain_count = 0

    for hop in range(1, max_hops + 1):
        if not frontier:
            break
        candidates: Dict[str, Tuple[float, int]] = {}
        for node_key in frontier:
            edges = graph.adjacency.get(node_key, [])
            per_node: List[Tuple[float, str, int]] = []
            for edge_idx in edges:
                edge = graph.edges[edge_idx]
                if not _edge_allowed(edge, enable_causal, enable_upward, enable_downward):
                    continue
                target = edge.target
                if target in sub_nodes:
                    continue
                weight = edge_weights.get(edge.edge_type, 1.0)
                gain = node_sims.get(target, 0.0) * (hop_decay**hop) * weight
                if gain_min_score > 0 and gain < gain_min_score:
                    continue
                per_node.append((gain, target, edge_idx))
            per_node.sort(key=lambda x: x[0], reverse=True)
            for gain, target, edge_idx in per_node[: max(0, max_fanout_per_node)]:
                prev = candidates.get(target)
                if prev is None or gain > prev[0]:
                    candidates[target] = (gain, edge_idx)

        if not candidates:
            break

        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)
        newly_added = []
        gains = []
        for target, (gain, edge_idx) in sorted_candidates:
            if len(sub_nodes) >= max_nodes:
                break
            sub_nodes.add(target)
            newly_added.append(target)
            gains.append(gain)
        if hop_mode == "gain":
            avg_gain = sum(gains) / max(len(gains), 1)
            if avg_gain < gain_min_delta:
                low_gain_count += 1
            else:
                low_gain_count = 0
            if low_gain_count >= gain_patience:
                break
        frontier = newly_added

    sub_edges = set()
    for idx, edge in enumerate(graph.edges):
        if edge.source in sub_nodes and edge.target in sub_nodes:
            if _edge_allowed(edge, enable_causal, enable_upward, enable_downward):
                sub_edges.add(idx)
    return sub_nodes, sub_edges


def _prioritize_items(items: List[ContextItem]) -> List[ContextItem]:
    type_rank = {"text_unit": 0, "community": 1, "node": 1, "edge": 2}
    return sorted(items, key=lambda x: (type_rank.get(x.item_type, 3), -x.score))


def _normalize_with_map(text: str) -> Tuple[str, List[int]]:
    norm_chars: List[str] = []
    idx_map: List[int] = []
    for idx, ch in enumerate(text):
        normalized = unicodedata.normalize("NFKD", ch)
        ascii_chars = normalized.encode("ascii", "ignore").decode("ascii")
        if not ascii_chars:
            continue
        for c in ascii_chars:
            norm_chars.append(c.lower())
            idx_map.append(idx)
    return "".join(norm_chars), idx_map


def _snippet_text(text: str, query_norm: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text.strip()
    norm_text, idx_map = _normalize_with_map(text)
    if not norm_text:
        return _trim(text, max_chars)
    if not query_norm:
        return _trim(text, max_chars)

    tokens = _tokenize_filtered(query_norm, 3)
    if not tokens:
        tokens = _tokenize(query_norm)
    tokens = [t for t in tokens if t not in {"the", "and", "or"}]
    if tokens:
        seen = set()
        deduped = []
        for tok in tokens:
            if tok in seen:
                continue
            seen.add(tok)
            deduped.append(tok)
        tokens = deduped

    priority_tokens: List[str] = []
    if any(tok in tokens for tok in ("spouse", "husband", "wife")):
        for tok in ("spouse", "husband", "wife", "married"):
            if tok not in tokens:
                tokens.append(tok)
            if tok not in priority_tokens:
                priority_tokens.append(tok)

    occurrences: List[Tuple[int, str]] = []
    priority_positions: List[int] = []
    for tok in tokens:
        start = 0
        while True:
            pos = norm_text.find(tok, start)
            if pos == -1:
                break
            occurrences.append((pos, tok))
            if tok in priority_tokens:
                priority_positions.append(pos)
            start = pos + 1

    idx = -1
    if priority_positions:
        idx = max(priority_positions)
    elif query_norm:
        idx = norm_text.find(query_norm)
    if idx == -1 and tokens:
        tokens_sorted = sorted(tokens, key=len, reverse=True)
        for tok in tokens_sorted:
            pos = norm_text.find(tok)
            if pos != -1:
                idx = pos
                break
    if idx == -1 and occurrences:
        idx = occurrences[0][0]
    if idx == -1:
        return _trim(text, max_chars)
    start_norm = max(0, idx - max_chars // 3)
    end_norm = min(len(norm_text), start_norm + max_chars)
    start_orig = idx_map[start_norm]
    end_orig = idx_map[end_norm - 1] + 1
    snippet = text[start_orig:end_orig].strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "..."
    return snippet


def _relation_keywords(query: str) -> List[str]:
    if not query:
        return []
    tokens = set(_tokenize_filtered(query, 3))
    hits = [kw for kw in _RELATION_KEYWORDS if kw in tokens]
    if hits:
        return list(_RELATION_KEYWORDS)
    return []


def _pick_relation_text_unit(
    items: List[ContextItem],
    relation_keywords: List[str],
) -> Optional[ContextItem]:
    if not items:
        return None
    if relation_keywords:
        for item in items:
            content_norm = _normalize_text(item.content)
            if any(kw in content_norm for kw in relation_keywords):
                return item
    return items[0]


def _response_text(resp) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text
    output = getattr(resp, "output", None)
    if output:
        parts: List[str] = []
        for item in output:
            if isinstance(item, dict):
                content = item.get("content")
            else:
                content = getattr(item, "content", None)
            if not content:
                continue
            for part in content:
                if isinstance(part, dict):
                    part_text = part.get("text") or part.get("refusal")
                else:
                    part_text = getattr(part, "text", None) or getattr(part, "refusal", None)
                if part_text:
                    parts.append(part_text)
        joined = "".join(parts).strip()
        if joined:
            return joined
    return ""


def _openai_generate(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    temperature: Optional[float],
):
    _load_env()
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.openai.com/v1")
    merged_prompt = f"{system_prompt}\n\n{user_prompt}".strip()
    try:
        params = {"model": model, "input": merged_prompt, "max_output_tokens": max_tokens}
        if temperature is not None:
            if not str(model).startswith("gpt-5") or float(temperature) == 1.0:
                params["temperature"] = float(temperature)
        resp = client.responses.create(**params)
        text = _response_text(resp)
        if text:
            usage = coerce_usage(getattr(resp, "usage", None))
            return text, usage
    except Exception:
        pass
    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": max_tokens,
    }
    if temperature is not None:
        if not str(model).startswith("gpt-5") or float(temperature) == 1.0:
            params["temperature"] = float(temperature)
    resp = client.chat.completions.create(**params)
    message = resp.choices[0].message
    text = message.content or ""
    if not text:
        text = getattr(message, "refusal", "") or ""
    usage = coerce_usage(getattr(resp, "usage", None))
    return text, usage


def _extract_json(text: str) -> Dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return {}


def _usage_add(usage_total: Dict[str, int], usage: Dict[str, int]) -> Dict[str, int]:
    for key in ["input_tokens", "output_tokens", "cached_input_tokens"]:
        usage_total[key] = usage_total.get(key, 0) + int(usage.get(key, 0) or 0)
    return usage_total


def _estimate_usage(prompt: str, response: str) -> Dict[str, int]:
    return {
        "input_tokens": estimate_tokens(prompt),
        "output_tokens": estimate_tokens(response),
        "cached_input_tokens": 0,
    }


def run_single(
    example: Dict,
    graph_root: str,
    seed_top_k: int = 6,
    seed_mix_nodes: bool = True,
    seed_k_entities: int = 3,
    seed_k_communities: int = 3,
    seed_mmr_lambda: float = 0.7,
    seed_lexical_alpha: float = 0.7,
    seed_edge_top_k: int = 8,
    seed_edge_min_score: float = 0.0,
    seed_edge_lexical_alpha: float = 0.7,
    seed_edge_min_token_len: int = 3,
    seed_text_units_top_k: int = 6,
    seed_text_units_max_nodes: int = 8,
    seed_text_units_min_score: float = 0.2,
    seed_text_units_min_token_len: int = 3,
    seed_community_summary_top_k: int = 4,
    seed_community_summary_min_score: float = 0.35,
    hop_mode: str = "fixed",
    max_hops: int = 5,
    gain_min_delta: float = 0.02,
    gain_min_score: float = 0.0,
    gain_patience: int = 2,
    enable_causal: bool = True,
    enable_upward: bool = True,
    enable_downward: bool = True,
    max_fanout_per_node: int = 5,
    max_extended_nodes: Optional[int] = None,
    hop_decay: float = 0.7,
    edge_weight_causal: float = 1.2,
    edge_weight_structural: float = 1.0,
    edge_weight_membership: float = 1.0,
    edge_weight_relationship: float = 0.8,
    path_max_len: int = 5,
    path_beam_width: int = 10,
    path_candidates: int = 20,
    report_mode: str = "counterfactual",
    counterfactual: Optional[bool] = None,
    report_max_words: int = 350,
    max_prompt_nodes_causal_llm: Optional[int] = None,
    max_prompt_edges_causal_llm: Optional[int] = None,
    max_each_node_chars_causal_llm: Optional[int] = None,
    max_each_edge_chars_causal_llm: Optional[int] = None,
    max_each_text_unit_chars_causal_llm: Optional[int] = None,
    causal_llm_mode: Optional[str] = None,
    causal_llm_return_p_answer: bool = True,
    causal_precise_max_items: int = 10,
    causal_ct_precise_max_items: int = 10,
    causal_p_answer_max_words: int = 80,
    causal_prompt_max_tokens: int = 0,
    context_include_community_summary: bool = True,
    context_include_text_units: bool = True,
    context_text_units_scope: str = "subgraph",
    context_text_units_top_k: Optional[int] = None,
    context_text_units_min_score: Optional[float] = None,
    context_text_units_min_token_len: Optional[int] = None,
    answer_use_p_answer: bool = True,
    answer_context_max_items: int = 0,
    answer_prompt_max_tokens: int = 0,
    force_relation_text_unit: bool = True,
    retrieved_context_mode: str = "recall",
    retrieved_context_max_nodes: Optional[int] = None,
    retrieved_context_max_edges: Optional[int] = None,
    retrieved_context_max_text_units: Optional[int] = None,
    retrieved_context_max_items: int = 0,
    evidence_text_units: int = 5,
    evidence_max_chars: int = 400,
    define_include_evidence: bool = True,
    dedupe_bidirectional_edges: bool = True,
    include_evidence_in_context: bool = False,
    save_evidence: bool = False,
    save_query_meta: bool = True,
    k_bottom: Optional[int] = None,
    k_communities: Optional[int] = None,
    causal_llm_max_output_tokens: Optional[int] = None,
    answer_llm_max_output_tokens: Optional[int] = None,
    debug_save_prompt: bool = False,
    debug_prompt_chars: int = 0,
    debug_response_chars: int = 0,
    embed_model: str = "all-MiniLM-L6-v2",
    model: str = "gpt-5-nano",
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_nodes: Optional[int] = None,
    max_prompt_nodes: Optional[int] = None,
    max_prompt_edges: Optional[int] = None,
    max_node_chars: Optional[int] = None,
    max_edge_chars: Optional[int] = None,
    define_max_tokens: Optional[int] = None,
    answer_max_tokens: Optional[int] = None,
    **_,
) -> Dict:
    question = example.get("gold_question") or example.get("question") or ""
    if max_extended_nodes is None:
        max_extended_nodes = max_nodes if max_nodes is not None else 200
    if max_prompt_nodes_causal_llm is None:
        max_prompt_nodes_causal_llm = max_prompt_nodes if max_prompt_nodes is not None else 60
    if max_prompt_edges_causal_llm is None:
        max_prompt_edges_causal_llm = max_prompt_edges if max_prompt_edges is not None else 120
    if max_each_node_chars_causal_llm is None:
        max_each_node_chars_causal_llm = max_node_chars if max_node_chars is not None else 240
    if max_each_edge_chars_causal_llm is None:
        max_each_edge_chars_causal_llm = max_edge_chars if max_edge_chars is not None else 200
    if max_each_text_unit_chars_causal_llm is None:
        max_each_text_unit_chars_causal_llm = evidence_max_chars if evidence_max_chars is not None else 400
    if context_text_units_top_k is None:
        context_text_units_top_k = evidence_text_units
    if context_text_units_min_score is None:
        context_text_units_min_score = seed_text_units_min_score
    if context_text_units_min_token_len is None:
        context_text_units_min_token_len = seed_text_units_min_token_len
    if include_evidence_in_context:
        context_include_text_units = True
    if causal_llm_mode is None:
        mode_from_report = str(report_mode or "").lower()
        if counterfactual is not None:
            causal_llm_mode = "ct" if counterfactual else "causal"
        elif mode_from_report in {"counterfactual", "cf", "ct", "cf_mode", "cf_causal", "cf_define", "counter"}:
            causal_llm_mode = "ct"
        else:
            causal_llm_mode = "causal"
    causal_llm_mode = str(causal_llm_mode or "ct").lower()
    if not context_include_text_units:
        context_text_units_top_k = 0
    if causal_llm_max_output_tokens is None:
        if define_max_tokens is not None:
            causal_llm_max_output_tokens = define_max_tokens
        else:
            causal_llm_max_output_tokens = max_tokens
    if answer_llm_max_output_tokens is None:
        if answer_max_tokens is not None:
            answer_llm_max_output_tokens = answer_max_tokens
        else:
            answer_llm_max_output_tokens = max_tokens
    graph_root_path = Path(graph_root)
    graph = load_graph(graph_root_path)

    embeddings, _ = load_or_build_embeddings(graph_root_path, graph.nodes, embed_model)
    if k_bottom is not None:
        seed_k_entities = int(k_bottom)
    if k_communities is not None:
        seed_k_communities = int(k_communities)

    (
        seed_keys,
        node_sims,
        _,
        seed_entity_indices,
        seed_community_indices,
        seed_sims,
        seed_lexical,
        seed_combined,
    ) = _seed_topk_by_type(
        question,
        graph.nodes,
        embeddings,
        embed_model,
        seed_top_k,
        seed_mix_nodes,
        seed_k_entities,
        seed_k_communities,
        seed_lexical_alpha,
        seed_mmr_lambda,
    )
    title_map = {node.key: node.title or node.key for node in graph.nodes}
    edge_scores_all, base_to_edge_all, _ = _score_edges(
        question,
        graph.edges,
        node_sims,
        title_map,
        seed_edge_lexical_alpha,
        seed_edge_min_token_len,
    )
    seed_edge_ids, seed_edge_keys = _select_seed_edges(
        edge_scores_all,
        base_to_edge_all,
        seed_edge_top_k,
        seed_edge_min_score,
    )
    seed_text_unit_ids: List[str] = []
    community_seed_keys = _seed_communities_from_summary(
        graph.nodes,
        seed_sims,
        seed_community_summary_top_k,
        seed_community_summary_min_score,
    )
    text_unit_seed_keys, seed_text_unit_ids = _seed_nodes_from_text_units(
        question,
        graph,
        node_sims,
        seed_text_units_top_k,
        seed_text_units_max_nodes,
        min_score=seed_text_units_min_score,
        min_token_len=seed_text_units_min_token_len,
    )
    seed_keys = _merge_seed_keys(seed_keys, community_seed_keys, text_unit_seed_keys, seed_edge_keys)
    if max_extended_nodes is not None and max_extended_nodes > 0 and len(seed_keys) > max_extended_nodes:
        seed_keys = seed_keys[:max_extended_nodes]

    edge_weights = {
        "causal": edge_weight_causal,
        "structural": edge_weight_structural,
        "membership": edge_weight_membership,
        "relationship": edge_weight_relationship,
    }

    sub_nodes, sub_edges = expand_subgraph(
        seed_keys,
        graph,
        node_sims,
        max_hops,
        hop_mode,
        gain_min_delta,
        gain_min_score,
        gain_patience,
        enable_causal,
        enable_upward,
        enable_downward,
        max_fanout_per_node,
        max_extended_nodes,
        hop_decay,
        edge_weights,
    )

    sub_nodes_list = [graph.nodes[graph.node_index[k]] for k in sub_nodes if k in graph.node_index]
    sub_edges_list = [graph.edges[idx] for idx in sub_edges]

    usage_total: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
    estimated = False
    debug_info: Dict[str, str] = {}
    causal_prompt = ""
    causal_response = ""
    causal_json: Dict[str, object] = {}
    precise_ids: List[str] = []
    ct_precise_ids: List[str] = []
    p_answer = ""

    sub_edge_base_to_edge: Dict[str, Edge] = {}
    sub_edge_base_to_ids: Dict[str, List[str]] = {}
    for edge in sub_edges_list:
        base_id = _edge_base_id(edge.edge_id)
        sub_edge_base_to_ids.setdefault(base_id, []).append(edge.edge_id)
        if base_id not in sub_edge_base_to_edge or edge.edge_id == base_id:
            sub_edge_base_to_edge[base_id] = edge
    sub_edge_scores = {base_id: edge_scores_all.get(base_id, 0.0) for base_id in sub_edge_base_to_edge}

    scope_norm = str(context_text_units_scope or "subgraph").lower()
    if scope_norm in {"global", "all"}:
        candidate_text_unit_ids = list(graph.text_units.keys())
    else:
        candidate_text_unit_ids: List[str] = []
        for node in sub_nodes_list:
            candidate_text_unit_ids.extend(node.text_unit_ids)
        for edge in sub_edges_list:
            candidate_text_unit_ids.extend(edge.text_unit_ids)
        candidate_text_unit_ids.extend(seed_text_unit_ids)
        candidate_text_unit_ids = _dedupe_keep_order(candidate_text_unit_ids)
        if not candidate_text_unit_ids and graph.text_units:
            candidate_text_unit_ids = list(graph.text_units.keys())

    recall_nodes_limit = 0 if retrieved_context_max_nodes is None else int(retrieved_context_max_nodes)
    recall_edges_limit = 0 if retrieved_context_max_edges is None else int(retrieved_context_max_edges)
    recall_text_limit = (
        int(retrieved_context_max_text_units)
        if retrieved_context_max_text_units is not None
        else int(context_text_units_top_k or 0)
    )

    recall_node_items = _select_nodes_for_context(
        sub_nodes_list,
        node_sims,
        recall_nodes_limit,
        max_each_node_chars_causal_llm,
        context_include_community_summary,
        community_seed_keys,
    )
    recall_edge_items = _select_edges_for_context(
        sub_edges_list,
        sub_edge_scores,
        sub_edge_base_to_edge,
        sub_edge_base_to_ids,
        title_map,
        recall_edges_limit,
        max_each_edge_chars_causal_llm,
    )
    recall_text_items = []
    if recall_text_limit != 0:
        recall_text_items = _select_text_units_for_context(
            question,
            graph.text_units,
            candidate_text_unit_ids,
            recall_text_limit,
            max_each_text_unit_chars_causal_llm,
            float(context_text_units_min_score or 0.0),
            int(context_text_units_min_token_len or 3),
        )

    causal_node_items = _select_nodes_for_context(
        sub_nodes_list,
        node_sims,
        max_prompt_nodes_causal_llm or 0,
        max_each_node_chars_causal_llm,
        context_include_community_summary,
        community_seed_keys,
    )
    causal_edge_items = _select_edges_for_context(
        sub_edges_list,
        sub_edge_scores,
        sub_edge_base_to_edge,
        sub_edge_base_to_ids,
        title_map,
        max_prompt_edges_causal_llm or 0,
        max_each_edge_chars_causal_llm,
    )
    if seed_edge_ids:
        edge_lookup = {item.original_id: item for item in causal_edge_items}
        for base_id in seed_edge_ids:
            if base_id in edge_lookup:
                continue
            edge = sub_edge_base_to_edge.get(base_id)
            if edge is None:
                continue
            edge_lookup[base_id] = ContextItem(
                item_type="edge",
                original_id=base_id,
                content=_edge_content(edge, title_map, max_each_edge_chars_causal_llm),
                score=float(sub_edge_scores.get(base_id, 0.0)),
                edge_ids=sub_edge_base_to_ids.get(base_id, [edge.edge_id]),
            )
        causal_edge_items = sorted(edge_lookup.values(), key=lambda x: x.score, reverse=True)
        if max_prompt_edges_causal_llm and max_prompt_edges_causal_llm > 0:
            causal_edge_items = causal_edge_items[: max_prompt_edges_causal_llm]

    causal_text_items = []
    if context_text_units_top_k and int(context_text_units_top_k) > 0:
        causal_text_items = _select_text_units_for_context(
            question,
            graph.text_units,
            candidate_text_unit_ids,
            int(context_text_units_top_k),
            max_each_text_unit_chars_causal_llm,
            float(context_text_units_min_score or 0.0),
            int(context_text_units_min_token_len or 3),
        )

    def _split_nodes(items: List[ContextItem]) -> Tuple[List[ContextItem], List[ContextItem]]:
        communities = [i for i in items if i.item_type == "community"]
        nodes = [i for i in items if i.item_type != "community"]
        return communities, nodes

    recall_communities, recall_entities = _split_nodes(recall_node_items)
    recall_items = recall_communities + recall_entities + recall_edge_items + recall_text_items
    if retrieved_context_max_items and retrieved_context_max_items > 0:
        recall_items = sorted(recall_items, key=lambda x: x.score, reverse=True)[: retrieved_context_max_items]

    causal_communities, causal_entities = _split_nodes(causal_node_items)
    causal_items = causal_communities + causal_entities + causal_edge_items + causal_text_items
    causal_items = _prioritize_items(causal_items)
    _assign_short_ids(causal_items)
    protected_ids = {item.short_id for item in causal_items if item.original_id in set(community_seed_keys)}

    def _build_prompt_for_items(items: List[ContextItem]) -> str:
        context_table = _build_context_table(items)
        template = CT_CAUSAL_RERANK_PROMPT if causal_llm_mode == "ct" else CAUSAL_RERANK_PROMPT
        max_precise = int(causal_precise_max_items) if causal_precise_max_items and causal_precise_max_items > 0 else len(items)
        max_ct_precise = (
            int(causal_ct_precise_max_items) if causal_ct_precise_max_items and causal_ct_precise_max_items > 0 else len(items)
        )
        max_answer_words = int(causal_p_answer_max_words) if causal_p_answer_max_words and causal_p_answer_max_words > 0 else 80
        prompt = template.format(
            query=question,
            context_table=context_table,
            max_precise_items=max(1, max_precise),
            max_ct_precise_items=max(1, max_ct_precise),
            max_answer_words=max(1, max_answer_words),
        )
        if not causal_llm_return_p_answer:
            prompt = prompt + "\n\nNote: Set p_answer to an empty string."
        return prompt

    if causal_prompt_max_tokens and causal_prompt_max_tokens > 0:
        causal_items = _trim_context_by_tokens(
            causal_items,
            _build_prompt_for_items,
            causal_prompt_max_tokens,
            protected_ids,
        )
        _assign_short_ids(causal_items)

    causal_prompt = _build_prompt_for_items(causal_items)
    if debug_save_prompt and (debug_prompt_chars != 0):
        preview = causal_prompt if debug_prompt_chars < 0 else causal_prompt[:debug_prompt_chars]
        debug_info["llm_define_prompt"] = preview
    usage = {}
    define_error = ""
    try:
        causal_response, usage = _openai_generate(
            system_prompt="You are a careful analyst.",
            user_prompt=causal_prompt,
            model=model,
            max_tokens=causal_llm_max_output_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        define_error = str(exc)
    if define_error:
        debug_info["llm_define_error"] = define_error
    if debug_save_prompt and (debug_response_chars != 0):
        preview = causal_response if debug_response_chars < 0 else causal_response[:debug_response_chars]
        debug_info["llm_define_response"] = preview
    if not usage and not define_error:
        usage = _estimate_usage(causal_prompt, causal_response)
        estimated = True
    usage_total = _usage_add(usage_total, usage)

    parsed = _extract_json(causal_response)
    if isinstance(parsed, dict):
        causal_json = parsed
        precise_ids = _coerce_list(parsed.get("precise"))
        ct_precise_ids = _coerce_list(parsed.get("ct_precise"))
        p_answer = str(parsed.get("p_answer") or "").strip()
    if not causal_llm_return_p_answer:
        p_answer = ""
    if causal_precise_max_items and causal_precise_max_items > 0:
        precise_ids = precise_ids[: causal_precise_max_items]
    if causal_ct_precise_max_items and causal_ct_precise_max_items > 0:
        ct_precise_ids = ct_precise_ids[: causal_ct_precise_max_items]

    id_map = {item.short_id: item for item in causal_items}
    precise_items = _resolve_short_ids(precise_ids, id_map, int(causal_precise_max_items))
    ct_precise_items = _resolve_short_ids(ct_precise_ids, id_map, int(causal_ct_precise_max_items))
    if not precise_items:
        precise_items = _prioritize_items(causal_items)
        if causal_precise_max_items and causal_precise_max_items > 0:
            precise_items = precise_items[: causal_precise_max_items]
    if force_relation_text_unit:
        relation_keywords = _relation_keywords(question)
        has_text_unit = any(item.item_type == "text_unit" for item in precise_items)
        if relation_keywords and not has_text_unit:
            candidate = _pick_relation_text_unit(causal_text_items, relation_keywords)
            if candidate and candidate not in precise_items:
                precise_items = [candidate] + precise_items
                if causal_precise_max_items and causal_precise_max_items > 0:
                    precise_items = precise_items[: causal_precise_max_items]
                if candidate.short_id:
                    precise_ids = [candidate.short_id] + [pid for pid in precise_ids if pid != candidate.short_id]
                    if causal_precise_max_items and causal_precise_max_items > 0:
                        precise_ids = precise_ids[: causal_precise_max_items]
                    if isinstance(causal_json, dict):
                        causal_json["precise"] = list(precise_ids)

    answer_items = list(precise_items)
    if answer_context_max_items and answer_context_max_items > 0:
        answer_items = answer_items[: answer_context_max_items]
    answer_context_lines = [item.content for item in answer_items if item.content]
    report_context = "\n".join(f"- {line}" for line in answer_context_lines) if answer_context_lines else "(none)"
    draft_answer = p_answer if answer_use_p_answer and p_answer else "(none)"

    final_prompt = FINAL_ANSWER_PROMPT.format(
        report_context=report_context,
        draft_answer=draft_answer,
        query=question,
    )
    if answer_prompt_max_tokens and answer_prompt_max_tokens > 0:
        while answer_context_lines:
            if estimate_tokens(final_prompt) <= answer_prompt_max_tokens:
                break
            answer_context_lines = answer_context_lines[:-1]
            report_context = "\n".join(f"- {line}" for line in answer_context_lines) if answer_context_lines else "(none)"
            final_prompt = FINAL_ANSWER_PROMPT.format(
                report_context=report_context,
                draft_answer=draft_answer,
                query=question,
            )

    if debug_save_prompt and (debug_prompt_chars != 0):
        preview = final_prompt if debug_prompt_chars < 0 else final_prompt[:debug_prompt_chars]
        debug_info["llm_answer_prompt"] = preview
    answer_text = ""
    usage = {}
    answer_error = ""
    try:
        answer_text, usage = _openai_generate(
            system_prompt="You are a helpful assistant.",
            user_prompt=final_prompt,
            model=model,
            max_tokens=answer_llm_max_output_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        answer_error = str(exc)
    if answer_error:
        debug_info["llm_answer_error"] = answer_error
    if debug_save_prompt and (debug_response_chars != 0):
        preview = answer_text if debug_response_chars < 0 else answer_text[:debug_response_chars]
        debug_info["llm_answer_response"] = preview
    if not usage and not answer_error:
        usage = _estimate_usage(final_prompt, answer_text)
        estimated = True
    usage_total = _usage_add(usage_total, usage)
    if not answer_text.strip():
        answer_text = "Insufficient context."

    retrieved_context_items = recall_items
    if str(retrieved_context_mode or "").lower() in {"precise", "rerank", "causal", "refined"}:
        retrieved_context_items = precise_items
    retrieved_context = [item.content for item in retrieved_context_items if item.content]

    if estimated:
        usage_total["estimated"] = True

    def _node_meta(node: Node, cosine: float, lex: float, combined: float) -> Dict[str, object]:
        return {
            "id": node.key,
            "type": node.node_type,
            "level": node.level,
            "title": node.title,
            "summary": node.summary,
            "cosine": round(float(cosine), 6),
            "lexical": round(float(lex), 6),
            "combined": round(float(combined), 6),
        }

    seed_entities = []
    seed_communities = []
    for key in seed_keys:
        idx = graph.node_index.get(key)
        if idx is None:
            continue
        node = graph.nodes[idx]
        meta = _node_meta(node, seed_sims[idx], seed_lexical[idx], seed_combined[idx])
        if node.node_type == "community":
            seed_communities.append(meta)
        else:
            seed_entities.append(meta)

    subgraph_nodes_meta = []
    for node in sub_nodes_list:
        idx = graph.node_index.get(node.key)
        if idx is None:
            continue
        subgraph_nodes_meta.append(
            _node_meta(
                node,
                seed_sims[idx],
                seed_lexical[idx],
                seed_combined[idx],
            )
        )
    subgraph_edges_meta = []
    for edge in sub_edges_list:
        subgraph_edges_meta.append(
            {
                "id": edge.edge_id,
                "type": edge.edge_type,
                "source": edge.source,
                "target": edge.target,
                "source_title": title_map.get(edge.source, edge.source),
                "target_title": title_map.get(edge.target, edge.target),
                "description": edge.description,
                "direction": edge.direction,
            }
        )

    meta = {
        "seed_nodes": seed_keys,
        "seed_edges": seed_edge_ids,
        "subgraph_nodes": len(sub_nodes),
        "subgraph_edges": len(sub_edges),
        "causal_llm_mode": causal_llm_mode,
        "retrieved_context_mode": retrieved_context_mode,
        "seed_entities": seed_entities,
        "seed_communities": seed_communities,
        "seed_text_unit_ids": seed_text_unit_ids,
        "seed_community_summary_keys": community_seed_keys,
        "subgraph_nodes_detail": subgraph_nodes_meta,
        "subgraph_edges_detail": subgraph_edges_meta,
        "seed_params": {
            "k_bottom": seed_k_entities,
            "k_communities": seed_k_communities,
            "seed_mix_nodes": seed_mix_nodes,
            "seed_lexical_alpha": seed_lexical_alpha,
            "seed_lexical_disabled": seed_lexical_alpha >= 1.0,
            "seed_mmr_lambda": seed_mmr_lambda,
            "seed_edge_top_k": seed_edge_top_k,
            "seed_edge_min_score": seed_edge_min_score,
            "seed_edge_lexical_alpha": seed_edge_lexical_alpha,
            "seed_edge_min_token_len": seed_edge_min_token_len,
            "seed_text_units_top_k": seed_text_units_top_k,
            "seed_text_units_max_nodes": seed_text_units_max_nodes,
            "seed_text_units_min_score": seed_text_units_min_score,
            "seed_text_units_min_token_len": seed_text_units_min_token_len,
            "seed_community_summary_top_k": seed_community_summary_top_k,
            "seed_community_summary_min_score": seed_community_summary_min_score,
        },
        "hop_params": {
            "max_hops": max_hops,
            "gain_min_score": gain_min_score,
            "enable_causal": enable_causal,
            "enable_upward": enable_upward,
            "enable_downward": enable_downward,
            "max_fanout_per_node": max_fanout_per_node,
        },
        "context_params": {
            "max_prompt_nodes_causal_llm": max_prompt_nodes_causal_llm,
            "max_prompt_edges_causal_llm": max_prompt_edges_causal_llm,
            "max_each_node_chars_causal_llm": max_each_node_chars_causal_llm,
            "max_each_edge_chars_causal_llm": max_each_edge_chars_causal_llm,
            "max_each_text_unit_chars_causal_llm": max_each_text_unit_chars_causal_llm,
            "context_include_community_summary": context_include_community_summary,
            "context_include_text_units": context_include_text_units,
            "context_text_units_scope": context_text_units_scope,
            "context_text_units_top_k": context_text_units_top_k,
            "context_text_units_min_score": context_text_units_min_score,
            "context_text_units_min_token_len": context_text_units_min_token_len,
            "retrieved_context_max_nodes": retrieved_context_max_nodes,
            "retrieved_context_max_edges": retrieved_context_max_edges,
            "retrieved_context_max_text_units": retrieved_context_max_text_units,
            "retrieved_context_max_items": retrieved_context_max_items,
        },
        "llm_params": {
            "causal_precise_max_items": causal_precise_max_items,
            "causal_ct_precise_max_items": causal_ct_precise_max_items,
            "causal_p_answer_max_words": causal_p_answer_max_words,
            "causal_prompt_max_tokens": causal_prompt_max_tokens,
            "causal_llm_return_p_answer": causal_llm_return_p_answer,
            "answer_use_p_answer": answer_use_p_answer,
            "answer_context_max_items": answer_context_max_items,
            "answer_prompt_max_tokens": answer_prompt_max_tokens,
            "force_relation_text_unit": force_relation_text_unit,
        },
    }
    if debug_info:
        meta.update(debug_info)
    meta["causal_precise_ids"] = precise_ids
    if ct_precise_items:
        meta["causal_ct_precise_ids"] = ct_precise_ids
    meta["causal_p_answer"] = p_answer
    if save_query_meta:
        meta["causal_llm_prompt"] = causal_prompt
        meta["causal_llm_response"] = causal_response
        meta["causal_llm_json"] = causal_json
        meta["answer_llm_prompt"] = final_prompt
        meta["answer_llm_response"] = answer_text
        meta["short_id_map"] = {
            item.short_id: {
                "type": item.item_type,
                "original_id": item.original_id,
                "content": item.content,
                "score": item.score,
                "edge_ids": item.edge_ids or [],
            }
            for item in causal_items
        }

    return {
        "answer": answer_text,
        "retrieved_context": retrieved_context,
        "usage": usage_total,
        "meta": meta,
    }
