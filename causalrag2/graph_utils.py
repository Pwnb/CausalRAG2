from __future__ import annotations

"""
Graph utilities shared by the CausalRAG2 indexer: text embeddings (with a
TF-IDF fallback) and Leiden / NetworkX community detection.
"""

import logging
import math
import random
from collections import Counter
from typing import Dict, Iterable, List

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike


class LightTfidfVectorizer:
    """Lightweight TF-IDF vectorizer used when scikit-learn is unavailable or for state restoration."""

    def __init__(self, stop_words: str | None = None):
        self.stop_words = set(["the", "a", "an", "is", "are", "of", "to", "in"]) if stop_words else set()
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Dict[str, float] = {}

    def fit_transform(self, texts: List[str]):
        self._build_vocab(texts)
        return self.transform(texts)

    def transform(self, texts: List[str]):
        if not self.vocabulary_:
            return np.zeros((len(texts), 1), dtype=np.float32)
        matrix = np.zeros((len(texts), len(self.vocabulary_)), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = [t.lower() for t in text.split() if t.lower() not in self.stop_words]
            counts = Counter(tokens)
            for token, count in counts.items():
                if token in self.vocabulary_:
                    col = self.vocabulary_[token]
                    idf = self.idf_.get(token, 1.0)
                    matrix[row, col] = (count / max(len(tokens), 1)) * idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

    def _build_vocab(self, texts: List[str]) -> None:
        vocab = Counter()
        for text in texts:
            tokens = set([t.lower() for t in text.split() if t.lower() not in self.stop_words])
            vocab.update(tokens)
        self.vocabulary_ = {token: idx for idx, (token, _) in enumerate(vocab.items())}
        total_docs = len(texts)
        self.idf_ = {token: math.log((total_docs + 1) / (freq + 1)) + 1 for token, freq in vocab.items()}

    def get_state(self) -> Dict:
        return {
            "type": "light_tfidf",
            "vocabulary": self.vocabulary_,
            "idf": self.idf_,
        }

    def load_state(self, state: Dict) -> None:
        self.vocabulary_ = {k: int(v) for k, v in state.get("vocabulary", {}).items()}
        self.idf_ = {k: float(v) for k, v in state.get("idf", {}).items()}


try:  # pragma: no cover - optional dependency
    from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer

    _TFIDF_CLASS = SklearnTfidfVectorizer
except Exception:  # pragma: no cover - fallback implementation
    _TFIDF_CLASS = LightTfidfVectorizer


LOGGER = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer

    _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    LOGGER.info("Loaded sentence-transformers model 'all-MiniLM-L6-v2'.")
except Exception:  # pragma: no cover - optional dependency
    _ST_MODEL = None
    LOGGER.warning("sentence-transformers not available. Falling back to TF-IDF embeddings.")

try:  # pragma: no cover - optional dependency
    import igraph as ig
    import leidenalg

    _HAVE_LEIDEN = True
except Exception:
    _HAVE_LEIDEN = False
    LOGGER.warning("igraph/leidenalg not available. Falling back to networkx community detection.")

random.seed(42)
np.random.seed(42)


def _ensure_embedding_array(vectors: ArrayLike) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


class EmbeddingBackend:
    """Manage text embeddings with sentence-transformers or TF-IDF fallback."""

    def __init__(self) -> None:
        self._tfidf_vectorizer: object | None = None
        self._light_vectorizer: LightTfidfVectorizer | None = None
        self._tfidf_corpus: List[str] = []
        self._loaded_state = False
        self._force_tfidf = False

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts = [t or "" for t in texts]
        if _ST_MODEL and not self._force_tfidf:
            return _ensure_embedding_array(_ST_MODEL.encode(texts, convert_to_numpy=True))

        # lazy fit TF-IDF on corpus accumulation
        if self._tfidf_vectorizer is None:
            if self._light_vectorizer and self._loaded_state:
                self._tfidf_vectorizer = self._light_vectorizer
                matrix = self._tfidf_vectorizer.transform(texts)
            else:
                self._tfidf_corpus = texts
                self._tfidf_vectorizer = _TFIDF_CLASS(stop_words="english")
                matrix = self._tfidf_vectorizer.fit_transform(self._tfidf_corpus)
            if hasattr(matrix, "toarray"):
                matrix = matrix.toarray()
            return np.asarray(matrix, dtype=np.float32)

        matrix = self._tfidf_vectorizer.transform(texts)
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        return np.asarray(matrix, dtype=np.float32)

    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        a = _ensure_embedding_array(a)
        b = _ensure_embedding_array(b)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        sim = float(np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b)))
        return max(min(sim, 1.0), -1.0)

    def get_state(self) -> Dict:
        if _ST_MODEL and not self._force_tfidf:
            return {"type": "sentence_transformer", "model": "all-MiniLM-L6-v2"}
        if self._tfidf_vectorizer is None:
            return {"type": "tfidf", "vocabulary": {}, "idf": {}}

        if isinstance(self._tfidf_vectorizer, LightTfidfVectorizer):
            state = self._tfidf_vectorizer.get_state()
            state["type"] = "tfidf"
            return state

        vocab = {k: int(v) for k, v in self._tfidf_vectorizer.vocabulary_.items()}
        idf = {term: float(self._tfidf_vectorizer.idf_[idx]) for term, idx in vocab.items()}
        return {"type": "tfidf", "vocabulary": vocab, "idf": idf}

    def load_state(self, state: Dict) -> None:
        if not state:
            return
        if state.get("type") == "sentence_transformer":
            self._force_tfidf = False
            return  # nothing to do for transformer model
        self._force_tfidf = True
        vec = LightTfidfVectorizer(stop_words="english")
        vec.load_state(state)
        self._light_vectorizer = vec
        self._tfidf_vectorizer = None  # ensure refit uses loaded vectorizer
        self._loaded_state = True


EMBEDDINGS = EmbeddingBackend()


def _community_detection_with_leiden(H: nx.Graph) -> Dict[str, int]:  # pragma: no cover - optional dependency
    mapping = {node: idx for idx, node in enumerate(H.nodes())}
    reverse = {idx: node for node, idx in mapping.items()}

    ig_graph = ig.Graph(len(mapping))
    edges = [(mapping[u], mapping[v]) for u, v in H.edges()]
    ig_graph.add_edges(edges)

    weights = [H[u][v].get("weight", 1.0) for u, v in H.edges()]
    if weights:
        ig_graph.es["weight"] = weights

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=ig_graph.es["weight"] if "weight" in ig_graph.es.attributes() else None,
        seed=42,
    )
    community_map = {}
    for community_id, members in enumerate(partition):
        for node_idx in members:
            community_map[reverse[node_idx]] = community_id
    return community_map


def _community_detection_with_networkx(H: nx.Graph) -> Dict[str, int]:
    try:
        from networkx.algorithms.community import greedy_modularity_communities

        communities = list(greedy_modularity_communities(H, weight="weight"))
    except Exception:  # pragma: no cover - fallback
        LOGGER.warning("Fallback to label propagation community detection.")
        from networkx.algorithms.community import asyn_lpa_communities

        communities = list(asyn_lpa_communities(H, weight="weight", seed=42))

    community_map: Dict[str, int] = {}
    for idx, community in enumerate(communities):
        for node in community:
            community_map[node] = idx
    return community_map


__all__ = [
    "EMBEDDINGS",
    "EmbeddingBackend",
    "LightTfidfVectorizer",
]
