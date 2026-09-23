"""Embedder: item text -> vector.

Two backends behind one interface.

`HashingEmbedder` is the default and has no model download: hashed character
n-grams plus word unigrams, L2-normalised. It is a bag-of-features
approximation, so it captures topical overlap but not paraphrase. The point is
that a fresh clone produces a working feed on the first run.

`SentenceTransformerEmbedder` is the quality option (`pip install
driftfeed[st]`, then `DRIFTFEED_EMBEDDER=sentence-transformers`). Switching
backends changes `model_id`, which invalidates stored vectors — they are
re-embedded lazily rather than migrated.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

WORD_RE = re.compile(r"[a-z0-9_+#.]+")


class Embedder(ABC):
    #: Identifies the vector space. Stored alongside each embedding.
    model_id: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed one string. Must return a `dim`-length vector."""

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class HashingEmbedder(Embedder):
    """Hashed bag of words + character trigrams, L2-normalised.

    Sublinear term weighting (1 + log tf) stands in for TF-IDF: with no corpus
    statistics on hand, it still stops a repeated word from dominating.
    """

    def __init__(self, dim: int = 512, *, char_ngrams: int = 3) -> None:
        self.dim = dim
        self.char_ngrams = char_ngrams
        self.model_id = f"hashing-{dim}d-c{char_ngrams}"

    def embed(self, text: str) -> list[float]:
        counts: dict[int, float] = {}
        low = (text or "").lower()
        for token in WORD_RE.findall(low):
            _bump(counts, f"w:{token}", self.dim)
        squashed = re.sub(r"\s+", " ", low)
        n = self.char_ngrams
        for i in range(max(0, len(squashed) - n + 1)):
            _bump(counts, f"c:{squashed[i : i + n]}", self.dim, weight=0.5)

        vec = [0.0] * self.dim
        for idx, tf in counts.items():
            vec[idx] = 1.0 + math.log(tf)
        return l2_normalize(vec)


class SentenceTransformerEmbedder(Embedder):
    """Wraps sentence-transformers. Model loads on first use, not at import."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.environ.get(
            "DRIFTFEED_ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model_id = f"st:{self.model_name}"
        self._model = None
        self.dim = 384  # all-MiniLM-L6-v2; corrected after the model loads.

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Install with `pip install 'driftfeed[st]'`, or set "
                    "DRIFTFEED_EMBEDDER=hashing to use the dependency-free backend."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        raw = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in row] for row in raw]


def build_embedder(backend: str | None = None) -> Embedder:
    """Resolve the backend from an explicit name, else DRIFTFEED_EMBEDDER."""
    name = (backend or os.environ.get("DRIFTFEED_EMBEDDER") or "hashing").strip().lower()
    if name in ("hashing", "hash", "tfidf", "default"):
        return HashingEmbedder()
    if name in ("st", "sentence-transformers", "sentence_transformers"):
        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedder backend: {name!r} (expected 'hashing' or 'st')")


def _bump(counts: dict[int, float], key: str, dim: int, *, weight: float = 1.0) -> None:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    idx = int.from_bytes(digest, "little") % dim
    counts[idx] = counts.get(idx, 0.0) + weight


def l2_normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Returns 0.0 for empty or mismatched vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))
