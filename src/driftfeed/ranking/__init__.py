"""Ranking layer: Embedder, Scorer, Explorer, Ranker.

Each is an interface with a working minimal implementation. The interfaces are
the stable part; the implementations are meant to be replaced.
"""

from driftfeed.ranking.embedder import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
    cosine,
)
from driftfeed.ranking.explorer import UCB1, Explorer, ThompsonSampling, build_explorer
from driftfeed.ranking.features import UserProfile, extract, popularity, recency_decay, topic_arm
from driftfeed.ranking.ranker import Ranker, Scored
from driftfeed.ranking.scorer import ColdStartScorer, LogisticSGDScorer, Scorer

__all__ = [
    "ColdStartScorer",
    "Embedder",
    "Explorer",
    "HashingEmbedder",
    "LogisticSGDScorer",
    "Ranker",
    "Scored",
    "Scorer",
    "SentenceTransformerEmbedder",
    "ThompsonSampling",
    "UCB1",
    "UserProfile",
    "build_embedder",
    "build_explorer",
    "cosine",
    "extract",
    "popularity",
    "recency_decay",
    "topic_arm",
]
