"""Feature extraction — the seam between raw items and the scorer.

Features are a named dict rather than a bare vector so `driftfeed stats` can
print learned weights against readable names, and so adding a feature never
silently shifts an index.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

from driftfeed.models import Item
from driftfeed.ranking.embedder import cosine
from driftfeed.util.urls import domain_of


@dataclass
class UserProfile:
    """What the system believes the user likes.

    `centroid` is the mean of embeddings of positively-received items; `seed_vec`
    covers the cold-start case where there is nothing to average yet.
    """

    centroid: list[float] | None = None
    seed_vec: list[float] | None = None
    positives: int = 0

    def similarity(self, vec: Sequence[float] | None) -> float:
        if not vec:
            return 0.0
        if self.centroid:
            return cosine(vec, self.centroid)
        if self.seed_vec:
            return cosine(vec, self.seed_vec)
        return 0.0


def recency_decay(created_at: float, *, half_life_hours: float = 36.0, now: float | None = None)\
        -> float:
    """Exponential decay in [0, 1]. One half-life old scores 0.5."""
    now = now if now is not None else time.time()
    if created_at <= 0 or half_life_hours <= 0:
        return 0.5  # Unknown age: neither fresh nor stale.
    age_h = max(0.0, (now - created_at) / 3600.0)
    return 0.5 ** (age_h / half_life_hours)


def popularity(score: int, comment_count: int) -> float:
    """Log-compressed engagement, roughly in [0, 1].

    Raw HN points and GitHub stars differ by orders of magnitude, so the log keeps
    a 20k-star repo from swamping every feature it is summed with.
    """
    raw = max(0, score) + 2 * max(0, comment_count)
    return min(1.0, math.log1p(raw) / math.log1p(1000))


def extract(
    item: Item,
    profile: UserProfile,
    *,
    embedding: Sequence[float] | None = None,
    half_life_hours: float = 36.0,
    now: float | None = None,
) -> dict[str, float]:
    """Feature dict for one item. `bias` is always present so the model has an intercept."""
    sim = profile.similarity(embedding)
    feats: dict[str, float] = {
        "bias": 1.0,
        "sim": sim,
        # Squared term lets the model express "only strong matches count"
        # without waiting for a nonlinear scorer.
        "sim_sq": sim * sim,
        "recency": recency_decay(item.created_at, half_life_hours=half_life_hours, now=now),
        "popularity": popularity(item.score, item.comment_count),
        f"source:{item.source}": 1.0,
        "has_body": 1.0 if item.body.strip() else 0.0,
        "title_len": min(1.0, len(item.title) / 120.0),
    }
    domain = domain_of(item.url)
    if domain:
        feats[f"domain:{domain}"] = 1.0
    for tag in item.tags[:6]:
        feats[f"tag:{tag.lower()}"] = 1.0
    return feats


def topic_arm(item: Item) -> str:
    """Bandit arm label.

    Source plus primary tag: coarse enough to gather feedback quickly, fine
    enough that "Rust posts" and "ML posts" are separable. A real topic
    clustering over embeddings is the obvious upgrade.
    """
    tag = next(
        (t.lower() for t in item.tags if t.lower() not in (item.source, "hn", "reddit", "github")),
        "",
    )
    return f"{item.source}/{tag}" if tag else item.source
