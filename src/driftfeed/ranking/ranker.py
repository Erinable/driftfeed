"""Ranker: combines embedder, scorer and explorer into a final ordering.

Final score = (1 - eps) * relevance + eps * exploration_bonus, then a diversity
pass caps how many consecutive slots one source may take. The blend weight `eps`
decays as the model accumulates updates: heavy exploration while the model knows
nothing, light exploration once it does — but never zero, or the feed calcifies.
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from driftfeed.models import EVENT_LABELS, Item
from driftfeed.ranking.embedder import Embedder, build_embedder, l2_normalize
from driftfeed.ranking.explorer import Explorer, build_explorer
from driftfeed.ranking.features import UserProfile, extract, topic_arm
from driftfeed.ranking.scorer import ColdStartScorer, LogisticSGDScorer, Scorer
from driftfeed.storage import Database

STATE_KEY_SCORER = "scorer"
STATE_KEY_EXPLORER = "explorer"
STATE_KEY_FEED = "last_feed"

#: Below this many gradient updates the learned model is blended with the
#: hand-weighted cold-start scorer instead of trusted outright.
WARMUP_UPDATES = 25


@dataclass
class Scored:
    """One ranked item plus the arithmetic that put it there — `feed` prints it,
    tests assert on it."""

    item: Item
    relevance: float
    exploration: float
    final: float
    arm: str
    features: dict[str, float] = field(default_factory=dict)
    also_on: list[str] = field(default_factory=list)


class Ranker:
    def __init__(
        self,
        db: Database,
        *,
        embedder: Embedder | None = None,
        scorer: Scorer | None = None,
        explorer: Explorer | None = None,
        half_life_hours: float = 36.0,
        max_per_source_in_top: int = 4,
        rng: random.Random | None = None,
    ) -> None:
        self.db = db
        self.embedder = embedder or build_embedder()
        self.scorer = scorer or LogisticSGDScorer()
        self.explorer = explorer or build_explorer(rng=rng)
        self.cold = ColdStartScorer()
        self.half_life_hours = half_life_hours
        self.max_per_source_in_top = max_per_source_in_top
        self.rng = rng or random.Random()
        self.load_state()

    # --- persistence -----------------------------------------------------

    def load_state(self) -> None:
        scorer_state = self.db.get_state(STATE_KEY_SCORER)
        if scorer_state:
            self.scorer.load_state(scorer_state)
        explorer_state = self.db.get_state(STATE_KEY_EXPLORER)
        if explorer_state:
            self.explorer.load_state(explorer_state)

    def save_state(self) -> None:
        self.db.set_state(STATE_KEY_SCORER, self.scorer.state())
        self.db.set_state(STATE_KEY_EXPLORER, self.explorer.state())

    # --- embeddings ------------------------------------------------------

    def embedding_for(self, item: Item) -> list[float]:
        """Cached per item and per model id; recomputed when the backend changes."""
        cached = self.db.get_embedding(item.id, model=self.embedder.model_id)
        if cached is not None:
            return cached
        vec = self.embedder.embed(item.text())
        self.db.put_embedding(item.id, self.embedder.model_id, vec)
        return vec

    def build_profile(self, seed_keywords: Sequence[str] = ()) -> UserProfile:
        """Centroid of positively-received items, with seed keywords as the prior."""
        positives = [
            fb for fb in self.db.all_feedback() if EVENT_LABELS.get(fb.event) == 1.0
        ]
        vectors: list[list[float]] = []
        for fb in positives:
            item = self.db.get_item(fb.item_id)
            if item is not None:
                vectors.append(self.embedding_for(item))

        centroid = None
        if vectors:
            dim = len(vectors[0])
            acc = [0.0] * dim
            for vec in vectors:
                for i, x in enumerate(vec[:dim]):
                    acc[i] += x
            centroid = l2_normalize([x / len(vectors) for x in acc])

        seed_vec = None
        if seed_keywords:
            seed_vec = self.embedder.embed(", ".join(seed_keywords))
        return UserProfile(centroid=centroid, seed_vec=seed_vec, positives=len(vectors))

    # --- scoring ---------------------------------------------------------

    @property
    def epsilon(self) -> float:
        """Exploration weight, decaying with model confidence but floored at 0.08."""
        updates = getattr(self.scorer, "updates", 0)
        return max(0.08, 0.45 * (1.0 / (1.0 + updates / 20.0)))

    def relevance(self, features: dict[str, float]) -> float:
        learned = self.scorer.score(features)
        updates = getattr(self.scorer, "updates", 0)
        if updates >= WARMUP_UPDATES:
            return learned
        # Linear hand-off from the prior to the learned model.
        w = updates / WARMUP_UPDATES
        return (1.0 - w) * self.cold.score(features) + w * learned

    def rank(
        self,
        items: Sequence[Item],
        *,
        profile: UserProfile | None = None,
        seed_keywords: Sequence[str] = (),
        limit: int = 20,
        now: float | None = None,
        diversify: bool = True,
    ) -> list[Scored]:
        profile = profile if profile is not None else self.build_profile(seed_keywords)
        now = now if now is not None else time.time()
        eps = self.epsilon

        scored: list[Scored] = []
        for item in items:
            vec = self.embedding_for(item)
            feats = extract(
                item, profile, embedding=vec, half_life_hours=self.half_life_hours, now=now
            )
            arm = topic_arm(item)
            rel = self.relevance(feats)
            exp = self.explorer.bonus(arm)
            scored.append(
                Scored(
                    item=item,
                    relevance=rel,
                    exploration=exp,
                    final=(1.0 - eps) * rel + eps * exp,
                    arm=arm,
                    features=feats,
                )
            )

        scored.sort(key=lambda s: s.final, reverse=True)
        if diversify:
            scored = self._diversify(scored, limit)
        top = scored[:limit]
        for s in top:
            s.also_on = [
                src for src in self.db.aliases_for(s.item.canonical_url) if src != s.item.source
            ]
        return top

    def _diversify(self, scored: list[Scored], limit: int) -> list[Scored]:
        """Cap per-source density in the visible window.

        Greedy: walk the sorted list and defer an item whose source already filled
        its quota. Deferred items keep their relative order and come back once the
        window rolls over, so nothing is dropped — only delayed.
        """
        if self.max_per_source_in_top <= 0:
            return scored
        out: list[Scored] = []
        deferred: list[Scored] = []
        counts: dict[str, int] = {}
        window = max(limit, 1)
        for s in scored:
            src = s.item.source
            if len(out) < window and counts.get(src, 0) >= self.max_per_source_in_top:
                deferred.append(s)
                continue
            out.append(s)
            counts[src] = counts.get(src, 0) + 1
            if len(out) == window:
                counts = {}
        return out + deferred

    # --- learning --------------------------------------------------------

    def learn_from(self, item: Item, event: str, *, profile: UserProfile | None = None,
                   seed_keywords: Sequence[str] = ()) -> None:
        """Apply one feedback event to both the scorer and the bandit."""
        label = EVENT_LABELS.get(event)
        if label is None:
            return  # Impressions alone carry no supervision.
        profile = profile if profile is not None else self.build_profile(seed_keywords)
        feats = extract(
            item, profile, embedding=self.embedding_for(item), half_life_hours=self.half_life_hours
        )
        self.scorer.update(feats, label)
        self.explorer.observe(topic_arm(item), label)
        self.save_state()

    def replay_feedback(self, *, seed_keywords: Sequence[str] = (), epochs: int = 1) -> int:
        """Retrain from the full feedback log. Used after changing the feature set."""
        self.scorer = LogisticSGDScorer()
        self.explorer = build_explorer(rng=self.rng)
        events = [fb for fb in self.db.all_feedback() if EVENT_LABELS.get(fb.event) is not None]
        profile = self.build_profile(seed_keywords)
        applied = 0
        for _ in range(max(1, epochs)):
            for fb in events:
                item = self.db.get_item(fb.item_id)
                if item is None:
                    continue
                feats = extract(
                    item, profile, embedding=self.embedding_for(item),
                    half_life_hours=self.half_life_hours,
                )
                label = EVENT_LABELS[fb.event]
                self.scorer.update(feats, float(label))
                self.explorer.observe(topic_arm(item), float(label))
                applied += 1
        self.save_state()
        return applied
