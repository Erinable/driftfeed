"""Scorer: (item features, user profile) -> relevance in [0, 1].

The default is online logistic regression trained by SGD on implicit feedback.
Online rather than batch because a single user generates feedback one event at a
time, and a model that only improves after a nightly retrain feels broken.

Weights live in a dict keyed by feature name, so a feature appearing for the
first time (a new source, a new tag) costs nothing and starts at zero.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping


class Scorer(ABC):
    @abstractmethod
    def score(self, features: Mapping[str, float]) -> float:
        """Relevance in [0, 1]."""

    @abstractmethod
    def update(self, features: Mapping[str, float], label: float) -> None:
        """One gradient step against a 0/1 label."""

    def state(self) -> dict:
        return {}

    def load_state(self, state: Mapping) -> None:
        return None


class LogisticSGDScorer(Scorer):
    """Logistic regression with per-feature L2 and an SGD update.

    `lr` is deliberately on the high side (0.1): with a few dozen feedback events
    total, a conservative rate would never move off the prior.
    """

    def __init__(self, *, lr: float = 0.1, l2: float = 1e-4) -> None:
        self.lr = lr
        self.l2 = l2
        self.weights: dict[str, float] = {}
        self.updates = 0

    def _logit(self, features: Mapping[str, float]) -> float:
        return sum(self.weights.get(name, 0.0) * val for name, val in features.items())

    def score(self, features: Mapping[str, float]) -> float:
        return _sigmoid(self._logit(features))

    def update(self, features: Mapping[str, float], label: float) -> None:
        pred = self.score(features)
        err = pred - label
        for name, val in features.items():
            if val == 0.0:
                continue
            w = self.weights.get(name, 0.0)
            self.weights[name] = w - self.lr * (err * val + self.l2 * w)
        self.updates += 1

    def top_weights(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]

    def state(self) -> dict:
        return {"weights": self.weights, "updates": self.updates, "lr": self.lr, "l2": self.l2}

    def load_state(self, state: Mapping) -> None:
        raw = state.get("weights") or {}
        self.weights = {str(k): float(v) for k, v in raw.items()}
        self.updates = int(state.get("updates") or 0)
        self.lr = float(state.get("lr") or self.lr)
        self.l2 = float(state.get("l2") or self.l2)


class ColdStartScorer(Scorer):
    """Hand-weighted fallback for a model with no feedback yet.

    Used as a blend partner rather than a replacement: the learned model takes
    over gradually as `updates` grows, so early rankings are never pure noise.
    """

    WEIGHTS = {"sim": 1.6, "recency": 1.0, "popularity": 0.8, "bias": -1.0}

    def score(self, features: Mapping[str, float]) -> float:
        return _sigmoid(sum(w * features.get(name, 0.0) for name, w in self.WEIGHTS.items()))

    def update(self, features: Mapping[str, float], label: float) -> None:
        return None


def _sigmoid(x: float) -> float:
    # Clamped to avoid math.exp overflow on extreme logits.
    if x < -35:
        return 1e-15
    if x > 35:
        return 1.0 - 1e-15
    return 1.0 / (1.0 + math.exp(-x))
