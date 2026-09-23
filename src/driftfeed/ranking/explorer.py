"""Explorer: explore/exploit balance. Not optional.

A single-user recommender trained on its own suggestions is a filter bubble
generator: the scorer only ever sees feedback on what it already ranked highly,
so the feed narrows with every session and the narrowing is invisible from
inside. The explorer is the only component that deliberately surfaces things the
model is not yet confident about.

Arms are topic/source clusters (see `features.topic_arm`). Two policies ship:
Thompson sampling over a Beta posterior (default) and UCB1.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod


class Explorer(ABC):
    @abstractmethod
    def bonus(self, arm: str) -> float:
        """Additive exploration bonus for this arm, roughly in [0, 1]."""

    @abstractmethod
    def observe(self, arm: str, reward: float) -> None:
        """Record a reward in [0, 1] for an arm."""

    def state(self) -> dict:
        return {}

    def load_state(self, state: dict) -> None:
        return None


class ThompsonSampling(Explorer):
    """Beta-Bernoulli Thompson sampling.

    `bonus` draws from each arm's posterior, so an arm with little data has a wide
    posterior and gets a high draw often enough to be tried. As evidence
    accumulates the posterior tightens and the bonus converges on the true rate.
    """

    def __init__(self, *, prior_alpha: float = 1.0, prior_beta: float = 1.0,
                 rng: random.Random | None = None) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.rng = rng or random.Random()
        self.arms: dict[str, list[float]] = {}

    def _arm(self, arm: str) -> list[float]:
        return self.arms.setdefault(arm, [self.prior_alpha, self.prior_beta])

    def bonus(self, arm: str) -> float:
        alpha, beta = self._arm(arm)
        return self.rng.betavariate(max(alpha, 1e-6), max(beta, 1e-6))

    def observe(self, arm: str, reward: float) -> None:
        reward = max(0.0, min(1.0, reward))
        stats = self._arm(arm)
        stats[0] += reward
        stats[1] += 1.0 - reward

    def expected(self, arm: str) -> float:
        alpha, beta = self._arm(arm)
        return alpha / (alpha + beta)

    def pulls(self, arm: str) -> float:
        alpha, beta = self._arm(arm)
        return (alpha - self.prior_alpha) + (beta - self.prior_beta)

    def state(self) -> dict:
        return {
            "policy": "thompson",
            "arms": {k: list(v) for k, v in self.arms.items()},
            "prior": [self.prior_alpha, self.prior_beta],
        }

    def load_state(self, state: dict) -> None:
        self.arms = {
            str(k): [float(v[0]), float(v[1])]
            for k, v in (state.get("arms") or {}).items()
        }
        prior = state.get("prior") or [self.prior_alpha, self.prior_beta]
        self.prior_alpha, self.prior_beta = float(prior[0]), float(prior[1])


class UCB1(Explorer):
    """Deterministic alternative: mean reward plus a confidence radius."""

    def __init__(self, *, c: float = 1.4) -> None:
        self.c = c
        self.counts: dict[str, float] = {}
        self.rewards: dict[str, float] = {}

    @property
    def total(self) -> float:
        return sum(self.counts.values())

    def bonus(self, arm: str) -> float:
        n = self.counts.get(arm, 0.0)
        if n <= 0:
            return 1.0  # Never-tried arms are maximally uncertain.
        mean = self.rewards.get(arm, 0.0) / n
        radius = self.c * math.sqrt(math.log(max(self.total, 2.0)) / n)
        return min(1.0, mean + radius)

    def observe(self, arm: str, reward: float) -> None:
        self.counts[arm] = self.counts.get(arm, 0.0) + 1.0
        self.rewards[arm] = self.rewards.get(arm, 0.0) + max(0.0, min(1.0, reward))

    def state(self) -> dict:
        return {"policy": "ucb1", "counts": self.counts, "rewards": self.rewards, "c": self.c}

    def load_state(self, state: dict) -> None:
        self.counts = {str(k): float(v) for k, v in (state.get("counts") or {}).items()}
        self.rewards = {str(k): float(v) for k, v in (state.get("rewards") or {}).items()}
        self.c = float(state.get("c") or self.c)


def build_explorer(policy: str | None = None, *, rng: random.Random | None = None) -> Explorer:
    name = (policy or "thompson").strip().lower()
    if name in ("thompson", "ts", "default"):
        return ThompsonSampling(rng=rng)
    if name in ("ucb", "ucb1"):
        return UCB1()
    raise ValueError(f"unknown exploration policy: {name!r} (expected 'thompson' or 'ucb1')")
