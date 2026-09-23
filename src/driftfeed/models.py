"""Core data shapes shared by every layer.

The `Item` schema is the contract between the aggregation layer and everything
downstream: a source adapter's only job is to produce these.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Feedback event types, ordered roughly by how strong a signal they carry.
EVENT_IMPRESSION = "impression"
EVENT_CLICK = "click"
EVENT_DWELL = "dwell"
EVENT_SAVE = "save"
EVENT_SKIP = "skip"
EVENT_HIDE = "hide"

EVENT_TYPES = (
    EVENT_IMPRESSION,
    EVENT_CLICK,
    EVENT_DWELL,
    EVENT_SAVE,
    EVENT_SKIP,
    EVENT_HIDE,
)

#: Implicit-feedback label used to train the scorer. `None` means "unlabelled"
#: (an impression on its own tells us nothing until we see what followed).
EVENT_LABELS: dict[str, float | None] = {
    EVENT_IMPRESSION: None,
    EVENT_CLICK: 1.0,
    EVENT_DWELL: 1.0,
    EVENT_SAVE: 1.0,
    EVENT_SKIP: 0.0,
    EVENT_HIDE: 0.0,
}


@dataclass(slots=True)
class Item:
    """One piece of content, normalised across sources."""

    source: str
    source_id: str
    url: str
    title: str
    body: str = ""
    author: str = ""
    score: int = 0
    comment_count: int = 0
    tags: list[str] = field(default_factory=list)
    created_at: float = 0.0
    fetched_at: float = field(default_factory=time.time)
    #: URL with tracking params and other noise stripped — the dedup key.
    canonical_url: str = ""

    @property
    def id(self) -> str:
        """Stable primary key: source plus the id within that source."""
        return f"{self.source}:{self.source_id}"

    def text(self) -> str:
        """The text handed to the embedder."""
        parts = [self.title, " ".join(self.tags), self.body]
        return "\n".join(p for p in parts if p).strip()

    def as_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "source_id": self.source_id,
            "url": self.url,
            "canonical_url": self.canonical_url or self.url,
            "title": self.title,
            "body": self.body,
            "author": self.author,
            "score": self.score,
            "comment_count": self.comment_count,
            "tags": ",".join(self.tags),
            "created_at": self.created_at,
            "fetched_at": self.fetched_at,
        }


@dataclass(slots=True)
class Feedback:
    """A single interaction with an item."""

    item_id: str
    event: str
    duration_s: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def label(self) -> float | None:
        return EVENT_LABELS.get(self.event)
