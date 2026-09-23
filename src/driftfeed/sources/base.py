"""The aggregation-layer contract.

A source is anything that can turn a config dict into `Item`s. Adapters must not
touch the database: `fetch()` returns items, the caller persists them. That keeps
adapters testable with a mocked HTTP session and nothing else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from driftfeed.models import Item
from driftfeed.util.http import HttpClient
from driftfeed.util.urls import canonicalize


class SourceUnavailable(RuntimeError):
    """The source cannot run right now — usually missing credentials.

    Distinct from a transport failure: `fetch` raising this means "skip me and
    carry on", not "the network is broken".
    """


class Source(ABC):
    #: Stable short name; also the `source` field on every Item it produces.
    name: str = "base"
    #: Seconds between requests. Defaults are deliberately conservative.
    min_interval_s: float = 0.5

    def __init__(self, *, client: HttpClient | None = None, config: dict | None = None) -> None:
        self.config = config or {}
        self.client = client or HttpClient(min_interval_s=self.min_interval_s)

    @abstractmethod
    def fetch(self, *, limit: int = 50) -> list[Item]:
        """Pull the configured listings. Raises SourceUnavailable if unusable."""

    def search(self, query: str, *, limit: int = 25) -> list[Item]:
        """Keyword search, used for cold-start seeding. Optional per source."""
        return []

    # --- helpers for subclasses -----------------------------------------

    def finalize(self, items: list[Item]) -> list[Item]:
        """Fill canonical_url and drop within-batch duplicates."""
        out: list[Item] = []
        seen: set[str] = set()
        for item in items:
            if not item.title or not item.source_id:
                continue
            item.canonical_url = canonicalize(item.url)
            if item.id in seen:
                continue
            seen.add(item.id)
            out.append(item)
        return out
