"""Aggregation layer: one adapter per source, all speaking `Source`."""

from __future__ import annotations

from driftfeed.sources.base import Source, SourceUnavailable
from driftfeed.sources.github import GitHubSource
from driftfeed.sources.hackernews import HackerNewsSource
from driftfeed.sources.reddit import RedditSource

#: Registry keyed by the same short name the CLI's `--source` takes.
REGISTRY: dict[str, type[Source]] = {
    HackerNewsSource.name: HackerNewsSource,
    RedditSource.name: RedditSource,
    GitHubSource.name: GitHubSource,
}


def build_source(name: str, config: dict | None = None) -> Source:
    try:
        cls = REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown source {name!r}; known sources: {known}") from None
    from driftfeed.util.http import HttpClient

    return cls(config=config or {}, client=HttpClient(min_interval_s=cls.min_interval_s))


__all__ = [
    "GitHubSource",
    "HackerNewsSource",
    "REGISTRY",
    "RedditSource",
    "Source",
    "SourceUnavailable",
    "build_source",
]
