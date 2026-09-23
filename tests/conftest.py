"""Shared fixtures.

`FakeSession` stands in for `requests.Session`, so no test in this suite touches
the network. An adapter that reaches an unstubbed URL raises loudly rather than
silently returning empty.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest

from driftfeed.config import Config
from driftfeed.models import Item
from driftfeed.storage import Database
from driftfeed.util.http import HttpClient


@dataclass
class FakeResponse:
    status_code: int = 200
    payload: object = None
    headers: dict = field(default_factory=dict)
    _raise_on_json: bool = False

    def json(self):
        if self._raise_on_json:
            raise ValueError("not json")
        return self.payload


class FakeSession:
    """Routes requests by URL substring. Records every call for assertions."""

    def __init__(self, routes: dict | None = None) -> None:
        self.routes = dict(routes or {})
        self.calls: list[dict] = []

    def add(self, url_fragment: str, payload, *, status: int = 200, headers: dict | None = None):
        self.routes[url_fragment] = FakeResponse(
            status_code=status, payload=payload, headers=headers or {}
        )
        return self

    def request(self, method, url, *, params=None, data=None, headers=None, timeout=None):
        self.calls.append(
            {"method": method, "url": url, "params": params, "data": data, "headers": headers}
        )
        # Longest match wins, so "/item/" beats "/" when both are registered.
        best = None
        for fragment, resp in self.routes.items():
            if fragment in url and (best is None or len(fragment) > len(best[0])):
                best = (fragment, resp)
        if best is None:
            raise AssertionError(f"unstubbed request: {method} {url}")
        return best[1]


@pytest.fixture
def no_sleep(monkeypatch):
    """Make backoff instant so retry paths do not cost wall-clock time."""
    monkeypatch.setattr("time.sleep", lambda *_: None)


@pytest.fixture
def session():
    return FakeSession()


@pytest.fixture
def client(session):
    return HttpClient(session=session, max_retries=1, backoff_base_s=0.0, sleep=lambda *_: None)


@pytest.fixture
def db(tmp_path):
    with Database(tmp_path / "test.sqlite3") as database:
        yield database


@pytest.fixture
def config():
    return Config.load(path=tmp_nonexistent())


def tmp_nonexistent():
    from pathlib import Path

    return Path("/nonexistent-driftfeed-config.json")


def make_item(
    *,
    source: str = "hn",
    source_id: str = "1",
    title: str = "A post about rust and compilers",
    url: str = "https://example.com/a",
    body: str = "",
    score: int = 10,
    comment_count: int = 2,
    tags: list[str] | None = None,
    age_hours: float = 1.0,
) -> Item:
    item = Item(
        source=source,
        source_id=source_id,
        url=url,
        title=title,
        body=body,
        score=score,
        comment_count=comment_count,
        tags=tags if tags is not None else [source],
        created_at=time.time() - age_hours * 3600,
    )
    from driftfeed.util.urls import canonicalize

    item.canonical_url = canonicalize(url)
    return item
