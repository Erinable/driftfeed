"""A network-free smoke test through the application service layer."""

from __future__ import annotations

from tests.conftest import make_item

from driftfeed.app import App
from driftfeed.models import EVENT_SAVE
from driftfeed.sources.base import Source


class FakeSource(Source):
    name = "fake"

    def fetch(self, *, limit: int = 50):
        return [
            make_item(
                source="fake", source_id="1", title="Rust release notes", tags=["fake", "rust"]
            ),
            make_item(
                source="fake", source_id="2", title="Python release notes", tags=["fake", "python"]
            ),
        ][:limit]


def test_fetch_feed_and_feedback_round_trip(db, config):
    app = App(db, config)
    reports = app.fetch(["hn"], source_factory=lambda _name, _config: FakeSource())

    assert reports[0].ok
    assert reports[0].source == "hn"
    assert reports[0].inserted == 2

    feed = app.feed(limit=2)
    assert len(feed) == 2
    chosen = app.resolve_position(1)
    app.record(chosen, EVENT_SAVE)

    assert db.count_items() == 2
    assert db.counts_by_event()[EVENT_SAVE] == 1
    assert app.stats()["scorer_updates"] == 1
