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
                source="fake", source_id="1", title="Rust release notes",
                url="https://example.com/rust", tags=["fake", "rust"]
            ),
            make_item(
                source="fake", source_id="2", title="Python release notes",
                url="https://example.com/python", tags=["fake", "python"]
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


def test_cli_init_fetch_feed_read_save_skip_stats(monkeypatch, session, capsys):
    import time

    from driftfeed.cli import main
    from driftfeed.config import db_path
    from driftfeed.storage import Database

    monkeypatch.setattr("driftfeed.cli.github_token", lambda: None)
    monkeypatch.setattr("requests.Session.request", session.request)
    session.add("/topstories.json", [42])
    session.add("/beststories.json", [42])
    session.add("/item/42.json", {
        "id": 42, "type": "story", "title": "Rust compiler release",
        "url": "https://example.com/rust", "time": time.time(), "score": 50,
    })

    assert main(["init"]) == 0
    assert main(["fetch", "--source", "hn"]) == 0
    assert main(["feed"]) == 0
    assert main(["read", "1", "--dwell", "45"]) == 0
    assert main(["save", "1"]) == 0
    assert main(["skip", "1"]) == 0
    assert main(["stats"]) == 0

    output = capsys.readouterr().out
    assert "Rust compiler release" in output
    with Database(db_path()) as database:
        assert database.counts_by_event() == {
            "impression": 1, "click": 1, "dwell": 1, "save": 1, "skip": 1,
        }
        assert database.feedback_for("hn:42")[2].duration_s == 45
        assert database.get_state("scorer")["updates"] == 4
