"""Storage layer: migrations, upsert semantics, feedback, embeddings, kv."""

from __future__ import annotations

import pytest
from tests.conftest import make_item

from driftfeed.models import EVENT_CLICK, EVENT_HIDE, Feedback
from driftfeed.storage import Database, pack_vector, unpack_vector
from driftfeed.storage.schema import LATEST_VERSION, migrate


def test_migrate_is_idempotent(tmp_path):
    path = tmp_path / "db.sqlite3"
    with Database(path) as db:
        assert db.schema_version == LATEST_VERSION
        # Re-running must not raise or double-apply.
        assert migrate(db.conn) == LATEST_VERSION
    with Database(path) as db:
        assert db.schema_version == LATEST_VERSION


def test_vector_roundtrip_preserves_values_within_float32():
    vec = [0.5, -0.25, 0.125, 0.0]
    assert unpack_vector(pack_vector(vec)) == pytest.approx(vec, abs=1e-6)


def test_upsert_inserts_then_updates_volatile_fields(db):
    item = make_item(score=10, comment_count=1)
    assert db.upsert_items([item]) == (1, 0)

    item.score = 99
    item.comment_count = 42
    item.created_at = 0.0  # Must be ignored on conflict.
    assert db.upsert_items([item]) == (0, 1)

    stored = db.get_item(item.id)
    assert (stored.score, stored.comment_count) == (99, 42)
    assert stored.created_at > 0, "created_at must not be overwritten by a re-fetch"
    assert db.count_items() == 1


def test_tags_roundtrip(db):
    item = make_item(tags=["hn", "rust", "compilers"])
    db.upsert_items([item])
    assert db.get_item(item.id).tags == ["hn", "rust", "compilers"]


def test_cross_source_aliases_record_every_source(db):
    url = "https://example.com/same?utm_source=x"
    db.upsert_items(
        [
            make_item(source="hn", source_id="1", url=url),
            make_item(source="reddit", source_id="abc", url="https://www.example.com/same/"),
        ]
    )
    canonical = db.get_item("hn:1").canonical_url
    assert db.aliases_for(canonical) == ["hn", "reddit"]


def test_candidate_items_excludes_hidden_and_old(db):
    fresh = make_item(source_id="fresh", age_hours=1)
    hidden = make_item(source_id="hidden", age_hours=1)
    ancient = make_item(source_id="ancient", age_hours=24 * 60)
    db.upsert_items([fresh, hidden, ancient])
    db.add_feedback(Feedback(item_id=hidden.id, event=EVENT_HIDE))

    ids = {i.id for i in db.candidate_items(max_age_days=21)}
    assert ids == {fresh.id}


def test_candidate_items_orders_newest_first(db):
    db.upsert_items(
        [
            make_item(source_id="old", age_hours=48),
            make_item(source_id="new", age_hours=1),
        ]
    )
    assert [i.source_id for i in db.candidate_items()] == ["new", "old"]


def test_add_feedback_rejects_unknown_event(db):
    item = make_item()
    db.upsert_items([item])
    with pytest.raises(ValueError, match="unknown feedback event"):
        db.add_feedback(Feedback(item_id=item.id, event="telepathy"))


def test_feedback_counts_and_lookup(db):
    item = make_item()
    db.upsert_items([item])
    db.add_feedback(Feedback(item_id=item.id, event=EVENT_CLICK))
    db.add_feedback(Feedback(item_id=item.id, event=EVENT_CLICK))
    assert db.counts_by_event() == {EVENT_CLICK: 2}
    assert len(db.feedback_for(item.id)) == 2


def test_embeddings_are_scoped_to_their_model(db):
    item = make_item()
    db.upsert_items([item])
    db.put_embedding(item.id, "model-a", [1.0, 0.0])

    assert db.get_embedding(item.id, model="model-a") == pytest.approx([1.0, 0.0])
    # A different backend must miss the cache rather than reuse a foreign space.
    assert db.get_embedding(item.id, model="model-b") is None

    db.put_embedding(item.id, "model-b", [0.0, 1.0, 0.0])
    assert db.get_embedding(item.id, model="model-b") == pytest.approx([0.0, 1.0, 0.0])
    assert db.count_embeddings() == 1


def test_kv_state_roundtrip_and_default(db):
    assert db.get_state("missing", "fallback") == "fallback"
    db.set_state("weights", {"sim": 0.5})
    assert db.get_state("weights") == {"sim": 0.5}


def test_mark_fetched_records_source_state(db):
    db.mark_fetched("hn", cursor="42")
    state = db.source_state()["hn"]
    assert state["cursor"] == "42"
    assert state["last_fetch_at"] > 0
