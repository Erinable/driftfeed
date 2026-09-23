"""Ranking layer contracts: embeddings, online learning, exploration, diversity."""

from __future__ import annotations

import random

import pytest

from driftfeed.ranking import HashingEmbedder, Ranker, ThompsonSampling
from driftfeed.ranking.features import UserProfile, extract, recency_decay
from driftfeed.ranking.scorer import LogisticSGDScorer
from tests.conftest import make_item


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbedder(dim=64)
    first = embedder.embed("Rust compilers and tooling")
    second = embedder.embed("Rust compilers and tooling")

    assert first == second
    assert len(first) == 64
    assert sum(value * value for value in first) == pytest.approx(1.0)


def test_scorer_learns_positive_feedback():
    scorer = LogisticSGDScorer(lr=0.5)
    features = {"bias": 1.0, "tag:rust": 1.0}
    before = scorer.score(features)

    scorer.update(features, 1.0)

    assert scorer.updates == 1
    assert scorer.score(features) > before


def test_ranker_explores_arms_and_caps_source_density(db):
    items = [
        make_item(source="hn", source_id="1", url="https://example.com/1", score=100),
        make_item(source="hn", source_id="2", url="https://example.com/2", score=90),
        make_item(source="reddit", source_id="3", url="https://example.com/3", score=80),
    ]
    db.upsert_items(items)
    ranker = Ranker(
        db,
        embedder=HashingEmbedder(dim=64),
        explorer=ThompsonSampling(rng=random.Random(0)),
        max_per_source_in_top=1,
        rng=random.Random(0),
    )

    ranked = ranker.rank(items, seed_keywords=["rust"], limit=2)

    assert len(ranked) == 2
    assert len({result.item.source for result in ranked}) == 2
    assert all(0.0 <= result.exploration <= 1.0 for result in ranked)
    assert db.count_embeddings() == len(items)


def test_feature_profile_uses_seed_similarity_and_recency():
    embedder = HashingEmbedder(dim=64)
    item = make_item(title="Rust compiler tooling")
    profile = UserProfile(seed_vec=embedder.embed("rust compiler"))
    features = extract(item, profile, embedding=embedder.embed(item.text()), now=item.created_at)

    assert features["sim"] > 0
    assert recency_decay(item.created_at, now=item.created_at) == 1.0


def test_feed_collapses_cross_source_urls_and_keeps_aliases(db):
    items = [
        make_item(source="hn", url="https://example.com/article?utm_source=hn"),
        make_item(source="reddit", url="https://www.example.com/article/"),
    ]
    db.upsert_items(items)

    ranked = Ranker(db).rank(items)

    assert len(ranked) == 1
    assert {ranked[0].item.source, *ranked[0].also_on} == {"hn", "reddit"}


def test_bandit_uses_one_draw_per_arm_per_ranking(db):
    items = [
        make_item(source_id="1", url="https://example.com/1"),
        make_item(source_id="2", url="https://example.com/2"),
    ]
    db.upsert_items(items)

    ranked = Ranker(db, rng=random.Random(42)).rank(items)

    assert ranked[0].exploration == ranked[1].exploration


def test_config_selects_ucb_and_persists_feedback_across_invocations(db, config):
    from driftfeed.app import App

    config.ranking = {**config.ranking, "exploration": "ucb1"}
    item = make_item()
    db.upsert_items([item])
    app = App(db, config)
    app.record(item, "save")

    restored = App(db, config)

    assert restored.stats()["exploration_policy"] == "ucb1"
    assert restored.stats()["scorer_updates"] == 1
    assert restored.ranker.explorer.state()["counts"]["hn"] == 1
