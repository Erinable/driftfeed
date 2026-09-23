"""Aggregation layer. Every HTTP call is stubbed — the suite never hits a network."""

from __future__ import annotations

import pytest

from driftfeed.sources import REGISTRY, GitHubSource, HackerNewsSource, RedditSource
from driftfeed.sources.base import SourceUnavailable
from driftfeed.util.http import HttpClient, HttpError, RateLimiter


def make_client(session):
    return HttpClient(session=session, max_retries=1, backoff_base_s=0.0, sleep=lambda *_: None)


# --- HTTP client ---------------------------------------------------------


def test_client_sends_user_agent(session, client):
    session.add("example.com", {"ok": True})
    client.get_json("https://example.com/x")
    assert "driftfeed/" in session.calls[0]["headers"]["User-Agent"]


def test_client_retries_then_succeeds(session):
    calls = {"n": 0}

    class Flaky:
        def request(self, *a, **kw):
            calls["n"] += 1
            from tests.conftest import FakeResponse

            if calls["n"] == 1:
                return FakeResponse(status_code=503, payload=None)
            return FakeResponse(status_code=200, payload={"ok": True})

    client = HttpClient(session=Flaky(), max_retries=2, backoff_base_s=0.0, sleep=lambda *_: None)
    assert client.get_json("https://example.com") == {"ok": True}
    assert calls["n"] == 2


def test_client_does_not_retry_client_errors(session):
    session.add("example.com", None, status=404)
    client = make_client(session)
    with pytest.raises(HttpError) as exc:
        client.get_json("https://example.com/missing")
    assert exc.value.status == 404
    assert len(session.calls) == 1, "4xx must not be retried"


def test_client_honours_retry_after(session):
    slept: list[float] = []
    session.add("example.com", None, status=429, headers={"Retry-After": "3"})
    client = HttpClient(
        session=session, max_retries=1, backoff_base_s=0.0, sleep=lambda s: slept.append(s)
    )
    with pytest.raises(HttpError):
        client.get_json("https://example.com")
    assert 3.0 in slept


def test_rate_limiter_waits_for_the_gap():
    slept: list[float] = []
    now = {"t": 100.0}
    limiter = RateLimiter(1.0, sleep=slept.append, clock=lambda: now["t"])
    limiter.wait()
    now["t"] += 0.25
    limiter.wait()
    assert slept and slept[-1] == pytest.approx(0.75)


# --- Hacker News ---------------------------------------------------------

HN_STORY = {
    "id": 42,
    "type": "story",
    "title": "Rust 2.0 &amp; beyond",
    "url": "https://blog.example.com/rust?utm_source=hn",
    "by": "alice",
    "score": 250,
    "descendants": 88,
    "time": 1_700_000_000,
    "text": "<p>Some <i>html</i> body</p>",
}


def test_hn_fetch_maps_firebase_story(session):
    session.add("/topstories.json", [42])
    session.add("/item/42.json", HN_STORY)
    src = HackerNewsSource(client=make_client(session), config={"lists": ["topstories"]})

    items = src.fetch(limit=1)
    assert len(items) == 1
    item = items[0]
    assert item.id == "hn:42"
    assert item.title == "Rust 2.0 & beyond", "HTML entities must be unescaped"
    assert item.body == "Some html body", "tags must be stripped from the body"
    assert item.score == 250 and item.comment_count == 88
    assert item.canonical_url == "https://blog.example.com/rust", "tracking params stripped"


def test_hn_skips_dead_deleted_and_non_stories(session):
    session.add("/topstories.json", [1, 2, 3])
    session.add("/item/1.json", {"id": 1, "type": "story", "title": "gone", "deleted": True})
    session.add("/item/2.json", {"id": 2, "type": "comment", "title": "a comment"})
    session.add("/item/3.json", {"id": 3, "type": "story", "title": "ok", "time": 1})
    src = HackerNewsSource(client=make_client(session), config={"lists": ["topstories"]})
    assert [i.source_id for i in src.fetch(limit=3)] == ["3"]


def test_hn_ask_post_without_url_falls_back_to_the_thread(session):
    session.add("/topstories.json", [7])
    session.add("/item/7.json", {"id": 7, "type": "story", "title": "Ask HN: why?", "time": 1})
    src = HackerNewsSource(client=make_client(session), config={"lists": ["topstories"]})
    assert src.fetch(limit=1)[0].url == "https://news.ycombinator.com/item?id=7"


def test_hn_one_failing_list_does_not_sink_the_fetch(session):
    session.add("/topstories.json", None, status=500)
    session.add("/beststories.json", [42])
    session.add("/item/42.json", HN_STORY)
    src = HackerNewsSource(
        client=make_client(session), config={"lists": ["topstories", "beststories"]}
    )
    assert len(src.fetch(limit=2)) == 1


def test_hn_search_uses_algolia(session):
    session.add(
        "hn.algolia.com",
        {"hits": [{"objectID": "9", "title": "rust things", "url": "https://x.example/1",
                   "points": 5, "num_comments": 1, "created_at_i": 1_700_000_000}]},
    )
    src = HackerNewsSource(client=make_client(session))
    items = src.search("rust", limit=5)
    assert items[0].id == "hn:9"
    assert "seed:rust" in items[0].tags
    assert session.calls[0]["params"]["query"] == "rust"


# --- Reddit --------------------------------------------------------------


def reddit_listing(post_id="abc", **overrides):
    data = {
        "id": post_id,
        "title": "A reddit post",
        "url": "https://example.com/post",
        "permalink": f"/r/programming/comments/{post_id}/x/",
        "subreddit": "programming",
        "author": "bob",
        "score": 120,
        "num_comments": 15,
        "created_utc": 1_700_000_000,
        "selftext": "body text",
    }
    data.update(overrides)
    return {"data": {"children": [{"data": data}]}}


def test_reddit_without_credentials_is_unavailable(session):
    src = RedditSource(client=make_client(session), config={"subreddits": ["programming"]},
                       credentials=None)
    assert src.available is False
    with pytest.raises(SourceUnavailable, match="DRIFTFEED_REDDIT_CLIENT_ID"):
        src.fetch()


def test_reddit_fetch_authenticates_then_maps_posts(session):
    session.add("api/v1/access_token", {"access_token": "tok", "expires_in": 3600})
    session.add("/r/programming/hot", reddit_listing())
    src = RedditSource(
        client=make_client(session),
        config={"subreddits": ["programming"], "listing": "hot"},
        credentials=("id", "secret"),
    )

    items = src.fetch(limit=10)
    assert items[0].id == "reddit:abc"
    assert "r/programming" in items[0].tags
    assert items[0].score == 120

    token_call, listing_call = session.calls[0], session.calls[1]
    assert token_call["data"] == {"grant_type": "client_credentials"}
    assert token_call["headers"]["Authorization"].startswith("Basic ")
    assert listing_call["headers"]["Authorization"] == "Bearer tok"


def test_reddit_token_is_reused_across_subreddits(session):
    session.add("api/v1/access_token", {"access_token": "tok", "expires_in": 3600})
    session.add("/r/rust/hot", reddit_listing("r1"))
    session.add("/r/programming/hot", reddit_listing("p1"))
    src = RedditSource(
        client=make_client(session),
        config={"subreddits": ["rust", "programming"]},
        credentials=("id", "secret"),
    )
    src.fetch(limit=10)
    token_calls = [c for c in session.calls if "access_token" in c["url"]]
    assert len(token_calls) == 1, "the token must be cached, not refetched per subreddit"


def test_reddit_skips_stickied_posts(session):
    session.add("api/v1/access_token", {"access_token": "tok", "expires_in": 3600})
    session.add("/r/programming/hot", reddit_listing(stickied=True))
    src = RedditSource(
        client=make_client(session), config={"subreddits": ["programming"]},
        credentials=("id", "secret"),
    )
    assert src.fetch(limit=10) == []


def test_reddit_bad_token_response_raises_unavailable(session):
    session.add("api/v1/access_token", {"error": "invalid_grant"})
    src = RedditSource(
        client=make_client(session), config={"subreddits": ["programming"]},
        credentials=("id", "secret"),
    )
    with pytest.raises(SourceUnavailable, match="no access_token"):
        src.fetch()


# --- GitHub --------------------------------------------------------------

GH_REPO = {
    "id": 777,
    "full_name": "acme/widget",
    "description": "A fast widget",
    "html_url": "https://github.com/acme/widget",
    "owner": {"login": "acme"},
    "stargazers_count": 1500,
    "open_issues_count": 12,
    "language": "Rust",
    "topics": ["cli", "performance"],
    "created_at": "2026-09-01T00:00:00Z",
}


def test_github_fetch_uses_search_with_created_and_stars(session):
    session.add("search/repositories", {"items": [GH_REPO]})
    src = GitHubSource(
        client=make_client(session),
        config={"languages": ["rust"], "created_within_days": 30},
        token="t",
    )

    items = src.fetch(limit=10)
    assert items[0].id == "github:777"
    assert items[0].title.startswith("acme/widget:")
    assert items[0].score == 1500
    assert "rust" in items[0].tags and "cli" in items[0].tags

    params = session.calls[0]["params"]
    assert params["sort"] == "stars" and params["order"] == "desc"
    assert params["q"].startswith("created:>") and "language:rust" in params["q"]
    assert session.calls[0]["headers"]["Authorization"] == "Bearer t"


def test_github_works_without_a_token(session):
    session.add("search/repositories", {"items": [GH_REPO]})
    src = GitHubSource(client=make_client(session), config={"languages": []}, token="")
    assert len(src.fetch(limit=5)) == 1
    assert "Authorization" not in session.calls[0]["headers"]


def test_github_repo_without_description_still_maps(session):
    repo = dict(GH_REPO, description=None)
    session.add("search/repositories", {"items": [repo]})
    src = GitHubSource(client=make_client(session), config={"languages": []}, token="")
    assert "no description" in src.fetch(limit=1)[0].title


# --- registry ------------------------------------------------------------


def test_registry_covers_all_three_sources():
    assert set(REGISTRY) == {"hn", "reddit", "github"}


def test_finalize_drops_within_batch_duplicates(session):
    session.add("/topstories.json", [42, 42])
    session.add("/item/42.json", HN_STORY)
    src = HackerNewsSource(client=make_client(session), config={"lists": ["topstories"]})
    assert len(src.fetch(limit=2)) == 1
