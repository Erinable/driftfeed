"""Hacker News.

Two APIs, no authentication on either:
  * Firebase (`hacker-news.firebaseio.com`) for the ranked id lists. One request
    per story, so `limit` directly bounds the request count.
  * Algolia (`hn.algolia.com/api/v1`) for keyword search, which Firebase lacks.
"""

from __future__ import annotations

import html
import time

from driftfeed.models import Item
from driftfeed.sources.base import Source
from driftfeed.util.http import HttpError

FIREBASE = "https://hacker-news.firebaseio.com/v0"
ALGOLIA = "https://hn.algolia.com/api/v1"
VALID_LISTS = ("topstories", "newstories", "beststories", "askstories", "showstories")


class HackerNewsSource(Source):
    name = "hn"
    min_interval_s = 0.05  # Firebase is a CDN-fronted static read; this is polite enough.

    def fetch(self, *, limit: int = 50) -> list[Item]:
        lists = [
            name for name in self.config.get("lists", ["topstories"]) if name in VALID_LISTS
        ] or ["topstories"]
        per_list = max(1, limit // len(lists))

        items: list[Item] = []
        for listing in lists:
            try:
                ids = self.client.get_json(f"{FIREBASE}/{listing}.json") or []
            except HttpError:
                continue
            for story_id in ids[:per_list]:
                item = self._fetch_story(story_id)
                if item is not None:
                    items.append(item)
        return self.finalize(items)

    def _fetch_story(self, story_id: int) -> Item | None:
        try:
            raw = self.client.get_json(f"{FIREBASE}/item/{story_id}.json")
        except HttpError:
            return None
        if not isinstance(raw, dict) or raw.get("deleted") or raw.get("dead"):
            return None
        if raw.get("type") not in (None, "story"):
            return None
        return self._to_item(raw)

    def _to_item(self, raw: dict) -> Item | None:
        story_id = raw.get("id")
        title = (raw.get("title") or "").strip()
        if story_id is None or not title:
            return None
        discussion = f"https://news.ycombinator.com/item?id={story_id}"
        # Ask/Show posts have no external link; the thread itself is the item.
        url = (raw.get("url") or "").strip() or discussion
        body = html.unescape(raw.get("text") or "")
        return Item(
            source=self.name,
            source_id=str(story_id),
            url=url,
            title=html.unescape(title),
            body=_strip_tags(body),
            author=raw.get("by") or "",
            score=int(raw.get("score") or 0),
            comment_count=int(raw.get("descendants") or 0),
            tags=["hn"],
            created_at=float(raw.get("time") or 0.0),
        )

    def search(self, query: str, *, limit: int = 25) -> list[Item]:
        try:
            payload = self.client.get_json(
                f"{ALGOLIA}/search",
                params={"query": query, "tags": "story", "hitsPerPage": min(limit, 100)},
            )
        except HttpError:
            return []
        items: list[Item] = []
        for hit in (payload or {}).get("hits", []):
            item = self._from_algolia(hit, query)
            if item is not None:
                items.append(item)
        return self.finalize(items)

    def _from_algolia(self, hit: dict, query: str) -> Item | None:
        story_id = hit.get("objectID")
        title = (hit.get("title") or hit.get("story_title") or "").strip()
        if not story_id or not title:
            return None
        discussion = f"https://news.ycombinator.com/item?id={story_id}"
        url = (hit.get("url") or hit.get("story_url") or "").strip() or discussion
        return Item(
            source=self.name,
            source_id=str(story_id),
            url=url,
            title=html.unescape(title),
            body=_strip_tags(html.unescape(hit.get("story_text") or "")),
            author=hit.get("author") or "",
            score=int(hit.get("points") or 0),
            comment_count=int(hit.get("num_comments") or 0),
            tags=["hn", f"seed:{query}"],
            created_at=float(hit.get("created_at_i") or time.time()),
        )


def _strip_tags(text: str) -> str:
    """HN bodies carry a little HTML; the embedder wants plain text."""
    out, depth = [], 0
    for ch in text:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return " ".join("".join(out).split())
