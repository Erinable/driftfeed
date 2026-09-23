"""Reddit.

Requires OAuth2. We use the client-credentials grant, which an "installed app"
or "script" app gets for free and which is enough for reading public listings —
no user context needed. Credentials come from the environment only.

Rate limits: Reddit publishes 60 requests/minute for OAuth clients, hence the
1s floor between calls. `X-Ratelimit-Remaining` is read opportunistically so we
slow down before being told to.
"""

from __future__ import annotations

import time

from driftfeed.config import reddit_credentials
from driftfeed.models import Item
from driftfeed.sources.base import Source, SourceUnavailable
from driftfeed.util.http import HttpError

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
OAUTH_BASE = "https://oauth.reddit.com"
VALID_LISTINGS = ("hot", "new", "top", "rising", "best")


class RedditSource(Source):
    name = "reddit"
    min_interval_s = 1.0  # 60 req/min published limit.

    def __init__(self, *, credentials: tuple[str, str] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._credentials = credentials or reddit_credentials()
        self._token: str | None = None
        self._token_expires_at = 0.0

    @property
    def available(self) -> bool:
        return self._credentials is not None

    def _access_token(self) -> str:
        if self._credentials is None:
            raise SourceUnavailable(
                "Reddit needs DRIFTFEED_REDDIT_CLIENT_ID and "
                "DRIFTFEED_REDDIT_CLIENT_SECRET in the environment (see .env.example)"
            )
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token

        client_id, client_secret = self._credentials
        import base64

        basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        try:
            payload = self.client.post_json(
                TOKEN_URL,
                data={"grant_type": "client_credentials"},
                headers={
                    "Authorization": f"Basic {basic}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
        except HttpError as exc:
            raise SourceUnavailable(f"Reddit token request failed: {exc}") from exc

        token = (payload or {}).get("access_token")
        if not token:
            raise SourceUnavailable("Reddit token response contained no access_token")
        self._token = str(token)
        self._token_expires_at = time.time() + float((payload or {}).get("expires_in") or 3600)
        return self._token

    def fetch(self, *, limit: int = 50) -> list[Item]:
        subs = [s for s in self.config.get("subreddits", []) if s]
        if not subs:
            return []
        listing = self.config.get("listing", "hot")
        if listing not in VALID_LISTINGS:
            listing = "hot"
        token = self._access_token()
        per_sub = max(1, min(100, limit // len(subs)))

        items: list[Item] = []
        for sub in subs:
            try:
                payload = self.client.get_json(
                    f"{OAUTH_BASE}/r/{sub}/{listing}",
                    params={"limit": per_sub, "raw_json": 1},
                    headers={"Authorization": f"Bearer {token}"},
                )
            except HttpError:
                # One dead or private subreddit should not sink the whole fetch.
                continue
            items.extend(self._parse_listing(payload))
        return self.finalize(items)

    def search(self, query: str, *, limit: int = 25) -> list[Item]:
        if not self.available:
            return []
        try:
            payload = self.client.get_json(
                f"{OAUTH_BASE}/search",
                params={"q": query, "limit": min(limit, 100), "type": "link", "raw_json": 1},
                headers={"Authorization": f"Bearer {self._access_token()}"},
            )
        except (HttpError, SourceUnavailable):
            return []
        return self.finalize(self._parse_listing(payload, extra_tags=[f"seed:{query}"]))

    def _parse_listing(self, payload: object, *, extra_tags: list[str] | None = None) -> list[Item]:
        children = ((payload or {}).get("data", {}) if isinstance(payload, dict) else {}).get(
            "children", []
        )
        out: list[Item] = []
        for child in children:
            data = (child or {}).get("data") or {}
            item = self._to_item(data, extra_tags or [])
            if item is not None:
                out.append(item)
        return out

    def _to_item(self, data: dict, extra_tags: list[str]) -> Item | None:
        post_id = data.get("id")
        title = (data.get("title") or "").strip()
        if not post_id or not title or data.get("stickied"):
            return None
        permalink = data.get("permalink") or ""
        discussion = f"https://www.reddit.com{permalink}" if permalink else ""
        url = (data.get("url_overridden_by_dest") or data.get("url") or "").strip() or discussion
        subreddit = data.get("subreddit") or ""
        tags = ["reddit"]
        if subreddit:
            tags.append(f"r/{subreddit}")
        flair = data.get("link_flair_text")
        if flair:
            tags.append(str(flair))
        return Item(
            source=self.name,
            source_id=str(post_id),
            url=url,
            title=title,
            body=(data.get("selftext") or "")[:4000],
            author=data.get("author") or "",
            score=int(data.get("score") or 0),
            comment_count=int(data.get("num_comments") or 0),
            tags=tags + extra_tags,
            created_at=float(data.get("created_utc") or 0.0),
        )
