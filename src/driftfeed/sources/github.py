"""GitHub.

There is no official trending API — the trending page is HTML-only and scraping
it is both brittle and rude. We approximate it with the Search API:
`created:>DATE sort:stars`, which surfaces repos that gathered stars quickly.
That is not the same ranking, but it is the same intent and it is supported.

A token is optional (60 req/h unauthenticated, 5000 authenticated) and is picked
up from the environment or the local `gh` login.
"""

from __future__ import annotations

import datetime as dt

from driftfeed.config import github_token
from driftfeed.models import Item
from driftfeed.sources.base import Source
from driftfeed.util.http import HttpError

API = "https://api.github.com"


class GitHubSource(Source):
    name = "github"
    min_interval_s = 2.0  # Search API is rate-limited harder than the REST core.

    def __init__(self, *, token: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._token = token if token is not None else github_token()

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def fetch(self, *, limit: int = 40) -> list[Item]:
        days = int(self.config.get("created_within_days", 30) or 30)
        since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).date().isoformat()
        languages = list(self.config.get("languages", [])) or [None]
        per_lang = max(1, min(100, limit // len(languages)))

        items: list[Item] = []
        for language in languages:
            query = f"created:>{since}"
            if language:
                query += f" language:{language}"
            try:
                payload = self.client.get_json(
                    f"{API}/search/repositories",
                    params={
                        "q": query,
                        "sort": "stars",
                        "order": "desc",
                        "per_page": per_lang,
                    },
                    headers=self._headers(),
                )
            except HttpError:
                continue
            for repo in (payload or {}).get("items", []):
                item = self._to_item(repo)
                if item is not None:
                    items.append(item)
        return self.finalize(items)

    def search(self, query: str, *, limit: int = 25) -> list[Item]:
        try:
            payload = self.client.get_json(
                f"{API}/search/repositories",
                params={"q": query, "sort": "stars", "order": "desc",
                        "per_page": min(limit, 100)},
                headers=self._headers(),
            )
        except HttpError:
            return []
        items = []
        for repo in (payload or {}).get("items", []):
            item = self._to_item(repo, extra_tags=[f"seed:{query}"])
            if item is not None:
                items.append(item)
        return self.finalize(items)

    def _to_item(self, repo: dict, *, extra_tags: list[str] | None = None) -> Item | None:
        repo_id = repo.get("id")
        full_name = repo.get("full_name") or repo.get("name")
        if repo_id is None or not full_name:
            return None
        tags = ["github"]
        language = repo.get("language")
        if language:
            tags.append(str(language).lower())
        tags.extend(str(t) for t in (repo.get("topics") or [])[:8])
        owner = (repo.get("owner") or {}).get("login") or ""
        return Item(
            source=self.name,
            source_id=str(repo_id),
            url=repo.get("html_url") or f"https://github.com/{full_name}",
            # The repo name carries real signal ("ripgrep", "tokio"), so keep it
            # in the title rather than relying on the description alone.
            title=f"{full_name}: {repo.get('description') or 'no description'}".strip(),
            body=repo.get("description") or "",
            author=owner,
            score=int(repo.get("stargazers_count") or 0),
            comment_count=int(repo.get("open_issues_count") or 0),
            tags=tags + (extra_tags or []),
            created_at=_parse_ts(repo.get("created_at")),
        )


def _parse_ts(value: object) -> float:
    if not isinstance(value, str) or not value:
        return 0.0
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0
