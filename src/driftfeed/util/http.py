"""A small polite HTTP client.

Every outbound request carries a User-Agent, respects a minimum interval between
calls, and backs off on 429/5xx while honouring `Retry-After`. Adapters are
expected to go through this rather than calling `requests` directly.
"""

from __future__ import annotations

import random
import time
from typing import Any

import requests

from driftfeed.config import user_agent


class HttpError(RuntimeError):
    """A request failed after exhausting retries."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class RateLimiter:
    """Minimum wall-clock gap between requests on one client."""

    def __init__(self, min_interval_s: float = 0.0, sleep=time.sleep, clock=time.monotonic) -> None:
        self.min_interval_s = min_interval_s
        self._sleep = sleep
        self._clock = clock
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval_s <= 0:
            return
        elapsed = self._clock() - self._last
        remaining = self.min_interval_s - elapsed
        if remaining > 0:
            self._sleep(remaining)
        self._last = self._clock()


class HttpClient:
    """Retrying JSON client. `session` is injectable so tests never touch the network."""

    RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        *,
        session: Any | None = None,
        min_interval_s: float = 0.0,
        max_retries: int = 3,
        timeout_s: float = 15.0,
        backoff_base_s: float = 0.5,
        sleep=time.sleep,
    ) -> None:
        self.session = session if session is not None else requests.Session()
        self.limiter = RateLimiter(min_interval_s, sleep=sleep)
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.backoff_base_s = backoff_base_s
        self._sleep = sleep

    def _headers(self, extra: dict[str, str] | None) -> dict[str, str]:
        headers = {"User-Agent": user_agent(), "Accept": "application/json"}
        if extra:
            headers.update(extra)
        return headers

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict | None = None,
        data: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Perform the request and return decoded JSON. Raises HttpError on failure."""
        last_error: str = "no attempt made"
        last_status: int | None = None
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            try:
                resp = self.session.request(
                    method,
                    url,
                    params=params,
                    data=data,
                    headers=self._headers(headers),
                    timeout=self.timeout_s,
                )
            except requests.RequestException as exc:
                last_error, last_status = f"{type(exc).__name__}: {exc}", None
            else:
                status = int(getattr(resp, "status_code", 0))
                if 200 <= status < 300:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise HttpError(
                            f"{url} returned non-JSON body: {exc}", status=status
                        ) from exc
                last_status = status
                last_error = f"HTTP {status} from {url}"
                if status not in self.RETRY_STATUS:
                    raise HttpError(last_error, status=status)
                retry_after = _retry_after(resp)
                if retry_after is not None and attempt < self.max_retries:
                    self._sleep(retry_after)
                    continue

            if attempt >= self.max_retries:
                break
            # Exponential backoff with jitter, so repeated failures do not
            # hammer a source that is already struggling.
            delay = self.backoff_base_s * (2**attempt)
            self._sleep(delay + random.uniform(0, delay * 0.25))

        raise HttpError(f"giving up after {self.max_retries + 1} attempts: {last_error}",
                        status=last_status)

    def get_json(self, url: str, **kwargs: Any) -> Any:
        return self.request("GET", url, **kwargs)

    def post_json(self, url: str, **kwargs: Any) -> Any:
        return self.request("POST", url, **kwargs)


def _retry_after(resp: Any) -> float | None:
    headers = getattr(resp, "headers", None) or {}
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if not raw:
        return None
    try:
        # Cap it: a source asking us to sleep for an hour means "come back later",
        # not "block the CLI".
        return min(float(raw), 60.0)
    except (TypeError, ValueError):
        return None
