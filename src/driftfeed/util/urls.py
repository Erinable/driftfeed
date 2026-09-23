"""URL normalisation for cross-source dedup.

The same article reaches us as an HN story, a Reddit link post and sometimes a
GitHub repo reference. Collapsing them needs a canonical form: lowercase host,
no `www.`, no tracking params, no trailing slash, no fragment.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_PREFIXES = ("utm_", "ga_", "mc_")
TRACKING_PARAMS = frozenset(
    {
        "fbclid",
        "gclid",
        "gclsrc",
        "dclid",
        "msclkid",
        "igshid",
        "mkt_tok",
        "ref",
        "ref_src",
        "ref_url",
        "referrer",
        "source",
        "spm",
        "share_id",
        "yclid",
        "_hsenc",
        "_hsmi",
    }
)

DEFAULT_PORTS = {"http": "80", "https": "443"}


def canonicalize(url: str) -> str:
    """Best-effort canonical form. Returns the input stripped if unparseable."""
    raw = (url or "").strip()
    if not raw:
        return ""
    if "//" not in raw.split("?", 1)[0]:
        raw = "https://" + raw.lstrip("/")

    parts = urlsplit(raw)
    scheme = (parts.scheme or "https").lower()
    if scheme not in ("http", "https"):
        # Leave mailto:, magnet: and friends alone rather than mangling them.
        return raw

    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    netloc = host
    if parts.port and str(parts.port) != DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{parts.port}"

    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"

    query = urlencode(
        sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if _keep(k))
    )
    # https is the canonical scheme even when the source reported http: the same
    # article served over both should dedup to one row.
    return urlunsplit(("https", netloc, path, query, ""))


def _keep(key: str) -> bool:
    low = key.lower()
    if low in TRACKING_PARAMS:
        return False
    return not any(low.startswith(p) for p in TRACKING_PREFIXES)


def domain_of(url: str) -> str:
    host = urlsplit(canonicalize(url)).hostname or ""
    return host
