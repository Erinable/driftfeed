"""Paths, environment lookups and the on-disk config file.

Everything credential-shaped is read from the environment only. The config file
holds subscriptions and tuning knobs — things you would happily commit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "driftfeed"

DEFAULT_CONFIG: dict = {
    "seed_keywords": [
        "rust",
        "python",
        "distributed systems",
        "compilers",
        "machine learning",
        "developer tools",
    ],
    "sources": {
        "hn": {"enabled": True, "lists": ["topstories", "beststories"], "limit": 60},
        "reddit": {
            "enabled": True,
            "subreddits": ["programming", "rust", "MachineLearning"],
            "listing": "hot",
            "limit": 40,
        },
        "github": {
            "enabled": True,
            "languages": ["python", "rust"],
            "created_within_days": 30,
            "limit": 40,
        },
    },
    "ranking": {
        "half_life_hours": 36.0,
        "max_per_source_in_top": 4,
        "exploration": "thompson",
    },
}


def data_dir() -> Path:
    """Platform-conventional app-data directory, overridable via DRIFTFEED_HOME."""
    override = os.environ.get("DRIFTFEED_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_NAME


def db_path() -> Path:
    override = os.environ.get("DRIFTFEED_DB")
    if override:
        return Path(override).expanduser()
    return data_dir() / "driftfeed.sqlite3"


def config_path() -> Path:
    override = os.environ.get("DRIFTFEED_CONFIG")
    if override:
        return Path(override).expanduser()
    return data_dir() / "config.json"


def user_agent() -> str:
    from driftfeed import __version__

    contact = os.environ.get("DRIFTFEED_USER_AGENT_CONTACT", "").strip()
    suffix = f" (+{contact})" if contact else " (+https://github.com/Erinable/driftfeed)"
    return f"driftfeed/{__version__}{suffix}"


def github_token() -> str | None:
    """Env first, then fall back to whatever `gh` is already logged in as."""
    for key in ("DRIFTFEED_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    try:
        out = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    token = out.stdout.strip()
    return token or None


def reddit_credentials() -> tuple[str, str] | None:
    cid = os.environ.get("DRIFTFEED_REDDIT_CLIENT_ID", "").strip()
    secret = os.environ.get("DRIFTFEED_REDDIT_CLIENT_SECRET", "").strip()
    if cid and secret:
        return cid, secret
    return None


@dataclass
class Config:
    """The parsed config file, with defaults filled in for anything missing."""

    seed_keywords: list[str] = field(default_factory=list)
    sources: dict = field(default_factory=dict)
    ranking: dict = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        path = path or config_path()
        raw = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"config at {path} is not valid JSON: {exc}") from exc
        merged = _deep_merge(DEFAULT_CONFIG, raw)
        return cls(
            seed_keywords=list(merged["seed_keywords"]),
            sources=merged["sources"],
            ranking=merged["ranking"],
            path=path,
        )

    def write_default(self, path: Path | None = None, *, overwrite: bool = False) -> Path:
        path = path or self.path or config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            return path
        path.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n", encoding="utf-8")
        return path

    def source(self, name: str) -> dict:
        return dict(self.sources.get(name, {}))

    def enabled_sources(self) -> list[str]:
        return [name for name, cfg in self.sources.items() if cfg.get("enabled", True)]


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out
