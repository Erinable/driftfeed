"""Command line interface.

`argparse` over click/typer: the command set is small and flat, argparse is in
the standard library, and `requests` is the only runtime dependency worth
carrying for a tool that is supposed to be trivial to install. See the README.
"""

from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from collections.abc import Sequence

from driftfeed import __version__
from driftfeed.app import App
from driftfeed.config import Config, config_path, db_path, github_token, reddit_credentials
from driftfeed.models import (
    EVENT_CLICK,
    EVENT_DWELL,
    EVENT_HIDE,
    EVENT_SAVE,
    EVENT_SKIP,
)
from driftfeed.ranking.ranker import Scored
from driftfeed.sources import REGISTRY
from driftfeed.storage import Database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="driftfeed",
        description="Single-user local feed recommender (embeddings + online learning + bandit).",
    )
    parser.add_argument("--version", action="version", version=f"driftfeed {__version__}")
    parser.add_argument("--db", help="override the sqlite path for this invocation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create the database and a default config")
    p_init.add_argument("--force-config", action="store_true",
                        help="overwrite an existing config file with defaults")

    p_fetch = sub.add_parser("fetch", help="pull items from sources into the database")
    p_fetch.add_argument("--source", action="append", dest="sources",
                         choices=sorted(REGISTRY), help="limit to one source (repeatable)")
    p_fetch.add_argument("--seed", action="store_true",
                         help="also run keyword searches for the configured seed terms")

    p_feed = sub.add_parser("feed", help="show the ranked feed")
    p_feed.add_argument("-n", "--limit", type=int, default=20)
    p_feed.add_argument("--explain", action="store_true",
                        help="show the relevance/exploration breakdown per item")
    p_feed.add_argument("--no-impressions", action="store_true",
                        help="do not record impression feedback for what is shown")

    for name, help_text in (
        ("open", "open an item in the browser and record a click"),
        ("read", "record a click plus dwell time without opening a browser"),
        ("save", "record an explicit positive"),
        ("skip", "record an explicit negative"),
        ("hide", "record a negative and never show the item again"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("position", type=int, help="1-based position in the last rendered feed")
        if name in ("open", "read"):
            p.add_argument("--dwell", type=float, default=0.0,
                           help="seconds spent on the item (recorded as dwell feedback)")

    sub.add_parser("stats", help="feedback and model overview")

    p_retrain = sub.add_parser("retrain", help="replay the feedback log from scratch")
    p_retrain.add_argument("--epochs", type=int, default=1)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    path = args.db or db_path()

    try:
        with Database(path) as db:
            config = Config.load()
            app = App(db, config)
            return _dispatch(args, app, db, config)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


def _dispatch(args: argparse.Namespace, app: App, db: Database, config: Config) -> int:
    command = args.command
    if command == "init":
        return _cmd_init(args, app, db, config)
    if command == "fetch":
        return _cmd_fetch(args, app)
    if command == "feed":
        return _cmd_feed(args, app)
    if command in ("open", "read", "save", "skip", "hide"):
        return _cmd_feedback(args, app)
    if command == "stats":
        return _cmd_stats(app)
    if command == "retrain":
        applied = app.ranker.replay_feedback(
            seed_keywords=config.seed_keywords, epochs=args.epochs
        )
        print(f"replayed {applied} feedback events over {args.epochs} epoch(s)")
        return 0
    raise AssertionError(f"unhandled command {command!r}")


def _cmd_init(args: argparse.Namespace, app: App, db: Database, config: Config) -> int:
    written = config.write_default(overwrite=args.force_config)
    print(f"database  {db.path} (schema v{db.schema_version})")
    print(f"config    {written}")
    print(f"embedder  {app.ranker.embedder.model_id}")
    print()
    print("credentials detected:")
    reddit_status = "yes" if reddit_credentials() else "no  (DRIFTFEED_REDDIT_CLIENT_ID/SECRET)"
    print(f"  reddit  {reddit_status}")
    print(f"  github  {'yes' if github_token() else 'no  (GITHUB_TOKEN, or `gh auth login`)'}")
    print()
    print("next: driftfeed fetch --source hn && driftfeed feed")
    return 0


def _cmd_fetch(args: argparse.Namespace, app: App) -> int:
    reports = app.fetch(args.sources, seed=args.seed)
    failed = 0
    for r in reports:
        if r.ok:
            print(
                f"{r.source:8s} fetched {r.fetched:4d}  "
                f"new {r.inserted:4d}  updated {r.updated:4d}"
            )
        else:
            failed += 1
            print(f"{r.source:8s} skipped: {r.error}", file=sys.stderr)
    if not reports:
        print("no sources enabled — check your config", file=sys.stderr)
        return 1
    # Partial success is still success: HN working while Reddit lacks credentials
    # is the expected first-run state.
    return 0 if failed < len(reports) else 1


def _cmd_feed(args: argparse.Namespace, app: App) -> int:
    ranked = app.feed(limit=args.limit, record_impressions=not args.no_impressions)
    if not ranked:
        print("feed is empty — run `driftfeed fetch` first")
        return 0
    for i, s in enumerate(ranked, start=1):
        print(_format_row(i, s))
        if args.explain:
            print(_format_explain(s))
    print()
    print(f"eps={app.ranker.epsilon:.2f}  "
          f"updates={getattr(app.ranker.scorer, 'updates', 0)}  "
          f"open <n> / save <n> / skip <n> to teach it")
    return 0


def _cmd_feedback(args: argparse.Namespace, app: App) -> int:
    try:
        item = app.resolve_position(args.position)
    except LookupError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    command = args.command
    if command == "open":
        app.record(item, EVENT_CLICK)
        print(f"opening {item.url}")
        opened = webbrowser.open(item.url)
        if not opened:
            print("(no browser available — the click was still recorded)", file=sys.stderr)
        dwell = getattr(args, "dwell", 0.0)
        if dwell > 0:
            app.record(item, EVENT_DWELL, duration_s=dwell)
    elif command == "read":
        app.record(item, EVENT_CLICK)
        dwell = getattr(args, "dwell", 0.0) or 30.0
        app.record(item, EVENT_DWELL, duration_s=dwell)
        print(f"recorded read ({dwell:.0f}s): {item.title}")
        print(item.url)
    else:
        event = {"save": EVENT_SAVE, "skip": EVENT_SKIP, "hide": EVENT_HIDE}[command]
        app.record(item, event)
        print(f"{event}: {item.title}")
    return 0


def _cmd_stats(app: App) -> int:
    s = app.stats()
    print(f"database   {s['db_path']} (schema v{s['schema_version']})")
    print(f"items      {s['items']}  " + ", ".join(
        f"{k}={v}" for k, v in s["items_by_source"].items()) or "items      0")
    print(f"embeddings {s['embeddings']}  via {s['embedder']}")
    feedback = s["feedback"]
    feedback_summary = ", ".join(f"{k}={v}" for k, v in sorted(feedback.items())) or "none yet"
    print("feedback   " + feedback_summary)
    print(f"model      {s['scorer_updates']} updates, "
          f"eps={s['epsilon']}, {s['exploration_policy']} over {s['arms']} arms")
    if s["top_weights"]:
        print("weights    " + ", ".join(f"{n}={w:+.3f}" for n, w in s["top_weights"]))
    for name, state in s["sources"].items():
        when = state["last_fetch_at"]
        ago = f"{(time.time() - when) / 3600:.1f}h ago" if when else "never"
        print(f"source     {name:8s} last fetch {ago}")
    print(f"config     {config_path()}")
    return 0


def _format_row(i: int, s: Scored) -> str:
    also = f" +{'/'.join(s.also_on)}" if s.also_on else ""
    age = _age(s.item.created_at)
    return (
        f"{i:>3}. [{s.final:.3f}] {_clip(s.item.title, 84)}\n"
        f"     {s.item.source}{also} · {s.item.score}pts · {s.item.comment_count}c · {age}"
    )


def _format_explain(s: Scored) -> str:
    sim = s.features.get("sim", 0.0)
    return (
        f"     relevance={s.relevance:.3f} exploration={s.exploration:.3f} "
        f"arm={s.arm} sim={sim:+.3f} recency={s.features.get('recency', 0.0):.3f} "
        f"pop={s.features.get('popularity', 0.0):.3f}"
    )


def _clip(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _age(created_at: float) -> str:
    if created_at <= 0:
        return "unknown age"
    hours = (time.time() - created_at) / 3600
    if hours < 1:
        return f"{max(1, int(hours * 60))}m"
    if hours < 48:
        return f"{hours:.0f}h"
    return f"{hours / 24:.0f}d"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
