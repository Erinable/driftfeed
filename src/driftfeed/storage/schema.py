"""Schema versioning.

Migrations are an ordered list of (version, SQL) pairs. `migrate()` applies every
migration whose version exceeds the recorded one, inside a single transaction,
and bumps `schema_version`. Adding a column later means appending a new pair —
never editing an existing one.
"""

from __future__ import annotations

import sqlite3

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS items (
            id             TEXT PRIMARY KEY,
            source         TEXT NOT NULL,
            source_id      TEXT NOT NULL,
            url            TEXT NOT NULL,
            canonical_url  TEXT NOT NULL,
            title          TEXT NOT NULL,
            body           TEXT NOT NULL DEFAULT '',
            author         TEXT NOT NULL DEFAULT '',
            score          INTEGER NOT NULL DEFAULT 0,
            comment_count  INTEGER NOT NULL DEFAULT 0,
            tags           TEXT NOT NULL DEFAULT '',
            created_at     REAL NOT NULL DEFAULT 0,
            fetched_at     REAL NOT NULL DEFAULT 0,
            UNIQUE (source, source_id)
        );
        CREATE INDEX IF NOT EXISTS idx_items_canonical ON items (canonical_url);
        CREATE INDEX IF NOT EXISTS idx_items_created ON items (created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_items_source ON items (source);

        -- A canonical URL seen from more than one source keeps a row per source,
        -- so "also on HN" stays visible after dedup collapses the feed.
        CREATE TABLE IF NOT EXISTS item_aliases (
            canonical_url TEXT NOT NULL,
            item_id       TEXT NOT NULL REFERENCES items (id) ON DELETE CASCADE,
            source        TEXT NOT NULL,
            PRIMARY KEY (canonical_url, item_id)
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id    TEXT NOT NULL,
            event      TEXT NOT NULL,
            duration_s REAL NOT NULL DEFAULT 0,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_item ON feedback (item_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback (created_at DESC);

        CREATE TABLE IF NOT EXISTS embeddings (
            item_id TEXT PRIMARY KEY REFERENCES items (id) ON DELETE CASCADE,
            model   TEXT NOT NULL,
            dim     INTEGER NOT NULL,
            vector  BLOB NOT NULL
        );

        -- Per-source subscription state the config file does not own (e.g. cursors).
        CREATE TABLE IF NOT EXISTS sources (
            name          TEXT PRIMARY KEY,
            enabled       INTEGER NOT NULL DEFAULT 1,
            last_fetch_at REAL NOT NULL DEFAULT 0,
            cursor        TEXT NOT NULL DEFAULT ''
        );

        -- Small mutable key/value state: model weights, bandit arms, the last
        -- rendered feed (so `driftfeed open 3` knows what "3" was).
        CREATE TABLE IF NOT EXISTS kv (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """,
    ),
]

LATEST_VERSION = MIGRATIONS[-1][0]


def current_version(conn: sqlite3.Connection) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations. Returns the version now on disk."""
    version = current_version(conn)
    for target, sql in MIGRATIONS:
        if target <= version:
            continue
        with conn:
            conn.executescript(sql)
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (target,))
        version = target
    return version
