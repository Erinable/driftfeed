"""The storage layer: a thin, typed wrapper over sqlite3.

No ORM. The queries are few and the schema is small enough that raw SQL is the
clearer choice, and it keeps the dependency list to `requests`.
"""

from __future__ import annotations

import array
import json
import sqlite3
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from driftfeed.models import EVENT_TYPES, Feedback, Item
from driftfeed.storage.schema import LATEST_VERSION, current_version, migrate


def pack_vector(vec: Sequence[float]) -> bytes:
    """Float32 little-endian blob. Compact, and numpy-free on the read path."""
    return array.array("f", list(vec)).tobytes()


def unpack_vector(blob: bytes) -> list[float]:
    arr = array.array("f")
    arr.frombytes(blob)
    return list(arr)


class Database:
    """Owns the connection. Use as a context manager, or call `close()`."""

    def __init__(self, path: Path | str, *, auto_migrate: bool = True) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        if auto_migrate:
            migrate(self.conn)

    # --- lifecycle -------------------------------------------------------

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    @property
    def schema_version(self) -> int:
        return current_version(self.conn)

    @property
    def expected_schema_version(self) -> int:
        return LATEST_VERSION

    # --- items -----------------------------------------------------------

    def upsert_items(self, items: Iterable[Item]) -> tuple[int, int]:
        """Insert or refresh items. Returns (inserted, updated).

        Volatile fields (score, comment_count, fetched_at) are refreshed on
        conflict; `created_at` is not, since a source re-reporting a different
        creation time is noise rather than news.
        """
        inserted = updated = 0
        with self.conn:
            for item in items:
                row = item.as_row()
                existing = self.conn.execute(
                    "SELECT 1 FROM items WHERE id = ?", (row["id"],)
                ).fetchone()
                self.conn.execute(
                    """
                    INSERT INTO items (id, source, source_id, url, canonical_url, title,
                                       body, author, score, comment_count, tags,
                                       created_at, fetched_at)
                    VALUES (:id, :source, :source_id, :url, :canonical_url, :title,
                            :body, :author, :score, :comment_count, :tags,
                            :created_at, :fetched_at)
                    ON CONFLICT (id) DO UPDATE SET
                        title         = excluded.title,
                        body          = excluded.body,
                        score         = excluded.score,
                        comment_count = excluded.comment_count,
                        tags          = excluded.tags,
                        fetched_at    = excluded.fetched_at
                    """,
                    row,
                )
                self.conn.execute(
                    """
                    INSERT INTO item_aliases (canonical_url, item_id, source)
                    VALUES (?, ?, ?) ON CONFLICT DO NOTHING
                    """,
                    (row["canonical_url"], row["id"], row["source"]),
                )
                if existing:
                    updated += 1
                else:
                    inserted += 1
        return inserted, updated

    def get_item(self, item_id: str) -> Item | None:
        row = self.conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        return _row_to_item(row) if row else None

    def candidate_items(
        self,
        *,
        limit: int = 300,
        max_age_days: float | None = 21.0,
        exclude_hidden: bool = True,
    ) -> list[Item]:
        """Items eligible for ranking, newest first.

        Hidden items are dropped here rather than in the ranker: a hide is a
        standing instruction, not a score adjustment.
        """
        clauses, params = [], []
        if max_age_days is not None:
            clauses.append("created_at >= ?")
            params.append(time.time() - max_age_days * 86400)
        if exclude_hidden:
            clauses.append("id NOT IN (SELECT item_id FROM feedback WHERE event = 'hide')")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self.conn.execute(
            f"SELECT * FROM items {where} ORDER BY created_at DESC LIMIT ?", params
        ).fetchall()
        return [_row_to_item(r) for r in rows]

    def aliases_for(self, canonical_url: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT source FROM item_aliases WHERE canonical_url = ? ORDER BY source",
            (canonical_url,),
        ).fetchall()
        return [r["source"] for r in rows]

    def count_items(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM items").fetchone()[0])

    def counts_by_source(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT source, COUNT(*) AS n FROM items GROUP BY source ORDER BY n DESC"
        ).fetchall()
        return {r["source"]: int(r["n"]) for r in rows}

    # --- feedback --------------------------------------------------------

    def add_feedback(self, fb: Feedback) -> None:
        if fb.event not in EVENT_TYPES:
            raise ValueError(f"unknown feedback event: {fb.event!r}")
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO feedback (item_id, event, duration_s, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (fb.item_id, fb.event, fb.duration_s, fb.created_at),
            )

    def feedback_for(self, item_id: str) -> list[Feedback]:
        rows = self.conn.execute(
            "SELECT * FROM feedback WHERE item_id = ? ORDER BY created_at", (item_id,)
        ).fetchall()
        return [_row_to_feedback(r) for r in rows]

    def all_feedback(self, *, events: Sequence[str] | None = None) -> list[Feedback]:
        if events:
            placeholders = ",".join("?" * len(events))
            rows = self.conn.execute(
                f"SELECT * FROM feedback WHERE event IN ({placeholders}) ORDER BY created_at",
                list(events),
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM feedback ORDER BY created_at").fetchall()
        return [_row_to_feedback(r) for r in rows]

    def counts_by_event(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT event, COUNT(*) AS n FROM feedback GROUP BY event"
        ).fetchall()
        return {r["event"]: int(r["n"]) for r in rows}

    # --- embeddings ------------------------------------------------------

    def put_embedding(self, item_id: str, model: str, vector: Sequence[float]) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO embeddings (item_id, model, dim, vector) VALUES (?, ?, ?, ?)
                ON CONFLICT (item_id) DO UPDATE SET
                    model = excluded.model, dim = excluded.dim, vector = excluded.vector
                """,
                (item_id, model, len(vector), pack_vector(vector)),
            )

    def get_embedding(self, item_id: str, *, model: str | None = None) -> list[float] | None:
        row = self.conn.execute(
            "SELECT model, vector FROM embeddings WHERE item_id = ?", (item_id,)
        ).fetchone()
        if row is None or (model is not None and row["model"] != model):
            return None
        return unpack_vector(row["vector"])

    def count_embeddings(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    # --- sources ---------------------------------------------------------

    def mark_fetched(self, source: str, *, cursor: str = "") -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO sources (name, enabled, last_fetch_at, cursor)
                VALUES (?, 1, ?, ?)
                ON CONFLICT (name) DO UPDATE SET
                    last_fetch_at = excluded.last_fetch_at, cursor = excluded.cursor
                """,
                (source, time.time(), cursor),
            )

    def source_state(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM sources ORDER BY name").fetchall()
        return {
            r["name"]: {
                "enabled": bool(r["enabled"]),
                "last_fetch_at": r["last_fetch_at"],
                "cursor": r["cursor"],
            }
            for r in rows
        }

    # --- kv --------------------------------------------------------------

    def set_state(self, key: str, value: Any) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO kv (key, value) VALUES (?, ?)
                ON CONFLICT (key) DO UPDATE SET value = excluded.value
                """,
                (key, json.dumps(value)),
            )

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return default


def _row_to_item(row: sqlite3.Row) -> Item:
    return Item(
        source=row["source"],
        source_id=row["source_id"],
        url=row["url"],
        canonical_url=row["canonical_url"],
        title=row["title"],
        body=row["body"],
        author=row["author"],
        score=int(row["score"]),
        comment_count=int(row["comment_count"]),
        tags=[t for t in (row["tags"] or "").split(",") if t],
        created_at=float(row["created_at"]),
        fetched_at=float(row["fetched_at"]),
    )


def _row_to_feedback(row: sqlite3.Row) -> Feedback:
    return Feedback(
        item_id=row["item_id"],
        event=row["event"],
        duration_s=float(row["duration_s"]),
        created_at=float(row["created_at"]),
    )
