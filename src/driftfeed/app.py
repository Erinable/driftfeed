"""Application service layer — what the CLI calls, minus the argument parsing.

Keeping this separate means the end-to-end tests exercise the same code paths as
the CLI without going through `argparse` or stdout.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from driftfeed.config import Config
from driftfeed.models import EVENT_IMPRESSION, Feedback, Item
from driftfeed.ranking import Ranker
from driftfeed.ranking.ranker import STATE_KEY_FEED, Scored
from driftfeed.sources import REGISTRY, Source, SourceUnavailable, build_source
from driftfeed.storage import Database


@dataclass
class FetchReport:
    source: str
    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


class App:
    """Ties config, storage and ranking together for one invocation."""

    def __init__(self, db: Database, config: Config, *, ranker: Ranker | None = None) -> None:
        self.db = db
        self.config = config
        self._ranker = ranker

    @property
    def ranker(self) -> Ranker:
        if self._ranker is None:
            ranking = self.config.ranking
            self._ranker = Ranker(
                self.db,
                half_life_hours=float(ranking.get("half_life_hours", 36.0)),
                max_per_source_in_top=int(ranking.get("max_per_source_in_top", 4)),
            )
        return self._ranker

    # --- fetch -----------------------------------------------------------

    def fetch(
        self,
        source_names: Sequence[str] | None = None,
        *,
        seed: bool = False,
        source_factory=build_source,
    ) -> list[FetchReport]:
        """Run the named adapters (default: all enabled) and persist their items."""
        names = list(source_names) if source_names else self.config.enabled_sources()
        reports: list[FetchReport] = []
        for name in names:
            if name not in REGISTRY:
                reports.append(FetchReport(source=name, error=f"unknown source {name!r}"))
                continue
            cfg = self.config.source(name)
            report = FetchReport(source=name)
            try:
                adapter: Source = source_factory(name, cfg)
                items: list[Item] = adapter.fetch(limit=int(cfg.get("limit", 50)))
                if seed:
                    items.extend(self._seed_items(adapter))
            except SourceUnavailable as exc:
                report.error = str(exc)
            except Exception as exc:  # noqa: BLE001 - one bad source must not abort the rest
                report.error = f"{type(exc).__name__}: {exc}"
            else:
                report.fetched = len(items)
                report.inserted, report.updated = self.db.upsert_items(items)
                self.db.mark_fetched(name)
            reports.append(report)
        return reports

    def _seed_items(self, adapter: Source) -> list[Item]:
        """Cold-start seeding: keyword search per configured seed term."""
        out: list[Item] = []
        for keyword in self.config.seed_keywords[:6]:
            try:
                out.extend(adapter.search(keyword, limit=10))
            except Exception:  # noqa: BLE001 - search is best-effort
                continue
        return out

    # --- feed ------------------------------------------------------------

    def feed(self, *, limit: int = 20, record_impressions: bool = True) -> list[Scored]:
        candidates = self.db.candidate_items(limit=400)
        ranked = self.ranker.rank(
            candidates, seed_keywords=self.config.seed_keywords, limit=limit
        )
        # Persist the rendered order so `open 3` / `save 3` resolve the same "3".
        self.db.set_state(
            STATE_KEY_FEED,
            {"at": time.time(), "item_ids": [s.item.id for s in ranked]},
        )
        if record_impressions:
            for s in ranked:
                self.db.add_feedback(Feedback(item_id=s.item.id, event=EVENT_IMPRESSION))
        return ranked

    def resolve_position(self, position: int) -> Item:
        """Map a 1-based feed position back to an item."""
        state = self.db.get_state(STATE_KEY_FEED) or {}
        ids = state.get("item_ids") or []
        if not ids:
            raise LookupError("no feed has been rendered yet — run `driftfeed feed` first")
        if position < 1 or position > len(ids):
            raise LookupError(f"position {position} is outside the last feed (1-{len(ids)})")
        item = self.db.get_item(ids[position - 1])
        if item is None:
            raise LookupError(f"item {ids[position - 1]} is no longer in the database")
        return item

    # --- feedback --------------------------------------------------------

    def record(self, item: Item, event: str, *, duration_s: float = 0.0) -> None:
        """Store a feedback event and learn from it immediately."""
        self.db.add_feedback(Feedback(item_id=item.id, event=event, duration_s=duration_s))
        self.ranker.learn_from(item, event, seed_keywords=self.config.seed_keywords)

    # --- stats -----------------------------------------------------------

    def stats(self) -> dict:
        scorer = self.ranker.scorer
        explorer = self.ranker.explorer
        arms = explorer.state().get("arms") or explorer.state().get("counts") or {}
        return {
            "db_path": str(self.db.path),
            "schema_version": self.db.schema_version,
            "items": self.db.count_items(),
            "items_by_source": self.db.counts_by_source(),
            "embeddings": self.db.count_embeddings(),
            "embedder": self.ranker.embedder.model_id,
            "feedback": self.db.counts_by_event(),
            "scorer_updates": getattr(scorer, "updates", 0),
            "top_weights": getattr(scorer, "top_weights", lambda n=8: [])(8),
            "epsilon": round(self.ranker.epsilon, 4),
            "exploration_policy": explorer.state().get("policy", "unknown"),
            "arms": len(arms),
            "sources": self.db.source_state(),
        }
