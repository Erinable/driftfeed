# driftfeed

driftfeed is a single-user, local-first information feed recommender. It pulls
items from Hacker News, Reddit, and GitHub into SQLite, ranks them with content
embeddings plus online implicit-feedback learning, and keeps exploration active
with a bandit policy.

## Why a single-user model?

There is no useful population for collaborative filtering when one person owns
the system. driftfeed instead learns from that user's clicks, dwell time, saves,
skips, and hides. A seed-keyword profile provides a cold-start prior, while
Thompson sampling over source/topic arms keeps unfamiliar material in the feed.

## Architecture

The package is split into three replaceable layers:

1. **Aggregation** (`driftfeed.sources`) implements one `Source` interface per
   provider and emits the shared `Item` schema. Requests use a User-Agent,
   rate limits, retry/backoff, and URL canonicalisation for cross-source aliases.
2. **Storage** (`driftfeed.storage`) is a standard-library SQLite wrapper with
   ordered migrations. It stores items, feedback events, embeddings, source
   state, and model state. The default database is in the platform app-data
   directory and can be overridden with `DRIFTFEED_HOME` or `DRIFTFEED_DB`.
3. **Ranking** (`driftfeed.ranking`) separates `Embedder`, `Scorer`,
   `Explorer`, and `Ranker`. Hashing embeddings run without downloads. The
   optional Sentence Transformers backend is selected with
   `DRIFTFEED_EMBEDDER=sentence-transformers`.

The CLI uses `argparse` because the command set is small, flat, and covered by
the standard library; the only runtime dependency is `requests`.

## Install

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

For the optional local sentence-transformer model:

```bash
python -m pip install -e '.[st]'
export DRIFTFEED_EMBEDDER=sentence-transformers
```

## Quick start

Hacker News needs no credentials and is the easiest first-run path:

```bash
driftfeed init
driftfeed fetch --source hn
driftfeed feed
driftfeed read 1 --dwell 45
driftfeed save 2
driftfeed stats
```

`open N` launches the URL and records a click, `read N` records a click and
dwell time without launching a browser, and `skip N` / `hide N` teach the model
negative preferences. The position refers to the most recently rendered feed.

## Source configuration and credentials

`driftfeed init` writes a default JSON config with seed keywords and enabled
sources. Edit it at the path printed by `driftfeed stats`; set
`DRIFTFEED_CONFIG` to use another file.

Credentials are read only from environment variables and are never stored in
SQLite:

* Reddit OAuth client credentials: `DRIFTFEED_REDDIT_CLIENT_ID` and
  `DRIFTFEED_REDDIT_CLIENT_SECRET`.
* GitHub: `GITHUB_TOKEN`, `GH_TOKEN`, or `DRIFTFEED_GITHUB_TOKEN`; if none is
  set, a logged-in `gh auth token` is used when available.

See `.env.example` for all path and model settings. The adapters use conservative
request intervals and retry 429/5xx responses with exponential backoff.

## Development

All tests use mocked HTTP sessions and never call real providers:

```bash
pytest
ruff check .
```

The GitHub Actions workflow runs both checks on pushes and pull requests for
Python 3.10, 3.11, and 3.12.

## Roadmap

* Evaluate embedding backends on a user's saved/skipped corpus and make model
  selection configurable per profile.
* Tune Thompson priors and arm definitions from observed feedback, then compare
  against UCB1 with offline replay metrics.
* Add more source adapters and optional background refresh scheduling.
* Add richer ranking explanations and export/import of local model state.

## License

MIT; see `LICENSE`.
