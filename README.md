# trav-ml-pipeline

Data pipeline and prediction models for Scandinavian harness racing (trav).
Phase 1 ingests raw data from ATG's public racing info API. Later phases add
feature engineering and probabilistic race outcome models evaluated against
the betting market.

Status: Phase 1 (ingestion) complete. Phase 2 (normalisation and the market
benchmark) written and validated offline, pending a real backfill.

## Research question

The objective is to produce win probabilities with lower log loss than the
normalised public betting distribution, out of sample. Betting profit is not
a goal. Pari-mutuel takeout makes profit unrealistic. The betting market
serves as the public baseline, and every prediction has a verifiable outcome.

## Architecture

```
atg/
  client.py    Rate-limited, retrying HTTP client for the ATG racing info API
  db.py        SQLite storage of complete raw JSON responses
  ingest.py    Idempotent day-by-day ingestion CLI, resumable and re-runnable
  explore.py   Sanity checks on the collected data
  normalize.py Raw JSON to analytical tables, re-runnable, reads only raw data
  benchmark.py Market win probabilities and benchmark log loss and Brier
  metrics.py   Log loss, Brier, calibration, paired bootstrap
tools/
  fetch_samples.py  Pull a few raw JSON files for parser validation
tests/
  test_phase2.py     Offline end-to-end check on fabricated payloads
  build_sample_db.py Build a small DB from raw JSON in data/samples/
```

Design principles:

- Raw-first storage. The complete API response is stored for every race and
  game. Parsing and normalisation happen in a later phase and can be re-run
  from the raw data. A parsing bug never forces a re-download.
- Idempotent ingestion. Completed days are recorded in an ingest log. The CLI
  can be interrupted and resumed, and re-runs skip completed days. A day with
  any failed request is left incomplete and is retried on the next run.
- Polite crawling. Requests are separated by a fixed delay (default 0.4 s)
  with exponential backoff on errors and an identifiable User-Agent. A racing
  day takes roughly 50 to 70 requests.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Ingest one week
python -m atg.ingest --from 2025-06-01 --to 2025-06-07

# Inspect what was collected
python -m atg.explore

# Backfill. Safe to interrupt and resume. Default scope is Scandinavia
# (SE, DK, NO). Use --countries SE to narrow, or --countries all to widen.
python -m atg.ingest --from 2023-01-01 --to 2025-12-31

# Lean backfill for tight disk. --skip-games drops the V game pools, which
# are about 60 percent of the storage and hold only the secondary spelprocent.
# The primary odds benchmark and all features come from races, so results are
# unaffected. Roughly 2 to 3 GB instead of 5 to 8 GB for 3.5 years.
python -m atg.ingest --from 2023-01-01 --to 2025-12-31 --skip-games

# Pull a few raw JSON files for parser validation, on a machine with access.
python -m tools.fetch_samples --date 2025-06-07
```

Phase 2 runs on the local SQLite file and needs no network.

```bash
# Normalise raw JSON into analytical tables and print a validation report.
python -m atg.normalize --db data/atg.sqlite

# Compute the market benchmark (log loss, Brier, calibration plot).
python -m atg.benchmark --db data/atg.sqlite

# Offline check of the whole Phase 2 chain on fabricated data.
python tests/test_phase2.py
```

Data is written to `data/atg.sqlite`, which is gitignored.

## Data model

```
calendar/day/{date}          tracks -> races (ids, status), games (V75, V86, ...)
races/{raceId}               distance, startMethod, track
                             starts[] -> postPosition, distance (handicap)
                                         horse: age, sex, money, shoes, sulky,
                                                trainer, pedigree, statistics
                                         driver: id, name, license, stats
                                         result: place, finishOrder, kmTime,
                                                 prizeMoney, finalOdds
games/{gameId}               pool turnover, per-start betDistribution
                             (the market data used as the benchmark)
```

Id formats: race `YYYY-MM-DD_trackId_raceNo`, game
`TYPE_YYYY-MM-DD_trackId_legRaceNo`.

Normalisation produces three derived tables in the same SQLite file, rebuilt
on each run:

```
norm_races             one row per race
norm_starts            one row per horse per race, the analytical unit.
                       Post-race columns (place, km_time_s, final_odds,
                       is_winner) are labels and benchmark inputs only.
norm_bet_distribution  one row per horse per pool leg, share of the pool
```

## Roadmap

- [x] Phase 1: ingestion pipeline (this repo)
- [~] Phase 2: normalisation and market benchmark. Code written and validated
      offline. Real benchmark numbers pending a backfill.
- [ ] Phase 3: models. Conditional logit baseline, then LightGBM ranking.
      Calibration and time-split backtest against market log loss.
- [ ] Phase 4: live prediction logging (timestamped, pre-race) and a dashboard

## Notes on data use

This project uses ATG's public, read-only API for personal research. It is
the same API that powers atg.se. The crawler is slow and identifiable. Raw
collected data is not redistributed. Only code and aggregated results are
published.
