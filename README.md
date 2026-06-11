# trav-ml-pipeline

Data pipeline and prediction models for Scandinavian harness racing (trav).
Phase 1 ingests raw data from ATG's public racing info API. Later phases add
feature engineering and probabilistic race outcome models evaluated against
the betting market.

Status: Phase 1, data ingestion.

## Research question

The objective is to produce win probabilities with lower log loss than the
normalised public betting distribution, out of sample. Betting profit is not
a goal. Pari-mutuel takeout makes profit unrealistic. The betting market
serves as the public baseline, and every prediction has a verifiable outcome.

## Architecture (Phase 1)

```
atg/
  client.py    Rate-limited, retrying HTTP client for the ATG racing info API
  db.py        SQLite storage of complete raw JSON responses
  ingest.py    Idempotent day-by-day ingestion CLI, resumable and re-runnable
  explore.py   Sanity checks on the collected data
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

# Backfill a full year. This takes a few hours and is safe to interrupt
# and resume.
python -m atg.ingest --from 2024-01-01 --to 2024-12-31
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

## Roadmap

- [x] Phase 1: ingestion pipeline (this repo)
- [ ] Phase 2: normalisation and feature engineering (form, class,
      driver and trainer statistics, post position, equipment changes)
- [ ] Phase 3: models. Conditional logit baseline, then LightGBM ranking.
      Calibration and time-split backtest against market log loss.
- [ ] Phase 4: live prediction logging (timestamped, pre-race) and a dashboard

## Notes on data use

This project uses ATG's public, read-only API for personal research. It is
the same API that powers atg.se. The crawler is slow and identifiable. Raw
collected data is not redistributed. Only code and aggregated results are
published.
