# trav-ml-pipeline

Data pipeline and prediction models for Scandinavian harness racing (trav).
Phase 1 ingests raw data from ATG's public racing info API. Later phases add
feature engineering and probabilistic race outcome models evaluated against
the betting market.

Status: Phases 1 to 3 complete. The first real backfill is done (28,687 races,
2024 to 2026, Scandinavia, trot). On a walk-forward over 16,224 held-out races,
a forecaster combining the public odds with a point-in-time fundamental model
beats the market on out-of-sample log loss by a small but significant margin (see
Results below). The detailed plan is in docs/ROADMAP.md and the research notes in
docs/RESEARCH.md.

## Results

On a walk-forward over 16,224 held-out races (2025 to mid 2026), scored on
multinomial log loss against the public win market:

- The market is sharp and well calibrated. Raw market log loss 1.6351.
- A favourite-longshot recalibration of the market beats it by 0.40 percent
  skill, almost entirely a calibration fix since favourites are underbet.
- The combination of the fundamental model with the market beats the raw market
  by 0.64 percent and the recalibrated market by 0.24 percent skill, both
  significant. The extra gain is a small resolution gain from point-in-time
  features, mostly classic Scandinavian factors (barefoot, draw bias,
  start-method specialism, class movement).
- A standalone fundamental model that ignores the odds loses to the market, as
  expected for public data without pace or trip information.

The objective, lower out-of-sample log loss than the market, is met, modestly.
The effect is small and the cleanest confirmation is fresh future races
(Phase 4).

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
  splits.py    Date-based walk-forward splits with purge and embargo
  evaluate.py  Skill score, day-blocked bootstrap, Diebold-Mariano, Murphy
  features.py  Point-in-time features, re-runnable, reads only norm_ tables
  ratings.py   Running Elo and time-decayed shrunk rates
  model.py     Conditional logit, LightGBM, market recalibration, combination
  predict.py   Frozen-model prediction for one race day, the Phase 4 core
  score.py     Score stored prediction snapshots against realised outcomes
tools/
  fetch_samples.py  Pull a few raw JSON files for parser validation
  inspect_live.py   Dump a live pre-race payload to confirm the Phase 4 unknowns
tests/
  test_phase2.py     Offline end-to-end check on fabricated payloads
  test_evaluate.py   Offline checks of the evaluation harness
  test_features.py   Offline point-in-time checks of the feature build
  test_model.py      Offline checks of the modelling maths
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

# Build point-in-time features into the norm_features table.
python -m atg.features --db data/atg.sqlite

# Fit and evaluate the models. A fast fixed split (provisional, pre-walk-forward)
# by default, or the quotable walk-forward with --walk.
python -m atg.model --db data/atg.sqlite --walk

# Predict one race day with the frozen model (the Phase 4 prediction core).
python -m atg.predict --db data/atg.sqlite --date 2026-06-28 --save

# Score stored prediction snapshots against the realised outcomes.
python -m atg.score --db data/atg.sqlite

# Offline checks (no network, no database needed).
python tests/test_phase2.py
python tests/test_evaluate.py
python tests/test_features.py
python tests/test_model.py
python tests/test_predict.py
python tests/test_score.py
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

- [x] Phase 1: ingestion pipeline
- [x] Phase 2: normalisation and market benchmark. Real benchmark on the 2024
      to 2026 backfill: de-vigged win odds, trot, log loss 1.6352, Brier 0.7235.
- [x] Phase 3: features, models and walk-forward evaluation. The combination of
      the fundamental model with the market beats the market on held-out log
      loss by a small significant margin. See docs/ROADMAP.md and the Results
      section above.
- [ ] Phase 4: live pre-race logging (timestamped, before the off) to confirm
      the edge on fresh races, and a dashboard

## Notes on data use

This project uses ATG's public, read-only API for personal research. It is
the same API that powers atg.se. The crawler is slow and identifiable. Raw
collected data is not redistributed. Only code and aggregated results are
published.
