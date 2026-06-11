# CLAUDE.md

Project context for Claude Code sessions. Read this before making changes.

## Project

Research and portfolio project on Scandinavian harness racing (trav). The
measurable objective is to produce win probabilities with lower log loss
than the normalised public betting distribution, out of sample. Betting
profit is not a goal.

Data source: ATG's public racing info API.
Base URL: `https://www.atg.se/services/racinginfo/v1/api`

## Current state (June 2026)

Phase 1 (ingestion) is complete and tested live. The pipeline ingested 83
races and 35 games for 2025-06-04 to 2025-06-05 correctly, and re-runs are
idempotent. Phase 2 (normalisation and features) has not started.

Environment notes (June 2026):

- The NBIM corporate network blocks atg.se through its web filter, so
  ingestion cannot run from this machine on that network. The client
  handles this correctly: it retries, fails loudly, and leaves the day
  incomplete. Run ingestion from a network where atg.se is reachable.
- pip cannot reach files.pythonhosted.org on this network. Install
  packages with uv, which is configured against NBIM's JFrog Artifactory:
  `uv pip install -r requirements.txt --python .venv\Scripts\python.exe`
- Storage and explore were verified offline with fabricated payloads.
- This repository is local only. Do not push to any remote or create
  remote repositories unless explicitly instructed.

Code layout:

```
atg/
  client.py    Rate-limited, retrying HTTP client
  db.py        SQLite storage of complete raw JSON responses
  ingest.py    Idempotent day-by-day ingestion CLI
  explore.py   Sanity checks on the collected data
```

## Verified API facts (tested June 2026, treat as ground truth)

Endpoints:

- `calendar/day/{YYYY-MM-DD}`: tracks[] with race stubs (id, status) and
  games{} keyed by type (V75, V86, V64, V65, V5, V4, V3, dd, ld, vinnare,
  plats, trio, ...).
- `races/{raceId}`: distance, startMethod (auto/volte), track, starts[] with
  postPosition, per-start distance (handicap), horse (age, sex, money,
  shoes, sulky, trainer, pedigree, statistics), driver, result (place,
  finishOrder, kmTime, prizeMoney, finalOdds).
- `games/{gameId}`: pools with turnover and per-start betDistribution.

Id formats:

- Race: `YYYY-MM-DD_trackId_raceNo`
- Game: `TYPE_YYYY-MM-DD_trackId_legRaceNo`

Units:

- betDistribution is in hundredths of a percent (141 means 1.41%).
- Odds inside pool objects are in hundredths (3303 means 33.03).
- finalOdds in race results is a plain decimal.
- Statistics percentages are scaled by 100 (winPercentage 3076 means 30.76%).

Quirks:

- Horses that gallop or are disqualified can have place 0 and a missing
  kmTime while still having odds.
- shoes and sulky carry "changed" flags. These are useful features later.
- The API also covers Danish and French trot.

## Standing conventions (apply to all work)

- Raw-first: store complete JSON responses. Parsing and normalisation is a
  separate, re-runnable step. Ingestion must never depend on parsing.
- Ingestion stays idempotent and resumable. A day is marked complete only
  when every request succeeded.
- Politeness: request delay of at least 0.3 s, identifiable User-Agent,
  never parallelise requests against the API.
- Point-in-time discipline: every model feature must have been knowable
  before race start. finalOdds and betDistribution are post-race values.
  They are the evaluation benchmark and must never leak into features.
- Evaluation: time-based train/test splits only, never random splits.
- Writing style everywhere (README, docstrings, comments, docs): plain,
  formal, simple English. Short declarative sentences. Avoid patterns that
  read as AI-generated: em dashes, bold lead-in bullet lists, rule-of-three
  constructions, marketing adjectives, rhetorical questions, exclamation
  marks. A comment states what the code does, not why it is great.

## Phase 2 plan (not started)

1. Normalise raw JSON into tables: races, starts, horses, drivers,
   trainers, game_legs.
2. Build the market benchmark first: calibration and log loss of normalised
   win odds. Every model must beat this number.
3. Feature engineering: recent form (km times adjusted for distance and
   start method), post position by track and start method, driver and
   trainer statistics, class (prize money), equipment changes, days since
   last start.
4. Baseline: conditional logistic regression per race, then LightGBM with a
   ranking objective.
