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
idempotent.

Phase 1 was later hardened. The client treats non-JSON 2xx bodies and
non-retryable 4xx responses explicitly, ingestion continues past a failed
day instead of aborting the batch, and games already stored are skipped.
Ingestion takes a --countries filter, default SE,DK,NO, since the project is
Scandinavian. Only marking-bet pools (V75, V64, ...) are collected, because
they alone carry betDistribution. vinnare and plats are not collected: the
win pool only repeats finalOdds and there is roughly one win and one place
pool per race, so collecting them tripled the crawl for no benefit.

Phase 2 (normalisation and the market benchmark) is written and validated.
The parser was confirmed against real race and game payloads (a 24 file
sample for 2020-06-06, 2022-06-04 and 2025-06-07), and the full chain runs on
a 51 race sample built from it. The modules are normalize.py, benchmark.py and
metrics.py. The market benchmark uses the de-vigged final win odds as the
primary signal. On the sample the median overround was 1.179, a takeout near
15 percent, which is the expected figure for a Swedish trot win pool. The
V game spelprocent is reported as a second signal where it exists. The sample
is far too small for the benchmark number itself to mean anything. A real
backfill is the next step. Feature engineering and models (Phase 3) have not
started.

Local helpers: tests/build_sample_db.py builds data/sample.sqlite from the raw
JSON in data/samples/. Both are gitignored.

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

Two-machine workflow (because of the network block):

1. Ingestion runs on a personal machine where atg.se is reachable.
   Transfer the repo there as a zip. Setup on that machine: create a
   venv, `pip install -r requirements.txt`, then run `python -m
   atg.ingest`. Backfill in month-sized chunks. One year takes a few
   hours and is safe to interrupt and resume.
2. The result is a single file, `data/atg.sqlite`. Zip it (raw JSON
   compresses roughly 10x) and copy it back to this laptop into
   `data/`. The directory is gitignored.
3. All later phases (normalisation, features, models, evaluation) run
   on this laptop against the local SQLite file and need no network.
4. Phase 4 (live pre-race logging) must eventually run on the personal
   machine or a small always-on box, since it needs the API near race
   start. The day-based idempotent CLI already fits a nightly
   incremental run.

Code layout:

```
atg/
  client.py    Rate-limited, retrying HTTP client
  db.py        SQLite storage of complete raw JSON responses
  ingest.py    Idempotent day-by-day ingestion CLI
  explore.py   Sanity checks on the collected data
  normalize.py Raw JSON to analytical tables (norm_races, norm_starts,
               norm_bet_distribution). Re-runnable, reads only raw tables.
  benchmark.py Market win probabilities and benchmark log loss and Brier
  metrics.py   Log loss, Brier, calibration, paired bootstrap
tools/
  fetch_samples.py  Pull a few raw JSON files for parser validation
tests/
  test_phase2.py    Offline end-to-end check on fabricated payloads
```

Derived tables added by normalize.py, all rebuilt from the raw tables on
each run:

- `norm_races`: one row per race.
- `norm_starts`: one row per horse per race. Post-race columns (place,
  km_time_s, final_odds, prize_money, is_winner, galloped_or_dq) are kept in
  the same table but are labels and benchmark inputs only, never features.
- `norm_bet_distribution`: one row per horse per pool leg, share as a
  fraction of the pool.

## Verified API facts (confirmed against real payloads, June 2026)

These were corrected after inspecting real race and game JSON. The earlier
version of this section had three errors, now fixed: betDistribution location,
the gallop encoding, and the assumption that sport is always trot. Data is
available back to at least 2020-06-06.

Endpoints:

- `calendar/day/{YYYY-MM-DD}`: tracks[] with race stubs (id, status) and
  games{} keyed by type (V75, V86, V64, V65, V5, V4, V3, dd, ld, vinnare,
  plats, trio, ...).
- `races/{raceId}`: id, date, number, distance, startMethod (auto/volte),
  startTime and scheduledStartTime, sport, prize (a text string), track
  (id, name, condition, countryCode), status, starts[].
- `games/{gameId}`: top-level pools and a races[] list. Each race in races[]
  is a full race object (same shape as the races endpoint) whose starts carry
  a per-start pools block.

Start shape (in both races[] and the race endpoint):

- start: id, number, postPosition, distance (per-start handicap), horse,
  driver, result, and inside a game also pools.
- horse: id, name, age, sex (full words: gelding, mare, stallion, ...),
  money (career earnings), record (best), shoes, sulky, trainer, owner,
  breeder, pedigree, statistics (per year and life).
- driver and trainer: id, firstName, lastName, shortName, license,
  statistics. There is no single name field.
- shoes: {reported, front{hasShoe, changed}, back{hasShoe, changed}}.
  sulky: {reported, type, changed} when reported, else {reported:false}.
- result of a finisher: place, finishOrder, kmTime{minutes,seconds,tenths},
  prizeMoney, finalOdds, startNumber.
- result of a non-finisher: galloped and/or disqualified set true, no place,
  kmTime is {code: ...}. It still has finalOdds.
- result of a scratched horse: the start has scratched:true and finalOdds 0.0.
- per-start pools inside a game: vinnare{odds}, plats{minOdds,maxOdds}, and
  for a marking-bet leg the bet type pool, for example V75{betDistribution}.

Id formats:

- Race: `YYYY-MM-DD_trackId_raceNo`
- Start: `YYYY-MM-DD_trackId_raceNo_startNo`
- Game: `TYPE_YYYY-MM-DD_trackId_legRaceNo`

Units:

- betDistribution is in hundredths of a percent (a leg sums to 10000 = 100%).
  It exists only in marking-bet pools (V75, V64, ...), at
  start.pools.<betType>.betDistribution. The win and place pools do not have
  it. The win-pool implied probability comes from finalOdds instead.
- Pool odds are in hundredths (1926 means 19.26), equal to finalOdds.
- finalOdds in race results is a plain decimal.
- Statistics percentages are scaled by 100 (winPercentage 3076 means 30.76%).

Quirks:

- A non-finisher has no place. Use the galloped and disqualified flags, not a
  place of zero. A gait break is common in trot, so this is frequent.
- sport is per race and is trot or monte. monte is mounted trotting (a rider,
  not a sulky), with different dynamics. Store it and model it separately.
- shoes and sulky carry changed flags. These are useful pre-race features.
- The horse and driver statistics blocks may be current rather than as-of-race.
  Until this is verified, do not use them as features. Reconstruct history from
  prior races instead. To verify later: check whether a horse's life.starts
  grows across races over time (as-of-race) or is constant (current).
- The API also covers Danish, Norwegian and French trot (countryCode).

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

## Phase 2 status

Done and validated against real payloads:

1. Normalisation into norm_races, norm_starts and norm_bet_distribution. The
   parser handles the real schema: firstName and lastName names, nested shoes,
   the galloped and disqualified flags, the scratched flag, sport (trot or
   monte), and the betDistribution under start.pools. It prints a validation
   report (parse errors, missing fields, races without a winner, sport split).
2. The market benchmark in benchmark.py. It builds win probabilities from
   de-vigged final odds (the primary, full-coverage signal) and from the
   V game spelprocent (partial coverage). It scores them with log loss and
   Brier and writes a calibration plot. A race is included only if it has
   exactly one winner and every runner that started has a usable value.
   Excluded races are counted, as is the overround for a takeout sanity check.

Remaining before Phase 3 results mean anything: a real backfill. The sample
has only 51 races, so the benchmark numbers are noise. One open question to
settle on the backfill is whether the statistics blocks are as-of-race.

## Phase 3 plan (not started)

1. Point-in-time features computed only from prior races: recent km-time form
   adjusted for distance and start method, days since last start, post
   position with track and start method, time-decayed driver and trainer win
   rates, class from prior earnings, equipment change flags, field size. The
   API statistics blocks are not used as features, because they may be
   current rather than as-of-race.
2. Baseline: conditional (multinomial) logistic regression over the runners
   in each race. Then LightGBM, first as a binary objective with per-race
   renormalisation, then with a custom grouped-softmax objective.
3. Time-based split by race date, never random. Report model log loss and
   Brier on the held-out test set next to the market benchmark on the same
   races, with a paired bootstrap on the per-race difference and a model
   calibration plot. A value-betting backtest is secondary to calibration.
