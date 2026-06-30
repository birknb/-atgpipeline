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
V game spelprocent is reported as a second signal where it exists. The first
real backfill is now complete and the real benchmark is computed (see the
backfill section below). Feature engineering and models (Phase 3) have not
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
- The code is synced through a personal GitHub repo, birknb/-atgpipeline,
  branch main. Only code is pushed. The data (data/, the sqlite, zips) is
  gitignored and must never be pushed. Do not push to any other remote, in
  particular anything under an NBIM or corporate organisation.

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

## Backfill and real market benchmark (June 2026)

The first real backfill is complete. It covers 2024-01-01 to 2026-06-28 for
Scandinavia (SE, NO, DK), filtered to trot tracks, run with --skip-games. The
result is 28,687 races and 304,067 starts in data/atg.sqlite, transferred to
the analysis laptop as a split zip over email and reassembled there.
Normalisation ran with no parse errors. The country split is SE 18,577, NO
5,791, DK 4,319. Of the starts, 288,299 ran, 15,768 were scratched, and 53,574
galloped or were disqualified, the expected high gait-break rate in trot.
288,129 runners that ran have positive final odds.

The track filter keys on the calendar, but sport is per race, so the backfill
picked up some non-trot races on trot tracks: 27,384 trot, 1,171 monté, 132
gallop. Model trot and monté separately. The gallop handful can be dropped.

The real market benchmark is the de-vigged final win odds. Trot only, 27,306
clean races (exactly one winner, every runner with positive odds):

- log loss 1.6352
- Brier 0.7235
- median overround 1.180, a takeout near 15 to 18 percent, matching the sample
  and confirming the de-vigging.

All sports together, 28,601 races: log loss 1.6372. The calibration curve
(results/trot/calibration_market.png) sits almost exactly on the diagonal
across every decile from about 1 to 40 percent, so the win market is close to
unbiased. This is the number every model must beat out of sample. For scale, a
uniform guess over a 10 runner field scores ln(10) = 2.30.

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
- The horse statistics blocks are as-of-race and exclude the current race.
  This was verified on the 2024 to 2026 backfill. A horse's life.starts grows
  monotonically across its races over time (11,200 horses with at least five
  starts increase, only 6 violations), and on adjacent close races the career
  win count rises in step with the win in the current race, not the next one
  (10,415 cases against 10). The blocks therefore reflect the horse's record
  going into the race and are point-in-time safe. They may be used as features,
  although they are coarse (yearly and life aggregates). Reconstructed
  prior-race features stay the backbone, with the API blocks as extra features
  and a cross-check. Driver and trainer blocks are presumed the same but were
  not separately verified.
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
- Evaluation: time-based train/test splits only, never random splits. Numbers
  from the fast fixed-split phase are provisional. Never quote them anywhere,
  even internally, without the label pre-walk-forward. Only walk-forward numbers,
  with purge and embargo, are quotable as results.
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

The backfill is now complete and the real benchmark is computed (see the
backfill section above). The open question about the statistics blocks is
resolved: they are as-of-race and exclude the current race. Phase 3 is the
next step, with a detailed, research-informed plan in docs/ROADMAP.md.

## Phase 3 plan (not started)

1. Point-in-time features computed only from prior races: recent km-time form
   adjusted for distance and start method, days since last start, post
   position with track and start method, time-decayed driver and trainer win
   rates, class from prior earnings, equipment change flags, field size. The
   API statistics blocks may also be used as extra features and a cross-check,
   since they are verified as-of-race. See docs/ROADMAP.md for the detailed,
   research-informed plan that supersedes this summary.
2. Baseline: conditional (multinomial) logistic regression over the runners
   in each race. Then LightGBM, first as a binary objective with per-race
   renormalisation, then with a custom grouped-softmax objective.
3. Time-based split by race date, never random. Report model log loss and
   Brier on the held-out test set next to the market benchmark on the same
   races, with a paired bootstrap on the per-race difference and a model
   calibration plot. A value-betting backtest is secondary to calibration.
