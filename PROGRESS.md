# Progress

## Status

Phase 1 (ingestion) is complete and hardened. Phase 2 (normalisation and the
market benchmark) is complete and has now run on the first real backfill.
Phase 3 (features, models, evaluation) has not started; a research-informed
plan for it is in docs/ROADMAP.md.

The backfill covers 2024-01-01 to 2026-06-28 for Scandinavia, trot, run with
--skip-games: 28,687 races and 304,067 starts in data/atg.sqlite. The real
market benchmark, de-vigged final win odds, trot only over 27,306 clean races,
is log loss 1.6352 and Brier 0.7235, with a median overround of 1.180 and a
calibration curve close to the diagonal. This is the benchmark Phase 3 must
beat out of sample.

## Phase 1: ingestion

Raw-first storage of complete JSON responses, idempotent by day. Hardened
after the first version: the client treats non-JSON 2xx bodies and
non-retryable 4xx responses explicitly, a failed day no longer aborts the
batch, and games already stored are skipped on resume.

Ingestion options worth knowing:

- `--countries`, default `SE,DK,NO`. The project is Scandinavian, so French
  and Finnish trot are excluded by default. Use `--countries all` to widen.
- `--skip-games`. The V game pools are about 60 percent of the storage and
  carry only the secondary spelprocent. Skipping them cuts a multi-year
  backfill to roughly 2 to 3 GB with no effect on the primary benchmark or on
  any model feature, since both come from the race data.

## Phase 2: normalisation and benchmark

`normalize.py` parses the raw tables into `norm_races`, `norm_starts` and
`norm_bet_distribution`. It matches the real API schema, confirmed against
sample payloads: `firstName` and `lastName` names, the `galloped` and
`disqualified` flags (a non-finisher has no `place`), the `scratched` flag,
`sport` (trot or monte), and the `betDistribution` under `start.pools` for
marking-bet legs. It prints a validation report.

`benchmark.py` builds market win probabilities from the de-vigged final odds
(the primary, full-coverage signal, equal to the win-pool spelprocent) and from
the V game spelprocent (secondary, partial coverage). It scores them with log
loss and Brier and writes a calibration plot.

`metrics.py` holds log loss, Brier, a calibration table and a paired bootstrap.

Validation. `tests/test_phase2.py` runs the whole chain on fabricated payloads
and checks the arithmetic. `tests/build_sample_db.py` builds `data/sample.sqlite`
from the real JSON in `data/samples/`. On the 51 race sample the median
overround was 1.179, a takeout near 15 percent, which is the expected figure
for a Swedish trot win pool and confirms the de-vigging.

## Phase 3 progress

The detailed plan is in docs/ROADMAP.md and the literature behind it in
docs/RESEARCH.md. Progress so far:

- Step 1, the evaluation harness, is built and tested. atg/splits.py does
  date-based walk-forward splits with purge and embargo. atg/evaluate.py scores a
  model against the market on the same races, with a log-loss skill score, a
  day-blocked paired bootstrap, a Diebold-Mariano test, the Murphy decomposition
  and the minimum detectable effect. tests/test_evaluate.py passes.
- An empirical probe found a favourite-longshot bias in the de-vigged odds, so
  the honest benchmark to beat is the recalibrated market, not the raw market.
- Step 2, the point-in-time feature build, has a first version. atg/features.py
  and atg/ratings.py build norm_features, one row per runner from prior-race state
  only: normalised speed-figure form, a horse Elo, time-decayed shrunk driver and
  trainer rates, reconstructed class and earnings, layoff, post position, field
  size and equipment flags. On the backfill it wrote 288,299 rows, 94 percent of
  them carrying a form figure and an Elo history. tests/test_features.py checks
  the point-in-time discipline, including that a race's own result never enters
  its feature row.

- Step 3, models, first cut (atg/model.py, tests/test_model.py). Within-race
  centred features feed a conditional logit and a LightGBM model, scored through
  the harness on a fixed split. All numbers here are PRE-WALK-FORWARD and
  provisional. On the 2026 first-half test block the favourite-longshot
  recalibration of the market beat the raw market by about 0.55 percent skill with
  a tight interval, which confirms it as the honest benchmark. The fundamental
  models trailed the market clearly, near 1.92 to 1.96 log loss against the
  market's 1.60. Diagnostics show this is genuine feature limitation, not a bug:
  the models beat a uniform guess and have sensible coefficients, but capture only
  about half the market's signal with the current coarse features and no pace.

Still to do in Phase 3: stronger features (track variant and class-adjusted speed
figures, sex, better ratings), the market-combination ceiling to test whether the
features hold any signal orthogonal to the crowd, a grouped-softmax objective and
calibration (Step 4), and the walk-forward evaluation that turns these provisional
numbers into results (Step 5).

## Data and workflow

The corporate network blocks atg.se, so ingestion runs on a machine with
access (a personal network or a second laptop). The result is a single file,
`data/atg.sqlite`, transferred back as a file. `data/` is gitignored and is
never pushed. Because storage is raw-first, parsing can be re-run at any time
without re-downloading.

## Next steps

1. Done: backfill 2024-01-01 to 2026-06-28 (Scandinavia, trot, --skip-games),
   normalisation, and the real market benchmark.
2. Phase 3, planned in detail in docs/ROADMAP.md: point-in-time features from
   prior races, the API as-of-race blocks as extra features, a conditional
   logistic regression baseline, then LightGBM with a grouped softmax objective,
   calibration on a forward time block, and a walk-forward split with purge and
   embargo, evaluated against the market benchmark with a paired bootstrap and
   calibration plots.

## Resolved question

Whether the API statistics blocks are as-of-race or current. Resolved on the
2024 to 2026 backfill: they are as-of-race and exclude the current race, so they
are point-in-time safe. A horse's `life.starts` grows monotonically across its
races (11,200 horses up, 6 violations), and the career win count rises in step
with the current-race win, not the next race (10,415 against 10). They may be
used as features, though they are coarse. Reconstructed prior-race features stay
the backbone.
