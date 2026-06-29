# Progress

## Status

Phase 1 (ingestion) is complete and hardened. Phase 2 (normalisation and the
market benchmark) is written and validated against real sample payloads, and
the full chain runs. Phase 3 (features, models, evaluation) has not started.
The next milestone is a real backfill, after which the benchmark numbers become
meaningful. The current sample of 51 races is far too small for the numbers to
mean anything.

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

## Data and workflow

The corporate network blocks atg.se, so ingestion runs on a machine with
access (a personal network or a second laptop). The result is a single file,
`data/atg.sqlite`, transferred back as a file. `data/` is gitignored and is
never pushed. Because storage is raw-first, parsing can be re-run at any time
without re-downloading.

## Next steps

1. Backfill, for example:
   `python -m atg.ingest --from 2023-01-01 --to 2026-06-28 --skip-games`
   Resumable. Bring `data/atg.sqlite` into `data/`.
2. `python -m atg.normalize --db data/atg.sqlite`
   `python -m atg.benchmark --db data/atg.sqlite`
   Review the market benchmark before any modelling.
3. Phase 3: point-in-time features from prior races only, a conditional logistic
   regression baseline, then LightGBM with a grouped-softmax objective,
   evaluated on a time-based split against the market benchmark with a paired
   bootstrap and a calibration plot.

## Open question

Whether the API statistics blocks are as-of-race or current. On the backfill,
check whether a horse's `life.starts` grows across races over time. Until this
is confirmed, do not use those blocks as features. Reconstruct history from
prior races instead.
