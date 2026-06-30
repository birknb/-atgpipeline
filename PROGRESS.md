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
- The market-combination ceiling (fundamental plus market, stage-two weights fit
  on the validation block) put almost all weight on the market, 1.10, and only
  0.06 on the fundamental model, and did not significantly beat the recalibrated
  market (skill plus 0.02 percent, not significant, p 0.51). So with the current
  features the fundamental model holds essentially no signal orthogonal to the
  crowd. The favourite-longshot recalibration remains the one genuine edge. This
  is a pre-walk-forward finding and it motivates the feature work.
- First feature enrichment: the as-of-race API statistics (career win and place
  rates, earnings per start, life starts, best record time), verified
  point-in-time safe, plus sex and an age curve. This lifted the fundamental
  models (conditional logit 1.96 to 1.85, LightGBM 1.92 to 1.81 log loss,
  pre-walk-forward), and the market combination now beats the recalibrated market
  by a tiny but significant margin, about 0.05 percent skill, p 0.01, right at the
  detection floor. So the enriched features hold a sliver of signal orthogonal to
  the crowd. It is fragile and pre-walk-forward, to be confirmed later. The
  standalone fundamental model still trails the market clearly.

- The speed figure was improved (track added to the par, plus a point-in-time
  daily track variant). It barely moved the models, because the API best-time and
  career-rate features already carry the ability signal. Kept anyway as a sounder,
  low-overfit feature.
- Walk-forward evaluation (atg/model.py --walk), six expanding-window folds over
  2025-01-01 to 2026-06-28, 16,224 test races. These are quotable, not
  pre-walk-forward. The favourite-longshot recalibration beats the raw market by
  0.40 percent skill (log loss 1.6286 against 1.6351), highly significant and
  above the detection floor, so a robust edge. The standalone fundamental models
  trail the market clearly (conditional logit 1.88, LightGBM 1.85 against 1.64).
  The market combination does not significantly beat the recalibrated market
  (skill 0.07 percent, p 0.07, interval includes zero): the tiny edge seen on the
  single fixed-split block did not survive walk-forward. So the one robust edge is
  the favourite-longshot recalibration. The fundamental features add no edge
  beyond the crowd that survives rigorous evaluation, which is exactly why
  fixed-split numbers are kept provisional.

- A disciplined niche-feature batch was added: barefoot (shoes off, front and
  back), American sulky, driver change, and a point-in-time draw-bias win rate by
  track, start method and post. These are classic Scandinavian-trot factors,
  domain-motivated rather than searched on the test set. On the walk-forward they
  lifted the combination from not beating the recalibrated market (p 0.07) to
  beating it by 0.20 percent skill, p 0.0002, with the fundamental weight stable
  and positive across all six folds (0.09 to 0.23) and the per-fold combo-versus-
  recalibrated difference positive in five of six folds, including the final one.
  So the features now carry a small but real signal orthogonal to the crowd that
  survives rigorous evaluation. The standalone fundamental model still loses to
  the market; the edge is realised only in combination with the market odds,
  which is a legitimate forecaster's use of public information.
- A final gated batch added a start-method specialism feature (the horse's form
  under today's start method) and a class-move feature (today's field class minus
  the horse's recent field class). Consistent small lift: the combination versus
  the recalibrated market rose to 0.24 percent skill (p below 0.0001, z 4.2),
  positive in five of six folds including the last, fundamental weights stable.

Headline result (walk-forward, 16,224 held-out races): the combination of the
fundamental model with the market beats the raw market by 0.64 percent skill and
the favourite-longshot-recalibrated market by 0.24 percent, both significant. The
recalibration alone beats the raw market by 0.40 percent. This meets the project
objective, lower out-of-sample log loss than the market. The effect is small and
this test period has been examined repeatedly, so a fully fresh future period
(Phase 4 live logging) would be the cleanest confirmation.

Still to do in Phase 3: probability calibration on a forward block (Step 4), then
the write-up. Any further feature ideas stay gated through the walk-forward
harness, and the small edge argues for restraint to avoid overfitting.

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
