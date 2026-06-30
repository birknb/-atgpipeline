# Phase 3 roadmap: models and evaluation

This document is the detailed plan for Phase 3. It supersedes the short Phase 3
summary in CLAUDE.md. It was written after a literature review of racing
prediction models, machine learning for grouped choice problems, probability
calibration, time-aware validation, and harness feature engineering. Sources
are listed at the end.

## Objective and the benchmark to beat

The goal is win probabilities with lower out-of-sample multinomial log loss than
the public win market, with good calibration. Profit is not a goal.

The benchmark is the de-vigged final win odds. On the 2024 to 2026 backfill,
trot only, 27,306 clean races, it scores log loss 1.6352 and Brier 0.7235, and
its calibration curve is almost exactly on the diagonal. A uniform guess over a
ten runner field scores 2.30. The market is therefore sharp and close to
unbiased.

The central scientific fact that shapes the whole plan: a near-efficient market
already prices most public information, so a model's edge lives only in the
residual, the small part the crowd misprices. Realistic edges over a sharp pool
are on the order of a few thousandths of a nat, which is a log-loss skill score
around half a percent to one percent. Any result much larger than that is almost
certainly leakage, not skill. The plan is built to find a small real edge and to
not fool ourselves into reporting a large fake one.

## Guiding principles

- Point-in-time first. Every feature must be computable from data timestamped
  strictly before the race start. The post-race fields (place, km_time,
  finalOdds, betDistribution) feed only prior-race aggregates, rating updates,
  and the evaluation benchmark, never the current race's feature row.
- Features are within-race relative. A race is a choice among its runners, so
  each feature should be centred or standardised within the race. Otherwise the
  model learns field strength rather than relative ability, which is the most
  common implementation error in this problem.
- The market is the benchmark, not an input. The de-vigged odds are what we
  score against, so a model that ingests them cannot be compared to them without
  circularity. Market-informed models are built and reported separately, clearly
  labelled, to measure how much orthogonal signal our features carry.
- Identical race sets. The model and the market are always scored on exactly the
  same races, decided by one eligibility mask, so the comparison is fair.
- Time-based evaluation only. Splits are by race date with purge and embargo.
  Random splits are invalid because features pool a horse's, driver's and
  trainer's history across races.
- Build the ruler before the thing it measures. The evaluation harness is built
  and tested before any model, so no model is ever judged by an instrument we
  have not checked.

## Empirical findings before modelling (pre-walk-forward, indicative)

A probe of the de-vigged final odds on the full data, for orientation only and to
be re-checked out of sample, found the following.

- A favourite-longshot bias is present even in final odds. The shortest-price
  group is underbet, predicted 0.48 and won 0.52, and the longest-price groups
  are overbet, predicted 0.006 and won 0.002. The largest bin deviation is about
  four percentage points.
- Correcting it with a single power-transform parameter, near 1.15, which shifts
  probability mass toward favourites, lowers the market log loss from 1.6352 to
  1.6281, a gain of about 0.4 percent skill. An in-sample isotonic recalibration,
  an optimistic ceiling, gains about 0.7 percent. The one-parameter figure should
  largely hold out of sample because it barely overfits.
- Raw market log loss rises with field size, from 1.32 at seven or fewer runners
  to 1.97 at fourteen or more, and differs by country, Norway 1.56 against Sweden
  1.66. These differences are mostly entropy, not market softness, so they do not
  by themselves point to exploitable races.

Two consequences for the plan. The favourite-longshot recalibration is a real,
capturable first gain and the right Step 3 baseline. And the honest benchmark to
beat is the recalibrated market, not the raw market, because part of any win over
the raw market is only the recalibration that any model gets for free.

## Order of work

### Step 1. Evaluation harness and data spine

Build and unit-test the evaluation machinery first. New module atg/evaluate.py
extends metrics.py.

1. Eligibility mask. One function that decides which races count: exactly one
   winner, every runner that started has a usable value, sport filter (trot
   first, monté modelled separately later, gallop dropped). Both the model and
   the market are restricted to this mask. Assert the two race sets are
   identical before scoring.
2. Walk-forward splits. New module atg/splits.py. Expanding-window and
   rolling-window splits keyed on race date, not row index, so a single day is
   never split across train and test. Start with one fixed train, validation,
   test split for fast iteration, then move to walk-forward for the final
   numbers.
3. Purge and embargo. Drop training races whose feature window overlaps the test
   period, and embargo a buffer after each test block before training resumes.
   Size the purge and embargo to the longest feature memory, for example the
   form half-life and the rating look-back. This prevents entity-overlap
   leakage, where a test race shares horses or drivers with adjacent training
   races.
4. Skill score. Report 1 minus L_model over L_market on the common races, so the
   headline number is the edge over the market, not the raw log loss, which is
   dominated by field size.
5. Significance. Keep the existing paired bootstrap but resample race days as
   blocks, not individual races, to respect within-day dependence. Add a
   Diebold-Mariano test on the per-race log-loss differential with the
   Harvey-Leybourne-Newbold small-sample correction. Claim an edge only when
   both agree, and control for multiple comparisons across model variants with
   Holm or Benjamini-Hochberg.
6. Calibration and decomposition. Reliability diagrams fitted only on past data.
   The Murphy decomposition of the Brier score into reliability, resolution and
   uncertainty, and the equivalent for the log score, so we can see whether a
   model wins through better calibration or better discrimination.
7. Minimum detectable effect. From the observed standard deviation of the
   per-race log-loss differential and the test-set size, compute the smallest
   improvement we could detect before looking at any model result. Report it, so
   a point gain smaller than the detectable effect is never read as real. A
   conservative power preview on the real data, using the market's own per-race
   log-loss variation as an upper bound on the differential, puts the detectable
   effect at about 0.96 percent skill over all 27,306 trot races and about 2.25
   percent over a single six-month window. The true floor is lower, since a real
   model's differential varies less than the market loss. The implication is that
   a single six-month holdout is underpowered for a realistic edge near one
   percent, so walk-forward folds tiling the whole period, which test on close to
   every race, are needed for power as well as for rigor.

Deliverable: a tested harness that, given two probability columns on the same
races, returns log loss, Brier, skill score, a significance verdict, calibration
curves, the score decomposition, and the minimum detectable effect.

### Step 2. Point-in-time feature engineering

New module atg/features.py reads the norm_ tables in race-date order and writes a
feature table, one row per runner, with strict causal construction. A separate
module atg/ratings.py holds the running ratings. An early task is to check the
raw payload for sectional or running-position data, since the pace features in
tier three depend on it.

Tier one, build first:

- Speed figure. Convert each prior race's km-time into a comparable rating by
  subtracting a par expected km-time conditional on distance bucket, start
  method (auto or volte), track and condition, then standardising. Raw km-time
  is not comparable across configurations. The horse's feature is an aggregate
  of its prior speed figures only, never the current race.
- Daily track variant. Estimate each race-day-track speed bias as the average
  deviation of completed runners from par, and adjust prior speed figures by it.
  Compute the variant only from races finished before the target race start, or
  it leaks.
- Recency-weighted form. Exponentially weight prior speed figures and finishes
  by recency, with a half-life around 60 to 120 days, and include both a level
  and a trend (recent minus long-run).
- Field size and a clean race-level context block.

Tier two, build next:

- Driver and trainer rates. Time-decayed win and top-three rates per driver and
  trainer, shrunk toward the population mean by sample size with empirical Bayes,
  so low-sample entities do not look elite. Minor-driver skill is a plausible
  underpriced signal.
- Elo or Glicko-2 ratings for horses and, separately, drivers, updated after
  each race in date order from the finishing order. Glicko-2 handles layoffs
  through a rating deviation that grows with idle time. The rating entering a
  race is the feature, the post-race update is not.
- Post position by start method and track. Encode the draw as a historical
  win-rate-by-(post, track, start method) table estimated on the training period,
  not as a raw number. Inside draws matter most on short ovals and standing
  starts.
- Class from prior earnings. Reconstruct cumulative earnings and average
  prize-money per start from prior race results rather than the career money
  field, to stay point-in-time.
- Layoff. Days since last start with non-linear buckets, a first-start-after-
  long-layoff flag, and a second-start-back flag. Career debutants get a distinct
  flag.
- Equipment changes. Shoe front and back changed and sulky changed flags. These
  are genuine pre-race signals and may be underpriced.

Tier three, higher effort and higher potential edge:

- Pace and trip. The literature points to pace setup as the most promising
  unpriced signal, but the ingested payload does not carry it. A recursive scan
  of all 271 key paths across 800 races found only the final per-km time, the
  finishing place and the finish order, plus the starting post position. There
  are no sectional or quarter times, no running positions, and no running
  comments, so running-style classification and a race-level pace-pressure
  feature are not feasible from this data. The only weak proxies are start
  method, post position and distance. Two ways to recover pace remain, both
  outside the current scope: a richer or different ATG endpoint may expose
  running comments or positions, which needs a network-connected check on a
  machine that can reach atg.se; and each start carries a video field, so pace
  could be extracted by computer vision. Until then this tempers the expected
  edge, since the clearest place a market can be beaten is not observable here.
- Bayesian hierarchical random effects for horse, driver and track on the speed
  figure, giving shrunk latent abilities as features. Fit on the training period
  only and forward-filter.
- Entity embeddings and sequence models over a horse's prior-race history. Likely
  only marginal over a well-tuned gradient boosting model at this data size and
  leakage-prone, so treated as a late experiment. Time-decayed target encoding of
  the ids captures most of the value at far less risk, and the rating features
  above already do much of this.

The verified as-of-race API statistics blocks may be added as extra features and
as a cross-check on the reconstructed ones, with the reconstructed features as
the backbone.

### Step 3. Models

Build in increasing order of complexity, scoring each against the market with the
Step 1 harness before moving on.

1. Favourite-longshot sanity baseline. A monotone recalibration of the de-vigged
   odds, for example a power or Shin transform that mildly shrinks toward
   favourites, fit on the training period. If this beats the raw market, it
   measures the residual favourite-longshot bias in final odds and sets a floor
   that any real model must clear. Estimate it separately for trot and monté.
2. Conditional logit baseline. A multinomial logit over the runners in each race,
   probability proportional to exp of a linear index in within-race-centred
   features, fit by maximum likelihood with the winner as the chosen runner. This
   is the Bolton and Chapman model and the correct baseline. It handles variable
   field size and outputs probabilities that sum to one within a race.
3. LightGBM, two forms. First the pragmatic form: a binary is-winner objective
   per runner with within-race renormalisation, which is quick in stock LightGBM.
   Then the principled form: a custom grouped-softmax objective, where the trees
   fit a per-runner index and the softmax is taken within each race, with
   gradient p_i minus y_i and the standard diagonal Hessian bound. The
   grouped-softmax form directly minimises the evaluation metric and needs no
   post-hoc renormalisation.
4. Finishing-order training signal. Train on the full finishing order with a
   Plackett-Luce or ListMLE loss instead of the winner alone, which extracts more
   signal per race. Treat non-finishers, the frequent gait breaks and
   disqualifications, as censored below the finishers, and truncate the order to
   the top few places where the signal is reliable. The evaluation metric stays
   win log loss; this is a training choice, not a metric change.

The built-in LightGBM lambdarank and rank_xendcg objectives are ranking-only and
not calibrated, so they are used only as ranking diagnostics or as a feature,
never as a probability source.

Later, optional, only if the above plateaus:

- Boosted conditional logit, which keeps the correct per-race softmax structure
  while adding tree-like nonlinearity to the index.
- A set-based neural model, Deep Sets or a Set Transformer, where each runner
  conditions on its actual opponents. This is a speculative experiment given the
  modest data size, where gradient boosting usually wins on tabular data.

### Step 4. Calibration

Trees are not calibrated out of the box, and even the conditional logit may be
mildly over or under confident. Add a calibration layer fit only on a forward
time block, never on the training races and never on the test races.

- Start with temperature scaling, a single scalar that divides the within-race
  indices before the softmax. It preserves the sum-to-one structure and the
  ranking and only sharpens or softens.
- If calibration error is more than a global scale, consider isotonic or beta
  calibration of the per-runner scores followed by within-race renormalisation,
  or Dirichlet calibration. Isotonic overfits on small samples, so prefer the
  smoother options unless there are enough calibration races.

### Step 5. Evaluation and reporting

For each model, on the held-out test races and on the identical market race set,
report log loss, Brier, the skill score against the market, the significance
verdict from the block bootstrap and Diebold-Mariano, a calibration curve next to
the market, and the score decomposition. Report the skill score against both the
raw market and the favourite-longshot-recalibrated market, since beating only the
raw market may be recovering the recalibration that any model gets for free.
Break the numbers down by segment: trot against monté, by country, and by field
size, since the edge may live in a subset. A value-betting backtest is secondary to calibration and is not the goal.

### Step 6. Research payoff, market combination and bias

This is where the interesting scientific question is answered, kept strictly
separate from the headline benchmark comparison.

- Market combination. Build a Benter-style second-stage model that combines the
  fundamental model probability with the public implied probability. Benter found
  the combination beat both the market alone and the fundamentals alone, because
  the model adds information orthogonal to the crowd. Because the public
  probability here is the benchmark, this model is reported only as a separately
  labelled ceiling. Its value is the estimated weights and the gain over the
  market, which quantify how much orthogonal signal our features hold. It is
  never the model we headline against the benchmark.
- Favourite-longshot analysis. Report the estimated bias in the final odds for
  trot and monté, which connects the project to the racing-market literature.
- A genuinely fair market input, if one exists, would be an earlier pre-close
  odds snapshot that is not the final benchmark odds. Phase 4 below could log
  this.

## Anti-self-deception checklist

Run through this before believing any result.

- Did the model and the market score on exactly the same races, with N reported.
- Was every preprocessing step, scaler, calibrator, and hyperparameter choice fit
  inside the training fold only.
- Were features audited so that no input is timestamped at or after race start,
  including the track variant, the ratings, and any reconstructed earnings.
- Is the reported improvement larger than the pre-computed minimum detectable
  effect.
- Do the block bootstrap and the Diebold-Mariano test agree.
- If the edge is large, assume leakage and find it before celebrating.

## Decisions for review

These shape the work and are for the project owner to confirm.

1. Scope. Start with trot only and model monté separately later. Drop gallop.
2. API as-of-race blocks. Use them as extra features and a cross-check, with
   reconstructed prior-race features as the backbone, and reconstruct cumulative
   earnings rather than using the snapshot money field.
3. Market use. The de-vigged odds are the benchmark and are never a feature in
   the headline models. Benter-style combination is reported only as a separate,
   clearly labelled ceiling.
4. Split design. A first pass on a fixed train, validation, test split for speed,
   then walk-forward with purge and embargo for the final numbers. With six
   months of test data this is a few thousand races, enough to detect a small
   edge if one exists. Every number from the fixed-split phase is provisional and
   is never quoted anywhere, even internally, without the label pre-walk-forward.
   Fixed-split numbers have a way of escaping into conclusions. Only walk-forward
   numbers, with purge and embargo, are quotable as results.
5. Pace features, resolved. The payload has no sectional times or running
   positions, only the final per-km time and the finishing order. Pace and trip
   handicapping is not feasible from this data, which removes the most promising
   unpriced signal. Expectations are set accordingly.

## References

Racing models and market efficiency:
- Bolton and Chapman 1986, multinomial logit for racing: https://gwern.net/doc/statistics/decision/1986-bolton.pdf
- Benter's Hong Kong model: https://datagolf.com/static/blogs/benter_paper.pdf , annotated: https://actamachina.com/posts/annotated-benter-paper
- Harville 1973 and order-statistics models: https://www.stat.berkeley.edu/~aldous/157/Papers/ali.pdf
- Favourite-longshot bias, German harness racing: https://www.researchgate.net/publication/23544905_Risk_Love_and_the_Favorite-Longshot_Bias_Evidence_from_German_Harness_Horse_Racing
- Boosted conditional logit (Tutz and Groll): https://www.sciencedirect.com/science/article/abs/pii/S1755534516301002

Machine learning for grouped choice and calibration:
- XGBoost custom softmax recipe: https://xgboost.readthedocs.io/en/stable/python/examples/custom_softmax.html
- rank_xendcg cross-entropy LTR (Bruch 2019): https://arxiv.org/pdf/1911.09798
- Temperature scaling and calibrated ranking: https://arxiv.org/pdf/2406.08010
- Dirichlet calibration (Kull et al. 2019): https://www.semanticscholar.org/paper/d4691aef27ae3c768b90c34ca5d8521d202eb47c
- scikit-learn calibration: https://scikit-learn.org/stable/modules/calibration.html

Time-aware evaluation:
- Purged and embargoed cross-validation (Lopez de Prado): https://en.wikipedia.org/wiki/Purged_cross-validation
- Diebold-Mariano test: https://www.sas.upenn.edu/~fdiebold/papers/paper68/pa.dm.pdf
- Murphy decomposition of proper scores: https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.2985
- Forecast verification, skill scores: https://www.cawcr.gov.au/projects/verification/verif_web_page.html

Feature engineering and ratings:
- Beyer speed figures: https://en.wikipedia.org/wiki/Beyer_Speed_Figure
- Empirical Bayes shrinkage: https://kiwidamien.github.io/shrinkage-and-empirical-bayes-to-improve-inference.html
- Rating systems, Elo, Glicko-2, TrueSkill: https://github.com/atomflunder/skillratings
- Hierarchical Bayesian racehorse ability: https://sciendo.com/article/10.2478/ijcss-2023-0007
- Machine learning the harness track (Schumaker): https://www.robschumaker.com/publications/DSS%20-%20Machine%20Learning%20the%20Harness%20Track%20-%20Crowdsourcing%20and%20Varying%20Race%20History.pdf
