# Decisions and experiment log

A single record of what was tried, what was found, and what was deliberately not
tried and why. It exists so the project is defensible: the choices, including the
omissions, are reasoned rather than accidental. The detailed plan is in
ROADMAP.md, the literature in RESEARCH.md, and the chronological log in
PROGRESS.md.

## Objective

Produce win probabilities for Scandinavian trot with lower out-of-sample
multinomial log loss than the public pari-mutuel market. Profit is not a goal.

## How the experiment was kept honest

- Walk-forward over 16,224 held-out races (six expanding folds, 2025 to mid 2026),
  with purge and embargo. Never a random split.
- The model and the market are scored on exactly the same races. The market is
  never an input to the headline fundamental model, so the comparison is not
  circular.
- Significance requires two methods to agree: a bootstrap that resamples whole
  race days, and a Diebold-Mariano test with a day-clustered standard error.
- The minimum detectable effect was computed before looking at any model result,
  so a gain below the detection floor is never read as real.
- Fixed-split numbers were labelled provisional and never quoted as results. Only
  walk-forward numbers count. A fixed-split edge that looked significant was
  correctly rejected when it did not survive walk-forward.
- A point-in-time leakage test guards the feature builder, and the API statistics
  blocks were verified to be as-of-race before being used.

## What was tried, and what was found

All figures are walk-forward, the quotable numbers.

- Favourite-longshot recalibration of the market (a one-parameter power fit on
  past data). Beats the raw market by 0.40 percent skill. The Murphy
  decomposition shows it is a calibration fix: favourites are underbet. Robust.
- Conditional (multinomial) logit on within-race-centred point-in-time features.
  Standalone it loses to the market clearly (log loss about 1.88 against 1.64).
- LightGBM with a binary objective and per-race renormalisation. Also loses
  standalone (about 1.85). Expected for public, pace-less data against a sharp
  market.
- Market combination, Benter style, fitting the fundamental probability against
  the market probability on a held-out block. Beats the raw market by 0.64
  percent and the recalibrated market by 0.24 percent skill, both significant
  (z 4.2, p below 0.0001), consistent across folds. This is the headline.
- Feature rounds, each gated through walk-forward. The base reconstructed
  features alone left the combination no better than the recalibrated market
  (p 0.07). Adding the verified as-of-race API statistics plus sex and an age
  curve made it significant. A domain-motivated niche batch (barefoot, American
  sulky, driver change, draw bias) lifted it further and held across folds. A
  start-method specialism and a class-move feature added a final small,
  consistent gain.
- Speed-figure refinement (track added to the par, plus a daily track variant).
  Barely moved the models, because the API best-time and career-rate features
  already carry the ability signal. Kept anyway as a sounder, lower-overfit
  feature. An honest negative result.

Conclusion: the objective is met, modestly. A forecaster combining the public
odds with the fundamental model beats the market on out-of-sample log loss. The
robust pieces are the favourite-longshot recalibration and a small orthogonal
signal from classic Scandinavian factors. A standalone fundamental model does not
beat the market.

## Feature engineering coverage by tier

- Tier 1, built in full: distance, start-method and track-normalised speed
  figure; daily track variant; recency-weighted form; field size. Only the form
  trend (recent minus long-run) was left out; the level is used.
- Tier 2, mostly built: empirical-Bayes time-decayed driver and trainer rates;
  horse Elo; draw bias by track, start method and post; class from reconstructed
  prior earnings; equipment changes; layoff as days since last start. Simpler
  choices where the roadmap allowed a fancier one: plain Elo rather than
  Glicko-2, and no separate driver Elo since the decayed driver rates cover it;
  layoff as a continuous value rather than non-linear buckets.
- Tier 3, mostly not built, by design: pace and trip is impossible, the data
  does not exist in any structured source (confirmed); hierarchical Bayesian
  random effects and entity-embedding or sequence models were skipped as marginal
  over gradient boosting at this scale and leakage-prone, with the Elo and
  shrunk-rate features already capturing most of that value. Two higher-effort
  custom features were built: start-method specialism and class movement.
- Beyond the tiers: domain-specific niche features not in the original plan were
  added and gated through walk-forward: barefoot (shoes off), American sulky and
  driver change. These produced the orthogonal signal that made the combination
  beat the recalibrated market.

## What was deliberately not tried, and why

- Grouped-softmax LightGBM objective. The binary objective with renormalisation
  was already competitive, and the conditional logit covers the principled
  per-race-softmax form. The expected gain was marginal relative to the build and
  the overfitting risk on a small edge.
- Plackett-Luce or ListMLE finishing-order training. It would extract more signal
  per race, but trot has frequent gait breaks and disqualifications, so the
  finishing tail is noisy, and the marginal gain did not justify it once the
  combination edge was established.
- A fitted calibration layer (temperature scaling, isotonic, Dirichlet). The
  combination is already essentially perfectly calibrated, Murphy reliability
  about 0.00002, so a calibrator would add nothing. Calibration was measured, not
  assumed.
- Hierarchical Bayesian random effects, entity embeddings and sequence models.
  Likely marginal over gradient boosting at this data size and leakage-prone. The
  Elo and time-decayed shrunk rate features already capture most of that value at
  far less risk.
- Pace and trip features. The strongest signal in the literature, but absent from
  every structured Scandinavian source (confirmed by a recursive scan of the ATG
  payload and the Svensk Travsport schema). Only recoverable from race video by
  computer vision or a paid positioning feed, both out of scope. Recorded as a
  future extension in ROADMAP.md.
- Monté modelling. Mounted trot is a small subset (about 1,200 races) with
  different dynamics. The focus is sulky trot. Monté is left for a separate model.
- Live pre-race logging. Everything the test needs is reconstructable after a race
  finishes (result, as-of-race statistics, closing odds, prior-race features), and
  the benchmark is the closing odds, so a daily before-the-off logger adds
  operational overhead with no methodological gain. The recommended confirmation
  is a backfill of further data and a re-run.
- Multiple-comparison correction (Holm or Benjamini-Hochberg) across model
  variants. The number of headline comparisons is small and the surviving edge
  has p below 0.0001, well inside any correction, so it was noted rather than
  formally applied.

## What would strengthen it further, if revisited

- More data by backfilling 2020 to 2023, which roughly triples the test races and
  adds an earlier era the features were not tuned on, a cross-era robustness check.
- A segment breakdown of the edge by pool size, to see whether it concentrates in
  the softer, thinner weekday markets.
