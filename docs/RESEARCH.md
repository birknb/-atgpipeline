# Research notes for Phase 3

These are the detailed findings from four literature reviews carried out in June
2026 to inform the Phase 3 plan in docs/ROADMAP.md. They are a synthesis of the
open literature, not validated results. Anything here that suggests an edge must
still be confirmed out of sample under the evaluation protocol. The four streams
are racing-specific models and market blending, machine learning for grouped
choice and calibration, time-aware evaluation, and feature engineering and
ratings. Sources are listed inline.

## 1. Win-probability methods for harness racing, and market blending

The central tension: the market is the benchmark, so the strongest known method,
Benter's market combination, uses the odds as an input, which would make a direct
log-loss comparison circular. The non-circular methods come first.

1. Conditional (multinomial) logit over the runners in each race. One race is one
   choice set, p_i proportional to exp(V_i) over the runners, V_i linear in
   point-in-time features, fit by maximum likelihood with the winner as the
   chosen alternative. Handles varying field size and outputs probabilities that
   sum to one. This is Bolton and Chapman (1986) and the fundamental layer of
   Benter (1994). Pitfall: features must be within-race relative, centred or
   z-scored within the race, or the model learns field strength rather than
   relative ability. This is the most common implementation error. Scratched
   horses must be dropped from the choice set.
   Source: https://gwern.net/doc/statistics/decision/1986-bolton.pdf
2. Plackett-Luce, the rank-ordered or exploded logit. Same strengths, but the
   likelihood uses the full finishing order, so each race gives about N-1
   observations instead of one, which sharpens estimation. Pitfall: truncate the
   order to the top few places, and treat the frequent gallops and
   disqualifications as censored below the finishers rather than dropping them.
   Source: https://en.wikipedia.org/wiki/Discrete_choice
3. Harville, Henery and Stern order-statistics models. These convert win
   probabilities into full finishing-order probabilities. Out of scope for a
   win-only model, useful only if the project later extends to place or exotic
   pools. Harville is known to overstate the favourite's place probability.
   Source: https://www.stat.berkeley.edu/~aldous/157/Papers/ali.pdf
4. LightGBM with a per-race-renormalised binary objective, or a grouped softmax.
   Captures nonlinearities a linear logit misses, for example distance by start
   method or post by field size. Pitfall: trees are not calibrated out of the
   box, so a calibration step is needed before scoring log loss. The lambdarank
   objective optimises ranking, not probability, so prefer binary plus
   renormalisation or a custom grouped softmax for log loss.
5. Isotonic, beta or Platt recalibration as a final monotone layer, fit on a
   time-separated slice and renormalised per race afterwards. Cheap and reliably
   lowers log loss for tree models.
6. Benter two-stage market combination. Build the fundamental model, then fit a
   second conditional logit on the log fundamental probability and the log public
   probability. Benter's combined fit beat both the market alone and the
   fundamentals alone, because the model adds information orthogonal to the crowd.
   For this project the public probability is the benchmark, so this model is
   reported only as a clearly labelled ceiling that quantifies how much orthogonal
   signal the features hold. It is never the model headlined against the market.
   Source: https://datagolf.com/static/blogs/benter_paper.pdf
7. Favourite-longshot bias. Longshots are over-bet and favourites under-bet, a
   robust pattern for over fifty years, and present in German harness racing
   specifically. A monotone recalibration of the de-vigged odds that mildly
   shrinks toward favourites is a strong sanity-check baseline. Recent evidence is
   mixed, since late money can remove much of the bias by the close, so final odds
   may be near-efficient. Estimate it separately for trot and monté.
   Source: https://www.researchgate.net/publication/23544905

Less common but promising: boosted conditional logit (Tutz and Groll), which
keeps the per-race softmax structure while adding tree-like nonlinearity
(https://www.sciencedirect.com/science/article/abs/pii/S1755534516301002); and a
time-decayed rank-ordered logit for time-varying ability.

## 2. Machine learning for grouped win-probability and calibration

The key point: the target, a distribution over a variable-size set where exactly
one wins, is the Plackett-Luce or conditional-logit winner model,
P(i) = exp(s_i) / sum_j exp(s_j). The listwise learning-to-rank top-one loss
(ListNet) is identical to this softmax cross-entropy. So the planned approach and
the literature converge.

1. Grouped softmax, the conditional logit, as the primary objective. Softmax over
   the runners in a race, cross-entropy against the one-hot winner, gradient
   p_i - y_i and a diagonal Hessian upper bound. Guarantees sum-to-one within a
   race and directly minimises multinomial log loss.
   Source: https://xgboost.readthedocs.io/en/stable/python/examples/custom_softmax.html
2. Conditional or multinomial logistic regression baseline. The right inductive
   bias, fast, interpretable, the published method that has beaten the market.
   Fit as a conditional logit grouped by race, not plain multiclass.
3. LightGBM with the grouped-softmax custom objective. Trees fit the per-runner
   index, softmax-normalised within each race through a custom objective. The
   GBDT workhorse for this tabular size.
4. lambdarank and rank_xendcg are ranking objectives, not probabilities. Outputs
   are ordinal and look poor under log loss until calibrated. Use only as ranking
   diagnostics or as a feature. rank_xendcg is Bruch's cross-entropy loss
   (https://arxiv.org/pdf/1911.09798).
5. Plackett-Luce or ListMLE full-ranking loss. Uses the finishing order for more
   signal per race than the winner alone. A training-signal choice, not a metric
   change. Truncate the noisy tail.
6. Set-based neural models, Deep Sets or a Set Transformer, where each runner
   conditions on its actual opponents. A later experiment only, since around
   300k rows is small and gradient boosting usually wins on tabular data.
   Source: http://proceedings.mlr.press/v97/lee19d/lee19d.pdf
7. Temperature scaling, the first calibration step. One scalar divides the
   within-race logits, preserving the sum-to-one structure and the ranking. Fit
   on a temporally-later validation slice.
8. Dirichlet calibration, a more expressive multiclass post-hoc method that still
   outputs a normalised distribution. Adapting it to variable race sizes is
   awkward, so per-runner binary calibration plus renormalisation, or temperature
   scaling, is safer (https://www.semanticscholar.org/paper/d4691aef27ae3c768b90c34ca5d8521d202eb47c).
9. Per-runner binary objective plus within-race renormalisation. The pragmatic
   LightGBM path, trivial in stock LightGBM, a fast strong baseline. Theoretically
   inferior to the trained grouped softmax because the model never knows
   probabilities compete within a race.
10. Scoring rules. Optimise and report log loss, the project metric. Report Brier
    as a secondary robustness check. Clip probabilities away from zero. Use the
    paired bootstrap on per-race differences.

Recommended order: conditional logit baseline, then grouped softmax in LightGBM,
then temperature scaling on a forward-time slice, reporting log loss and Brier
with a paired bootstrap against the market. Do not over-engineer. The recent or
less-common items to keep in mind are rank_xendcg, Dirichlet calibration, the Set
Transformer, and the self-boosted calibrated ranking framework
(https://arxiv.org/abs/2406.08010).

## 3. Evaluation and validation protocol against a sharp market

The benchmark is sharp and well-calibrated, so the only real risk is fooling
ourselves. Items are ordered by how badly getting them wrong would mislead.

1. Walk-forward or expanding-window splits only, keyed on race date, never random
   k-fold. Split on the date value so a single day is never half in train and
   half in test. Source: https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/
2. Purge and embargo around each boundary (Lopez de Prado). Features built from a
   horse's, driver's or trainer's prior races create overlapping-history overlap
   between nearby train and test races. Purge the tail of training before the
   test window and embargo a band after it, sized to the longest feature memory,
   for example the form half-life. Source: https://en.wikipedia.org/wiki/Purged_cross-validation
3. Fit everything inside the training fold: scaling, the calibrator, the de-vig,
   hyperparameters. Fitting any of these on test data is the most common cause of
   overstated performance. Source: https://arxiv.org/pdf/2308.07832
4. Score model and market on exactly the same races, decided by one eligibility
   mask. Assert the two sets are identical. Watch survivorship, where dropping
   races with a missing feature quietly removes the hard ones.
5. Report a log-loss skill score against the market, 1 - L_model / L_market, not
   raw log loss, which is dominated by field size. Source: https://www.cawcr.gov.au/projects/verification/verif_web_page.html
6. Murphy decomposition of the Brier and log scores into reliability, resolution
   and uncertainty. Shows whether a model wins through calibration or through
   discrimination. Source: https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.2985
7. Reliability diagrams fit only on past data. Fit any calibrator on the training
   fold and apply forward. Isotonic overfits on small samples, so prefer Platt or
   logistic when a fold is small.
8. Significance with Diebold-Mariano and a paired bootstrap, and claim an edge
   only when both agree. Resample race days as blocks and cluster the standard
   error by day to respect within-day dependence. Control for multiple
   comparisons across model variants. Source: https://www.sas.upenn.edu/~fdiebold/papers/paper68/pa.dm.pdf
9. Pre-compute the minimum detectable effect from the per-race differential
   standard deviation and the test size. An honest edge over a sharp pool is
   single-digit thousandths of a nat, roughly half a percent to one percent skill.
   Anything much larger is almost certainly leakage.
10. Audit the feature pipeline for look-ahead and survivorship at construction
    time. Assert every feature input is timestamped before the race start.

Most overlooked for this project: entity-overlap leakage, which chronological
splitting alone does not remove; the block bootstrap by race day; and deciding the
minimum detectable effect before looking at any model result. Combinatorial purged
cross-validation can later give a distribution of skill scores rather than one
number.

## 4. Feature engineering and rating systems for harness racing

Each item notes whether it likely encodes information the market already prices,
in which case it mainly improves calibration and stability, or where mispriced
signal may remain.

Tier 1, build first:
- Distance and start-method normalised speed figure. Turn per-km time into a
  rating by subtracting a par expected km-time for distance bucket, auto versus
  volte, track and going, then standardising. The harness analogue of Beyer speed
  figures. Use only a horse's prior races. Auto starts run about a second per km
  faster than volte, so never pool them. Source: https://en.wikipedia.org/wiki/Beyer_Speed_Figure
- Daily track variant. Adjust each day's figures by the average deviation of
  completed runners from par, computed only from races finished before the target
  race.
- Recency-weighted form. Exponentially weight prior figures and finishes, with a
  half-life around 60 to 120 days, including a level and a trend.
- The de-vigged market as a calibration target and meta-feature, never a feature.

Tier 2:
- Empirical-Bayes-shrunk, time-decayed driver and trainer win and place rates.
  Shrink toward the population mean by sample size so low-sample entities do not
  look elite. Minor-driver skill is plausibly under-priced. Source: https://kiwidamien.github.io/shrinkage-and-empirical-bayes-to-improve-inference.html
- Elo or Glicko-2 ratings for horses and, separately, drivers, updated from
  finishing order in date order. Glicko-2 handles layoffs through a rating
  deviation that grows with idle time. Source: https://github.com/atomflunder/skillratings
- Post position by start method and track. Encode the draw as a historical
  win-rate-by-(post, track, start method) table estimated on the training period.
  Inside draws matter most on short ovals and standing starts.
- Class from prior earnings. Reconstruct cumulative earnings from prior results
  rather than the snapshot money field, to stay point-in-time.
- Layoff, days since last start, with non-linear buckets and a second-start-back
  flag.
- Equipment-change flags for shoes and sulky, genuine pre-race signal that may be
  under-priced.

Tier 3, higher effort:
- Pace and trip. The most promising unpriced signal, but not available in the
  data, see the pace note in docs/ROADMAP.md.
- Bayesian hierarchical random effects for horse, driver and track on the speed
  figure, giving shrunk latent abilities. Source: https://sciendo.com/article/10.2478/ijcss-2023-0007
- Entity embeddings and sequence models over a horse's history. Likely marginal
  over a good gradient boosting model at this size and leakage-prone, so a late
  experiment. Time-decayed target encoding captures most of the value at less
  risk. Source: https://www.robschumaker.com/publications/DSS%20-%20Machine%20Learning%20the%20Harness%20Track%20-%20Crowdsourcing%20and%20Varying%20Race%20History.pdf

Cheap supporting features: field size, sport split (trot versus monté, modelled
separately), sex and age with a peak around four to seven, and per-start handicap
distance.

Cross-cutting: every aggregate must be computed from strictly pre-start data in
race-date order; labels feed only prior-race aggregates and the benchmark; the
edge lives in residuals, so train and report against the de-vigged market. The
most plausible unpriced signal would be pace and well-shrunk minor-driver skill,
and most other features mainly buy calibration and robustness.
