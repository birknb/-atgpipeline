"""Evaluation harness for comparing a model to the market.

This is the ruler, built and tested before any model. Every model is scored
against the market on exactly the same races, with the difference assessed for
significance in a way that respects the fact that runners within a race, and
races within a day, are not independent.

All forecasts are long-format frames with one row per runner and the columns
race_id, number, p (predicted win probability that sums to one within a race)
and y (1 for the winner, 0 otherwise), the same shape metrics.py expects. The
race_dates argument maps race_id to a date string and is used to resample and
cluster by race day.

The headline number is the log-loss skill score against the market, 1 minus
L_model over L_market on the common races. Zero means no edge, positive means
the model beats the market. Significance comes from a paired bootstrap that
resamples race days and a Diebold-Mariano style test with a day-clustered
standard error. An edge is claimed only when both agree. The minimum detectable
effect says how small an edge the test set could even detect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import metrics


def race_dates_from_frame(frame: pd.DataFrame) -> dict:
    """Map race_id to its date, taken from a runners frame with a date column."""
    d = frame[["race_id", "date"]].drop_duplicates("race_id")
    return dict(zip(d["race_id"], d["date"]))


def common_races(*frames: pd.DataFrame) -> set:
    """Race ids present in every frame."""
    sets = [set(f["race_id"].unique()) for f in frames]
    out = sets[0]
    for s in sets[1:]:
        out = out & s
    return out


def _per_race_diff(
    model_frame: pd.DataFrame, market_frame: pd.DataFrame
) -> pd.Series:
    """Per-race log-loss difference, market minus model, on common races.

    A positive value means the model assigned higher probability to the winner
    than the market did, so positive favours the model.
    """
    ll_model = metrics.per_race_log_loss(model_frame)
    ll_market = metrics.per_race_log_loss(market_frame)
    common = ll_model.index.intersection(ll_market.index)
    return ll_market.loc[common] - ll_model.loc[common]


def _day_arrays(values: pd.Series, race_dates) -> tuple[np.ndarray, np.ndarray]:
    """Group per-race values by day. Returns per-day sums and per-day counts."""
    df = pd.DataFrame({"v": values.to_numpy()})
    df["day"] = [race_dates[r] for r in values.index]
    g = df.groupby("day")["v"]
    return g.sum().to_numpy(), g.count().to_numpy()


def paired_bootstrap_by_day(
    per_race_diff: pd.Series,
    race_dates,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Bootstrap the mean per-race log-loss difference, resampling race days.

    Resampling whole days rather than individual races respects within-day
    dependence, which an across-races resample would ignore. The estimand is the
    overall per-race mean, so each bootstrap sample sums the difference over the
    resampled days and divides by the resampled race count.
    """
    sums, counts = _day_arrays(per_race_diff, race_dates)
    n_days = len(sums)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_days, size=n_days)
        means[i] = sums[idx].sum() / counts[idx].sum()
    return {
        "estimate": float(sums.sum() / counts.sum()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n_days": int(n_days),
        "n_races": int(counts.sum()),
    }


def clustered_mean_test(per_race_diff: pd.Series, race_dates) -> dict:
    """Diebold-Mariano style test of the mean difference with a day-clustered
    standard error.

    For a one-step forecast the Diebold-Mariano test reduces to testing whether
    the mean loss difference is zero. The standard error clusters residuals by
    race day, which is robust to the correlation between races on the same day.
    A positive z with a small p-value means the model beats the market by more
    than day-to-day noise.
    """
    d = per_race_diff.to_numpy()
    n = len(d)
    mean = float(d.mean())
    resid = pd.Series(d - mean, index=per_race_diff.index)
    cluster_sums, _ = _day_arrays(resid, race_dates)
    var_mean = float((cluster_sums ** 2).sum()) / (n ** 2)
    se = float(np.sqrt(var_mean))
    if se > 0:
        z = mean / se
        p = float(2.0 * (1.0 - norm.cdf(abs(z))))
    elif mean == 0:
        # No mean difference and no variance: the two forecasts are identical.
        z, p = 0.0, 1.0
    else:
        # A nonzero mean with zero variance is a degenerate, perfectly
        # consistent difference. It is infinitely significant in principle.
        z, p = float(np.copysign(np.inf, mean)), 0.0
    return {
        "mean": mean,
        "se": se,
        "z": float(z),
        "p_value": p,
        "n_races": int(n),
    }


def minimum_detectable_effect(
    per_race_diff: pd.Series,
    race_dates,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Smallest true mean log-loss difference the test set could detect.

    Uses the day-clustered standard error, so it is consistent with the
    significance test. A measured improvement smaller than this is below the
    detection floor and must not be read as a real edge.
    """
    se = clustered_mean_test(per_race_diff, race_dates)["se"]
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    z_power = norm.ppf(power)
    return float((z_alpha + z_power) * se)


def murphy_decomposition(
    frame: pd.DataFrame, n_bins: int = 10, strategy: str = "quantile"
) -> dict:
    """Reliability, resolution and uncertainty of the runner-level forecast.

    The decomposition is computed on the runner level, treating each runner's p
    as a binary forecast of the event that this runner wins. The Brier score
    equals reliability minus resolution plus uncertainty when forecasts are
    binned. Lower reliability is better calibration, higher resolution is better
    discrimination, and uncertainty is fixed by the base win rate.
    """
    tab = metrics.calibration_table(frame, n_bins=n_bins, strategy=strategy)
    y = frame["y"].to_numpy(dtype=float)
    n = len(y)
    base = float(y.mean())
    uncertainty = base * (1.0 - base)

    counts = tab["count"].to_numpy(dtype=float)
    p_mean = tab["p_mean"].to_numpy(dtype=float)
    y_rate = tab["y_rate"].to_numpy(dtype=float)
    reliability = float((counts * (p_mean - y_rate) ** 2).sum() / n)
    resolution = float((counts * (y_rate - base) ** 2).sum() / n)

    brier_runner = float(((frame["p"].to_numpy(dtype=float) - y) ** 2).mean())
    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier_runner": brier_runner,
        "binned_reconstruction": reliability - resolution + uncertainty,
        "base_rate": base,
    }


def compare(
    model_frame: pd.DataFrame,
    market_frame: pd.DataFrame,
    race_dates,
    label: str = "model",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Score a model against the market on the common races.

    Restricts both forecasts to the races they share, then reports both log
    losses, the skill score, the paired day bootstrap, the clustered test and
    the minimum detectable effect. Records how many races were unique to each
    side, which should be zero when the caller passes a shared eligible set.
    """
    m_races = set(model_frame["race_id"].unique())
    k_races = set(market_frame["race_id"].unique())
    common = m_races & k_races

    modf = model_frame[model_frame["race_id"].isin(common)]
    mktf = market_frame[market_frame["race_id"].isin(common)]

    l_model = metrics.log_loss(modf)
    l_market = metrics.log_loss(mktf)
    skill = 1.0 - l_model / l_market if l_market > 0 else float("nan")

    diff = _per_race_diff(modf, mktf)
    boot = paired_bootstrap_by_day(diff, race_dates, n_boot=n_boot, seed=seed)
    test = clustered_mean_test(diff, race_dates)
    mde = minimum_detectable_effect(diff, race_dates)

    ci_excludes_zero = boot["ci_low"] > 0 or boot["ci_high"] < 0
    significant = bool(ci_excludes_zero and test["p_value"] < 0.05)

    return {
        "label": label,
        "n_races": int(len(common)),
        "races_model_only": int(len(m_races - k_races)),
        "races_market_only": int(len(k_races - m_races)),
        "log_loss_model": l_model,
        "log_loss_market": l_market,
        "brier_model": metrics.brier(modf),
        "brier_market": metrics.brier(mktf),
        "skill_score": float(skill),
        "mean_diff": test["mean"],
        "boot_ci": (boot["ci_low"], boot["ci_high"]),
        "dm_z": test["z"],
        "dm_p_value": test["p_value"],
        "min_detectable_effect": mde,
        "significant": significant,
    }


def print_report(res: dict) -> None:
    """Readable summary of a compare result."""
    print(f"\n[{res['label']} vs market] {res['n_races']:,} common races")
    if res["races_model_only"] or res["races_market_only"]:
        print(
            f"  WARNING race sets differ: model only {res['races_model_only']}, "
            f"market only {res['races_market_only']}"
        )
    print(f"  log loss   model {res['log_loss_model']:.4f}   market {res['log_loss_market']:.4f}")
    print(f"  Brier      model {res['brier_model']:.4f}   market {res['brier_market']:.4f}")
    print(f"  skill score          {res['skill_score']:+.4f}  (positive means model beats market)")
    print(f"  mean per-race diff   {res['mean_diff']:+.5f}  (positive favours model)")
    print(f"  bootstrap 95% CI     [{res['boot_ci'][0]:+.5f}, {res['boot_ci'][1]:+.5f}]")
    print(f"  DM z {res['dm_z']:+.2f}  p {res['dm_p_value']:.4f}")
    print(f"  min detectable effect {res['min_detectable_effect']:.5f}")
    print(f"  significant edge: {'YES' if res['significant'] else 'no'}")
