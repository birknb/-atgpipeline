"""Scoring metrics for win-probability forecasts.

All functions operate on a long-format table with one row per runner and the
columns race_id, p (predicted win probability) and y (1 for the winner, 0
otherwise). Within each race the rows are the runners that started, the
predicted probabilities are expected to sum to one, and exactly one runner
has y = 1.

The headline metric is the multinomial log loss: the average over races of
-log(p) assigned to the actual winner. This is the quantity the project tries
to make smaller than the market benchmark. The Brier score and the
calibration table are reported alongside it. The paired bootstrap quantifies
whether a difference in log loss between two forecasts is larger than race to
race noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Lower bound on probabilities before taking a logarithm. A forecast that puts
# zero probability on the winner would otherwise produce an infinite loss.
EPS = 1e-15


def normalize_within_group(
    df: pd.DataFrame, score_col: str, group_col: str = "race_id"
) -> pd.Series:
    """Return scores divided by their per-group sum, so each group sums to one.

    Used to turn raw implied weights (for example 1 / final_odds) or model
    scores into a probability distribution over the runners in a race.
    """
    totals = df.groupby(group_col)[score_col].transform("sum")
    return df[score_col] / totals


def _check_one_winner(df: pd.DataFrame, group_col: str, y_col: str) -> None:
    wins = df.groupby(group_col)[y_col].sum()
    bad = wins[wins != 1]
    if len(bad) > 0:
        raise ValueError(
            f"{len(bad)} races do not have exactly one winner. "
            f"Filter these before scoring. Examples: {list(bad.index[:5])}"
        )


def per_race_log_loss(
    df: pd.DataFrame,
    p_col: str = "p",
    y_col: str = "y",
    group_col: str = "race_id",
) -> pd.Series:
    """Multinomial log loss per race, indexed by race_id.

    The per-race value is -log(p) of the runner that won. Returning one value
    per race lets the caller pair two forecasts race by race for the bootstrap.
    """
    _check_one_winner(df, group_col, y_col)
    winners = df[df[y_col] == 1].copy()
    p = winners[p_col].clip(lower=EPS, upper=1.0)
    return pd.Series(-np.log(p.to_numpy()), index=winners[group_col].to_numpy())


def log_loss(
    df: pd.DataFrame,
    p_col: str = "p",
    y_col: str = "y",
    group_col: str = "race_id",
) -> float:
    """Mean multinomial log loss over races."""
    return float(per_race_log_loss(df, p_col, y_col, group_col).mean())


def per_race_brier(
    df: pd.DataFrame,
    p_col: str = "p",
    y_col: str = "y",
    group_col: str = "race_id",
) -> pd.Series:
    """Multi-category Brier score per race, indexed by race_id.

    The per-race value is the sum over runners of (p - y) squared. With one
    winner this lies between 0 and 2. It rewards both a high probability on the
    winner and low probability on the losers.
    """
    _check_one_winner(df, group_col, y_col)
    sq = (df[p_col] - df[y_col]) ** 2
    return sq.groupby(df[group_col]).sum()


def brier(
    df: pd.DataFrame,
    p_col: str = "p",
    y_col: str = "y",
    group_col: str = "race_id",
) -> float:
    """Mean multi-category Brier score over races."""
    return float(per_race_brier(df, p_col, y_col, group_col).mean())


def calibration_table(
    df: pd.DataFrame,
    p_col: str = "p",
    y_col: str = "y",
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Bin runner-level predictions and compare predicted to observed wins.

    Returns one row per non-empty bin with the mean predicted probability, the
    observed win frequency, and the count. strategy 'quantile' uses equal-count
    bins, 'uniform' uses equal-width bins on [0, 1]. A well calibrated forecast
    has mean predicted close to observed in every bin.
    """
    p = df[p_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.unique(edges)
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    # np.digitize with the last edge inclusive on the right.
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "p_mean": float(p[mask].mean()),
                "y_rate": float(y[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_logloss_diff(
    loss_a: pd.Series,
    loss_b: pd.Series,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Bootstrap the mean per-race log loss difference (a minus b).

    loss_a and loss_b are per-race losses indexed by race_id, as returned by
    per_race_log_loss. Only races present in both are used, so the comparison
    is on a common set. Resampling is over races, which respects the fact that
    runners within a race are not independent. A positive estimate with a
    confidence interval above zero means forecast b has lower log loss than
    forecast a by more than race to race noise.
    """
    common = loss_a.index.intersection(loss_b.index)
    a = loss_a.loc[common].to_numpy()
    b = loss_b.loc[common].to_numpy()
    diff = a - b
    n = len(diff)
    if n == 0:
        raise ValueError("No races in common between the two forecasts")

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.integers(0, n, size=n)
        means[i] = diff[sample].mean()

    return {
        "n_races": int(n),
        "estimate": float(diff.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }
