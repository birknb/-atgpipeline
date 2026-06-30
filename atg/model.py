"""Phase 3 models: market baselines and fundamental models scored against the market.

This fits and evaluates win-probability models on a time-based split. It is the
fast fixed-split phase, so every number it prints is provisional and is labelled
pre-walk-forward. Only the later walk-forward evaluation produces quotable
results.

Models:

  market        de-vigged final win odds, the benchmark.
  market_flb    the market recalibrated for the favourite-longshot bias, a power
                transform fitted on the training races. This is the honest
                baseline, since part of any win over the raw market is only this
                recalibration.
  clogit        a conditional (multinomial) logit over the runners in each race,
                on within-race-centred point-in-time features. No market input.
  lgbm          LightGBM with a binary objective and per-race renormalisation, on
                the same features. No market input.

The fundamental models never see the odds, so the comparison against the market
is not circular. Features are centred within each race, because only within-race
differences affect a runner's win probability.

Usage:
    python -m atg.model --db data/atg.sqlite
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from . import evaluate, metrics

log = logging.getLogger("atg.model")

# Runner-varying features. Race-constant fields such as field size or distance
# are excluded, since they cancel in a within-race comparison.
RAW_FEATURES = [
    "post_rel", "age", "is_debut", "form_speed", "elo", "elo_default",
    "driver_win_rate", "driver_place_rate", "trainer_win_rate",
    "trainer_place_rate", "shoes_changed", "sulky_changed",
    # as-of-race API statistics, verified point-in-time safe
    "stat_win_pct", "stat_place_pct", "best_km_time_s", "stat_start_points",
]
# Skewed counts and money, entered after a log transform.
LOG_FEATURES = {
    "days_since_last": "t_days_since_last",
    "form_n": "t_form_n",
    "cum_earnings": "t_cum_earnings",
    "avg_earn_per_start": "t_avg_earn",
    "driver_n": "t_driver_n",
    "trainer_n": "t_trainer_n",
    "stat_earn_per_start": "t_stat_earn",
    "stat_life_starts": "t_stat_starts",
}
# Derived in prepare: an age curve and sex indicators (gelding is the baseline).
DERIVED_FEATURES = ["age2", "is_mare", "is_stallion"]

# Fixed split. Provisional, pre-walk-forward.
TRAIN_END = "2025-09-30"
VAL_END = "2025-12-31"
# test is everything after VAL_END.


def load_features(conn: sqlite3.Connection, sport: str = "trot") -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM norm_features", conn)
    if sport:
        df = df[df["sport"] == sport].copy()
    return df


def eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Keep races with exactly one winner and a positive final odds for every
    runner. This matches the market benchmark's eligibility so the two are scored
    on the same races."""
    wins = df.groupby("race_id")["is_winner"].sum()
    one_winner = set(wins[wins == 1].index)
    bad = set(df.loc[~(df["final_odds"] > 0), "race_id"])
    keep = one_winner - bad
    return df[df["race_id"].isin(keep)].copy()


def prepare(df: pd.DataFrame, train_mask: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Build the model matrix. Apply log transforms, impute missing values with
    the race mean, centre each feature within its race, and scale by the global
    training standard deviation so coefficients share one scale. The scale is fit
    on training rows only."""
    df = df.copy()
    df["age2"] = df["age"].astype(float) ** 2
    df["is_mare"] = (df["sex"] == "mare").astype(float)
    df["is_stallion"] = (df["sex"] == "stallion").astype(float)
    for src, dst in LOG_FEATURES.items():
        df[dst] = np.log1p(df[src].clip(lower=0))
    feats = RAW_FEATURES + DERIVED_FEATURES + list(LOG_FEATURES.values())

    cols = []
    for f in feats:
        race_mean = df.groupby("race_id")[f].transform("mean")
        filled = df[f].fillna(race_mean).fillna(0.0)
        centred = filled - filled.groupby(df["race_id"]).transform("mean")
        std = centred[train_mask].std()
        df[f + "_z"] = centred / (std if std and std > 0 else 1.0)
        cols.append(f + "_z")
    return df[cols].to_numpy(dtype=float), cols


def _codes(race_ids: pd.Series) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(race_ids, sort=False)
    return codes.astype(np.intp), len(uniques)


def grouped_softmax(v: np.ndarray, codes: np.ndarray, g: int) -> np.ndarray:
    m = np.full(g, -np.inf)
    np.maximum.at(m, codes, v)
    e = np.exp(v - m[codes])
    s = np.zeros(g)
    np.add.at(s, codes, e)
    return e / s[codes]


def _group_lse(v: np.ndarray, codes: np.ndarray, g: int) -> float:
    m = np.full(g, -np.inf)
    np.maximum.at(m, codes, v)
    e = np.exp(v - m[codes])
    s = np.zeros(g)
    np.add.at(s, codes, e)
    return float((m + np.log(s)).sum())


def fit_conditional_logit(X: np.ndarray, y: np.ndarray, codes: np.ndarray,
                          g: int, lam: float = 1.0) -> np.ndarray:
    """Maximise the conditional-logit likelihood with an L2 penalty. Each race is
    a choice set, the winner is the chosen runner, the probability is the softmax
    of a linear index within the race."""
    win = y == 1

    def negll(beta):
        v = X @ beta
        ll = v[win].sum() - _group_lse(v, codes, g)
        return -(ll - 0.5 * lam * beta @ beta)

    def grad(beta):
        v = X @ beta
        p = grouped_softmax(v, codes, g)
        gll = X[win].sum(axis=0) - X.T @ p
        return -(gll - lam * beta)

    res = minimize(negll, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B")
    return res.x


def fit_flb_power(market: pd.DataFrame) -> float:
    """Fit the single power exponent that best recalibrates the market on the
    training races. Above one shifts mass toward favourites."""
    def loss(a):
        g = market[["race_id", "y"]].copy()
        w = market["p"].to_numpy() ** a
        g["p"] = w / pd.Series(w, index=market.index).groupby(market["race_id"]).transform("sum")
        return metrics.log_loss(g)
    return float(minimize_scalar(loss, bounds=(0.5, 2.0), method="bounded").x)


def apply_power(market: pd.DataFrame, a: float) -> pd.DataFrame:
    w = market["p"].to_numpy() ** a
    out = market.copy()
    out["p"] = w / pd.Series(w, index=market.index).groupby(market["race_id"]).transform("sum")
    return out


def fit_market_combination(f: np.ndarray, m: np.ndarray, df: pd.DataFrame,
                           lam: float = 0.01) -> np.ndarray:
    """Benter-style stage two. Fit weights on the log fundamental probability and
    the log market probability so the combined probability is proportional to
    f**alpha times m**beta within each race. Fitting uses a held-out block the
    fundamental model did not train on."""
    eps = 1e-12
    X = np.column_stack([np.log(np.clip(f, eps, 1.0)), np.log(np.clip(m, eps, 1.0))])
    codes, g = _codes(df["race_id"])
    y = df["is_winner"].to_numpy().astype(int)
    return fit_conditional_logit(X, y, codes, g, lam=lam)


def apply_market_combination(f: np.ndarray, m: np.ndarray, df: pd.DataFrame,
                             coef: np.ndarray) -> np.ndarray:
    eps = 1e-12
    X = np.column_stack([np.log(np.clip(f, eps, 1.0)), np.log(np.clip(m, eps, 1.0))])
    codes, g = _codes(df["race_id"])
    return grouped_softmax(X @ coef, codes, g)


def market_frame(df: pd.DataFrame) -> pd.DataFrame:
    raw = 1.0 / df["final_odds"].to_numpy()
    out = df[["race_id", "number", "date"]].copy()
    out["p"] = raw / pd.Series(raw, index=df.index).groupby(df["race_id"]).transform("sum")
    out["y"] = df["is_winner"].astype(int).to_numpy()
    return out


def model_frame(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    out = df[["race_id", "number", "date"]].copy()
    out["p"] = p
    out["y"] = df["is_winner"].astype(int).to_numpy()
    return out


def fit_lgbm(Xtr, ytr, Xval, yval):
    import lightgbm as lgb
    dtr = lgb.Dataset(Xtr, label=ytr)
    dval = lgb.Dataset(Xval, label=yval, reference=dtr)
    params = {
        "objective": "binary",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
    }
    return lgb.train(
        params, dtr, num_boost_round=2000, valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )


def renormalise(df: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    s = pd.Series(raw, index=df.index)
    return (raw / s.groupby(df["race_id"]).transform("sum")).to_numpy()


def walk_forward(df: pd.DataFrame, first_test: str = "2025-01-01",
                 span_days: int = 91, val_days: int = 60, n_boot: int = 2000) -> None:
    """Expanding-window walk-forward evaluation. Each fold trains on all races
    before its test window, refits the feature scale, the models, the
    favourite-longshot power and the combination weights, and predicts the test
    window. Predictions are accumulated across folds and scored once on the union.
    These are the quotable results, not pre-walk-forward."""
    df = df.sort_values("date").reset_index(drop=True)
    y = df["is_winner"].to_numpy().astype(int)
    mkt_all = market_frame(df)
    dmax = df["date"].max()

    acc = {k: [] for k in ["market", "market_flb", "clogit", "lgbm", "combo"]}
    t0 = date.fromisoformat(first_test)
    n_folds = 0
    while t0.isoformat() <= dmax:
        t1 = t0 + timedelta(days=span_days)
        ts, te = t0.isoformat(), t1.isoformat()
        val_cut = (t0 - timedelta(days=val_days)).isoformat()
        train_mask = (df["date"] < ts).to_numpy()
        val_mask = ((df["date"] < ts) & (df["date"] >= val_cut)).to_numpy()
        core_mask = ((df["date"] < val_cut)).to_numpy()
        test_mask = ((df["date"] >= ts) & (df["date"] < te)).to_numpy()
        if test_mask.sum() == 0 or core_mask.sum() == 0 or val_mask.sum() == 0:
            t0 = t1
            continue

        X, _ = prepare(df, pd.Series(train_mask))
        cc, gc = _codes(df.loc[core_mask, "race_id"])
        beta = fit_conditional_logit(X[core_mask], y[core_mask], cc, gc)
        booster = fit_lgbm(X[core_mask], y[core_mask], X[val_mask], y[val_mask])

        test = df[test_mask]
        ct, gt = _codes(test["race_id"])
        p_cl = grouped_softmax(X[test_mask] @ beta, ct, gt)
        p_lg = renormalise(test, booster.predict(X[test_mask]))

        a = fit_flb_power(mkt_all[train_mask])
        mkt_test = mkt_all[test_mask].copy()
        cv, gv = _codes(df.loc[val_mask, "race_id"])
        f_val = grouped_softmax(X[val_mask] @ beta, cv, gv)
        coef = fit_market_combination(f_val, mkt_all.loc[val_mask, "p"].to_numpy(), df[val_mask])
        p_combo = apply_market_combination(p_cl, mkt_test["p"].to_numpy(), test, coef)

        acc["market"].append(mkt_test)
        acc["market_flb"].append(apply_power(mkt_test, a))
        acc["clogit"].append(model_frame(test, p_cl))
        acc["lgbm"].append(model_frame(test, p_lg))
        acc["combo"].append(model_frame(test, p_combo))
        n_folds += 1
        t0 = t1

    frames = {k: pd.concat(v).reset_index(drop=True) for k, v in acc.items()}
    rd = evaluate.race_dates_from_frame(frames["market"])
    n_races = frames["market"]["race_id"].nunique()
    print("=== WALK-FORWARD evaluation (quotable) ===")
    print(f"{n_folds} folds, {n_races:,} test races from {first_test} to {dmax}")
    pairs = [
        ("market_flb vs market", "market_flb", "market"),
        ("clogit vs market", "clogit", "market"),
        ("lgbm vs market", "lgbm", "market"),
        ("combo vs market", "combo", "market"),
        ("combo vs market_flb", "combo", "market_flb"),
    ]
    for label, model, market in pairs:
        res = evaluate.compare(frames[model], frames[market], rd, label=label, n_boot=n_boot)
        evaluate.print_report(res)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/atg.sqlite")
    parser.add_argument("--sport", default="trot")
    parser.add_argument("--walk", action="store_true", help="run the walk-forward evaluation")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    conn = sqlite3.connect(args.db)
    df = eligible(load_features(conn, args.sport))
    conn.close()
    df = df.sort_values("race_id").reset_index(drop=True)

    if args.walk:
        walk_forward(df)
        return 0

    train_mask = df["date"] <= TRAIN_END
    val_mask = (df["date"] > TRAIN_END) & (df["date"] <= VAL_END)
    test_mask = df["date"] > VAL_END
    print("=== Phase 3 models (PRE-WALK-FORWARD, provisional) ===")
    for name, m in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        print(f"  {name:5s} races {df.loc[m, 'race_id'].nunique():>7,}  runners {int(m.sum()):>8,}")

    X, cols = prepare(df, train_mask)
    y = df["is_winner"].to_numpy().astype(int)

    # Conditional logit, fit on training races only.
    codes_tr, g_tr = _codes(df.loc[train_mask, "race_id"])
    beta = fit_conditional_logit(X[train_mask.to_numpy()], y[train_mask.to_numpy()], codes_tr, g_tr)

    test = df[test_mask].copy()
    codes_te, g_te = _codes(test["race_id"])
    p_clogit = grouped_softmax(X[test_mask.to_numpy()] @ beta, codes_te, g_te)

    # LightGBM binary plus per-race renormalisation, early-stopped on the val block.
    booster = fit_lgbm(
        X[train_mask.to_numpy()], y[train_mask.to_numpy()],
        X[val_mask.to_numpy()], y[val_mask.to_numpy()],
    )
    p_lgbm = renormalise(test, booster.predict(X[test_mask.to_numpy()]))

    # Market frames and the favourite-longshot recalibration fitted on train.
    mkt_all = market_frame(df)
    a = fit_flb_power(mkt_all[train_mask.to_numpy()])
    mkt_test = mkt_all[test_mask.to_numpy()].copy()
    mkt_flb_test = apply_power(mkt_test, a)
    print(f"\nfavourite-longshot power a = {a:.3f} (fit on train)")

    # Market-combination ceiling. Stage-two weights are fit on the validation
    # block, then applied to test. This uses the odds, so it is a labelled
    # ceiling that measures whether the features hold signal orthogonal to the
    # crowd, never a fair standalone beat.
    val = df[val_mask].copy()
    cv, gv = _codes(val["race_id"])
    f_val = grouped_softmax(X[val_mask.to_numpy()] @ beta, cv, gv)
    m_val = mkt_all.loc[val_mask.to_numpy(), "p"].to_numpy()
    coef = fit_market_combination(f_val, m_val, val)
    p_combo = apply_market_combination(p_clogit, mkt_test["p"].to_numpy(), test, coef)
    print(f"market combination weights: fundamental {coef[0]:+.3f}, market {coef[1]:+.3f}")

    rd = evaluate.race_dates_from_frame(test)
    f_clogit = model_frame(test, p_clogit)
    f_lgbm = model_frame(test, p_lgbm)
    f_combo = model_frame(test, p_combo)

    print("\n--- test set, PRE-WALK-FORWARD, provisional ---")
    pairs = [
        ("market_flb vs market", mkt_flb_test, mkt_test),
        ("clogit vs market", f_clogit, mkt_test),
        ("lgbm vs market", f_lgbm, mkt_test),
        ("combo vs market", f_combo, mkt_test),
        ("combo vs market_flb", f_combo, mkt_flb_test),
    ]
    for label, model, market in pairs:
        res = evaluate.compare(model, market, rd, label=label, n_boot=2000)
        res["label"] = label
        evaluate.print_report(res)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
