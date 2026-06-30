"""Frozen-model prediction for one race day, the core of Phase 4.

Phase 4 logs predictions before each race and scores them later against the
outcome, on fresh data no model has seen. The prediction engine here is the
reusable core. It fits the model on all races before a target date, then
produces win probabilities for that date's races: the fundamental model, the raw
market, the favourite-longshot-recalibrated market, and the combination.

The live wrapper, which fetches the race card near the off and must run on a
machine that can reach atg.se, is not here. Its job is only to bring today's
cards into the database (the existing ingestion does this), after which this
engine predicts and a snapshot is stored. Because the model is refit on all data
up to the target date and the features are strictly point-in-time, there is no
look-ahead.

Usage:
    python -m atg.predict --db data/atg.sqlite --date 2026-06-28
    python -m atg.predict --db data/atg.sqlite --date 2026-06-28 --save
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from . import model

log = logging.getLogger("atg.predict")


def predict(df: pd.DataFrame, target_date: str, val_days: int = 60):
    """Fit on races before target_date, predict that date's races.

    df is the eligible feature frame. Returns the per-runner prediction frame and
    a dict of the fitted weights. Raises if the target date has no races or there
    is too little history before it."""
    df = df.sort_values("date").reset_index(drop=True)
    y = df["is_winner"].to_numpy().astype(int)

    train_mask = (df["date"] < target_date).to_numpy()
    val_cut = (date.fromisoformat(target_date) - timedelta(days=val_days)).isoformat()
    val_mask = ((df["date"] < target_date) & (df["date"] >= val_cut)).to_numpy()
    core_mask = (df["date"] < val_cut).to_numpy()
    test_mask = (df["date"] == target_date).to_numpy()
    if test_mask.sum() == 0:
        raise ValueError(f"no races on {target_date}")
    if core_mask.sum() == 0 or val_mask.sum() == 0:
        raise ValueError(f"not enough history before {target_date}")

    X, _ = model.prepare(df, pd.Series(train_mask))
    cc, gc = model._codes(df.loc[core_mask, "race_id"])
    beta = model.fit_conditional_logit(X[core_mask], y[core_mask], cc, gc)

    mkt_all = model.market_frame(df)
    a = model.fit_flb_power(mkt_all[train_mask])
    cv, gv = model._codes(df.loc[val_mask, "race_id"])
    f_val = model.grouped_softmax(X[val_mask] @ beta, cv, gv)
    coef = model.fit_market_combination(f_val, mkt_all.loc[val_mask, "p"].to_numpy(), df[val_mask])

    test = df[test_mask]
    ct, gt = model._codes(test["race_id"])
    p_fund = model.grouped_softmax(X[test_mask] @ beta, ct, gt)
    mkt_test = mkt_all[test_mask].copy()
    p_combo = model.apply_market_combination(p_fund, mkt_test["p"].to_numpy(), test, coef)

    out = test[["start_id", "race_id", "number", "horse_id", "date", "is_winner"]].copy()
    out["p_market"] = mkt_test["p"].to_numpy()
    out["p_market_flb"] = model.apply_power(mkt_test, a)["p"].to_numpy()
    out["p_fundamental"] = p_fund
    out["p_combination"] = p_combo
    info = {
        "flb_power": float(a),
        "combo_fund_weight": float(coef[0]),
        "combo_market_weight": float(coef[1]),
        "n_races": int(test["race_id"].nunique()),
        "n_runners": int(len(test)),
    }
    return out.reset_index(drop=True), info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/atg.sqlite")
    parser.add_argument("--sport", default="trot")
    parser.add_argument("--date", required=True, help="target race date, YYYY-MM-DD")
    parser.add_argument("--save", action="store_true", help="append to a predictions table")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    conn = sqlite3.connect(args.db)
    df = model.eligible(model.load_features(conn, args.sport))
    out, info = predict(df, args.date)

    print(f"=== predictions for {args.date} ===")
    print(f"races {info['n_races']}, runners {info['n_runners']}")
    print(f"combination weights: fundamental {info['combo_fund_weight']:+.3f}, "
          f"market {info['combo_market_weight']:+.3f}, flb power {info['flb_power']:.3f}")
    # Sanity: probabilities sum to one within each race.
    sums = out.groupby("race_id")["p_combination"].sum()
    print(f"per-race probability sum: min {sums.min():.4f}, max {sums.max():.4f}")

    if args.save:
        out = out.copy()
        out["predicted_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        out.to_sql("predictions", conn, if_exists="append", index=False)
        print(f"saved {len(out)} rows to the predictions table")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
