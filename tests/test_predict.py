"""Offline plumbing checks for the prediction engine (atg/predict.py).

Builds a synthetic eligible-feature frame spanning enough history, then predicts
the last day and checks the engine runs and returns valid within-race
distributions. The required feature columns are taken from the model module, so
this stays in step if the feature set changes. Run:

    python tests/test_predict.py
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atg import model, predict  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, cond: bool) -> None:
    checks.append((name, bool(cond)))


def build_frame() -> tuple[pd.DataFrame, str]:
    # Every feature column the model reads, defaulted to neutral values. The
    # barefoot and American-sulky features are derived in prepare from these
    # sources, so they are not set directly.
    base = [c for c in model.RAW_FEATURES
            if c not in ("barefoot_front", "barefoot_back", "is_american_sulky")]
    zero_cols = set(base) | set(model.LOG_FEATURES.keys())

    start = date(2024, 1, 1)
    rows = []
    n_days = 130  # history, then a target day after it

    def add_race(rid, d, n_runners):
        winner = rid % n_runners
        for i in range(n_runners):
            row = {c: 0.0 for c in zero_cols}
            row.update(
                start_id=f"{rid}_{i}", race_id=str(rid), number=i + 1,
                horse_id=1000 + i, date=d, is_winner=1 if i == winner else 0,
                # Winner gets shorter odds, so the market is informative.
                final_odds=2.5 if i == winner else 9.0,
                age=5.0, sex="gelding", shoe_front_on=1, shoe_back_on=1,
                sulky_type="Standard",
            )
            rows.append(row)

    rid = 0
    for day in range(n_days):
        add_race(rid, (start + timedelta(days=day)).isoformat(), 6)
        rid += 1
    target = (start + timedelta(days=n_days + 1)).isoformat()
    for _ in range(3):  # three races on the target day
        add_race(rid, target, 6)
        rid += 1
    return pd.DataFrame(rows), target


def main() -> int:
    df, target = build_frame()
    out, info = predict.predict(df, target)

    check("returns rows for the target day", (out["date"] == target).all() and len(out) == 18)
    check("three target races", out["race_id"].nunique() == 3)
    for col in ["p_market", "p_market_flb", "p_fundamental", "p_combination"]:
        check(f"{col} present", col in out.columns)
    sums = out.groupby("race_id")["p_combination"].sum()
    check("combination sums to one per race", bool(((sums - 1.0).abs() < 1e-6).all()))
    msums = out.groupby("race_id")["p_market"].sum()
    check("market sums to one per race", bool(((msums - 1.0).abs() < 1e-6).all()))
    check("weights returned", "combo_fund_weight" in info and "flb_power" in info)

    raised = False
    try:
        predict.predict(df, "2099-01-01")
    except ValueError:
        raised = True
    check("raises on a date with no races", raised)

    print("=== Prediction engine test results ===")
    ok = True
    for name, cond in checks:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond
    print("\nALL PASSED" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
