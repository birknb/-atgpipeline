"""Offline checks for the evaluation harness (atg/splits.py, atg/evaluate.py).

Runs without network or a database. Builds small synthetic forecast frames with
known log losses and checks the arithmetic, the split logic with purge, and the
significance machinery. Run directly:

    python tests/test_evaluate.py
"""
from __future__ import annotations

import math
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atg import evaluate, splits  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, cond: bool) -> None:
    checks.append((name, bool(cond)))


def approx(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol


def make_frame(races: list[dict]) -> pd.DataFrame:
    """races: list of {race_id, date, p (list), winner (index)}."""
    rows = []
    for r in races:
        for i, p in enumerate(r["p"]):
            rows.append(
                {
                    "race_id": r["race_id"],
                    "date": r["date"],
                    "number": i + 1,
                    "p": p,
                    "y": 1 if i == r["winner"] else 0,
                }
            )
    return pd.DataFrame(rows)


def test_arithmetic() -> None:
    market = make_frame(
        [
            {"race_id": "A", "date": "2024-01-01", "p": [0.5, 0.5], "winner": 0},
            {"race_id": "B", "date": "2024-01-02", "p": [0.25] * 4, "winner": 0},
        ]
    )
    model = make_frame(
        [
            {"race_id": "A", "date": "2024-01-01", "p": [0.8, 0.2], "winner": 0},
            {"race_id": "B", "date": "2024-01-02", "p": [0.4, 0.2, 0.2, 0.2], "winner": 0},
        ]
    )
    rd = evaluate.race_dates_from_frame(market)
    res = evaluate.compare(model, market, rd, label="toy", n_boot=500)

    exp_market = (math.log(2) + math.log(4)) / 2
    exp_model = (-math.log(0.8) - math.log(0.4)) / 2
    exp_skill = 1 - exp_model / exp_market
    check("market log loss arithmetic", approx(res["log_loss_market"], exp_market))
    check("model log loss arithmetic", approx(res["log_loss_model"], exp_model))
    check("skill score arithmetic", approx(res["skill_score"], exp_skill))
    check("mean diff arithmetic", approx(res["mean_diff"], 0.470004))
    check("no race-set mismatch", res["races_model_only"] == 0 and res["races_market_only"] == 0)
    check("bootstrap estimate equals mean diff", approx(res["boot_ci"][0], res["mean_diff"], 0.2))


def _synthetic_races(n_races: int, field: int, seed: int):
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 1)
    market_rows, model_rows = [], []
    for k in range(n_races):
        rid = f"r{k:04d}"
        d = (start + timedelta(days=k // 4)).isoformat()  # 4 races per day
        winner = int(rng.integers(0, field))
        # market: uniform
        mp = [1.0 / field] * field
        # model: extra weight on the winner, so the model is consistently
        # better. The weight varies across races, so the per-race difference
        # has non-zero variance and the clustered test is exercised properly.
        w = np.ones(field)
        w[winner] = float(rng.uniform(1.5, 3.5))
        w = w / w.sum()
        for i in range(field):
            market_rows.append({"race_id": rid, "date": d, "number": i + 1, "p": mp[i], "y": 1 if i == winner else 0})
            model_rows.append({"race_id": rid, "date": d, "number": i + 1, "p": float(w[i]), "y": 1 if i == winner else 0})
    return pd.DataFrame(model_rows), pd.DataFrame(market_rows)


def test_significance() -> None:
    model, market = _synthetic_races(n_races=400, field=8, seed=0)
    rd = evaluate.race_dates_from_frame(market)
    res = evaluate.compare(model, market, rd, n_boot=500)
    check("model better gives positive skill", res["skill_score"] > 0)
    check("model better gives positive mean diff", res["mean_diff"] > 0)
    check("model better gives positive DM z", res["dm_z"] > 0)
    check("consistent edge is significant", res["significant"] is True)
    check("MDE is positive", res["min_detectable_effect"] > 0)

    # Equal forecasts: no edge, not significant.
    eq = evaluate.compare(market.copy(), market, rd, n_boot=200)
    check("equal forecasts give zero skill", approx(eq["skill_score"], 0.0))
    check("equal forecasts not significant", eq["significant"] is False)
    check("equal forecasts DM p is 1", approx(eq["dm_p_value"], 1.0))


def test_murphy() -> None:
    model, market = _synthetic_races(n_races=400, field=8, seed=1)
    dec = evaluate.murphy_decomposition(model, n_bins=10)
    base = dec["base_rate"]
    check("uncertainty equals base rate variance", approx(dec["uncertainty"], base * (1 - base)))
    check("reliability non-negative", dec["reliability"] >= 0)
    check("resolution non-negative", dec["resolution"] >= 0)
    check("base rate near one over field", approx(base, 1 / 8, 0.02))


def test_splits() -> None:
    start = date(2024, 1, 1)
    rows = []
    for k in range(91):  # one race per day, Jan to end of March 2024
        d = (start + timedelta(days=k)).isoformat()
        rows.append({"race_id": f"d{d}", "date": d})
    races = pd.DataFrame(rows)

    fs = splits.fixed_split(races, train_end="2024-02-29", val_end="2024-03-15", purge_days=5)
    train, val, test = set(fs["train"]), set(fs["val"]), set(fs["test"])
    check("split partitions disjoint", not (train & val) and not (val & test) and not (train & test))
    check("train ends at purge cut", "d2024-02-24" in train and "d2024-02-25" not in train)
    purged = {f"d2024-02-2{x}" for x in (5, 6, 7, 8, 9)}
    check("purge gap excluded from all", not (purged & (train | val | test)))
    check("val window correct", "d2024-03-01" in val and "d2024-03-16" not in val)
    check("test after val_end", "d2024-03-16" in test)

    folds = splits.walk_forward_splits(
        races, first_test_start="2024-03-01", test_span_days=10, purge_days=3, n_folds=2
    )
    check("two folds produced", len(folds) == 2)
    f0 = folds[0]
    train0_dates = [r[1:] for r in f0["train"]]  # strip leading 'd'
    check("fold0 train before purge cut", max(train0_dates) < "2024-02-27")
    check("fold0 test within window", all("2024-03-01" <= r[1:] < "2024-03-11" for r in f0["test"]))


def main() -> int:
    test_arithmetic()
    test_significance()
    test_murphy()
    test_splits()

    print("=== Evaluation harness test results ===")
    ok = True
    for name, cond in checks:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond
    print("\nALL PASSED" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
