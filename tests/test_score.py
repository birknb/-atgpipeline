"""Offline checks for the snapshot scorer (atg/score.py).

Builds synthetic prediction snapshots and outcomes, where the combination is a
sharper forecast than the market, and checks the scorer joins to the realised
winners and recovers a positive skill. Run:

    python tests/test_score.py
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atg import score  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, cond: bool) -> None:
    checks.append((name, bool(cond)))


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    start = date(2026, 1, 1)
    pred_rows, out_rows = [], []
    for r in range(600):
        n = 8
        a = rng.normal(size=n)
        true_p = np.exp(a) / np.exp(a).sum()
        winner = rng.choice(n, p=true_p)
        # Market is under-confident, combination is sharper (closer to truth).
        mp = np.exp(0.6 * a) / np.exp(0.6 * a).sum()
        cp = np.exp(0.9 * a) / np.exp(0.9 * a).sum()
        fp = np.exp(0.9 * a) / np.exp(0.9 * a).sum()
        d = (start + timedelta(days=r // 5)).isoformat()
        for i in range(n):
            sid = f"{r}_{i}"
            pred_rows.append({
                "start_id": sid, "race_id": str(r), "number": i + 1, "date": d,
                "p_market": mp[i], "p_market_flb": mp[i], "p_fundamental": fp[i],
                "p_combination": cp[i],
            })
            out_rows.append({"start_id": sid, "is_winner": 1 if i == winner else 0})
    return pd.DataFrame(pred_rows), pd.DataFrame(out_rows)


def main() -> int:
    pred, outcomes = build()
    results = score.score(pred, outcomes)

    cvm = results["combo_vs_market"]
    check("scores all races", cvm["n_races"] == 600)
    check("combination beats the under-confident market", cvm["skill_score"] > 0)
    check("edge is significant", cvm["significant"] is True)
    check("three comparisons returned", len(results) == 3)

    raised = False
    try:
        score.score(pred.head(0), outcomes)
    except (ValueError, KeyError):
        raised = True
    check("raises on empty input", raised)

    print("=== Scorer test results ===")
    ok = True
    for name, cond in checks:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond
    print("\nALL PASSED" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
