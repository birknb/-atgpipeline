"""Offline checks for the modelling maths in atg/model.py.

Synthetic, no database. Verifies the grouped softmax, that the conditional logit
recovers a known signal, that the favourite-longshot power fit sharpens an
under-confident market, and that per-race renormalisation sums to one. Run:

    python tests/test_model.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atg import metrics, model  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, cond: bool) -> None:
    checks.append((name, bool(cond)))


def test_grouped_softmax() -> None:
    v = np.array([1.0, 2.0, 0.5, 1.5, -1.0])
    codes = np.array([0, 0, 1, 1, 1], dtype=np.intp)
    p = model.grouped_softmax(v, codes, 2)
    check("softmax group 0 sums to 1", abs(p[codes == 0].sum() - 1.0) < 1e-9)
    check("softmax group 1 sums to 1", abs(p[codes == 1].sum() - 1.0) < 1e-9)
    check("softmax orders by score", p[1] > p[0])


def test_conditional_logit_recovers_signal() -> None:
    rng = np.random.default_rng(0)
    n_races, n = 800, 6
    rows = []
    for r in range(n_races):
        x = rng.normal(size=n)
        # True model is a logit with coefficient 2 on x, sampled via Gumbel.
        u = 2.0 * x + rng.gumbel(size=n)
        winner = int(np.argmax(u))
        for i in range(n):
            rows.append({"race_id": r, "x": x[i], "y": 1 if i == winner else 0})
    df = pd.DataFrame(rows)
    xc = df["x"] - df.groupby("race_id")["x"].transform("mean")
    X = xc.to_numpy().reshape(-1, 1)
    codes, g = model._codes(df["race_id"])
    beta = model.fit_conditional_logit(X, df["y"].to_numpy(), codes, g, lam=1.0)
    check("clogit recovers positive coefficient", beta[0] > 1.0)
    p = model.grouped_softmax(X[:, 0] * beta[0], codes, g)
    df["p"] = p
    # The highest-x runner should usually carry the most probability in a race.
    top_is_max_x = df.loc[df.groupby("race_id")["p"].idxmax()].reset_index(drop=True)
    max_x = df.loc[df.groupby("race_id")["x"].idxmax()].reset_index(drop=True)
    agree = (top_is_max_x["race_id"].values == max_x["race_id"].values).all()
    check("clogit ranks highest-x runner top", bool((top_is_max_x["x"].values == max_x["x"].values).all()) and agree)


def test_flb_power_sharpens_underconfident_market() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for r in range(1500):
        a = rng.normal(size=8)
        true_p = np.exp(a) / np.exp(a).sum()
        winner = rng.choice(8, p=true_p)
        # Market is under-confident: it uses 0.6*a, so favourites are underbet.
        mp = np.exp(0.6 * a) / np.exp(0.6 * a).sum()
        for i in range(8):
            rows.append({"race_id": r, "p": mp[i], "y": 1 if i == winner else 0})
    market = pd.DataFrame(rows)
    a_hat = model.fit_flb_power(market)
    check("flb power above one for underconfident market", a_hat > 1.0)
    l0 = metrics.log_loss(market)
    l1 = metrics.log_loss(model.apply_power(market, a_hat))
    check("flb recalibration lowers log loss", l1 < l0)


def test_renormalise() -> None:
    df = pd.DataFrame({"race_id": [0, 0, 1, 1, 1]})
    raw = np.array([2.0, 2.0, 1.0, 1.0, 2.0])
    p = model.renormalise(df, raw)
    check("renormalise group 0 sums to 1", abs(p[:2].sum() - 1.0) < 1e-9)
    check("renormalise group 1 sums to 1", abs(p[2:].sum() - 1.0) < 1e-9)


def main() -> int:
    test_grouped_softmax()
    test_conditional_logit_recovers_signal()
    test_flb_power_sharpens_underconfident_market()
    test_renormalise()

    print("=== Model maths test results ===")
    ok = True
    for name, cond in checks:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond
    print("\nALL PASSED" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
