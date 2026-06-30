"""Score stored prediction snapshots against the realised outcomes (Phase 4).

The prediction core (predict.py) stores a timestamped snapshot of the win
probabilities before each race. After the races run, the ordinary ingestion
records the results. This module joins the snapshots to the realised winners,
taken from norm_starts rather than the snapshot, and runs the evaluation harness:
the combination against the raw market and against the recalibrated market.

This is how the small walk-forward edge is confirmed on fresh races, with no
look-ahead, since every probability was committed before the off.

Usage:
    python -m atg.score --db data/atg.sqlite
"""
from __future__ import annotations

import argparse
import logging
import sqlite3

import pandas as pd

from . import evaluate

log = logging.getLogger("atg.score")


def score(pred: pd.DataFrame, outcomes: pd.DataFrame) -> dict:
    """pred has start_id, race_id, number, date and the probability columns.
    outcomes has start_id and the authoritative is_winner. Returns the harness
    results for the combination against both market forms."""
    df = pred.merge(outcomes[["start_id", "is_winner"]], on="start_id", how="inner",
                    suffixes=("", "_actual"))
    df["y"] = df["is_winner_actual"] if "is_winner_actual" in df else df["is_winner"]

    # Keep races with exactly one realised winner, so the scoring is well posed.
    wins = df.groupby("race_id")["y"].sum()
    keep = set(wins[wins == 1].index)
    df = df[df["race_id"].isin(keep)].copy()
    if df.empty:
        raise ValueError("no scorable races (need exactly one winner per race)")

    rd = evaluate.race_dates_from_frame(df)

    def frame(prob_col):
        out = df[["race_id", "number", "date"]].copy()
        out["p"] = df[prob_col].to_numpy()
        out["y"] = df["y"].to_numpy()
        return out

    market = frame("p_market")
    market_flb = frame("p_market_flb")
    combo = frame("p_combination")
    return {
        "combo_vs_market": evaluate.compare(combo, market, rd, label="combo vs market"),
        "combo_vs_market_flb": evaluate.compare(combo, market_flb, rd, label="combo vs market_flb"),
        "flb_vs_market": evaluate.compare(market_flb, market, rd, label="market_flb vs market"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/atg.sqlite")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    conn = sqlite3.connect(args.db)
    pred = pd.read_sql_query("SELECT * FROM predictions", conn)
    outcomes = pd.read_sql_query("SELECT start_id, is_winner FROM norm_starts", conn)
    conn.close()

    results = score(pred, outcomes)
    print(f"=== scoring {pred['race_id'].nunique():,} predicted races against outcomes ===")
    for res in results.values():
        evaluate.print_report(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
