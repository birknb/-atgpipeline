"""Market benchmark for win probabilities.

This is the number every model must beat. Two market signals are turned into
win probabilities over the runners in each race:

  odds      de-vigged final win odds. raw_i = 1 / final_odds_i, normalised
            across the runners that started. In a pari-mutuel win pool this
            equals each horse's share of the pool, that is the published win
            percentage, so it has full coverage from race results alone.
  vinnare   the win pool bet distribution, when it has been ingested. This is
            the same quantity measured directly rather than through the odds.

Both are post-race quantities. They are the benchmark and are never used as
model features.

For each signal the script reports multinomial log loss and Brier score over a
clean set of races, and writes a calibration plot. A race is included only if
it has exactly one winner and every runner that started has a usable value.
Excluded races are counted and reported.

Usage:
    python -m atg.benchmark --db data/atg.sqlite
    python -m atg.benchmark --db data/atg.sqlite --country SE
"""
from __future__ import annotations

import argparse
import os
import sqlite3

import matplotlib

matplotlib.use("Agg")  # No display. Write plots to files.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import metrics  # noqa: E402


def load_runners(
    conn: sqlite3.Connection, country: str | None, sport: str | None
) -> pd.DataFrame:
    """Runners that started, joined to their race, optionally filtered."""
    sql = """
        SELECT s.race_id, s.number, s.horse_id, s.final_odds, s.is_winner,
               r.country, r.sport, r.date, r.n_starters
        FROM norm_starts s
        JOIN norm_races r ON r.race_id = s.race_id
        WHERE s.scratched = 0
    """
    df = pd.read_sql_query(sql, conn)
    if country:
        df = df[df["country"] == country].copy()
    if sport:
        df = df[df["sport"] == sport].copy()
    return df


def _clean_races(df: pd.DataFrame, value_col: str) -> tuple[pd.DataFrame, dict]:
    """Keep races with exactly one winner and a positive value for every runner.

    Returns the kept rows and a dict of exclusion counts.
    """
    info: dict = {}
    n_races_all = df["race_id"].nunique()

    # Exactly one winner.
    wins = df.groupby("race_id")["is_winner"].sum()
    one_winner = set(wins[wins == 1].index)
    info["races_excluded_winner"] = n_races_all - len(one_winner)

    # Every runner has a usable positive value.
    usable = df[value_col].notna() & (df[value_col] > 0)
    bad_races = set(df.loc[~usable, "race_id"])
    keep = set(one_winner) - bad_races
    info["races_excluded_value"] = len(one_winner) - len(keep)

    out = df[df["race_id"].isin(keep)].copy()
    info["races_kept"] = out["race_id"].nunique()
    info["runners_kept"] = len(out)
    return out, info


def market_frame_from_odds(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out, info = _clean_races(df, "final_odds")
    out["raw"] = 1.0 / out["final_odds"]
    out["p"] = metrics.normalize_within_group(out, "raw", "race_id")
    out["y"] = out["is_winner"].astype(int)
    # Median overround is a sanity check on the takeout. A trot win pool sits
    # near 1.15 to 1.25, which is a takeout of roughly 15 to 20 percent.
    overround = out.groupby("race_id")["raw"].sum()
    info["overround_median"] = float(overround.median()) if len(overround) else float("nan")
    return out[["race_id", "number", "p", "y"]], info


def market_frame_from_betdist(
    df: pd.DataFrame, conn: sqlite3.Connection
) -> tuple[pd.DataFrame, dict]:
    """Win probabilities from the marking-bet distribution (the V game spelprocent).

    This signal exists only for races that are legs in a V game, so coverage is
    partial. It reflects how V game players spread their bets across a leg,
    which is related to but not the same as a pure win probability. When a race
    is a leg in more than one V game, the most liquid pool is used.
    """
    bd = pd.read_sql_query(
        "SELECT race_id, number, share, game_id, pool_turnover FROM norm_bet_distribution",
        conn,
    )
    if bd.empty:
        return pd.DataFrame(columns=["race_id", "number", "p", "y"]), {"races_kept": 0}

    # Pick one game per race: the pool with the highest turnover.
    rank = (
        bd.groupby(["race_id", "game_id"])["pool_turnover"].max().reset_index()
        .sort_values("pool_turnover", ascending=False, na_position="last")
        .drop_duplicates("race_id")[["race_id", "game_id"]]
    )
    bd = bd.merge(rank, on=["race_id", "game_id"])

    merged = df.merge(bd[["race_id", "number", "share"]], on=["race_id", "number"], how="left")
    out, info = _clean_races(merged, "share")
    out["p"] = metrics.normalize_within_group(out, "share", "race_id")
    out["y"] = out["is_winner"].astype(int)
    return out[["race_id", "number", "p", "y"]], info


def evaluate(frame: pd.DataFrame, label: str) -> dict | None:
    if frame.empty:
        print(f"\n[{label}] no races available")
        return None
    ll = metrics.log_loss(frame)
    br = metrics.brier(frame)
    n_races = frame["race_id"].nunique()
    print(f"\n[{label}]")
    print(f"  races      {n_races:>8,}")
    print(f"  log loss   {ll:>8.4f}")
    print(f"  Brier      {br:>8.4f}")
    return {
        "label": label,
        "log_loss": ll,
        "brier": br,
        "n_races": n_races,
        "per_race_ll": metrics.per_race_log_loss(frame),
    }


def plot_calibration(frames: dict[str, pd.DataFrame], out_path: str, n_bins: int) -> None:
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="perfect")
    for label, frame in frames.items():
        if frame.empty:
            continue
        tab = metrics.calibration_table(frame, n_bins=n_bins, strategy="quantile")
        plt.plot(tab["p_mean"], tab["y_rate"], marker="o", label=label)
    plt.xlabel("mean predicted win probability")
    plt.ylabel("observed win frequency")
    plt.title("Market calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/atg.sqlite")
    parser.add_argument("--country", default=None, help="optional country filter, e.g. SE")
    parser.add_argument("--sport", default=None, help="optional sport filter, e.g. trot")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--out", default="results", help="directory for plots")
    args = parser.parse_args(argv)

    conn = sqlite3.connect(args.db)
    runners = load_runners(conn, args.country, args.sport)
    if runners.empty:
        print("No runners found. Run normalisation first, or check the filters.")
        conn.close()
        return 1

    print("=== Market benchmark ===")
    scope = ", ".join(x for x in (args.country, args.sport) if x) or "all races"
    print(f"scope: {scope}, {runners['race_id'].nunique():,} races before cleaning")
    by_sport = runners.groupby("sport")["race_id"].nunique().to_dict()
    print(f"races by sport: {by_sport}")

    odds_frame, odds_info = market_frame_from_odds(runners)
    bd_frame, bd_info = market_frame_from_betdist(runners, conn)

    print("\n--- race cleaning (odds) ---")
    print(f"  kept                  {odds_info['races_kept']:>8,}")
    print(f"  excluded, winner      {odds_info['races_excluded_winner']:>8,}")
    print(f"  excluded, missing odds{odds_info['races_excluded_value']:>8,}")
    print(f"  median overround      {odds_info['overround_median']:>8.3f}")

    odds_res = evaluate(odds_frame, "market: de-vigged odds")
    bd_res = evaluate(bd_frame, "market: V-game spelprocent")

    # If both signals exist, compare them on their common races.
    if odds_res and bd_res:
        boot = metrics.paired_bootstrap_logloss_diff(
            odds_res["per_race_ll"], bd_res["per_race_ll"]
        )
        print("\n[odds minus spelprocent] paired bootstrap of log loss")
        print(f"  common races {boot['n_races']:,}")
        print(f"  estimate {boot['estimate']:+.4f}  95% CI [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]")

    os.makedirs(args.out, exist_ok=True)
    plot_path = os.path.join(args.out, "calibration_market.png")
    plot_calibration(
        {"de-vigged odds": odds_frame, "V-game spelprocent": bd_frame},
        plot_path,
        args.bins,
    )
    print(f"\nCalibration plot written to {plot_path}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
