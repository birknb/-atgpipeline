"""Sanity checks on the contents of the database.

Usage:
    python -m atg.explore --db data/atg.sqlite
"""
from __future__ import annotations

import argparse
import json
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/atg.sqlite")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    q = conn.execute

    print("=== Coverage ===")
    for table in ("raw_calendar_days", "raw_races", "raw_games", "ingest_log"):
        (n,) = q(f"SELECT COUNT(*) FROM {table}").fetchone()
        print(f"{table:20s} {n:>8,} rows")

    row = q("SELECT MIN(date), MAX(date) FROM raw_races").fetchone()
    print(f"\nRace date range: {row[0]} -> {row[1]}")

    print("\n=== Races per country ===")
    for country, n in q(
        "SELECT country, COUNT(*) FROM raw_races GROUP BY country ORDER BY 2 DESC"
    ):
        print(f"  {country}: {n:,}")

    print("\n=== Games per type ===")
    for gtype, n in q(
        "SELECT game_type, COUNT(*) FROM raw_games GROUP BY game_type ORDER BY 2 DESC"
    ):
        print(f"  {gtype}: {n:,}")

    # Print one race to confirm that the fields needed for modelling are present.
    row = q("SELECT json FROM raw_races LIMIT 1").fetchone()
    if row:
        race = json.loads(row[0])
        starts = race.get("starts", [])
        print(f"\n=== Sample race {race['id']} ({race['track']['name']}) ===")
        print(f"distance={race.get('distance')} startMethod={race.get('startMethod')} "
              f"starts={len(starts)}")
        for s in starts[:3]:
            h, res = s["horse"], s.get("result", {})
            km = res.get("kmTime") or {}
            kmtime = (
                f"{km.get('minutes')}:{km.get('seconds'):02d},{km.get('tenths')}"
                if km.get("seconds") is not None else "-"
            )
            print(
                f"  #{s['number']:>2} {h['name']:<22} spor={s.get('postPosition')} "
                f"plass={res.get('place')} kmtid={kmtime} odds={res.get('finalOdds')}"
            )

    conn.close()


if __name__ == "__main__":
    main()
