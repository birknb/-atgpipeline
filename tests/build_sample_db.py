"""Build a small SQLite database from the JSON files in data/samples/.

This lets normalisation and the benchmark be validated against real payloads
without a full backfill. Standalone race files and the race objects embedded
in game files are both loaded into raw_races, so the V game legs are present
and the bet distribution can join to them.

Run:
    python tests/build_sample_db.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atg.db import Db  # noqa: E402


def build(samples_dir: str = "data/samples", out: str = "data/sample.sqlite") -> str:
    if os.path.exists(out):
        os.remove(out)
    db = Db(out)
    races: set[str] = set()
    n_games = 0

    for f in sorted(glob.glob(os.path.join(samples_dir, "race_*.json"))):
        p = json.load(open(f, encoding="utf-8"))
        db.upsert_race(p)
        races.add(p["id"])

    for f in sorted(glob.glob(os.path.join(samples_dir, "game_*.json"))):
        p = json.load(open(f, encoding="utf-8"))
        db.upsert_game(p)
        n_games += 1
        # The game payload embeds the full race object for each leg. Load these
        # too, so the marking-bet legs exist as races.
        for leg in p.get("races", []):
            if leg.get("id") and leg.get("starts"):
                db.upsert_race(leg)
                races.add(leg["id"])

    db.close()
    print(f"built {out}: {len(races)} races, {n_games} games")
    return out


if __name__ == "__main__":
    build()
