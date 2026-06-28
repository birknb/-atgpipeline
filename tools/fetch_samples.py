"""Fetch a small sample of raw ATG JSON for offline inspection.

Run this on a machine where atg.se is reachable. It writes raw, unmodified
JSON responses to data/samples/ so the analytical parser can be checked against
real field names without a full backfill. The data source is ATG's public
racing info API, the same API that powers atg.se. It is free and needs no
account or key.

The files this writes are tiny. Zip the folder and copy it back to the
development machine, or paste a single race file and a single game file into
the conversation. That is enough to confirm the parser field names.

Usage:
    python -m tools.fetch_samples --date 2025-06-07
    python tools/fetch_samples.py --date 2025-06-07
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atg.client import AtgClient  # noqa: E402


def save(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print("wrote", path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    # A Saturday is a good default because the large V games run then, which
    # gives the win pool and several pool types in one day.
    p.add_argument("--date", default="2025-06-07", help="YYYY-MM-DD")
    p.add_argument("--out", default="data/samples")
    p.add_argument("--n-races", type=int, default=3)
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    client = AtgClient()

    cal = client.calendar_day(args.date)
    if cal is None:
        print(f"No calendar for {args.date}. Try another date.")
        return 1
    save(cal, os.path.join(args.out, f"calendar_{args.date}.json"))

    # A few finished trot races. Prefer the first results races found.
    saved = 0
    for track in cal.get("tracks", []):
        if track.get("sport") != "trot":
            continue
        for stub in track.get("races", []):
            if stub.get("status") != "results":
                continue
            race = client.race(stub["id"])
            if race:
                save(race, os.path.join(args.out, f"race_{stub['id']}.json"))
                saved += 1
            if saved >= args.n_races:
                break
        if saved >= args.n_races:
            break

    # One game of each available type, so the bet distribution layout is visible.
    # vinnare is the win pool, the most important one for this project.
    games = cal.get("games") or {}
    for gtype in ("vinnare", "plats", "V75", "V86", "V64", "V65"):
        stubs = games.get(gtype)
        if not stubs:
            continue
        game = client.game(stubs[0]["id"])
        if game:
            save(game, os.path.join(args.out, f"game_{stubs[0]['id']}.json"))

    print(f"\nDone. {client.request_count} requests made. Files are in {args.out}")
    print("Zip that folder and copy it into data/samples/ on the repo machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
