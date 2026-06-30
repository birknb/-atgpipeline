"""Inspect a live, pre-race ATG payload to settle the Phase 4 unknowns.

Run this on a machine that can reach atg.se, during a race day before the races
have run, so it can find a race that has not finished yet:

    python -m tools.inspect_live
    python -m tools.inspect_live --date 2026-07-01

It fetches the calendar, picks a Scandinavian trot race that is not finished,
fetches that race and the day's win pool, saves both as JSON next to the script,
and prints what the pre-race payload contains. The three questions it answers,
which decide how the live logger is written:

  1. What status does an upcoming race have, and can its card be fetched before
     the off?
  2. Where do the live win odds sit before the off? The de-vigged market needs a
     current odds figure, not the post-race finalOdds.
  3. Does the pre-race card carry the as-of-race statistics and record blocks the
     features rely on?

Paste the printed summary back into the Claude Code session on the work laptop to
continue. Only requests is needed, the same as for ingestion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atg.client import AtgClient  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _save(name: str, payload) -> str:
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--countries", default="SE,DK,NO")
    args = parser.parse_args(argv)
    countries = {c.strip().upper() for c in args.countries.split(",")}

    client = AtgClient()
    cal = client.calendar_day(args.date)
    if not cal:
        print(f"No calendar for {args.date}. Is atg.se reachable, and is it a race day?")
        return 1

    # Prefer a trot race that has not finished. Fall back to any trot race.
    upcoming = None
    fallback = None
    for track in cal.get("tracks", []):
        if track.get("sport") != "trot":
            continue
        country = track.get("countryCode") or track.get("country")
        if country not in countries:
            continue
        for rs in track.get("races", []):
            fallback = fallback or rs
            if rs.get("status") != "results":
                upcoming = rs
                break
        if upcoming:
            break

    target = upcoming or fallback
    if target is None:
        print(f"No Scandinavian trot races found on {args.date}.")
        return 1
    if upcoming is None:
        print("WARNING: every race has already finished. Re-run earlier on a race")
        print("day to see the live odds before the off. Showing a finished race.\n")

    race = client.race(target["id"])
    rp = _save("inspect_race.json", race)
    starts = race.get("starts", [])
    print(f"=== race {race.get('id')} ===")
    print(f"status            {race.get('status')}")
    print(f"scheduledStartTime {race.get('scheduledStartTime')}")
    print(f"starts            {len(starts)}")
    if starts:
        s0 = starts[0]
        horse = s0.get("horse") or {}
        print(f"start keys        {sorted(s0.keys())}")
        print(f"has horse.statistics {'statistics' in horse}, has horse.record {'record' in horse}")
        print(f"start has pools   {'pools' in s0}")
        if "pools" in s0:
            print(f"start.pools       {json.dumps(s0.get('pools'), ensure_ascii=False)[:400]}")
    print(f"saved {rp}")

    # The win pool, where live odds should sit before the off.
    games = cal.get("games") or {}
    vinnare = games.get("vinnare")
    if vinnare:
        gid = vinnare[0]["id"]
        game = client.game(gid)
        gp = _save("inspect_winpool.json", game)
        legs = game.get("races") or []
        if legs and (legs[0].get("starts") or []):
            st = legs[0]["starts"][0]
            pools = st.get("pools") or {}
            print(f"\n=== win pool {gid} ===")
            print(f"game status       {game.get('status')}")
            print(f"first start pools {json.dumps(pools, ensure_ascii=False)[:400]}")
        print(f"saved {gp}")
    else:
        print("\nNo 'vinnare' win pool in the calendar games for this day.")

    print("\nPaste this summary back to the work laptop. The two JSON files next to")
    print("this script can be transferred too if a closer look is needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
