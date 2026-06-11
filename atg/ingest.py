"""Incremental ingestion of historical ATG racing data.

Usage:
    python -m atg.ingest --from 2025-06-01 --to 2025-06-07
    python -m atg.ingest --from 2025-06-01 --to 2025-06-07 --db data/atg.sqlite

Behaviour:
  - Walks the date range day by day.
  - Skips days already marked complete in ingest_log. The run is idempotent
    and can be interrupted with Ctrl-C and resumed later.
  - For each day: fetches the calendar, then every finished trot race
    (status == 'results'), then every pool game of the configured types.
  - A day is only marked complete if every fetch succeeded. Partial days
    are retried on the next run.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta

from .client import AtgClient
from .db import Db

log = logging.getLogger("atg.ingest")

# Pool game types to store. These carry the bet distribution (spelprocent),
# the market data the models are benchmarked against.
GAME_TYPES = {"V75", "V86", "V64", "V65", "GS75", "V5", "V4", "V3", "dd", "ld"}

# Only trot races are modelled. Gallop days are skipped.
SPORTS = {"trot"}


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def ingest_day(client: AtgClient, db: Db, day: str) -> tuple[int, int, int]:
    """Fetch one day. Returns (n_races, n_games, n_failures)."""
    cal = client.calendar_day(day)
    if cal is None:
        log.info("%s: no calendar (404)", day)
        db.mark_day_done(day, 0, 0)
        return 0, 0, 0
    db.upsert_calendar_day(day, cal)

    n_races = n_games = n_fail = 0

    for track in cal.get("tracks", []):
        if track.get("sport") not in SPORTS:
            continue
        for race_stub in track.get("races", []):
            if race_stub.get("status") != "results":
                # The race is not finished or was cancelled.
                continue
            race_id = race_stub["id"]
            if db.has_race(race_id):
                n_races += 1
                continue
            try:
                payload = client.race(race_id)
            except RuntimeError as exc:
                log.error("%s: race %s failed: %s", day, race_id, exc)
                n_fail += 1
                continue
            if payload is not None:
                db.upsert_race(payload)
                n_races += 1

    for game_type, games in (cal.get("games") or {}).items():
        if game_type not in GAME_TYPES:
            continue
        for game_stub in games:
            game_id = game_stub["id"]
            try:
                payload = client.game(game_id)
            except RuntimeError as exc:
                log.error("%s: game %s failed: %s", day, game_id, exc)
                n_fail += 1
                continue
            if payload is not None and payload.get("status") == "results":
                db.upsert_game(payload)
                n_games += 1

    db.commit()
    if n_fail == 0:
        db.mark_day_done(day, n_races, n_games)
    return n_races, n_games, n_fail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD")
    parser.add_argument("--db", default="data/atg.sqlite", help="SQLite file path")
    parser.add_argument("--delay", type=float, default=0.4, help="seconds between requests")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    start = date.fromisoformat(args.date_from)
    end = date.fromisoformat(args.date_to)
    if end < start:
        parser.error("--to must be on or after --from")

    from pathlib import Path

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    db = Db(args.db)
    client = AtgClient(delay_s=args.delay)

    total_races = total_games = 0
    try:
        for d in daterange(start, end):
            day = d.isoformat()
            if db.day_is_done(day):
                log.info("%s: already complete, skipping", day)
                continue
            n_races, n_games, n_fail = ingest_day(client, db, day)
            total_races += n_races
            total_games += n_games
            log.info(
                "%s: %d races, %d games stored%s",
                day, n_races, n_games,
                f" ({n_fail} failures, day left incomplete)" if n_fail else "",
            )
    except KeyboardInterrupt:
        log.warning("Interrupted. Progress is saved. Re-run to resume.")
    finally:
        db.close()

    log.info(
        "Done. %d races and %d games stored, %d HTTP requests made.",
        total_races, total_games, client.request_count,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
