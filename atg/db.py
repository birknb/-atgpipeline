"""SQLite storage for raw ATG API responses.

The complete, untouched JSON response is stored for every entity, keyed by
its natural id. Normalisation into analytical tables happens in a later
phase and can always be re-run from the raw data. A parsing bug therefore
never forces a re-download.

Tables:
  raw_calendar_days  one row per calendar day fetched
  raw_races          one row per race (the core dataset)
  raw_games          one row per pool game (V75, V86, ...) with bet distribution
  ingest_log         one row per completed day, which makes ingestion idempotent
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_calendar_days (
    date        TEXT PRIMARY KEY,           -- YYYY-MM-DD
    json        TEXT NOT NULL,
    fetched_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_races (
    race_id     TEXT PRIMARY KEY,           -- e.g. 2026-06-10_23_5
    date        TEXT NOT NULL,
    track_id    INTEGER,
    track_name  TEXT,
    sport       TEXT,                       -- trot / gallop
    country     TEXT,                       -- SE / DK / NO ...
    status      TEXT,                       -- results / cancelled / ...
    json        TEXT NOT NULL,
    fetched_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_races_date ON raw_races(date);

CREATE TABLE IF NOT EXISTS raw_games (
    game_id     TEXT PRIMARY KEY,           -- e.g. V86_2026-06-10_23_3
    game_type   TEXT,                       -- V75 / V86 / V64 / ...
    date        TEXT,
    status      TEXT,
    json        TEXT NOT NULL,
    fetched_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_games_date ON raw_games(date);

CREATE TABLE IF NOT EXISTS ingest_log (
    date          TEXT PRIMARY KEY,
    completed_at  TEXT NOT NULL,
    n_races       INTEGER NOT NULL,
    n_games       INTEGER NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Db:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------- writes
    def upsert_calendar_day(self, date: str, payload: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO raw_calendar_days(date, json, fetched_at) VALUES (?,?,?)",
            (date, json.dumps(payload, ensure_ascii=False), _now()),
        )

    def upsert_race(self, payload: dict) -> None:
        track = payload.get("track") or {}
        self.conn.execute(
            """INSERT OR REPLACE INTO raw_races
               (race_id, date, track_id, track_name, sport, country, status, json, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                payload["id"],
                payload.get("date"),
                track.get("id"),
                track.get("name"),
                payload.get("sport"),
                track.get("countryCode"),
                payload.get("status"),
                json.dumps(payload, ensure_ascii=False),
                _now(),
            ),
        )

    def upsert_game(self, payload: dict) -> None:
        game_id = payload["id"]
        game_type = game_id.split("_", 1)[0]
        date = game_id.split("_")[1] if "_" in game_id else None
        self.conn.execute(
            """INSERT OR REPLACE INTO raw_games
               (game_id, game_type, date, status, json, fetched_at)
               VALUES (?,?,?,?,?,?)""",
            (
                game_id,
                game_type,
                date,
                payload.get("status"),
                json.dumps(payload, ensure_ascii=False),
                _now(),
            ),
        )

    def mark_day_done(self, date: str, n_races: int, n_games: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO ingest_log(date, completed_at, n_races, n_games) VALUES (?,?,?,?)",
            (date, _now(), n_races, n_games),
        )
        self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()

    # -------------------------------------------------------------- reads
    def day_is_done(self, date: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM ingest_log WHERE date = ?", (date,)
        ).fetchone()
        return row is not None

    def has_race(self, race_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM raw_races WHERE race_id = ?", (race_id,)
        ).fetchone()
        return row is not None

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
