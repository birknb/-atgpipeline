"""Normalise raw ATG JSON into analytical tables.

This step reads only the raw_* tables and writes derived tables back into the
same SQLite file. It never touches the network. The raw JSON is the source of
truth, so this step can be re-run at any time and a parsing change never
requires a re-download. Each run drops and rebuilds the derived tables, which
makes it idempotent.

Derived tables:
  norm_races            one row per race
  norm_starts           one row per horse per race, the core analytical unit
  norm_bet_distribution one row per horse per marking-bet leg (V75, V64, ...)

Field names were confirmed against real API payloads in June 2026. The notable
points, which differ from the first guess:

  - Driver and trainer have firstName and lastName, not name.
  - A non-finisher has result.galloped or result.disqualified set to true and
    no place key at all. place is therefore absent, not zero.
  - A withdrawn horse has scratched set to true and finalOdds 0.0. It did not
    run and is excluded from the win simplex by the benchmark.
  - shoes is nested as front and back, each with hasShoe and changed. sulky
    carries reported and, when present, changed.
  - The win odds per horse are in result.finalOdds (a plain decimal). The
    marking-bet bet distribution (the V game spelprocent) is the only
    betDistribution field and sits at start.pools.<betType>.betDistribution,
    in hundredths of a percent. The win pool itself has odds, not a
    distribution, so de-vigged finalOdds is the win-market signal.
  - sport is per race and is trot or monte. Both are stored.

Usage:
    python -m atg.normalize --db data/atg.sqlite
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from collections import Counter

log = logging.getLogger("atg.normalize")

NORM_SCHEMA = """
DROP TABLE IF EXISTS norm_races;
DROP TABLE IF EXISTS norm_starts;
DROP TABLE IF EXISTS norm_bet_distribution;

CREATE TABLE norm_races (
    race_id              TEXT PRIMARY KEY,
    date                 TEXT,
    start_time           TEXT,    -- actual off time
    scheduled_start_time TEXT,    -- the point-in-time field for ordering
    track_id             INTEGER,
    track_name           TEXT,
    track_condition      TEXT,
    country              TEXT,
    race_no              INTEGER,
    distance_m           INTEGER,
    start_method         TEXT,    -- auto or volte
    sport                TEXT,    -- trot or monte
    status               TEXT,
    n_starters           INTEGER  -- runners that were not scratched
);

CREATE TABLE norm_starts (
    start_id           TEXT PRIMARY KEY,
    race_id            TEXT NOT NULL,
    number             INTEGER,
    post_position      INTEGER,
    start_distance_m   INTEGER,
    horse_id           INTEGER,
    horse_name         TEXT,
    age                INTEGER,
    sex                TEXT,
    career_money       INTEGER,   -- horse.money. May be current, not as-of-race.
    stat_life_starts   INTEGER,   -- statistics.life.starts, as-of-race (verified)
    stat_win_pct       REAL,      -- as-of-race career win fraction
    stat_place_pct     REAL,      -- as-of-race career place fraction
    stat_earn_per_start INTEGER,  -- statistics.life.earningsPerStart
    stat_start_points  INTEGER,   -- statistics.life.startPoints
    best_km_time_s     REAL,      -- horse.record best time, as-of-race (verified)
    shoe_front_on      INTEGER,
    shoe_back_on       INTEGER,
    shoe_front_changed INTEGER,
    shoe_back_changed  INTEGER,
    shoes_changed      INTEGER,   -- front or back changed
    sulky_changed      INTEGER,
    driver_id          INTEGER,
    driver_name        TEXT,
    trainer_id         INTEGER,
    trainer_name       TEXT,
    scratched          INTEGER,
    -- The columns below are post-race outcomes. They are labels and benchmark
    -- inputs only. They must never be used as model features.
    place              INTEGER,   -- absent for non-finishers, stored as NULL
    finish_order       INTEGER,
    km_time_s          REAL,
    prize_money        INTEGER,
    final_odds         REAL,
    is_winner          INTEGER,
    galloped           INTEGER,
    disqualified       INTEGER,
    galloped_or_dq     INTEGER
);
CREATE INDEX idx_norm_starts_race ON norm_starts(race_id);
CREATE INDEX idx_norm_starts_horse ON norm_starts(horse_id);

CREATE TABLE norm_bet_distribution (
    game_id       TEXT,
    bet_type      TEXT,           -- V75, V64, V65, V86, ...
    date          TEXT,
    race_id       TEXT,
    number        INTEGER,
    horse_id      INTEGER,
    share         REAL,           -- fraction of the pool, betDistribution / 10000
    pool_turnover INTEGER,
    PRIMARY KEY (game_id, race_id, number)
);
CREATE INDEX idx_norm_betdist_race ON norm_bet_distribution(race_id);
"""


# ---------------------------------------------------------------- safe casts
def _int(v):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _bool01(v):
    """1 if truthy, 0 if the key was present and falsey, None if absent."""
    if v is None:
        return None
    return 1 if v else 0


def person_name(d: dict | None) -> str | None:
    """Driver and trainer names come as firstName and lastName."""
    if not isinstance(d, dict):
        return None
    if d.get("name"):
        return d["name"]
    first = (d.get("firstName") or "").strip()
    last = (d.get("lastName") or "").strip()
    full = f"{first} {last}".strip()
    return full or d.get("shortName")


def km_time_seconds(km: dict | None) -> float | None:
    """Convert a kmTime {minutes, seconds, tenths} to seconds.

    A non-finisher often has kmTime {code: ...} or no kmTime, in which case
    this returns None.
    """
    if not isinstance(km, dict):
        return None
    seconds = _int(km.get("seconds"))
    if seconds is None:
        return None
    minutes = _int(km.get("minutes")) or 0
    tenths = _int(km.get("tenths")) or 0
    return minutes * 60 + seconds + tenths / 10.0


def horse_stats(horse: dict) -> dict:
    """Extract the as-of-race statistics blocks. These were verified to be
    as-of-race and to exclude the current race, so they are point-in-time safe.
    Percentages are scaled by 10000 in the API, so 3076 means a 0.3076 fraction.
    The record best time was verified to exclude the current race as well."""
    life = (horse.get("statistics") or {}).get("life") or {}
    record_time = (horse.get("record") or {}).get("time")

    def pct(v):
        f = _float(v)
        return f / 10000.0 if f is not None else None

    return {
        "stat_life_starts": _int(life.get("starts")),
        "stat_win_pct": pct(life.get("winPercentage")),
        "stat_place_pct": pct(life.get("placePercentage")),
        "stat_earn_per_start": _int(life.get("earningsPerStart")),
        "stat_start_points": _int(life.get("startPoints")),
        "best_km_time_s": km_time_seconds(record_time),
    }


def shoe_fields(shoes: dict | None):
    """Return (front_on, back_on, front_changed, back_changed, any_changed)."""
    if not isinstance(shoes, dict):
        return (None, None, None, None, None)
    front = shoes.get("front") or {}
    back = shoes.get("back") or {}
    fon = _bool01(front.get("hasShoe")) if "hasShoe" in front else None
    bon = _bool01(back.get("hasShoe")) if "hasShoe" in back else None
    fch = _bool01(front.get("changed")) if "changed" in front else None
    bch = _bool01(back.get("changed")) if "changed" in back else None
    if fch is None and bch is None:
        any_changed = None
    else:
        any_changed = 1 if (fch or bch) else 0
    return (fon, bon, fch, bch, any_changed)


# ---------------------------------------------------------------- race level
def parse_race(payload: dict, report: Counter) -> tuple[dict, list[dict]]:
    race_id = payload["id"]
    track = payload.get("track") or {}
    starts_raw = payload.get("starts") or []

    race_no = _int(payload.get("number"))
    if race_no is None and "_" in race_id:
        race_no = _int(race_id.rsplit("_", 1)[-1])

    start_rows: list[dict] = []
    n_starters = 0
    for s in starts_raw:
        horse = s.get("horse") or {}
        driver = s.get("driver") or {}
        trainer = horse.get("trainer") or s.get("trainer") or {}
        result = s.get("result") or {}
        fon, bon, fch, bch, shoes_changed = shoe_fields(horse.get("shoes"))
        sulky = horse.get("sulky") or {}

        number = _int(s.get("number"))
        scratched = 1 if s.get("scratched") else 0
        if not scratched:
            n_starters += 1

        place = _int(result.get("place"))
        galloped = _bool01(result.get("galloped"))
        disqualified = _bool01(result.get("disqualified"))
        galloped_or_dq = 1 if (galloped or disqualified) else 0
        is_winner = 1 if place == 1 else 0

        if number is None:
            report["start_missing_number"] += 1
        if s.get("postPosition") is None:
            report["start_missing_postposition"] += 1

        start_rows.append(
            {
                "start_id": s.get("id") or f"{race_id}_{number}",
                "race_id": race_id,
                "number": number,
                "post_position": _int(s.get("postPosition")),
                "start_distance_m": _int(s.get("distance")),
                "horse_id": _int(horse.get("id")),
                "horse_name": horse.get("name"),
                "age": _int(horse.get("age")),
                "sex": horse.get("sex"),
                "career_money": _int(horse.get("money")),
                **horse_stats(horse),
                "shoe_front_on": fon,
                "shoe_back_on": bon,
                "shoe_front_changed": fch,
                "shoe_back_changed": bch,
                "shoes_changed": shoes_changed,
                "sulky_changed": _bool01(sulky.get("changed")) if "changed" in sulky else None,
                "driver_id": _int(driver.get("id")),
                "driver_name": person_name(driver),
                "trainer_id": _int(trainer.get("id")),
                "trainer_name": person_name(trainer),
                "scratched": scratched,
                "place": place,
                "finish_order": _int(result.get("finishOrder")),
                "km_time_s": km_time_seconds(result.get("kmTime")),
                "prize_money": _int(result.get("prizeMoney")),
                "final_odds": _float(result.get("finalOdds")),
                "is_winner": is_winner,
                "galloped": galloped,
                "disqualified": disqualified,
                "galloped_or_dq": galloped_or_dq,
            }
        )

    race_row = {
        "race_id": race_id,
        "date": payload.get("date"),
        "start_time": payload.get("startTime"),
        "scheduled_start_time": payload.get("scheduledStartTime"),
        "track_id": _int(track.get("id")),
        "track_name": track.get("name"),
        "track_condition": track.get("condition"),
        "country": track.get("countryCode"),
        "race_no": race_no,
        "distance_m": _int(payload.get("distance")),
        "start_method": payload.get("startMethod"),
        "sport": payload.get("sport"),
        "status": payload.get("status"),
        "n_starters": n_starters,
    }
    return race_row, start_rows


# ----------------------------------------------------------- bet distribution
def extract_bet_distribution(payload: dict, game_id: str, report: Counter):
    """Yield bet distribution records from a game payload.

    The bet distribution sits on each start, under the marking-bet pool, as
    start.pools.<betType>.betDistribution in hundredths of a percent. The win
    and place pools do not carry a distribution, so they yield nothing. The
    pool that carries betDistribution is found by scanning the start's pools,
    which keeps this independent of the exact bet type name.
    """
    found = False
    for leg in payload.get("races") or []:
        race_id = leg.get("id")
        leg_pools = leg.get("pools") or {}
        for st in leg.get("starts") or []:
            pools = st.get("pools") or {}
            for bet_type, pool in pools.items():
                if not isinstance(pool, dict) or "betDistribution" not in pool:
                    continue
                bd = _float(pool.get("betDistribution"))
                horse = st.get("horse") or {}
                # Turnover sits on the leg pool, not the per-start pool.
                turnover = _int((leg_pools.get(bet_type) or {}).get("turnover"))
                found = True
                yield {
                    "game_id": game_id,
                    "bet_type": bet_type,
                    "date": leg.get("date"),
                    "race_id": race_id,
                    "number": _int(st.get("number")),
                    "horse_id": _int(horse.get("id")),
                    "share": bd / 10000.0 if bd is not None else None,
                    "pool_turnover": turnover,
                }
    if not found:
        report["game_without_betdist"] += 1


# ------------------------------------------------------------------- runner
def _insert(conn, table, rows):
    if not rows:
        return
    cols = list(rows[0])
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) "
        f"VALUES ({','.join('?' for _ in cols)})",
        [tuple(r[c] for c in cols) for r in rows],
    )


def normalize(db_path: str) -> Counter:
    conn = sqlite3.connect(db_path)
    conn.executescript(NORM_SCHEMA)
    report: Counter = Counter()

    race_rows: list[dict] = []
    start_rows: list[dict] = []
    for (raw,) in conn.execute("SELECT json FROM raw_races"):
        report["raw_races_read"] += 1
        try:
            payload = json.loads(raw)
            race_row, srows = parse_race(payload, report)
        except (KeyError, ValueError) as exc:
            report["race_parse_errors"] += 1
            log.warning("race parse error: %s", exc)
            continue
        race_rows.append(race_row)
        start_rows.extend(srows)
    _insert(conn, "norm_races", race_rows)
    _insert(conn, "norm_starts", start_rows)
    report["norm_races"] = len(race_rows)
    report["norm_starts"] = len(start_rows)

    dist_rows: list[dict] = []
    seen = set()
    for (raw,) in conn.execute("SELECT json FROM raw_games"):
        report["raw_games_read"] += 1
        try:
            payload = json.loads(raw)
        except ValueError:
            report["game_parse_errors"] += 1
            continue
        game_id = payload.get("id") or ""
        for rec in extract_bet_distribution(payload, game_id, report):
            key = (rec["game_id"], rec["race_id"], rec["number"])
            if key in seen:
                continue
            seen.add(key)
            dist_rows.append(rec)
    _insert(conn, "norm_bet_distribution", dist_rows)
    report["norm_bet_distribution"] = len(dist_rows)

    conn.commit()

    def scalar(sql):
        return conn.execute(sql).fetchone()[0]

    report["starts_ran"] = scalar("SELECT COUNT(*) FROM norm_starts WHERE scratched=0")
    report["scratched"] = scalar("SELECT COUNT(*) FROM norm_starts WHERE scratched=1")
    report["galloped_or_dq"] = scalar("SELECT COUNT(*) FROM norm_starts WHERE galloped_or_dq=1")
    report["starts_with_final_odds"] = scalar(
        "SELECT COUNT(*) FROM norm_starts WHERE scratched=0 AND final_odds > 0"
    )
    report["races_no_winner"] = scalar(
        "SELECT COUNT(*) FROM (SELECT race_id FROM norm_starts GROUP BY race_id HAVING SUM(is_winner)=0)"
    )
    report["races_multi_winner"] = scalar(
        "SELECT COUNT(*) FROM (SELECT race_id FROM norm_starts GROUP BY race_id HAVING SUM(is_winner)>1)"
    )
    report["date_min"] = scalar("SELECT MIN(date) FROM norm_races")
    report["date_max"] = scalar("SELECT MAX(date) FROM norm_races")
    for sport, n in conn.execute("SELECT sport, COUNT(*) FROM norm_races GROUP BY sport"):
        report[f"races_sport_{sport}"] = n

    conn.close()
    return report


def print_report(report: Counter) -> None:
    print("=== Normalisation report ===")
    print(f"raw races read        {report['raw_races_read']:>8,}")
    print(f"races written         {report['norm_races']:>8,}")
    print(f"starts written        {report['norm_starts']:>8,}")
    print(f"  of which ran        {report['starts_ran']:>8,}")
    print(f"  scratched           {report['scratched']:>8,}")
    print(f"  galloped or DQ      {report['galloped_or_dq']:>8,}")
    print(f"  ran, with win odds  {report['starts_with_final_odds']:>8,}")
    print(f"raw games read        {report['raw_games_read']:>8,}")
    print(f"bet distribution rows {report['norm_bet_distribution']:>8,}")
    print(f"date range            {report['date_min']} -> {report['date_max']}")
    print("races by sport:", {k[12:]: v for k, v in report.items() if k.startswith("races_sport_")})

    print("\n=== Checks (investigate if non-zero) ===")
    for key in (
        "race_parse_errors", "game_parse_errors",
        "start_missing_number", "start_missing_postposition",
        "races_no_winner", "races_multi_winner",
    ):
        print(f"{key:28s} {report.get(key, 0):>8,}")
    if report.get("norm_bet_distribution", 0) == 0 and report.get("raw_games_read", 0) > 0:
        print(
            "\nWARNING: games were read but no bet distribution parsed. Check the "
            "start.pools layout against a real game payload."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/atg.sqlite")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = normalize(args.db)
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
