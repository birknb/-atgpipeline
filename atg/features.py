"""Point-in-time feature engineering.

This builds one feature row per runner per race, using only information that was
knowable before that race started. It reads the norm_ tables, walks the races in
race-date order, and for each race first emits feature rows from the current
running state, then updates that state with the race result. The post-race
fields (place, km-time, final odds) are written to the table as labels and a
benchmark only. They are never read back as features.

The features in this first version are reconstructed from prior races:

  history     prior start count, days since last start, debut flag
  form        recency-weighted speed figure from prior km-times, normalised by a
              point-in-time par for sport, start method and distance bucket
  ability     a horse Elo updated from finishing order
  class       cumulative prior earnings and earnings per start
  people      time-decayed, shrunk driver and trainer win and place rates
  context     field size, post position, distance, start method, equipment change
              flags, age, sex, sport

The market odds are not a feature here. They are the benchmark and are handled
in benchmark.py and evaluate.py. Within-race normalisation of these features is
left to the modelling step, where each feature is centred within its race.

Usage:
    python -m atg.features --db data/atg.sqlite
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
from collections import Counter
from datetime import date

from .ratings import DecayedRate, Elo

log = logging.getLogger("atg.features")

FORM_HALFLIFE_DAYS = 120.0
RATE_HALFLIFE_DAYS = 365.0
RATE_PSEUDO = 20.0
PAR_MIN_COUNT = 30  # a par key needs this many prior times before it is trusted

FEATURE_SCHEMA = """
DROP TABLE IF EXISTS norm_features;
CREATE TABLE norm_features (
    start_id            TEXT PRIMARY KEY,
    race_id             TEXT NOT NULL,
    date                TEXT,
    scheduled_start_time TEXT,
    number              INTEGER,
    horse_id            INTEGER,
    sport               TEXT,
    country             TEXT,
    -- pre-race context
    field_size          INTEGER,
    distance_m          INTEGER,
    start_method        TEXT,
    start_distance_m    INTEGER,
    post_position       INTEGER,
    post_rel            REAL,
    track_id            INTEGER,
    age                 INTEGER,
    sex                 TEXT,
    shoes_changed       INTEGER,
    shoe_front_changed  INTEGER,
    shoe_back_changed   INTEGER,
    sulky_changed       INTEGER,
    -- horse history, prior races only
    hist_starts         INTEGER,
    is_debut            INTEGER,
    days_since_last     REAL,
    form_speed          REAL,
    form_n              INTEGER,
    elo                 REAL,
    elo_default         INTEGER,
    cum_earnings        INTEGER,
    avg_earn_per_start  REAL,
    -- people, prior races only
    driver_id           INTEGER,
    driver_win_rate     REAL,
    driver_place_rate   REAL,
    driver_n            REAL,
    trainer_id          INTEGER,
    trainer_win_rate    REAL,
    trainer_place_rate  REAL,
    trainer_n           REAL,
    -- labels and benchmark, never features
    is_winner           INTEGER,
    place               INTEGER,
    final_odds          REAL,
    galloped_or_dq      INTEGER
);
CREATE INDEX idx_feat_race ON norm_features(race_id);
CREATE INDEX idx_feat_date ON norm_features(date);
CREATE INDEX idx_feat_horse ON norm_features(horse_id);
"""

COLUMNS = [
    "start_id", "race_id", "date", "scheduled_start_time", "number", "horse_id",
    "sport", "country", "field_size", "distance_m", "start_method",
    "start_distance_m", "post_position", "post_rel", "track_id", "age", "sex",
    "shoes_changed", "shoe_front_changed", "shoe_back_changed", "sulky_changed",
    "hist_starts", "is_debut", "days_since_last", "form_speed", "form_n", "elo",
    "elo_default", "cum_earnings", "avg_earn_per_start", "driver_id",
    "driver_win_rate", "driver_place_rate", "driver_n", "trainer_id",
    "trainer_win_rate", "trainer_place_rate", "trainer_n", "is_winner", "place",
    "final_odds", "galloped_or_dq",
]


def dist_bucket(distance) -> str:
    if distance is None:
        return "na"
    if distance < 1800:
        return "sprint"
    if distance < 2200:
        return "mid"
    if distance < 2700:
        return "long"
    return "xlong"


class Par:
    """Point-in-time par km-times. Stores a sum and count per key, with a
    fallback from the specific key to coarser keys. A key is used only once it
    has enough observations, so an early, thin estimate does not dominate."""

    def __init__(self) -> None:
        self.s: dict = {}

    def add(self, km: float, keys: tuple) -> None:
        for k in keys + (("__global__",),):
            v = self.s.setdefault(k, [0.0, 0.0])
            v[0] += km
            v[1] += 1.0

    def mean(self, keys: tuple):
        for k in keys:
            v = self.s.get(k)
            if v and v[1] >= PAR_MIN_COUNT:
                return v[0] / v[1]
        g = self.s.get(("__global__",))
        return g[0] / g[1] if g and g[1] > 0 else None


class DecayedMean:
    """Recency-weighted mean of a value per entity. New observations decay older
    ones by a half-life. The decay cancels in the ratio, so the stored sum and
    count give the recency-weighted mean directly."""

    def __init__(self, halflife_days: float) -> None:
        self.hl = halflife_days
        self.s: dict = {}

    def add(self, eid: int, day: int, value: float) -> None:
        st = self.s.get(eid)
        if st is None:
            self.s[eid] = [value, 1.0, day]
            return
        dsum, dcount, last = st
        f = 0.5 ** ((day - last) / self.hl) if day > last else 1.0
        self.s[eid] = [dsum * f + value, dcount * f + 1.0, day]

    def mean(self, eid: int):
        st = self.s.get(eid)
        if st is None or st[1] <= 0:
            return None
        return st[0] / st[1]


def par_keys(sport, start_method, distance) -> tuple:
    b = dist_bucket(distance)
    return (
        (sport, start_method, b),
        (sport, start_method),
        (sport,),
    )


def build(db_path: str) -> Counter:
    conn = sqlite3.connect(db_path)
    conn.executescript(FEATURE_SCHEMA)
    report: Counter = Counter()

    races = conn.execute(
        """SELECT race_id, date, scheduled_start_time, track_id, country,
                  distance_m, start_method, sport, n_starters
           FROM norm_races
           ORDER BY date, COALESCE(scheduled_start_time, ''), race_no"""
    ).fetchall()

    starts_by_race: dict[str, list] = {}
    for row in conn.execute(
        """SELECT start_id, race_id, number, post_position, start_distance_m,
                  horse_id, age, sex, shoes_changed, shoe_front_changed,
                  shoe_back_changed, sulky_changed, driver_id, trainer_id,
                  scratched, place, km_time_s, prize_money, final_odds,
                  is_winner, galloped_or_dq
           FROM norm_starts"""
    ):
        starts_by_race.setdefault(row[1], []).append(row)

    # Running state, all updated only after a race's rows are emitted.
    h_starts: dict[int, int] = {}
    h_last_day: dict[int, int] = {}
    h_cum_earn: dict[int, int] = {}
    h_form_n: dict[int, int] = {}
    form = DecayedMean(FORM_HALFLIFE_DAYS)
    elo = Elo()
    drv_win = DecayedRate(RATE_HALFLIFE_DAYS, RATE_PSEUDO)
    drv_place = DecayedRate(RATE_HALFLIFE_DAYS, RATE_PSEUDO)
    trn_win = DecayedRate(RATE_HALFLIFE_DAYS, RATE_PSEUDO)
    trn_place = DecayedRate(RATE_HALFLIFE_DAYS, RATE_PSEUDO)
    par = Par()
    g_wins = g_top3 = g_starts = 0

    out_rows: list[tuple] = []

    for (race_id, rdate, sched, track_id, country, distance_m, start_method,
         sport, n_starters) in races:
        starts = starts_by_race.get(race_id, [])
        if not rdate:
            report["race_missing_date"] += 1
            continue
        day = date.fromisoformat(rdate).toordinal()

        ran = [s for s in starts if not s[14]]  # scratched is index 14
        field_size = len(ran)
        g_win_rate = g_wins / g_starts if g_starts > 0 else 0.10
        g_place_rate = g_top3 / g_starts if g_starts > 0 else 0.30

        # 1) Emit feature rows from pre-race state.
        for s in ran:
            (start_id, _rid, number, post_position, start_distance_m, horse_id,
             age, sex, shoes_changed, shoe_front_changed, shoe_back_changed,
             sulky_changed, driver_id, trainer_id, _scratched, place, km_time_s,
             prize_money, final_odds, is_winner, galloped_or_dq) = s

            hs = h_starts.get(horse_id, 0)
            last_day = h_last_day.get(horse_id)
            cum_earn = h_cum_earn.get(horse_id, 0)
            row = {
                "start_id": start_id,
                "race_id": race_id,
                "date": rdate,
                "scheduled_start_time": sched,
                "number": number,
                "horse_id": horse_id,
                "sport": sport,
                "country": country,
                "field_size": field_size,
                "distance_m": distance_m,
                "start_method": start_method,
                "start_distance_m": start_distance_m,
                "post_position": post_position,
                "post_rel": (post_position / field_size)
                if post_position is not None and field_size else None,
                "track_id": track_id,
                "age": age,
                "sex": sex,
                "shoes_changed": shoes_changed,
                "shoe_front_changed": shoe_front_changed,
                "shoe_back_changed": shoe_back_changed,
                "sulky_changed": sulky_changed,
                "hist_starts": hs,
                "is_debut": 1 if hs == 0 else 0,
                "days_since_last": float(day - last_day) if last_day is not None else None,
                "form_speed": form.mean(horse_id),
                "form_n": h_form_n.get(horse_id, 0),
                "elo": elo.get(horse_id),
                "elo_default": 0 if elo.has(horse_id) else 1,
                "cum_earnings": cum_earn,
                "avg_earn_per_start": (cum_earn / hs) if hs > 0 else None,
                "driver_id": driver_id,
                "driver_win_rate": drv_win.rate(driver_id, day, g_win_rate)
                if driver_id is not None else None,
                "driver_place_rate": drv_place.rate(driver_id, day, g_place_rate)
                if driver_id is not None else None,
                "driver_n": drv_win.count(driver_id, day) if driver_id is not None else None,
                "trainer_id": trainer_id,
                "trainer_win_rate": trn_win.rate(trainer_id, day, g_win_rate)
                if trainer_id is not None else None,
                "trainer_place_rate": trn_place.rate(trainer_id, day, g_place_rate)
                if trainer_id is not None else None,
                "trainer_n": trn_win.count(trainer_id, day) if trainer_id is not None else None,
                "is_winner": is_winner,
                "place": place,
                "final_odds": final_odds,
                "galloped_or_dq": galloped_or_dq,
            }
            out_rows.append(tuple(row[c] for c in COLUMNS))
            report["rows"] += 1
            if hs == 0:
                report["debuts"] += 1
            if row["form_speed"] is not None:
                report["with_form"] += 1
            if row["elo_default"] == 0:
                report["with_elo_history"] += 1

        # 2) Update state from this race's result. The par is read once, before
        # any of this race's times are added, so a speed figure is always
        # relative to earlier races only.
        finishers = [(s[5], s[15]) for s in ran if s[15] is not None and s[15] > 0]
        max_place = max((p for _, p in finishers), default=0)
        ranked = list(finishers) + [
            (s[5], max_place + 1) for s in ran if not (s[15] is not None and s[15] > 0)
        ]
        elo.update(ranked)

        keys = par_keys(sport, start_method, distance_m)
        par_mean = par.mean(keys)
        race_km: list[float] = []
        for s in ran:
            horse_id = s[5]
            km_time_s = s[16]
            prize_money = s[17] or 0
            place = s[15]
            driver_id = s[12]
            trainer_id = s[13]
            is_winner = 1 if (place == 1) else 0
            is_top3 = 1 if (place is not None and 1 <= place <= 3) else 0

            if km_time_s is not None:
                if par_mean is not None:
                    form.add(horse_id, day, par_mean - km_time_s)
                    h_form_n[horse_id] = h_form_n.get(horse_id, 0) + 1
                race_km.append(km_time_s)

            h_starts[horse_id] = h_starts.get(horse_id, 0) + 1
            h_cum_earn[horse_id] = h_cum_earn.get(horse_id, 0) + prize_money
            h_last_day[horse_id] = day

            if driver_id is not None:
                drv_win.update(driver_id, day, bool(is_winner))
                drv_place.update(driver_id, day, bool(is_top3))
            if trainer_id is not None:
                trn_win.update(trainer_id, day, bool(is_winner))
                trn_place.update(trainer_id, day, bool(is_top3))

            g_wins += is_winner
            g_top3 += is_top3
            g_starts += 1

        for km in race_km:
            par.add(km, keys)

    conn.executemany(
        f"INSERT OR REPLACE INTO norm_features ({','.join(COLUMNS)}) "
        f"VALUES ({','.join('?' for _ in COLUMNS)})",
        out_rows,
    )
    conn.commit()

    report["date_min"] = conn.execute("SELECT MIN(date) FROM norm_features").fetchone()[0]
    report["date_max"] = conn.execute("SELECT MAX(date) FROM norm_features").fetchone()[0]
    conn.close()
    return report


def print_report(report: Counter) -> None:
    rows = report["rows"] or 1
    print("=== Feature build report ===")
    print(f"feature rows written  {report['rows']:>9,}")
    print(f"  debuts (no history) {report['debuts']:>9,}  ({report['debuts'] / rows * 100:.1f}%)")
    print(f"  with form speed     {report['with_form']:>9,}  ({report['with_form'] / rows * 100:.1f}%)")
    print(f"  with elo history    {report['with_elo_history']:>9,}  ({report['with_elo_history'] / rows * 100:.1f}%)")
    print(f"date range            {report['date_min']} -> {report['date_max']}")
    if report.get("race_missing_date"):
        print(f"races skipped, no date {report['race_missing_date']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/atg.sqlite")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = build(args.db)
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
