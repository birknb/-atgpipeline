"""Offline point-in-time checks for the feature builder (atg/features.py).

Builds a tiny synthetic database of dated races and verifies that each feature
row uses only prior-race information. The strongest checks are the leakage ones:
a horse that wins a race must not see that race's win, earnings or start count in
its own feature row. Run directly:

    python tests/test_features.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atg import features  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, cond: bool) -> None:
    checks.append((name, bool(cond)))


def approx(a, b, tol: float = 1e-6) -> bool:
    return a is not None and abs(a - b) <= tol


RACE_COLS = [
    "race_id", "date", "scheduled_start_time", "track_id", "country",
    "distance_m", "start_method", "sport", "n_starters", "race_no",
]
START_COLS = [
    "start_id", "race_id", "number", "post_position", "start_distance_m",
    "horse_id", "age", "sex", "shoes_changed", "shoe_front_changed",
    "shoe_back_changed", "sulky_changed", "driver_id", "trainer_id", "scratched",
    "place", "km_time_s", "prize_money", "final_odds", "is_winner",
    "galloped_or_dq", "stat_life_starts", "stat_win_pct", "stat_place_pct",
    "stat_earn_per_start", "stat_start_points", "best_km_time_s",
]


def make_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE norm_races ({', '.join(RACE_COLS)})")
    conn.execute(f"CREATE TABLE norm_starts ({', '.join(START_COLS)})")

    def race(rid, d, rno, n):
        conn.execute(
            f"INSERT INTO norm_races ({','.join(RACE_COLS)}) VALUES ({','.join('?' * len(RACE_COLS))})",
            (rid, d, f"{d}T19:00:00", 1, "SE", 2140, "auto", "trot", n, rno),
        )

    def start(rid, horse, post, place, km, prize, winner, driver, trainer, gallop=0):
        sid = f"{rid}_{horse}"
        conn.execute(
            f"INSERT INTO norm_starts ({','.join(START_COLS)}) VALUES ({','.join('?' * len(START_COLS))})",
            (sid, rid, post, post, 2140, horse, 5, "gelding", 0, 0, 0, 0,
             driver, trainer, 0, place, km, prize, 5.0, winner, gallop,
             None, None, None, None, None, None),
        )

    # R0 seeds par. R1 then produces form figures. R2 is the test race for H1.
    race("R0", "2024-01-01", 1, 3)
    start("R0", 1, 1, 1, 75.0, 1000, 1, 10, 20)
    start("R0", 2, 2, 2, 76.0, 500, 0, 11, 21)
    start("R0", 3, 3, 3, 77.0, 250, 0, 11, 21)

    race("R1", "2024-01-08", 1, 3)
    start("R1", 1, 1, 1, 74.0, 1000, 1, 10, 20)
    start("R1", 2, 2, 2, 78.0, 500, 0, 11, 21)
    start("R1", 3, 3, None, None, 0, 0, 11, 21, gallop=1)  # galloped

    race("R2", "2024-01-20", 1, 2)
    start("R2", 1, 1, 1, 73.0, 5000, 1, 10, 20)  # big prize to expose leakage
    start("R2", 2, 2, 2, 79.0, 500, 0, 11, 21)

    conn.commit()
    conn.close()


def feat(conn, rid, horse) -> dict:
    cur = conn.execute("SELECT * FROM norm_features WHERE start_id = ?", (f"{rid}_{horse}",))
    cols = [c[0] for c in cur.description]
    row = cur.fetchone()
    return dict(zip(cols, row)) if row else {}


def main() -> int:
    features.PAR_MIN_COUNT = 1  # tiny synthetic data, trust the par immediately
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "t.sqlite")
    make_db(db)
    rep = features.build(db)
    check("rows written", rep["rows"] == 8)

    conn = sqlite3.connect(db)

    r0h1 = feat(conn, "R0", 1)
    check("debut: hist_starts 0", r0h1["hist_starts"] == 0)
    check("debut: is_debut flag", r0h1["is_debut"] == 1)
    check("debut: days_since_last is null", r0h1["days_since_last"] is None)
    check("debut: form_speed is null", r0h1["form_speed"] is None)
    check("debut: elo is default 1500", approx(r0h1["elo"], 1500.0))
    check("debut: elo_default flag", r0h1["elo_default"] == 1)
    check("debut: cum_earnings 0", r0h1["cum_earnings"] == 0)
    check("debut: field_size 3", r0h1["field_size"] == 3)

    r1h1 = feat(conn, "R1", 1)
    check("R1 H1 hist_starts 1", r1h1["hist_starts"] == 1)
    check("R1 H1 days_since_last 7", approx(r1h1["days_since_last"], 7.0))
    check("R1 H1 cum_earnings 1000 (only R0)", r1h1["cum_earnings"] == 1000)
    check("R1 H1 elo above default", r1h1["elo"] > 1500.0)

    r1h3 = feat(conn, "R1", 3)
    check("R1 H3 hist_starts 1", r1h3["hist_starts"] == 1)

    r2h1 = feat(conn, "R2", 1)
    # Leakage checks: R2 result must not enter R2's own feature row for H1.
    check("LEAK hist_starts excludes current (2 not 3)", r2h1["hist_starts"] == 2)
    check("LEAK cum_earnings excludes current (2000 not 7000)", r2h1["cum_earnings"] == 2000)
    check("R2 H1 days_since_last 12", approx(r2h1["days_since_last"], 12.0))
    check("R2 H1 avg_earn_per_start 1000", approx(r2h1["avg_earn_per_start"], 1000.0))
    check("R2 H1 field_size 2", r2h1["field_size"] == 2)
    # form from R1 only: par after R0 for (trot,auto,mid) = mean(75,76,77)=76; R1 km 74 -> +2.
    check("R2 H1 form_speed approx +2.0", approx(r2h1["form_speed"], 2.0, 1e-6))
    check("R2 H1 form_n 1", r2h1["form_n"] == 1)
    check("R2 H1 is_winner label 1", r2h1["is_winner"] == 1)

    # Driver D1 (id 10) won R0 and R1, so its rate before R2 beats its debut prior.
    check("driver win rate rises with history", r2h1["driver_win_rate"] > r0h1["driver_win_rate"])

    conn.close()

    print("=== Feature builder test results ===")
    ok = True
    for name, cond in checks:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond
    print("\nALL PASSED" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
