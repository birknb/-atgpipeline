"""Offline validation of Phase 2 against fabricated payloads.

The payloads here mirror the real API shape confirmed in June 2026: driver and
trainer with firstName and lastName, non-finishers with galloped or
disqualified flags and no place, withdrawn horses with scratched true and
finalOdds 0.0, nested shoes, and the marking-bet betDistribution under
start.pools.<betType>. The test runs the real db, normalize and benchmark code
over them and checks the results against values computed by hand.

For a check against real payloads, see build_sample_db.py.

Run:
    python tests/test_phase2.py
"""
from __future__ import annotations

import math
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atg import benchmark, normalize  # noqa: E402
from atg.db import Db  # noqa: E402


def horse_start(num, hid, odds, place=None, km=None, gallop=False, scratched=False):
    start = {
        "number": num,
        "postPosition": num,
        "distance": 2140,
        "horse": {
            "id": hid,
            "name": f"Horse {hid}",
            "age": 6,
            "sex": "gelding",
            "money": 100000 + hid,
            "shoes": {
                "reported": True,
                "front": {"hasShoe": True, "changed": num == 1},
                "back": {"hasShoe": True, "changed": False},
            },
            "sulky": {"reported": False},
            "trainer": {"id": 500 + hid, "firstName": "Tr", "lastName": str(hid)},
        },
        "driver": {"id": 300 + hid, "firstName": "Dr", "lastName": str(hid)},
    }
    if scratched:
        start["scratched"] = True
        start["result"] = {"finishOrder": 99, "finalOdds": 0.0, "startNumber": num}
        return start
    if gallop:
        start["result"] = {
            "finishOrder": 32, "kmTime": {"code": "2"},
            "galloped": True, "disqualified": True,
            "finalOdds": odds, "startNumber": num,
        }
        return start
    start["result"] = {
        "place": place, "finishOrder": place,
        "kmTime": {"minutes": 1, "seconds": km, "tenths": 0},
        "prizeMoney": 1000, "finalOdds": odds, "startNumber": num,
    }
    return start


def race(race_id, date, starts):
    for st in starts:
        st["id"] = f"{race_id}_{st['number']}"  # real start ids are race-qualified
    return {
        "id": race_id, "date": date, "number": int(race_id.rsplit("_", 1)[-1]),
        "distance": 2140, "startMethod": "auto",
        "startTime": f"{date}T19:30:05", "scheduledStartTime": f"{date}T19:30:00",
        "sport": "trot", "status": "results",
        "track": {"id": 23, "name": "Solvalla", "condition": "light", "countryCode": "SE"},
        "starts": starts,
    }


def v75_game(leg_race):
    """A V75 game embedding one leg with a per-start betDistribution."""
    leg = {k: v for k, v in leg_race.items() if k != "starts"}
    leg["pools"] = {"V75": {"turnover": 100000}}
    leg["starts"] = []
    shares = {1: 5000, 2: 3000, 3: 2000}  # hundredths of a percent, sum 10000
    for st in leg_race["starts"]:
        n = st["number"]
        leg["starts"].append({
            "number": n,
            "horse": {"id": st["horse"]["id"]},
            "result": st.get("result", {}),
            "pools": {
                "vinnare": {"@type": ".VinnareStartPool", "odds": 100},
                "V75": {"@type": ".MarkingBetStartPool", "betDistribution": shares[n]},
            },
        })
    return {
        "id": f"V75_{leg_race['id']}", "status": "results",
        "pools": {"V75": {"turnover": 100000}}, "races": [leg],
    }


def build_fixture(path):
    db = Db(path)
    r1 = race("2025-06-04_23_5", "2025-06-04", [
        horse_start(1, 101, 2.0, place=1, km=12),
        horse_start(2, 102, 4.0, place=2, km=13),
        horse_start(3, 103, 5.0, gallop=True),
        horse_start(4, 104, 0.0, scratched=True),
    ])
    r2 = race("2025-06-04_23_6", "2025-06-04", [
        horse_start(1, 111, 1.5, place=2, km=12),
        horse_start(2, 112, 3.0, place=1, km=11),
        horse_start(3, 113, 6.0, place=3, km=14),
    ])
    db.upsert_race(r1)
    db.upsert_race(r2)
    db.upsert_game(v75_game(r2))  # makes r2 a V75 leg with a spelprocent
    db.close()


def approx(a, b, tol=2e-3):
    return abs(a - b) <= tol


def main():
    tmp = tempfile.mkdtemp(prefix="atg_test_")
    path = os.path.join(tmp, "fixture.sqlite")
    build_fixture(path)

    report = normalize.normalize(path)
    normalize.print_report(report)

    checks = []

    def check(name, cond):
        checks.append((name, cond))

    check("2 races", report["norm_races"] == 2)
    check("7 starts", report["norm_starts"] == 7)
    check("1 scratched", report["scratched"] == 1)
    check("1 galloper/DQ", report["galloped_or_dq"] == 1)
    check("no parse errors", report["race_parse_errors"] == 0)
    check("no missing winner", report["races_no_winner"] == 0)
    check("3 bet distribution rows", report["norm_bet_distribution"] == 3)

    # Spot-check the corrected field parsing on the real-shaped payload.
    conn = sqlite3.connect(path)
    row = conn.execute(
        "SELECT trainer_name, driver_name, shoes_changed, galloped, disqualified, place "
        "FROM norm_starts WHERE start_id='2025-06-04_23_5_3'"
    ).fetchone()
    check("trainer name from firstName/lastName", row[0] == "Tr 103")
    check("galloper flagged", row[3] == 1 and row[4] == 1)
    check("galloper has no place", row[5] is None)
    win = conn.execute(
        "SELECT shoes_changed FROM norm_starts WHERE start_id='2025-06-04_23_5_1'"
    ).fetchone()
    check("shoes_changed parsed from nested front", win[0] == 1)

    runners = benchmark.load_runners(conn, country=None, sport=None)
    check("6 ran runners", len(runners) == 6)

    odds_frame, _ = benchmark.market_frame_from_odds(runners)
    odds_res = benchmark.evaluate(odds_frame, "test: odds")
    expected_odds_ll = (-math.log(0.5 / 0.95) - math.log((1 / 3) / (1 / 1.5 + 1 / 3 + 1 / 6))) / 2.0
    check("odds races kept = 2", odds_res["n_races"] == 2)
    check(f"odds log loss approx {expected_odds_ll:.4f}",
          approx(odds_res["log_loss"], expected_odds_ll))

    bd_frame, _ = benchmark.market_frame_from_betdist(runners, conn)
    bd_res = benchmark.evaluate(bd_frame, "test: spelprocent")
    check("spelprocent races kept = 1", bd_res["n_races"] == 1)
    check("spelprocent log loss approx 1.2040",
          approx(bd_res["log_loss"], -math.log(0.30)))
    conn.close()

    print("\n=== Test results ===")
    ok = True
    for name, cond in checks:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond
    print("\nALL PASSED" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
