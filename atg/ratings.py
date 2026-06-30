"""Running rating systems updated in race-date order.

These are point-in-time by construction. The value read before a race is the
feature, and the update from the race result happens only after that race's
feature rows are emitted. Nothing here looks at a race's own outcome when
producing that race's features.

Two systems are provided. Elo is a compact ability rating for horses, updated
from the finishing order. DecayedRate keeps a time-decayed, shrunk win or place
rate for an entity such as a driver or trainer.
"""
from __future__ import annotations


class Elo:
    """Multiplayer Elo over a finishing order.

    Each runner is compared with every other runner in the race. A runner that
    finished ahead scores one against a runner behind it, and a half for a tie.
    Ratings move toward the average pairwise surprise, divided by the field size
    so a large field does not cause an outsized update. Non-finishers are passed
    in ranked behind all finishers.
    """

    def __init__(self, k: float = 24.0, base: float = 1500.0, scale: float = 400.0):
        self.k = k
        self.base = base
        self.scale = scale
        self.r: dict[int, float] = {}

    def get(self, eid: int) -> float:
        return self.r.get(eid, self.base)

    def has(self, eid: int) -> bool:
        return eid in self.r

    def update(self, ranked: list[tuple[int, int]]) -> None:
        """ranked is a list of (entity_id, rank). Lower rank finished better.
        Ties are allowed by giving equal ranks."""
        ids = [e for e, _ in ranked]
        rank = {e: rk for e, rk in ranked}
        n = len(ids)
        if n < 2:
            return
        cur = {e: self.get(e) for e in ids}
        delta = {e: 0.0 for e in ids}
        for a in ids:
            ra = cur[a]
            for b in ids:
                if a == b:
                    continue
                if rank[a] < rank[b]:
                    s = 1.0
                elif rank[a] > rank[b]:
                    s = 0.0
                else:
                    s = 0.5
                expected = 1.0 / (1.0 + 10 ** ((cur[b] - ra) / self.scale))
                delta[a] += s - expected
        for e in ids:
            self.r[e] = cur[e] + self.k * delta[e] / (n - 1)


class DecayedRate:
    """Time-decayed binary-event rate per entity, with empirical-Bayes shrinkage.

    For each entity it keeps decayed counts of starts and of the event, a win or
    a top-three finish. Counts decay by a half-life in days between observations.
    The rate read before a race is shrunk toward a supplied global rate by a
    pseudo-count, so an entity with few decayed starts does not look extreme.
    """

    def __init__(self, halflife_days: float = 365.0, pseudo: float = 20.0):
        self.halflife = halflife_days
        self.pseudo = pseudo
        # entity -> [decayed_events, decayed_starts, last_day_ordinal]
        self.state: dict[int, list[float]] = {}

    def _decayed(self, eid: int, day: int) -> tuple[float, float]:
        st = self.state.get(eid)
        if st is None:
            return 0.0, 0.0
        ev, n, last = st
        dt = day - last
        if dt > 0:
            f = 0.5 ** (dt / self.halflife)
            ev *= f
            n *= f
        return ev, n

    def rate(self, eid: int, day: int, global_rate: float) -> float:
        ev, n = self._decayed(eid, day)
        return (ev + self.pseudo * global_rate) / (n + self.pseudo)

    def count(self, eid: int, day: int) -> float:
        _, n = self._decayed(eid, day)
        return n

    def update(self, eid: int, day: int, event: bool) -> None:
        ev, n = self._decayed(eid, day)
        ev += 1.0 if event else 0.0
        n += 1.0
        self.state[eid] = [ev, n, day]
