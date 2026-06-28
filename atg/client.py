"""HTTP client for ATG's public racing info API.

Endpoints used (all public and read-only, the same API that powers atg.se):
  - /calendar/day/{YYYY-MM-DD}  -> race days: tracks, races, games
  - /races/{raceId}             -> full race card with horses, drivers, results
  - /games/{gameId}             -> pool game (V75, V86, ...) with bet distribution

The client enforces a fixed delay between requests and sends an identifiable
User-Agent. Transient errors are retried with exponential backoff. A 404
returns None, since the data does not exist. Persistent failure raises, so
the caller never silently loses data.
"""
from __future__ import annotations

import logging
import time

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://www.atg.se/services/racinginfo/v1/api"

# The User-Agent identifies the crawler. Put real contact information here
# so the API operator can reach you.
USER_AGENT = "trav-ml-pipeline/0.1 (hobby research project)"


class AtgClient:
    """Rate-limited, retrying HTTP client for the ATG racing info API."""

    def __init__(
        self,
        delay_s: float = 0.4,
        max_retries: int = 4,
        timeout_s: float = 30.0,
    ) -> None:
        self.delay_s = delay_s
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})
        self._last_request_ts = 0.0
        self.request_count = 0

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------
    def calendar_day(self, date_str: str) -> dict | None:
        """Calendar for one day: tracks, race ids, game ids. date_str = YYYY-MM-DD."""
        return self._get(f"/calendar/day/{date_str}")

    def race(self, race_id: str) -> dict | None:
        """Full race card. race_id like '2026-06-10_23_5' (date_trackId_raceNo)."""
        return self._get(f"/races/{race_id}")

    def game(self, game_id: str) -> dict | None:
        """Pool game. game_id like 'V86_2026-06-10_23_3' (type_date_trackId_legRaceNo)."""
        return self._get(f"/games/{game_id}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.delay_s:
            time.sleep(self.delay_s - elapsed)

    def _get(self, path: str) -> dict | None:
        url = f"{BASE_URL}{path}"
        backoff = 1.0
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self._session.get(url, timeout=self.timeout_s)
                self._last_request_ts = time.monotonic()
                self.request_count += 1

                if resp.status_code == 404:
                    log.debug("404 for %s", path)
                    return None
                if resp.status_code == 429 or resp.status_code >= 500:
                    # Rate limiting and server errors are transient. Retry.
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                if resp.status_code >= 400:
                    # Other client errors will not change on retry. Fail now.
                    raise RuntimeError(f"HTTP {resp.status_code} for {url}")
                # A 2xx body that is not JSON usually means an intercepting
                # proxy returned an HTML page. resp.json() raises ValueError,
                # which is caught below and retried.
                return resp.json()

            except (requests.ConnectionError, requests.Timeout, requests.HTTPError,
                    ValueError) as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"Giving up on {url} after {attempt} attempts") from exc
                log.warning(
                    "Attempt %d/%d failed for %s (%s), retrying in %.1fs",
                    attempt, self.max_retries, path, exc, backoff,
                )
                time.sleep(backoff)
                backoff *= 2
        return None  # unreachable
