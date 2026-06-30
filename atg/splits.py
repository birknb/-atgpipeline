"""Time-based train and test splits for the evaluation harness.

A race is the unit. Splits are keyed on race date, never on row order, so a
single day is never divided between train and test. Two leakage guards are
available. Purge removes the tail of the training period immediately before a
test window. Embargo removes a band immediately after a test window from
training.

Features in this project look only backwards in time. For a strict forward
split the training period is always before the test window, so purge is the
guard that matters and embargo is a no-op kept for rolling or middle-test
schemes used later. Purge and embargo are a safety margin against any
accidental forward-looking feature and against shared-history overlap.

Two entry points:

  fixed_split          one train, validation and test partition by cut dates.
                       For fast iteration. Numbers from it are provisional and
                       are labelled pre-walk-forward, never quoted as results.
  walk_forward_splits  expanding-window folds. The source of quotable results.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def _to_date(x) -> date:
    if isinstance(x, date):
        return x
    return date.fromisoformat(str(x)[:10])


def _race_dates(races: pd.DataFrame) -> pd.Series:
    """One row per race: a Series of python dates indexed by race_id."""
    d = races[["race_id", "date"]].drop_duplicates("race_id")
    return pd.Series(
        [_to_date(x) for x in d["date"]],
        index=d["race_id"].to_numpy(),
    )


def fixed_split(
    races: pd.DataFrame,
    train_end: str,
    val_end: str,
    purge_days: int = 0,
) -> dict[str, list[str]]:
    """Partition race ids into train, val and test by date.

    train is on or before train_end, val is from train_end to val_end, test is
    after val_end. purge_days removes the last purge_days of the training period
    before the validation window, leaving a gap so the two do not touch.
    """
    dates = _race_dates(races)
    train_end_d = _to_date(train_end)
    val_end_d = _to_date(val_end)
    purge_cut = train_end_d - timedelta(days=purge_days)

    train = dates.index[dates <= purge_cut]
    val = dates.index[(dates > train_end_d) & (dates <= val_end_d)]
    test = dates.index[dates > val_end_d]
    return {"train": list(train), "val": list(val), "test": list(test)}


def walk_forward_splits(
    races: pd.DataFrame,
    first_test_start: str,
    test_span_days: int,
    step_days: int | None = None,
    n_folds: int | None = None,
    purge_days: int = 0,
    embargo_days: int = 0,
) -> list[dict]:
    """Expanding-window folds. Each fold trains on all races before the test
    window (minus the purge gap) and tests on a window of test_span_days.

    step_days defaults to test_span_days, which gives non-overlapping test
    windows that tile the period after first_test_start. Each fold is a dict
    with test_start, test_end, train (race ids) and test (race ids). Folds with
    no test races are skipped.
    """
    dates = _race_dates(races)
    first = _to_date(first_test_start)
    step = step_days if step_days is not None else test_span_days
    last_date = dates.max()

    folds: list[dict] = []
    t0 = first
    while t0 <= last_date:
        t1 = t0 + timedelta(days=test_span_days)
        test = dates.index[(dates >= t0) & (dates < t1)]
        if len(test) > 0:
            purge_cut = t0 - timedelta(days=purge_days)
            embargo_hi = t1 + timedelta(days=embargo_days)
            in_embargo = (dates >= t1) & (dates < embargo_hi)
            train = dates.index[(dates < purge_cut) & (~in_embargo)]
            folds.append(
                {
                    "test_start": t0.isoformat(),
                    "test_end": t1.isoformat(),
                    "train": list(train),
                    "test": list(test),
                }
            )
            if n_folds is not None and len(folds) >= n_folds:
                break
        t0 = t0 + timedelta(days=step)
    return folds
