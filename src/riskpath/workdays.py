"""Working-day calendar (Mon-Fri plus an optional holiday list).

Convention: the project starts at the beginning of working day 1 (the start
date, rolled forward to a working day). A task or project that takes D
working days, D > 0, finishes at the end of working day ceil(D). Fractional
durations therefore round up to the working day in which work completes.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Iterable

import numpy as np

_EPS = 1e-9


def _to_date(value) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


class WorkCalendar:
    def __init__(self, start, holidays: Iterable = ()):  # noqa: D401
        self.holidays = np.array(sorted({_to_date(h) for h in holidays}), dtype="datetime64[D]")
        raw = np.datetime64(_to_date(start), "D")
        self.start = np.busday_offset(raw, 0, roll="forward", holidays=self.holidays)

    @property
    def start_date(self) -> date:
        return self.start.astype(object)

    def is_workday(self, d) -> bool:
        return bool(np.is_busday(np.datetime64(_to_date(d), "D"), holidays=self.holidays))

    def add_workdays(self, n: int) -> date:
        """Date of working day index n (0 = start date)."""
        return np.busday_offset(self.start, int(n), roll="forward", holidays=self.holidays).astype(object)

    def finish_date(self, days: float) -> date:
        n = max(math.ceil(float(days) - _EPS), 1)
        return self.add_workdays(n - 1)

    def finish_dates(self, days) -> np.ndarray:
        d = np.asarray(days, dtype=float)
        n = np.maximum(np.ceil(d - _EPS), 1).astype(int)
        return np.busday_offset(self.start, n - 1, roll="forward", holidays=self.holidays)

    def workdays_through(self, target) -> int:
        """Number of working days from the start date through ``target`` inclusive."""
        end = np.datetime64(_to_date(target) + timedelta(days=1), "D")
        return int(np.busday_count(self.start, end, holidays=self.holidays))

    def workday_index(self, d) -> int:
        """Working days elapsed between the start date and ``d`` (exclusive of d)."""
        return int(np.busday_count(self.start, np.datetime64(_to_date(d), "D"), holidays=self.holidays))
