from datetime import date

import numpy as np

from riskpath import analysis
from riskpath.workdays import WorkCalendar


def test_working_day_calendar_skips_weekends_and_holidays():
    cal = WorkCalendar("2027-01-01", holidays=["2027-01-01", "2027-01-05"])  # Fri holiday, Tue holiday
    assert cal.start_date == date(2027, 1, 4)  # rolled forward to Monday
    assert cal.finish_date(1) == date(2027, 1, 4)
    assert cal.finish_date(1.2) == date(2027, 1, 6)  # day 2 is Wed (Tue is a holiday)
    assert cal.finish_date(5) == date(2027, 1, 11)  # Mon, Wed, Thu, Fri, Mon
    assert cal.workdays_through(date(2027, 1, 8)) == 4
    assert not cal.is_workday("2027-01-09")
    arr = cal.finish_dates(np.array([1.0, 5.0]))
    assert [d.astype(object) for d in arr] == [date(2027, 1, 4), date(2027, 1, 11)]


def test_target_probability_and_percentiles():
    finish = np.arange(1, 101, dtype=float)  # 1..100 working days, uniform
    assert analysis.prob_within(finish, 80) == 0.80
    assert analysis.prob_within(finish - 0.5, 80) == 0.80  # fractional days round up
    cal = WorkCalendar("2027-01-04")
    assert analysis.prob_by_date(finish, cal, date(2027, 1, 8)) == 0.05
    p = analysis.percentiles(finish)
    assert p[50] == np.percentile(finish, 50) and p[80] < p[95]


def test_schedule_margin_is_p80_minus_deterministic():
    m = analysis.schedule_margin(p80=211.19, deterministic=170)
    assert abs(m["raw_days"] - 41.19) < 1e-9
    assert m["recommended_days"] == 42
    assert abs(m["pct_of_deterministic"] - 100 * 41.19 / 170) < 1e-9
    assert analysis.schedule_margin(150, 170)["recommended_days"] == 0
