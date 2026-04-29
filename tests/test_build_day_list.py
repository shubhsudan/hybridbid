"""Unit tests for build_day_list CT-day alignment fix."""
import numpy as np
import pandas as pd
import pytest

from src.data.postbreak_milp import build_day_list, STEPS_PER_DAY, SEQ_LEN


def _make_merged(start_utc: str, end_utc: str) -> pd.DataFrame:
    """Build a minimal merged DataFrame with 5-min UTC index."""
    idx = pd.date_range(start_utc, end_utc, freq="5min", tz="UTC", inclusive="left")
    df = pd.DataFrame({"rt_lmp": 1.0}, index=idx)
    return df


class TestBuildDayListCTAlignment:
    def test_jan26_ct_starts_at_ct_midnight(self):
        """CT Jan 26 day must start at 2026-01-26 00:00 CT = 2026-01-26 06:00 UTC."""
        # Build enough data for a full CT Jan 26 + lookback
        # SEQ_LEN=32 intervals lookback before day start
        # CT Jan 26 00:00 = UTC Jan 26 06:00
        # We need at least SEQ_LEN rows before that UTC timestamp
        lookback_start_utc = "2026-01-25 22:00:00"  # plenty of lookback before UTC 06:00
        end_utc = "2026-01-27 06:00:00"              # through CT Jan 26 midnight
        merged = _make_merged(lookback_start_utc, end_utc)

        days = build_day_list(merged, "2026-01-26", "2026-01-26")

        assert len(days) == 1, f"Expected exactly 1 day, got {len(days)}: {days}"
        date_str, first_idx = days[0]
        assert date_str == "2026-01-26", f"date_str should be '2026-01-26', got '{date_str}'"

        # The first_idx must correspond to UTC 2026-01-26 06:00:00 (= CT midnight Jan 26)
        actual_utc_ts = merged.index[first_idx]
        expected_utc_ts = pd.Timestamp("2026-01-26 06:00:00", tz="UTC")
        assert actual_utc_ts == expected_utc_ts, (
            f"CT Jan 26 day should start at UTC 06:00, but got {actual_utc_ts}"
        )

    def test_jan26_not_starting_at_utc_midnight(self):
        """CT Jan 26 must NOT start at UTC midnight (which is CT Jan 25 18:00)."""
        lookback_start_utc = "2026-01-25 22:00:00"
        end_utc = "2026-01-27 06:00:00"
        merged = _make_merged(lookback_start_utc, end_utc)

        days = build_day_list(merged, "2026-01-26", "2026-01-26")

        assert len(days) == 1
        _, first_idx = days[0]

        utc_wrong = pd.Timestamp("2026-01-26 00:00:00", tz="UTC")  # = CT Jan 25 18:00
        assert merged.index[first_idx] != utc_wrong, (
            "CT day must not start at UTC midnight (that is CT Jan 25 18:00, not Jan 26 00:00)"
        )

    def test_day_has_288_intervals(self):
        """A CT day in the result must cover exactly STEPS_PER_DAY=288 intervals."""
        lookback_start_utc = "2026-01-25 22:00:00"
        end_utc = "2026-01-27 07:00:00"
        merged = _make_merged(lookback_start_utc, end_utc)

        days = build_day_list(merged, "2026-01-26", "2026-01-26")

        assert len(days) == 1
        date_str, first_idx = days[0]

        # Count how many rows in merged.index fall on CT Jan 26
        ts_ct = merged.index.tz_convert("US/Central")
        ct_dates = np.array([ts.date() for ts in ts_ct])
        import datetime
        n_rows = int((ct_dates == datetime.date(2026, 1, 26)).sum())
        assert n_rows == STEPS_PER_DAY, (
            f"CT Jan 26 should have {STEPS_PER_DAY} rows, got {n_rows}"
        )

    def test_fern_spike_on_ct_jan25(self):
        """The $938 spike interval (UTC 2026-01-26 00:00 = CT Jan 25 18:00) belongs to CT Jan 25."""
        lookback_start_utc = "2026-01-24 22:00:00"
        end_utc = "2026-01-26 06:10:00"  # just past CT Jan 25 midnight
        merged = _make_merged(lookback_start_utc, end_utc)

        # Mark the spike interval
        spike_utc = pd.Timestamp("2026-01-26 00:00:00", tz="UTC")  # CT Jan 25 18:00
        merged.loc[spike_utc, "rt_lmp"] = 938.06

        days = build_day_list(merged, "2026-01-25", "2026-01-25")

        assert len(days) == 1
        date_str, first_idx = days[0]
        assert date_str == "2026-01-25"

        # The spike interval must fall within the CT Jan 25 day slice
        ts_ct = merged.index.tz_convert("US/Central")
        import datetime
        jan25_mask = np.array([ts.date() for ts in ts_ct]) == datetime.date(2026, 1, 25)
        jan25_lmp = merged["rt_lmp"].values[jan25_mask]
        assert 938.06 in jan25_lmp, "Spike interval should be in CT Jan 25 day slice"

    def test_returns_ct_date_string_format(self):
        """date_str in output must be YYYY-MM-DD matching the CT operating day."""
        lookback_start_utc = "2026-01-09 22:00:00"
        end_utc = "2026-01-11 06:05:00"
        merged = _make_merged(lookback_start_utc, end_utc)

        days = build_day_list(merged, "2026-01-10", "2026-01-10")

        assert len(days) == 1
        date_str, _ = days[0]
        assert date_str == "2026-01-10"
        # Verify format
        import datetime
        parsed = datetime.date.fromisoformat(date_str)
        assert parsed == datetime.date(2026, 1, 10)

    def test_excludes_days_outside_range(self):
        """build_day_list must respect the [start, end] date filter."""
        lookback_start_utc = "2026-01-08 22:00:00"
        end_utc = "2026-01-13 06:05:00"
        merged = _make_merged(lookback_start_utc, end_utc)

        days = build_day_list(merged, "2026-01-10", "2026-01-11")
        date_strs = {d for d, _ in days}
        assert "2026-01-09" not in date_strs
        assert "2026-01-12" not in date_strs
        assert "2026-01-10" in date_strs
        assert "2026-01-11" in date_strs

    def test_requires_sufficient_lookback(self):
        """Days whose first_idx < SEQ_LEN must be dropped."""
        # Start data only 5 rows before CT day start — not enough for SEQ_LEN=32 lookback
        lookback_start_utc = "2026-01-10 05:35:00"  # only 5 rows before UTC 06:00 (CT midnight)
        end_utc = "2026-01-11 06:05:00"
        merged = _make_merged(lookback_start_utc, end_utc)

        days = build_day_list(merged, "2026-01-10", "2026-01-10")
        assert len(days) == 0, "Day with insufficient lookback should be excluded"
