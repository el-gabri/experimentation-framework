"""Calendar validation must not equate row counts with daily coverage."""

import numpy as np
import pandas as pd
import pytest

from supply_experiments.panel import CityPanel, ExperimentWindow


def _frame(index):
    return pd.DataFrame({name: np.arange(len(index)) + 100.0 for name in ("T", "A", "B")},
                        index=index)


@pytest.mark.parametrize("fill_value", [None, 0.0])
def test_intraday_duplicate_cannot_replace_missing_day(fill_value):
    index = pd.date_range("2025-01-01", periods=12).tolist()
    index[2] = pd.Timestamp("2025-01-02 12:00")
    with pytest.raises(ValueError, match="uma observação por dia"):
        CityPanel(_frame(index), fill_value=fill_value)


def test_inconsistent_local_times_are_rejected_even_with_fill():
    index = pd.date_range("2025-01-01", periods=8).tolist()
    index[2] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="horário local consistente"):
        CityPanel(_frame(index), fill_value=0.0)


@pytest.mark.parametrize("time", ["00:00", "12:00"])
def test_daily_local_times_and_dst_preserve_dates_and_anticipation(time):
    index = pd.date_range(f"2025-03-04 {time}", periods=12, tz="America/New_York")
    panel = CityPanel(_frame(index))
    window = ExperimentWindow(index[8].date(), index[-1].date(), 5, anticipation_days=2)
    sample = panel.slice_for(["T"], ["A", "B"], window)
    pd.testing.assert_index_equal(panel.index, index)
    np.testing.assert_array_equal(sample.y_pre, panel.outcome["T"].iloc[1:6])
    assert len(sample.y_post) == 4


def test_explicit_fill_adds_only_missing_dates():
    index = pd.date_range("2025-01-01", periods=8)
    observed = _frame(index).drop(index[2])
    with pytest.raises(ValueError, match="ausentes"):
        CityPanel(observed)
    with pytest.warns(UserWarning, match="1 dia"):
        panel = CityPanel(observed, fill_value=0.0)
    assert (panel.outcome.loc[index[2]] == 0.0).all()
    pd.testing.assert_frame_equal(panel.outcome.loc[observed.index], observed)
    observed.iloc[0, 0] = np.nan
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="NaN ou infinito"):
        CityPanel(observed, fill_value=0.0)


@pytest.mark.parametrize("role", ["numerators", "denominators"])
def test_ratio_frames_reject_intraday_duplicates_before_alignment(role):
    index = pd.date_range("2025-01-01", periods=8)
    broken = index.tolist()
    broken[2] = pd.Timestamp("2025-01-02 12:00")
    kwargs = {"numerators": {"rate": _frame(index)}, "denominators": {"rate": _frame(index)}}
    kwargs[role] = {"rate": _frame(broken)}
    with pytest.raises(ValueError, match="uma observação por dia"):
        CityPanel(_frame(index), fill_value=0.0, **kwargs)


def test_ratio_alignment_does_not_replace_unknown_values_or_shift_clock():
    index = pd.date_range("2025-01-01", periods=8)
    numerator = _frame(index).drop(index[2])
    denominator = _frame(index)
    with pytest.warns(UserWarning):
        panel = CityPanel(_frame(index), {"rate": numerator}, {"rate": denominator}, 0.0)
    assert (panel.numerators["rate"].loc[index[2]] == 0.0).all()
    numerator.iloc[0, 0] = np.nan
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="NaN ou infinito"):
        CityPanel(_frame(index), {"rate": numerator}, {"rate": denominator}, 0.0)
    shifted = _frame(index + pd.Timedelta(hours=12))
    with pytest.raises(ValueError, match="timezone e horário"):
        CityPanel(_frame(index), {"rate": shifted}, {"rate": denominator}, 0.0)
