"""Offline checks for the adapter's observed-data and daily-grid boundary."""

import numpy as np
import pandas as pd
import pytest

from supply_experiments.io.etl import _panel_from_daily


def _daily():
    return pd.DataFrame({
        "order_date": ["2025-01-01", "2025-01-03"], "city_norm": ["A", "A"],
        "gmv": [100., 120.], "orders": [1, 1], "merchants": [1, 1],
        "rupture_orders": [0, 0], "source_rows": [1, 1], "rupture_observed_rows": [1, 1],
        "city_address": ["A", "A"], "state_address": ["SP", "SP"],
    })


def test_missing_city_days_require_explicit_zero_activity_semantics():
    with pytest.raises(ValueError, match="city-days ausentes"):
        _panel_from_daily(_daily(), "2025-01-01", "2025-01-03")
    panel, stats = _panel_from_daily(_daily(), "2025-01-01", "2025-01-03",
                                     assume_missing_city_days_zero=True)
    np.testing.assert_array_equal(panel.outcome.A, [100., 0., 120.])
    assert stats.loc["A", "avg_daily_gmv"] == pytest.approx(220 / 3)
    assert stats.loc["A", "rupture_rate"] == 0


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_invalid_observed_gmv_is_never_zero_filled(value):
    daily = _daily()
    daily.loc[0, "gmv"] = value
    with pytest.raises(ValueError, match="agregados observados"):
        _panel_from_daily(daily, "2025-01-01", "2025-01-03",
                          assume_missing_city_days_zero=True)


@pytest.mark.parametrize("observed", [[0, 0], [0, 1]])
def test_incomplete_optional_rupture_is_unavailable_and_required_rupture_fails(observed):
    daily = _daily()
    daily["rupture_observed_rows"] = observed
    kwargs = dict(assume_missing_city_days_zero=True)
    with pytest.warns(UserWarning, match="indisponível"):
        panel, stats = _panel_from_daily(daily, "2025-01-01", "2025-01-03", **kwargs)
    assert "rupture_rate_order" not in panel.numerators
    assert stats.rupture_rate.isna().all()
    assert not stats.rupture_telemetry_complete.any()
    with pytest.raises(ValueError, match="telemetria de ruptura incompleta"):
        _panel_from_daily(daily, "2025-01-01", "2025-01-03",
                          require_rupture_telemetry=True, **kwargs)


def test_requested_city_cannot_disappear_from_loaded_panel():
    with pytest.raises(ValueError, match="cidades solicitadas ausentes"):
        _panel_from_daily(_daily(), "2025-01-01", "2025-01-03", cities=["A", "MISSING"],
                          assume_missing_city_days_zero=True)
