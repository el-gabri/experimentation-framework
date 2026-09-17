"""Preserve requested DiD groups/windows and refit bootstrap fixed effects."""

import numpy as np
import pandas as pd
import pytest

from supply_experiments.design.control_selection import revalidate_fixed_control
from supply_experiments.estimators.did import fit_did_panel
from supply_experiments.inference.bootstrap import wild_cluster_bootstrap
from supply_experiments.panel import CityPanel, ExperimentWindow
from supply_experiments.synthetic import make_synthetic_panel


def _panel():
    rng = np.random.default_rng(7)
    index = pd.date_range("2025-01-01", periods=30)
    values = 100 + np.arange(30)[:, None] * 0.3 + rng.normal(size=(30, 10))
    values[20:, :4] += 0.4
    panel = CityPanel(pd.DataFrame(values, index=index, columns=[f"c{i}" for i in range(10)]))
    return panel, ExperimentWindow(index[20].date(), index[-1].date(), 20)


@pytest.mark.parametrize("group,extra", [("treated", "MISSING"), ("treated", "c0"),
                                        ("control", "MISSING"), ("control", "c4"),
                                        ("control", "c0")])
def test_did_rejects_changed_membership(group, extra):
    panel, window = _panel()
    treated, control = panel.cities[:4], panel.cities[4:]
    (treated if group == "treated" else control).append(extra)
    with pytest.raises(ValueError):
        fit_did_panel(panel, treated, control, window)


@pytest.mark.parametrize("start,end", [(0, 20), (0, 22), (20, 30), (1, 30)])
def test_did_rejects_incomplete_requested_windows(start, end):
    panel, window = _panel()
    truncated = CityPanel(panel.outcome.iloc[start:end])
    with pytest.raises(ValueError, match="janela .* completa"):
        fit_did_panel(truncated, panel.cities[:4], panel.cities[4:], window)


def test_did_valid_fit_preserves_multi_treated_absolute_effect_scope():
    panel, window = _panel()
    result = fit_did_panel(panel, panel.cities[:4], panel.cities[4:], window)
    assert result.success and np.isfinite(result.att_pct)
    assert np.isnan(result.att)
    assert result.design_info["treated_cities"] == panel.cities[:4]


def test_fixed_control_cannot_pass_missing_duplicate_or_overlapping_members():
    panel = make_synthetic_panel(seed=14, n_cities=25, n_days=380)
    controls = ["CITY_01", "CITY_04", "CITY_07", "CITY_10", "CITY_13"]
    kwargs = dict(window_days=120, equivalence_margin=0.15, min_corr=0.5)
    baseline = revalidate_fixed_control(panel, controls, **kwargs)
    assert baseline.passed
    assert baseline.corr == pytest.approx(0.9606998432349353)
    for names, exclude in [(controls + ["MISSING"], set()),
                           (controls + [controls[0]], set()),
                           (controls, {controls[0]})]:
        invalid = revalidate_fixed_control(panel, names, target_exclude=exclude, **kwargs)
        assert not invalid.passed and invalid.note
    assert not revalidate_fixed_control(
        CityPanel(panel.outcome.iloc[:50]), controls, **kwargs,
    ).passed


@pytest.mark.parametrize("weight_type", ["webb", "rademacher"])
def test_wild_bootstrap_matches_full_dummy_regression(weight_type):
    panel, window = _panel()
    treated, controls = panel.cities[:4], panel.cities[4:]
    fit = fit_did_panel(panel, treated, controls, window, normalize_scale=False)
    result = wild_cluster_bootstrap(fit, n_boot=39, seed=11, weight_type=weight_type)
    long = panel.long_format(treated + controls)
    cities = long.city.astype("category").cat.codes.to_numpy()
    dates = long.date.astype("category").cat.codes.to_numpy()
    y = long.y.to_numpy()
    tp = (long.city.isin(treated) & (long.date.dt.date >= window.start_date)).to_numpy(float)
    nuisance = np.column_stack((np.eye(10)[cities], np.eye(30)[dates, 1:]))
    design = np.column_stack((tp, nuisance))
    inverse = np.linalg.pinv(design.T @ design)

    def refit(response):
        coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
        residuals = response - design @ coefficients
        scores = np.array([design[cities == c].T @ residuals[cities == c] for c in range(10)])
        # Match the package's CR1 convention (one non-absorbed regressor), so
        # this comparison isolates the FE refit instead of differing df choices.
        adjustment = (10 / 9) * ((len(y) - 1) / (len(y) - 1))
        variance = adjustment * inverse @ (scores.T @ scores) @ inverse
        return coefficients[0] / np.sqrt(variance[0, 0])

    observed = refit(y)
    fitted_null = nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
    residual_null = y - fitted_null
    weights = (np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
               if weight_type == "webb" else np.array([-1., 1.]))
    rng = np.random.default_rng(11)
    expected = np.array([refit(fitted_null + residual_null * rng.choice(weights, 10)[cities])
                         for _ in range(39)])
    np.testing.assert_allclose(result.boot_t, expected, atol=1e-9, rtol=1e-9)
    assert result.t_stat == pytest.approx(observed)
    assert result.p_value == (1 + np.sum(np.abs(expected) >= abs(observed))) / 40
    assert not result.valid_for_inference


def test_wild_bootstrap_requires_balanced_date_projection_metadata():
    panel, window = _panel()
    fit = fit_did_panel(panel, panel.cities[:4], panel.cities[4:], window)
    fit.design_info["date_codes"] = np.zeros(len(fit.resid), dtype=int)
    with pytest.raises(ValueError, match="balanceado"):
        wild_cluster_bootstrap(fit, n_boot=1)
