"""Equation-level regression tests for estimator and inference validity guards."""

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from supply_experiments.estimators.ascm import fit_ascm
from supply_experiments.estimators.did import DiDFit
from supply_experiments.estimators.scm import _simplex_ols, fit_scm
from supply_experiments.estimators.sdid import (
    _solve_simplex_intercept_checked,
    fit_sdid,
)
from supply_experiments.inference.bootstrap import (
    UnsupportedFewTreatedClustersError,
    wild_cluster_bootstrap,
)
from supply_experiments.inference.conformal import conformal_inference
from supply_experiments.inference.permutation import placebo_inference


def _toy_trajectories(seed=123, t_pre=14, t_post=5, n_donors=6):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(t_pre + t_post, 2)).cumsum(axis=0)
    loadings = rng.normal(size=(2, n_donors))
    donors = 100.0 + factors @ loadings + rng.normal(scale=0.2, size=(t_pre + t_post, n_donors))
    true_weights = np.arange(1, n_donors + 1, dtype=float)
    true_weights /= true_weights.sum()
    treated = donors @ true_weights + rng.normal(scale=0.1, size=t_pre + t_post)
    return (
        treated[:t_pre], donors[:t_pre], treated[t_pre:], donors[t_pre:],
        [f"d{i}" for i in range(n_donors)],
    )


def test_scm_solver_reports_convergence_and_iteration_exhaustion():
    args = _toy_trajectories()
    fit = fit_scm(*args)
    assert fit.success
    assert fit.extras["solver_status"] == "converged"
    assert fit.extras["solver_iterations"] >= 1
    assert np.isfinite(fit.extras["solver_objective"])

    rng = np.random.default_rng(0)
    y = rng.normal(size=30)
    X = rng.normal(size=(30, 5))
    _, _, converged, diagnostics = _simplex_ols(
        y, X, max_iter=1, tol=1e-15,
    )
    assert not converged
    assert diagnostics["solver_status"] == "max_iter_reached"


def test_scm_solver_uses_scale_aware_kkt_certificate():
    rng = np.random.default_rng(187)
    X = rng.lognormal(mean=8.0, sigma=2.0, size=(30, 20))
    true_weights = rng.dirichlet(np.ones(20))
    y = X @ true_weights
    y += rng.normal(scale=np.std(y) * 0.01, size=len(y))

    weights, _, converged, diagnostics = _simplex_ols(y, X)
    scale = float(np.std(y)) or 1.0
    ys, Xs = y / scale, X / scale
    residual = ys - Xs @ weights
    gradient = -2.0 * Xs.T @ residual
    fw_gap = float(gradient @ weights - np.min(gradient))

    assert converged
    assert np.isclose(weights.sum(), 1.0)
    assert weights.min() >= -1e-12
    assert fw_gap <= diagnostics["solver_gap_tolerance"]
    assert np.isclose(fw_gap, diagnostics["solver_fw_gap"], atol=1e-10)
    assert float(residual @ residual) < 0.01


def test_scm_family_fails_closed_on_nonfinite_or_misaligned_panels():
    y_pre, Y_pre, y_post, Y_post, names = _toy_trajectories()
    bad_scm = fit_scm(y_pre, Y_pre, y_post, Y_post[:, :-1], names)
    assert not bad_scm.success

    Y_pre_nan = Y_pre.copy()
    Y_pre_nan[0, 0] = np.nan
    bad_ascm = fit_ascm(y_pre, Y_pre_nan, y_post, Y_post, names)
    assert not bad_ascm.success

    with pytest.raises(ValueError, match="min_weight"):
        fit_scm(y_pre, Y_pre, y_post, Y_post, names, min_weight=-0.1)


def test_ascm_lambda_uses_only_pre_period_and_exposes_effective_weights():
    y_pre, Y_pre, y_post, Y_post, names = _toy_trajectories(seed=321)
    fit_a = fit_ascm(y_pre, Y_pre, y_post, Y_post, names)
    fit_b = fit_ascm(
        y_pre,
        Y_pre,
        y_post + 500.0,
        Y_post[:, ::-1] * 7.0,
        names,
    )
    assert fit_a.success and fit_b.success
    assert fit_a.extras["lambda_cv_method"] == "leave_one_pre_period_out"
    assert fit_a.extras["lambda"] == fit_b.extras["lambda"]

    effective = np.asarray(fit_a.extras["effective_weights"])
    assert np.isclose(effective.sum(), 1.0)
    assert np.allclose(fit_a.y_synth_post, Y_post @ effective)
    assert np.isfinite(fit_a.extras["augmented_pre_rmspe"])
    assert np.isclose(
        fit_a.extras["extrapolation_l2"], np.linalg.norm(effective - fit_a.w),
    )


def test_sdid_mean_estimand_and_algorithm_one_time_penalty():
    y_mean_pre, Y_pre, y_mean_post, Y_post, names = _toy_trajectories(seed=77)
    mean_fit = fit_sdid(
        y_mean_pre, Y_pre, y_mean_post, Y_post, names,
        n_treated_units=2, treated_aggregation="mean",
    )
    legacy_sum_fit = fit_sdid(
        2.0 * y_mean_pre, Y_pre, 2.0 * y_mean_post, Y_post, names,
        n_treated_units=2, treated_aggregation="sum",
    )
    assert mean_fit.success and legacy_sum_fit.success
    assert np.allclose(mean_fit.w, legacy_sum_fit.w)
    assert np.allclose(
        mean_fit.extras["time_weights"], legacy_sum_fit.extras["time_weights"],
    )
    assert np.isclose(mean_fit.att, legacy_sum_fit.att)

    diffs = np.diff(Y_pre, axis=0)
    sigma = np.sqrt(np.mean((diffs - diffs.mean()) ** 2))
    expected_time_ridge = (1e-6 * sigma) ** 2 * Y_pre.shape[1]
    assert np.isclose(mean_fit.extras["sigma_first_differences"], sigma)
    assert np.isclose(mean_fit.extras["time_ridge"], expected_time_ridge)
    assert mean_fit.extras["unit_solver_status"] == "converged"
    assert mean_fit.extras["time_solver_status"] == "converged"
    assert (
        mean_fit.extras["unit_solver_fw_gap"]
        <= mean_fit.extras["unit_solver_gap_tolerance"]
    )
    assert (
        mean_fit.extras["time_solver_fw_gap"]
        <= mean_fit.extras["time_solver_gap_tolerance"]
    )


def test_sdid_simplex_solver_reports_iteration_exhaustion():
    rng = np.random.default_rng(93)
    A = rng.normal(size=(20, 80))
    b = rng.normal(size=20)
    _, converged, diagnostics = _solve_simplex_intercept_checked(
        A, b, ridge=0.0, max_iter=1,
    )
    assert not converged
    assert diagnostics["solver_status"] == "max_iter_reached"


def test_sdid_fit_fails_closed_when_a_weight_qp_is_uncertified(monkeypatch):
    module = importlib.import_module("supply_experiments.estimators.sdid")
    args = _toy_trajectories(seed=404)

    def failed_solver(A, b, ridge, **kwargs):
        weights = np.full(A.shape[1], 1.0 / A.shape[1])
        return weights, False, {
            "solver_status": "max_iter_reached",
            "solver_fw_gap": 1.0,
            "solver_gap_tolerance": 1e-8,
        }

    monkeypatch.setattr(module, "_solve_simplex_intercept_checked", failed_solver)
    fit = module.fit_sdid(*args)
    assert not fit.success
    assert fit.extras["failure_reason"] == "unit_weight_solver_nonconvergence"
    assert fit.extras["unit_solver_status"] == "max_iter_reached"


def _occasionally_failing_fit(y_pre, Y_pre, y_post, Y_post, names):
    if np.isclose(y_pre[0], 20.0):
        raise RuntimeError("deterministic alternative-assignment failure")
    synth_pre = Y_pre.mean(axis=1)
    synth_post = Y_post.mean(axis=1)
    effects = y_post - synth_post
    return SimpleNamespace(
        success=True,
        y_synth_pre=synth_pre,
        y_synth_post=synth_post,
        att=float(np.mean(effects)),
        att_pct=float(np.mean(effects) / np.mean(synth_post)),
    )


def _permutation_failure_panel():
    y_pre = np.array([10.0, 11.0, 12.0])
    y_post = np.array([13.0, 14.0])
    Y_pre = np.column_stack([
        np.array([20.0, 21.0, 22.0]),
        np.array([30.0, 31.0, 32.0]),
        np.array([40.0, 41.0, 42.0]),
    ])
    Y_post = np.column_stack([
        np.array([23.0, 24.0]),
        np.array([33.0, 34.0]),
        np.array([43.0, 44.0]),
    ])
    return y_pre, Y_pre, y_post, Y_post, ["d0", "d1", "d2"]


def test_randomization_inference_fails_closed_if_any_assignment_fit_fails():
    result = placebo_inference(
        _occasionally_failing_fit,
        *_permutation_failure_panel(),
        assignment_mechanism="randomized",
        max_group_placebos=None,
        min_placebos=1,
    )
    assert not result.valid_for_decision
    assert np.isnan(result.p_value)
    assert np.isnan(result.p_value_att)
    assert result.inference_basis == "incomplete_randomization_reference"
    assert result.n_attempted_assignments == 3
    assert result.n_failed_assignments == 1
    assert result.n_filtered_assignments == 0
    assert result.n_placebos == 2


def test_observational_placebo_failures_remain_explicit_diagnostics():
    result = placebo_inference(
        _occasionally_failing_fit,
        *_permutation_failure_panel(),
        assignment_mechanism="observational",
        calibrated=True,
        max_group_placebos=None,
        min_placebos=1,
    )
    assert result.valid_for_decision
    assert np.isfinite(result.p_value)
    assert result.n_attempted_assignments == 3
    assert result.n_failed_assignments == 1
    assert result.n_filtered_assignments == 0
    assert result.n_placebos == 2

    coarse = placebo_inference(
        _occasionally_failing_fit,
        *_permutation_failure_panel(),
        assignment_mechanism="observational",
        calibrated=True,
        max_group_placebos=None,
        min_placebos=3,
    )
    assert np.isfinite(coarse.p_value)
    assert not coarse.valid_for_decision


def test_conformal_rejects_non_scm_and_validates_alpha_and_grid():
    args = _toy_trajectories(t_pre=12, t_post=4)
    with pytest.raises(ValueError, match="apenas fit_scm"):
        conformal_inference(fit_sdid, *args)
    with pytest.raises(ValueError, match="alpha"):
        conformal_inference(fit_scm, *args, alpha=1.0)
    with pytest.raises(ValueError, match="estritamente crescente"):
        conformal_inference(fit_scm, *args, rel_grid=np.array([0.0, -0.1]))


def test_conformal_reports_disconnected_sets_and_grid_boundaries(monkeypatch):
    module = importlib.import_module("supply_experiments.inference.conformal")
    y_pre, Y_pre, y_post, Y_post, names = _toy_trajectories(t_pre=8, t_post=2)
    grid = np.linspace(-0.5, 0.5, 11)

    def disconnected_pvalue(*call_args, **call_kwargs):
        tau = np.asarray(call_args[6], float)
        relative = float(tau[0] / 100.0)
        eps = 1e-12
        accepted = (-0.4 - eps <= relative <= -0.2 + eps
                    or 0.2 - eps <= relative <= 0.4 + eps)
        return 0.5 if accepted else 0.01

    monkeypatch.setattr(module, "_pvalue_for_tau", disconnected_pvalue)
    result = conformal_inference(
        fit_scm, y_pre, Y_pre, y_post, Y_post, names,
        alpha=0.10, rel_grid=grid, counterfactual_level=100.0,
    )
    assert result.confidence_level == 0.90
    assert result.acceptance_set_disconnected
    assert np.allclose(result.acceptance_intervals_pct, [(-0.4, -0.2), (0.2, 0.4)])
    assert not result.boundary_truncated

    monkeypatch.setattr(module, "_pvalue_for_tau", lambda *args, **kwargs: 0.5)
    boundary = conformal_inference(
        fit_scm, y_pre, Y_pre, y_post, Y_post, names,
        alpha=0.10, rel_grid=grid, counterfactual_level=100.0,
    )
    assert boundary.boundary_truncated
    assert boundary.lower_boundary_truncated and boundary.upper_boundary_truncated


def _fake_did_fit(n_treated, n_clusters=8, periods=5):
    rng = np.random.default_rng(991)
    codes = np.repeat(np.arange(n_clusters), periods)
    Xd = rng.normal(size=(len(codes), 1))
    yd = 0.2 * Xd[:, 0] + rng.normal(scale=0.5, size=len(codes))
    return DiDFit(
        att=0.2,
        att_pct=0.2,
        resid=pd.DataFrame(),
        design_info={
            "Xd": Xd,
            "yd": yd,
            "city_codes": codes,
            "date_codes": np.tile(np.arange(periods), n_clusters),
            "treated_cities": [f"t{i}" for i in range(n_treated)],
        },
        success=True,
    )


def test_wild_cluster_bootstrap_fails_closed_for_few_treated_clusters():
    with pytest.raises(UnsupportedFewTreatedClustersError) as exc_info:
        wild_cluster_bootstrap(_fake_did_fit(n_treated=2), n_boot=19, seed=1)
    assert exc_info.value.n_treated_clusters == 2


def test_wild_cluster_bootstrap_exposes_treated_count_when_guard_passes():
    result = wild_cluster_bootstrap(_fake_did_fit(n_treated=4), n_boot=19, seed=1)
    assert result.n_treated_clusters == 4
    assert not result.valid_for_inference
    assert "diagnostic only" in result.diagnostic
    assert "not a validity certificate" in result.diagnostic
