"""Regression tests for release-blocking statistical safeguards."""

from dataclasses import replace
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import supply_experiments.calibration.aa as aa_module
import supply_experiments.reporting as reporting_module
from supply_experiments.calibration.aa import AACalibration, run_aa_calibration
from supply_experiments.design.control_selection import revalidate_fixed_control
from supply_experiments.design.spec import DecisionRule, DesignSpec
from supply_experiments.estimators.scm import fit_scm
from supply_experiments.estimators.sdid import fit_sdid
from supply_experiments.inference.permutation import placebo_inference
from supply_experiments.metrics import ratio_did
from supply_experiments.panel import CityPanel, ExperimentWindow
from supply_experiments.reporting import analyze_experiment
from supply_experiments.synthetic import make_synthetic_panel


def test_multi_treated_default_is_mean_and_has_no_scale_artifact():
    rng = np.random.default_rng(20260824)
    index = pd.date_range("2026-01-01", periods=25)
    t = np.arange(25.0)
    base = 100.0 + 0.4 * t + 2.0 * np.sin(t / 3.0)
    frame = pd.DataFrame(
        {
            "t1": base,
            "t2": base,
            **{f"d{j}": base + rng.normal(0.0, 0.08, 25) for j in range(10)},
        },
        index=index,
    )
    panel = CityPanel(frame)
    window = ExperimentWindow(
        start_date=index[20].date(), end_date=index[-1].date(), pre_window_days=20,
    )
    sliced = panel.slice_for(["t1", "t2"], [f"d{j}" for j in range(10)], window)
    assert np.allclose(sliced.y_pre, base[:20])

    scm = fit_scm(*sliced.as_args())
    sdid = fit_sdid(*sliced.as_args(), n_treated_units=2)
    assert scm.success and abs(scm.att_pct) < 0.005
    assert sdid.success and abs(sdid.att_pct) < 0.005


def test_panel_city_inputs_fail_closed():
    panel = make_synthetic_panel(seed=1, n_cities=8, n_days=100)
    window = ExperimentWindow(
        start_date=panel.index[80].date(), end_date=panel.index[89].date(),
        pre_window_days=50,
    )
    with pytest.raises(ValueError, match="fora do painel"):
        panel.slice_for(["CITY_00", "TYPO"], panel.cities[2:7], window)
    with pytest.raises(ValueError, match="duplicadas"):
        panel.aggregate(["CITY_00", "CITY_00"])
    with pytest.raises(ValueError, match="sobrepõem"):
        panel.slice_for(["CITY_00"], ["CITY_00", "CITY_01"], window)


def test_missing_dates_require_explicit_semantics():
    index = pd.to_datetime(["2026-01-01", "2026-01-03"])
    frame = pd.DataFrame({"a": [1.0, 2.0]}, index=index)
    with pytest.raises(ValueError, match=r"dia\(s\) ausentes"):
        CityPanel(frame)
    with pytest.warns(UserWarning, match="preenchidos"):
        filled = CityPanel(frame, fill_value=0.0)
    assert list(filled.outcome["a"]) == [1.0, 0.0, 2.0]


def test_anticipation_gap_is_excluded_from_pre_period():
    index = pd.date_range("2026-01-01", periods=30)
    window = ExperimentWindow(
        start_date=date(2026, 1, 21), end_date=date(2026, 1, 25),
        pre_window_days=10, anticipation_days=3,
    )
    pre, post = window.masks(index)
    assert index[pre][-1].date() == date(2026, 1, 17)
    assert int(pre.sum()) == 10 and int(post.sum()) == 5


def test_placebo_rank_declares_its_validity_basis():
    panel = make_synthetic_panel(seed=2, n_cities=12, n_days=180)
    treated = ["CITY_00"]
    donors = [c for c in panel.cities if c not in treated]
    y = panel.aggregate(treated).to_numpy(float)
    Y = panel.outcome[donors].to_numpy(float)
    args = (y[40:140], Y[40:140], y[140:170], Y[140:170], donors)

    observational = placebo_inference(fit_scm, *args)
    randomized = placebo_inference(
        fit_scm, *args, assignment_mechanism="randomized",
    )
    assert np.isfinite(observational.p_value)
    assert not observational.valid_for_decision
    assert observational.inference_basis == "observational_placebo_rank"
    assert randomized.valid_for_decision
    assert randomized.inference_basis == "randomization_inference"


def test_randomization_inference_refits_symmetric_assignment_universe():
    donor_counts = []

    def recording_fit(y_pre, Y_pre, y_post, Y_post, names):
        donor_counts.append((Y_pre.shape[1], tuple(names)))
        synth_pre = Y_pre.mean(axis=1)
        synth_post = Y_post.mean(axis=1)
        att = float(np.mean(y_post - synth_post))
        return SimpleNamespace(
            success=True,
            y_synth_pre=synth_pre,
            y_synth_post=synth_post,
            att=att,
            att_pct=att / max(abs(float(np.mean(synth_post))), 1e-12),
        )

    y_pre = np.array([1.0, 2.0, 3.0])
    y_post = np.array([4.0, 5.0])
    donor_pre = np.column_stack([y_pre + offset for offset in (0.1, 0.2, 0.3)])
    donor_post = np.column_stack([y_post + offset for offset in (0.1, 0.2, 0.3)])
    result = placebo_inference(
        recording_fit, y_pre, donor_pre, y_post, donor_post,
        ["d1", "d2", "d3"], min_placebos=1, max_group_placebos=None,
        assignment_mechanism="randomized", treated_names=["t1"],
    )

    assert result.n_placebos == 3
    assert len(donor_counts) == 4  # observed fit plus three alternative assignments
    assert all(count == 3 for count, _ in donor_counts)
    assert any("t1" in names for _, names in donor_counts[1:])


def test_multi_treated_randomization_requires_individual_trajectories():
    panel = make_synthetic_panel(seed=21, n_cities=12, n_days=150)
    treated = panel.cities[:2]
    donors = panel.cities[2:]
    y = panel.outcome[treated].mean(axis=1).to_numpy(float)
    Y = panel.outcome[donors].to_numpy(float)
    with pytest.raises(ValueError, match="Y_treated_pre"):
        placebo_inference(
            fit_scm, y[:100], Y[:100], y[100:130], Y[100:130], donors,
            n_treated_units=2, assignment_mechanism="randomized",
        )


def test_analysis_requires_bound_design_for_decisions_and_supports_interim_only():
    panel = make_synthetic_panel(seed=11, n_cities=12, n_days=180)
    treated = (panel.cities[0],)
    donors = tuple(panel.cities[1:11])
    start = panel.index[140].date()
    end = panel.index[149].date()
    window = ExperimentWindow(start_date=start, end_date=end, pre_window_days=100)
    spec = DesignSpec(
        experiment_id="randomized-smoke",
        treated_units=treated,
        donor_units=donors,
        estimator_names=("scm",),
        pre_days=100,
        post_days=10,
        start_date=start,
        end_date=end,
        primary_kpi="gmv",
        assignment_mechanism="randomized",
        require_calibration=False,
        calibration_scope="none",
        max_group_placebos=None,
        max_pre_rmspe=1.0,
        decision_rule=DecisionRule(methods=("scm",)),
    )

    strict = analyze_experiment(
        panel, "randomized-smoke", treated, donors, window,
        today=end + timedelta(days=1), run_conformal=False,
        max_group_placebos=None, design_spec=spec,
    )
    assert strict.decision_eligible
    assert strict.decision in {"RULE_PASSED", "RULE_NOT_PASSED"}
    assert strict.estimators[0].inference_basis == "randomization_inference"

    legacy = analyze_experiment(
        panel, "legacy", treated, donors, window,
        today=end + timedelta(days=1), run_conformal=False,
        max_group_placebos=None,
    )
    assert not legacy.decision_eligible
    assert legacy.decision == "DIAGNOSTIC_ONLY"

    interim = analyze_experiment(
        panel, "randomized-smoke", treated, donors, window,
        today=start, allow_interim=True, run_conformal=False,
        max_group_placebos=None, design_spec=spec,
    )
    assert interim.interim and not interim.estimators
    assert interim.decision == "INTERIM_MONITORING_ONLY"


def test_analysis_validity_matches_partial_method_decision_rule(monkeypatch):
    panel = make_synthetic_panel(seed=32, n_cities=10, n_days=220)
    treated = ["CITY_00"]
    donors = [city for city in panel.cities if city not in treated][:8]
    start = panel.index[180].date()
    end = panel.index[194].date()
    window = ExperimentWindow(start, end, pre_window_days=90)
    rule = DecisionRule(
        methods=("ascm", "scm"), min_rejections=1, require_all_valid=False
    )
    spec = DesignSpec(
        experiment_id="partial-validity",
        treated_units=tuple(treated), donor_units=tuple(donors),
        estimator_names=("ascm", "scm"), pre_days=90, post_days=15,
        start_date=start, end_date=end, alpha=0.20, max_group_placebos=None,
        assignment_mechanism="randomized", max_pre_rmspe=1.0,
        primary_kpi="gmv", require_calibration=False, calibration_scope="none",
        decision_rule=rule,
        estimator_config_json='{"ascm":{"lambda_grid":[]},"scm":{}}',
    )

    uncalibrated = analyze_experiment(
        panel, "partial-validity", treated, donors, window,
        today=end + timedelta(days=1), run_conformal=False,
        alpha=0.20, max_group_placebos=None, design_spec=spec,
    )
    assert not uncalibrated.decision_eligible
    assert uncalibrated.decision == "DIAGNOSTIC_ONLY"

    monkeypatch.setattr(reporting_module, "_matching_complete_calibration", lambda *a: True)
    report = analyze_experiment(
        panel, "partial-validity", treated, donors, window,
        today=end + timedelta(days=1), run_conformal=False,
        alpha=0.20, max_group_placebos=None, design_spec=spec,
    )

    rows = {row.method: row for row in report.estimators}
    assert not rows["ascm"].valid_for_decision
    assert rows["scm"].valid_for_decision
    assert report.decision_eligible
    assert report.decision in {"RULE_PASSED", "RULE_NOT_PASSED"}


def test_ratio_guardrail_uses_joint_deterministic_block_bootstrap():
    rng = np.random.default_rng(3)
    n = 210
    index = pd.date_range("2025-01-01", periods=n)
    common = rng.normal(0.0, 0.004, n)
    den_t = pd.Series(rng.poisson(3000, n).astype(float), index=index)
    den_c = pd.Series(rng.poisson(3000, n).astype(float), index=index)
    pre = np.arange(n) < 150
    post = ~pre
    rate_c = np.clip(0.05 + common, 0.001, 0.2)
    rate_t = np.clip(rate_c + np.where(post, 0.015, 0.0), 0.001, 0.2)
    num_t = pd.Series(rng.binomial(den_t.astype(int), rate_t), index=index, dtype=float)
    num_c = pd.Series(rng.binomial(den_c.astype(int), rate_c), index=index, dtype=float)

    first = ratio_did(num_t, den_t, num_c, den_c, pre, post, n_boot=399, seed=7)
    second = ratio_did(num_t, den_t, num_c, den_c, pre, post, n_boot=399, seed=7)
    assert first.method == "paired_circular_block_bootstrap"
    assert first.n_boot == 399
    assert first.p_value < 0.05
    assert first.se == second.se and first.p_value == second.p_value


def test_fixed_control_requires_equivalence_and_correlation():
    panel = make_synthetic_panel(seed=14, n_cities=25, n_days=380)
    control = ["CITY_01", "CITY_04", "CITY_07", "CITY_10", "CITY_13"]
    result = revalidate_fixed_control(
        panel, control, window_days=120, equivalence_margin=0.15, min_corr=0.5,
    )
    assert result.passed
    assert result.p_value <= 0.05
    assert result.corr >= 0.5


def test_joint_aa_calibration_fingerprints_the_decision_procedure(monkeypatch):
    panel = make_synthetic_panel(seed=31, n_cities=8, n_days=100)

    def fake_joint(*args, **kwargs):
        return {
            "p_value": 0.5,
            "reject": False,
            "valid": True,
            "method_results": {"scm": {"att_pct": 0.0}},
            "n_placebos": 4,
        }

    monkeypatch.setattr(aa_module, "_decision_simulation", fake_joint)
    fit = fit_scm
    result = run_aa_calibration(
        panel, panel.cities, fit, pre_days=30, post_days=10,
        n_runs=20, n_treated=1, n_donors=4, alpha=0.20,
        min_valid_runs=20, min_ks_p_value=0.0,
        fit_fns={"scm": fit},
        decision_rule=DecisionRule(methods=("scm",)),
        design_fingerprint="d" * 64,
    )
    assert result.passed
    assert result.procedure_scope == "joint_decision"
    assert np.isnan(result.ks_p_value)
    assert result.design_fingerprint == "d" * 64
    assert len(result.calibration_fingerprint) == 64


def test_selected_design_calibration_contract_must_match_every_bound_input(monkeypatch):
    panel = make_synthetic_panel(seed=41, n_cities=8, n_days=100)
    eligible = tuple(panel.cities[:-1])
    treated = (panel.cities[0],)
    donors = tuple(panel.cities[1:5])
    rule = DecisionRule(methods=("scm",))
    spec = DesignSpec(
        treated_units=treated, donor_units=donors, estimator_names=("scm",),
        pre_days=30, post_days=10, alpha=0.20, max_group_placebos=4,
        decision_rule=rule, selection_procedure_id="selector-v1",
        eligible_units=eligible,
        selector_config_json='{"top_k":8}',
        calibration_n_runs=20, calibration_min_valid_runs=20,
        calibration_min_ks_p_value=0.0,
    )

    def fake_joint(*args, **kwargs):
        return {
            "p_value": 0.5, "reject": False, "valid": True,
            "method_results": {"scm": {"att_pct": 0.0}}, "n_placebos": 4,
        }

    def replay_selection(*args):
        return list(treated), list(donors)

    monkeypatch.setattr(aa_module, "_decision_simulation", fake_joint)
    fit = fit_scm
    calibration = run_aa_calibration(
        panel, eligible, fit, pre_days=30, post_days=10,
        n_runs=20, n_treated=1, n_donors=4, alpha=0.20,
        min_valid_runs=20, min_ks_p_value=0.0, max_group_placebos=4,
        fit_fns={"scm": fit}, decision_rule=rule,
        design_fingerprint=spec.fingerprint, selection_fn=replay_selection,
        selection_procedure_id="selector-v1", max_pre_rmspe=0.10,
        design_spec=spec,
    )
    bound = replace(spec, calibration_fingerprint=calibration.calibration_fingerprint)

    assert calibration.matches_design(bound)
    restored = AACalibration.from_json(calibration.to_json())
    assert restored.calibration_fingerprint == calibration.calibration_fingerprint
    assert restored.matches_design(bound)
    tampered = replace(restored, passed=not restored.passed)
    assert not tampered.matches_design(bound)
    changed = replace(bound, selection_procedure_id="selector-v2")
    assert not calibration.matches_design(changed)
    relaxed = replace(bound, calibration_max_fpr_inflation=0.10)
    assert not calibration.matches_design(relaxed)
    changed_universe = replace(bound, eligible_units=tuple(panel.cities[:-2]))
    assert not calibration.matches_design(changed_universe)
    changed_selector_config = replace(bound, selector_config_json='{"top_k":7}')
    assert not calibration.matches_design(changed_selector_config)

    def outside_selector(*args):
        return [panel.cities[-1]], list(donors)

    with pytest.raises(ValueError, match="fora de eligible"):
        run_aa_calibration(
            panel, eligible, fit, pre_days=30, post_days=10,
            n_runs=20, n_treated=1, n_donors=4, alpha=0.20,
            min_valid_runs=20, min_ks_p_value=0.0, max_group_placebos=4,
            fit_fns={"scm": fit}, decision_rule=rule,
            selection_fn=outside_selector,
            selection_procedure_id="selector-v1", max_pre_rmspe=0.10,
            design_spec=spec, seed=spec.calibration_seed,
        )


def test_randomized_aa_rejects_universe_mismatch_before_simulation():
    panel = make_synthetic_panel(seed=47, n_cities=8, n_days=100)
    treated = (panel.cities[0],)
    donors = tuple(panel.cities[1:5])
    rule = DecisionRule(methods=("scm",))
    spec = DesignSpec(
        treated_units=treated, donor_units=donors, estimator_names=("scm",),
        eligible_units=tuple(panel.cities), pre_days=30, post_days=10,
        alpha=0.20, max_group_placebos=4, assignment_mechanism="randomized",
        decision_rule=rule, calibration_n_runs=20,
        calibration_min_valid_runs=20, calibration_min_ks_p_value=0.0,
        seed=123,
    )
    fit = fit_scm

    with pytest.raises(ValueError, match="eligible_units"):
        run_aa_calibration(
            panel, panel.cities[:-1], fit, pre_days=30, post_days=10,
            n_runs=20, n_treated=1, n_donors=4, alpha=0.20,
            min_valid_runs=20, min_ks_p_value=0.0, max_group_placebos=4,
            fit_fns={"scm": fit}, decision_rule=rule,
            assignment_mechanism="randomized", design_spec=spec,
        )
