"""Focused tests for the immutable design and design-tooling contract."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import supply_experiments.design.power as power_module
import supply_experiments.design.treated_selection as selection_module
import supply_experiments.io.registry as registry_module
from supply_experiments.design.power import PowerResult, _draw_windows, simulate_once
from supply_experiments.design.spec import DecisionRule, DesignSpec
from supply_experiments.design.spillover import spillover_exclusions
from supply_experiments.estimators import fit_scm
from supply_experiments.io.registry import (
    ExperimentRecord,
    validate_experiment_transition,
)
from supply_experiments.panel import CityPanel


def _panel(n_cities: int = 12, n_days: int = 120) -> CityPanel:
    index = pd.date_range("2025-01-01", periods=n_days, freq="D")
    trend = np.linspace(100.0, 140.0, n_days)
    outcome = pd.DataFrame(
        {f"CITY_{j:02d}": trend * (1.0 + 0.02 * j) for j in range(n_cities)},
        index=index,
    )
    return CityPanel(outcome=outcome)


def _spec(**overrides) -> DesignSpec:
    values = {
        "experiment_id": "exp-1",
        "treated_units": ("T1",),
        "donor_units": tuple(f"C{i}" for i in range(9)),
        "estimator_names": ("scm",),
        "pre_days": 30,
        "post_days": 11,
        "start_date": date(2027, 1, 10),
        "end_date": date(2027, 1, 20),
        "effect_grid": (-0.05, 0.0, 0.05),
        "assignment_mechanism": "randomized",
        "eligible_units": ("T1",) + tuple(f"C{i}" for i in range(9)),
        "anticipation_days": 3,
        "calibration_fingerprint": "a" * 64,
        "estimator_config_json": json.dumps({"scm": {"ridge": 0.1}}),
    }
    values.update(overrides)
    return DesignSpec(**values)


def test_design_spec_round_trip_is_immutable_and_fingerprint_is_stable():
    spec = _spec()
    restored = DesignSpec.from_json(spec.to_json())

    assert restored == spec
    assert restored.schema_version == "2.0"
    assert restored.window_spacing_days == 1
    assert restored.fingerprint == spec.fingerprint
    assert restored.estimand == "average_treated_relative_att"
    assert json.loads(restored.estimator_config_json) == {"scm": {"ridge": 0.1}}
    with pytest.raises(FrozenInstanceError):
        spec.alpha = 0.05

    # Artifact digest is downstream; calibration policy is part of the design.
    assert replace(spec, calibration_fingerprint="b" * 64).fingerprint == spec.fingerprint
    assert replace(spec, calibration_scope="estimator_only").fingerprint != spec.fingerprint
    assert replace(spec, calibration_max_invalid_rate=0.10).fingerprint != spec.fingerprint
    assert replace(spec, power_target=0.90).fingerprint != spec.fingerprint
    assert replace(spec, power_max_invalid_rate=0.10).fingerprint != spec.fingerprint


def test_authoritative_sdid_config_cannot_change_the_mean_estimand():
    with pytest.raises(ValueError, match="treated_aggregation='mean'"):
        _spec(
            estimator_names=("sdid",),
            estimator_config_json='{"sdid":{"treated_aggregation":"sum"}}',
        )
    with pytest.raises(ValueError, match="n_treated_units"):
        _spec(
            estimator_names=("sdid",),
            estimator_config_json='{"sdid":{"n_treated_units":2}}',
        )

    valid = _spec(
        estimator_names=("sdid",),
        estimator_config_json=(
            '{"sdid":{"n_treated_units":1,"treated_aggregation":"mean"}}'
        ),
    )
    assert json.loads(valid.estimator_config_json)["sdid"] == {
        "n_treated_units": 1,
        "treated_aggregation": "mean",
    }


def test_design_spec_rejects_date_and_permutation_resolution_mismatches():
    with pytest.raises(ValueError, match="post_days"):
        _spec(end_date=date(2027, 1, 21))
    with pytest.raises(ValueError, match="resolução de permutação"):
        _spec(donor_units=tuple(f"C{i}" for i in range(8)))
    with pytest.raises(ValueError, match="estimand não suportado"):
        _spec(estimand="total_volume_att")
    with pytest.raises(ValueError, match="simulações iid"):
        _spec(window_spacing_days=2)


def test_design_spec_uses_assignment_specific_placebo_resolution_and_ceiling():
    common = {
        "treated_units": ("T1", "T2"),
        "donor_units": ("C1", "C2", "C3"),
        "estimator_names": ("scm",),
        "pre_days": 30,
        "post_days": 10,
        "alpha": 0.10,
        "effect_grid": (0.0,),
        "require_calibration": False,
        "calibration_scope": "none",
    }
    randomized = DesignSpec(**common, assignment_mechanism="randomized")
    assert randomized.max_group_placebos == 30
    with pytest.raises(ValueError, match="placebos observacionais"):
        DesignSpec(**common, assignment_mechanism="observational")

    with pytest.raises(ValueError, match="10.000"):
        DesignSpec(
            treated_units=tuple(f"T{i}" for i in range(5)),
            donor_units=tuple(f"C{i}" for i in range(20)),
            estimator_names=("scm",), pre_days=30, post_days=10,
            alpha=0.10, effect_grid=(0.0,), max_group_placebos=None,
            require_calibration=False, calibration_scope="none",
        )


def test_decision_rule_requires_explicit_decision_validity_by_default():
    rule = DecisionRule(min_rejections=1, methods=("scm",), direction="positive")
    result = {"scm": {"p_value": 0.01, "att_pct": 0.10}}
    assert not rule.evaluate(result, alpha=0.10)
    result["scm"]["valid_for_decision"] = True
    assert rule.evaluate(result, alpha=0.10)


def test_draw_windows_enforces_requested_spacing_and_anticipation_history():
    index = pd.date_range("2025-01-01", periods=200, freq="D")
    starts = _draw_windows(
        index, pre_days=20, post_days=10, n=20,
        rng=np.random.default_rng(7), min_start_spacing_days=10,
        anticipation_days=3,
    )
    assert min(starts) >= 23
    assert all(b - a >= 10 for a, b in zip(starts, starts[1:], strict=False))

    iid_starts = _draw_windows(
        pd.date_range("2025-01-01", periods=31, freq="D"),
        pre_days=20, post_days=10, n=20,
        rng=np.random.default_rng(11), min_start_spacing_days=1,
    )
    assert len(iid_starts) == 20
    assert len(set(iid_starts)) < len(iid_starts)


def test_simulate_once_returns_real_att_and_marks_coarse_rank_invalid(monkeypatch):
    captured = {}

    def fake_inference(*args, assignment_mechanism=None, **kwargs):
        captured["pre"] = np.asarray(args[1])
        captured["assignment"] = assignment_mechanism
        return SimpleNamespace(
            p_value=0.05,
            placebo_atts=[],
            n_placebos=8,
            real_fit=SimpleNamespace(att_pct=0.12),
            valid_for_decision=True,
        )

    monkeypatch.setattr(power_module, "placebo_inference", fake_inference)
    result = simulate_once(
        _panel(), ["CITY_00"], [f"CITY_{j:02d}" for j in range(1, 10)],
        lambda *args, **kwargs: None, start_idx=50, pre_days=20,
        post_days=10, delta=0.0, alpha=0.10, anticipation_days=4,
    )

    assert len(captured["pre"]) == 20
    assert captured["assignment"] == "observational"
    assert result["att_pct"] == pytest.approx(0.12)
    assert not result["valid_for_decision"]
    assert not result["reject"]


def test_power_reports_signed_crossing_magnitude_uncertainty_and_invalid_rate(
    monkeypatch,
):
    calls = {}

    def fake_decision(*args):
        delta = float(args[7])
        calls[delta] = calls.get(delta, 0) + 1
        valid = calls[delta] % 2 == 1
        reject = valid and delta <= -0.05
        return {
            "delta": delta,
            "start_idx": int(args[4]),
            "p_value": 0.05 if reject else 0.50,
            "reject": reject,
            "valid": valid,
            "method_results": {},
            "n_placebos": 9,
        }

    def dummy(*args, **kwargs):
        return None

    monkeypatch.setattr(power_module, "_decision_simulation", fake_decision)
    result = power_module.power_analysis(
        _panel(), ["CITY_00"], [f"CITY_{j:02d}" for j in range(1, 11)],
        dummy, pre_days=30, post_days=10,
        effect_grid=(-0.10, -0.05, 0.0), n_sims_per_point=4,
        decision_rule=DecisionRule(direction="negative"),
    )

    assert result.mde_effect == pytest.approx(-0.05)
    assert result.mde_80 == pytest.approx(0.05)
    assert result.invalid_rate[-0.05] == pytest.approx(0.5)
    assert result.n_valid[-0.05] == 2
    assert result.power_ci[-0.05][0] < result.power[-0.05] <= result.power_ci[-0.05][1]
    assert not result.valid_for_approval
    assert any("invalid_rate" in reason for reason in result.failure_reasons)


def test_power_requires_and_replays_nonfixed_market_selection(monkeypatch):
    panel = _panel()
    treated = ("CITY_00",)
    donors = tuple(f"CITY_{j:02d}" for j in range(1, 10))
    eligible_universe = tuple(panel.cities[:-1])
    rule = DecisionRule(methods=("scm",))
    spec = DesignSpec(
        treated_units=treated, donor_units=donors, estimator_names=("scm",),
        pre_days=30, post_days=10, effect_grid=(0.0, 0.05),
        decision_rule=rule, selection_procedure_id="selector-v1",
        eligible_units=eligible_universe,
        selector_config_json='{"top_k":8}',
        require_calibration=False, calibration_scope="none",
        power_n_sims_per_point=2, power_min_valid_sims_per_point=2,
    )

    def dummy(*args, **kwargs):
        return None

    with pytest.raises(ValueError, match="exige selection_fn"):
        power_module.power_analysis(
            panel, treated, donors, fit_scm, pre_days=30, post_days=10,
            effect_grid=(0.0, 0.05), n_sims_per_point=2,
            fit_fns={"scm": fit_scm}, decision_rule=rule, design_spec=spec,
        )

    seen = []

    def selector(panel_arg, eligible, rng, run, start):
        seen.append((run, start))
        return list(treated), list(donors)

    def fake_decision(*args):
        return {
            "delta": float(args[7]), "start_idx": int(args[4]),
            "p_value": 0.5, "reject": False, "valid": True,
            "method_results": {}, "n_placebos": 9,
        }

    monkeypatch.setattr(power_module, "_decision_simulation", fake_decision)
    def wrapped_scm(*args, **kwargs):
        return fit_scm(*args, **kwargs)

    with pytest.raises(ValueError, match="callables exatos"):
        power_module.power_analysis(
            panel, treated, donors, wrapped_scm, pre_days=30, post_days=10,
            effect_grid=(0.0, 0.05), n_sims_per_point=2,
            fit_fns={"scm": wrapped_scm}, decision_rule=rule,
            design_spec=spec, selection_fn=selector,
            eligible=eligible_universe, selection_procedure_id="selector-v1",
        )
    with pytest.raises(ValueError, match="eligible_units"):
        power_module.power_analysis(
            panel, treated, donors, fit_scm, pre_days=30, post_days=10,
            effect_grid=(0.0, 0.05), n_sims_per_point=2,
            fit_fns={"scm": fit_scm}, decision_rule=rule, design_spec=spec,
            selection_fn=selector, eligible=panel.cities[:-2],
            selection_procedure_id="selector-v1",
        )
    def out_of_universe_selector(*args):
        return [panel.cities[-1]], list(donors)

    with pytest.raises(ValueError, match="fora de eligible"):
        power_module.power_analysis(
            panel, treated, donors, fit_scm, pre_days=30, post_days=10,
            effect_grid=(0.0, 0.05), n_sims_per_point=2,
            fit_fns={"scm": fit_scm}, decision_rule=rule, design_spec=spec,
            selection_fn=out_of_universe_selector,
            eligible=eligible_universe,
            selection_procedure_id="selector-v1",
        )
    result = power_module.power_analysis(
        panel, treated, donors, fit_scm, pre_days=30, post_days=10,
        effect_grid=(0.0, 0.05), n_sims_per_point=2,
        fit_fns={"scm": fit_scm}, decision_rule=rule, design_spec=spec,
        selection_fn=selector, eligible=eligible_universe,
        selection_procedure_id="selector-v1",
    )
    assert len(seen) == 2
    assert result.selection_scope == "replayed_selection"
    assert result.selection_procedure_id == "selector-v1"
    assert result.eligible_units == tuple(sorted(eligible_universe))


def test_joint_power_estimators_receive_the_same_injected_effect_path(monkeypatch):
    seen = {}

    def first(*args, **kwargs):
        return None

    def second(*args, **kwargs):
        return None

    def fake_inference(fit_fn, y_pre, donor_pre, y_post, *args, **kwargs):
        seen[fit_fn.__name__] = np.asarray(y_post).copy()
        return SimpleNamespace(
            p_value=0.5, placebo_atts=[0.0] * 9, n_placebos=9,
            real_fit=SimpleNamespace(att_pct=0.05, pre_rmspe=0.01),
        )

    monkeypatch.setattr(power_module, "placebo_inference", fake_inference)
    power_module._decision_simulation(
        _panel(), ["CITY_00"], [f"CITY_{j:02d}" for j in range(1, 10)],
        {"first": first, "second": second}, 50, 20, 10, 0.05, 0.10,
        DecisionRule(methods=("first", "second")), None,
        {"first": {}, "second": {}}, np.random.default_rng(3), 30, 123,
        0, "randomized", 0.10,
    )
    np.testing.assert_allclose(seen["first"], seen["second"])


def test_spillover_protection_fails_closed_on_missing_coordinates():
    with pytest.raises(ValueError, match="cobertura espacial incompleta"):
        spillover_exclusions(
            ["A"], ["B"], {"A": (0.0, 0.0)},
            require_complete_coverage=True,
        )


def test_treated_selection_uses_requested_power_fit_and_train_only_screening(
    monkeypatch,
):
    panel = _panel(n_cities=12, n_days=120)
    stats = pd.DataFrame({"sum_gmv": panel.outcome.sum()})
    correlation_lengths = []
    captured = {}

    def custom_power_fit(*args, **kwargs):
        return None

    def fake_sort(y, panel, donors, sl):
        correlation_lengths.append((len(y), sl.stop - sl.start))
        return list(donors)

    def fake_power(*args, **kwargs):
        captured["fit_fns"] = kwargs["fit_fns"]
        spec = kwargs["design_spec"]
        return PowerResult(
            effect_grid=[0.0, 0.10], power={0.0: 0.0, 0.10: 1.0},
            fpr=0.0, mde_80=0.10, n_sims_per_point=2, alpha=0.10,
            power_ci={0.0: (0.0, 0.5), 0.10: (0.5, 1.0)},
            invalid_rate={0.0: 0.0, 0.10: 0.0},
            design_fingerprint=spec.fingerprint if spec else "exploratory",
        )

    monkeypatch.setattr(selection_module, "_sort_donors_by_corr", fake_sort)
    monkeypatch.setattr(selection_module, "_holdout_rmspe", lambda *a, **k: 0.01)
    monkeypatch.setattr(selection_module, "power_analysis", fake_power)

    recommendations = selection_module.recommend_treated_sets(
        panel, stats, panel.cities, custom_power_fit,
        pre_days=60, post_days=10, n_treated=(1,), min_donors=9,
        top_k_cities=2, top_sets=1, effect_grid=(0.0, 0.10),
        n_sims_per_point=2, max_gmv_share_per_city=1.0,
    )

    assert recommendations
    assert custom_power_fit in captured["fit_fns"].values()
    assert captured.get("design_spec") is None
    assert correlation_lengths
    assert all(length == train_length == 42
               for length, train_length in correlation_lengths)
    assert recommendations[0].design_fingerprint
    assert recommendations[0].power_scope == "conditional_screening_only"
    candidate = DesignSpec.from_json(recommendations[0].design_spec_json)
    assert candidate.selection_procedure_id == "recommend_treated_sets-v1"
    assert recommendations[0].power_design_fingerprint != candidate.fingerprint


def _record() -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id="exp-1", name="test", hypothesis="positive lift",
        status="approved", design_method="scm", primary_kpi="outcome",
        guardrail_kpis=[], start_date=date(2027, 1, 10),
        end_date=date(2027, 1, 20), pre_window_days=30,
        treated_units=[{"city_norm": "T1"}],
        control_units=[{"city_norm": f"C{i}"} for i in range(9)],
        mde_estimated=0.05, expected_effect=0.05,
        decision_rule="serialized DesignSpec rule is authoritative",
    )


def _valid_power(spec: DesignSpec) -> PowerResult:
    n = spec.power_n_sims_per_point
    power = {
        effect: (0.0 if np.isclose(effect, 0.0) else 1.0)
        for effect in spec.effect_grid
    }
    rejections = {
        effect: int(round(power[effect] * n)) for effect in spec.effect_grid
    }
    intervals = {
        effect: power_module._wilson_interval(rejections[effect], n)
        for effect in spec.effect_grid
    }
    candidates = [effect for effect in spec.effect_grid if not np.isclose(effect, 0.0)]
    if spec.decision_rule.direction == "positive":
        candidates = [effect for effect in candidates if effect > 0]
    elif spec.decision_rule.direction == "negative":
        candidates = [effect for effect in candidates if effect < 0]
    mde_effect = min(candidates, key=lambda effect: (abs(effect), effect))
    return PowerResult(
        effect_grid=list(spec.effect_grid), power=power, fpr=power[0.0],
        mde_80=abs(mde_effect), mde_effect=mde_effect, n_sims_per_point=n,
        requested_sims_per_point=n,
        min_valid_sims_per_point=spec.power_min_valid_sims_per_point,
        max_invalid_rate=spec.power_max_invalid_rate,
        effect_model=spec.power_effect_model,
        alpha=spec.alpha,
        power_ci=intervals, fpr_ci=intervals[0.0],
        invalid_rate={effect: 0.0 for effect in spec.effect_grid},
        n_valid={effect: n for effect in spec.effect_grid},
        n_rejections=rejections,
        target_power=spec.power_target,
        design_fingerprint=spec.fingerprint,
        window_spacing_days=spec.window_spacing_days,
        selection_procedure_id=spec.selection_procedure_id,
        eligible_units=spec.eligible_units,
        selector_config_json=spec.selector_config_json,
        assignment_mechanism=spec.assignment_mechanism,
        authoritative=True,
    )


def test_registry_binds_design_calibration_and_immutable_transitions():
    spec = _spec()
    power = _valid_power(spec)
    bound = _record().with_design_contract(spec, power_result=power)
    assert bound.validate_for_approval(alpha=0.10) == []
    restored_spec = DesignSpec.from_json(bound.design_spec_json)
    assert restored_spec.calibration_fingerprint == "a" * 64
    assert PowerResult.from_json(bound.power_result_json).computed_fingerprint == (
        bound.power_fingerprint
    )

    running = replace(bound, status="running")
    assert validate_experiment_transition(bound, running) == []
    changed = replace(running, hypothesis="changed after seeing data")
    problems = validate_experiment_transition(bound, changed)
    assert any("imutáveis" in problem and "hypothesis" in problem for problem in problems)

    tampered = replace(bound, design_fingerprint="b" * 64)
    assert any("não corresponde" in p for p in tampered.validate_for_approval())


def test_registry_embeds_supplied_calibration_digest_and_rejects_weak_power():
    draft_spec = _spec(calibration_fingerprint=None)
    power = _valid_power(draft_spec)
    bound = _record().with_design_contract(
        draft_spec, calibration_fingerprint="c" * 64, power_result=power,
    )
    assert DesignSpec.from_json(bound.design_spec_json).calibration_fingerprint == "c" * 64

    weak_power = replace(
        power,
        n_valid={effect: draft_spec.power_min_valid_sims_per_point - 1
                 for effect in draft_spec.effect_grid},
        invalid_rate={
            effect: 1.0 - (draft_spec.power_min_valid_sims_per_point - 1)
            / draft_spec.power_n_sims_per_point
            for effect in draft_spec.effect_grid
        },
    )
    weak = _record().with_design_contract(
        draft_spec, calibration_fingerprint="c" * 64, power_result=weak_power,
    )
    assert any("simulações válidas" in p for p in weak.validate_for_approval())


def test_registry_rejects_internally_inconsistent_power_artifacts():
    spec = _spec()
    power = _valid_power(spec)
    inconsistent_fpr = replace(power, fpr=0.5)
    bound = _record().with_design_contract(
        spec, power_result=inconsistent_fpr,
    )
    assert any("fpr diverge" in p for p in bound.validate_for_approval())

    impossible_curve = dict(power.power)
    impossible_curve[-0.05] = 0.83
    impossible = _record().with_design_contract(
        spec, power_result=replace(power, power=impossible_curve),
    )
    assert any(
        "n_rejections/n_valid" in p
        for p in impossible.validate_for_approval()
    )

    with pytest.raises(ValueError, match="MDE não corresponde"):
        _record().with_design_contract(
            spec, power_result=replace(power, mde_effect=0.05),
        )
    with pytest.raises(ValueError, match="implementation_version"):
        _record().with_design_contract(
            replace(spec, implementation_version="1.9.0"),
            power_result=power,
        )


def test_registry_legacy_unbound_path_is_explicit():
    legacy = _record()
    assert any("design contract ausente" in p for p in legacy.validate_for_approval())
    assert legacy.validate_for_approval(allow_legacy_unbound=True) == []


def test_registry_rejects_approval_on_start_date():
    today = date.today()
    end = today + timedelta(days=10)
    spec = _spec(start_date=today, end_date=end)
    record = replace(_record(), start_date=today, end_date=end)
    bound = record.with_design_contract(spec, power_result=_valid_power(spec))

    with pytest.raises(ValueError, match="em ou após start_date"):
        registry_module.save_experiment(object(), "schema.registry", bound)
