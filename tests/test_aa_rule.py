"""The recorded A/A decision must equal the procedure actually executed."""

import numpy as np
import pytest

from supply_experiments import fit_scm, make_synthetic_panel
from supply_experiments.calibration.aa import run_aa_calibration
from supply_experiments.design.spec import DecisionRule, DesignSpec


@pytest.mark.parametrize("direction", ["positive", "negative", "any"])
def test_single_estimator_aa_executes_rule_and_matches_method_mapping(direction):
    panel = make_synthetic_panel(n_cities=10, n_days=120, seed=7)
    rule = DecisionRule(direction=direction, methods=("scm",))
    spec = DesignSpec(
        treated_units=tuple(panel.cities[:1]), donor_units=tuple(panel.cities[1:]),
        estimator_names=("scm",), pre_days=60, post_days=14,
        decision_rule=rule, assignment_mechanism="randomized",
        eligible_units=tuple(panel.cities), calibration_n_runs=20,
        calibration_min_valid_runs=20, max_pre_rmspe=None,
    )
    kwargs = dict(
        n_runs=20, n_treated=1, n_donors=9, min_valid_runs=20,
        assignment_mechanism="randomized", max_pre_rmspe=None,
        decision_rule=rule, design_spec=spec,
    )
    single = run_aa_calibration(panel, panel.cities, fit_scm, 60, 14, **kwargs)
    mapped = run_aa_calibration(panel, panel.cities, fit_scm, 60, 14,
                                fit_fns={"scm": fit_scm}, **kwargs)
    expected = single.details.apply(
        lambda row: rule.evaluate({"scm": row.to_dict()}, 0.1), axis=1,
    )
    np.testing.assert_array_equal(single.details.reject, expected)
    np.testing.assert_array_equal(single.details.reject, mapped.details.reject)
    np.testing.assert_array_equal(single.details.valid, mapped.details.valid)
    assert single.fpr == mapped.fpr
    assert single.authoritative and single.matches_design(spec)
    assert single.procedure_scope == "single_estimator"
    assert np.isfinite(single.ks_p_value)
    assert mapped.procedure_scope == "joint_decision"
    if direction == "positive":
        assert not single.details.reject.any()  # previous behavior counted two negative effects


def test_single_estimator_aa_invalid_fits_do_not_reject_with_relaxed_rule():
    panel = make_synthetic_panel(n_cities=10, n_days=120, seed=7)
    rule = DecisionRule(require_all_valid=False, require_decision_validity=False)
    kwargs = dict(n_runs=2, n_treated=1, n_donors=9, min_valid_runs=1,
                  max_pre_rmspe=1e-20, decision_rule=rule)
    single = run_aa_calibration(panel, panel.cities, fit_scm, 60, 14, **kwargs)
    mapped = run_aa_calibration(panel, panel.cities, fit_scm, 60, 14,
                                fit_fns={"scm": fit_scm}, **kwargs)
    assert single.n_runs == mapped.n_runs == 0
    assert not single.details.reject.any()
    assert not mapped.details.reject.any()


def test_single_estimator_rejects_an_incompatible_rule_before_simulation():
    panel = make_synthetic_panel(n_cities=10, n_days=120, seed=7)
    with pytest.raises(ValueError, match="incompatível"):
        run_aa_calibration(panel, panel.cities, fit_scm, 60, 14,
                           n_treated=1, n_donors=9,
                           decision_rule=DecisionRule(methods=("sdid",)))
