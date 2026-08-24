"""Post-period analysis bound to an immutable geo-experiment design.

Point-estimate sensitivity is kept separate from inference. Observational
in-space placebo ranks remain visible, but cannot satisfy a decision rule
without matching complete-procedure calibration. Randomized assignments can
use the corresponding randomization rank.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from supply_experiments._version import require_runtime_version
from supply_experiments.calibration.aa import AACalibration
from supply_experiments.design.spec import DesignSpec
from supply_experiments.estimators import ESTIMATORS, fit_did_panel
from supply_experiments.inference.bootstrap import (
    UnsupportedFewTreatedClustersError,
    wild_cluster_bootstrap,
)
from supply_experiments.inference.conformal import conformal_inference
from supply_experiments.inference.permutation import placebo_inference
from supply_experiments.metrics import ratio_did
from supply_experiments.panel import CityPanel, ExperimentWindow


def benjamini_hochberg(p_values: Dict[str, float], q: float = 0.10) -> Dict[str, bool]:
    """Return which finite p-values pass the Benjamini-Hochberg FDR rule."""
    if not (0.0 < q < 1.0):
        raise ValueError("q deve estar em (0, 1)")
    items = [(key, value) for key, value in p_values.items() if np.isfinite(value)]
    items.sort(key=lambda item: item[1])
    out = {key: False for key in p_values}
    threshold_index = 0
    m = len(items)
    for index, (_, p_value) in enumerate(items, start=1):
        if p_value <= q * index / m:
            threshold_index = index
    for index, (key, _) in enumerate(items, start=1):
        out[key] = index <= threshold_index
    return out


@dataclass
class EstimatorRow:
    method: str
    att_pct: float
    p_value: float
    p_value_att: float
    ci: Optional[tuple] = None
    pre_rmspe: float = np.nan
    n_placebos: int = 0
    note: str = ""
    confidence_level: float = np.nan
    ci_hypothesis: str = ""
    prefit_valid: bool = False
    valid_for_decision: bool = False
    inference_basis: str = "diagnostic"
    failure_reason: str = ""

    def decision_payload(self) -> dict:
        return {
            "p_value": self.p_value,
            "att_pct": self.att_pct,
            "valid_for_decision": self.valid_for_decision,
        }


@dataclass
class ExperimentReport:
    experiment_id: str
    primary_kpi: str
    window: ExperimentWindow
    estimators: List[EstimatorRow]
    guardrails: pd.DataFrame
    triangulation: str
    decision_inputs: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    decision: str = "DIAGNOSTIC_ONLY"
    decision_eligible: bool = False
    design_fingerprint: str = ""
    calibration_fingerprint: str = ""
    interim: bool = False

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "method": row.method,
                "att_pct": row.att_pct,
                "p_value": row.p_value,
                "p_value_att": row.p_value_att,
                "ci_low": row.ci[0] if row.ci else np.nan,
                "ci_high": row.ci[1] if row.ci else np.nan,
                "confidence_level": row.confidence_level,
                "ci_hypothesis": row.ci_hypothesis,
                "pre_rmspe": row.pre_rmspe,
                "prefit_valid": row.prefit_valid,
                "n_placebos": row.n_placebos,
                "valid_for_decision": row.valid_for_decision,
                "inference_basis": row.inference_basis,
                "failure_reason": row.failure_reason,
                "note": row.note,
            }
            for row in self.estimators
        ])

    def summary(self) -> str:
        lines = [f"=== Experiment {self.experiment_id} | primary KPI: {self.primary_kpi} ==="]
        lines.append(
            f"Window: pre {self.window.pre_start}..{self.window.pre_end} | "
            f"post {self.window.start_date}..{self.window.end_date}"
        )
        if self.interim:
            lines.append("INTERIM MONITORING ONLY: point estimates and decisions are suppressed")
        for row in self.estimators:
            ci = ""
            if row.ci and np.isfinite(row.ci[0]):
                level = f"{row.confidence_level:.0%}" if np.isfinite(row.confidence_level) else "?"
                ci = f" sharp-set {level}[{row.ci[0]:+.1%}, {row.ci[1]:+.1%}]"
            rank_label = "p" if row.valid_for_decision else "placebo-rank"
            lines.append(
                f"  {row.method:>10}: ATT={row.att_pct:+.2%} | "
                f"{rank_label}(RMSPE)={row.p_value:.3f}{ci} | "
                f"pre-RMSPE={row.pre_rmspe:.1%} | valid={row.valid_for_decision}"
            )
        lines.append(f"Sensitivity agreement: {self.triangulation}")
        lines.append(f"Decision: {self.decision}")
        for warning in self.warnings:
            lines.append(f"  WARNING: {warning}")
        return "\n".join(lines)


def _triangulate(
    rows: List[EstimatorRow], alpha: float, agreement_tol: float = 0.05,
) -> str:
    """Descriptive cross-estimator sensitivity check, not independent evidence."""
    finite = [row for row in rows if np.isfinite(row.att_pct)]
    if len(finite) < 2:
        return "INCONCLUSIVE: fewer than two finite point estimates"
    atts = np.array([row.att_pct for row in finite])
    same_sign = bool(np.all(atts > 0) or np.all(atts < 0))
    spread = float(atts.max() - atts.min())
    if not all(np.isfinite(row.p_value) for row in finite):
        verdict = "INCONCLUSIVE INFERENCE: one or more ranks are missing"
    elif not all(row.valid_for_decision for row in finite):
        verdict = "DESCRIPTIVE AGREEMENT ONLY: inference is not decision-valid"
    else:
        significant = [row.p_value <= alpha for row in finite]
        if same_sign and spread < agreement_tol and len(set(significant)) == 1:
            verdict = "AGREEMENT"
        elif same_sign and spread < agreement_tol:
            verdict = "PARTIAL: magnitude agrees, significance differs"
        else:
            verdict = "DIVERGENT: investigate fit, interference and sensitivity"
    return (
        f"{verdict} (spread tolerance={agreement_tol:.0%}) | "
        + ", ".join(f"{row.method}={row.att_pct:+.1%}" for row in finite)
    )


def _validate_design_arguments(
    design: DesignSpec,
    experiment_id: str,
    treated: Sequence[str],
    donors: Sequence[str],
    window: ExperimentWindow,
    primary_kpi: str,
    alpha: float,
    max_group_placebos: Optional[int],
) -> None:
    expected = {
        "experiment_id": experiment_id,
        "treated": tuple(sorted(treated)),
        "donors": tuple(sorted(donors)),
        "pre_days": window.pre_window_days,
        "post_days": window.post_days,
        "start_date": window.start_date,
        "end_date": window.end_date,
        "anticipation_days": window.anticipation_days,
        "primary_kpi": primary_kpi,
        "alpha": alpha,
        "max_group_placebos": max_group_placebos,
    }
    actual = {
        "experiment_id": design.experiment_id,
        "treated": design.treated_units,
        "donors": design.donor_units,
        "pre_days": design.pre_days,
        "post_days": design.post_days,
        "start_date": design.start_date,
        "end_date": design.end_date,
        "anticipation_days": design.anticipation_days,
        "primary_kpi": design.primary_kpi,
        "alpha": design.alpha,
        "max_group_placebos": design.max_group_placebos,
    }
    mismatched = [key for key in expected if expected[key] != actual[key]]
    if mismatched:
        raise ValueError(f"analysis arguments diverge from DesignSpec: {mismatched}")


def _matching_complete_calibration(
    design: DesignSpec,
    calibration: Optional[AACalibration],
) -> bool:
    rule_methods = design.decision_rule.methods or design.estimator_names
    if design.assignment_mechanism == "randomized" and len(rule_methods) == 1:
        return True
    if not design.require_calibration or calibration is None or not calibration.passed:
        return False
    if design.calibration_scope != "exact_design":
        return False
    if calibration.design_fingerprint != design.fingerprint:
        return False
    if design.calibration_fingerprint != calibration.calibration_fingerprint:
        return False
    if not calibration.matches_design(design):
        return False
    valid_scope = (
        calibration.selection_scope == "selected_design"
        if design.assignment_mechanism == "observational"
        else calibration.selection_scope in {"random_assignment", "selected_design"}
    )
    return valid_scope and calibration.procedure_scope == "joint_decision"


def _guardrail_report(
    panel: CityPanel,
    treated: Sequence[str],
    donors: Sequence[str],
    guardrail_kpis: Sequence[str],
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
    fdr_q: float,
    warnings: List[str],
) -> pd.DataFrame:
    rows = []
    if not np.any(post_mask):
        if guardrail_kpis:
            warnings.append("no observed post-period days available for guardrails")
        return pd.DataFrame()
    for kpi in guardrail_kpis:
        if kpi not in panel.numerators or kpi not in panel.denominators:
            warnings.append(f"guardrail {kpi!r} lacks numerator/denominator and was skipped")
            continue
        numerator, denominator = panel.numerators[kpi], panel.denominators[kpi]
        missing = [
            city for city in list(treated) + list(donors)
            if city not in numerator.columns or city not in denominator.columns
        ]
        if missing:
            warnings.append(f"guardrail {kpi!r} lacks cities {sorted(set(missing))}")
            continue
        result = ratio_did(
            numerator[list(treated)].sum(axis=1), denominator[list(treated)].sum(axis=1),
            numerator[list(donors)].sum(axis=1), denominator[list(donors)].sum(axis=1),
            pre_mask, post_mask,
        )
        rows.append(
            {
                "kpi": kpi, "effect_abs": result.effect_abs,
                "effect_rel": result.effect_rel, "se": result.se,
                "p_value": result.p_value, "ci_low": result.ci_lower,
                "ci_high": result.ci_upper, "n_boot": result.n_boot,
                "method": result.method,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        significant = benjamini_hochberg(
            dict(zip(frame["kpi"], frame["p_value"], strict=True)), q=fdr_q,
        )
        frame["significant_bh"] = frame["kpi"].map(significant)
    return frame


def analyze_experiment(
    panel: CityPanel,
    experiment_id: str,
    treated: Sequence[str],
    donors: Sequence[str],
    window: ExperimentWindow,
    primary_kpi: str = "gmv",
    guardrail_kpis: Sequence[str] = (),
    alpha: float = 0.10,
    fdr_q: float = 0.10,
    holidays: Optional[Sequence[str]] = None,
    did_control: Optional[Sequence[str]] = None,
    run_conformal: bool = True,
    conformal_method: str = "scm",
    agreement_tol: float = 0.05,
    today: Optional[date] = None,
    allow_interim: bool = False,
    max_group_placebos: Optional[int] = 30,
    *,
    design_spec: Optional[DesignSpec] = None,
    calibration: Optional[AACalibration] = None,
) -> ExperimentReport:
    """Analyze after the window closes; legacy calls are diagnostic only."""
    if not (0.0 < alpha < 1.0 and 0.0 < fdr_q < 1.0):
        raise ValueError("alpha e fdr_q devem estar em (0, 1)")
    if agreement_tol <= 0:
        raise ValueError("agreement_tol deve ser > 0")
    if conformal_method != "scm":
        raise ValueError("conformal_method suporta somente 'scm'")

    warnings: List[str] = []
    analysis_date = today or date.today()
    interim = analysis_date <= window.end_date
    if interim and not allow_interim:
        raise ValueError(
            f"experiment ends on {window.end_date}; analysis is blocked until the next day "
            "(anti-peeking). allow_interim=True returns guardrails only."
        )

    if design_spec is not None:
        design = design_spec.validate()
        require_runtime_version(design.implementation_version)
        _validate_design_arguments(
            design, experiment_id, treated, donors, window, primary_kpi,
            alpha, max_group_placebos,
        )
    else:
        design = DesignSpec(
            experiment_id=experiment_id,
            treated_units=tuple(treated), donor_units=tuple(donors),
            estimator_names=tuple(ESTIMATORS), pre_days=window.pre_window_days,
            post_days=window.post_days, start_date=window.start_date,
            end_date=window.end_date, anticipation_days=window.anticipation_days,
            alpha=alpha, max_group_placebos=max_group_placebos,
            primary_kpi=primary_kpi, assignment_mechanism="observational",
            require_calibration=False, calibration_scope="none",
        )
        warnings.append("no DesignSpec supplied: results are exploratory only")

    panel.aggregate(treated)
    panel.aggregate(donors)
    if set(treated) & set(donors):
        raise ValueError("treated and donors overlap")

    pre_mask, post_mask = window.masks(panel.index)
    if interim:
        post_mask = post_mask & (panel.index.date <= analysis_date)
    guardrails = _guardrail_report(
        panel, treated, donors, guardrail_kpis, pre_mask, post_mask, fdr_q, warnings,
    )
    if interim:
        return ExperimentReport(
            experiment_id=experiment_id, primary_kpi=primary_kpi, window=window,
            estimators=[], guardrails=guardrails,
            triangulation="NOT RUN DURING INTERIM MONITORING",
            decision_inputs={"analysis_date": analysis_date.isoformat()},
            warnings=warnings, decision="INTERIM_MONITORING_ONLY",
            decision_eligible=False, design_fingerprint=design.fingerprint,
            interim=True,
        )

    pslice = panel.slice_for(treated, donors, window, aggregation="mean")
    calibrated = _matching_complete_calibration(design, calibration)
    if not calibrated:
        if design.assignment_mechanism == "randomized":
            warnings.append(
                "multi-estimator randomized decision lacks matching joint calibration; "
                "per-estimator randomization p-values do not control rule multiplicity"
            )
        elif design.require_calibration:
            warnings.append(
                "matching selected-design joint calibration is absent or failed; "
                "placebo ranks are diagnostic only"
            )

    estimator_configs: Mapping[str, dict] = json.loads(design.estimator_config_json)
    rows: List[EstimatorRow] = []
    unknown_estimators = set(design.estimator_names) - set(ESTIMATORS)
    if unknown_estimators:
        raise ValueError(f"unsupported estimators in DesignSpec: {sorted(unknown_estimators)}")
    for method in design.estimator_names:
        fit_fn = ESTIMATORS[method]
        fit_kwargs = dict(estimator_configs.get(method, {}))
        if method == "sdid":
            # The panel slice is already the treated mean. These parameters are
            # framework-owned so a stale legacy config cannot divide it again.
            fit_kwargs["n_treated_units"] = len(treated)
            fit_kwargs["treated_aggregation"] = "mean"
        required_placebos = math.ceil((1.0 / alpha) - 1.0 - 1e-12)
        inference = placebo_inference(
            fit_fn, *pslice.as_args(), fit_kwargs=fit_kwargs,
            min_placebos=required_placebos,
            n_treated_units=len(treated), max_group_placebos=max_group_placebos,
            seed=design.permutation_seed,
            assignment_mechanism=design.assignment_mechanism,
            calibrated=calibrated,
            Y_treated_pre=pslice.Y_treated_pre,
            Y_treated_post=pslice.Y_treated_post,
            treated_names=pslice.treated_names,
        )
        fit = inference.real_fit
        if fit is None or not fit.success:
            rows.append(
                EstimatorRow(
                    method=method, att_pct=np.nan, p_value=np.nan,
                    p_value_att=np.nan, note=inference.note,
                    failure_reason="estimator fit failed",
                    inference_basis=inference.inference_basis,
                )
            )
            continue
        prefit_valid = design.max_pre_rmspe is None or fit.pre_rmspe <= design.max_pre_rmspe
        resolution_valid = inference.n_placebos >= required_placebos
        valid_for_decision = bool(
            calibrated
            and inference.valid_for_decision
            and prefit_valid
            and resolution_valid
        )
        notes = [inference.note] if inference.note else []
        if not prefit_valid:
            notes.append(
                f"pre-RMSPE {fit.pre_rmspe:.2%} exceeds design gate "
                f"{design.max_pre_rmspe:.2%}"
            )
        if not resolution_valid:
            notes.append(
                f"only {inference.n_placebos} valid placebos; {required_placebos} "
                f"are required to attain alpha={alpha:.3f}"
            )
        ci = None
        confidence_level = np.nan
        ci_hypothesis = ""
        if run_conformal and method == "scm":
            try:
                conformal = conformal_inference(
                    fit_fn, *pslice.as_args(), alpha=alpha, fit_kwargs=fit_kwargs,
                )
                ci = (conformal.ci_lower_pct, conformal.ci_upper_pct)
                confidence_level = conformal.confidence_level
                ci_hypothesis = conformal.hypothesis
                notes.append(conformal.assumption_note)
                if conformal.acceptance_set_disconnected:
                    notes.append(
                        f"conformal acceptance set is disconnected: "
                        f"{conformal.acceptance_intervals_pct}"
                    )
                if conformal.boundary_truncated:
                    notes.append("conformal set touches the search-grid boundary")
            except ValueError as exc:
                notes.append(f"conformal unavailable: {exc}")
        rows.append(
            EstimatorRow(
                method=method, att_pct=fit.att_pct, p_value=inference.p_value,
                p_value_att=inference.p_value_att, ci=ci,
                confidence_level=confidence_level, ci_hypothesis=ci_hypothesis,
                pre_rmspe=fit.pre_rmspe, prefit_valid=prefit_valid,
                n_placebos=inference.n_placebos, note=" | ".join(notes),
                valid_for_decision=valid_for_decision,
                inference_basis=inference.inference_basis,
            )
        )

    if did_control:
        did_fit = fit_did_panel(panel, treated, did_control, window, holidays)
        if did_fit.success:
            try:
                bootstrap = wild_cluster_bootstrap(did_fit, n_boot=999, seed=11)
                rows.append(
                    EstimatorRow(
                        method="did_panel", att_pct=did_fit.att_pct,
                        p_value=bootstrap.p_value, p_value_att=bootstrap.p_value,
                        note=bootstrap.diagnostic, valid_for_decision=False,
                        inference_basis="ordinary_wild_cluster_bootstrap",
                    )
                )
            except UnsupportedFewTreatedClustersError as exc:
                rows.append(
                    EstimatorRow(
                        method="did_panel", att_pct=did_fit.att_pct,
                        p_value=np.nan, p_value_att=np.nan, note=str(exc),
                        valid_for_decision=False,
                        inference_basis="unsupported_few_treated_clusters",
                    )
                )

    primary_rows = [row for row in rows if row.method in design.estimator_names]
    result_map = {row.method: row.decision_payload() for row in primary_rows}
    rule_methods = design.decision_rule.methods or design.estimator_names
    method_validity = [
        result_map.get(method, {}).get("valid_for_decision") is True
        for method in rule_methods
    ]
    if design.decision_rule.require_all_valid:
        rule_rows_valid = len(method_validity) == len(rule_methods) and all(method_validity)
    else:
        rule_rows_valid = sum(method_validity) >= design.decision_rule.min_rejections
    decision_eligible = bool(design_spec is not None and rule_rows_valid)
    if not decision_eligible:
        decision = "DIAGNOSTIC_ONLY"
    elif design.decision_rule.evaluate(result_map, alpha):
        decision = "RULE_PASSED"
    else:
        decision = "RULE_NOT_PASSED"

    calibration_fingerprint = calibration.calibration_fingerprint if calibration else ""
    return ExperimentReport(
        experiment_id=experiment_id, primary_kpi=primary_kpi, window=window,
        estimators=rows, guardrails=guardrails,
        triangulation=_triangulate(primary_rows, alpha, agreement_tol),
        decision_inputs={
            "alpha": alpha, "fdr_q": fdr_q,
            "n_donors": len(pslice.donor_names),
            "decision_rule": design.decision_rule.to_dict(),
            "assignment_mechanism": design.assignment_mechanism,
            "design_fingerprint": design.fingerprint,
            "calibration_fingerprint": calibration_fingerprint,
        },
        warnings=warnings, decision=decision,
        decision_eligible=decision_eligible,
        design_fingerprint=design.fingerprint,
        calibration_fingerprint=calibration_fingerprint,
    )
