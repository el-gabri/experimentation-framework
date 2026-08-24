"""Reproduce the public statistical-calibration certificate on synthetic data.

The script prints a human-readable A/A and power report and writes the same
results as versioned text and JSON artifacts. It never reads external or
production data: every input comes from ``make_synthetic_panel`` with fixed
seeds.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy

from supply_experiments.calibration.aa import AACalibration, run_aa_calibration
from supply_experiments.design.power import PowerResult, power_analysis
from supply_experiments.estimators.scm import fit_scm
from supply_experiments.synthetic import make_synthetic_panel

SCHEMA_VERSION = "1.0.0"
DEFAULT_OUTPUT_DIR = Path("calibration-artifacts")
TEXT_FILENAME = "calibration_certificate.txt"
JSON_FILENAME = "calibration_certificate.json"

PANEL_N_CITIES = 40
PANEL_N_DAYS = 430
PANEL_SEED = 99
PRE_DAYS = 120
POST_DAYS = 35
AA_RUNS = 120
N_TREATED = 2
NAIVE_CONTROL_CITIES = 10
NAIVE_SEED = 2026
FRAMEWORK_DONORS = 15
FRAMEWORK_ALPHA = 0.10
FRAMEWORK_SEED = 7
FRAMEWORK_MIN_VALID_RUNS = 100
FRAMEWORK_MIN_KS_P_VALUE = 0.01
FRAMEWORK_MAX_GROUP_PLACEBOS = 30
PERMUTATION_GROUP_SEED = 123
POWER_TREATED = ("CITY_03", "CITY_07")
POWER_DONORS = 18
POWER_EFFECT_GRID = (0.0, 0.03, 0.05, 0.08, 0.12)
POWER_SIMS_PER_POINT = 25
POWER_SEED = 3
POWER_MAX_GROUP_PLACEBOS = 10


# ---------------------------------------------------------------------------
# Naive baseline: aggregate OLS with a classical t-test
# ---------------------------------------------------------------------------
def naive_ols_pvalue(dates, y_control, y_treated, start_date):
    from scipy.stats import t as t_dist

    y0, y1 = np.asarray(y_control, float), np.asarray(y_treated, float)
    n_periods = len(y0)
    dt = pd.to_datetime(dates)
    post = (dt.date >= start_date).astype(float)
    y = np.concatenate([y0, y1])
    group = np.concatenate([np.zeros(n_periods), np.ones(n_periods)])
    post2 = np.concatenate([post, post])
    interaction = group * post2
    columns = [np.ones(2 * n_periods), group, post2, interaction]
    dow = pd.Series(dt).dt.dayofweek.to_numpy()
    dow = np.concatenate([dow, dow])
    for day in range(1, 7):
        columns.append((dow == day).astype(float))
    design = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta
    n_observations, n_parameters = design.shape
    degrees_of_freedom = n_observations - n_parameters
    sigma2 = float(residuals @ residuals) / degrees_of_freedom
    xtx_inverse = np.linalg.pinv(design.T @ design)
    standard_error = np.sqrt(sigma2 * xtx_inverse[3, 3])
    t_statistic = beta[3] / standard_error
    return float(2 * t_dist.sf(abs(t_statistic), df=degrees_of_freedom))


def _package_version() -> str:
    try:
        return version("supply-experiments")
    except PackageNotFoundError:
        return "source-tree"


def _finite_float(value: Any) -> Optional[float]:
    """Return a JSON-safe float, using null for unavailable diagnostics."""
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _format_percent(
    value: Optional[float], digits: int = 1, signed: bool = False
) -> str:
    if value is None:
        return "unavailable"
    sign = "+" if signed else ""
    return f"{value:{sign}.{digits}%}"


def _format_decimal(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.{digits}f}"


def _summarize_power(
    power: PowerResult,
) -> Tuple[List[Dict[str, Any]], bool, Optional[float], str]:
    required_columns = {"delta", "p_value", "reject"}
    if not required_columns.issubset(power.details.columns):
        raise ValueError(
            "Power details must contain delta, p_value, and reject columns"
        )

    points: List[Dict[str, Any]] = []
    for effect in power.effect_grid:
        effect_value = float(effect)
        effect_rows = power.details[
            np.isclose(power.details["delta"].astype(float), effect_value)
        ]
        attempted = len(effect_rows)
        p_values = pd.to_numeric(effect_rows["p_value"], errors="coerce").to_numpy(float)
        valid_mask = np.isfinite(p_values)
        valid_rows = effect_rows.loc[valid_mask]
        valid = len(valid_rows)
        rejections = int(valid_rows["reject"].astype(bool).sum())
        points.append(
            {
                "effect": effect_value,
                "attempted_simulations": attempted,
                "valid_simulations": valid,
                "failed_simulations": attempted - valid,
                "rejections": rejections,
                "rejection_rate": rejections / valid if valid else None,
            }
        )

    complete = bool(
        points
        and len(power.details) == len(points) * power.n_sims_per_point
        and all(
            point["attempted_simulations"] == power.n_sims_per_point
            and point["failed_simulations"] == 0
            for point in points
        )
    )
    nonzero_points = [point for point in points if point["effect"] != 0.0]
    if not complete:
        return points, False, None, "unavailable_incomplete_simulations"
    if not nonzero_points:
        return points, True, None, "not_tested"

    detectable = [
        point
        for point in nonzero_points
        if point["rejection_rate"] is not None and point["rejection_rate"] >= 0.80
    ]
    if not detectable:
        return points, True, None, "above_tested_grid"
    mde = min(float(point["effect"]) for point in detectable)
    return points, True, mde, "estimated_on_tested_grid"


def build_certificate(
    naive_p_values: Sequence[float],
    aa: AACalibration,
    power: PowerResult,
) -> Dict[str, Any]:
    """Build the stable, machine-readable public certificate contract."""
    if len(naive_p_values) != AA_RUNS:
        raise ValueError("Naive baseline does not match the canonical requested run count")
    naive_p_values_array = np.asarray(naive_p_values, dtype=float)
    naive_valid_p_values = naive_p_values_array[np.isfinite(naive_p_values_array)]
    naive_valid_runs = len(naive_valid_p_values)
    naive_failed_runs = AA_RUNS - naive_valid_runs
    naive_complete = naive_failed_runs == 0
    naive_rejections_05 = int((naive_valid_p_values <= 0.05).sum())
    naive_rejections_10 = int((naive_valid_p_values <= 0.10).sum())
    if not np.isclose(aa.alpha, FRAMEWORK_ALPHA):
        raise ValueError("A/A alpha does not match the canonical certificate design")
    if [float(effect) for effect in power.effect_grid] != list(POWER_EFFECT_GRID):
        raise ValueError("Power effect grid does not match the canonical certificate design")
    if power.n_sims_per_point != POWER_SIMS_PER_POINT:
        raise ValueError(
            "Power simulation count does not match the canonical certificate design"
        )
    if not np.isclose(power.alpha, FRAMEWORK_ALPHA):
        raise ValueError("Power alpha does not match the canonical certificate design")

    if not {"p_value", "reject"}.issubset(aa.details.columns):
        raise ValueError("A/A details must contain p_value and reject columns")
    if len(aa.details) != AA_RUNS:
        raise ValueError("A/A details do not match the canonical requested run count")
    aa_p_values = pd.to_numeric(aa.details["p_value"], errors="coerce").to_numpy(float)
    aa_valid_rows = aa.details.loc[np.isfinite(aa_p_values)]
    if len(aa_valid_rows) != aa.n_runs:
        raise ValueError("A/A valid-run summary contradicts its run details")
    framework_rejections = int(aa_valid_rows["reject"].astype(bool).sum())
    detailed_fpr = framework_rejections / aa.n_runs if aa.n_runs else None
    reported_fpr = _finite_float(aa.fpr)
    if (detailed_fpr is None) != (reported_fpr is None) or (
        detailed_fpr is not None
        and reported_fpr is not None
        and not np.isclose(detailed_fpr, reported_fpr)
    ):
        raise ValueError("A/A false-positive rate contradicts its run details")
    interval_lower = _finite_float(aa.fpr_ci[0])
    interval_upper = _finite_float(aa.fpr_ci[1])
    ks_p_value = _finite_float(aa.ks_p_value)
    nominal_alpha_inside_interval = bool(
        interval_lower is not None
        and interval_upper is not None
        and interval_lower <= aa.alpha <= interval_upper
    )
    minimum_valid_runs_met = bool(aa.n_runs >= FRAMEWORK_MIN_VALID_RUNS)
    minimum_ks_p_value_met = bool(
        ks_p_value is not None and ks_p_value >= FRAMEWORK_MIN_KS_P_VALUE
    )
    aa_calibrated = bool(
        minimum_valid_runs_met
        and nominal_alpha_inside_interval
        and minimum_ks_p_value_met
    )
    if bool(aa.passed) != aa_calibrated:
        raise ValueError("A/A result contradicts its published pass criteria")

    power_points, power_complete, power_mde, power_mde_status = _summarize_power(power)
    if power_complete:
        for point in power_points:
            reported_rate = _finite_float(power.power.get(point["effect"]))
            if reported_rate is None or not np.isclose(
                reported_rate, point["rejection_rate"]
            ):
                raise ValueError("Power summary contradicts its simulation details")
        reported_mde = _finite_float(power.mde_80)
        if (reported_mde is None) != (power_mde is None) or (
            reported_mde is not None
            and power_mde is not None
            and not np.isclose(reported_mde, power_mde)
        ):
            raise ValueError("Power MDE contradicts its simulation details")
    power_fpr = next(
        (
            point["rejection_rate"]
            for point in power_points
            if point["effect"] == 0.0
        ),
        None,
    )
    certificate_passed = naive_complete and aa_calibrated and power_complete

    return {
        "schema_version": SCHEMA_VERSION,
        "certificate_type": "synthetic_statistical_calibration",
        "data_scope": {
            "kind": "synthetic_only",
            "contains_real_data": False,
            "generator": "supply_experiments.synthetic.make_synthetic_panel",
            "description": (
                "Deterministic simulated city-day panel; no external, customer, "
                "production, or internal data."
            ),
        },
        "software": {
            "python": platform.python_version(),
            "supply_experiments": _package_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "reproducibility": {
            "command": (
                "python calibration_certificate.py --output-dir calibration-artifacts"
            ),
            "random_seeds": {
                "panel": PANEL_SEED,
                "naive_aa": NAIVE_SEED,
                "framework_aa": FRAMEWORK_SEED,
                "power": POWER_SEED,
                "permutation_group_sampling": PERMUTATION_GROUP_SEED,
            },
        },
        "design": {
            "panel": {
                "n_cities": PANEL_N_CITIES,
                "n_days": PANEL_N_DAYS,
                "data_generating_process": [
                    "common_factor",
                    "day_of_week_seasonality",
                    "ar1_daily_noise",
                ],
            },
            "aa": {
                "requested_runs": AA_RUNS,
                "pre_days": PRE_DAYS,
                "post_days": POST_DAYS,
                "n_treated": N_TREATED,
                "naive_control_cities": NAIVE_CONTROL_CITIES,
                "framework_donors": FRAMEWORK_DONORS,
                "framework_alpha": FRAMEWORK_ALPHA,
                "framework_max_group_placebos": FRAMEWORK_MAX_GROUP_PLACEBOS,
                "minimum_valid_runs": FRAMEWORK_MIN_VALID_RUNS,
                "minimum_ks_p_value": FRAMEWORK_MIN_KS_P_VALUE,
            },
            "power": {
                "treated_synthetic_ids": list(POWER_TREATED),
                "donor_selection": (
                    "first 18 synthetic city IDs excluding treated_synthetic_ids"
                ),
                "n_donors": POWER_DONORS,
                "pre_days": PRE_DAYS,
                "post_days": POST_DAYS,
                "alpha": FRAMEWORK_ALPHA,
                "effect_grid": [float(effect) for effect in power.effect_grid],
                "requested_simulations_per_effect": POWER_SIMS_PER_POINT,
                "max_group_placebos": POWER_MAX_GROUP_PLACEBOS,
                "target_power": 0.80,
            },
        },
        "results": {
            "naive_baseline": {
                "method": "aggregated_ols_classical_t_test",
                "attempted_runs": AA_RUNS,
                "valid_runs": naive_valid_runs,
                "failed_runs": naive_failed_runs,
                "complete": naive_complete,
                "false_positive_rates": [
                    {
                        "alpha": 0.05,
                        "rejections": naive_rejections_05,
                        "valid_runs": naive_valid_runs,
                        "fpr": (
                            naive_rejections_05 / naive_valid_runs
                            if naive_valid_runs
                            else None
                        ),
                    },
                    {
                        "alpha": 0.10,
                        "rejections": naive_rejections_10,
                        "valid_runs": naive_valid_runs,
                        "fpr": (
                            naive_rejections_10 / naive_valid_runs
                            if naive_valid_runs
                            else None
                        ),
                    },
                ],
            },
            "framework_aa": {
                "estimator": "scm",
                "inference": "in_space_permutation",
                "alpha": float(aa.alpha),
                "requested_runs": AA_RUNS,
                "valid_runs": int(aa.n_runs),
                "failed_runs": int(AA_RUNS - aa.n_runs),
                "rejections": framework_rejections,
                "false_positive_rate": detailed_fpr,
                "false_positive_rate_interval": {
                    "method": "clopper_pearson",
                    "level": 0.95,
                    "lower": interval_lower,
                    "upper": interval_upper,
                },
                "ks_uniformity_p_value": ks_p_value,
                "median_placebo_att_bias": _finite_float(aa.median_att_bias),
                "median_placebo_att_bias_role": "diagnostic_only",
                "pass_criteria": {
                    "minimum_valid_runs": FRAMEWORK_MIN_VALID_RUNS,
                    "minimum_ks_p_value": FRAMEWORK_MIN_KS_P_VALUE,
                    "minimum_valid_runs_met": minimum_valid_runs_met,
                    "nominal_alpha_inside_fpr_interval": nominal_alpha_inside_interval,
                    "minimum_ks_p_value_met": minimum_ks_p_value_met,
                },
                "passed": aa_calibrated,
            },
            "power_analysis": {
                "estimator": "scm",
                "inference": "in_space_permutation",
                "requested_simulations_per_effect": POWER_SIMS_PER_POINT,
                "points": power_points,
                "false_positive_rate": power_fpr,
                "complete": power_complete,
                "mde_grid_at_80pct_power": power_mde,
                "mde_status": power_mde_status,
            },
        },
        "verdict": {
            "naive_baseline_complete": naive_complete,
            "calibrated": aa_calibrated,
            "power_analysis_complete": power_complete,
            "passed": certificate_passed,
            "rule": (
                "At least 100 valid A/A runs, nominal alpha inside the exact 95% "
                "binomial interval, KS uniformity p-value at least 0.01, and no "
                "failed naive-baseline or power simulations."
            ),
        },
    }


def render_human_report(certificate: Dict[str, Any]) -> str:
    """Render the certificate as a readable console report."""
    results = certificate["results"]
    baseline_result = results["naive_baseline"]
    baseline = baseline_result["false_positive_rates"]
    framework = results["framework_aa"]
    power = results["power_analysis"]
    panel = certificate["design"]["panel"]
    aa_design = certificate["design"]["aa"]
    fpr_interval = framework["false_positive_rate_interval"]

    lines = [
        "=" * 72,
        "STATISTICAL CALIBRATION CERTIFICATE — A/A (true effect = 0)",
        "=" * 72,
        "",
        (
            f"Synthetic-only panel: {panel['n_cities']} cities, {panel['n_days']} days, "
            "common factor + DOW + AR(1)"
        ),
        (
            f"A/A runs: requested={framework['requested_runs']}, "
            f"valid={framework['valid_runs']}, failed={framework['failed_runs']} | "
            f"pre={aa_design['pre_days']}d, post={aa_design['post_days']}d, "
            f"treated={aa_design['n_treated']}"
        ),
        "",
        "NAIVE BASELINE (aggregated OLS, classical t-test):",
        (
            f"  Runs: attempted={baseline_result['attempted_runs']}, "
            f"valid={baseline_result['valid_runs']}, "
            f"failed={baseline_result['failed_runs']}"
        ),
        f"  FPR @ alpha=0.05: {_format_percent(baseline[0]['fpr']):>6}   (expected: 5%)",
        f"  FPR @ alpha=0.10: {_format_percent(baseline[1]['fpr']):>6}   (expected: 10%)",
        "",
        "THIS FRAMEWORK (SCM + in-space permutation):",
        (
            f"  FPR @ alpha={framework['alpha']:.2f}: "
            f"{_format_percent(framework['false_positive_rate']):>6}   "
            "exact 95% binomial CI "
            f"[{_format_percent(fpr_interval['lower'])}, "
            f"{_format_percent(fpr_interval['upper'])}]"
        ),
        (
            "  KS p-value (p-values ~ U(0,1)): "
            f"{_format_decimal(framework['ks_uniformity_p_value'])}"
        ),
        (
            "  Median placebo ATT bias: "
            f"{_format_percent(framework['median_placebo_att_bias'], digits=2, signed=True)}"
        ),
        f"  A/A CALIBRATED: {'YES' if framework['passed'] else 'NO'}",
        "",
        "=" * 72,
        "POWER CURVE (design: 2 treated, 18 donors, 35 days, SCM)",
        "=" * 72,
    ]
    for point in power["points"]:
        rejection_rate = point["rejection_rate"]
        bar = "█" * int(rejection_rate * 40) if rejection_rate is not None else ""
        lines.append(
            f"  effect={point['effect']:5.1%} | "
            f"power={_format_percent(rejection_rate):>6} | "
            f"valid={point['valid_simulations']}/{point['attempted_simulations']} {bar}"
        )

    mde = power["mde_grid_at_80pct_power"]
    if power["mde_status"] == "estimated_on_tested_grid":
        mde_text = f"{mde:.1%}"
    elif power["mde_status"] == "above_tested_grid":
        maximum_effect = max(point["effect"] for point in power["points"])
        mde_text = f"> {maximum_effect:.0%}"
    else:
        mde_text = "unavailable"
    failed_power_simulations = sum(
        point["failed_simulations"] for point in power["points"]
    )
    lines.extend(
        [
            "",
            "  FPR (effect=0): "
            f"{_format_percent(power['false_positive_rate'])} | MDE@80%: {mde_text}",
            (
                "  POWER ANALYSIS COMPLETE: "
                f"{'YES' if power['complete'] else 'NO'} "
                f"(failed simulations: {failed_power_simulations})"
            ),
            "",
            "Framework rule: approve a design only when MDE <= expected effect.",
            (
                "CERTIFICATE VERDICT: "
                f"{'PASS' if certificate['verdict']['passed'] else 'FAIL'}"
            ),
        ]
    )
    return "\n".join(lines)


def write_certificate(
    certificate: Dict[str, Any], output_dir: Union[str, Path]
) -> Tuple[Path, Path]:
    """Write deterministic UTF-8 text and strict JSON from the same payload."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    text_path = directory / TEXT_FILENAME
    json_path = directory / JSON_FILENAME
    text_path.write_text(render_human_report(certificate) + "\n", encoding="utf-8")
    json_payload = json.dumps(
        certificate,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    json_path.write_text(json_payload + "\n", encoding="utf-8")
    return text_path, json_path


def run_calibration() -> Dict[str, Any]:
    """Run the full deterministic synthetic calibration workflow."""
    rng = np.random.default_rng(NAIVE_SEED)
    panel = make_synthetic_panel(
        n_cities=PANEL_N_CITIES,
        n_days=PANEL_N_DAYS,
        seed=PANEL_SEED,
    )
    cities = panel.cities
    n_periods = len(panel.index)

    naive_p_values = []
    for _ in range(AA_RUNS):
        selected = rng.choice(
            cities,
            size=N_TREATED + NAIVE_CONTROL_CITIES,
            replace=False,
        ).tolist()
        treated = selected[:N_TREATED]
        control = selected[N_TREATED:]
        start = int(rng.integers(PRE_DAYS, n_periods - POST_DAYS))
        window = slice(start - PRE_DAYS, start + POST_DAYS)
        naive_p_values.append(
            naive_ols_pvalue(
                panel.index[window],
                panel.aggregate(control).to_numpy()[window],
                panel.aggregate(treated).to_numpy()[window],
                panel.index[start].date(),
            )
        )

    aa = run_aa_calibration(
        panel,
        cities,
        fit_scm,
        PRE_DAYS,
        POST_DAYS,
        n_runs=AA_RUNS,
        n_treated=N_TREATED,
        n_donors=FRAMEWORK_DONORS,
        alpha=FRAMEWORK_ALPHA,
        seed=FRAMEWORK_SEED,
        min_valid_runs=FRAMEWORK_MIN_VALID_RUNS,
        min_ks_p_value=FRAMEWORK_MIN_KS_P_VALUE,
        max_group_placebos=FRAMEWORK_MAX_GROUP_PLACEBOS,
        permutation_seed=PERMUTATION_GROUP_SEED,
    )

    donors = [city for city in cities if city not in POWER_TREATED][:POWER_DONORS]
    power = power_analysis(
        panel,
        POWER_TREATED,
        donors,
        fit_scm,
        PRE_DAYS,
        POST_DAYS,
        effect_grid=POWER_EFFECT_GRID,
        n_sims_per_point=POWER_SIMS_PER_POINT,
        alpha=FRAMEWORK_ALPHA,
        seed=POWER_SEED,
        max_group_placebos=POWER_MAX_GROUP_PLACEBOS,
        permutation_seed=PERMUTATION_GROUP_SEED,
    )
    return build_certificate(naive_p_values, aa, power)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print the deterministic synthetic calibration report and write its "
            "versioned text and JSON artifacts."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Artifact directory (default: {DEFAULT_OUTPUT_DIR.as_posix()})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    certificate = run_calibration()
    text_path, json_path = write_certificate(certificate, args.output_dir)
    stdout_encoding = str(getattr(sys.stdout, "encoding", "")).lower().replace("-", "")
    if hasattr(sys.stdout, "reconfigure") and stdout_encoding not in {"utf8", "utf8sig"}:
        sys.stdout.reconfigure(encoding="utf-8")
    print(render_human_report(certificate))
    print(f"\nArtifacts: {text_path.resolve()} | {json_path.resolve()}")
    return 0 if certificate["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
