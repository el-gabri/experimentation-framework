"""Calibração A/A de um procedimento de desenho/inferência especificado.

Roda N experimentos nulos (tratadas e janelas sorteadas do histórico, sem
tratamento) por todo o pipeline de inferência e verifica:
  - FPR ≈ α (taxa de falsos positivos);
  - distribuição de ranks compatível com a grade uniforme discreta;
  - viés mediano do ATT ≈ 0.

Um A/A aleatório não valida seleção otimizada de mercados. Use
``selection_fn`` para repetir o seletor implantado ou rotule o resultado como
calibração de alocação aleatória apenas.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from supply_experiments._version import require_runtime_version
from supply_experiments.design.power import (
    _decision_simulation,
    _require_builtin_estimators,
    simulate_once,
)
from supply_experiments.design.spec import DecisionRule, DesignSpec
from supply_experiments.panel import CityPanel


@dataclass
class AACalibration:
    n_runs: int
    alpha: float
    fpr: float
    fpr_ci: tuple            # IC binomial 95% do FPR
    ks_p_value: float        # KS marginal; NaN para regras conjuntas
    median_att_bias: float
    passed: bool
    details: pd.DataFrame
    requested_runs: int = 0
    invalid_rate: float = np.nan
    fpr_upper_bound: float = np.nan
    max_fpr: float = np.nan
    median_placebo_att: float = np.nan
    estimator_name: str = ""
    design_fingerprint: str = ""
    calibration_fingerprint: str = ""
    selection_scope: str = "random_assignment"
    assignment_mechanism: str = "observational"
    procedure_scope: str = "single_estimator"
    failure_reasons: Tuple[str, ...] = ()
    calibration_spec_json: str = ""
    authoritative: bool = False

    @staticmethod
    def _finite_or_none(value: Any) -> Optional[float]:
        return float(value) if value is not None and np.isfinite(value) else None

    def integrity_payload(self) -> Dict[str, Any]:
        """Canonical procedure plus decision-relevant summary bound by the digest."""
        spec = json.loads(self.calibration_spec_json)
        return {
            "calibration_spec": spec,
            "summary": {
                "n_runs": int(self.n_runs),
                "requested_runs": int(self.requested_runs),
                "alpha": float(self.alpha),
                "fpr": self._finite_or_none(self.fpr),
                "fpr_ci": [self._finite_or_none(value) for value in self.fpr_ci],
                "ks_p_value": self._finite_or_none(self.ks_p_value),
                "median_att_bias": self._finite_or_none(self.median_att_bias),
                "median_placebo_att": self._finite_or_none(self.median_placebo_att),
                "invalid_rate": self._finite_or_none(self.invalid_rate),
                "fpr_upper_bound": self._finite_or_none(self.fpr_upper_bound),
                "max_fpr": self._finite_or_none(self.max_fpr),
                "passed": bool(self.passed),
                "estimator_name": self.estimator_name,
                "design_fingerprint": self.design_fingerprint,
                "selection_scope": self.selection_scope,
                "assignment_mechanism": self.assignment_mechanism,
                "procedure_scope": self.procedure_scope,
                "failure_reasons": list(self.failure_reasons),
                "authoritative": bool(self.authoritative),
            },
        }

    def computed_fingerprint(self) -> str:
        canonical = json.dumps(
            self.integrity_payload(), sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self, include_details: bool = False) -> Dict[str, Any]:
        """Return a portable artifact payload; details are opt-in due to size."""
        payload: Dict[str, Any] = {
            "n_runs": int(self.n_runs),
            "requested_runs": int(self.requested_runs),
            "alpha": float(self.alpha),
            "fpr": self._finite_or_none(self.fpr),
            "fpr_ci": [self._finite_or_none(value) for value in self.fpr_ci],
            "ks_p_value": self._finite_or_none(self.ks_p_value),
            "median_att_bias": self._finite_or_none(self.median_att_bias),
            "median_placebo_att": self._finite_or_none(self.median_placebo_att),
            "invalid_rate": self._finite_or_none(self.invalid_rate),
            "fpr_upper_bound": self._finite_or_none(self.fpr_upper_bound),
            "max_fpr": self._finite_or_none(self.max_fpr),
            "passed": bool(self.passed),
            "estimator_name": self.estimator_name,
            "design_fingerprint": self.design_fingerprint,
            "calibration_fingerprint": self.calibration_fingerprint,
            "selection_scope": self.selection_scope,
            "assignment_mechanism": self.assignment_mechanism,
            "procedure_scope": self.procedure_scope,
            "failure_reasons": list(self.failure_reasons),
            "calibration_spec_json": self.calibration_spec_json,
            "authoritative": bool(self.authoritative),
        }
        if include_details:
            payload["details"] = json.loads(
                self.details.to_json(orient="records", date_format="iso")
            )
        return payload

    def to_json(self, include_details: bool = False) -> str:
        return json.dumps(
            self.to_dict(include_details=include_details),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AACalibration":
        def float_or_nan(item):
            return float(item) if item is not None else np.nan

        interval = list(value.get("fpr_ci", (None, None)))
        if len(interval) != 2:
            raise ValueError("fpr_ci deve conter dois limites")
        return cls(
            n_runs=int(value["n_runs"]),
            requested_runs=int(value.get("requested_runs", value["n_runs"])),
            alpha=float(value["alpha"]),
            fpr=float_or_nan(value.get("fpr")),
            fpr_ci=(float_or_nan(interval[0]), float_or_nan(interval[1])),
            ks_p_value=float_or_nan(value.get("ks_p_value")),
            median_att_bias=float_or_nan(value.get("median_att_bias")),
            median_placebo_att=float_or_nan(value.get("median_placebo_att")),
            invalid_rate=float_or_nan(value.get("invalid_rate")),
            fpr_upper_bound=float_or_nan(value.get("fpr_upper_bound")),
            max_fpr=float_or_nan(value.get("max_fpr")),
            passed=bool(value["passed"]),
            details=pd.DataFrame(value.get("details", [])),
            estimator_name=str(value.get("estimator_name", "")),
            design_fingerprint=str(value.get("design_fingerprint", "")),
            calibration_fingerprint=str(value.get("calibration_fingerprint", "")),
            selection_scope=str(value.get("selection_scope", "random_assignment")),
            assignment_mechanism=str(value.get("assignment_mechanism", "observational")),
            procedure_scope=str(value.get("procedure_scope", "single_estimator")),
            failure_reasons=tuple(value.get("failure_reasons", ())),
            calibration_spec_json=str(value.get("calibration_spec_json", "")),
            authoritative=bool(value.get("authoritative", False)),
        )

    @classmethod
    def from_json(cls, value: str) -> "AACalibration":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("AACalibration JSON deve codificar um objeto")
        return cls.from_dict(payload)

    def matches_design(self, design: DesignSpec) -> bool:
        """Verify artifact integrity and exact procedure compatibility."""
        if (
            not self.authoritative
            or not self.calibration_spec_json
            or not self.calibration_fingerprint
        ):
            return False
        try:
            payload = json.loads(self.calibration_spec_json)
        except (TypeError, json.JSONDecodeError):
            return False
        try:
            digest = self.computed_fingerprint()
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if digest != self.calibration_fingerprint:
            return False
        expected_selection_id = design.selection_procedure_id
        if (
            payload.get("selection_scope") == "random_assignment"
            and design.assignment_mechanism == "randomized"
        ):
            expected_selection_id = "random_assignment"
        expected = {
            "design_fingerprint": design.fingerprint,
            "estimator_names": list(design.estimator_names),
            "decision_rule": design.decision_rule.to_dict(),
            "pre_days": design.pre_days,
            "post_days": design.post_days,
            "anticipation_days": design.anticipation_days,
            "n_treated": len(design.treated_units),
            "n_donors": len(design.donor_units),
            "alpha": design.alpha,
            "max_group_placebos": design.max_group_placebos,
            "seed": design.calibration_seed,
            "permutation_seed": design.permutation_seed,
            "assignment_mechanism": design.assignment_mechanism,
            "max_pre_rmspe": design.max_pre_rmspe,
            "fit_kwargs_by_estimator": json.loads(design.estimator_config_json),
            "selection_procedure_id": expected_selection_id,
            "eligible_units": list(design.eligible_units),
            "selector_config": json.loads(design.selector_config_json),
            "requested_runs": design.calibration_n_runs,
            "min_valid_runs": design.calibration_min_valid_runs,
            "min_ks_p_value": design.calibration_min_ks_p_value,
            "max_fpr_inflation": design.calibration_max_fpr_inflation,
            "max_invalid_rate": design.calibration_max_invalid_rate,
        }
        return all(payload.get(key) == value for key, value in expected.items())


def run_aa_calibration(
    panel: CityPanel,
    eligible: Sequence[str],
    fit_fn: Callable,
    pre_days: int,
    post_days: int,
    n_runs: int = 200,
    n_treated: int = 2,
    n_donors: int = 15,
    alpha: float = 0.10,
    seed: int = 123,
    fit_kwargs: Optional[dict] = None,
    min_valid_runs: int = 100,
    min_ks_p_value: float = 0.01,
    max_group_placebos: Optional[int] = 30,
    permutation_seed: Optional[int] = 123,
    max_fpr_inflation: float = 0.05,
    max_invalid_rate: float = 0.05,
    selection_fn: Optional[
        Callable[[CityPanel, Sequence[str], np.random.Generator, int, int], tuple]
    ] = None,
    design_fingerprint: str = "",
    fit_fns: Optional[Mapping[str, Callable]] = None,
    fit_kwargs_by_estimator: Optional[Mapping[str, dict]] = None,
    decision_rule: Optional[DecisionRule] = None,
    anticipation_days: int = 0,
    assignment_mechanism: str = "observational",
    max_pre_rmspe: Optional[float] = 0.10,
    selection_procedure_id: str = "",
    selector_config_json: Optional[str] = None,
    design_spec: Optional[DesignSpec] = None,
) -> AACalibration:
    from scipy.stats import beta as beta_dist
    from scipy.stats import kstest

    rng = np.random.default_rng(seed)
    eligible = sorted(str(city).strip() for city in eligible)
    if any(not city for city in eligible):
        raise ValueError("eligible contém cidade vazia")
    if len(eligible) != len(set(eligible)):
        raise ValueError("eligible contém cidades duplicadas")
    missing_eligible = sorted(set(eligible) - set(panel.cities))
    if missing_eligible:
        raise ValueError(f"eligible contém cidades fora do painel: {missing_eligible}")
    resolved_selector_config = (
        design_spec.selector_config_json
        if selector_config_json is None and design_spec is not None
        else ("{}" if selector_config_json is None else selector_config_json)
    )
    try:
        selector_config = json.loads(resolved_selector_config)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("selector_config_json deve ser JSON válido") from exc
    if not isinstance(selector_config, dict):
        raise ValueError("selector_config_json deve codificar um objeto JSON")
    canonical_selector_config = json.dumps(
        selector_config, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )
    T = len(panel.index)
    if n_treated < 1:
        raise ValueError("n_treated deve ser >= 1")
    if n_donors < 2:
        raise ValueError("n_donors deve ser >= 2")
    if len(eligible) < n_treated + n_donors:
        raise ValueError(
            f"Elegíveis insuficientes: precisa de {n_treated + n_donors}, "
            f"recebeu {len(eligible)}"
        )
    if anticipation_days < 0:
        raise ValueError("anticipation_days deve ser >= 0")
    if assignment_mechanism not in {"observational", "randomized"}:
        raise ValueError("assignment_mechanism deve ser 'observational' ou 'randomized'")
    if max_pre_rmspe is not None and (
        not np.isfinite(max_pre_rmspe) or max_pre_rmspe <= 0
    ):
        raise ValueError("max_pre_rmspe deve ser finito e > 0, ou None")
    if selection_fn is not None and not selection_procedure_id.strip():
        raise ValueError(
            "selection_procedure_id versionado é obrigatório quando selection_fn é usado"
        )
    if fit_fns is None:
        estimator_name = getattr(fit_fn, "__name__", fit_fn.__class__.__name__)
        method_name = str(estimator_name).removeprefix("fit_")
        procedure_scope = "single_estimator"
        estimator_names = [method_name]
        rule_payload = decision_rule.to_dict() if decision_rule else None
        method_kwargs = {method_name: dict(fit_kwargs or {})}
    else:
        if decision_rule is None:
            raise ValueError("decision_rule é obrigatório quando fit_fns é fornecido")
        if not fit_fns or any(not callable(fn) for fn in fit_fns.values()):
            raise ValueError("fit_fns deve mapear nomes para callables")
        unknown_kwargs = set(fit_kwargs_by_estimator or {}) - set(fit_fns)
        if unknown_kwargs:
            raise ValueError(
                "fit_kwargs_by_estimator referencia estimadores ausentes: "
                f"{sorted(unknown_kwargs)}"
            )
        estimator_names = sorted(fit_fns)
        estimator_name = "joint[" + ",".join(estimator_names) + "]"
        procedure_scope = "joint_decision"
        rule_payload = decision_rule.to_dict()
        method_kwargs = {
            name: dict((fit_kwargs_by_estimator or {}).get(name, fit_kwargs or {}))
            for name in estimator_names
        }
    scope = "selected_design" if selection_fn is not None else "random_assignment"
    selection_id = (
        selection_procedure_id.strip() if selection_fn is not None else "random_assignment"
    )
    if design_spec is not None:
        design_spec.validate()
        require_runtime_version(design_spec.implementation_version)
        _require_builtin_estimators(
            fit_fns or {estimator_names[0]: fit_fn}
        )
        if design_fingerprint and design_fingerprint != design_spec.fingerprint:
            raise ValueError("design_fingerprint diverge do DesignSpec")
        design_fingerprint = design_spec.fingerprint
        requires_universe = (
            selection_fn is not None or assignment_mechanism == "randomized"
        )
        if requires_universe and tuple(eligible) != design_spec.eligible_units:
            raise ValueError("eligible diverge de eligible_units do DesignSpec")
        if canonical_selector_config != design_spec.selector_config_json:
            raise ValueError("selector_config_json diverge do DesignSpec")
        if (
            selection_fn is not None
            and selection_procedure_id.strip()
            != design_spec.selection_procedure_id
        ):
            raise ValueError("selection_procedure_id diverge do DesignSpec")
        design_selection_id = design_spec.selection_procedure_id
        if selection_fn is None and assignment_mechanism == "randomized":
            design_selection_id = "random_assignment"
        expected = {
            "pre_days": pre_days,
            "post_days": post_days,
            "anticipation_days": anticipation_days,
            "n_treated": n_treated,
            "n_donors": n_donors,
            "alpha": alpha,
            "max_group_placebos": max_group_placebos,
            "seed": seed,
            "permutation_seed": permutation_seed,
            "assignment_mechanism": assignment_mechanism,
            "max_pre_rmspe": max_pre_rmspe,
            "estimator_names": estimator_names,
            "decision_rule": rule_payload,
            "fit_kwargs_by_estimator": method_kwargs,
            "selection_procedure_id": selection_id,
            "requested_runs": n_runs,
            "min_valid_runs": min_valid_runs,
            "min_ks_p_value": min_ks_p_value,
            "max_fpr_inflation": max_fpr_inflation,
            "max_invalid_rate": max_invalid_rate,
        }
        actual = {
            "pre_days": design_spec.pre_days,
            "post_days": design_spec.post_days,
            "anticipation_days": design_spec.anticipation_days,
            "n_treated": len(design_spec.treated_units),
            "n_donors": len(design_spec.donor_units),
            "alpha": design_spec.alpha,
            "max_group_placebos": design_spec.max_group_placebos,
            "seed": design_spec.calibration_seed,
            "permutation_seed": design_spec.permutation_seed,
            "assignment_mechanism": design_spec.assignment_mechanism,
            "max_pre_rmspe": design_spec.max_pre_rmspe,
            "estimator_names": list(design_spec.estimator_names),
            "decision_rule": design_spec.decision_rule.to_dict(),
            "fit_kwargs_by_estimator": json.loads(
                design_spec.estimator_config_json
            ),
            "selection_procedure_id": design_selection_id,
            "requested_runs": design_spec.calibration_n_runs,
            "min_valid_runs": design_spec.calibration_min_valid_runs,
            "min_ks_p_value": design_spec.calibration_min_ks_p_value,
            "max_fpr_inflation": design_spec.calibration_max_fpr_inflation,
            "max_invalid_rate": design_spec.calibration_max_invalid_rate,
        }
        mismatched = [name for name in expected if expected[name] != actual[name]]
        if mismatched:
            raise ValueError(f"argumentos divergem do DesignSpec: {mismatched}")
    if T < pre_days + anticipation_days + post_days:
        raise ValueError(
            "Histórico insuficiente: precisa de >= "
            f"{pre_days + anticipation_days + post_days} dias"
        )
    if n_runs < 1 or min_valid_runs < 1 or min_valid_runs > n_runs:
        raise ValueError("use 1 <= min_valid_runs <= n_runs")
    if max_fpr_inflation < 0 or alpha + max_fpr_inflation >= 1:
        raise ValueError("max_fpr_inflation inválido")
    if not (0.0 <= max_invalid_rate < 1.0):
        raise ValueError("max_invalid_rate deve estar em [0, 1)")

    rows: List[dict] = []
    for run in range(n_runs):
        start = int(rng.integers(pre_days + anticipation_days, T - post_days + 1))
        if selection_fn is None:
            cities = rng.choice(eligible, size=n_treated + n_donors, replace=False).tolist()
            treated, donors = cities[:n_treated], cities[n_treated:]
        else:
            selected = selection_fn(panel, eligible, rng, run, start)
            if not isinstance(selected, tuple) or len(selected) != 2:
                raise ValueError("selection_fn deve retornar (treated, donors)")
            treated, donors = list(selected[0]), list(selected[1])
            if len(treated) != n_treated or len(donors) != n_donors:
                raise ValueError("selection_fn retornou tamanhos incompatíveis")
            outside_eligible = sorted(
                (set(str(city) for city in treated) | set(str(city) for city in donors))
                - set(eligible)
            )
            if outside_eligible:
                raise ValueError(
                    "selection_fn retornou cidades fora de eligible: "
                    f"{outside_eligible}"
                )
        try:
            if fit_fns is None:
                r = simulate_once(
                    panel, treated, donors, fit_fn, start,
                    pre_days, post_days, delta=0.0, alpha=alpha,
                    fit_kwargs=fit_kwargs,
                    max_group_placebos=max_group_placebos,
                    permutation_seed=permutation_seed,
                    anticipation_days=anticipation_days,
                    assignment_mechanism=assignment_mechanism,
                    max_pre_rmspe=max_pre_rmspe,
                )
            else:
                if decision_rule is None:
                    raise ValueError("decision_rule é obrigatório quando fit_fns é fornecido")
                r = _decision_simulation(
                    panel, treated, donors, fit_fns, start,
                    pre_days, post_days, 0.0, alpha, decision_rule,
                    fit_kwargs, dict(fit_kwargs_by_estimator or {}), rng,
                    max_group_placebos, permutation_seed, anticipation_days,
                    assignment_mechanism, max_pre_rmspe,
                )
                method_atts = [
                    result.get("att_pct", np.nan)
                    for result in r.get("method_results", {}).values()
                ]
                r["att_pct"] = (
                    float(np.nanmedian(method_atts)) if method_atts else np.nan
                )
        except Exception as e:  # pragma: no cover
            rows.append({
                "run": run, "p_value": np.nan, "reject": False,
                "valid": False, "error": str(e),
            })
            continue
        r["valid"] = bool(r.get("valid", r.get("valid_for_decision", True)))
        r["run"] = run
        rows.append(r)

    df = pd.DataFrame(rows)
    finite_p = np.isfinite(df["p_value"])
    validity_gate = df["valid"].fillna(False).astype(bool)
    valid = df[finite_p & validity_gate]
    n = len(valid)
    fpr = float(valid["reject"].mean()) if n else np.nan
    k = int(valid["reject"].sum())
    # IC binomial exato (Clopper-Pearson) + limite unilateral de inflação.
    if n:
        lo = float(beta_dist.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
        hi = float(beta_dist.ppf(0.975, k + 1, n - k)) if k < n else 1.0
        upper = float(beta_dist.ppf(0.95, k + 1, n - k)) if k < n else 1.0
    else:
        lo, hi, upper = 0.0, 1.0, 1.0

    # Sob o rank nulo de um estimador, p=k/(M+1) é uniforme numa grade discreta. O PIT
    # aleatorizado U=(k-1+V)/(M+1) transforma essa referência em U(0,1),
    # permitindo usar o KS contínuo sem o erro de ties do teste anterior. Uma
    # regra conjunta expõe abaixo um mínimo de p-values apenas como diagnóstico;
    # esse mínimo não é uniforme, então não se aplica o KS marginal a ele.
    ks_p = np.nan
    if fit_fns is None and n >= 20 and "n_placebos" in valid:
        pit_rng = np.random.default_rng(seed + 1_000_003)
        uniforms = []
        for p_value, m in zip(valid["p_value"], valid["n_placebos"], strict=True):
            grid = int(m) + 1
            rank = int(np.clip(round(float(p_value) * grid), 1, grid))
            uniforms.append((rank - 1 + pit_rng.random()) / grid)
        ks_p = float(kstest(uniforms, "uniform").pvalue)

    real_att_col = next(
        (name for name in ("att_pct", "real_att_pct") if name in valid.columns), None,
    )
    med_bias = (
        float(valid[real_att_col].median()) if n and real_att_col is not None else np.nan
    )
    med_placebo = (
        float(valid["att_pct_placebo_med"].median())
        if n and "att_pct_placebo_med" in valid else np.nan
    )
    invalid_rate = float(1.0 - n / n_runs)
    max_fpr = alpha + max_fpr_inflation
    failures = []
    if n < min_valid_runs:
        failures.append(f"valid_runs={n} < {min_valid_runs}")
    if invalid_rate > max_invalid_rate:
        failures.append(f"invalid_rate={invalid_rate:.3f} > {max_invalid_rate:.3f}")
    if upper > max_fpr:
        failures.append(f"FPR upper bound={upper:.3f} > {max_fpr:.3f}")
    if fit_fns is None and (not np.isfinite(ks_p) or ks_p < min_ks_p_value):
        failures.append(f"discrete-rank PIT KS p={ks_p!r} < {min_ks_p_value:.3f}")

    payload = {
        "estimator": estimator_name,
        "estimator_names": estimator_names,
        "decision_rule": rule_payload,
        "design_fingerprint": design_fingerprint,
        "pre_days": pre_days, "post_days": post_days,
        "anticipation_days": anticipation_days,
        "assignment_mechanism": assignment_mechanism,
        "max_pre_rmspe": max_pre_rmspe,
        "n_treated": n_treated, "n_donors": n_donors,
        "alpha": alpha, "max_group_placebos": max_group_placebos,
        "seed": seed, "permutation_seed": permutation_seed,
        "selection_scope": scope,
        "selection_procedure_id": selection_id,
        "eligible_units": eligible,
        "selector_config": selector_config,
        "fit_kwargs_by_estimator": method_kwargs,
        "requested_runs": n_runs,
        "min_valid_runs": min_valid_runs,
        "min_ks_p_value": min_ks_p_value,
        "max_fpr_inflation": max_fpr_inflation,
        "max_invalid_rate": max_invalid_rate,
    }
    calibration_spec_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    result = AACalibration(
        n_runs=n, alpha=alpha, fpr=fpr, fpr_ci=(lo, hi),
        ks_p_value=ks_p, median_att_bias=med_bias,
        passed=not failures, details=df, requested_runs=n_runs,
        invalid_rate=invalid_rate, fpr_upper_bound=upper, max_fpr=max_fpr,
        median_placebo_att=med_placebo, estimator_name=estimator_name,
        design_fingerprint=design_fingerprint,
        calibration_fingerprint="",
        selection_scope=scope, assignment_mechanism=assignment_mechanism,
        procedure_scope=procedure_scope,
        failure_reasons=tuple(failures),
        calibration_spec_json=calibration_spec_json,
        authoritative=design_spec is not None,
    )
    result.calibration_fingerprint = result.computed_fingerprint()
    return result
