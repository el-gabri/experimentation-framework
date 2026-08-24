"""Immutable, serializable contracts for geo-experiment design.

The fingerprint is a SHA-256 digest of canonical JSON.  It binds power,
calibration, registration and analysis to the same units, estimator family,
inference resolution and executable decision rule.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from supply_experiments._version import IMPLEMENTATION_VERSION


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def fingerprint_payload(value: Any) -> str:
    """Return a stable SHA-256 fingerprint for a JSON-serializable payload."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalized_names(values: Tuple[str, ...], label: str) -> Tuple[str, ...]:
    names = tuple(sorted(str(value).strip() for value in values))
    if not names or any(not name for name in names):
        raise ValueError(f"{label} não pode ser vazio")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} contém valores duplicados")
    return names


@dataclass(frozen=True)
class DecisionRule:
    """Serializable decision rule executable by simulation-based power.

    ``min_rejections`` implements rules such as "at least two of three
    estimators reject".  Direction is applied to the estimated relative ATT,
    not to the injected effect label.
    """

    min_rejections: int = 1
    direction: str = "any"  # any | positive | negative
    methods: Tuple[str, ...] = ()
    require_all_valid: bool = True
    require_decision_validity: bool = True
    rule_type: str = "minimum_rejections"

    def __post_init__(self) -> None:
        if self.rule_type != "minimum_rejections":
            raise ValueError(f"rule_type não suportado: {self.rule_type!r}")
        if self.min_rejections < 1:
            raise ValueError("min_rejections deve ser >= 1")
        if self.direction not in {"any", "positive", "negative"}:
            raise ValueError("direction deve ser 'any', 'positive' ou 'negative'")
        methods = tuple(sorted(str(method).strip() for method in self.methods))
        if any(not method for method in methods) or len(set(methods)) != len(methods):
            raise ValueError("methods contém nome vazio ou duplicado")
        object.__setattr__(self, "methods", methods)

    def evaluate(self, results: Mapping[str, Mapping[str, Any]], alpha: float) -> bool:
        """Evaluate estimator results containing ``p_value`` and ``att_pct``."""
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha deve estar em (0, 1)")
        methods = self.methods or tuple(sorted(results))
        if self.min_rejections > len(methods):
            raise ValueError("min_rejections excede o número de estimadores da regra")

        valid = []
        rejected = 0
        for method in methods:
            result = results.get(method)
            if result is None:
                valid.append(False)
                continue
            p_value = float(result.get("p_value", np.nan))
            att_pct = float(result.get("att_pct", np.nan))
            method_valid = np.isfinite(p_value)
            if self.require_decision_validity:
                method_valid = method_valid and result.get("valid_for_decision") is True
            if self.direction != "any":
                method_valid = method_valid and np.isfinite(att_pct)
            valid.append(bool(method_valid))
            if not method_valid or p_value > alpha:
                continue
            if self.direction == "positive" and att_pct <= 0:
                continue
            if self.direction == "negative" and att_pct >= 0:
                continue
            rejected += 1

        if self.require_all_valid and not all(valid):
            return False
        return rejected >= self.min_rejections

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_type": self.rule_type,
            "min_rejections": self.min_rejections,
            "direction": self.direction,
            "methods": list(self.methods),
            "require_all_valid": self.require_all_valid,
            "require_decision_validity": self.require_decision_validity,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecisionRule":
        return cls(
            rule_type=str(value.get("rule_type", "minimum_rejections")),
            min_rejections=int(value.get("min_rejections", 1)),
            direction=str(value.get("direction", "any")),
            methods=tuple(value.get("methods", ())),
            require_all_valid=bool(value.get("require_all_valid", True)),
            require_decision_validity=bool(value.get("require_decision_validity", True)),
        )


@dataclass(frozen=True)
class DesignSpec:
    """Complete immutable design used to produce a power/calibration result."""

    treated_units: Tuple[str, ...]
    donor_units: Tuple[str, ...]
    estimator_names: Tuple[str, ...]
    pre_days: int
    post_days: int
    experiment_id: str = ""
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    alpha: float = 0.10
    max_group_placebos: Optional[int] = 30
    effect_grid: Tuple[float, ...] = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)
    decision_rule: DecisionRule = field(default_factory=DecisionRule)
    primary_kpi: str = "outcome"
    estimand: str = "average_treated_relative_att"
    assignment_mechanism: str = "observational"  # observational | randomized
    selection_procedure_id: str = "fixed_prespecified"
    eligible_units: Tuple[str, ...] = ()
    selector_config_json: str = "{}"
    max_pre_rmspe: Optional[float] = 0.10
    anticipation_days: int = 0
    require_calibration: bool = True
    calibration_scope: str = "exact_design"  # exact_design | estimator_only | none
    calibration_fingerprint: Optional[str] = None
    calibration_seed: int = 123
    calibration_n_runs: int = 200
    calibration_min_valid_runs: int = 100
    calibration_min_ks_p_value: float = 0.01
    calibration_max_fpr_inflation: float = 0.05
    calibration_max_invalid_rate: float = 0.05
    power_target: float = 0.80
    power_n_sims_per_point: int = 30
    power_min_valid_sims_per_point: int = 20
    power_max_invalid_rate: float = 0.05
    power_effect_model: str = "constant_multiplicative"
    seed: int = 42
    permutation_seed: Optional[int] = 123
    window_spacing_days: Optional[int] = None
    estimator_config_json: str = "{}"
    schema_version: str = "2.0"
    implementation_version: str = IMPLEMENTATION_VERSION

    def __post_init__(self) -> None:
        treated = _normalized_names(tuple(self.treated_units), "treated_units")
        donors = _normalized_names(tuple(self.donor_units), "donor_units")
        estimators = _normalized_names(tuple(self.estimator_names), "estimator_names")
        object.__setattr__(self, "treated_units", treated)
        object.__setattr__(self, "donor_units", donors)
        object.__setattr__(self, "estimator_names", estimators)
        eligible = tuple(sorted(str(value).strip() for value in self.eligible_units))
        if any(not name for name in eligible) or len(set(eligible)) != len(eligible):
            raise ValueError("eligible_units contém nome vazio ou duplicado")
        object.__setattr__(self, "eligible_units", eligible)
        object.__setattr__(self, "experiment_id", str(self.experiment_id).strip())

        for field_name in ("start_date", "end_date"):
            value = getattr(self, field_name)
            if isinstance(value, datetime):
                object.__setattr__(self, field_name, value.date())
            elif value is not None and not isinstance(value, date):
                raise TypeError(f"{field_name} deve ser datetime.date ou None")

        overlap = set(treated) & set(donors)
        if overlap:
            raise ValueError(f"tratadas também aparecem como doadoras: {sorted(overlap)}")
        if eligible:
            outside_universe = sorted((set(treated) | set(donors)) - set(eligible))
            if outside_universe:
                raise ValueError(
                    "treated/donor fora de eligible_units: "
                    f"{outside_universe}"
                )
        if self.pre_days <= 0 or self.post_days <= 0:
            raise ValueError("pre_days e post_days devem ser > 0")
        if (self.start_date is None) != (self.end_date is None):
            raise ValueError("start_date e end_date devem ser fornecidas juntas")
        if self.start_date is not None and self.end_date is not None:
            if self.start_date > self.end_date:
                raise ValueError("start_date é posterior a end_date")
            observed_post_days = (self.end_date - self.start_date).days + 1
            if observed_post_days != self.post_days:
                raise ValueError(
                    f"post_days={self.post_days} diverge das datas ({observed_post_days})"
                )
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha deve estar em (0, 1)")
        if self.max_group_placebos is not None and self.max_group_placebos < 1:
            raise ValueError("max_group_placebos deve ser >= 1 ou None")
        if self.assignment_mechanism == "randomized":
            if len(donors) < 2:
                raise ValueError("alocação randomizada exige ao menos duas controles")
            total_placebos = (
                math.comb(len(treated) + len(donors), len(treated)) - 1
            )
        else:
            if len(donors) - len(treated) < 2:
                raise ValueError(
                    "placebos observacionais precisam deixar ao menos duas doadoras"
                )
            total_placebos = math.comb(len(donors), len(treated))
        if self.max_group_placebos is None and total_placebos > 10_000:
            raise ValueError(
                "mais de 10.000 alocações placebo exigem max_group_placebos "
                "explícito para Monte Carlo"
            )
        available_placebos = (
            total_placebos
            if self.max_group_placebos is None
            else min(total_placebos, self.max_group_placebos)
        )
        required_placebos = math.ceil((1.0 / self.alpha) - 1.0 - 1e-12)
        if available_placebos < required_placebos:
            raise ValueError(
                "resolução de permutação insuficiente: "
                f"{available_placebos} placebos permitem p mínimo "
                f"{1.0 / (available_placebos + 1):.4f} > alpha={self.alpha:.4f}"
            )

        effects = tuple(sorted({float(effect) for effect in self.effect_grid}))
        if not effects or not all(np.isfinite(effect) for effect in effects):
            raise ValueError("effect_grid deve conter valores finitos")
        if not any(np.isclose(effect, 0.0) for effect in effects):
            raise ValueError("effect_grid deve incluir 0 para estimar o FPR")
        if any(effect <= -1.0 for effect in effects):
            raise ValueError("efeitos multiplicativos devem ser > -1")
        object.__setattr__(self, "effect_grid", effects)

        if self.decision_rule.methods:
            unknown = set(self.decision_rule.methods) - set(estimators)
            if unknown:
                raise ValueError(f"regra de decisão referencia estimadores ausentes: {sorted(unknown)}")
            n_rule_methods = len(self.decision_rule.methods)
        else:
            n_rule_methods = len(estimators)
        if self.decision_rule.min_rejections > n_rule_methods:
            raise ValueError("min_rejections excede os estimadores do design")

        spacing = 1 if self.window_spacing_days is None else self.window_spacing_days
        if spacing < 1:
            raise ValueError("window_spacing_days deve ser >= 1")
        if self.schema_version == "2.0" and spacing != 1:
            raise ValueError(
                "DesignSpec schema 2.0 exige window_spacing_days=1 para "
                "simulações iid com reposição"
            )
        object.__setattr__(self, "window_spacing_days", int(spacing))
        if not self.primary_kpi.strip() or not self.estimand.strip():
            raise ValueError("primary_kpi e estimand não podem ser vazios")
        if self.estimand != "average_treated_relative_att":
            raise ValueError(
                "estimand não suportado: o pipeline implementa somente "
                "'average_treated_relative_att'"
            )
        if not self.implementation_version.strip():
            raise ValueError("implementation_version não pode ser vazio")
        if self.assignment_mechanism not in {"observational", "randomized"}:
            raise ValueError("assignment_mechanism deve ser 'observational' ou 'randomized'")
        if not self.selection_procedure_id.strip():
            raise ValueError("selection_procedure_id não pode ser vazio")
        if (
            self.schema_version != "1.0"
            and self.selection_procedure_id != "fixed_prespecified"
            and not eligible
        ):
            raise ValueError(
                "seleção não fixa exige eligible_units no contrato de design"
            )
        if self.max_pre_rmspe is not None and (
            not np.isfinite(self.max_pre_rmspe) or self.max_pre_rmspe <= 0
        ):
            raise ValueError("max_pre_rmspe deve ser finito e > 0, ou None")
        if self.anticipation_days < 0:
            raise ValueError("anticipation_days deve ser >= 0")
        if self.calibration_scope not in {"exact_design", "estimator_only", "none"}:
            raise ValueError(
                "calibration_scope deve ser 'exact_design', 'estimator_only' ou 'none'"
            )
        if self.require_calibration and self.calibration_scope == "none":
            raise ValueError("require_calibration=True exige um calibration_scope")
        if self.calibration_n_runs < 1:
            raise ValueError("calibration_n_runs deve ser >= 1")
        if not (1 <= self.calibration_min_valid_runs <= self.calibration_n_runs):
            raise ValueError(
                "calibration_min_valid_runs deve estar entre 1 e calibration_n_runs"
            )
        if not (0.0 <= self.calibration_min_ks_p_value <= 1.0):
            raise ValueError("calibration_min_ks_p_value deve estar em [0, 1]")
        if (
            self.calibration_max_fpr_inflation < 0
            or self.alpha + self.calibration_max_fpr_inflation >= 1.0
        ):
            raise ValueError("calibration_max_fpr_inflation é incompatível com alpha")
        if not (0.0 <= self.calibration_max_invalid_rate < 1.0):
            raise ValueError("calibration_max_invalid_rate deve estar em [0, 1)")
        if not (0.0 < self.power_target < 1.0):
            raise ValueError("power_target deve estar em (0, 1)")
        if self.power_n_sims_per_point < 1:
            raise ValueError("power_n_sims_per_point deve ser >= 1")
        if not (
            1
            <= self.power_min_valid_sims_per_point
            <= self.power_n_sims_per_point
        ):
            raise ValueError(
                "power_min_valid_sims_per_point deve estar entre 1 e "
                "power_n_sims_per_point"
            )
        if not (0.0 <= self.power_max_invalid_rate < 1.0):
            raise ValueError("power_max_invalid_rate deve estar em [0, 1)")
        if self.power_effect_model != "constant_multiplicative":
            raise ValueError(
                "power_effect_model suportado: 'constant_multiplicative'"
            )
        if self.calibration_fingerprint is not None:
            digest = self.calibration_fingerprint.lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("calibration_fingerprint deve ser SHA-256 hexadecimal")
            object.__setattr__(self, "calibration_fingerprint", digest)

        try:
            selector_config = json.loads(self.selector_config_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("selector_config_json deve ser JSON válido") from exc
        if not isinstance(selector_config, dict):
            raise ValueError("selector_config_json deve codificar um objeto JSON")
        object.__setattr__(
            self, "selector_config_json", _canonical_json(selector_config)
        )

        try:
            config = json.loads(self.estimator_config_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("estimator_config_json deve ser JSON válido") from exc
        if not isinstance(config, dict):
            raise ValueError("estimator_config_json deve codificar um objeto JSON")
        unknown_config = set(config) - set(estimators)
        if unknown_config:
            raise ValueError(
                f"configuração referencia estimadores ausentes: {sorted(unknown_config)}"
            )
        if any(not isinstance(params, dict) for params in config.values()):
            raise ValueError("cada configuração de estimador deve ser um objeto JSON")
        # Persist even empty configs by method so downstream code cannot
        # silently apply an unbound shared kwargs dictionary.
        method_config = {name: dict(config.get(name, {})) for name in estimators}
        if self.schema_version == "2.0" and "sdid" in method_config:
            sdid_config = method_config["sdid"]
            if sdid_config.get("treated_aggregation", "mean") != "mean":
                raise ValueError(
                    "DesignSpec schema 2.0 exige SDID treated_aggregation='mean' "
                    "para o estimando average_treated_relative_att"
                )
            configured_count = sdid_config.get("n_treated_units", len(treated))
            if (
                isinstance(configured_count, bool)
                or not isinstance(configured_count, (int, float))
                or not np.isfinite(configured_count)
                or int(configured_count) != configured_count
                or int(configured_count) != len(treated)
            ):
                raise ValueError(
                    "n_treated_units do SDID deve ser igual ao número de "
                    "treated_units registrado"
                )
        object.__setattr__(self, "estimator_config_json", _canonical_json(method_config))

    @property
    def fingerprint(self) -> str:
        # Calibration is downstream of design and may itself include this
        # fingerprint. Bind the requirement and scope, but omit only the
        # downstream artifact digest to avoid a circular hash.
        payload = self.to_dict()
        calibration = dict(payload["calibration"])
        calibration.pop("fingerprint", None)
        payload["calibration"] = calibration
        return fingerprint_payload(payload)

    def validate(self) -> "DesignSpec":
        """Explicit validation hook; construction already validates eagerly."""
        DesignSpec.from_dict(self.to_dict())
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "implementation_version": self.implementation_version,
            "treated_units": list(self.treated_units),
            "donor_units": list(self.donor_units),
            "estimator_names": list(self.estimator_names),
            "pre_days": self.pre_days,
            "post_days": self.post_days,
            "experiment_id": self.experiment_id,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "alpha": self.alpha,
            "max_group_placebos": self.max_group_placebos,
            "effect_grid": list(self.effect_grid),
            "decision_rule": self.decision_rule.to_dict(),
            "primary_kpi": self.primary_kpi,
            "estimand": self.estimand,
            "assignment_mechanism": self.assignment_mechanism,
            "selection_procedure_id": self.selection_procedure_id,
            "eligible_units": list(self.eligible_units),
            "selector_config": json.loads(self.selector_config_json),
            "max_pre_rmspe": self.max_pre_rmspe,
            "anticipation_days": self.anticipation_days,
            "calibration": {
                "required": self.require_calibration,
                "scope": self.calibration_scope,
                "fingerprint": self.calibration_fingerprint,
                "seed": self.calibration_seed,
                "n_runs": self.calibration_n_runs,
                "min_valid_runs": self.calibration_min_valid_runs,
                "min_ks_p_value": self.calibration_min_ks_p_value,
                "max_fpr_inflation": self.calibration_max_fpr_inflation,
                "max_invalid_rate": self.calibration_max_invalid_rate,
            },
            "power": {
                "target": self.power_target,
                "n_sims_per_point": self.power_n_sims_per_point,
                "min_valid_sims_per_point": self.power_min_valid_sims_per_point,
                "max_invalid_rate": self.power_max_invalid_rate,
                "effect_model": self.power_effect_model,
            },
            "seed": self.seed,
            "permutation_seed": self.permutation_seed,
            "window_spacing_days": self.window_spacing_days,
            "estimator_config": json.loads(self.estimator_config_json),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DesignSpec":
        return cls(
            schema_version=str(value.get("schema_version", "1.0")),
            implementation_version=str(
                value.get("implementation_version", "legacy-unbound")
            ),
            treated_units=tuple(value["treated_units"]),
            donor_units=tuple(value["donor_units"]),
            estimator_names=tuple(value["estimator_names"]),
            pre_days=int(value["pre_days"]),
            post_days=int(value["post_days"]),
            experiment_id=str(value.get("experiment_id", "")),
            start_date=(
                date.fromisoformat(str(value["start_date"]))
                if value.get("start_date")
                else None
            ),
            end_date=(
                date.fromisoformat(str(value["end_date"]))
                if value.get("end_date")
                else None
            ),
            alpha=float(value.get("alpha", 0.10)),
            max_group_placebos=value.get("max_group_placebos", 30),
            effect_grid=tuple(value.get("effect_grid", (0.0,))),
            decision_rule=DecisionRule.from_dict(value.get("decision_rule", {})),
            primary_kpi=str(value.get("primary_kpi", "outcome")),
            estimand=str(value.get("estimand", "average_treated_relative_att")),
            assignment_mechanism=str(value.get("assignment_mechanism", "observational")),
            selection_procedure_id=str(
                value.get("selection_procedure_id", "fixed_prespecified")
            ),
            eligible_units=tuple(value.get("eligible_units", ())),
            selector_config_json=_canonical_json(value.get("selector_config", {})),
            max_pre_rmspe=value.get("max_pre_rmspe", 0.10),
            anticipation_days=int(value.get("anticipation_days", 0)),
            require_calibration=bool(value.get("calibration", {}).get("required", True)),
            calibration_scope=str(value.get("calibration", {}).get("scope", "exact_design")),
            calibration_fingerprint=value.get("calibration", {}).get("fingerprint"),
            calibration_seed=int(
                value.get("calibration", {}).get("seed", 123)
            ),
            calibration_n_runs=int(
                value.get("calibration", {}).get("n_runs", 200)
            ),
            calibration_min_valid_runs=int(
                value.get("calibration", {}).get("min_valid_runs", 100)
            ),
            calibration_min_ks_p_value=float(
                value.get("calibration", {}).get("min_ks_p_value", 0.01)
            ),
            calibration_max_fpr_inflation=float(
                value.get("calibration", {}).get("max_fpr_inflation", 0.05)
            ),
            calibration_max_invalid_rate=float(
                value.get("calibration", {}).get("max_invalid_rate", 0.05)
            ),
            power_target=float(value.get("power", {}).get("target", 0.80)),
            power_n_sims_per_point=int(
                value.get("power", {}).get("n_sims_per_point", 30)
            ),
            power_min_valid_sims_per_point=int(
                value.get("power", {}).get(
                    "min_valid_sims_per_point",
                    min(
                        20,
                        int(value.get("power", {}).get("n_sims_per_point", 30)),
                    ),
                )
            ),
            power_max_invalid_rate=float(
                value.get("power", {}).get("max_invalid_rate", 0.05)
            ),
            power_effect_model=str(
                value.get("power", {}).get(
                    "effect_model", "constant_multiplicative"
                )
            ),
            seed=int(value.get("seed", 42)),
            permutation_seed=value.get("permutation_seed"),
            window_spacing_days=value.get("window_spacing_days"),
            estimator_config_json=_canonical_json(value.get("estimator_config", {})),
        )

    @classmethod
    def from_json(cls, value: str) -> "DesignSpec":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("DesignSpec JSON deve codificar um objeto")
        return cls.from_dict(payload)
