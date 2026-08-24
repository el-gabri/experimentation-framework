"""Simulation-based power on historical geo panels.

Power is evaluated with the same estimator(s), permutation resolution and
executable decision rule recorded by :class:`DesignSpec`. Legacy calls remain
supported, but their default start spacing is one day and therefore reuses
overlapping post windows; the result records that spacing explicitly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd

from supply_experiments._version import require_runtime_version
from supply_experiments.design.spec import (
    DecisionRule,
    DesignSpec,
    fingerprint_payload,
)
from supply_experiments.inference.permutation import placebo_inference
from supply_experiments.panel import CityPanel


@dataclass
class PowerResult:
    effect_grid: List[float]
    power: Dict[float, float]
    fpr: float
    mde_80: Optional[float]
    n_sims_per_point: int
    alpha: float
    details: pd.DataFrame = field(default_factory=pd.DataFrame)
    power_ci: Dict[float, Tuple[float, float]] = field(default_factory=dict)
    invalid_rate: Dict[float, float] = field(default_factory=dict)
    n_valid: Dict[float, int] = field(default_factory=dict)
    n_rejections: Dict[float, int] = field(default_factory=dict)
    fpr_ci: Tuple[float, float] = (np.nan, np.nan)
    mde_effect: Optional[float] = None
    target_power: float = 0.80
    design_fingerprint: Optional[str] = None
    window_spacing_days: int = 1
    selection_scope: str = "fixed_prespecified"
    selection_procedure_id: str = "fixed_prespecified"
    eligible_units: Tuple[str, ...] = ()
    selector_config_json: str = "{}"
    assignment_mechanism: str = "observational"
    requested_sims_per_point: Optional[int] = None
    min_valid_sims_per_point: Optional[int] = None
    max_invalid_rate: float = 0.05
    effect_model: str = "constant_multiplicative"
    artifact_schema_version: str = "1.0"
    authoritative: bool = False

    def __post_init__(self) -> None:
        self.effect_grid = [float(effect) for effect in self.effect_grid]
        self.n_sims_per_point = int(self.n_sims_per_point)
        if self.requested_sims_per_point is None:
            self.requested_sims_per_point = self.n_sims_per_point
        else:
            self.requested_sims_per_point = int(self.requested_sims_per_point)
        if self.min_valid_sims_per_point is None:
            self.min_valid_sims_per_point = min(
                20, int(self.requested_sims_per_point)
            )
        else:
            self.min_valid_sims_per_point = int(self.min_valid_sims_per_point)
        eligible = tuple(sorted(str(value).strip() for value in self.eligible_units))
        if any(not value for value in eligible) or len(set(eligible)) != len(eligible):
            raise ValueError("eligible_units contém nome vazio ou duplicado")
        self.eligible_units = eligible
        try:
            selector_config = json.loads(self.selector_config_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("selector_config_json deve ser JSON válido") from exc
        if not isinstance(selector_config, dict):
            raise ValueError("selector_config_json deve codificar um objeto JSON")
        self.selector_config_json = json.dumps(
            selector_config, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        try:
            if len(self.fpr_ci) != 2:
                raise ValueError
            if any(len(interval) != 2 for interval in self.power_ci.values()):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError("power_ci e fpr_ci devem conter exatamente dois limites") from exc
        if not (0.0 < self.target_power < 1.0):
            raise ValueError("target_power deve estar em (0, 1)")
        if self.n_sims_per_point < 1 or self.requested_sims_per_point < 1:
            raise ValueError("contagens de simulações devem ser >= 1")
        if not (
            1
            <= self.min_valid_sims_per_point
            <= self.requested_sims_per_point
        ):
            raise ValueError(
                "min_valid_sims_per_point deve estar entre 1 e "
                "requested_sims_per_point"
            )
        if not (0.0 <= self.max_invalid_rate < 1.0):
            raise ValueError("max_invalid_rate deve estar em [0, 1)")
        if self.effect_model != "constant_multiplicative":
            raise ValueError("effect_model suportado: 'constant_multiplicative'")

    @staticmethod
    def _json_number(value: Any) -> Optional[float]:
        if value is None:
            return None
        number = float(value)
        return number if np.isfinite(number) else None

    def approval_problems(self) -> List[str]:
        """Return simulation sufficiency/validity failures for approval use."""
        problems: List[str] = []
        if not self.authoritative:
            problems.append(
                "PowerResult exploratório: execução autoritativa exige DesignSpec"
            )
        requested = int(cast(int, self.requested_sims_per_point))
        minimum = int(cast(int, self.min_valid_sims_per_point))
        if self.n_sims_per_point != requested:
            problems.append(
                f"power executou {self.n_sims_per_point} de {requested} "
                "simulações solicitadas por ponto"
            )
        for effect in self.effect_grid:
            if effect not in self.n_valid:
                problems.append(f"power sem n_valid para efeito {effect:g}")
                continue
            if effect not in self.invalid_rate:
                problems.append(f"power sem invalid_rate para efeito {effect:g}")
                continue
            valid = int(self.n_valid[effect])
            if effect not in self.n_rejections:
                problems.append(f"power sem n_rejections para efeito {effect:g}")
                continue
            rejected = int(self.n_rejections[effect])
            invalid = float(self.invalid_rate[effect])
            if valid < 0 or valid > self.n_sims_per_point:
                problems.append(f"n_valid inválido para efeito {effect:g}: {valid}")
                continue
            if rejected < 0 or rejected > valid:
                problems.append(
                    f"n_rejections inválido para efeito {effect:g}: {rejected}"
                )
                continue
            expected_invalid = 1.0 - valid / self.n_sims_per_point
            if not np.isfinite(invalid) or not math.isclose(
                invalid, expected_invalid, rel_tol=1e-9, abs_tol=1e-9
            ):
                problems.append(
                    f"invalid_rate inconsistente para efeito {effect:g}"
                )
                continue
            if valid < minimum:
                problems.append(
                    f"somente {valid} simulações válidas para efeito {effect:g} "
                    f"(mínimo {minimum})"
                )
            if invalid > self.max_invalid_rate + 1e-12:
                problems.append(
                    f"invalid_rate={invalid:.1%} para efeito {effect:g} excede "
                    f"{self.max_invalid_rate:.1%}"
                )
            estimate = self.power.get(effect, np.nan)
            if valid >= minimum and (
                not np.isfinite(estimate) or not (0.0 <= estimate <= 1.0)
            ):
                problems.append(f"power inválido para efeito {effect:g}")
            expected_power = rejected / valid if valid else np.nan
            if valid and (
                not np.isfinite(estimate)
                or not math.isclose(
                    float(estimate), expected_power, rel_tol=1e-9, abs_tol=1e-12
                )
            ):
                problems.append(
                    f"power não corresponde a n_rejections/n_valid para efeito {effect:g}"
                )
            interval = self.power_ci.get(effect)
            if interval is None or len(interval) != 2:
                problems.append(f"power_ci ausente para efeito {effect:g}")
            else:
                lower, upper = float(interval[0]), float(interval[1])
                if (
                    not np.isfinite(lower)
                    or not np.isfinite(upper)
                    or not (0.0 <= lower <= upper <= 1.0)
                    or (
                        np.isfinite(estimate)
                        and not (lower - 1e-12 <= estimate <= upper + 1e-12)
                    )
                ):
                    problems.append(f"power_ci inválido para efeito {effect:g}")
                expected_interval = _wilson_interval(rejected, valid)
                if valid and not all(
                    math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)
                    for left, right in zip(
                        (lower, upper), expected_interval, strict=True
                    )
                ):
                    problems.append(
                        f"power_ci não corresponde ao intervalo Wilson para efeito {effect:g}"
                    )
        zero_effect = next(
            (effect for effect in self.effect_grid if np.isclose(effect, 0.0)),
            None,
        )
        if zero_effect is None:
            problems.append("PowerResult não contém ponto de efeito zero")
        else:
            zero_power = self.power.get(zero_effect, np.nan)
            if (
                not np.isfinite(self.fpr)
                or not np.isfinite(zero_power)
                or not math.isclose(self.fpr, zero_power, rel_tol=1e-9, abs_tol=1e-12)
            ):
                problems.append("fpr diverge do power no efeito zero")
            zero_interval = self.power_ci.get(zero_effect)
            if (
                zero_interval is None
                or len(zero_interval) != 2
                or any(not np.isfinite(value) for value in self.fpr_ci)
                or not all(
                    math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
                    for left, right in zip(self.fpr_ci, zero_interval, strict=True)
                )
            ):
                problems.append("fpr_ci diverge do power_ci no efeito zero")
        return problems

    @property
    def valid_for_approval(self) -> bool:
        return not self.approval_problems()

    @property
    def failure_reasons(self) -> Tuple[str, ...]:
        return tuple(self.approval_problems())

    @property
    def mde_at_target(self) -> Optional[float]:
        """MDE magnitude at ``target_power`` (legacy field name is ``mde_80``)."""
        return self.mde_80

    def integrity_payload(self) -> Dict[str, Any]:
        """Compact, stable, decision-relevant power artifact payload."""
        points = []
        for effect in self.effect_grid:
            interval = self.power_ci.get(effect, (np.nan, np.nan))
            points.append(
                {
                    "effect": effect,
                    "power": self._json_number(self.power.get(effect)),
                    "power_ci": [
                        self._json_number(interval[0]),
                        self._json_number(interval[1]),
                    ],
                    "invalid_rate": self._json_number(
                        self.invalid_rate.get(effect)
                    ),
                    "n_valid": (
                        int(self.n_valid[effect])
                        if effect in self.n_valid
                        else None
                    ),
                    "n_rejections": (
                        int(self.n_rejections[effect])
                        if effect in self.n_rejections
                        else None
                    ),
                }
            )
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "authoritative": self.authoritative,
            "design_fingerprint": self.design_fingerprint,
            "alpha": self.alpha,
            "target_power": self.target_power,
            "requested_sims_per_point": self.requested_sims_per_point,
            "actual_sims_per_point": self.n_sims_per_point,
            "min_valid_sims_per_point": self.min_valid_sims_per_point,
            "max_invalid_rate": self.max_invalid_rate,
            "effect_model": self.effect_model,
            "window_spacing_days": self.window_spacing_days,
            "selection_scope": self.selection_scope,
            "selection_procedure_id": self.selection_procedure_id,
            "eligible_units": list(self.eligible_units),
            "selector_config": json.loads(self.selector_config_json),
            "assignment_mechanism": self.assignment_mechanism,
            "fpr": self._json_number(self.fpr),
            "fpr_ci": [
                self._json_number(self.fpr_ci[0]),
                self._json_number(self.fpr_ci[1]),
            ],
            "mde_at_target": self._json_number(self.mde_80),
            "mde_effect": self._json_number(self.mde_effect),
            "points": points,
        }

    @property
    def computed_fingerprint(self) -> str:
        return fingerprint_payload(self.integrity_payload())

    def to_dict(self) -> Dict[str, Any]:
        payload = self.integrity_payload()
        payload["artifact_fingerprint"] = self.computed_fingerprint
        return payload

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PowerResult":
        raw_points = value.get("points", ())
        if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
            raise ValueError("PowerResult points deve ser uma sequência")
        effect_grid: List[float] = []
        power: Dict[float, float] = {}
        power_ci: Dict[float, Tuple[float, float]] = {}
        invalid_rate: Dict[float, float] = {}
        n_valid: Dict[float, int] = {}
        n_rejections: Dict[float, int] = {}
        for raw_point in raw_points:
            if not isinstance(raw_point, Mapping):
                raise ValueError("cada ponto de PowerResult deve ser um objeto")
            effect = float(raw_point["effect"])
            effect_grid.append(effect)
            point_power = raw_point.get("power")
            power[effect] = float(point_power) if point_power is not None else np.nan
            interval = raw_point.get("power_ci", (None, None))
            if (
                not isinstance(interval, Sequence)
                or isinstance(interval, (str, bytes))
                or len(interval) != 2
            ):
                raise ValueError("power_ci deve conter exatamente dois limites")
            power_ci[effect] = tuple(
                float(bound) if bound is not None else np.nan for bound in interval
            )  # type: ignore[assignment]
            point_invalid = raw_point.get("invalid_rate")
            if point_invalid is not None:
                invalid_rate[effect] = float(point_invalid)
            point_valid = raw_point.get("n_valid")
            if point_valid is not None:
                n_valid[effect] = int(point_valid)
            point_rejections = raw_point.get("n_rejections")
            if point_rejections is not None:
                n_rejections[effect] = int(point_rejections)
        fpr_value = value.get("fpr")
        fpr_ci_value = value.get("fpr_ci", (None, None))
        if (
            not isinstance(fpr_ci_value, Sequence)
            or isinstance(fpr_ci_value, (str, bytes))
            or len(fpr_ci_value) != 2
        ):
            raise ValueError("fpr_ci deve conter exatamente dois limites")
        mde_value = value.get("mde_at_target")
        mde_effect = value.get("mde_effect")
        result = cls(
            effect_grid=effect_grid,
            power=power,
            fpr=float(fpr_value) if fpr_value is not None else np.nan,
            mde_80=float(mde_value) if mde_value is not None else None,
            n_sims_per_point=int(value["actual_sims_per_point"]),
            alpha=float(value["alpha"]),
            power_ci=power_ci,
            invalid_rate=invalid_rate,
            n_valid=n_valid,
            n_rejections=n_rejections,
            fpr_ci=tuple(
                float(bound) if bound is not None else np.nan
                for bound in fpr_ci_value
            ),  # type: ignore[arg-type]
            mde_effect=float(mde_effect) if mde_effect is not None else None,
            target_power=float(value["target_power"]),
            design_fingerprint=(
                str(value["design_fingerprint"])
                if value.get("design_fingerprint") is not None
                else None
            ),
            window_spacing_days=int(value.get("window_spacing_days", 1)),
            selection_scope=str(value.get("selection_scope", "fixed_prespecified")),
            selection_procedure_id=str(
                value.get("selection_procedure_id", "fixed_prespecified")
            ),
            eligible_units=tuple(value.get("eligible_units", ())),
            selector_config_json=json.dumps(
                value.get("selector_config", {}), sort_keys=True,
                separators=(",", ":"), ensure_ascii=False, allow_nan=False,
            ),
            assignment_mechanism=str(
                value.get("assignment_mechanism", "observational")
            ),
            requested_sims_per_point=int(value["requested_sims_per_point"]),
            min_valid_sims_per_point=int(value["min_valid_sims_per_point"]),
            max_invalid_rate=float(value["max_invalid_rate"]),
            effect_model=str(value.get("effect_model", "constant_multiplicative")),
            artifact_schema_version=str(value.get("artifact_schema_version", "1.0")),
            authoritative=bool(value.get("authoritative", False)),
        )
        embedded = value.get("artifact_fingerprint")
        if embedded is not None and str(embedded) != result.computed_fingerprint:
            raise ValueError("artifact_fingerprint não corresponde ao PowerResult")
        return result

    @classmethod
    def from_json(cls, value: str) -> "PowerResult":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("PowerResult JSON deve codificar um objeto")
        return cls.from_dict(payload)

    def design_problems(self, design_spec: DesignSpec) -> List[str]:
        """Return mismatches between this artifact and its immutable design."""
        expected = {
            "design_fingerprint": design_spec.fingerprint,
            "alpha": design_spec.alpha,
            "target_power": design_spec.power_target,
            "requested_sims_per_point": design_spec.power_n_sims_per_point,
            "min_valid_sims_per_point": (
                design_spec.power_min_valid_sims_per_point
            ),
            "max_invalid_rate": design_spec.power_max_invalid_rate,
            "effect_model": design_spec.power_effect_model,
            "window_spacing_days": design_spec.window_spacing_days,
            "selection_procedure_id": design_spec.selection_procedure_id,
            "eligible_units": design_spec.eligible_units,
            "selector_config_json": design_spec.selector_config_json,
            "assignment_mechanism": design_spec.assignment_mechanism,
        }
        actual = {
            "design_fingerprint": self.design_fingerprint,
            "alpha": self.alpha,
            "target_power": self.target_power,
            "requested_sims_per_point": self.requested_sims_per_point,
            "min_valid_sims_per_point": self.min_valid_sims_per_point,
            "max_invalid_rate": self.max_invalid_rate,
            "effect_model": self.effect_model,
            "window_spacing_days": self.window_spacing_days,
            "selection_procedure_id": self.selection_procedure_id,
            "eligible_units": self.eligible_units,
            "selector_config_json": self.selector_config_json,
            "assignment_mechanism": self.assignment_mechanism,
        }
        problems = [
            f"PowerResult {name} diverge do DesignSpec"
            for name in expected
            if actual[name] != expected[name]
        ]
        if not self.authoritative:
            problems.append("PowerResult exploratório não pode vincular DesignSpec")
        if self.effect_grid != list(design_spec.effect_grid):
            problems.append("PowerResult effect_grid diverge do DesignSpec")
        expected_scope = (
            "fixed_prespecified"
            if design_spec.selection_procedure_id == "fixed_prespecified"
            else "replayed_selection"
        )
        if self.selection_scope != expected_scope:
            problems.append("PowerResult selection_scope diverge do DesignSpec")

        candidates = [
            effect for effect in design_spec.effect_grid
            if not np.isclose(effect, 0.0)
        ]
        if design_spec.decision_rule.direction == "positive":
            candidates = [effect for effect in candidates if effect > 0]
        elif design_spec.decision_rule.direction == "negative":
            candidates = [effect for effect in candidates if effect < 0]
        expected_effect = next(
            (
                float(effect)
                for effect in sorted(candidates, key=lambda value: (abs(value), value))
                if np.isfinite(self.power.get(float(effect), np.nan))
                and self.power[float(effect)] >= design_spec.power_target
            ),
            None,
        )
        if expected_effect is None:
            if self.mde_effect is not None or self.mde_at_target is not None:
                problems.append("PowerResult MDE não corresponde à curva de power")
        elif (
            self.mde_effect is None
            or not math.isclose(self.mde_effect, expected_effect)
            or self.mde_at_target is None
            or not math.isclose(self.mde_at_target, abs(expected_effect))
        ):
            problems.append("PowerResult MDE não corresponde à curva de power")
        return problems

    def matches_design(self, design_spec: DesignSpec) -> bool:
        return not self.design_problems(design_spec)


def _draw_windows(
    index: pd.DatetimeIndex,
    pre_days: int,
    post_days: int,
    n: int,
    rng: np.random.Generator,
    buffer_end_days: int = 0,
    min_start_spacing_days: int = 1,
    anticipation_days: int = 0,
) -> List[int]:
    """Draw placebo starts under the requested historical-window design.

    ``min_start_spacing_days=1`` draws exactly ``n`` iid starts with replacement,
    which is the schema-2 power design required by binomial uncertainty. Larger
    spacing values retain the legacy without-replacement heuristic.
    """
    if pre_days <= 0 or post_days <= 0 or n <= 0:
        raise ValueError("pre_days, post_days e n devem ser > 0")
    if buffer_end_days < 0 or min_start_spacing_days < 1 or anticipation_days < 0:
        raise ValueError("buffer/anticipation devem ser >= 0 e spacing >= 1")
    lo = pre_days + anticipation_days
    hi_exclusive = len(index) - post_days - buffer_end_days + 1
    if hi_exclusive <= lo:
        raise ValueError(
            f"Histórico insuficiente: precisa de > "
            f"{pre_days + anticipation_days + post_days + buffer_end_days - 1} dias"
        )

    if min_start_spacing_days == 1:
        return [int(value) for value in rng.integers(lo, hi_exclusive, size=n)]

    phases = [
        list(range(lo + offset, hi_exclusive, min_start_spacing_days))
        for offset in range(min_start_spacing_days)
        if lo + offset < hi_exclusive
    ]
    max_count = max(len(starts) for starts in phases)
    fullest = [starts for starts in phases if len(starts) == max_count]
    starts = fullest[int(rng.integers(0, len(fullest)))]
    if len(starts) > n:
        starts = sorted(rng.choice(starts, size=n, replace=False).tolist())
    return starts


def _callable_name(fit_fn: Callable) -> str:
    name = getattr(fit_fn, "__name__", fit_fn.__class__.__name__)
    return str(name).removeprefix("fit_")


def _require_builtin_estimators(fit_fns: Mapping[str, Callable]) -> None:
    """Require the exact callables used by final authoritative reporting."""
    from supply_experiments.estimators import ESTIMATORS

    unknown = sorted(set(fit_fns) - set(ESTIMATORS))
    replaced = sorted(
        name for name, fit_fn in fit_fns.items()
        if name in ESTIMATORS and fit_fn is not ESTIMATORS[name]
    )
    if unknown or replaced:
        raise ValueError(
            "DesignSpec autoritativo exige callables exatos de ESTIMATORS; "
            f"desconhecidos={unknown}, substituídos={replaced}"
        )


def _wilson_interval(
    successes: int,
    n: int,
    z: float = 1.959963984540054,
) -> Tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    rate = successes / n
    denom = 1.0 + z * z / n
    center = (rate + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(
        rate * (1.0 - rate) / n + z * z / (4.0 * n * n)
    ) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def simulate_once(
    panel: CityPanel,
    treated: Sequence[str],
    donors: Sequence[str],
    fit_fn: Callable,
    start_idx: int,
    pre_days: int,
    post_days: int,
    delta: float,
    alpha: float,
    fit_kwargs: Optional[dict] = None,
    seasonal_effect: bool = False,
    rng: Optional[np.random.Generator] = None,
    max_group_placebos: Optional[int] = 30,
    permutation_seed: Optional[int] = 123,
    assignment_mechanism: str = "observational",
    anticipation_days: int = 0,
    max_pre_rmspe: Optional[float] = None,
    effect_multiplier: Optional[Sequence[float]] = None,
) -> Dict:
    """Run one injected-effect historical simulation for one estimator."""
    treated = [str(city) for city in treated]
    donors = [str(city) for city in donors]
    if len(treated) != len(set(treated)) or len(donors) != len(set(donors)):
        raise ValueError("treated/donors contêm cidades duplicadas")
    overlap = sorted(set(treated) & set(donors))
    if overlap:
        raise ValueError(f"treated e donors se sobrepõem: {overlap}")
    missing = sorted((set(treated) | set(donors)) - set(panel.cities))
    if missing:
        raise ValueError(f"cidades fora do painel: {missing}")
    if not treated or len(donors) < 2:
        raise ValueError("use ao menos uma tratada e duas doadoras")
    y = panel.aggregate(treated).to_numpy(float)
    Yt = panel.outcome[treated].to_numpy(float)
    Yd = panel.outcome[list(donors)].to_numpy(float)
    if anticipation_days < 0:
        raise ValueError("anticipation_days deve ser >= 0")
    pre_sl = slice(start_idx - anticipation_days - pre_days,
                   start_idx - anticipation_days)
    post_sl = slice(start_idx, start_idx + post_days)

    y_pre = y[pre_sl].copy()
    y_post = y[post_sl].copy()
    Yt_pre = Yt[pre_sl].copy()
    Yt_post = Yt[post_sl].copy()
    mult: "float | np.ndarray" = 1.0
    if effect_multiplier is not None:
        mult = np.asarray(effect_multiplier, dtype=float)
        if mult.shape != (len(y_post),):
            raise ValueError("effect_multiplier deve ter um valor por dia pós")
        if not np.all(np.isfinite(mult)) or np.any(mult <= 0):
            raise ValueError("effect_multiplier deve ser finito e estritamente positivo")
    elif delta != 0.0:
        mult = 1.0 + delta
        if seasonal_effect and rng is not None:
            mult = 1.0 + delta * (1.0 + 0.3 * rng.standard_normal(len(y_post)))
    if effect_multiplier is not None or delta != 0.0:
        y_post = y_post * mult
        Yt_post = Yt_post * np.asarray(mult)[:, None] if np.ndim(mult) else Yt_post * mult

    required_placebos = math.ceil((1.0 / alpha) - 1.0 - 1e-12)
    inf = placebo_inference(
        fit_fn,
        y_pre,
        Yd[pre_sl],
        y_post,
        Yd[post_sl],
        list(donors),
        fit_kwargs=fit_kwargs or {},
        min_placebos=required_placebos,
        n_treated_units=len(treated),
        max_group_placebos=max_group_placebos,
        seed=permutation_seed,
        assignment_mechanism=assignment_mechanism,
        Y_treated_pre=Yt_pre,
        Y_treated_post=Yt_post,
        treated_names=treated,
    )
    real_fit = getattr(inf, "real_fit", None)
    real_att = getattr(real_fit, "att_pct", np.nan)
    att_pct = float(real_att) if np.isfinite(real_att) else np.nan
    pre_rmspe = float(getattr(real_fit, "pre_rmspe", np.nan))
    prefit_valid = bool(
        max_pre_rmspe is None
        or (np.isfinite(pre_rmspe) and pre_rmspe <= max_pre_rmspe)
    )
    # A design simulation evaluates the candidate rejection algorithm. Its
    # empirical FPR/power is what later establishes calibration; requiring the
    # final-analysis validity label here would make observational calibration
    # impossible by construction.
    decision_valid = bool(
        np.isfinite(inf.p_value)
        and np.isfinite(att_pct)
        and int(inf.n_placebos) >= required_placebos
        and prefit_valid
    )
    return {
        "delta": float(delta),
        "start_idx": int(start_idx),
        "p_value": float(inf.p_value),
        "att_pct": att_pct,
        # Historical design simulation explicitly evaluates this decision path.
        "valid_for_decision": decision_valid,
        "prefit_valid": prefit_valid,
        "pre_rmspe": pre_rmspe,
        "assignment_mechanism": assignment_mechanism,
        "reject": bool(decision_valid and inf.p_value <= alpha),
        "att_pct_placebo_med": (
            float(np.nanmedian(inf.placebo_atts)) if inf.placebo_atts else np.nan
        ),
        "n_placebos": int(inf.n_placebos),
    }


def _decision_simulation(
    panel: CityPanel,
    treated: Sequence[str],
    donors: Sequence[str],
    fit_fns: Mapping[str, Callable],
    start_idx: int,
    pre_days: int,
    post_days: int,
    delta: float,
    alpha: float,
    decision_rule: DecisionRule,
    fit_kwargs: Optional[dict],
    fit_kwargs_by_estimator: Mapping[str, dict],
    rng: np.random.Generator,
    max_group_placebos: Optional[int],
    permutation_seed: Optional[int],
    anticipation_days: int,
    assignment_mechanism: str,
    max_pre_rmspe: Optional[float],
    effect_model: str = "constant_multiplicative",
) -> Dict:
    method_rows: Dict[str, Dict] = {}
    shared_multiplier: Optional[np.ndarray] = None
    if delta != 0.0:
        if effect_model != "constant_multiplicative":
            raise ValueError("effect_model suportado: 'constant_multiplicative'")
        shared_multiplier = np.full(post_days, 1.0 + delta, dtype=float)
    for method, method_fit in fit_fns.items():
        kwargs = fit_kwargs_by_estimator.get(method, fit_kwargs or {})
        method_rows[method] = simulate_once(
            panel,
            treated,
            donors,
            method_fit,
            start_idx,
            pre_days,
            post_days,
            delta,
            alpha,
            kwargs,
            seasonal_effect=False,
            rng=rng,
            max_group_placebos=max_group_placebos,
            permutation_seed=permutation_seed,
            assignment_mechanism=assignment_mechanism,
            anticipation_days=anticipation_days,
            max_pre_rmspe=max_pre_rmspe,
            effect_multiplier=shared_multiplier,
        )

    scoped = decision_rule.methods or tuple(sorted(method_rows))
    method_valid = {
        method: bool(method_rows[method]["valid_for_decision"])
        for method in scoped
        if method in method_rows
    }
    if decision_rule.require_all_valid:
        valid = len(method_valid) == len(scoped) and all(method_valid.values())
    else:
        valid = sum(method_valid.values()) >= decision_rule.min_rejections
    rejected = decision_rule.evaluate(method_rows, alpha) if valid else False
    finite_p = [
        row["p_value"] for row in method_rows.values() if np.isfinite(row["p_value"])
    ]
    return {
        "delta": float(delta),
        "start_idx": int(start_idx),
        "p_value": float(min(finite_p)) if valid and finite_p else np.nan,
        "reject": bool(rejected),
        "valid": bool(valid),
        "method_results": method_rows,
        "n_placebos": min(
            (row["n_placebos"] for row in method_rows.values()), default=0
        ),
    }


def power_analysis(
    panel: CityPanel,
    treated: Sequence[str],
    donors: Sequence[str],
    fit_fn: Callable,
    pre_days: int,
    post_days: int,
    effect_grid: Sequence[float] = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10),
    n_sims_per_point: Optional[int] = None,
    alpha: float = 0.10,
    seed: int = 42,
    fit_kwargs: Optional[dict] = None,
    max_group_placebos: Optional[int] = 30,
    permutation_seed: Optional[int] = 123,
    *,
    fit_fns: Optional[Mapping[str, Callable]] = None,
    fit_kwargs_by_estimator: Optional[Mapping[str, dict]] = None,
    decision_rule: Optional[DecisionRule] = None,
    design_spec: Optional[DesignSpec] = None,
    target_power: Optional[float] = None,
    min_valid_sims_per_point: Optional[int] = None,
    max_invalid_rate: Optional[float] = None,
    window_spacing_days: Optional[int] = None,
    selection_fn: Optional[
        Callable[[CityPanel, Sequence[str], np.random.Generator, int, int], tuple]
    ] = None,
    eligible: Optional[Sequence[str]] = None,
    selection_procedure_id: str = "",
    selector_config_json: Optional[str] = None,
) -> PowerResult:
    """Estimate power and MDE for the configured executable decision.

    Rates exclude invalid simulations and report their fraction. The MDE is a
    non-negative magnitude; ``mde_effect`` records the signed grid scenario.
    """
    requested_sims = int(
        design_spec.power_n_sims_per_point
        if n_sims_per_point is None and design_spec is not None
        else (30 if n_sims_per_point is None else n_sims_per_point)
    )
    resolved_target = float(
        design_spec.power_target
        if target_power is None and design_spec is not None
        else (0.80 if target_power is None else target_power)
    )
    resolved_min_valid = int(
        design_spec.power_min_valid_sims_per_point
        if min_valid_sims_per_point is None and design_spec is not None
        else (
            min(20, requested_sims)
            if min_valid_sims_per_point is None
            else min_valid_sims_per_point
        )
    )
    resolved_max_invalid = float(
        design_spec.power_max_invalid_rate
        if max_invalid_rate is None and design_spec is not None
        else (0.05 if max_invalid_rate is None else max_invalid_rate)
    )
    if not (0.0 < resolved_target < 1.0):
        raise ValueError("target_power deve estar em (0, 1)")
    if requested_sims < 1:
        raise ValueError("n_sims_per_point deve ser >= 1")
    if not (1 <= resolved_min_valid <= requested_sims):
        raise ValueError(
            "min_valid_sims_per_point deve estar entre 1 e n_sims_per_point"
        )
    if not (0.0 <= resolved_max_invalid < 1.0):
        raise ValueError("max_invalid_rate deve estar em [0, 1)")

    estimators = dict(fit_fns or {_callable_name(fit_fn): fit_fn})
    if not estimators or any(not callable(fn) for fn in estimators.values()):
        raise ValueError("fit_fns deve mapear nomes para callables")
    rule = decision_rule or (
        design_spec.decision_rule if design_spec else DecisionRule()
    )
    unknown_kwargs = set(fit_kwargs_by_estimator or {}) - set(estimators)
    if unknown_kwargs:
        raise ValueError(
            f"fit_kwargs_by_estimator referencia estimadores ausentes: {sorted(unknown_kwargs)}"
        )
    if fit_kwargs_by_estimator is None:
        supplied_estimator_kwargs = {
            method: dict(fit_kwargs or {}) for method in estimators
        }
    else:
        supplied_estimator_kwargs = {
            method: dict((fit_kwargs_by_estimator or {}).get(method, fit_kwargs or {}))
            for method in estimators
        }
    spacing = (
        design_spec.window_spacing_days
        if design_spec is not None and window_spacing_days is None
        else (1 if window_spacing_days is None else window_spacing_days)
    )
    if selection_fn is not None and eligible is None:
        raise ValueError("eligible é obrigatório com selection_fn")
    if selection_fn is not None and not selection_procedure_id.strip():
        raise ValueError("selection_procedure_id é obrigatório com selection_fn")
    eligible_values = (
        sorted(str(city).strip() for city in eligible)
        if eligible is not None
        else []
    )
    if any(not city for city in eligible_values):
        raise ValueError("eligible contém cidade vazia")
    if len(eligible_values) != len(set(eligible_values)):
        raise ValueError("eligible contém cidades duplicadas")
    missing_eligible = sorted(set(eligible_values) - set(panel.cities))
    if missing_eligible:
        raise ValueError(f"eligible contém cidades fora do painel: {missing_eligible}")
    bound_eligible = tuple(
        eligible_values
        if eligible is not None
        else (design_spec.eligible_units if design_spec is not None else ())
    )
    resolved_selector_config = (
        design_spec.selector_config_json
        if selector_config_json is None and design_spec is not None
        else ("{}" if selector_config_json is None else selector_config_json)
    )
    try:
        selector_config_payload = json.loads(resolved_selector_config)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("selector_config_json deve ser JSON válido") from exc
    if not isinstance(selector_config_payload, dict):
        raise ValueError("selector_config_json deve codificar um objeto JSON")
    canonical_selector_config = json.dumps(
        selector_config_payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )

    if design_spec is None:
        spec = DesignSpec(
            schema_version="1.0",
            treated_units=tuple(treated),
            donor_units=tuple(donors),
            estimator_names=tuple(estimators),
            pre_days=pre_days,
            post_days=post_days,
            alpha=alpha,
            max_group_placebos=max_group_placebos,
            effect_grid=tuple(effect_grid),
            decision_rule=rule,
            selection_procedure_id=(
                selection_procedure_id.strip()
                if selection_fn is not None
                else "fixed_prespecified"
            ),
            eligible_units=bound_eligible,
            selector_config_json=canonical_selector_config,
            power_target=resolved_target,
            power_n_sims_per_point=requested_sims,
            power_min_valid_sims_per_point=resolved_min_valid,
            power_max_invalid_rate=resolved_max_invalid,
            seed=seed,
            permutation_seed=permutation_seed,
            window_spacing_days=spacing,
            estimator_config_json=json.dumps(supplied_estimator_kwargs),
        )
    else:
        spec = design_spec.validate()
        require_runtime_version(spec.implementation_version)
        _require_builtin_estimators(estimators)
        expected = {
            "treated_units": tuple(sorted(treated)),
            "donor_units": tuple(sorted(donors)),
            "estimator_names": tuple(sorted(estimators)),
            "pre_days": pre_days,
            "post_days": post_days,
            "alpha": alpha,
            "max_group_placebos": max_group_placebos,
            "effect_grid": tuple(sorted({float(effect) for effect in effect_grid})),
            "decision_rule": rule,
            "eligible_units": bound_eligible,
            "selector_config_json": canonical_selector_config,
            "power_target": resolved_target,
            "power_n_sims_per_point": requested_sims,
            "power_min_valid_sims_per_point": resolved_min_valid,
            "power_max_invalid_rate": resolved_max_invalid,
            "seed": seed,
            "permutation_seed": permutation_seed,
            "window_spacing_days": int(cast(int, spacing)),
            "estimator_config": supplied_estimator_kwargs,
        }
        actual = {
            "treated_units": spec.treated_units,
            "donor_units": spec.donor_units,
            "estimator_names": spec.estimator_names,
            "pre_days": spec.pre_days,
            "post_days": spec.post_days,
            "alpha": spec.alpha,
            "max_group_placebos": spec.max_group_placebos,
            "effect_grid": spec.effect_grid,
            "decision_rule": spec.decision_rule,
            "eligible_units": spec.eligible_units,
            "selector_config_json": spec.selector_config_json,
            "power_target": spec.power_target,
            "power_n_sims_per_point": spec.power_n_sims_per_point,
            "power_min_valid_sims_per_point": (
                spec.power_min_valid_sims_per_point
            ),
            "power_max_invalid_rate": spec.power_max_invalid_rate,
            "seed": spec.seed,
            "permutation_seed": spec.permutation_seed,
            "window_spacing_days": spec.window_spacing_days,
            "estimator_config": json.loads(spec.estimator_config_json),
        }
        mismatched = [key for key in expected if expected[key] != actual[key]]
        if mismatched:
            raise ValueError(f"argumentos divergem do DesignSpec: {mismatched}")

    if selection_fn is None:
        if spec.selection_procedure_id != "fixed_prespecified":
            raise ValueError(
                "DesignSpec declara seleção não fixa; power_analysis exige selection_fn "
                "para replay do procedimento"
            )
        selection_scope = "fixed_prespecified"
    else:
        if not selection_procedure_id.strip():
            raise ValueError("selection_procedure_id é obrigatório com selection_fn")
        if selection_procedure_id.strip() != spec.selection_procedure_id:
            raise ValueError("selection_procedure_id diverge do DesignSpec")
        if tuple(eligible_values) != spec.eligible_units:
            raise ValueError("eligible diverge de eligible_units do DesignSpec")
        selection_scope = "replayed_selection"

    rng = np.random.default_rng(spec.seed)
    starts = _draw_windows(
        panel.index,
        pre_days,
        post_days,
        requested_sims,
        rng,
        min_start_spacing_days=int(cast(int, spacing)),
        anticipation_days=spec.anticipation_days,
    )
    selected_draws = []
    if selection_fn is not None:
        for simulation, start in enumerate(starts):
            selected = selection_fn(panel, eligible_values, rng, simulation, start)
            if not isinstance(selected, tuple) or len(selected) != 2:
                raise ValueError("selection_fn deve retornar (treated, donors)")
            selected_treated = [str(city) for city in selected[0]]
            selected_donors = [str(city) for city in selected[1]]
            if len(selected_treated) != len(spec.treated_units):
                raise ValueError("selection_fn retornou número incorreto de tratadas")
            if len(selected_donors) != len(spec.donor_units):
                raise ValueError("selection_fn retornou número incorreto de doadoras")
            outside_eligible = sorted(
                (set(selected_treated) | set(selected_donors))
                - set(eligible_values)
            )
            if outside_eligible:
                raise ValueError(
                    "selection_fn retornou cidades fora de eligible: "
                    f"{outside_eligible}"
                )
            # ``simulate_once`` performs the full duplicate/overlap/panel check.
            selected_draws.append((selected_treated, selected_donors))
    else:
        selected_draws = [
            (list(spec.treated_units), list(spec.donor_units)) for _ in starts
        ]
    rows = []
    estimator_kwargs = json.loads(spec.estimator_config_json)
    for delta in spec.effect_grid:
        for draw_index, start in enumerate(starts):
            simulated_treated, simulated_donors = selected_draws[draw_index]
            rows.append(
                _decision_simulation(
                    panel,
                    simulated_treated,
                    simulated_donors,
                    estimators,
                    start,
                    pre_days,
                    post_days,
                    float(delta),
                    spec.alpha,
                    rule,
                    fit_kwargs,
                    estimator_kwargs,
                    rng,
                    spec.max_group_placebos,
                    spec.permutation_seed,
                    spec.anticipation_days,
                    spec.assignment_mechanism,
                    spec.max_pre_rmspe,
                    spec.power_effect_model,
                )
            )
    details = pd.DataFrame(rows)

    power: Dict[float, float] = {}
    intervals: Dict[float, Tuple[float, float]] = {}
    invalid_rate: Dict[float, float] = {}
    n_valid: Dict[float, int] = {}
    n_rejections: Dict[float, int] = {}
    for delta, group in details.groupby("delta", sort=True):
        valid = group[group["valid"]]
        n_valid[float(delta)] = len(valid)
        invalid_rate[float(delta)] = 1.0 - len(valid) / len(group)
        successes = int(valid["reject"].sum())
        n_rejections[float(delta)] = successes
        power[float(delta)] = (
            float(successes / len(valid)) if len(valid) else np.nan
        )
        intervals[float(delta)] = _wilson_interval(successes, len(valid))

    fpr = next(
        (value for effect, value in power.items() if np.isclose(effect, 0.0)),
        np.nan,
    )
    fpr_ci = next(
        (value for effect, value in intervals.items() if np.isclose(effect, 0.0)),
        (np.nan, np.nan),
    )
    candidates = [effect for effect in power if not np.isclose(effect, 0.0)]
    if rule.direction == "positive":
        candidates = [effect for effect in candidates if effect > 0]
    elif rule.direction == "negative":
        candidates = [effect for effect in candidates if effect < 0]
    mde_effect = next(
        (
            float(effect)
            for effect in sorted(candidates, key=lambda value: (abs(value), value))
            if np.isfinite(power[effect]) and power[effect] >= resolved_target
        ),
        None,
    )
    mde = abs(mde_effect) if mde_effect is not None else None

    return PowerResult(
        effect_grid=list(spec.effect_grid),
        power=power,
        fpr=float(fpr),
        mde_80=mde,
        n_sims_per_point=len(starts),
        alpha=spec.alpha,
        details=details,
        power_ci=intervals,
        invalid_rate=invalid_rate,
        n_valid=n_valid,
        n_rejections=n_rejections,
        fpr_ci=fpr_ci,
        mde_effect=mde_effect,
        target_power=resolved_target,
        design_fingerprint=spec.fingerprint,
        window_spacing_days=int(cast(int, spacing)),
        selection_scope=selection_scope,
        selection_procedure_id=spec.selection_procedure_id,
        eligible_units=spec.eligible_units,
        selector_config_json=spec.selector_config_json,
        assignment_mechanism=spec.assignment_mechanism,
        requested_sims_per_point=requested_sims,
        min_valid_sims_per_point=resolved_min_valid,
        max_invalid_rate=resolved_max_invalid,
        effect_model=spec.power_effect_model,
        authoritative=design_spec is not None,
    )


def minimum_detectable_effect(power_result: PowerResult) -> Optional[float]:
    return power_result.mde_80
