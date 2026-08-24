"""Registry v2 — pré-registro como contrato.

`ExperimentRecord.validate_for_approval` é o gate que impede experimentos sem
hipótese, sem regra de decisão, ou sem poder estatístico suficiente de virar
"approved"/"running" via `save_experiment`.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

from supply_experiments._version import require_runtime_version
from supply_experiments.design.power import PowerResult
from supply_experiments.design.spec import DesignSpec

_VALID_STATUSES = {"draft", "approved", "running", "completed", "analyzed", "cancelled"}
_ALLOWED_TRANSITIONS = {
    "draft": {"draft", "approved", "cancelled"},
    "approved": {"approved", "running", "cancelled"},
    "running": {"running", "completed", "cancelled"},
    "completed": {"completed", "analyzed", "cancelled"},
    "analyzed": {"analyzed"},
    "cancelled": {"cancelled"},
}
_IMMUTABLE_AFTER_APPROVAL: Tuple[str, ...] = (
    "experiment_id", "name", "hypothesis", "design_method", "primary_kpi",
    "guardrail_kpis", "start_date", "end_date", "pre_window_days",
    "treated_units", "control_units", "mde_estimated", "expected_effect",
    "decision_rule", "merchant_type", "config_json", "design_spec_json",
    "design_fingerprint", "calibration_fingerprint", "power_result_json",
    "power_fingerprint", "contract_version",
)


def _is_sha256(value: str) -> bool:
    digest = str(value or "").lower()
    return len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest)


def _canonical_record_value(value):
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def validate_experiment_transition(
    previous: "ExperimentRecord",
    current: "ExperimentRecord",
    allow_legacy_unbound: bool = False,
) -> List[str]:
    """Validate append-only status movement and post-approval immutability."""
    problems = []
    allowed = _ALLOWED_TRANSITIONS.get(previous.status, set())
    if current.status not in allowed:
        problems.append(f"transição inválida: {previous.status} -> {current.status}")
    if previous.status != "draft":
        immutable_fields = _IMMUTABLE_AFTER_APPROVAL
        if allow_legacy_unbound:
            # A legacy row is normalized to ``legacy-unbound`` at write time;
            # callers should not need to manually mirror this storage label.
            immutable_fields = tuple(
                field_name for field_name in immutable_fields
                if field_name != "contract_version"
            )
        changed = [
            field_name for field_name in immutable_fields
            if _canonical_record_value(getattr(previous, field_name))
            != _canonical_record_value(getattr(current, field_name))
        ]
        if changed:
            problems.append(
                "campos de design imutáveis após approval foram alterados: "
                + ", ".join(changed)
            )
    if (
        not allow_legacy_unbound
        and previous.contract_version == "legacy-unbound"
        and current.status not in {"cancelled"}
    ):
        problems.append("registro legacy-unbound só pode permanecer legado ou ser cancelado")
    return problems


def minimum_donors_for_alpha(alpha: float) -> int:
    """Número de doadoras necessário para o p-value de permutação alcançar α."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha deve estar em (0, 1): {alpha}")
    # 1 / (J + 1) <= alpha. A tolerância evita ceil(9.000000000000002).
    return max(2, math.ceil((1.0 / alpha) - 1.0 - 1e-12))


@dataclass
class ExperimentRecord:
    experiment_id: str
    name: str
    hypothesis: str                     # obrigatório: o que se espera e por quê
    status: str                         # draft | approved | running | completed | analyzed
    design_method: str                  # scm | ascm | sdid | did
    primary_kpi: str
    guardrail_kpis: List[str]
    start_date: date
    end_date: date
    pre_window_days: int
    treated_units: List[Dict]           # {city_norm, city_address, state_address, weight}
    control_units: List[Dict]
    mde_estimated: Optional[float]      # do power_analysis — obrigatório p/ approve
    expected_effect: Optional[float]
    decision_rule: str                  # ex.: "ship se ATT>0 e p<=0.10 em >=2 estimadores"
    merchant_type: str = "Groceries"
    created_by: str = ""
    config_json: str = "{}"
    design_spec_json: str = ""
    design_fingerprint: str = ""
    calibration_fingerprint: str = ""
    power_result_json: str = ""
    power_fingerprint: str = ""
    contract_version: str = "design-v2"

    def with_design_contract(
        self,
        design_spec: DesignSpec,
        calibration_fingerprint: str = "",
        power_result: Optional[PowerResult] = None,
    ) -> "ExperimentRecord":
        """Bind immutable design, calibration, and optional power artifacts."""
        if design_spec.schema_version != "2.0":
            raise ValueError("with_design_contract exige DesignSpec schema_version='2.0'")
        require_runtime_version(design_spec.implementation_version)
        digest = (calibration_fingerprint or design_spec.calibration_fingerprint or "").lower()
        if digest and not _is_sha256(digest):
            raise ValueError("calibration_fingerprint deve ser SHA-256 hexadecimal")
        if (
            design_spec.calibration_fingerprint is not None
            and digest != design_spec.calibration_fingerprint
        ):
            raise ValueError("calibration_fingerprint diverge do DesignSpec")
        bound_spec = replace(
            design_spec,
            calibration_fingerprint=digest or None,
        )
        power_result_json = ""
        power_fingerprint = ""
        if power_result is not None:
            mismatches = power_result.design_problems(bound_spec)
            if mismatches:
                raise ValueError("; ".join(mismatches))
            power_result_json = power_result.to_json()
            power_fingerprint = power_result.computed_fingerprint
        return replace(
            self,
            design_spec_json=bound_spec.to_json(),
            design_fingerprint=bound_spec.fingerprint,
            calibration_fingerprint=digest,
            power_result_json=power_result_json,
            power_fingerprint=power_fingerprint,
            contract_version="design-v2",
        )

    def validate_for_approval(
        self,
        alpha: float = 0.10,
        allow_legacy_unbound: bool = False,
    ) -> List[str]:
        problems = []
        if self.status not in _VALID_STATUSES:
            problems.append(f"status inválido: {self.status!r}")
        if self.start_date > self.end_date:
            problems.append("start_date é posterior a end_date")
        if self.pre_window_days <= 0:
            problems.append("pre_window_days deve ser > 0 para um experimento")
        if not self.hypothesis.strip():
            problems.append("hypothesis vazia — pré-registro exige hipótese explícita")
        if not self.decision_rule.strip():
            problems.append("decision_rule vazia — defina a regra antes de ver dados")
        if self.mde_estimated is None:
            problems.append("mde_estimated ausente — rode power_analysis antes de aprovar")
        elif not math.isfinite(self.mde_estimated) or self.mde_estimated < 0:
            problems.append("mde_estimated deve ser finito e não negativo")
        elif self.expected_effect is None:
            problems.append("expected_effect ausente — defina o menor efeito relevante")
        elif not math.isfinite(self.expected_effect):
            problems.append("expected_effect deve ser finito")
        elif self.mde_estimated > abs(self.expected_effect):
            problems.append(
                f"MDE ({self.mde_estimated:.1%}) > efeito esperado "
                f"({self.expected_effect:.1%}): experimento sem poder. Aumente cidades/duração.")
        if not self.treated_units:
            problems.append("sem treated_units")
        treated_cities = {str(u.get("city_norm", "")).strip().upper()
                          for u in self.treated_units if u.get("city_norm")}
        control_cities = {str(u.get("city_norm", "")).strip().upper()
                          for u in self.control_units if u.get("city_norm")}
        if len(treated_cities) != len(self.treated_units):
            problems.append("treated_units contém city_norm ausente ou duplicado")
        if len(control_cities) != len(self.control_units):
            problems.append("control_units contém city_norm ausente ou duplicado")
        overlap = treated_cities & control_cities
        if overlap:
            problems.append(f"cidades tratadas também aparecem como doadoras: {sorted(overlap)}")
        has_bound_v2_design = bool(
            self.contract_version == "design-v2"
            and self.design_spec_json
            and self.design_fingerprint
        )
        if has_bound_v2_design:
            if len(control_cities) < 2:
                problems.append("design-v2 exige ao menos duas controles/doadoras")
        else:
            min_donors = minimum_donors_for_alpha(alpha)
            if len(control_cities) < min_donors:
                problems.append(f"só {len(control_cities)} controles/doadoras "
                                f"(mínimo {min_donors} para α={alpha:.3g})")
        problems.extend(
            self._design_binding_problems(
                treated_cities, control_cities, alpha, allow_legacy_unbound
            )
        )
        return problems

    def _design_binding_problems(
        self,
        treated_cities: set,
        control_cities: set,
        alpha: float,
        allow_legacy_unbound: bool,
    ) -> List[str]:
        if not self.design_spec_json or not self.design_fingerprint:
            if allow_legacy_unbound:
                return []
            return [
                "design contract ausente — use with_design_contract() ou "
                "allow_legacy_unbound=True explicitamente"
            ]
        try:
            spec = DesignSpec.from_json(self.design_spec_json)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return [f"design_spec_json inválido: {exc}"]
        problems = []
        if self.contract_version == "design-v2" and spec.schema_version != "2.0":
            problems.append("design-v2 exige DesignSpec schema_version='2.0'")
        try:
            require_runtime_version(spec.implementation_version)
        except ValueError as exc:
            problems.append(str(exc))
        if spec.fingerprint != self.design_fingerprint:
            problems.append("design_fingerprint não corresponde ao DesignSpec")
        if not spec.experiment_id or spec.experiment_id != self.experiment_id:
            problems.append("DesignSpec deve vincular o mesmo experiment_id")
        if spec.start_date != self.start_date or spec.end_date != self.end_date:
            problems.append("datas do DesignSpec divergem do registro")
        if spec.pre_days != self.pre_window_days:
            problems.append("pre_days do DesignSpec diverge do registro")
        if set(city.upper() for city in spec.treated_units) != treated_cities:
            problems.append("treated_units do DesignSpec divergem do registro")
        if set(city.upper() for city in spec.donor_units) != control_cities:
            problems.append("donor_units do DesignSpec divergem do registro")
        if self.design_method not in spec.estimator_names:
            problems.append("design_method não está entre os estimadores do DesignSpec")
        if spec.primary_kpi != self.primary_kpi:
            problems.append("primary_kpi do DesignSpec diverge do registro")
        if not math.isclose(spec.alpha, alpha):
            problems.append("alpha do DesignSpec diverge do gate do registro")
        if spec.require_calibration:
            if not _is_sha256(self.calibration_fingerprint):
                problems.append("calibration_fingerprint SHA-256 é obrigatório")
            if not _is_sha256(spec.calibration_fingerprint or ""):
                problems.append("DesignSpec não incorpora o calibration_fingerprint")
            elif spec.calibration_fingerprint != self.calibration_fingerprint.lower():
                problems.append("calibration_fingerprint diverge do DesignSpec")
        if not self.power_result_json or not self.power_fingerprint:
            if not allow_legacy_unbound:
                problems.append(
                    "power contract ausente — forneça power_result em "
                    "with_design_contract()"
                )
            return problems
        try:
            power_result = PowerResult.from_json(self.power_result_json)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            problems.append(f"power_result_json inválido: {exc}")
            return problems
        if not _is_sha256(self.power_fingerprint):
            problems.append("power_fingerprint deve ser SHA-256 hexadecimal")
        elif power_result.computed_fingerprint != self.power_fingerprint.lower():
            problems.append("power_fingerprint não corresponde ao PowerResult")
        problems.extend(power_result.design_problems(spec))
        problems.extend(power_result.approval_problems())
        if power_result.mde_at_target is None:
            problems.append("PowerResult não encontrou MDE no grid pré-registrado")
        elif self.mde_estimated is None or not math.isclose(
            power_result.mde_at_target,
            self.mde_estimated,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            problems.append("mde_estimated diverge do PowerResult")
        return problems


REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
  experiment_id STRING NOT NULL,
  name STRING, hypothesis STRING, status STRING,
  design_method STRING, primary_kpi STRING, guardrail_kpis ARRAY<STRING>,
  start_date DATE, end_date DATE, pre_window_days INT,
  treated_units ARRAY<STRUCT<city_norm:STRING, city_address:STRING,
                             state_address:STRING, weight:DOUBLE>>,
  control_units ARRAY<STRUCT<city_norm:STRING, city_address:STRING,
                             state_address:STRING, weight:DOUBLE>>,
  mde_estimated DOUBLE, expected_effect DOUBLE, decision_rule STRING,
  merchant_type STRING, created_by STRING, created_at TIMESTAMP,
  config_json STRING, design_spec_json STRING, design_fingerprint STRING,
  calibration_fingerprint STRING, power_result_json STRING,
  power_fingerprint STRING, contract_version STRING
) USING DELTA
"""


def ensure_registry(spark, table: str) -> None:
    schema = table.rsplit(".", 1)[0]
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    spark.sql(REGISTRY_DDL.format(table=table))
    existing = set(spark.table(table).columns)
    additions = {
        "design_spec_json": "STRING",
        "design_fingerprint": "STRING",
        "calibration_fingerprint": "STRING",
        "power_result_json": "STRING",
        "power_fingerprint": "STRING",
        "contract_version": "STRING",
    }
    missing = [f"{name} {kind}" for name, kind in additions.items() if name not in existing]
    if missing:
        spark.sql(f"ALTER TABLE {table} ADD COLUMNS ({', '.join(missing)})")


def _latest_by_experiment(spark, table: str):
    """Estado corrente de um registry append-only (uma linha por experimento)."""
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    latest = Window.partitionBy("experiment_id").orderBy(F.col("created_at").desc())
    return (spark.table(table)
            .withColumn("_registry_rank", F.row_number().over(latest))
            .where(F.col("_registry_rank") == 1)
            .drop("_registry_rank"))


def _registry_spark_schema():
    """Schema explícito espelhando REGISTRY_DDL — inferência falha com campos
    None e arrays vazios (CANNOT_DETERMINE_TYPE), ex.: controle fixo sem
    treated_units e com weight=None."""
    from pyspark.sql import types as T
    unit = T.ArrayType(T.StructType([
        T.StructField("city_norm", T.StringType()),
        T.StructField("city_address", T.StringType()),
        T.StructField("state_address", T.StringType()),
        T.StructField("weight", T.DoubleType()),
    ]))
    return T.StructType([
        T.StructField("experiment_id", T.StringType(), False),
        T.StructField("name", T.StringType()),
        T.StructField("hypothesis", T.StringType()),
        T.StructField("status", T.StringType()),
        T.StructField("design_method", T.StringType()),
        T.StructField("primary_kpi", T.StringType()),
        T.StructField("guardrail_kpis", T.ArrayType(T.StringType())),
        T.StructField("start_date", T.DateType()),
        T.StructField("end_date", T.DateType()),
        T.StructField("pre_window_days", T.IntegerType()),
        T.StructField("treated_units", unit),
        T.StructField("control_units", unit),
        T.StructField("mde_estimated", T.DoubleType()),
        T.StructField("expected_effect", T.DoubleType()),
        T.StructField("decision_rule", T.StringType()),
        T.StructField("merchant_type", T.StringType()),
        T.StructField("created_by", T.StringType()),
        T.StructField("created_at", T.TimestampType()),
        T.StructField("config_json", T.StringType()),
        T.StructField("design_spec_json", T.StringType()),
        T.StructField("design_fingerprint", T.StringType()),
        T.StructField("calibration_fingerprint", T.StringType()),
        T.StructField("power_result_json", T.StringType()),
        T.StructField("power_fingerprint", T.StringType()),
        T.StructField("contract_version", T.StringType()),
    ])


def save_experiment(
    spark,
    table: str,
    rec: ExperimentRecord,
    require_approval_checks: bool = True,
    alpha: float = 0.10,
    allow_legacy_unbound: bool = False,
) -> None:
    if rec.status not in _VALID_STATUSES:
        raise ValueError(f"status inválido: {rec.status!r}")
    legacy_path = allow_legacy_unbound or not require_approval_checks
    if legacy_path:
        warnings.warn(
            "LEGACY UNBOUND registry write: design/calibration/power binding is not enforced",
            DeprecationWarning,
            stacklevel=2,
        )
    if require_approval_checks and rec.status in ("approved", "running"):
        problems = rec.validate_for_approval(
            alpha=alpha, allow_legacy_unbound=allow_legacy_unbound
        )
        if rec.status == "approved" and date.today() >= rec.start_date:
            problems.append(
                "approval ocorreu em ou após start_date — não é pré-registro"
            )
        if problems:
            raise ValueError("Pré-registro reprovado:\n  - " + "\n  - ".join(problems))

    previous = None
    try:
        previous = load_experiment(spark, table, rec.experiment_id)
    except ValueError as exc:
        if "não encontrado" not in str(exc):
            raise
    if (
        previous is None
        and rec.status in {"running", "completed", "analyzed"}
        and not legacy_path
    ):
        raise ValueError(
            f"registro novo não pode iniciar em {rec.status}; aprove-o primeiro"
        )
    if previous is not None:
        transition_problems = validate_experiment_transition(
            previous, rec, allow_legacy_unbound=legacy_path
        )
        if transition_problems:
            raise ValueError(
                "Transição de registry reprovada:\n  - "
                + "\n  - ".join(transition_problems)
            )

    contract_version = "legacy-unbound" if legacy_path else rec.contract_version
    row = {
        **{k: getattr(rec, k) for k in [
            "experiment_id", "name", "hypothesis", "status", "design_method",
            "primary_kpi", "guardrail_kpis", "start_date", "end_date",
            "pre_window_days", "treated_units", "control_units", "mde_estimated",
            "expected_effect", "decision_rule", "merchant_type", "created_by",
            "config_json", "design_spec_json", "design_fingerprint",
            "calibration_fingerprint", "power_result_json",
            "power_fingerprint"]},
        "contract_version": contract_version,
        "created_at": datetime.now(timezone.utc),
    }
    row["pre_window_days"] = int(row["pre_window_days"] or 0)
    for k in ("mde_estimated", "expected_effect"):
        row[k] = float(row[k]) if row[k] is not None else None
    df = spark.createDataFrame([row], schema=_registry_spark_schema())
    df.write.format("delta").mode("append").saveAsTable(table)


def load_experiment(spark, table: str, experiment_id: str) -> ExperimentRecord:
    from pyspark.sql import functions as F

    rows = (_latest_by_experiment(spark, table)
            .where(F.col("experiment_id") == F.lit(experiment_id))
            .limit(1).collect())
    if not rows:
        raise ValueError(f"experiment_id não encontrado: {experiment_id}")
    r = rows[0].asDict(recursive=True)
    return ExperimentRecord(
        experiment_id=r["experiment_id"], name=r.get("name") or "",
        hypothesis=r.get("hypothesis") or "", status=r.get("status") or "",
        design_method=r.get("design_method") or "", primary_kpi=r.get("primary_kpi") or "gmv",
        guardrail_kpis=list(r.get("guardrail_kpis") or []),
        start_date=r["start_date"], end_date=r["end_date"],
        pre_window_days=int(r.get("pre_window_days") or 92),
        treated_units=list(r.get("treated_units") or []),
        control_units=list(r.get("control_units") or []),
        mde_estimated=r.get("mde_estimated"), expected_effect=r.get("expected_effect"),
        decision_rule=r.get("decision_rule") or "", merchant_type=r.get("merchant_type") or "",
        created_by=r.get("created_by") or "", config_json=r.get("config_json") or "{}",
        design_spec_json=r.get("design_spec_json") or "",
        design_fingerprint=r.get("design_fingerprint") or "",
        calibration_fingerprint=r.get("calibration_fingerprint") or "",
        power_result_json=r.get("power_result_json") or "",
        power_fingerprint=r.get("power_fingerprint") or "",
        contract_version=r.get("contract_version") or "legacy-unbound",
    )


def load_fixed_control(spark, table: str,
                       name: str = "fixed_control_groceries") -> Optional[List[str]]:
    """Cidades do controle fixo mais recente registrado; None se não houver."""
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    ensure_registry(spark, table)
    current = _latest_by_experiment(spark, table).where(F.col("name") == name)
    latest = (current.withColumn(
        "_fixed_rank", Window.partitionBy("name").orderBy(F.col("created_at").desc())
    ).where(F.col("_fixed_rank") == 1).drop("_fixed_rank"))
    rows = latest.where(F.col("status") == "approved").limit(1).collect()
    if not rows:
        return None
    r = rows[0].asDict(recursive=True)
    cities = [u["city_norm"] for u in (r.get("control_units") or []) if u.get("city_norm")]
    return cities or None


def active_blocked_cities(spark, table: str, merchant_type: str = "Groceries") -> dict:
    """Cidades bloqueadas: tratadas de experimentos ativos (não podem ser nada)
    e controles ativos (não podem ser tratadas)."""
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    ensure_registry(spark, table)  # primeira execução: registry ainda não existe
    current = _latest_by_experiment(spark, table)

    # O controle fixo é uma configuração versionada por nome, não um conjunto
    # de experimentos concorrentes. Só a versão mais recente deve bloquear uma
    # cidade; as anteriores permanecem no histórico auditável.
    is_fixed = ((F.col("design_method") == "did")
                & (F.size(F.col("treated_units")) == 0))
    fixed = current.where(is_fixed)
    latest_fixed = (fixed.withColumn(
        "_fixed_rank", Window.partitionBy("name").orderBy(F.col("created_at").desc())
    ).where(F.col("_fixed_rank") == 1).drop("_fixed_rank"))
    regular = current.where(~is_fixed)
    df = (regular.unionByName(latest_fixed)
          .where(F.col("merchant_type") == merchant_type)
          .where(F.col("status").isin(["approved", "running"])))

    def _norms(col):
        return set(r["cn"] for r in
                   df.select(F.explode_outer(col).alias("u"))
                     .select(F.upper(F.trim(F.col("u.city_norm"))).alias("cn"))
                     .where(F.col("cn").isNotNull()).distinct().collect())
    return {"treated_active": _norms("treated_units"), "control_active": _norms("control_units")}
