"""Registry v2 — pré-registro como contrato.

`ExperimentRecord.validate_for_approval` é o gate que impede experimentos sem
hipótese, sem regra de decisão, ou sem poder estatístico suficiente de virar
"approved"/"running" via `save_experiment`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

_VALID_STATUSES = {"draft", "approved", "running", "completed", "analyzed", "cancelled"}


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

    def validate_for_approval(self, alpha: float = 0.10) -> List[str]:
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
        elif self.expected_effect is not None and self.mde_estimated > abs(self.expected_effect):
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
        min_donors = minimum_donors_for_alpha(alpha)
        if len(control_cities) < min_donors:
            problems.append(f"só {len(control_cities)} controles/doadoras "
                            f"(mínimo {min_donors} para α={alpha:.3g})")
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
  config_json STRING
) USING DELTA
"""


def ensure_registry(spark, table: str) -> None:
    schema = table.rsplit(".", 1)[0]
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    spark.sql(REGISTRY_DDL.format(table=table))


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
    ])


def save_experiment(spark, table: str, rec: ExperimentRecord,
                    require_approval_checks: bool = True, alpha: float = 0.10) -> None:
    if rec.status not in _VALID_STATUSES:
        raise ValueError(f"status inválido: {rec.status!r}")
    if require_approval_checks and rec.status in ("approved", "running"):
        problems = rec.validate_for_approval(alpha=alpha)
        if problems:
            raise ValueError("Pré-registro reprovado:\n  - " + "\n  - ".join(problems))
    row = {
        **{k: getattr(rec, k) for k in [
            "experiment_id", "name", "hypothesis", "status", "design_method",
            "primary_kpi", "guardrail_kpis", "start_date", "end_date",
            "pre_window_days", "treated_units", "control_units", "mde_estimated",
            "expected_effect", "decision_rule", "merchant_type", "created_by",
            "config_json"]},
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
