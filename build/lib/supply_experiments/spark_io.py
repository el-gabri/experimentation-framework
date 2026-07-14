"""Adapters Databricks/Spark — única fonte de verdade para dados e registry.

Todo import de pyspark é local às funções: o resto do pacote roda e é testado
sem Spark. Os 4 notebooks antigos duplicavam build_orders_base com divergências
sutis; aqui existe UMA implementação.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from supply_experiments.panel import CityPanel

# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------

TABLES = {
    "sandbox": {
        "registry": "main.groceries_sandbox.experimentation_registry_v2",
        "results": "main.groceries_sandbox.experiments_results_v2",
        "calibration": "main.groceries_sandbox.experiments_calibration",
    },
    "prod": {
        "registry": "groceries_ops.tesselation.experimentation_registry_v2",
        "results": "groceries_ops.experiments_results_v2",
        "calibration": "groceries_ops.experiments_calibration",
    },
}
ORDERS_TABLE = "main.order_groceries.orders"

NATIONAL_HOLIDAYS = [
    "2024-12-25", "2024-12-31", "2025-01-01", "2025-03-03", "2025-03-04",
    "2025-04-18", "2025-04-21", "2025-05-01", "2025-06-19", "2025-09-07",
    "2025-10-12", "2025-11-02", "2025-11-15", "2025-12-25", "2025-12-31",
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-04-03", "2026-04-21",
    "2026-05-01", "2026-06-04", "2026-09-07", "2026-10-12", "2026-11-02",
    "2026-11-15",
]


# ---------------------------------------------------------------------------
# Orders base + painel
# ---------------------------------------------------------------------------

def build_orders_base(spark, source_table: str = ORDERS_TABLE):
    """Base filtrada (Groceries Marketplace Concluded) — implementação única."""
    from pyspark.sql import functions as F

    src = spark.table(source_table)
    cols = set(src.columns)

    exprs = [F.col(c) for c in ["dt", "order_id", "merchant_id", "merchant_city",
                                "last_status", "groceries_classification",
                                "marketplace_classification", "order_total"]]
    exprs.append(F.col("order_reference_date_local") if "order_reference_date_local" in cols
                 else F.lit(None).cast("timestamp").alias("order_reference_date_local"))
    if "merchant_state_label" in cols:
        exprs.append(F.col("merchant_state_label").alias("merchant_state"))
    elif "merchant_state" in cols:
        exprs.append(F.col("merchant_state"))
    else:
        exprs.append(F.lit(None).cast("string").alias("merchant_state"))
    exprs.append(F.col("otif_struct.has_rupture_ticket").alias("has_rupture_order")
                 if "otif_struct" in cols else F.lit(None).cast("boolean").alias("has_rupture_order"))

    return (
        src.select(*exprs)
        .where(F.col("last_status") == "CONCLUDED")
        .where(F.col("groceries_classification").isin(["Mercado", "Atacado"]))
        .where(F.col("marketplace_classification") == "Marketplace")
        .where(F.col("merchant_city").isNotNull())
        .where(F.col("merchant_id").isNotNull() & F.col("order_id").isNotNull())
        .withColumn("city_norm", F.upper(F.trim(F.col("merchant_city"))))
        .withColumn("order_date", F.coalesce(F.to_date("order_reference_date_local"),
                                             F.to_date("dt")))
        .where(F.col("order_date").isNotNull())
    )


def load_city_panel(spark, date_start: str, date_end: str,
                    cities: Optional[Sequence[str]] = None,
                    source_table: str = ORDERS_TABLE) -> tuple[CityPanel, pd.DataFrame]:
    """Carrega o painel diário (GMV + numeradores/denominadores de ruptura) e
    estatísticas por cidade (para elegibilidade)."""
    from pyspark.sql import functions as F

    base = build_orders_base(spark, source_table)
    base = base.where((F.col("order_date") >= F.lit(date_start))
                      & (F.col("order_date") <= F.lit(date_end)))
    if cities:
        base = base.where(F.col("city_norm").isin([str(c) for c in cities]))

    daily = (
        base.groupBy("order_date", "city_norm")
        .agg(F.sum(F.col("order_total").cast("double")).alias("gmv"),
             F.countDistinct("order_id").alias("orders"),
             F.countDistinct("merchant_id").alias("merchants"),
             F.sum(F.when(F.col("has_rupture_order") == True, 1).otherwise(0)).alias("rupture_orders"),
             F.first("merchant_city", ignorenulls=True).alias("city_address"),
             F.first("merchant_state", ignorenulls=True).alias("state_address"))
        .toPandas()
    )
    daily["order_date"] = pd.to_datetime(daily["order_date"])
    idx = pd.date_range(pd.to_datetime(date_start), pd.to_datetime(date_end), freq="D")

    def wide(col):
        w = daily.pivot_table(index="order_date", columns="city_norm", values=col,
                              aggfunc="sum").reindex(idx)
        return w

    gmv = wide("gmv").fillna(0.0)
    orders = wide("orders").fillna(0.0)
    rupt = wide("rupture_orders").fillna(0.0)

    panel = CityPanel(outcome=gmv,
                      numerators={"rupture_rate_order": rupt},
                      denominators={"rupture_rate_order": orders})

    from supply_experiments.design.spillover import max_zero_run
    stats = (
        daily.groupby("city_norm")
        .agg(avg_daily_gmv=("gmv", "mean"), sum_gmv=("gmv", "sum"),
             avg_daily_orders=("orders", "mean"), avg_daily_merchants=("merchants", "mean"),
             nonzero_days=("gmv", lambda s: int((s > 0).sum())),
             total_orders=("orders", "sum"), total_ruptures=("rupture_orders", "sum"),
             city_address=("city_address", "first"), state_address=("state_address", "first"))
    )
    stats["rupture_rate"] = stats["total_ruptures"] / stats["total_orders"].replace(0, np.nan)
    stats["cv_gmv"] = gmv.std() / gmv.mean().replace(0, np.nan)
    stats["max_zero_run"] = [max_zero_run(gmv[c].to_numpy()) if c in gmv.columns else 999
                             for c in stats.index]
    return panel, stats


# ---------------------------------------------------------------------------
# Registry v2 — pré-registro como contrato
# ---------------------------------------------------------------------------

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

    def validate_for_approval(self) -> List[str]:
        problems = []
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
        if len(self.control_units) < 8:
            problems.append(f"só {len(self.control_units)} controles/doadoras "
                            "(mínimo 8 p/ granularidade de p-value)")
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
    spark.sql(REGISTRY_DDL.format(table=table))


def save_experiment(spark, table: str, rec: ExperimentRecord,
                    require_approval_checks: bool = True) -> None:
    if require_approval_checks and rec.status in ("approved", "running"):
        problems = rec.validate_for_approval()
        if problems:
            raise ValueError("Pré-registro reprovado:\n  - " + "\n  - ".join(problems))
    row = {
        **{k: getattr(rec, k) for k in [
            "experiment_id", "name", "hypothesis", "status", "design_method",
            "primary_kpi", "guardrail_kpis", "start_date", "end_date",
            "pre_window_days", "treated_units", "control_units", "mde_estimated",
            "expected_effect", "decision_rule", "merchant_type", "created_by",
            "config_json"]},
        "created_at": datetime.utcnow(),
    }
    df = spark.createDataFrame([row])
    df.write.format("delta").mode("append").saveAsTable(table)


def load_experiment(spark, table: str, experiment_id: str) -> ExperimentRecord:
    rows = (spark.table(table)
            .where(f"experiment_id = '{experiment_id}'")
            .orderBy("created_at", ascending=False).limit(1).collect())
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


def active_blocked_cities(spark, table: str, merchant_type: str = "Groceries") -> dict:
    """Cidades bloqueadas: tratadas de experimentos ativos (não podem ser nada)
    e controles ativos (não podem ser tratadas)."""
    from pyspark.sql import functions as F
    df = (spark.table(table)
          .where(F.col("merchant_type") == merchant_type)
          .where(F.col("status").isin(["approved", "running"])))
    def _norms(col):
        return set(r["cn"] for r in
                   df.select(F.explode_outer(col).alias("u"))
                     .select(F.upper(F.trim(F.col("u.city_norm"))).alias("cn"))
                     .where(F.col("cn").isNotNull()).distinct().collect())
    return {"treated_active": _norms("treated_units"), "control_active": _norms("control_units")}
