"""ETL: base de pedidos + painel de cidades.

Import de pyspark é local às funções: o resto do pacote roda e é testado sem
Spark. `build_orders_base` tem UMA implementação, compartilhada por todos os
notebooks — os 4 notebooks antigos duplicavam essa lógica com divergências
sutis.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from supply_experiments.io.tables import ORDERS_TABLE
from supply_experiments.panel import CityPanel


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
    otif_type = src.schema["otif_struct"].dataType if "otif_struct" in cols else None
    has_rupture_field = "has_rupture_ticket" in getattr(otif_type, "names", ())
    exprs.append(F.col("otif_struct.has_rupture_ticket").alias("has_rupture_order")
                 if has_rupture_field else F.lit(None).cast("boolean").alias("has_rupture_order"))

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
                    source_table: str = ORDERS_TABLE, *,
                    assume_missing_city_days_zero: bool = False,
                    require_rupture_telemetry: bool = False) -> tuple[CityPanel, pd.DataFrame]:
    """Carrega o painel diário (GMV + numeradores/denominadores de ruptura) e
    estatísticas por cidade (para elegibilidade).

    Ausência de linhas só significa zero atividade quando o produtor da fonte
    garante sua completude e ``assume_missing_city_days_zero=True``. GMV
    observado inválido nunca é preenchido. Ruptura incompleta é omitida ou,
    quando ``require_rupture_telemetry=True``, impede a carga.
    """
    from pyspark.sql import functions as F

    base = build_orders_base(spark, source_table)
    base = base.where((F.col("order_date") >= F.lit(date_start))
                      & (F.col("order_date") <= F.lit(date_end)))
    if cities:
        base = base.where(F.col("city_norm").isin([str(c) for c in cities]))

    # A city name is usable as an ID only if it maps to one known state.
    # Check before aggregation, where first(state) would erase the collision.
    base = base.withColumn("merchant_state", F.upper(F.trim(F.col("merchant_state"))))
    ambiguous = (base.groupBy("city_norm")
                 .agg(F.countDistinct(F.when(F.col("merchant_state") != "",
                                            F.col("merchant_state"))).alias("n_states"))
                 .where(F.col("n_states") > 1).select("city_norm").limit(5).collect())
    if ambiguous:
        raise ValueError(
            "city_norm ambíguo entre estados; use identificadores geográficos únicos na fonte: "
            f"{[row['city_norm'] for row in ambiguous]}"
        )
    base = base.withColumn("_order_total", F.col("order_total").cast("double"))
    invalid_money = (F.col("_order_total").isNull() | F.isnan("_order_total")
                     | (F.abs(F.col("_order_total")) == F.lit(float("inf"))))
    if base.where(invalid_money).limit(1).count():
        raise ValueError("order_total observado contém valor ausente ou não finito")

    daily = (
        base.groupBy("order_date", "city_norm")
        .agg(F.sum("_order_total").alias("gmv"),
             F.countDistinct("order_id").alias("orders"),
             F.countDistinct("merchant_id").alias("merchants"),
             F.count(F.lit(1)).alias("source_rows"),
             F.count("has_rupture_order").alias("rupture_observed_rows"),
             # Null counts are tracked separately and invalidate the guardrail.
             F.sum(F.when(F.col("has_rupture_order") == True, 1).otherwise(0))  # noqa: E712
             .alias("rupture_orders"),
             F.first("merchant_city", ignorenulls=True).alias("city_address"),
             F.first("merchant_state", ignorenulls=True).alias("state_address"))
        .toPandas()
    )
    return _panel_from_daily(
        daily, date_start, date_end, cities=cities,
        assume_missing_city_days_zero=assume_missing_city_days_zero,
        require_rupture_telemetry=require_rupture_telemetry,
    )


def _panel_from_daily(daily: pd.DataFrame, date_start: str, date_end: str, *,
                      cities: Optional[Sequence[str]] = None,
                      assume_missing_city_days_zero: bool = False,
                      require_rupture_telemetry: bool = False) -> tuple[CityPanel, pd.DataFrame]:
    """Validate observed aggregates before making a complete daily grid."""
    daily = daily.copy()
    if daily.empty:
        raise ValueError("nenhum dado observado para as cidades e datas solicitadas")
    daily["order_date"] = pd.to_datetime(daily["order_date"])
    idx = pd.date_range(pd.to_datetime(date_start), pd.to_datetime(date_end), freq="D")
    if idx.empty:
        raise ValueError("date_end deve ser >= date_start")
    if cities is not None:
        missing = sorted(set(cities) - set(daily["city_norm"]))
        if missing or len(set(cities)) != len(cities) or not cities:
            raise ValueError(f"cidades solicitadas ausentes ou duplicadas: {missing}")
    numeric = ["gmv", "orders", "merchants", "rupture_orders", "source_rows", "rupture_observed_rows"]
    if not np.isfinite(daily[numeric].to_numpy(float)).all():
        raise ValueError("agregados observados contêm valores ausentes ou não finitos")
    complete_rupture = bool((daily["source_rows"] == daily["rupture_observed_rows"]).all())
    if not complete_rupture:
        message = "telemetria de ruptura incompleta; rupture_rate_order indisponível"
        if require_rupture_telemetry:
            raise ValueError(message)
        warnings.warn(message, stacklevel=3)

    def wide(col):
        return daily.pivot(index="order_date", columns="city_norm", values=col).reindex(idx)

    gmv = wide("gmv")
    if gmv.isna().any().any() and not assume_missing_city_days_zero:
        raise ValueError(
            "city-days ausentes; confirme completude da fonte antes de usar "
            "assume_missing_city_days_zero=True"
        )
    # Only absent rows remain NaN here; invalid observed aggregates were rejected.
    gmv = gmv.fillna(0.0)
    orders = wide("orders").fillna(0.0)
    merchants = wide("merchants").fillna(0.0)
    rupt = wide("rupture_orders").fillna(0.0)

    panel = CityPanel(outcome=gmv,
                      numerators={"rupture_rate_order": rupt} if complete_rupture else {},
                      denominators={"rupture_rate_order": orders} if complete_rupture else {})

    from supply_experiments.design.spillover import max_zero_run
    # Métricas diárias precisam incluir dias sem pedido como zero. Calcular as
    # médias diretamente em ``daily`` as transformava em médias condicionais a
    # haver atividade, enviesando a elegibilidade para cidades intermitentes.
    metadata = (daily.sort_values(["city_norm", "order_date"])
                .groupby("city_norm")
                .agg(city_address=("city_address", "first"),
                     state_address=("state_address", "first")))
    stats = metadata.reindex(gmv.columns)
    stats.index.name = "city_norm"
    stats["avg_daily_gmv"] = gmv.mean()
    stats["sum_gmv"] = gmv.sum()
    stats["avg_daily_orders"] = orders.mean()
    stats["avg_daily_merchants"] = merchants.mean()
    stats["nonzero_days"] = (gmv > 0).sum()
    stats["total_orders"] = orders.sum()
    stats["rupture_telemetry_complete"] = complete_rupture
    stats["total_ruptures"] = rupt.sum() if complete_rupture else np.nan
    stats["rupture_rate"] = stats["total_ruptures"] / stats["total_orders"].replace(0, np.nan)
    stats["cv_gmv"] = gmv.std() / gmv.mean().replace(0, np.nan)
    stats["max_zero_run"] = [max_zero_run(gmv[c].to_numpy()) for c in stats.index]
    return panel, stats
