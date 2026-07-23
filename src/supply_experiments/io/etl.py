"""ETL: base de pedidos + painel de cidades.

Import de pyspark é local às funções: o resto do pacote roda e é testado sem
Spark. `build_orders_base` tem UMA implementação, compartilhada por todos os
notebooks — os 4 notebooks antigos duplicavam essa lógica com divergências
sutis.
"""

from __future__ import annotations

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
             # `== True` (não truthy simples) trata NULL explicitamente como falso
             F.sum(F.when(F.col("has_rupture_order") == True, 1).otherwise(0))  # noqa: E712
             .alias("rupture_orders"),
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
    merchants = wide("merchants").fillna(0.0)
    rupt = wide("rupture_orders").fillna(0.0)

    panel = CityPanel(outcome=gmv,
                      numerators={"rupture_rate_order": rupt},
                      denominators={"rupture_rate_order": orders})

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
    stats["total_ruptures"] = rupt.sum()
    stats["rupture_rate"] = stats["total_ruptures"] / stats["total_orders"].replace(0, np.nan)
    stats["cv_gmv"] = gmv.std() / gmv.mean().replace(0, np.nan)
    stats["max_zero_run"] = [max_zero_run(gmv[c].to_numpy()) for c in stats.index]
    return panel, stats
