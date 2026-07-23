"""Adapters Databricks/Spark, divididos por responsabilidade:

  - tables:   nomes de tabela/catálogo e calendário de feriados
  - etl:      base de pedidos + painel de cidades (build_orders_base, load_city_panel)
  - registry: registry v2 de experimentos (pré-registro, aprovação, bloqueio de cidades)

`supply_experiments.spark_io` re-exporta tudo daqui para compatibilidade com
notebooks existentes — prefira importar de `supply_experiments.io.*` em código novo.
"""

from supply_experiments.io.etl import build_orders_base, load_city_panel
from supply_experiments.io.registry import (
    REGISTRY_DDL,
    ExperimentRecord,
    active_blocked_cities,
    ensure_registry,
    load_experiment,
    load_fixed_control,
    minimum_donors_for_alpha,
    save_experiment,
)
from supply_experiments.io.tables import NATIONAL_HOLIDAYS, ORDERS_TABLE, TABLES

__all__ = [
    "TABLES", "ORDERS_TABLE", "NATIONAL_HOLIDAYS",
    "build_orders_base", "load_city_panel",
    "ExperimentRecord", "REGISTRY_DDL", "ensure_registry",
    "save_experiment", "load_experiment", "load_fixed_control", "active_blocked_cities",
    "minimum_donors_for_alpha",
]
