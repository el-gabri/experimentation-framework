"""Compatibilidade: `supply_experiments.spark_io` foi dividido em
`supply_experiments.io.{tables,etl,registry}` por responsabilidade (ETL vs.
registry v2 vs. nomes de tabela). Este módulo reexporta a mesma superfície
pública para não quebrar os notebooks existentes — prefira importar de
`supply_experiments.io` em código novo.
"""

from __future__ import annotations

from supply_experiments.io import (
    NATIONAL_HOLIDAYS,
    ORDERS_TABLE,
    REGISTRY_DDL,
    TABLES,
    ExperimentRecord,
    active_blocked_cities,
    build_orders_base,
    ensure_registry,
    load_city_panel,
    load_experiment,
    load_fixed_control,
    save_experiment,
)

__all__ = [
    "TABLES", "ORDERS_TABLE", "NATIONAL_HOLIDAYS",
    "build_orders_base", "load_city_panel",
    "ExperimentRecord", "REGISTRY_DDL", "ensure_registry",
    "save_experiment", "load_experiment", "load_fixed_control", "active_blocked_cities",
]
