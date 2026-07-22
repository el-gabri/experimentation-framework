"""Nomes de tabela e calendário — dados compartilhados por `etl.py` e `registry.py`.

Os nomes de tabela abaixo são placeholders — aponte-os para o catálogo/schema
do seu ambiente antes de usar. `TABLES["*"]["results"]` está reservado para um
futuro par save_results/load_results; nenhuma função escreve nele hoje.
"""

from __future__ import annotations

TABLES = {
    "sandbox": {
        "registry": "catalog.sandbox.experimentation_registry",
        "results": "catalog.sandbox.experiments_results",
        "calibration": "catalog.sandbox.experiments_calibration",
    },
    "prod": {
        "registry": "catalog.prod.experimentation_registry",
        "results": "catalog.prod.experiments_results",
        "calibration": "catalog.prod.experiments_calibration",
    },
}
ORDERS_TABLE = "catalog.schema.orders"

NATIONAL_HOLIDAYS = [
    "2024-12-25", "2024-12-31", "2025-01-01", "2025-03-03", "2025-03-04",
    "2025-04-18", "2025-04-21", "2025-05-01", "2025-06-19", "2025-09-07",
    "2025-10-12", "2025-11-02", "2025-11-15", "2025-12-25", "2025-12-31",
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-04-03", "2026-04-21",
    "2026-05-01", "2026-06-04", "2026-09-07", "2026-10-12", "2026-11-02",
    "2026-11-15",
]
