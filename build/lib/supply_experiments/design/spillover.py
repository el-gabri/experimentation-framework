"""Regras de spillover geográfico e elegibilidade de cidades.

Doadoras/controles dentro do raio de spillover das tratadas são excluídas do
donor pool: merchants e consumidores transitam entre cidades vizinhas (região
metropolitana), contaminando o contrafactual e viesando o efeito para zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def spillover_exclusions(
    treated: Sequence[str],
    candidates: Sequence[str],
    coords: Dict[str, Tuple[float, float]],
    radius_km: float = 40.0,
    adjacency: Optional[Dict[str, Set[str]]] = None,
) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """
    Retorna (candidatas_limpas, excluídas com motivo).
    Usa coordenadas (raio em km) e/ou lista de adjacência explícita (ex.: mesma
    região metropolitana). Cidades sem coordenada não são excluídas por raio,
    mas são reportadas.
    """
    excluded: List[Tuple[str, str, float]] = []
    clean: List[str] = []
    adjacency = adjacency or {}
    adj_set: Set[str] = set()
    for t in treated:
        adj_set |= set(adjacency.get(t, set()))

    for c in candidates:
        if c in set(treated):
            continue
        if c in adj_set:
            excluded.append((c, "adjacente (região metropolitana)", np.nan))
            continue
        if c in coords:
            dmin = min(
                (haversine_km(*coords[c], *coords[t]) for t in treated if t in coords),
                default=np.inf,
            )
            if dmin < radius_km:
                excluded.append((c, f"raio < {radius_km:.0f} km", dmin))
                continue
        clean.append(c)
    return clean, excluded


@dataclass
class EligibilityCriteria:
    min_nonzero_days: int = 92
    min_avg_daily_gmv: float = 10_000.0
    min_avg_daily_merchants: float = 10.0
    max_avg_daily_merchants: float = 2_500.0
    min_avg_daily_orders: float = 50.0
    max_zero_run_days: int = 7          # NOVO: sequência máxima de zeros (buracos de dados)
    max_cv: float = 1.5                 # NOVO: coef. de variação máximo (séries erráticas)


def eligible_cities(stats: pd.DataFrame, crit: EligibilityCriteria) -> pd.DataFrame:
    """stats: DataFrame com colunas nonzero_days, avg_daily_gmv, avg_daily_merchants,
    avg_daily_orders, max_zero_run, cv_gmv (indexado por city_norm)."""
    m = (
        (stats["nonzero_days"] >= crit.min_nonzero_days)
        & (stats["avg_daily_gmv"] >= crit.min_avg_daily_gmv)
        & (stats["avg_daily_merchants"].between(crit.min_avg_daily_merchants,
                                                crit.max_avg_daily_merchants))
        & (stats["avg_daily_orders"] >= crit.min_avg_daily_orders)
    )
    if "max_zero_run" in stats.columns:
        m &= stats["max_zero_run"] <= crit.max_zero_run_days
    if "cv_gmv" in stats.columns:
        m &= stats["cv_gmv"] <= crit.max_cv
    return stats.loc[m].copy()


def max_zero_run(y: np.ndarray) -> int:
    run = best = 0
    for v in y:
        run = run + 1 if v <= 0 else 0
        best = max(best, run)
    return best
