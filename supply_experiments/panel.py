"""Estruturas de painel cidade x dia. Pandas puro (testável fora do Databricks)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentWindow:
    """Janelas pré/pós de um experimento."""

    start_date: date          # primeiro dia do tratamento
    end_date: date            # último dia do tratamento
    pre_window_days: int

    @property
    def pre_end(self) -> date:
        return self.start_date - timedelta(days=1)

    @property
    def pre_start(self) -> date:
        return self.pre_end - timedelta(days=self.pre_window_days - 1)

    @property
    def post_days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    def masks(self, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        d = index.date
        pre = (d >= self.pre_start) & (d <= self.pre_end)
        post = (d >= self.start_date) & (d <= self.end_date)
        return pre, post


@dataclass
class CityPanel:
    """
    Painel wide: index=DatetimeIndex diário contíguo, columns=city_norm, values=KPI.

    `outcome` é o painel do KPI primário (ex.: GMV). KPIs de razão são mantidos
    como pares numerador/denominador em `numerators`/`denominators` para permitir
    o método delta (nunca guardamos a razão diária diretamente).
    """

    outcome: pd.DataFrame
    numerators: Dict[str, pd.DataFrame] = field(default_factory=dict)
    denominators: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self) -> None:
        idx = self.outcome.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise TypeError("outcome.index deve ser DatetimeIndex")
        full = pd.date_range(idx.min(), idx.max(), freq="D")
        if len(full) != len(idx):
            self.outcome = self.outcome.reindex(full).fillna(0.0)
        self.outcome = self.outcome.astype(float)

    @property
    def cities(self) -> List[str]:
        return [str(c) for c in self.outcome.columns]

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.outcome.index

    def subset(self, cities: Sequence[str]) -> "CityPanel":
        cities = [c for c in cities if c in self.outcome.columns]
        return CityPanel(
            outcome=self.outcome[cities].copy(),
            numerators={k: v[[c for c in cities if c in v.columns]].copy() for k, v in self.numerators.items()},
            denominators={k: v[[c for c in cities if c in v.columns]].copy() for k, v in self.denominators.items()},
        )

    def aggregate(self, cities: Sequence[str], weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """Série agregada (soma ou média ponderada) do outcome para um grupo."""
        cities = [c for c in cities if c in self.outcome.columns]
        if not cities:
            raise ValueError("Nenhuma cidade do grupo existe no painel")
        if weights:
            w = _normalize_weights({c: weights.get(c, 0.0) for c in cities})
            return sum(self.outcome[c] * w[c] for c in cities if w.get(c, 0) > 0)
        return self.outcome[cities].sum(axis=1)

    def ratio_series(self, kpi: str, cities: Sequence[str], weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """Razão de somas diária (numerador agregado / denominador agregado)."""
        num, den = self.numerators[kpi], self.denominators[kpi]
        cities = [c for c in cities if c in num.columns]
        if weights:
            w = _normalize_weights({c: weights.get(c, 0.0) for c in cities})
            n = sum(num[c] * w[c] for c in cities if w.get(c, 0) > 0)
            d = sum(den[c] * w[c] for c in cities if w.get(c, 0) > 0)
        else:
            n, d = num[cities].sum(axis=1), den[cities].sum(axis=1)
        return n / d.replace(0, np.nan)

    def long_format(self, cities: Sequence[str]) -> pd.DataFrame:
        """Painel long (city, date, y) para estimadores em nível de unidade."""
        sub = self.outcome[[c for c in cities if c in self.outcome.columns]]
        out = sub.stack().rename("y").reset_index()
        out.columns = ["date", "city", "y"]
        return out


def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    w = {k: float(v) for k, v in w.items() if v and v > 0}
    total = sum(w.values())
    if total <= 0:
        raise ValueError("Pesos inválidos (soma <= 0)")
    return {k: v / total for k, v in w.items()}


def make_dow_dummies(index: pd.DatetimeIndex) -> np.ndarray:
    """Matriz (T, 6) de dummies de dia da semana (baseline = segunda)."""
    dow = index.dayofweek.to_numpy()
    return np.column_stack([(dow == k).astype(float) for k in range(1, 7)])


def make_holiday_dummy(index: pd.DatetimeIndex, holidays: Sequence[str]) -> np.ndarray:
    hs = set(pd.to_datetime(list(holidays)).date)
    return np.array([d in hs for d in index.date], dtype=float)
