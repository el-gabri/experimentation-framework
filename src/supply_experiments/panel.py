"""Estruturas de painel cidade x dia. Pandas puro (testável fora do Databricks)."""

from __future__ import annotations

import warnings
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
    fill_value: float = 0.0   # valor para dias ausentes do índice diário contíguo

    def __post_init__(self) -> None:
        idx = self.outcome.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise TypeError("outcome.index deve ser DatetimeIndex")
        if idx.empty:
            raise ValueError("outcome não pode ser vazio")
        if not idx.is_unique:
            raise ValueError("outcome.index deve ter datas únicas")
        if not idx.is_monotonic_increasing:
            raise ValueError("outcome.index deve estar em ordem cronológica")
        if not self.outcome.columns.is_unique:
            raise ValueError("outcome.columns deve ter cidades únicas")
        full = pd.date_range(idx.min(), idx.max(), freq="D")
        if len(full) != len(idx):
            n_missing = len(full) - len(idx)
            warnings.warn(
                f"CityPanel: {n_missing} dia(s) ausentes no índice diário "
                f"({idx.min().date()}..{idx.max().date()}) preenchidos com "
                f"fill_value={self.fill_value}. Um buraco real de dados fica "
                f"indistinguível de um dia com outcome=0 para os estimadores; "
                f"use `max_zero_run`/`cv_gmv` na elegibilidade para filtrar "
                f"séries com buracos longos, ou passe fill_value=np.nan e "
                f"trate os NaN explicitamente antes de estimar.",
                stacklevel=2,
            )
            self.outcome = self.outcome.reindex(full).fillna(self.fill_value)
        self.outcome = self.outcome.astype(float)
        kpis = set(self.numerators) | set(self.denominators)
        if set(self.numerators) != set(self.denominators):
            missing_num = sorted(set(self.denominators) - set(self.numerators))
            missing_den = sorted(set(self.numerators) - set(self.denominators))
            raise ValueError(
                f"KPIs sem par numerador/denominador: numeradores ausentes={missing_num}, "
                f"denominadores ausentes={missing_den}"
            )
        for kpi in kpis:
            num = self._align_ratio_frame(self.numerators[kpi], kpi, "numerador", full)
            den = self._align_ratio_frame(self.denominators[kpi], kpi, "denominador", full)
            if not num.columns.equals(den.columns):
                raise ValueError(f"KPI '{kpi}' tem cidades diferentes em numerador e denominador")
            self.numerators[kpi] = num
            self.denominators[kpi] = den

    def _align_ratio_frame(self, frame: pd.DataFrame, kpi: str, role: str,
                           full_index: pd.DatetimeIndex) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{role} do KPI '{kpi}' deve ser DataFrame")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"{role} do KPI '{kpi}' deve ter DatetimeIndex")
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{role} do KPI '{kpi}' deve ter datas únicas e ordenadas")
        if not frame.columns.is_unique:
            raise ValueError(f"{role} do KPI '{kpi}' deve ter cidades únicas")
        unknown = frame.columns.difference(self.outcome.columns)
        if len(unknown):
            raise ValueError(f"{role} do KPI '{kpi}' contém cidades fora do outcome: {list(unknown)}")
        if not frame.index.equals(full_index):
            warnings.warn(
                f"CityPanel: {role} do KPI '{kpi}' foi alinhado ao índice do outcome; "
                f"datas ausentes receberam fill_value={self.fill_value}.",
                stacklevel=3,
            )
            frame = frame.reindex(full_index).fillna(self.fill_value)
        return frame.astype(float)

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
            fill_value=self.fill_value,
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
        if kpi not in self.numerators:
            raise KeyError(f"KPI de razão inexistente: {kpi}")
        num, den = self.numerators[kpi], self.denominators[kpi]
        cities = [c for c in cities if c in num.columns and c in den.columns]
        if not cities:
            raise ValueError(f"Nenhuma cidade do grupo tem dados para o KPI '{kpi}'")
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

    def slice_for(
        self,
        treated: Sequence[str],
        donors: Sequence[str],
        window: "ExperimentWindow",
        weights: Optional[Dict[str, float]] = None,
    ) -> "PanelSlice":
        """Monta a fatia pré/pós (y_pre, Yd_pre, y_post, Yd_post, donor_names)
        que todo estimador/inferência consome, a partir de uma janela.

        Substitui o boilerplate repetido nos call sites (reporting.py e afins):
        `pre_mask, post_mask = window.masks(panel.index); y = panel.aggregate(...)...`
        """
        donors = [d for d in donors if d in self.outcome.columns and d not in set(treated)]
        pre_mask, post_mask = window.masks(self.index)
        y = self.aggregate(treated, weights).to_numpy(float)
        Yd = self.outcome[donors].to_numpy(float)
        return PanelSlice(
            y_pre=y[pre_mask], Yd_pre=Yd[pre_mask],
            y_post=y[post_mask], Yd_post=Yd[post_mask],
            donor_names=donors,
        )


@dataclass
class PanelSlice:
    """Fatia pré/pós de um painel, pronta para estimador/inferência.

    Encapsula a 5-tupla `(y_pre, Yd_pre, y_post, Yd_post, donor_names)` que
    `fit_scm`/`fit_ascm`/`fit_sdid`, `placebo_inference` e `conformal_inference`
    recebem posicionalmente — construída uma vez via `CityPanel.slice_for(...)`
    em vez de repetida em cada call site.
    """

    y_pre: np.ndarray
    Yd_pre: np.ndarray
    y_post: np.ndarray
    Yd_post: np.ndarray
    donor_names: List[str]

    def as_args(self) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]":
        """5-tupla posicional esperada pelos estimadores (`fit_fn(*slice.as_args())`)."""
        return (self.y_pre, self.Yd_pre, self.y_post, self.Yd_post, self.donor_names)

    def fit(self, fit_fn, **kwargs):
        return fit_fn(self.y_pre, self.Yd_pre, self.y_post, self.Yd_post,
                      self.donor_names, **kwargs)


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
