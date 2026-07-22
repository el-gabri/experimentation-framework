"""Análise de poder por simulação em dados históricos.

Procedimento: para o design candidato (tratadas, doadoras, método, duração),
sorteia janelas placebo no histórico, injeta efeito multiplicativo δ nas
tratadas no "pós" sintético, roda estimador+inferência e mede a taxa de
rejeição. δ=0 dá a calibração (FPR); a curva δ→poder dá o MDE.

Nenhum experimento deve ser lançado sem MDE ≤ efeito esperado da intervenção.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from supply_experiments.inference.permutation import placebo_inference
from supply_experiments.panel import CityPanel


@dataclass
class PowerResult:
    effect_grid: List[float]
    power: Dict[float, float]           # δ -> taxa de rejeição
    fpr: float                          # poder em δ=0 (deve ≈ α)
    mde_80: Optional[float]             # menor |δ| com poder >= 0.80
    n_sims_per_point: int
    alpha: float
    details: pd.DataFrame = field(default_factory=pd.DataFrame)


def _draw_windows(index: pd.DatetimeIndex, pre_days: int, post_days: int,
                  n: int, rng: np.random.Generator, buffer_end_days: int = 0) -> List[int]:
    """Sorteia posições de início do 'pós' placebo, espaçadas + jitter.

    Amostragem estratificada: divide [lo, hi) em n faixas quase iguais e
    sorteia uma posição por faixa. Isso evita que as janelas 'pós' placebo se
    sobreponham fortemente — sobreposição correlaciona as simulações e
    subestima a variância das estimativas de poder/FPR.
    """
    T = len(index)
    lo = pre_days
    hi = T - post_days - buffer_end_days
    if hi <= lo:
        raise ValueError(f"Histórico insuficiente: precisa de >= {pre_days + post_days} dias")
    n_candidates = hi - lo
    if n_candidates <= n:
        return list(range(lo, hi))

    edges = np.linspace(lo, hi, n + 1)
    picks = []
    for i in range(n):
        a, b = int(np.ceil(edges[i])), int(np.floor(edges[i + 1]))
        b = max(b, a + 1)
        picks.append(int(rng.integers(a, min(b, hi))))
    return sorted(set(picks)) or [lo]


def simulate_once(
    panel: CityPanel,
    treated: Sequence[str],
    donors: Sequence[str],
    fit_fn: Callable,
    start_idx: int,
    pre_days: int,
    post_days: int,
    delta: float,
    alpha: float,
    fit_kwargs: Optional[dict] = None,
    seasonal_effect: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """Um experimento placebo com efeito injetado δ (multiplicativo)."""
    donors = [d for d in donors if d not in set(treated)]
    y = panel.aggregate(treated).to_numpy(float)
    Yd = panel.outcome[list(donors)].to_numpy(float)

    pre_sl = slice(start_idx - pre_days, start_idx)
    post_sl = slice(start_idx, start_idx + post_days)

    y_pre = y[pre_sl].copy()
    y_post = y[post_sl].copy()
    if delta != 0.0:
        mult: "float | np.ndarray" = 1.0 + delta
        if seasonal_effect and rng is not None:
            # efeito com ruído dia-a-dia (mais realista que degrau perfeito)
            mult = 1.0 + delta * (1.0 + 0.3 * rng.standard_normal(len(y_post)))
        y_post = y_post * mult

    inf = placebo_inference(
        fit_fn, y_pre, Yd[pre_sl], y_post, Yd[post_sl], list(donors),
        fit_kwargs=fit_kwargs or {},
    )
    return {
        "delta": delta, "start_idx": start_idx,
        "p_value": inf.p_value, "reject": bool(np.isfinite(inf.p_value) and inf.p_value <= alpha),
        "att_pct_placebo_med": float(np.nanmedian(inf.placebo_atts)) if inf.placebo_atts else np.nan,
        "n_placebos": inf.n_placebos,
    }


def power_analysis(
    panel: CityPanel,
    treated: Sequence[str],
    donors: Sequence[str],
    fit_fn: Callable,
    pre_days: int,
    post_days: int,
    effect_grid: Sequence[float] = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10),
    n_sims_per_point: int = 30,
    alpha: float = 0.10,
    seed: int = 42,
    fit_kwargs: Optional[dict] = None,
) -> PowerResult:
    """
    alpha default 0.10: com poucos placebos a granularidade mínima do p-value é
    1/(J+1); com J=15 doadoras, p mínimo = 0.0625 — α=0.05 seria inatingível.
    O design deve garantir doadoras suficientes para o α desejado (ver relatório).
    """
    rng = np.random.default_rng(seed)
    starts = _draw_windows(panel.index, pre_days, post_days, n_sims_per_point, rng)

    rows = []
    for delta in effect_grid:
        for s in starts:
            rows.append(simulate_once(panel, treated, donors, fit_fn, s,
                                      pre_days, post_days, float(delta), alpha,
                                      fit_kwargs, seasonal_effect=True, rng=rng))
    df = pd.DataFrame(rows)
    power = df.groupby("delta")["reject"].mean().to_dict()
    fpr = float(power.get(0.0, np.nan))

    mde = None
    for d in sorted(x for x in power if x != 0.0):
        if power[d] >= 0.80:
            mde = float(d)
            break

    return PowerResult(effect_grid=list(effect_grid), power={float(k): float(v) for k, v in power.items()},
                       fpr=fpr, mde_80=mde, n_sims_per_point=len(starts), alpha=alpha, details=df)


def minimum_detectable_effect(power_result: PowerResult) -> Optional[float]:
    return power_result.mde_80
