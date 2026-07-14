"""Inferência conformal para SC/ASCM/SDID (Chernozhukov, Wüthrich & Zhu, JASA 2021).

Teste de H0: τ_t = τ0 ∀t no pós:
  1. Impõe H0 subtraindo τ0 dos outcomes pós da tratada.
  2. Reajusta o estimador na amostra completa (pré + pós ajustado).
  3. Estatística S = norma dos resíduos nos períodos pós.
  4. p-value por permutações em bloco (moving block) dos resíduos — válido sob
     estacionariedade fraca, sem hipótese de ausência de autocorrelação.

O IC de (1−α) é o conjunto de τ0 não rejeitados (busca em grade + refinamento).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class ConformalResult:
    p_value: float                       # H0: efeito relativo = 0
    ci_lower_pct: float                  # IC (1−α) para efeito relativo (fração)
    ci_upper_pct: float
    alpha: float
    grid_evaluated: int


def _pvalue_for_tau(fit_fn, y_pre, Yd_pre, y_post, Yd_post, names,
                    tau0_abs_per_t: np.ndarray, fit_kwargs) -> float:
    """p-value conformal para H0: efeito_t = tau0_abs_per_t."""
    T_pre, T_post = len(y_pre), len(y_post)
    y_post_h0 = y_post - tau0_abs_per_t

    # Reajusta com o pós "neutralizado" incorporado ao período de ajuste.
    y_all = np.concatenate([y_pre, y_post_h0])
    Yd_all = np.vstack([Yd_pre, Yd_post])
    fit = fit_fn(y_all, Yd_all, y_post_h0[:0] if T_post == 0 else y_post_h0,
                 Yd_post, names, **fit_kwargs)
    if not fit.success:
        return np.nan
    resid_all = y_all - np.concatenate([fit.y_synth_pre])  # synth_pre cobre y_all aqui
    T = len(resid_all)

    def stat(u: np.ndarray) -> float:
        return float(np.sqrt(np.mean(u[-T_post:] ** 2)))

    s_obs = stat(resid_all)
    # moving-block permutations: todas as rotações cíclicas (CWZ §3)
    count, total = 0, 0
    for shift in range(T):
        u = np.roll(resid_all, shift)
        if stat(u) >= s_obs - 1e-15:
            count += 1
        total += 1
    return count / total


def conformal_inference(
    fit_fn: Callable,
    y_pre: np.ndarray,
    Y_donors_pre: np.ndarray,
    y_post: np.ndarray,
    Y_donors_post: np.ndarray,
    donor_names: List[str],
    alpha: float = 0.05,
    fit_kwargs: Optional[dict] = None,
    rel_grid: Optional[np.ndarray] = None,
    counterfactual_level: Optional[float] = None,
) -> ConformalResult:
    fit_kwargs = fit_kwargs or {}
    y_pre = np.asarray(y_pre, float)
    y_post = np.asarray(y_post, float)

    base = counterfactual_level
    if base is None:
        f0 = fit_fn(y_pre, Y_donors_pre, y_post, Y_donors_post, donor_names, **fit_kwargs)
        base = float(np.mean(f0.y_synth_post)) if f0.success else float(np.mean(y_pre))
    if abs(base) < 1e-9:
        base = float(np.mean(y_pre)) or 1.0

    # p-value do nulo (efeito = 0)
    p0 = _pvalue_for_tau(fit_fn, y_pre, Y_donors_pre, y_post, Y_donors_post,
                         donor_names, np.zeros_like(y_post), fit_kwargs)

    # IC por inversão do teste em grade de efeitos relativos
    if rel_grid is None:
        rel_grid = np.linspace(-0.5, 0.5, 51)
    accepted = []
    for r in rel_grid:
        tau_abs = np.full_like(y_post, r * base)
        p = _pvalue_for_tau(fit_fn, y_pre, Y_donors_pre, y_post, Y_donors_post,
                            donor_names, tau_abs, fit_kwargs)
        if np.isfinite(p) and p > alpha:
            accepted.append(r)

    if accepted:
        lo, hi = float(min(accepted)), float(max(accepted))
        # refina bordas com meia malha
        step = float(rel_grid[1] - rel_grid[0]) if len(rel_grid) > 1 else 0.02
        for r in np.arange(lo - step, lo, step / 4):
            tau_abs = np.full_like(y_post, r * base)
            if _pvalue_for_tau(fit_fn, y_pre, Y_donors_pre, y_post, Y_donors_post,
                               donor_names, tau_abs, fit_kwargs) > alpha:
                lo = float(r)
                break
        for r in np.arange(hi + step, hi, -step / 4):
            tau_abs = np.full_like(y_post, r * base)
            if _pvalue_for_tau(fit_fn, y_pre, Y_donors_pre, y_post, Y_donors_post,
                               donor_names, tau_abs, fit_kwargs) > alpha:
                hi = float(r)
                break
    else:
        lo, hi = np.nan, np.nan

    return ConformalResult(p_value=float(p0), ci_lower_pct=lo, ci_upper_pct=hi,
                           alpha=alpha, grid_evaluated=len(rel_grid))
