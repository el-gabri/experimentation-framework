"""Seleção do controle fixo com holdout temporal — sem pre-testing contamination.

Mudanças vs. versão anterior:
  1. Janela dividida em TREINO (primeiros ~70%) e HOLDOUT (últimos ~30%).
     Otimização (greedy + swaps) roda SÓ no treino; tendências paralelas são
     avaliadas SÓ no holdout, como critério de aceite — nunca de otimização.
  2. O p-value NÃO entra no score de seleção (otimizar p-value = garimpagem).
     Score = correlação + forma (RMSE normalizado) + região + ruptura, no treino.
  3. O target exclui as cidades do próprio controle (evita correlação mecânica)
     e aceita lista de cidades a excluir do benchmark (tratadas ativas).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from supply_experiments.panel import CityPanel, make_dow_dummies, make_holiday_dummy


@dataclass
class ControlSelectionResult:
    cities: List[str]
    train_score: float
    holdout_p_value: float
    holdout_passed: bool
    holdout_corr: float
    metrics: dict = field(default_factory=dict)


def f_test_parallel_trends(
    dates: pd.DatetimeIndex,
    y_target: np.ndarray,
    y_control: np.ndarray,
    holidays: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Teste F de interação tempo×grupo em séries normalizadas pela média."""
    from scipy.stats import f as f_dist

    ya = np.asarray(y_target, float)
    yb = np.asarray(y_control, float)
    T = len(ya)
    ya = ya / (np.mean(ya) or 1.0)
    yb = yb / (np.mean(yb) or 1.0)

    y = np.concatenate([ya, yb])
    t = np.arange(T, dtype=float)
    t = (t - t.mean()) / max(t.std(), 1.0)
    group = np.concatenate([np.zeros(T), np.ones(T)])
    tt = np.concatenate([t, t])
    inter = group * tt

    dow = make_dow_dummies(dates)
    dow2 = np.vstack([dow, dow])
    blocks = [np.ones(2 * T), group, tt, inter] + [dow2[:, k] for k in range(dow2.shape[1])]
    if holidays:
        h = make_holiday_dummy(dates, holidays)
        blocks.append(np.concatenate([h, h]))
    X_u = np.column_stack(blocks)
    X_r = np.delete(X_u, 3, axis=1)

    def sse(X):
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        return float(r @ r), b

    sse_u, beta_u = sse(X_u)
    sse_r, _ = sse(X_r)
    df2 = max(2 * T - X_u.shape[1], 1)
    F = max(sse_r - sse_u, 0.0) / (sse_u / df2)
    p = float(f_dist.sf(F, 1, df2))
    return {"F": float(F), "p_value": p, "coef_interaction": float(beta_u[3])}


def _subset_score(y_target: np.ndarray, y_sub: np.ndarray,
                  states: Sequence[str], state_to_region: Dict[str, str],
                  rupture_target: float, rupture_sub: float,
                  weights: Dict[str, float]) -> float:
    a = y_target / (np.mean(y_target) or 1.0)
    b = y_sub / (np.mean(y_sub) or 1.0)
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(b) > 1e-12 else 0.0
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    shape_score = 1.0 / (1.0 + 10.0 * rmse)
    regions = {state_to_region.get((s or "")[:2].upper()) for s in states}
    regions.discard(None)
    reg_score = len(regions) / 5.0
    if np.isfinite(rupture_target) and np.isfinite(rupture_sub):
        rup_score = 1.0 / (1.0 + 10.0 * abs(rupture_target - rupture_sub))
    else:
        rup_score = 0.5
    return (weights.get("correlation", 0.45) * max(corr, 0.0)
            + weights.get("shape", 0.30) * shape_score
            + weights.get("regional", 0.10) * reg_score
            + weights.get("rupture", 0.15) * rup_score)


def select_fixed_control(
    panel: CityPanel,
    candidates: Sequence[str],
    city_states: Dict[str, str],
    state_to_region: Dict[str, str],
    max_cities: int = 9,
    train_frac: float = 0.7,
    alpha: float = 0.05,
    holidays: Optional[Sequence[str]] = None,
    city_rupture: Optional[Dict[str, float]] = None,
    target_exclude: Optional[Set[str]] = None,
    score_weights: Optional[Dict[str, float]] = None,
    max_swap_passes: int = 3,
    seed: int = 7,
) -> ControlSelectionResult:
    rng = np.random.default_rng(seed)
    weights = score_weights or {}
    city_rupture = city_rupture or {}
    target_exclude = set(target_exclude or set())

    T = len(panel.index)
    t_split = int(T * train_frac)
    train_idx, hold_idx = slice(0, t_split), slice(t_split, T)
    dates_hold = panel.index[hold_idx]

    all_cities = [c for c in panel.cities]
    candidates = [c for c in candidates if c in all_cities]

    def target_series(exclude: Sequence[str]) -> np.ndarray:
        keep = [c for c in all_cities if c not in set(exclude) | target_exclude]
        return panel.outcome[keep].sum(axis=1).to_numpy(float)

    vals_t = [city_rupture[c] for c in all_cities if c in city_rupture]
    rupture_target = float(np.mean(vals_t)) if vals_t else np.nan

    def evaluate(subset: List[str], idx_slice) -> float:
        tgt = target_series(subset)[idx_slice]
        sub = panel.outcome[subset].sum(axis=1).to_numpy(float)[idx_slice]
        states = [city_states.get(c, "") for c in subset]
        vals = [city_rupture[c] for c in subset if c in city_rupture]
        rup = float(np.mean(vals)) if vals else np.nan
        return _subset_score(tgt, sub, states, state_to_region, rupture_target, rup, weights)

    # ---- greedy forward no TREINO ----
    selected: List[str] = []
    best = -np.inf
    for _ in range(max_cities):
        pick, pick_score = None, best
        for c in candidates:
            if c in selected:
                continue
            s = evaluate(selected + [c], train_idx)
            if s > pick_score:
                pick, pick_score = c, s
        if pick is None:
            break
        selected.append(pick)
        best = pick_score

    # ---- busca local (swaps) no TREINO ----
    for _ in range(max_swap_passes):
        improved = False
        for i in range(len(selected)):
            for c in candidates:
                if c in selected:
                    continue
                trial = selected.copy()
                trial[i] = c
                s = evaluate(trial, train_idx)
                if s > best + 1e-9:
                    selected, best, improved = trial, s, True
                    break
            if improved:
                break
        if not improved:
            break

    # ---- ACEITE no HOLDOUT (nunca otimizado) ----
    tgt_h = target_series(selected)[hold_idx]
    sub_h = panel.outcome[selected].sum(axis=1).to_numpy(float)[hold_idx]
    tr = f_test_parallel_trends(dates_hold, tgt_h, sub_h, holidays)
    a = tgt_h / (np.mean(tgt_h) or 1.0)
    b = sub_h / (np.mean(sub_h) or 1.0)
    corr_h = float(np.corrcoef(a, b)[0, 1]) if np.std(b) > 1e-12 else 0.0
    passed = bool(tr["p_value"] >= alpha)

    return ControlSelectionResult(
        cities=selected, train_score=float(best),
        holdout_p_value=float(tr["p_value"]), holdout_passed=passed,
        holdout_corr=corr_h,
        metrics={"holdout_F": tr["F"], "holdout_coef": tr["coef_interaction"],
                 "train_frac": train_frac, "alpha": alpha},
    )
