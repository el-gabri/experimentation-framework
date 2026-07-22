"""Synthetic Difference-in-Differences (Arkhangelsky et al., AER 2021) — fiel ao paper.

Pesos unitários: min_{w0, w∈simplex} Σ_pre (w0 + Y_co,t'w − ȳ_tr,t)² + ζ² T_pre ||w||²
  com ζ = (N_tr · T_post)^{1/4} · σ̂,  σ̂ = dp das primeiras diferenças dos controles no pré.
Pesos temporais: min_{λ0, λ∈simplex} Σ_i∈co (λ0 + Y_i,pre'λ − ȳ_i,post)²  (+ ridge ínfimo p/ unicidade)
Estimador: τ̂ = (ȳ_tr,post − Σ_t λ_t ȳ_tr,t) − Σ_i w_i (ȳ_i,post − Σ_t λ_t Y_i,t)

Implementações simplificadas costumam usar pesos temporais lineares arbitrários
e sem regularização ζ — perdendo exatamente a propriedade de duplo-robustez
local que justifica o SDID.
"""

from __future__ import annotations

from typing import List

import numpy as np

from supply_experiments.estimators.scm import SCMFit, _failed


def _solve_simplex_intercept(A: np.ndarray, b: np.ndarray, ridge: float) -> np.ndarray:
    """min_{v0, v∈simplex} ||b − v0·1 − A v||² + ridge·||v||²  via Frank-Wolfe projetado.

    A: (n, k) — colunas são as variáveis com peso no simplex.
    Retorna v (k,). Frank-Wolfe com line search exato converge bem no simplex.
    """
    n, k = A.shape
    v = np.full(k, 1.0 / k)

    def resid(v):
        # intercepto ótimo dado v: média do resíduo
        r = b - A @ v
        return r - r.mean()

    for _ in range(2000):
        r = resid(v)
        grad = -2.0 * (A - A.mean(axis=0, keepdims=True)).T @ r + 2.0 * ridge * v
        s = np.zeros(k)
        s[int(np.argmin(grad))] = 1.0
        d = s - v
        gd = float(grad @ d)
        if gd > -1e-12:
            break
        # line search exato para objetivo quadrático
        Ad = (A - A.mean(axis=0, keepdims=True)) @ d
        denom = float(Ad @ Ad + ridge * (d @ d))
        step = min(1.0, max(0.0, -gd / (2.0 * denom))) if denom > 0 else 0.0
        if step <= 1e-14:
            break
        v = v + step * d
    return v


def fit_sdid(
    y_pre: np.ndarray,
    Y_donors_pre: np.ndarray,
    y_post: np.ndarray,
    Y_donors_post: np.ndarray,
    donor_names: List[str],
    n_treated_units: int = 1,
) -> SCMFit:
    y_pre = np.asarray(y_pre, float)
    y_post = np.asarray(y_post, float)
    T_pre, J = Y_donors_pre.shape
    T_post = len(y_post)
    if J < 2 or T_pre < 5 or T_post < 1:
        return _failed(y_pre, y_post, donor_names, "sdid")

    # ---- ζ de Arkhangelsky et al. ----
    diffs = np.diff(Y_donors_pre, axis=0)              # (T_pre-1, J)
    sigma = float(np.std(diffs, ddof=1)) if diffs.size else 1.0
    zeta = (n_treated_units * T_post) ** 0.25 * sigma
    ridge_w = (zeta ** 2) * T_pre

    # ---- pesos unitários ω ----
    w = _solve_simplex_intercept(Y_donors_pre, y_pre, ridge=ridge_w)

    # ---- pesos temporais λ ----
    ybar_post_donors = Y_donors_post.mean(axis=0)      # (J,)
    lam = _solve_simplex_intercept(Y_donors_pre.T, ybar_post_donors,
                                   ridge=1e-6 * float(np.var(Y_donors_pre)) * J)

    # ---- estimador DiD ponderado ----
    treated_post = float(np.mean(y_post))
    treated_pre_l = float(lam @ y_pre)
    donors_post_w = float(w @ Y_donors_post.mean(axis=0))
    donors_pre_wl = float(w @ (Y_donors_pre.T @ lam))
    att = (treated_post - treated_pre_l) - (donors_post_w - donors_pre_wl)

    # trajetória contrafactual implícita p/ gráficos e efeitos diários:
    # ŷ_t(0) = Y_co,t' w + (ȳ_tr,pre_λ − Y_co,pre_λ' w)   (nível ajustado por DiD)
    level_adj = treated_pre_l - donors_pre_wl
    synth_pre = Y_donors_pre @ w + level_adj
    synth_post = Y_donors_post @ w + level_adj
    effects = y_post - synth_post

    mean_pre = float(np.mean(y_pre)) or 1.0
    rmspe = float(np.sqrt(np.mean((y_pre - synth_pre) ** 2))) / abs(mean_pre)
    corr = float(np.corrcoef(y_pre, synth_pre)[0, 1]) if np.std(synth_pre) > 1e-12 else 0.0
    denom = float(np.mean(synth_post))
    return SCMFit(
        weights={n: float(x) for n, x in zip(donor_names, w) if x > 1e-6},
        donor_names=list(donor_names), w=w,
        y_synth_pre=synth_pre, y_synth_post=synth_post,
        pre_rmspe=rmspe, pre_correlation=corr,
        att=float(att), att_pct=float(att) / denom if abs(denom) > 1e-12 else np.nan,
        effects_post=effects, success=True, method="sdid",
        extras={"zeta": zeta, "time_weights": lam.tolist()},
    )
