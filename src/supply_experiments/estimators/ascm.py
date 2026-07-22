"""Augmented SCM com ridge (Ben-Michael, Feller & Rothstein 2021) — implementação fiel.

τ̂_t = Y_tr,t(post) − [ Σ_i w_i^scm Y_i,t + (X_tr − Σ_i w_i^scm X_i)' β̂_t ]

onde X são os outcomes pré-tratamento e β̂_t vem de uma regressão ridge dos
outcomes pós dos doadores sobre seus outcomes pré. O termo de correção remove
o viés de desbalanceamento residual do SCM, que o SCM puro ignora. Uma
simplificação comum e incorreta é "corrigir" isso somando a média do resíduo
pré ao contrafactual — o que zera o gap pré por construção e mascara o fit real.

λ é escolhido por validação cruzada leave-one-out nos doadores.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from supply_experiments.estimators.scm import SCMFit, _failed, _package, _simplex_ols


def _ridge_beta(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    """Ridge com intercepto (intercepto não penalizado). Retorna (b0, beta)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    p = X.shape[1]
    A = Xc.T @ Xc + lam * np.eye(p)
    beta = np.linalg.solve(A, Xc.T @ yc)
    b0 = float(y.mean() - X.mean(axis=0) @ beta)
    return b0, beta


def _loo_cv_lambda(X: np.ndarray, Y_post: np.ndarray, grid: Sequence[float]) -> float:
    """Escolhe λ minimizando erro LOO médio ao prever outcomes pós dos doadores."""
    J = X.shape[0]
    best_lam, best_err = grid[0], np.inf
    for lam in grid:
        errs = []
        for j in range(J):
            mask = np.arange(J) != j
            for t in range(Y_post.shape[1]):
                b0, beta = _ridge_beta(X[mask], Y_post[mask, t], lam)
                pred = b0 + X[j] @ beta
                errs.append((Y_post[j, t] - pred) ** 2)
        err = float(np.mean(errs))
        if err < best_err:
            best_err, best_lam = err, lam
    return best_lam


def fit_ascm(
    y_pre: np.ndarray,
    Y_donors_pre: np.ndarray,
    y_post: np.ndarray,
    Y_donors_post: np.ndarray,
    donor_names: List[str],
    lam: Optional[float] = None,
    lam_grid: Optional[Sequence[float]] = None,
) -> SCMFit:
    y_pre = np.asarray(y_pre, float)
    y_post = np.asarray(y_post, float)
    T_pre, J = Y_donors_pre.shape
    if J < 3:
        return _failed(y_pre, y_post, donor_names, "ascm")

    # 1) SCM base
    w, _, ok = _simplex_ols(y_pre, Y_donors_pre)
    if not np.isfinite(w).all():
        return _failed(y_pre, y_post, donor_names, "ascm")

    # 2) Ridge: outcomes pós dos doadores ~ outcomes pré dos doadores
    X = Y_donors_pre.T                      # (J, T_pre): features = trajetória pré
    Yp = Y_donors_post.T                    # (J, T_post)
    scale = float(np.std(X)) or 1.0
    Xs = X / scale
    if lam is None:
        grid = list(lam_grid) if lam_grid is not None else [0.1, 1.0, 10.0, 100.0, 1000.0]
        # CV em subamostra de períodos pós p/ custo controlado
        t_idx = np.linspace(0, Yp.shape[1] - 1, num=min(5, Yp.shape[1]), dtype=int)
        lam = _loo_cv_lambda(Xs, Yp[:, t_idx], grid)

    # 3) Correção de viés por período pós
    imbalance = (y_pre - Y_donors_pre @ w) / scale        # X_tr − X_co'w, escalado
    synth_post = np.empty_like(y_post)
    for t in range(len(y_post)):
        b0, beta = _ridge_beta(Xs, Yp[:, t], lam)
        synth_post[t] = Y_donors_post[t] @ w + imbalance @ beta

    # No pré, o ASCM balanceia por construção via ridge; reportamos o synthetic
    # SCM puro no pré (transparência do fit) e a correção só no pós.
    synth_pre = Y_donors_pre @ w
    fit = _package(y_pre, y_post, synth_pre, synth_post, w, donor_names, "ascm",
                   extras={"lambda": float(lam)})
    return fit
