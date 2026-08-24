"""Synthetic Control clássico (Abadie, Diamond & Hainmueller 2010)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SCMFit:
    weights: Dict[str, float]
    donor_names: List[str]
    w: np.ndarray                       # vetor completo alinhado a donor_names
    y_synth_pre: np.ndarray
    y_synth_post: np.ndarray
    pre_rmspe: float                    # relativo à média da tratada no pré
    pre_correlation: float
    att: float                          # efeito médio no pós (nível)
    att_pct: float                      # efeito relativo ao contrafactual
    effects_post: np.ndarray            # efeito por dia no pós
    success: bool
    method: str = "scm"
    extras: dict = field(default_factory=dict)


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Projeção euclidiana exata no simplex (Duchi et al. 2008)."""
    J = len(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    idx = np.nonzero(u * np.arange(1, J + 1) > (css - 1.0))[0]
    rho = idx[-1] if len(idx) else 0
    theta = (css[rho] - 1.0) / (rho + 1)
    return np.maximum(v - theta, 0.0)


def _simplex_ols(y: np.ndarray, X: np.ndarray, ridge: float = 0.0,
                 intercept: bool = False, max_iter: int = 5000,
                 tol: float = 1e-10) -> tuple[np.ndarray, float, bool, dict]:
    """min ||y - (a + X w)||^2 + ridge*||w||^2  s.t. w>=0, sum(w)=1.

    Resolvido por active-set para o QP no simplex. A convergência usa o gap de
    Frank-Wolfe/KKT em unidades do objetivo; um passo projetado dividido pelo
    Lipschitz não é certificado válido quando doadoras têm escalas díspares.
    Com intercepto livre, a* = mean(y - Xw): resolve-se nos dados centrados.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if y.ndim != 1 or X.ndim != 2 or X.shape[0] != len(y) or X.shape[1] == 0:
        return np.zeros(X.shape[1] if X.ndim == 2 else 0), 0.0, False, {
            "solver_status": "invalid_shape", "solver_iterations": 0,
            "solver_objective": np.nan, "solver_projected_gradient": np.nan,
        }
    if not np.isfinite(y).all() or not np.isfinite(X).all() or ridge < 0:
        return np.full(X.shape[1], 1.0 / X.shape[1]), 0.0, False, {
            "solver_status": "invalid_input", "solver_iterations": 0,
            "solver_objective": np.nan, "solver_projected_gradient": np.nan,
        }
    if max_iter < 1 or tol <= 0:
        return np.full(X.shape[1], 1.0 / X.shape[1]), 0.0, False, {
            "solver_status": "invalid_solver_options", "solver_iterations": 0,
            "solver_objective": np.nan, "solver_projected_gradient": np.nan,
        }

    _, J = X.shape
    scale = float(np.std(y)) or 1.0
    ys, Xs = y / scale, X / scale
    if intercept:
        ys = ys - ys.mean()
        Xs = Xs - Xs.mean(axis=0, keepdims=True)

    L = float(np.linalg.norm(Xs, 2) ** 2) * 2.0 + 2.0 * ridge
    if L <= 0:
        return np.full(J, 1.0 / J), 0.0, False, {
            "solver_status": "degenerate_objective", "solver_iterations": 0,
            "solver_objective": np.nan, "solver_projected_gradient": np.nan,
        }

    gram = Xs.T @ Xs + ridge * np.eye(J)
    rhs = Xs.T @ ys
    w = np.full(J, 1.0 / J)
    active = list(range(J))
    converged = False
    obj = np.inf
    projected_gradient = np.inf
    fw_gap = np.inf
    gap_tolerance = np.inf
    iterations = 0
    for iteration in range(1, max_iter + 1):
        idx = np.asarray(active, dtype=int)
        kkt = np.block([
            [2.0 * gram[np.ix_(idx, idx)], np.ones((len(idx), 1))],
            [np.ones((1, len(idx))), np.zeros((1, 1))],
        ])
        target = np.concatenate([2.0 * rhs[idx], np.ones(1)])
        solution, *_ = np.linalg.lstsq(kkt, target, rcond=None)
        candidate = np.zeros(J, dtype=float)
        candidate[idx] = solution[:-1]

        if np.any(candidate[idx] < -1e-12):
            direction = candidate - w
            decreasing = idx[direction[idx] < -1e-15]
            if len(decreasing) == 0:
                iterations = iteration
                break
            steps = w[decreasing] / (w[decreasing] - candidate[decreasing])
            step = float(np.clip(np.min(steps), 0.0, 1.0))
            w = w + step * direction
            w[np.abs(w) <= 1e-12] = 0.0
            active = [j for j in active if w[j] > 0.0]
            if not active:
                iterations = iteration
                break
            iterations = iteration
            continue

        w = np.maximum(candidate, 0.0)
        total = float(w.sum())
        if total <= 0.0:
            iterations = iteration
            break
        w /= total
        r = ys - Xs @ w
        obj = float(r @ r + ridge * (w @ w))
        g_w = -2.0 * Xs.T @ r + 2.0 * ridge * w
        projected_gradient = float(
            np.max(np.abs(w - _project_simplex(w - g_w / L)))
        )
        fw_gap = max(0.0, float(g_w @ w - np.min(g_w)))
        gap_tolerance = float(tol + 1e-8 * max(1.0, abs(obj)))
        feasible = abs(float(w.sum()) - 1.0) <= 1e-10 and float(w.min()) >= -1e-12
        if feasible and fw_gap <= gap_tolerance:
            converged = True
            iterations = iteration
            break
        entering = int(np.argmin(g_w))
        if entering not in active:
            active.append(entering)
        else:
            # The equality-constrained solve should have zero gap when every
            # violating coordinate is already active. If numerical rank loss
            # prevents that certificate, fail closed instead of looping.
            iterations = iteration
            break
        iterations = iteration

    a = 0.0
    if intercept:
        a = float(np.mean(y - X @ w))
    diagnostics = {
        "solver_status": "converged" if converged else "max_iter_reached",
        "solver_iterations": iterations,
        "solver_objective": float(obj),
        "solver_projected_gradient": projected_gradient,
        "solver_fw_gap": fw_gap,
        "solver_gap_tolerance": gap_tolerance,
        "solver_active_donors": len(active),
    }
    return w, a, converged, diagnostics


def fit_scm(
    y_pre: np.ndarray,
    Y_donors_pre: np.ndarray,
    y_post: np.ndarray,
    Y_donors_post: np.ndarray,
    donor_names: List[str],
    min_weight: float = 0.0,
) -> SCMFit:
    """
    Ajusta SCM no pré e projeta no pós.

    Nota: os pesos NÃO são truncados+renormalizados por padrão (min_weight=0):
    truncar altera o fit que foi otimizado. Se truncar, o synthetic é
    recomputado com os pesos finais.
    """
    y_pre = np.asarray(y_pre, float)
    y_post = np.asarray(y_post, float)
    Y_donors_pre = np.asarray(Y_donors_pre, float)
    Y_donors_post = np.asarray(Y_donors_post, float)
    if not np.isfinite(min_weight) or min_weight < 0:
        raise ValueError("min_weight deve ser finito e >= 0")
    if (
        y_pre.ndim != 1 or y_post.ndim != 1
        or Y_donors_pre.ndim != 2 or Y_donors_post.ndim != 2
    ):
        return _failed(y_pre, y_post, donor_names, "scm")
    T_pre, J = Y_donors_pre.shape
    if (
        J < 2 or T_pre < 2 or len(y_pre) != T_pre
        or Y_donors_post.shape != (len(y_post), J)
        or len(donor_names) != J or len(set(donor_names)) != J
        or any(
            not values.all()
            for values in (
                np.isfinite(y_pre), np.isfinite(y_post),
                np.isfinite(Y_donors_pre), np.isfinite(Y_donors_post),
            )
        )
    ):
        return _failed(y_pre, y_post, donor_names, "scm")

    w, _, ok, solver = _simplex_ols(y_pre, Y_donors_pre)
    if not ok or not np.isfinite(w).all():
        return _failed(y_pre, y_post, donor_names, "scm", extras=solver)

    if min_weight > 0:
        w = np.where(w >= min_weight, w, 0.0)
        s = w.sum()
        if s <= 0:
            return _failed(
                y_pre, y_post, donor_names, "scm",
                extras={**solver, "solver_status": "all_weights_removed"},
            )
        w = w / s

    synth_pre = Y_donors_pre @ w
    synth_post = Y_donors_post @ w
    return _package(
        y_pre, y_post, synth_pre, synth_post, w, donor_names, "scm",
        extras=solver,
    )


def _package(y_pre, y_post, synth_pre, synth_post, w, donor_names, method,
             extras: Optional[dict] = None) -> SCMFit:
    if (
        np.shape(synth_pre) != np.shape(y_pre)
        or np.shape(synth_post) != np.shape(y_post)
        or not np.isfinite(synth_pre).all()
        or not np.isfinite(synth_post).all()
    ):
        return _failed(y_pre, y_post, donor_names, method, extras=extras)
    mean_pre = float(np.mean(y_pre))
    if abs(mean_pre) < 1e-9:
        # painel degenerado (tratada ~0 no pré): RMSPE relativo não é
        # informativo aqui — falha explicitamente em vez de dividir por 1.0
        # silenciosamente e mascarar o problema.
        return _failed(y_pre, y_post, donor_names, method, extras=extras)
    rmspe = float(np.sqrt(np.mean((y_pre - synth_pre) ** 2))) / abs(mean_pre)
    if np.std(y_pre) > 1e-12 and np.std(synth_pre) > 1e-12:
        corr = float(np.corrcoef(y_pre, synth_pre)[0, 1])
    else:
        corr = 0.0
    effects = y_post - synth_post
    att = float(np.mean(effects)) if len(effects) else np.nan
    denom = float(np.mean(synth_post)) if len(synth_post) else np.nan
    att_pct = att / denom if denom and abs(denom) > 1e-12 else np.nan
    return SCMFit(
        weights={n: float(x) for n, x in zip(donor_names, w, strict=True) if x > 1e-6},
        donor_names=list(donor_names), w=np.asarray(w, float),
        y_synth_pre=synth_pre, y_synth_post=synth_post,
        pre_rmspe=rmspe, pre_correlation=corr,
        att=att, att_pct=att_pct, effects_post=effects,
        success=True, method=method, extras=extras or {},
    )


def _failed(y_pre, y_post, donor_names, method, extras: Optional[dict] = None) -> SCMFit:
    return SCMFit(weights={}, donor_names=list(donor_names),
                  w=np.zeros(len(donor_names)),
                  y_synth_pre=np.zeros_like(y_pre), y_synth_post=np.zeros_like(y_post),
                  pre_rmspe=np.inf, pre_correlation=0.0, att=np.nan, att_pct=np.nan,
                  effects_post=np.full_like(np.asarray(y_post, float), np.nan),
                  success=False, method=method, extras=extras or {})
