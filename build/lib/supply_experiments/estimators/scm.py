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
                 tol: float = 1e-10) -> tuple[np.ndarray, float, bool]:
    """min ||y - (a + X w)||^2 + ridge*||w||^2  s.t. w>=0, sum(w)=1.

    Resolvido via FISTA com projeção exata no simplex. (SLSQP foi abandonado:
    declara convergência com objetivo ~20x acima do ótimo em instâncias com
    doadoras de escalas muito díspares.)
    Com intercepto livre, a* = mean(y - Xw): resolve-se nos dados centrados.
    """
    T, J = X.shape
    scale = float(np.std(y)) or 1.0
    ys, Xs = y / scale, X / scale
    if intercept:
        ys = ys - ys.mean()
        Xs = Xs - Xs.mean(axis=0, keepdims=True)

    L = float(np.linalg.norm(Xs, 2) ** 2) * 2.0 + 2.0 * ridge
    if L <= 0:
        return np.full(J, 1.0 / J), 0.0, False

    w = np.full(J, 1.0 / J)
    z = w.copy()
    t_k = 1.0
    obj_prev = np.inf
    for _ in range(max_iter):
        g = -2.0 * Xs.T @ (ys - Xs @ z) + 2.0 * ridge * z
        w_new = _project_simplex(z - g / L)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
        z = w_new + ((t_k - 1.0) / t_new) * (w_new - w)
        w, t_k = w_new, t_new
        r = ys - Xs @ w
        obj = float(r @ r + ridge * (w @ w))
        if abs(obj_prev - obj) < tol * max(obj, 1.0):
            break
        obj_prev = obj

    a = 0.0
    if intercept:
        a = float(np.mean(y - X @ w))
    return w, a, True


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

    Nota: diferentemente da versão anterior do framework, os pesos NÃO são
    truncados+renormalizados por padrão (min_weight=0): truncar altera o fit
    que foi otimizado. Se truncar, o synthetic é recomputado com os pesos finais.
    """
    y_pre = np.asarray(y_pre, float)
    y_post = np.asarray(y_post, float)
    J = Y_donors_pre.shape[1]
    if J < 2:
        return _failed(y_pre, y_post, donor_names, "scm")

    w, _, ok = _simplex_ols(y_pre, Y_donors_pre)
    if not ok and not np.isfinite(w).all():
        return _failed(y_pre, y_post, donor_names, "scm")

    if min_weight > 0:
        w = np.where(w >= min_weight, w, 0.0)
        s = w.sum()
        if s <= 0:
            return _failed(y_pre, y_post, donor_names, "scm")
        w = w / s

    synth_pre = Y_donors_pre @ w
    synth_post = Y_donors_post @ w
    return _package(y_pre, y_post, synth_pre, synth_post, w, donor_names, "scm")


def _package(y_pre, y_post, synth_pre, synth_post, w, donor_names, method,
             extras: Optional[dict] = None) -> SCMFit:
    mean_pre = float(np.mean(y_pre)) or 1.0
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
        weights={n: float(x) for n, x in zip(donor_names, w) if x > 1e-6},
        donor_names=list(donor_names), w=np.asarray(w, float),
        y_synth_pre=synth_pre, y_synth_post=synth_post,
        pre_rmspe=rmspe, pre_correlation=corr,
        att=att, att_pct=att_pct, effects_post=effects,
        success=True, method=method, extras=extras or {},
    )


def _failed(y_pre, y_post, donor_names, method) -> SCMFit:
    return SCMFit(weights={}, donor_names=list(donor_names),
                  w=np.zeros(len(donor_names)),
                  y_synth_pre=np.zeros_like(y_pre), y_synth_post=np.zeros_like(y_post),
                  pre_rmspe=np.inf, pre_correlation=0.0, att=np.nan, att_pct=np.nan,
                  effects_post=np.full_like(np.asarray(y_post, float), np.nan),
                  success=False, method=method)
