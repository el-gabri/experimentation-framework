"""Augmented SCM com ridge (Ben-Michael, Feller & Rothstein 2021).

τ̂_t = Y_tr,t(post) − [ Σ_i w_i^scm Y_i,t + (X_tr − Σ_i w_i^scm X_i)' β̂_t ]

onde X são os outcomes pré-tratamento e β̂_t vem de uma regressão ridge dos
outcomes pós dos doadores sobre seus outcomes pré. O termo de correção remove
o viés de desbalanceamento residual do SCM, que o SCM puro ignora. Uma
simplificação comum e incorreta é "corrigir" isso somando a média do resíduo
pré ao contrafactual — o que zera o gap pré por construção e mascara o fit real.

Quando não é fornecido, λ é escolhido pela validação cruzada proposta no
paper: cada período pré é omitido, o SCM e o outcome model são reajustados sem
esse período, e o erro de previsão da unidade tratada no período omitido é
avaliado. Nenhum outcome pós-tratamento participa da escolha de λ.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from supply_experiments.estimators.scm import SCMFit, _failed, _package, _simplex_ols


def _ridge_beta(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    """Ridge com intercepto (intercepto não penalizado). Retorna (b0, beta)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    p = X.shape[1]
    A = Xc.T @ Xc + lam * np.eye(p)
    rhs = Xc.T @ yc
    try:
        beta = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(A) @ rhs
    b0 = float(y.mean() - X.mean(axis=0) @ beta)
    return b0, beta


def _loo_cv_lambda(
    y_pre: np.ndarray,
    Y_donors_pre: np.ndarray,
    grid: Sequence[float],
) -> tuple[float, Dict[float, float], Dict[float, np.ndarray]]:
    """Leave-one-pre-period-out CV from Ben-Michael et al., equation (27).

    Returns ``(best_lambda, mse_by_lambda, predictions_by_lambda)``.  At each
    held-out period both the base SCM weights and the ridge outcome model are
    fitted without that period, so the treated outcome being scored cannot
    leak into either component.
    """
    T_pre = len(y_pre)
    scale = float(np.std(Y_donors_pre)) or 1.0
    errors: Dict[float, float] = {}
    predictions: Dict[float, np.ndarray] = {}

    # SCM weights do not depend on lambda. Fit each held-out-period SCM once,
    # then reuse its imbalance for every ridge candidate.
    fold_type = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    folds: List[Optional[fold_type]] = []
    for held_out in range(T_pre):
        keep = np.arange(T_pre) != held_out
        w_loo, _, ok, _ = _simplex_ols(y_pre[keep], Y_donors_pre[keep])
        if not ok:
            folds.append(None)
            continue
        X_train = Y_donors_pre[keep].T / scale
        donor_target = Y_donors_pre[held_out]
        imbalance = (y_pre[keep] - Y_donors_pre[keep] @ w_loo) / scale
        folds.append((w_loo, X_train, donor_target, imbalance))

    for lam in grid:
        preds = np.full(T_pre, np.nan)
        for held_out, fold in enumerate(folds):
            if fold is None:
                continue
            w_loo, X_train, donor_target, imbalance = fold
            _, beta = _ridge_beta(X_train, donor_target, float(lam))
            preds[held_out] = float(donor_target @ w_loo + imbalance @ beta)

        valid = np.isfinite(preds)
        mse = float(np.mean((y_pre[valid] - preds[valid]) ** 2)) if valid.all() else np.inf
        errors[float(lam)] = mse
        predictions[float(lam)] = preds

    best_lam = min(errors, key=lambda candidate: errors[candidate])
    return float(best_lam), errors, predictions


def _effective_weights(Xs: np.ndarray, w: np.ndarray, imbalance: np.ndarray,
                       lam: float) -> np.ndarray:
    """Pesos ASCM implícitos: w_aug = w_scm + X_c (X_c'X_c+λI)^-1 imbalance."""
    Xc = Xs - Xs.mean(axis=0, keepdims=True)
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
    try:
        adjustment = Xc @ np.linalg.solve(A, imbalance)
    except np.linalg.LinAlgError:
        adjustment = Xc @ (np.linalg.pinv(A) @ imbalance)
    return w + adjustment


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
    Y_donors_pre = np.asarray(Y_donors_pre, float)
    Y_donors_post = np.asarray(Y_donors_post, float)
    if (
        y_pre.ndim != 1 or y_post.ndim != 1
        or Y_donors_pre.ndim != 2 or Y_donors_post.ndim != 2
    ):
        return _failed(y_pre, y_post, donor_names, "ascm")
    T_pre, J = Y_donors_pre.shape
    if (
        J < 3 or T_pre < 3 or len(y_pre) != T_pre
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
        return _failed(y_pre, y_post, donor_names, "ascm")

    # 1) SCM base
    w, _, ok, solver = _simplex_ols(y_pre, Y_donors_pre)
    if not ok or not np.isfinite(w).all():
        return _failed(y_pre, y_post, donor_names, "ascm", extras=solver)

    # 2) Ridge: outcomes pós dos doadores ~ outcomes pré dos doadores
    X = Y_donors_pre.T                      # (J, T_pre): features = trajetória pré
    scale = float(np.std(X)) or 1.0
    Xs = X / scale
    if lam is None:
        grid = list(lam_grid) if lam_grid is not None else [0.1, 1.0, 10.0, 100.0, 1000.0]
    else:
        grid = [float(lam)]
    if not grid or any(not np.isfinite(x) or x < 0 for x in grid):
        raise ValueError("lam/lam_grid deve conter apenas valores finitos e não negativos")

    selected_lam, cv_errors, cv_predictions = _loo_cv_lambda(
        y_pre, Y_donors_pre, grid,
    )
    if lam is None:
        lam = selected_lam
    else:
        lam = float(lam)
    if not np.isfinite(cv_errors[float(lam)]):
        return _failed(
            y_pre, y_post, donor_names, "ascm",
            extras={**solver, "solver_status": "ascm_cv_failed"},
        )

    # 3) Correção de viés por período pós
    imbalance = (y_pre - Y_donors_pre @ w) / scale        # X_tr − X_co'w, escalado
    w_aug = _effective_weights(Xs, w, imbalance, lam)
    synth_post = Y_donors_post @ w_aug

    # No pré, o ASCM balanceia por construção via ridge; reportamos o synthetic
    # SCM puro no pré (transparência do fit) e a correção só no pós.
    synth_pre = Y_donors_pre @ w
    cv_pred = cv_predictions[float(lam)]
    mean_pre = float(np.mean(y_pre))
    augmented_pre_rmspe = (
        float(np.sqrt(np.mean((y_pre - cv_pred) ** 2))) / abs(mean_pre)
        if abs(mean_pre) > 1e-12 else np.inf
    )
    effective_weight_map = {
        name: float(weight) for name, weight in zip(donor_names, w_aug, strict=True)
    }
    extras = {
        **solver,
        "lambda": float(lam),
        "lambda_cv_method": "leave_one_pre_period_out",
        "lambda_cv_mse": {str(k): float(v) for k, v in cv_errors.items()},
        "augmented_pre_predictions": cv_pred.tolist(),
        "augmented_pre_rmspe": augmented_pre_rmspe,
        "effective_weights": w_aug.tolist(),
        "effective_weight_map": effective_weight_map,
        "extrapolation_l2": float(np.linalg.norm(w_aug - w)),
        "negative_effective_weight_mass": float(np.abs(w_aug[w_aug < 0]).sum()),
    }
    fit = _package(
        y_pre, y_post, synth_pre, synth_post, w, donor_names, "ascm",
        extras=extras,
    )
    return fit
