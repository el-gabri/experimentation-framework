"""Synthetic Difference-in-Differences (Arkhangelsky et al., AER 2021).

Pesos unitários: min_{w0, w∈simplex} Σ_pre (w0 + Y_co,t'w − ȳ_tr,t)² + ζ² T_pre ||w||²
  com ζ = (N_tr · T_post)^{1/4} · σ̂,  σ̂ = dp das primeiras diferenças dos controles no pré.
Pesos temporais: min_{λ0, λ∈simplex} Σ_i∈co (λ0 + Y_i,pre'λ − ȳ_i,post)²  (+ ridge ínfimo p/ unicidade)
Estimador: τ̂ = (ȳ_tr,post − Σ_t λ_t ȳ_tr,t) − Σ_i w_i (ȳ_i,post − Σ_t λ_t Y_i,t)

Para múltiplas unidades, ``y_pre`` e ``y_post`` representam a *média* das
tratadas, como no estimando do paper. ``treated_aggregation="sum"`` existe
somente para adaptar dados legados e divide explicitamente pelo número de
tratadas antes do ajuste.

Implementações simplificadas costumam usar pesos temporais lineares arbitrários
e sem regularização ζ — perdendo exatamente a propriedade de duplo-robustez
local que justifica o SDID.
"""

from __future__ import annotations

from typing import List, Literal

import numpy as np

from supply_experiments.estimators.scm import SCMFit, _failed, _simplex_ols


def _solve_simplex_intercept_checked(
    A: np.ndarray,
    b: np.ndarray,
    ridge: float,
    *,
    max_iter: int = 5000,
    tol: float = 1e-10,
) -> tuple[np.ndarray, bool, dict]:
    """Solve the SDID weight QP and return a scale-aware KKT certificate.

    ``_simplex_ols`` standardizes the outcome and design by ``std(b)``.  The
    ridge is therefore divided by ``std(b)^2`` so this remains exactly the
    stated SDID objective, merely expressed in standardized objective units.
    """
    b = np.asarray(b, float)
    scale = float(np.std(b)) or 1.0
    scaled_ridge = float(ridge) / (scale ** 2)
    v, _, converged, diagnostics = _simplex_ols(
        b,
        np.asarray(A, float),
        ridge=scaled_ridge,
        intercept=True,
        max_iter=max_iter,
        tol=tol,
    )
    return v, converged, {
        **diagnostics,
        "solver_original_ridge": float(ridge),
        "solver_scaled_ridge": scaled_ridge,
    }


def _solve_simplex_intercept(A: np.ndarray, b: np.ndarray, ridge: float) -> np.ndarray:
    """Return certified simplex/intercept weights; raise if the QP fails."""
    v, converged, diagnostics = _solve_simplex_intercept_checked(A, b, ridge)
    if not converged:
        raise RuntimeError(
            "SDID simplex solver did not converge: "
            f"{diagnostics.get('solver_status', 'unknown')}"
        )
    return v


def fit_sdid(
    y_pre: np.ndarray,
    Y_donors_pre: np.ndarray,
    y_post: np.ndarray,
    Y_donors_post: np.ndarray,
    donor_names: List[str],
    n_treated_units: int = 1,
    treated_aggregation: Literal["mean", "sum"] = "mean",
) -> SCMFit:
    y_pre = np.asarray(y_pre, float)
    y_post = np.asarray(y_post, float)
    Y_donors_pre = np.asarray(Y_donors_pre, float)
    Y_donors_post = np.asarray(Y_donors_post, float)
    if n_treated_units < 1:
        raise ValueError("n_treated_units deve ser >= 1")
    if treated_aggregation not in {"mean", "sum"}:
        raise ValueError("treated_aggregation deve ser 'mean' ou 'sum'")
    if treated_aggregation == "sum":
        y_pre = y_pre / n_treated_units
        y_post = y_post / n_treated_units

    if y_pre.ndim != 1 or y_post.ndim != 1 or Y_donors_pre.ndim != 2 or Y_donors_post.ndim != 2:
        return _failed(y_pre, y_post, donor_names, "sdid")
    T_pre, J = Y_donors_pre.shape
    T_post = len(y_post)
    if (
        J < 2 or T_pre < 5 or T_post < 1
        or Y_donors_post.shape != (T_post, J)
        or len(y_pre) != T_pre or len(donor_names) != J
        or any(not np.isfinite(x).all() for x in (y_pre, y_post, Y_donors_pre, Y_donors_post))
    ):
        return _failed(y_pre, y_post, donor_names, "sdid")

    # ---- ζ de Arkhangelsky et al. ----
    diffs = np.diff(Y_donors_pre, axis=0)              # (T_pre-1, J)
    # Algorithm 1 usa o segundo momento populacional das primeiras diferenças.
    sigma = float(np.sqrt(np.mean((diffs - diffs.mean()) ** 2))) if diffs.size else 0.0
    zeta = (n_treated_units * T_post) ** 0.25 * sigma
    ridge_w = (zeta ** 2) * T_pre

    # ---- pesos unitários ω ----
    w, unit_converged, unit_solver = _solve_simplex_intercept_checked(
        Y_donors_pre, y_pre, ridge=ridge_w,
    )
    unit_extras = {f"unit_{key}": value for key, value in unit_solver.items()}
    if not unit_converged:
        return _failed(
            y_pre, y_post, donor_names, "sdid",
            extras={
                **unit_extras,
                "failure_reason": "unit_weight_solver_nonconvergence",
            },
        )

    # ---- pesos temporais λ ----
    ybar_post_donors = Y_donors_post.mean(axis=0)      # (J,)
    zeta_time = 1e-6 * sigma
    ridge_time = (zeta_time ** 2) * J
    lam, time_converged, time_solver = _solve_simplex_intercept_checked(
        Y_donors_pre.T, ybar_post_donors, ridge=ridge_time,
    )
    time_extras = {f"time_{key}": value for key, value in time_solver.items()}
    if not time_converged:
        return _failed(
            y_pre, y_post, donor_names, "sdid",
            extras={
                **unit_extras,
                **time_extras,
                "failure_reason": "time_weight_solver_nonconvergence",
            },
        )

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

    mean_pre = float(np.mean(y_pre))
    if abs(mean_pre) < 1e-9:
        return _failed(
            y_pre, y_post, donor_names, "sdid",
            extras={"failure_reason": "near_zero_treated_pre_mean"},
        )
    rmspe = float(np.sqrt(np.mean((y_pre - synth_pre) ** 2))) / abs(mean_pre)
    corr = float(np.corrcoef(y_pre, synth_pre)[0, 1]) if np.std(synth_pre) > 1e-12 else 0.0
    denom = float(np.mean(synth_post))
    return SCMFit(
        weights={n: float(x) for n, x in zip(donor_names, w, strict=True) if x > 1e-6},
        donor_names=list(donor_names), w=w,
        y_synth_pre=synth_pre, y_synth_post=synth_post,
        pre_rmspe=rmspe, pre_correlation=corr,
        att=float(att), att_pct=float(att) / denom if abs(denom) > 1e-12 else np.nan,
        effects_post=effects, success=True, method="sdid",
        extras={
            **unit_extras,
            **time_extras,
            "zeta": zeta,
            "unit_ridge": ridge_w,
            "time_zeta": zeta_time,
            "time_ridge": ridge_time,
            "sigma_first_differences": sigma,
            "time_weights": lam.tolist(),
            "n_treated_units": n_treated_units,
            "treated_aggregation": treated_aggregation,
            "att_total_across_treated": float(att) * n_treated_units,
        },
    )
