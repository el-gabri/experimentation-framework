"""Conformal inversion for the sharp constant-effect hypothesis under SCM.

For a candidate ``tau0``, the post-treatment outcomes are adjusted by
``tau0`` and classic SCM is refitted on the complete null-imposed trajectory.
The test statistic is the RMS residual in the designated post block and its
reference distribution uses all cyclic shifts of the full residual series.

This is deliberately restricted to :func:`fit_scm`. ASCM and SDID depend on
the pre/post partition in ways that are not permutation-equivariant under this
wrapper. The result is an inverted test of ``H0: tau_t = tau0`` for every post
period; it is not a generic confidence interval for an unrestricted ATT.
Exactness requires exchangeable residuals. The dependent-data justification
instead requires stable counterfactual estimation and stationary/strongly
mixing residuals; those substantive assumptions cannot be established here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from supply_experiments.estimators.scm import fit_scm


@dataclass
class ConformalResult:
    p_value: float
    ci_lower_pct: float
    ci_upper_pct: float
    alpha: float
    grid_evaluated: int
    confidence_level: float = np.nan
    hypothesis: str = "sharp_constant_post_effect"
    accepted_components_pct: List[Tuple[float, float]] = field(default_factory=list)
    acceptance_intervals_pct: List[Tuple[float, float]] = field(default_factory=list)
    acceptance_set_disconnected: bool = False
    boundary_truncated: bool = False
    lower_boundary_truncated: bool = False
    upper_boundary_truncated: bool = False
    grid_lower_pct: float = np.nan
    grid_upper_pct: float = np.nan
    assumption_note: str = (
        "Exact only under exchangeable residuals; cyclic-shift validity for dependent data "
        "requires stable estimation and stationary/strongly mixing residuals."
    )


def _pvalue_for_tau(fit_fn, y_pre, Yd_pre, y_post, Yd_post, names,
                    tau0_abs_per_t: np.ndarray, fit_kwargs) -> float:
    """Cyclic-shift p-value for a fully specified post-treatment effect path."""
    T_post = len(y_post)
    y_post_h0 = y_post - tau0_abs_per_t
    y_all = np.concatenate([y_pre, y_post_h0])
    Yd_all = np.vstack([Yd_pre, Yd_post])

    # fit_scm uses only its first two arrays to estimate weights. Empty post
    # arrays make explicit that the observed post block is not used a second
    # time after imposing the candidate sharp null.
    fit = fit_fn(
        y_all,
        Yd_all,
        np.empty(0, dtype=float),
        np.empty((0, Yd_all.shape[1]), dtype=float),
        names,
        **fit_kwargs,
    )
    if not fit.success or len(fit.y_synth_pre) != len(y_all):
        return np.nan
    resid_all = y_all - fit.y_synth_pre
    T = len(resid_all)

    def stat(u: np.ndarray) -> float:
        return float(np.sqrt(np.mean(u[-T_post:] ** 2)))

    s_obs = stat(resid_all)
    shifted_stats = np.array([stat(np.roll(resid_all, shift)) for shift in range(T)])
    return float(np.mean(shifted_stats >= s_obs - 1e-15))


def _accepted_components(
    grid: np.ndarray, accepted: np.ndarray,
) -> List[Tuple[float, float]]:
    components: List[Tuple[float, float]] = []
    start: Optional[int] = None
    for index, is_accepted in enumerate(accepted):
        if is_accepted and start is None:
            start = index
        if start is not None and (not is_accepted or index == len(accepted) - 1):
            end = index if is_accepted and index == len(accepted) - 1 else index - 1
            components.append((float(grid[start]), float(grid[end])))
            start = None
    return components


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
    """Invert cyclic-shift SCM tests over a relative constant-effect grid.

    ``ci_lower_pct`` and ``ci_upper_pct`` retain the backward-compatible hull
    of the accepted grid. Inspect ``accepted_components_pct`` before treating
    that hull as an interval. Boundary flags mean the requested grid was too
    narrow to identify the corresponding endpoint.
    """
    if fit_fn is not fit_scm:
        raise ValueError(
            "conformal_inference suporta apenas fit_scm; ASCM/SDID não são "
            "permutation-equivariant sob este wrapper"
        )
    if not 0 < alpha < 1:
        raise ValueError("alpha deve estar estritamente entre 0 e 1")

    fit_kwargs = dict(fit_kwargs or {})
    y_pre = np.asarray(y_pre, float)
    y_post = np.asarray(y_post, float)
    Y_donors_pre = np.asarray(Y_donors_pre, float)
    Y_donors_post = np.asarray(Y_donors_post, float)
    if y_pre.ndim != 1 or y_post.ndim != 1:
        raise ValueError("y_pre e y_post devem ser vetores")
    if Y_donors_pre.ndim != 2 or Y_donors_post.ndim != 2:
        raise ValueError("matrizes de doadoras devem ser bidimensionais")
    J = len(donor_names)
    if J < 2 or len(set(donor_names)) != J:
        raise ValueError("donor_names deve conter pelo menos 2 nomes únicos")
    if Y_donors_pre.shape != (len(y_pre), J):
        raise ValueError("shape de Y_donors_pre incompatível")
    if Y_donors_post.shape != (len(y_post), J):
        raise ValueError("shape de Y_donors_post incompatível")
    if len(y_post) < 1 or len(y_pre) < len(y_post) + 2:
        raise ValueError("conformal requer T_pre >= T_post + 2 e T_post >= 1")
    arrays = (y_pre, y_post, Y_donors_pre, Y_donors_post)
    if any(not np.isfinite(x).all() for x in arrays):
        raise ValueError("outcomes contêm NaN ou infinito")

    if rel_grid is None:
        grid = np.linspace(-0.5, 0.5, 51)
    else:
        grid = np.asarray(rel_grid, float)
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all():
        raise ValueError("rel_grid deve ser um vetor finito com pelo menos 2 pontos")
    if not np.all(np.diff(grid) > 0):
        raise ValueError("rel_grid deve ser estritamente crescente e sem duplicatas")

    if counterfactual_level is None:
        base_fit = fit_scm(
            y_pre, Y_donors_pre, y_post, Y_donors_post, donor_names, **fit_kwargs,
        )
        if not base_fit.success:
            raise ValueError("SCM base falhou; inferência conformal não é identificável")
        base = float(np.mean(base_fit.y_synth_post))
    else:
        base = float(counterfactual_level)
    if not np.isfinite(base) or abs(base) < 1e-9:
        raise ValueError("counterfactual_level deve ser finito e diferente de zero")

    p0 = _pvalue_for_tau(
        fit_scm, y_pre, Y_donors_pre, y_post, Y_donors_post,
        donor_names, np.zeros_like(y_post), fit_kwargs,
    )
    if not np.isfinite(p0):
        raise ValueError("ajuste conformal sob H0 falhou")

    p_values = np.empty(len(grid), dtype=float)
    for index, relative_effect in enumerate(grid):
        tau_abs = np.full_like(y_post, relative_effect * base)
        p_values[index] = _pvalue_for_tau(
            fit_scm, y_pre, Y_donors_pre, y_post, Y_donors_post,
            donor_names, tau_abs, fit_kwargs,
        )
    accepted = np.isfinite(p_values) & (p_values > alpha)
    components = _accepted_components(grid, accepted)

    if components:
        lo = components[0][0]
        hi = components[-1][1]
    else:
        lo = hi = np.nan

    return ConformalResult(
        p_value=float(p0),
        ci_lower_pct=float(lo),
        ci_upper_pct=float(hi),
        alpha=float(alpha),
        grid_evaluated=len(grid),
        confidence_level=float(1.0 - alpha),
        accepted_components_pct=components,
        acceptance_intervals_pct=components,
        acceptance_set_disconnected=len(components) > 1,
        boundary_truncated=bool(accepted[0] or accepted[-1]),
        lower_boundary_truncated=bool(accepted[0]),
        upper_boundary_truncated=bool(accepted[-1]),
        grid_lower_pct=float(grid[0]),
        grid_upper_pct=float(grid[-1]),
    )
