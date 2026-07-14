"""Wild cluster bootstrap-t para DiD em painel (Cameron, Gelbach & Miller 2008).

Com poucos clusters tratados usa pesos de Webb (6 pontos; MacKinnon & Webb 2018),
que evitam a degenerescência do Rademacher quando o nº de clusters é pequeno.
Impõe H0 (τ=0) no DGP do bootstrap — a versão "restricted" (WCR), com melhor
controle de tamanho.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from supply_experiments.estimators.did import DiDFit

_WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


@dataclass
class WildClusterResult:
    p_value: float
    t_stat: float
    boot_t: np.ndarray
    n_clusters: int
    n_boot: int
    weight_type: str


def _cluster_se(Xd: np.ndarray, resid: np.ndarray, codes: np.ndarray) -> float:
    """SE cluster-robusto (CR1) do primeiro coeficiente."""
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    G = codes.max() + 1
    meat = np.zeros((Xd.shape[1], Xd.shape[1]))
    for g in range(G):
        m = codes == g
        s = Xd[m].T @ resid[m]
        meat += np.outer(s, s)
    n, k = Xd.shape
    adj = (G / (G - 1)) * ((n - 1) / max(n - k, 1))
    V = adj * XtX_inv @ meat @ XtX_inv
    return float(np.sqrt(max(V[0, 0], 1e-300)))


def wild_cluster_bootstrap(
    did_fit: DiDFit,
    n_boot: int = 999,
    seed: Optional[int] = None,
    weight_type: str = "auto",
) -> WildClusterResult:
    rng = np.random.default_rng(seed)
    info = did_fit.design_info
    Xd, yd, codes = info["Xd"], info["yd"], info["city_codes"]
    G = int(codes.max() + 1)

    if weight_type == "auto":
        weight_type = "webb" if G < 12 else "rademacher"

    # ---- estimativa irrestrita e t observado ----
    beta_u, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    resid_u = yd - Xd @ beta_u
    se_u = _cluster_se(Xd, resid_u, codes)
    t_obs = float(beta_u[0] / se_u) if se_u > 0 else np.nan

    # ---- estimativa restrita (H0: beta[0]=0) ----
    Xr = Xd[:, 1:]
    beta_r, *_ = np.linalg.lstsq(Xr, yd, rcond=None)
    resid_r = yd - Xr @ beta_r
    fitted_r = Xr @ beta_r

    boot_t = np.empty(n_boot)
    for b in range(n_boot):
        if weight_type == "webb":
            wg = rng.choice(_WEBB, size=G)
        else:
            wg = rng.choice([-1.0, 1.0], size=G)
        y_b = fitted_r + resid_r * wg[codes]
        beta_b, *_ = np.linalg.lstsq(Xd, y_b, rcond=None)
        resid_b = y_b - Xd @ beta_b
        se_b = _cluster_se(Xd, resid_b, codes)
        boot_t[b] = beta_b[0] / se_b if se_b > 0 else 0.0

    p = float((1 + np.sum(np.abs(boot_t) >= abs(t_obs))) / (n_boot + 1))
    return WildClusterResult(p_value=p, t_stat=t_obs, boot_t=boot_t,
                             n_clusters=G, n_boot=n_boot, weight_type=weight_type)
