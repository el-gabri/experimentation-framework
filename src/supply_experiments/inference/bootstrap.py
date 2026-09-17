"""Restricted wild cluster bootstrap-t for panel DiD.

Webb weights improve support when the *total* number of clusters is small; they
do not repair the few-treated-cluster failure studied by MacKinnon and Webb.
This implementation therefore fails closed when fewer than four treated city
clusters are present. It is the ordinary WCR bootstrap, not the subcluster
bootstrap proposed for designs with very few treated clusters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from supply_experiments.estimators.did import DiDFit, _two_way_demean

_WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
_MIN_TREATED_CLUSTERS = 4


class UnsupportedFewTreatedClustersError(ValueError):
    """Ordinary WCB is unsupported because the design has too few treated clusters."""

    def __init__(self, n_treated_clusters: int) -> None:
        self.n_treated_clusters = int(n_treated_clusters)
        super().__init__(
            "ordinary wild cluster bootstrap is not valid with very few treated clusters: "
            f"found {self.n_treated_clusters}, require at least {_MIN_TREATED_CLUSTERS}; "
            "use design-based randomization inference or a validated few-treated method"
        )


@dataclass
class WildClusterResult:
    p_value: float
    t_stat: float
    boot_t: np.ndarray
    n_clusters: int
    n_boot: int
    weight_type: str
    n_treated_clusters: int = 0
    valid_for_inference: bool = False
    diagnostic: str = ""


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
    if not did_fit.success:
        raise ValueError("did_fit deve ser um ajuste DiD válido")
    if n_boot < 1:
        raise ValueError("n_boot deve ser >= 1")
    if weight_type not in {"auto", "webb", "rademacher"}:
        raise ValueError("weight_type deve ser 'auto', 'webb' ou 'rademacher'")

    rng = np.random.default_rng(seed)
    info = did_fit.design_info
    required = {"Xd", "yd", "city_codes", "date_codes"}
    if not required.issubset(info):
        raise ValueError(f"design_info ausente: {sorted(required - set(info))}")
    Xd = np.asarray(info["Xd"], float)
    yd = np.asarray(info["yd"], float)
    codes = np.asarray(info["city_codes"], int)
    date_codes = np.asarray(info["date_codes"], int)
    if Xd.ndim != 2 or yd.ndim != 1 or codes.ndim != 1 or len(Xd) != len(yd) or len(yd) != len(codes):
        raise ValueError("design_info possui shapes incompatíveis")
    if len(codes) == 0 or codes.min() < 0 or not np.isfinite(Xd).all() or not np.isfinite(yd).all():
        raise ValueError("design_info vazio ou não finito")
    G = int(codes.max() + 1)
    if set(np.unique(codes)) != set(range(G)):
        raise ValueError("city_codes deve ser contíguo de 0 a G-1")
    if (date_codes.ndim != 1 or len(date_codes) != len(yd)
            or date_codes.min() < 0
            or set(np.unique(date_codes)) != set(range(int(date_codes.max()) + 1))):
        raise ValueError("date_codes incompatível com o painel")
    pairs = np.column_stack((codes, date_codes))
    if (len(np.unique(pairs, axis=0)) != len(yd)
            or len(yd) != G * (int(date_codes.max()) + 1)):
        raise ValueError("bootstrap requer painel cidade × data balanceado")

    treated_cities = info.get("treated_cities")
    if treated_cities is not None:
        n_treated = len(set(treated_cities))
    elif {"city", "treated"}.issubset(did_fit.resid.columns):
        treated_rows = did_fit.resid.loc[did_fit.resid["treated"].astype(bool), "city"]
        n_treated = int(treated_rows.nunique())
    else:
        raise ValueError("não foi possível determinar o número de clusters tratados")
    if n_treated < _MIN_TREATED_CLUSTERS:
        raise UnsupportedFewTreatedClustersError(n_treated)
    if n_treated >= G:
        raise ValueError("wild cluster bootstrap requer ao menos um cluster de controle")

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
        # City multipliers break the zero date means of the original residuals.
        # Reabsorb both fixed effects before studentizing each bootstrap draw.
        y_b = _two_way_demean(y_b, codes, date_codes)
        beta_b, *_ = np.linalg.lstsq(Xd, y_b, rcond=None)
        resid_b = y_b - Xd @ beta_b
        se_b = _cluster_se(Xd, resid_b, codes)
        boot_t[b] = beta_b[0] / se_b if se_b > 0 else 0.0

    p = float((1 + np.sum(np.abs(boot_t) >= abs(t_obs))) / (n_boot + 1))
    return WildClusterResult(
        p_value=p,
        t_stat=t_obs,
        boot_t=boot_t,
        n_clusters=G,
        n_boot=n_boot,
        weight_type=weight_type,
        n_treated_clusters=n_treated,
        # Four treated clusters is only a coarse misuse guard. The package has
        # no design-matched size calibration that would justify promoting this
        # ordinary WCB result to decision-valid inference.
        valid_for_inference=False,
        diagnostic=(
            "ordinary restricted WCB diagnostic only; the four-treated-cluster "
            "cutoff is not a validity certificate, and size still depends on "
            "cluster independence and treated/control balance"
        ),
    )
