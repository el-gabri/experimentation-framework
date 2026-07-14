"""DiD em painel de cidades (nível de unidade, não agregado).

y_it = α_i + γ_t(dow, feriado, tendência) + τ·(Treated_i × Post_t) + ε_it

Estimado via within-transformation (FE de cidade) + OLS. A inferência NÃO usa
os SEs clássicos deste OLS (inválidos com autocorrelação e poucos clusters):
usar wild cluster bootstrap (inference.bootstrap) ou permutação.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from supply_experiments.panel import CityPanel, ExperimentWindow


@dataclass
class DiDFit:
    att: float
    att_pct: float
    resid: pd.DataFrame        # long: city, date, resid (p/ bootstrap)
    design_info: dict
    success: bool
    method: str = "did_panel"


def _build_design(long: pd.DataFrame, treated: set, start_date, holidays: Optional[Sequence[str]]):
    d = long.copy()
    d["treated"] = d["city"].isin(treated).astype(float)
    d["post"] = (d["date"].dt.date >= start_date).astype(float)
    d["tp"] = d["treated"] * d["post"]

    # dummies de tempo: dow + feriado + tendência linear + post
    dow = pd.get_dummies(d["date"].dt.dayofweek, prefix="dow", drop_first=True).astype(float)
    t_rel = (d["date"] - d["date"].min()).dt.days.astype(float)
    t_rel = (t_rel - t_rel.mean()) / max(t_rel.std(), 1.0)
    cols = [d["post"].to_numpy(), t_rel.to_numpy()]
    names = ["post", "trend"]
    for c in dow.columns:
        cols.append(dow[c].to_numpy())
        names.append(c)
    if holidays:
        hs = set(pd.to_datetime(list(holidays)).date)
        cols.append(d["date"].dt.date.isin(hs).astype(float).to_numpy())
        names.append("holiday")
    X = np.column_stack([d["tp"].to_numpy()] + cols)
    names = ["tp"] + names
    return d, X, names


def fit_did_panel(
    panel: CityPanel,
    treated_cities: Sequence[str],
    control_cities: Sequence[str],
    window: ExperimentWindow,
    holidays: Optional[Sequence[str]] = None,
    normalize_scale: bool = True,
) -> DiDFit:
    """
    normalize_scale: divide o y de cada cidade pela sua média pré (comparação de
    forma, não nível — cidades têm escalas de GMV muito diferentes e sem isso a
    within-FE só remove nível, não escala; o τ vira dominado pelas maiores).
    Com normalize_scale, τ já é aproximadamente o efeito relativo.
    """
    treated = [c for c in treated_cities if c in panel.cities]
    control = [c for c in control_cities if c in panel.cities]
    if not treated or len(control) < 2:
        return DiDFit(np.nan, np.nan, pd.DataFrame(), {}, False)

    pre_mask, post_mask = window.masks(panel.index)
    keep = pre_mask | post_mask
    sub = panel.outcome.loc[keep, treated + control].copy()

    scale = {}
    if normalize_scale:
        pre_idx = panel.index[pre_mask]
        for c in sub.columns:
            m = float(panel.outcome.loc[pre_idx, c].mean())
            scale[c] = m if m > 1e-9 else 1.0
            sub[c] = sub[c] / scale[c]

    long = sub.stack().rename("y").reset_index()
    long.columns = ["date", "city", "y"]

    d, X, names = _build_design(long, set(treated), window.start_date, holidays)
    y = d["y"].to_numpy(float)

    # within-transformation: demean por cidade (FE de unidade)
    city_codes = d["city"].astype("category").cat.codes.to_numpy()
    for arr in [y] + [X[:, j] for j in range(X.shape[1])]:
        pass  # demean feito vetorizado abaixo
    def demean(v):
        means = np.bincount(city_codes, weights=v) / np.bincount(city_codes)
        return v - means[city_codes]

    yd = demean(y)
    Xd = np.column_stack([demean(X[:, j]) for j in range(X.shape[1])])

    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    tau = float(beta[0])
    resid = yd - Xd @ beta

    d = d.assign(resid=resid)
    # efeito relativo: com normalização, tau já é ~relativo; sem, divide pelo baseline tratado
    if normalize_scale:
        att_pct = tau
        att_abs = tau * float(np.mean([scale[c] for c in treated])) * len(treated)
    else:
        base = float(panel.outcome.loc[panel.index[pre_mask], treated].sum(axis=1).mean())
        att_abs, att_pct = tau * len(treated), (tau * len(treated)) / base if base else np.nan

    return DiDFit(
        att=att_abs, att_pct=att_pct,
        resid=d[["city", "date", "resid", "treated"]],
        design_info={"X_names": names, "Xd": Xd, "yd": yd, "city_codes": city_codes,
                     "treated_cities": treated, "control_cities": control,
                     "normalize_scale": normalize_scale, "scale": scale,
                     "window": window, "holidays": list(holidays or [])},
        success=True,
    )
