"""DiD em painel de cidades (nível de unidade, não agregado).

y_it = α_i + γ_t + τ·(Treated_i × Post_t) + ε_it

Estimado via within-transformation (FE de cidade) + OLS. A inferência NÃO usa
os SEs clássicos deste OLS (inválidos com autocorrelação e poucos clusters):
usar wild cluster bootstrap (inference.bootstrap) ou permutação.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

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
    # Com efeitos fixos de data, DOW, feriado, tendência e o próprio ``post``
    # são colineares. Mantemos a assinatura por compatibilidade, mas o único
    # regressor identificado é a interação tratado × pós.
    return d, d[["tp"]].to_numpy(float), ["tp"]


def _two_way_demean(values: np.ndarray, city_codes: np.ndarray,
                    date_codes: np.ndarray) -> np.ndarray:
    """Remove FE de cidade e de data de um vetor de painel balanceado."""
    city_count = np.bincount(city_codes)
    date_count = np.bincount(date_codes)
    city_mean = np.bincount(city_codes, weights=values) / city_count
    date_mean = np.bincount(date_codes, weights=values) / date_count
    return values - city_mean[city_codes] - date_mean[date_codes] + float(np.mean(values))


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
    # Use the same membership and complete-window contract as the synthetic
    # estimators; never shrink a requested group or period silently.
    panel.slice_for(treated_cities, control_cities, window)
    treated = [str(city) for city in treated_cities]
    control = [str(city) for city in control_cities]

    pre_mask, post_mask = window.masks(panel.index)
    keep = pre_mask | post_mask
    sub = panel.outcome.loc[keep, treated + control].copy()
    if sub.isna().any().any():
        return DiDFit(np.nan, np.nan, pd.DataFrame(), {}, False)

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

    # Two-way within transformation: absorve FE de cidade e todos os choques
    # comuns de cada data. Isso é equivalente a estimar dummies de cidade e de
    # data, mas evita construir uma matriz densa N×T no painel diário.
    city_codes = d["city"].astype("category").cat.codes.to_numpy()
    date_codes = d["date"].astype("category").cat.codes.to_numpy()
    expected_rows = len(sub.index) * len(sub.columns)
    if len(d) != expected_rows:
        return DiDFit(np.nan, np.nan, pd.DataFrame(), {}, False)
    yd = _two_way_demean(y, city_codes, date_codes)
    Xd = np.column_stack([
        _two_way_demean(X[:, j], city_codes, date_codes) for j in range(X.shape[1])
    ])
    if np.linalg.matrix_rank(Xd) < Xd.shape[1]:
        raise ValueError("interação tratado × pós não identificada")

    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    tau = float(beta[0])
    resid = yd - Xd @ beta

    d = d.assign(resid=resid)
    # efeito relativo: com normalização, tau já é ~relativo; sem, divide pelo baseline tratado
    if normalize_scale:
        att_pct = tau
        # O coeficiente comum identifica a média relativa entre cidades. Com
        # múltiplas tratadas, convertê-lo em efeito absoluto agregado exigiria
        # homogeneidade de efeitos relativos. Não escondemos essa hipótese no
        # campo principal: a quantidade implícita fica apenas em design_info.
        att_abs_homogeneous = tau * float(sum(scale[c] for c in treated))
        att_abs = att_abs_homogeneous if len(treated) == 1 else np.nan
    else:
        base = float(panel.outcome.loc[panel.index[pre_mask], treated].sum(axis=1).mean())
        att_abs_homogeneous = tau * len(treated)
        att_abs = att_abs_homogeneous if len(treated) == 1 else np.nan
        att_pct = att_abs_homogeneous / base if base else np.nan

    if not np.isfinite(tau) or not np.isfinite(att_pct):
        raise ValueError("estimativa relativa DiD não finita")

    return DiDFit(
        att=att_abs, att_pct=att_pct,
        resid=d[["city", "date", "resid", "treated"]],
        design_info={"X_names": names, "Xd": Xd, "yd": yd, "city_codes": city_codes,
                     "date_codes": date_codes,
                     "treated_cities": treated, "control_cities": control,
                     "normalize_scale": normalize_scale, "scale": scale,
                     "estimand": "equal_city_average_relative_effect",
                     "att_abs_homogeneous_relative_effect": att_abs_homogeneous,
                     "att_abs_requires_homogeneous_effects": len(treated) > 1,
                     "window": window, "holidays": list(holidays or []),
                     "n_time_periods": int(date_codes.max() + 1)},
        success=True,
    )
