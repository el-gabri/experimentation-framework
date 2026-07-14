"""Inferência por permutação in-space (Abadie et al. 2010/2015).

O p-value é a posição da unidade tratada na distribuição de placebo do
estatístico. Estatístico padrão: razão RMSPE pós/pré (robusto a fit ruim de
placebos — placebos com pré ruim não dominam). Também reporta o p-value do
ATT bruto para transparência.

Este é o mesmo cálculo que o framework antigo fazia para *validar fit*, mas
aqui ele é a *inferência primária* — a única honesta com 1–3 unidades tratadas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class PlaceboInference:
    p_value: float                 # bilateral, razão RMSPE (recomendado p/ decisão)
    p_value_att: float             # bilateral, |ATT|
    p_value_att_onesided: float    # P(placebo ATT <= observado) — p/ efeitos com sinal esperado
    treated_stat: float
    placebo_stats: List[float]
    treated_att: float
    placebo_atts: List[float]
    n_placebos: int
    note: str = ""


def _rmspe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def placebo_inference(
    fit_fn: Callable,                 # (y_pre, Yd_pre, y_post, Yd_post, names, **kw) -> SCMFit
    y_pre: np.ndarray,
    Y_donors_pre: np.ndarray,
    y_post: np.ndarray,
    Y_donors_post: np.ndarray,
    donor_names: List[str],
    fit_kwargs: Optional[dict] = None,
    min_placebos: int = 8,
) -> PlaceboInference:
    fit_kwargs = fit_kwargs or {}
    J = len(donor_names)

    # fit real
    real = fit_fn(y_pre, Y_donors_pre, y_post, Y_donors_post, donor_names, **fit_kwargs)
    pre_r = _rmspe(y_pre, real.y_synth_pre)
    post_r = _rmspe(y_post, real.y_synth_post)
    treated_stat = post_r / max(pre_r, 1e-12)
    treated_att = real.att

    placebo_stats, placebo_atts = [], []
    for j in range(J):
        yj_pre, yj_post = Y_donors_pre[:, j], Y_donors_post[:, j]
        mask = np.arange(J) != j
        names_j = [donor_names[k] for k in range(J) if k != j]
        if mask.sum() < 2:
            continue
        try:
            pf = fit_fn(yj_pre, Y_donors_pre[:, mask], yj_post, Y_donors_post[:, mask],
                        names_j, **fit_kwargs)
        except Exception:
            continue
        if not pf.success:
            continue
        ppre = _rmspe(yj_pre, pf.y_synth_pre)
        ppost = _rmspe(yj_post, pf.y_synth_post)
        placebo_stats.append(ppost / max(ppre, 1e-12))
        placebo_atts.append(pf.att_pct if np.isfinite(pf.att_pct) else np.nan)

    note = ""
    if len(placebo_stats) < min_placebos:
        note = (f"AVISO: apenas {len(placebo_stats)} placebos válidos "
                f"(< {min_placebos}); p-value tem granularidade grosseira "
                f"(mínimo possível = 1/{len(placebo_stats) + 1}).")

    # p-values de permutação (incluem a tratada no denominador — Abadie)
    n = len(placebo_stats)
    if n == 0:
        return PlaceboInference(np.nan, np.nan, np.nan, treated_stat, [], treated_att, [], 0,
                                "Nenhum placebo válido")

    p_rmspe = (1 + sum(s >= treated_stat for s in placebo_stats)) / (n + 1)

    att_pcts = np.array([a for a in placebo_atts if np.isfinite(a)])
    treated_att_pct = real.att_pct
    if len(att_pcts) and np.isfinite(treated_att_pct):
        p_att = (1 + np.sum(np.abs(att_pcts) >= abs(treated_att_pct))) / (len(att_pcts) + 1)
        p_att_neg = (1 + np.sum(att_pcts <= treated_att_pct)) / (len(att_pcts) + 1)
    else:
        p_att, p_att_neg = np.nan, np.nan

    return PlaceboInference(
        p_value=float(p_rmspe), p_value_att=float(p_att),
        p_value_att_onesided=float(p_att_neg),
        treated_stat=float(treated_stat), placebo_stats=placebo_stats,
        treated_att=float(treated_att), placebo_atts=placebo_atts,
        n_placebos=n, note=note,
    )
