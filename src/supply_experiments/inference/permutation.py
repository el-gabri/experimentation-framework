"""Inferência por permutação in-space (Abadie et al. 2010/2015).

O p-value é a posição da unidade tratada na distribuição de placebo do
estatístico. Estatístico padrão: razão RMSPE pós/pré (robusto a fit ruim de
placebos — placebos com pré ruim não dominam). Também reporta o p-value do
ATT bruto para transparência.

É comum usar esse cálculo apenas para *validar o fit* do sintético; aqui ele é
promovido a *inferência primária* — a única honesta com 1–3 unidades tratadas.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, List, Optional, Set, Tuple

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
    real_fit: Optional[Any] = None  # fit real (SCMFit/DiDFit) — evita reajustar no chamador


def _rmspe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _fit_kwargs_for_group(fit_fn: Callable, fit_kwargs: dict,
                          n_treated_units: int) -> dict:
    """Passa o tamanho do grupo aos estimadores que o declaram.

    `fit_sdid` usa esse valor para a regularização zeta. O teste por
    permutação precisa usar a mesma definição de unidade tratada no fit real
    e em cada placebo; não deve depender de cada chamador lembrar esse kwarg.
    """
    out = dict(fit_kwargs)
    try:
        params = inspect.signature(fit_fn).parameters
    except (TypeError, ValueError):  # callable não inspecionável: mantém API genérica
        return out
    if "n_treated_units" in params:
        out.setdefault("n_treated_units", n_treated_units)
    return out


def _placebo_groups(
    n_donors: int,
    n_treated_units: int,
    max_group_placebos: Optional[int],
    rng: np.random.Generator,
) -> Tuple[List[Tuple[int, ...]], bool]:
    """Grupos placebo do mesmo tamanho da intervenção real.

    Para uma única tratada, devolve exatamente os placebos in-space clássicos.
    Para grupos, enumera todas as combinações quando isso é barato; caso
    contrário usa uma amostra sem reposição, mantendo o p-value de Monte Carlo
    explicitamente identificável na nota do resultado.
    """
    if n_treated_units < 1 or n_treated_units > n_donors:
        return [], False
    total = math.comb(n_donors, n_treated_units)
    if max_group_placebos is None or total <= max_group_placebos:
        return list(combinations(range(n_donors), n_treated_units)), False

    target = max_group_placebos
    sampled: Set[Tuple[int, ...]] = set()
    attempts = 0
    max_attempts = max(1000, target * 50)
    while len(sampled) < target and attempts < max_attempts:
        group = tuple(sorted(rng.choice(n_donors, size=n_treated_units, replace=False).tolist()))
        sampled.add(group)
        attempts += 1
    return sorted(sampled), True


def placebo_inference(
    fit_fn: Callable,                 # (y_pre, Yd_pre, y_post, Yd_post, names, **kw) -> SCMFit
    y_pre: np.ndarray,
    Y_donors_pre: np.ndarray,
    y_post: np.ndarray,
    Y_donors_post: np.ndarray,
    donor_names: List[str],
    fit_kwargs: Optional[dict] = None,
    min_placebos: int = 8,
    n_treated_units: int = 1,
    max_group_placebos: Optional[int] = 30,
    seed: Optional[int] = 123,
) -> PlaceboInference:
    """Executa inferência in-space com unidade placebo comparável à tratada.

    Quando a intervenção cobre mais de uma cidade, cada placebo é a soma de
    um grupo de doadoras do mesmo tamanho e o fit usa as doadoras restantes.
    Isso evita comparar uma série agregada contra placebos individuais, cujo
    nível e variância não representam o desenho real. Para pools grandes, a
    distribuição é uma amostra de Monte Carlo de combinações; ``note`` deixa
    isso explícito para o relatório e a calibração A/A deve usar o mesmo
    ``max_group_placebos`` do experimento.
    """
    if n_treated_units < 1:
        raise ValueError("n_treated_units deve ser >= 1")
    if max_group_placebos is not None and max_group_placebos < 1:
        raise ValueError("max_group_placebos deve ser >= 1 ou None")

    fit_kwargs = _fit_kwargs_for_group(fit_fn, fit_kwargs or {}, n_treated_units)
    J = len(donor_names)

    # fit real
    try:
        real = fit_fn(y_pre, Y_donors_pre, y_post, Y_donors_post, donor_names, **fit_kwargs)
    except Exception as exc:
        return PlaceboInference(
            np.nan, np.nan, np.nan, np.nan, [], np.nan, [], 0,
            f"Ajuste real falhou: {exc}", real_fit=None,
        )
    if not real.success:
        return PlaceboInference(
            np.nan, np.nan, np.nan, np.nan, [], np.nan, [], 0,
            "Ajuste real falhou; p-value não é reportado", real_fit=real,
        )
    pre_r = _rmspe(y_pre, real.y_synth_pre)
    post_r = _rmspe(y_post, real.y_synth_post)
    treated_stat = post_r / max(pre_r, 1e-12)
    treated_att = real.att

    placebo_stats, placebo_atts = [], []
    groups, sampled_groups = _placebo_groups(
        J, n_treated_units, max_group_placebos, np.random.default_rng(seed),
    )
    for group in groups:
        group_idx = np.asarray(group, dtype=int)
        yj_pre = Y_donors_pre[:, group_idx].sum(axis=1)
        yj_post = Y_donors_post[:, group_idx].sum(axis=1)
        mask = np.ones(J, dtype=bool)
        mask[group_idx] = False
        names_j = [donor_names[k] for k in range(J) if mask[k]]
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

    note_parts = []
    if n_treated_units > 1:
        kind = "amostrados" if sampled_groups else "exaustivos"
        note_parts.append(
            f"placebos em grupos de {n_treated_units} ({kind}; {len(groups)} candidatos)"
        )
    if len(placebo_stats) < min_placebos:
        note_parts.append(
            f"AVISO: apenas {len(placebo_stats)} placebos válidos "
            f"(< {min_placebos}); p-value tem granularidade grosseira "
            f"(mínimo possível = 1/{len(placebo_stats) + 1})."
        )
    note = " | ".join(note_parts)

    # p-values de permutação (incluem a tratada no denominador — Abadie)
    n = len(placebo_stats)
    if n == 0:
        return PlaceboInference(np.nan, np.nan, np.nan, treated_stat, [], treated_att, [], 0,
                                "Nenhum placebo válido", real_fit=real)

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
        n_placebos=n, note=note, real_fit=real,
    )
