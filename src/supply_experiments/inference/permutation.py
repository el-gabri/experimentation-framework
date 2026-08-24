"""Placebos in-space e inferência por randomização quando o desenho permite.

O p-value é a posição da unidade tratada na distribuição de placebo do
estatístico. Estatístico padrão: razão RMSPE pós/pré (robusto a fit ruim de
placebos — placebos com pré ruim não dominam). Também reporta o p-value do
ATT bruto para transparência.

Sem alocação aleatória/exchangeable ou calibração do procedimento completo, a
posição observada é um diagnóstico comparativo, não um p-value exato. O objeto
de resultado torna essa distinção explícita em ``valid_for_decision``.
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
    p_value: float                 # rank bilateral da razão RMSPE pós/pré
    p_value_att: float             # bilateral, |ATT|
    p_value_att_onesided: float    # P(placebo ATT <= observado) — p/ efeitos com sinal esperado
    treated_stat: float
    placebo_stats: List[float]
    treated_att: float             # ATT relativo; mesma unidade de placebo_atts
    placebo_atts: List[float]      # ATTs relativos
    n_placebos: int
    note: str = ""
    real_fit: Optional[Any] = None  # fit real (SCMFit/DiDFit) — evita reajustar no chamador
    treated_att_abs: float = np.nan
    valid_for_decision: bool = False
    inference_basis: str = "observational_placebo_rank"
    sampled_placebos: bool = False
    n_attempted_assignments: int = 0
    n_failed_assignments: int = 0
    n_filtered_assignments: int = 0


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
        out["n_treated_units"] = n_treated_units
    if "treated_aggregation" in params:
        # Every high-level caller and grouped placebo supplies a treated mean.
        # Legacy summed trajectories remain available only via direct fit_sdid.
        out["treated_aggregation"] = "mean"
    return out


def _placebo_groups(
    n_donors: int,
    n_treated_units: int,
    max_group_placebos: Optional[int],
    rng: np.random.Generator,
    max_exhaustive_placebos: int = 10_000,
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
    if max_group_placebos is None and total > max_exhaustive_placebos:
        raise ValueError(
            f"enumeração exaustiva teria {total} grupos; defina "
            "max_group_placebos para amostragem Monte Carlo"
        )
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


def _randomization_groups(
    n_units: int,
    n_treated_units: int,
    observed_group: Tuple[int, ...],
    max_group_placebos: Optional[int],
    rng: np.random.Generator,
    max_exhaustive_placebos: int = 10_000,
) -> Tuple[List[Tuple[int, ...]], bool]:
    """Alternative assignments from the full randomized unit universe.

    The observed assignment is excluded because the +1 correction adds it to
    the permutation-rank denominator. Every alternative fit therefore has the
    same number of treated and control units as the observed fit.
    """
    if n_treated_units < 1 or n_treated_units >= n_units:
        return [], False
    observed = tuple(sorted(observed_group))
    total = math.comb(n_units, n_treated_units) - 1
    if max_group_placebos is None and total > max_exhaustive_placebos:
        raise ValueError(
            f"enumeração exaustiva teria {total} alocações alternativas; "
            "defina max_group_placebos para amostragem Monte Carlo"
        )
    if max_group_placebos is None or total <= max_group_placebos:
        return [
            group
            for group in combinations(range(n_units), n_treated_units)
            if group != observed
        ], False

    target = min(max_group_placebos, total)
    sampled: Set[Tuple[int, ...]] = set()
    attempts = 0
    max_attempts = max(1000, target * 100)
    while len(sampled) < target and attempts < max_attempts:
        group = tuple(
            sorted(rng.choice(n_units, size=n_treated_units, replace=False).tolist())
        )
        if group != observed:
            sampled.add(group)
        attempts += 1
    if len(sampled) != target:  # pragma: no cover - defensive combinatorial guard
        raise RuntimeError("não foi possível amostrar alocações aleatórias distintas")
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
    assignment_mechanism: str = "observational",
    calibrated: bool = False,
    max_exhaustive_placebos: int = 10_000,
    max_pre_rmspe_multiple: Optional[float] = None,
    Y_treated_pre: Optional[np.ndarray] = None,
    Y_treated_post: Optional[np.ndarray] = None,
    treated_names: Optional[List[str]] = None,
) -> PlaceboInference:
    """Executa inferência in-space com unidade placebo comparável à tratada.

    Quando a intervenção cobre mais de uma cidade, cada placebo é a média de
    um grupo de doadoras do mesmo tamanho e o fit usa as doadoras restantes,
    reproduzindo o estimando médio da trajetória tratada. Para pools grandes, a
    distribuição é uma amostra de Monte Carlo de combinações; ``note`` deixa
    isso explícito para o relatório e a calibração A/A deve usar o mesmo
    ``max_group_placebos`` do experimento.
    """
    if n_treated_units < 1:
        raise ValueError("n_treated_units deve ser >= 1")
    if min_placebos < 1:
        raise ValueError("min_placebos deve ser >= 1")
    if max_group_placebos is not None and max_group_placebos < 1:
        raise ValueError("max_group_placebos deve ser >= 1 ou None")
    if assignment_mechanism not in {"observational", "randomized"}:
        raise ValueError("assignment_mechanism deve ser 'observational' ou 'randomized'")
    if max_exhaustive_placebos < 1:
        raise ValueError("max_exhaustive_placebos deve ser >= 1")
    if max_pre_rmspe_multiple is not None and max_pre_rmspe_multiple <= 0:
        raise ValueError("max_pre_rmspe_multiple deve ser > 0 ou None")
    if assignment_mechanism == "randomized" and max_pre_rmspe_multiple is not None:
        raise ValueError(
            "filtrar alocações por pre-RMSPE quebra a simetria da randomization inference"
        )

    fit_kwargs = _fit_kwargs_for_group(fit_fn, fit_kwargs or {}, n_treated_units)
    y_pre = np.asarray(y_pre, float)
    y_post = np.asarray(y_post, float)
    Y_donors_pre = np.asarray(Y_donors_pre, float)
    Y_donors_post = np.asarray(Y_donors_post, float)
    J = len(donor_names)
    if Y_donors_pre.ndim != 2 or Y_donors_post.ndim != 2:
        raise ValueError("matrizes de doadoras devem ser bidimensionais")
    if Y_donors_pre.shape != (len(y_pre), J):
        raise ValueError("shape de Y_donors_pre incompatível com y_pre/donor_names")
    if Y_donors_post.shape != (len(y_post), J):
        raise ValueError("shape de Y_donors_post incompatível com y_post/donor_names")
    if len(set(donor_names)) != J:
        raise ValueError("donor_names contém duplicatas")
    arrays = (y_pre, y_post, Y_donors_pre, Y_donors_post)
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("outcomes contêm NaN ou infinito")

    rng = np.random.default_rng(seed)
    if assignment_mechanism == "randomized":
        if Y_treated_pre is None or Y_treated_post is None:
            if n_treated_units != 1:
                raise ValueError(
                    "randomization inference com múltiplas tratadas exige "
                    "Y_treated_pre e Y_treated_post individuais"
                )
            Y_treated_pre = y_pre[:, None]
            Y_treated_post = y_post[:, None]
        Y_treated_pre = np.asarray(Y_treated_pre, float)
        Y_treated_post = np.asarray(Y_treated_post, float)
        expected_pre_shape = (len(y_pre), n_treated_units)
        expected_post_shape = (len(y_post), n_treated_units)
        if Y_treated_pre.shape != expected_pre_shape:
            raise ValueError(
                f"Y_treated_pre deve ter shape {expected_pre_shape}, "
                f"recebeu {Y_treated_pre.shape}"
            )
        if Y_treated_post.shape != expected_post_shape:
            raise ValueError(
                f"Y_treated_post deve ter shape {expected_post_shape}, "
                f"recebeu {Y_treated_post.shape}"
            )
        if not np.isfinite(Y_treated_pre).all() or not np.isfinite(Y_treated_post).all():
            raise ValueError("trajetórias tratadas contêm NaN ou infinito")
        if not np.allclose(Y_treated_pre.mean(axis=1), y_pre) or not np.allclose(
            Y_treated_post.mean(axis=1), y_post
        ):
            raise ValueError(
                "y_pre/y_post devem ser a média das trajetórias tratadas para "
                "randomization inference"
            )
        names_t = list(treated_names or [f"treated_{i}" for i in range(n_treated_units)])
        if len(names_t) != n_treated_units or len(set(names_t)) != len(names_t):
            raise ValueError("treated_names deve conter um nome único por unidade tratada")
        if set(names_t) & set(donor_names):
            raise ValueError("treated_names e donor_names se sobrepõem")
        placebo_pre = np.column_stack([Y_treated_pre, Y_donors_pre])
        placebo_post = np.column_stack([Y_treated_post, Y_donors_post])
        placebo_names = names_t + list(donor_names)
        observed_group = tuple(range(n_treated_units))
        groups, sampled_groups = _randomization_groups(
            placebo_pre.shape[1], n_treated_units, observed_group,
            max_group_placebos, rng,
            max_exhaustive_placebos=max_exhaustive_placebos,
        )
    else:
        placebo_pre = Y_donors_pre
        placebo_post = Y_donors_post
        placebo_names = list(donor_names)
        groups, sampled_groups = _placebo_groups(
            J, n_treated_units, max_group_placebos, rng,
            max_exhaustive_placebos=max_exhaustive_placebos,
        )

    # fit real
    try:
        real = fit_fn(y_pre, Y_donors_pre, y_post, Y_donors_post, donor_names, **fit_kwargs)
    except Exception as exc:
        return PlaceboInference(
            p_value=np.nan, p_value_att=np.nan, p_value_att_onesided=np.nan,
            treated_stat=np.nan, placebo_stats=[], treated_att=np.nan,
            placebo_atts=[], n_placebos=0, note=f"Ajuste real falhou: {exc}",
            real_fit=None,
        )
    if not real.success:
        return PlaceboInference(
            p_value=np.nan, p_value_att=np.nan, p_value_att_onesided=np.nan,
            treated_stat=np.nan, placebo_stats=[], treated_att=np.nan,
            placebo_atts=[], n_placebos=0,
            note="Ajuste real falhou; rank placebo não é reportado", real_fit=real,
        )
    pre_r = _rmspe(y_pre, real.y_synth_pre)
    post_r = _rmspe(y_post, real.y_synth_post)
    treated_stat = post_r / max(pre_r, 1e-12)
    treated_att = real.att_pct

    placebo_stats, placebo_atts = [], []
    n_attempted_assignments = len(groups)
    n_failed_assignments = 0
    n_filtered_assignments = 0
    for group in groups:
        group_idx = np.asarray(group, dtype=int)
        yj_pre = placebo_pre[:, group_idx].mean(axis=1)
        yj_post = placebo_post[:, group_idx].mean(axis=1)
        mask = np.ones(placebo_pre.shape[1], dtype=bool)
        mask[group_idx] = False
        names_j = [placebo_names[k] for k in range(len(placebo_names)) if mask[k]]
        if mask.sum() < 2:
            n_failed_assignments += 1
            continue
        try:
            pf = fit_fn(yj_pre, placebo_pre[:, mask], yj_post, placebo_post[:, mask],
                        names_j, **fit_kwargs)
        except Exception:
            n_failed_assignments += 1
            continue
        if not pf.success:
            n_failed_assignments += 1
            continue
        ppre = _rmspe(yj_pre, pf.y_synth_pre)
        if not np.isfinite(ppre):
            n_failed_assignments += 1
            continue
        if max_pre_rmspe_multiple is not None and ppre > max_pre_rmspe_multiple * pre_r:
            n_filtered_assignments += 1
            continue
        ppost = _rmspe(yj_post, pf.y_synth_post)
        placebo_stat = ppost / max(ppre, 1e-12)
        if not np.isfinite(placebo_stat):
            n_failed_assignments += 1
            continue
        placebo_stats.append(placebo_stat)
        placebo_atts.append(pf.att_pct if np.isfinite(pf.att_pct) else np.nan)

    note_parts = []
    if assignment_mechanism == "randomized":
        kind = "amostradas" if sampled_groups else "exaustivas"
        note_parts.append(
            "alocações simétricas no universo tratadas+doadoras "
            f"({kind}; {len(groups)} alternativas)"
        )
    elif n_treated_units > 1:
        kind = "amostrados" if sampled_groups else "exaustivos"
        note_parts.append(
            f"placebos em grupos de {n_treated_units} ({kind}; {len(groups)} candidatos)"
        )
    if n_failed_assignments:
        note_parts.append(
            f"{n_failed_assignments}/{n_attempted_assignments} alocações candidatas "
            "falharam no ajuste/estatística"
        )
    if n_filtered_assignments:
        note_parts.append(
            f"{n_filtered_assignments}/{n_attempted_assignments} placebos observacionais "
            "foram removidos pelo filtro de pré-RMSPE"
        )
    if len(placebo_stats) < min_placebos:
        note_parts.append(
            f"AVISO: apenas {len(placebo_stats)} placebos válidos "
            f"(< {min_placebos}); p-value tem granularidade grosseira "
            f"(mínimo possível = 1/{len(placebo_stats) + 1})."
        )
    incomplete_randomization_reference = (
        assignment_mechanism == "randomized"
        and (n_failed_assignments > 0 or len(placebo_stats) != n_attempted_assignments)
    )
    enough_placebos = len(placebo_stats) >= min_placebos
    valid_for_decision = enough_placebos and (
        (assignment_mechanism == "randomized" and not incomplete_randomization_reference)
        or (assignment_mechanism == "observational" and calibrated)
    )
    if incomplete_randomization_reference:
        note_parts.append(
            "inferência por randomização inválida: a estatística não foi definida "
            "para todas as alocações amostradas/enumeradas"
        )
    if assignment_mechanism == "observational" and not valid_for_decision:
        note_parts.append(
            "rank placebo observacional: diagnóstico, não p-value calibrado; "
            "requer alocação aleatória ou calibração do procedimento completo"
        )
    inference_basis = (
        "incomplete_randomization_reference" if incomplete_randomization_reference
        else "randomization_inference" if assignment_mechanism == "randomized"
        else "calibrated_procedure" if calibrated
        else "observational_placebo_rank"
    )
    note = " | ".join(note_parts)

    if incomplete_randomization_reference:
        return PlaceboInference(
            p_value=np.nan, p_value_att=np.nan, p_value_att_onesided=np.nan,
            treated_stat=float(treated_stat), placebo_stats=placebo_stats,
            treated_att=float(treated_att), placebo_atts=placebo_atts,
            n_placebos=len(placebo_stats), note=note, real_fit=real,
            treated_att_abs=float(real.att), valid_for_decision=False,
            inference_basis=inference_basis, sampled_placebos=sampled_groups,
            n_attempted_assignments=n_attempted_assignments,
            n_failed_assignments=n_failed_assignments,
            n_filtered_assignments=n_filtered_assignments,
        )

    # p-values de permutação (incluem a tratada no denominador — Abadie)
    n = len(placebo_stats)
    if n == 0:
        return PlaceboInference(
            p_value=np.nan, p_value_att=np.nan, p_value_att_onesided=np.nan,
            treated_stat=treated_stat, placebo_stats=[], treated_att=treated_att,
            placebo_atts=[], n_placebos=0, note="Nenhum placebo válido",
            real_fit=real, treated_att_abs=float(real.att),
            valid_for_decision=False, inference_basis=inference_basis,
            sampled_placebos=sampled_groups,
            n_attempted_assignments=n_attempted_assignments,
            n_failed_assignments=n_failed_assignments,
            n_filtered_assignments=n_filtered_assignments,
        )

    p_rmspe = (1 + sum(s >= treated_stat for s in placebo_stats)) / (n + 1)

    att_pcts = np.array([a for a in placebo_atts if np.isfinite(a)])
    treated_att_pct = real.att_pct
    complete_att_reference = (
        assignment_mechanism != "randomized" or len(att_pcts) == n
    )
    if complete_att_reference and len(att_pcts) and np.isfinite(treated_att_pct):
        p_att = (1 + np.sum(np.abs(att_pcts) >= abs(treated_att_pct))) / (len(att_pcts) + 1)
        p_att_neg = (1 + np.sum(att_pcts <= treated_att_pct)) / (len(att_pcts) + 1)
    else:
        p_att, p_att_neg = np.nan, np.nan

    return PlaceboInference(
        p_value=float(p_rmspe), p_value_att=float(p_att),
        p_value_att_onesided=float(p_att_neg),
        treated_stat=float(treated_stat), placebo_stats=placebo_stats,
        treated_att=float(treated_att), placebo_atts=placebo_atts,
        n_placebos=n, note=note, real_fit=real, treated_att_abs=float(real.att),
        valid_for_decision=valid_for_decision, inference_basis=inference_basis,
        sampled_placebos=sampled_groups,
        n_attempted_assignments=n_attempted_assignments,
        n_failed_assignments=n_failed_assignments,
        n_filtered_assignments=n_filtered_assignments,
    )
