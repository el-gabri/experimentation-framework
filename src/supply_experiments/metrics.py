"""Métricas de razão por razão de somas + bootstrap em blocos pareados.

Nunca modele a razão diária diretamente com OLS: dias com denominador pequeno
têm variância enorme e viesam o efeito. O caminho correto:
  - efeito em razão de somas: R = Σnum/Σden por grupo/período;
  - DiD em razões: Δ = (R_t,post − R_t,pre) − (R_c,post − R_c,pre);
  - inferência por blocos temporais conjuntos, preservando a covariância entre
    numeradores/denominadores e entre grupos no mesmo dia.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RatioDiD:
    effect_abs: float      # em pontos de taxa
    effect_rel: float      # relativo ao baseline pré tratado
    se: float
    z: float
    p_value: float
    r_pre_t: float
    r_post_t: float
    r_pre_c: float
    r_post_c: float
    ci_lower: float = np.nan
    ci_upper: float = np.nan
    n_boot: int = 0
    method: str = "paired_circular_block_bootstrap"


def _ratio_and_var(num: pd.Series, den: pd.Series, block_days: int = 7) -> tuple[float, float]:
    """Razão de somas e variância delta com agregação em blocos semanais.

    Var(R) ≈ (1/D̄²) · Var_blocos(n_b − R·d_b) / B, onde blocos reduzem o efeito
    da autocorrelação intra-semana.
    """
    if block_days < 1:
        raise ValueError("block_days deve ser >= 1")
    n = num.to_numpy(float)
    d = den.to_numpy(float)
    mask = np.isfinite(n) & np.isfinite(d)
    n, d = n[mask], d[mask]
    if d.sum() <= 0:
        return np.nan, np.nan
    R = n.sum() / d.sum()

    # Blocos que cobrem todos os dias, com o último absorvendo o resto. Com
    # menos de dois blocos não existe variância temporal estimável: retornar
    # NaN é mais seguro que inventar um bloco vazio e subestimar o SE.
    starts = list(range(0, len(n), block_days))
    B = len(starts)
    if B < 2:
        return float(R), np.nan
    nb = np.array([n[start:start + block_days].sum() for start in starts])
    db = np.array([d[start:start + block_days].sum() for start in starts])
    u = nb - R * db
    var_u = float(np.var(u, ddof=1))
    dbar = float(np.mean(db))
    var_R = var_u / (B * dbar ** 2) if dbar > 0 else np.nan
    return float(R), var_R


def ratio_did(
    num_t: pd.Series, den_t: pd.Series,
    num_c: pd.Series, den_c: pd.Series,
    pre_mask: np.ndarray, post_mask: np.ndarray,
    block_days: int = 7,
    n_boot: int = 999,
    seed: int = 2026,
    alpha: float = 0.05,
    min_blocks_per_period: int = 4,
) -> RatioDiD:
    """DiD de razões com bootstrap circular em blocos, aplicado conjuntamente.

    O mesmo índice reamostrado é aplicado às quatro séries dentro de cada
    período. Isso mantém choques comuns tratado-controle e a covariância entre
    numerador e denominador, omitidas pela soma de variâncias marginais.
    """
    if block_days < 1:
        raise ValueError("block_days deve ser >= 1")
    if n_boot < 99:
        raise ValueError("n_boot deve ser >= 99")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha deve estar em (0, 1)")

    arrays = [np.asarray(x, float) for x in (num_t, den_t, num_c, den_c)]
    n = len(arrays[0])
    if any(len(x) != n for x in arrays):
        raise ValueError("todas as séries de razão devem ter o mesmo comprimento")
    pre_mask = np.asarray(pre_mask, bool)
    post_mask = np.asarray(post_mask, bool)
    if len(pre_mask) != n or len(post_mask) != n or np.any(pre_mask & post_mask):
        raise ValueError("máscaras pré/pós inválidas")
    if any(not np.isfinite(x[pre_mask | post_mask]).all() for x in arrays):
        raise ValueError("séries de razão contêm NaN ou infinito nas janelas")

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
        total = float(np.sum(denominator))
        return float(np.sum(numerator) / total) if total > 0 else np.nan

    def effect(parts: list[np.ndarray]) -> tuple[float, tuple[float, float, float, float]]:
        nt, dt, nc, dc = parts
        rpt_ = ratio(nt[pre_mask], dt[pre_mask])
        rot_ = ratio(nt[post_mask], dt[post_mask])
        rpc_ = ratio(nc[pre_mask], dc[pre_mask])
        roc_ = ratio(nc[post_mask], dc[post_mask])
        return (rot_ - rpt_) - (roc_ - rpc_), (rpt_, rot_, rpc_, roc_)

    eff, (rpt, rot, rpc, roc) = effect(arrays)
    pre_idx = np.flatnonzero(pre_mask)
    post_idx = np.flatnonzero(post_mask)
    pre_blocks = int(np.ceil(len(pre_idx) / block_days))
    post_blocks = int(np.ceil(len(post_idx) / block_days))
    boot = np.array([], dtype=float)
    if min(pre_blocks, post_blocks) >= min_blocks_per_period and np.isfinite(eff):
        rng = np.random.default_rng(seed)

        def resample_period(indices: np.ndarray) -> np.ndarray:
            starts = rng.integers(0, len(indices), size=int(np.ceil(len(indices) / block_days)))
            positions = np.concatenate([
                (start + np.arange(block_days)) % len(indices) for start in starts
            ])[:len(indices)]
            return indices[positions]

        values = []
        for _ in range(n_boot):
            sampled_pre = resample_period(pre_idx)
            sampled_post = resample_period(post_idx)
            sampled = []
            for x in arrays:
                out = np.empty(len(pre_idx) + len(post_idx), dtype=float)
                out[:len(pre_idx)] = x[sampled_pre]
                out[len(pre_idx):] = x[sampled_post]
                sampled.append(out)
            pre_b = np.zeros(len(sampled[0]), dtype=bool)
            pre_b[:len(pre_idx)] = True
            post_b = ~pre_b
            nt, dt, nc, dc = sampled
            value = (
                ratio(nt[post_b], dt[post_b]) - ratio(nt[pre_b], dt[pre_b])
                - ratio(nc[post_b], dc[post_b]) + ratio(nc[pre_b], dc[pre_b])
            )
            if np.isfinite(value):
                values.append(value)
        boot = np.asarray(values, float)

    if len(boot) >= 99:
        se = float(np.std(boot, ddof=1))
        centered = boot - float(np.mean(boot))
        p = float((1 + np.sum(np.abs(centered) >= abs(eff))) / (len(boot) + 1))
        lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2]).astype(float)
        z = eff / se if se > 0 else np.nan
    else:
        se = z = p = lo = hi = np.nan
    rel = eff / rpt if rpt and abs(rpt) > 1e-12 else np.nan
    return RatioDiD(effect_abs=float(eff), effect_rel=float(rel), se=se, z=float(z),
                    p_value=p, r_pre_t=rpt, r_post_t=rot, r_pre_c=rpc, r_post_c=roc,
                    ci_lower=float(lo), ci_upper=float(hi), n_boot=int(len(boot)))
