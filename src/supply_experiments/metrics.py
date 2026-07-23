"""Métricas de razão (rupture_rate etc.) via razão de somas + método delta.

Nunca modele a razão diária diretamente com OLS: dias com denominador pequeno
têm variância enorme e viesam o efeito. O caminho correto:
  - efeito em razão de somas: R = Σnum/Σden por grupo/período;
  - DiD em razões: Δ = (R_t,post − R_t,pre) − (R_c,post − R_c,pre);
  - variância via método delta com blocos temporais (autocorrelação).
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
) -> RatioDiD:
    from scipy.stats import norm

    rpt, vpt = _ratio_and_var(num_t[pre_mask], den_t[pre_mask], block_days)
    rot, vot = _ratio_and_var(num_t[post_mask], den_t[post_mask], block_days)
    rpc, vpc = _ratio_and_var(num_c[pre_mask], den_c[pre_mask], block_days)
    roc, voc = _ratio_and_var(num_c[post_mask], den_c[post_mask], block_days)

    eff = (rot - rpt) - (roc - rpc)
    variances = np.array([vpt, vot, vpc, voc], dtype=float)
    var = float(np.sum(variances)) if np.isfinite(variances).all() else np.nan
    se = float(np.sqrt(var)) if var > 0 else np.nan
    z = eff / se if se and se > 0 else np.nan
    p = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
    rel = eff / rpt if rpt and abs(rpt) > 1e-12 else np.nan
    return RatioDiD(effect_abs=float(eff), effect_rel=float(rel), se=se, z=float(z),
                    p_value=p, r_pre_t=rpt, r_post_t=rot, r_pre_c=rpc, r_post_c=roc)
