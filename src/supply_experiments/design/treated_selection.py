"""Recomendação de cidades tratadas — "market selection" orientado por poder.

Inverte o fluxo de design: em vez do stakeholder propor tratadas e o power gate
reprovar, o framework ranqueia os conjuntos mais MENSURÁVEIS e o stakeholder
escolhe da lista (cf. GeoLift/Meta). Mensurabilidade aqui = MDE baixo.

Pipeline (barato → caro):
  1. Filtros de negócio: estados/lista, exclusões, teto de share de GMV nacional
     (cidade grande demais contamina o benchmark e encarece o rollout).
  2. Screening por cidade: fit sintético individual com holdout temporal
     (treino 70% / holdout 30% do pré) — corta para as top_k_cities.
  3. Screening por conjunto: mesmo fit no agregado do conjunto, donor pool já
     limpo de spillover — corta para top_sets.
  4. power_analysis só nos top_sets → MDE@80% por conjunto.

CUSTO: todo o pipeline usa fit_scm (FISTA, ~30ms/fit) com doadoras capadas por
correlação, independentemente do estimador do experimento — o ranking por MDE é
estável entre estimadores e o ASCM (LOO-CV ridge) custa ~100x mais por fit. O
estimador pré-registrado entra só no power gate confirmatório do design final
(notebook 03, passo 3). `fit_fn` é aceito por compatibilidade mas o screening
sempre roda SCM; passe `power_fit_fn` para forçar outro estimador no ranking.

O resultado NÃO enviesa a inferência: a seleção usa apenas dados pré-tratamento
e o experimento ainda passa por pré-registro + análise pós-período normalmente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from supply_experiments.design.power import power_analysis
from supply_experiments.design.spillover import spillover_exclusions
from supply_experiments.estimators.scm import fit_scm
from supply_experiments.panel import CityPanel


@dataclass
class TreatedRecommendation:
    cities: List[str]
    n_donors: int
    screen_rmspe: float                 # RMSPE no holdout do screening (menor = melhor)
    mde_80: Optional[float]             # menor δ detectável com poder >= 80%
    fpr: float                          # rejeição em δ=0 (deve ≈ α)
    power_curve: Dict[float, float] = field(default_factory=dict)
    gmv_share: float = np.nan           # share do conjunto no GMV nacional do painel
    donors: List[str] = field(default_factory=list)

    def label(self) -> str:
        mde = f"{self.mde_80:.1%}" if self.mde_80 is not None else "> grid"
        return (f"{{{', '.join(self.cities)}}} | MDE@80%: {mde} | "
                f"doadoras: {self.n_donors} | share GMV: {self.gmv_share:.1%}")


def _holdout_rmspe(y: np.ndarray, Yd: np.ndarray, fit_fn: Callable,
                   train_frac: float = 0.7) -> float:
    """Fit no treino, RMSPE (normalizado pela média) no holdout — nunca otimiza
    onde avalia, mesmo racional do select_fixed_control."""
    T = len(y)
    t = int(T * train_frac)
    if t < 30 or T - t < 10:
        return np.inf
    fit = fit_fn(y[:t], Yd[:t], y[t:], Yd[t:], [f"d{j}" for j in range(Yd.shape[1])])
    if not fit.success:
        return np.inf
    resid = y[t:] - fit.y_synth_post
    mean_h = float(np.mean(y[t:])) or 1.0
    return float(np.sqrt(np.mean(resid ** 2))) / abs(mean_h)


def _sort_donors_by_corr(y: np.ndarray, panel: CityPanel, donors: Sequence[str],
                         sl: slice) -> List[str]:
    """Doadoras ordenadas por correlação (desc) com a série tratada na janela.
    Barato (O(T*J)) e garante que qualquer cap [:N] pegue as mais informativas."""
    a = y / (np.mean(y) or 1.0)
    corrs = {}
    for d in donors:
        b = panel.outcome[d].to_numpy(float)[sl]
        b = b / (np.mean(b) or 1.0)
        corrs[d] = float(np.corrcoef(a, b)[0, 1]) if np.std(b) > 1e-12 else -1.0
    return sorted(donors, key=lambda d: corrs[d], reverse=True)


def recommend_treated_sets(
    panel: CityPanel,
    stats: pd.DataFrame,
    candidates: Sequence[str],
    fit_fn: Callable,
    pre_days: int,
    post_days: int,
    n_treated: Sequence[int] = (2, 3),
    states: Optional[Sequence[str]] = None,
    must_include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    coords: Optional[Dict[str, Tuple[float, float]]] = None,
    adjacency: Optional[Dict[str, Set[str]]] = None,
    radius_km: float = 40.0,
    max_gmv_share_per_city: float = 0.05,
    min_donors: int = 10,
    top_k_cities: int = 12,
    top_sets: int = 5,
    effect_grid: Sequence[float] = (0.0, 0.02, 0.03, 0.05, 0.08),
    n_sims_per_point: int = 20,
    alpha: float = 0.10,
    seed: int = 42,
    screen_donor_cap: int = 40,
    power_fit_fn: Optional[Callable] = None,
) -> List[TreatedRecommendation]:
    """Retorna conjuntos de tratadas ranqueados por MDE (mais mensurável primeiro).

    `candidates` deve vir já filtrado por elegibilidade e cidades bloqueadas
    (tratadas/controles de experimentos ativos). `stats` é o DataFrame do
    load_city_panel (precisa de sum_gmv e, se `states` for usado, state_address).
    """
    coords = coords or {}
    must_include = [c for c in must_include]
    exclude_set = set(exclude)

    # ---- 1. filtros de negócio ------------------------------------------------
    pool = [c for c in candidates if c in panel.cities and c not in exclude_set]
    if states:
        st = set(s.upper() for s in states)
        pool = [c for c in pool
                if str(stats.loc[c, "state_address"] if c in stats.index else "")[:2].upper() in st]
    total_gmv = float(stats["sum_gmv"].sum()) or 1.0
    share = {c: float(stats.loc[c, "sum_gmv"]) / total_gmv if c in stats.index else 0.0
             for c in pool}
    pool = [c for c in pool if share[c] <= max_gmv_share_per_city or c in must_include]
    missing = [c for c in must_include if c not in pool]
    if missing:
        raise ValueError(f"must_include fora do pool (inelegível/bloqueada/filtrada): {missing}")
    if len(pool) < min_donors + max(n_treated):
        raise ValueError(f"Pool de {len(pool)} cidades é insuficiente "
                         f"(precisa >= {min_donors + max(n_treated)})")

    # janela de screening: os últimos pre_days do painel
    sl = slice(max(len(panel.index) - pre_days, 0), len(panel.index))

    # screening/ranking SEMPRE com SCM (custo ~30ms/fit); ver docstring do módulo
    screen_fn = fit_scm
    power_fn = power_fit_fn or fit_scm

    # ---- 2. screening por cidade ----------------------------------------------
    city_score: Dict[str, float] = {}
    for c in pool:
        y = panel.outcome[c].to_numpy(float)[sl]
        donors_c = _sort_donors_by_corr(y, panel, [d for d in pool if d != c],
                                        sl)[:screen_donor_cap]
        Yd = panel.outcome[donors_c].to_numpy(float)[sl]
        city_score[c] = _holdout_rmspe(y, Yd, screen_fn)
    ranked = sorted(pool, key=lambda c: city_score[c])
    shortlist = list(dict.fromkeys(must_include + ranked))[:max(top_k_cities, len(must_include))]

    # ---- 3. screening por conjunto (com spillover no donor pool) ---------------
    set_rows = []
    for n in n_treated:
        for combo in combinations(shortlist, n):
            if must_include and not set(must_include) <= set(combo):
                continue
            donors, _ = spillover_exclusions(list(combo), [c for c in pool if c not in combo],
                                             coords, radius_km=radius_km, adjacency=adjacency)
            if len(donors) < min_donors:
                continue
            y = panel.aggregate(list(combo)).to_numpy(float)[sl]
            donors = _sort_donors_by_corr(y, panel, donors, sl)  # melhores primeiro
            Yd = panel.outcome[donors[:screen_donor_cap]].to_numpy(float)[sl]
            rmspe = _holdout_rmspe(y, Yd, screen_fn)
            if np.isfinite(rmspe):
                set_rows.append((list(combo), donors, rmspe))
    if not set_rows:
        raise ValueError("Nenhum conjunto viável (spillover/min_donors muito restritivos?)")
    set_rows.sort(key=lambda r: r[2])

    # ---- 4. power analysis nos finalistas ---------------------------------------
    out: List[TreatedRecommendation] = []
    for cities, donors, rmspe in set_rows[:top_sets]:
        pw = power_analysis(panel, cities, donors[:20], power_fn,
                            pre_days=pre_days, post_days=post_days,
                            effect_grid=effect_grid,
                            n_sims_per_point=n_sims_per_point,
                            alpha=alpha, seed=seed)
        out.append(TreatedRecommendation(
            cities=cities, n_donors=min(len(donors), 20), screen_rmspe=rmspe,
            mde_80=pw.mde_80, fpr=pw.fpr, power_curve=pw.power,
            gmv_share=sum(share.get(c, 0.0) for c in cities), donors=donors[:20],
        ))
    out.sort(key=lambda r: (r.mde_80 is None, r.mde_80 if r.mde_80 is not None else np.inf,
                            r.screen_rmspe))
    return out


def recommendations_frame(recs: List[TreatedRecommendation]) -> pd.DataFrame:
    """Tabela amigável para stakeholders (uma linha por conjunto recomendado)."""
    return pd.DataFrame([{
        "rank": i + 1,
        "cidades_tratadas": ", ".join(r.cities),
        "mde_80": r.mde_80,
        "n_doadoras": r.n_donors,
        "fpr_calibracao": r.fpr,
        "share_gmv_nacional": r.gmv_share,
        "screen_rmspe": r.screen_rmspe,
    } for i, r in enumerate(recs)])
