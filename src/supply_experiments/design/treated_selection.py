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
  4. power_analysis condicional só nos top_sets → MDE de *screening* por conjunto.

CUSTO: o screening usa SCM por padrão, mas o ``fit_fn`` fornecido é o estimador
do power gate. ``screen_fit_fn`` e ``power_fit_fns`` tornam ambas as escolhas
explícitas. Selecionar mercados por mensurabilidade ainda é parte do mecanismo
de assignment; inferência por ranks espaciais precisa de randomização posterior
ou calibração que reproduza esta seleção, não apenas de dados pré-tratamento.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import date
from itertools import combinations
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from supply_experiments.design.power import power_analysis
from supply_experiments.design.spec import DecisionRule, DesignSpec
from supply_experiments.design.spillover import spillover_exclusions
from supply_experiments.estimators import ESTIMATORS
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
    power_ci: Dict[float, Tuple[float, float]] = field(default_factory=dict)
    invalid_rate: Dict[float, float] = field(default_factory=dict)
    design_fingerprint: Optional[str] = None
    design_spec_json: Optional[str] = None
    power_design_fingerprint: Optional[str] = None
    power_scope: str = "conditional_screening_only"

    def label(self) -> str:
        mde = f"{self.mde_80:.1%}" if self.mde_80 is not None else "> grid"
        return (f"{{{', '.join(self.cities)}}} | screening MDE@80%: {mde} | "
                f"doadoras: {self.n_donors} | share GMV: {self.gmv_share:.1%}")


def _holdout_rmspe(y: np.ndarray, Yd: np.ndarray, fit_fn: Callable,
                   train_frac: float = 0.7,
                   fit_kwargs: Optional[dict] = None) -> float:
    """Fit no treino, RMSPE (normalizado pela média) no holdout — nunca otimiza
    onde avalia, mesmo racional do select_fixed_control."""
    T = len(y)
    t = int(T * train_frac)
    if t < 30 or T - t < 10:
        return np.inf
    fit = fit_fn(
        y[:t], Yd[:t], y[t:], Yd[t:],
        [f"d{j}" for j in range(Yd.shape[1])], **(fit_kwargs or {}),
    )
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
    target_power: float = 0.80,
    min_valid_sims_per_point: Optional[int] = None,
    max_invalid_rate: float = 0.05,
    alpha: float = 0.10,
    seed: int = 42,
    screen_donor_cap: int = 40,
    power_fit_fn: Optional[Callable] = None,
    screen_fit_fn: Optional[Callable] = None,
    screen_fit_kwargs: Optional[dict] = None,
    power_fit_fns: Optional[Mapping[str, Callable]] = None,
    power_fit_kwargs: Optional[Mapping[str, dict]] = None,
    decision_rule: Optional[DecisionRule] = None,
    max_group_placebos: Optional[int] = 30,
    permutation_seed: Optional[int] = 123,
    window_spacing_days: Optional[int] = None,
    require_spillover_coverage: bool = False,
    assignment_mechanism: str = "observational",
    max_pre_rmspe: Optional[float] = 0.10,
    anticipation_days: int = 0,
    require_calibration: bool = True,
    calibration_fingerprint: Optional[str] = None,
    primary_kpi: str = "outcome",
    experiment_id: str = "",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    selection_procedure_id: str = "recommend_treated_sets-v1",
) -> List[TreatedRecommendation]:
    """Retorna conjuntos ranqueados por MDE condicional de screening.

    O MDE desta função não é confirmatório porque as unidades já foram escolhidas
    no mesmo painel. Antes da aprovação, rode ``power_analysis`` novamente com
    ``selection_fn`` reproduzindo ``selection_procedure_id``.

    `candidates` deve vir já filtrado por elegibilidade e cidades bloqueadas
    (tratadas/controles de experimentos ativos). `stats` é o DataFrame do
    load_city_panel (precisa de sum_gmv e, se `states` for usado, state_address).
    """
    coords = coords or {}
    if not n_treated or any(not isinstance(n, int) or n < 1 for n in n_treated):
        raise ValueError("n_treated deve conter inteiros >= 1")
    if top_k_cities < 1 or top_sets < 1:
        raise ValueError("top_k_cities e top_sets devem ser >= 1")
    if "sum_gmv" not in stats.columns:
        raise ValueError("stats deve conter a coluna sum_gmv")
    if states and "state_address" not in stats.columns:
        raise ValueError("stats deve conter state_address quando states é usado")
    must_include = [c for c in must_include]
    exclude_set = set(exclude)

    # ---- 1. filtros de negócio ------------------------------------------------
    pool = list(dict.fromkeys(
        c for c in candidates if c in panel.cities and c not in exclude_set
    ))
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
    screening_length = len(panel.index[sl])
    training_length = int(screening_length * 0.7)
    correlation_sl = slice(sl.start, sl.start + training_length)

    screen_fn = screen_fit_fn or fit_scm
    power_fn = power_fit_fn or fit_fn
    if not callable(screen_fn) or not callable(power_fn):
        raise ValueError("fit_fn, screen_fit_fn e power_fit_fn devem ser callables")
    estimator_map = dict(power_fit_fns or {
        str(getattr(power_fn, "__name__", power_fn.__class__.__name__)).removeprefix("fit_"):
            power_fn
    })
    rule = decision_rule or DecisionRule()
    estimator_kwargs = dict(power_fit_kwargs or {})
    power_min_valid = (
        min(20, n_sims_per_point)
        if min_valid_sims_per_point is None
        else min_valid_sims_per_point
    )
    selector_config_json = json.dumps(
        {
            "selector": "recommend_treated_sets",
            "n_treated": sorted(n_treated),
            "states": sorted(str(state).upper() for state in (states or ())),
            "must_include": sorted(str(city) for city in must_include),
            "exclude": sorted(str(city) for city in exclude_set),
            "radius_km": radius_km,
            "max_gmv_share_per_city": max_gmv_share_per_city,
            "min_donors": min_donors,
            "top_k_cities": top_k_cities,
            "top_sets": top_sets,
            "screen_donor_cap": screen_donor_cap,
            "require_spillover_coverage": require_spillover_coverage,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    # ---- 2. screening por cidade ----------------------------------------------
    city_score: Dict[str, float] = {}
    for c in pool:
        y = panel.outcome[c].to_numpy(float)[sl]
        y_train = panel.outcome[c].to_numpy(float)[correlation_sl]
        donors_c = _sort_donors_by_corr(
            y_train, panel, [d for d in pool if d != c], correlation_sl,
        )[:screen_donor_cap]
        Yd = panel.outcome[donors_c].to_numpy(float)[sl]
        city_score[c] = _holdout_rmspe(
            y, Yd, screen_fn, fit_kwargs=screen_fit_kwargs,
        )
    ranked = sorted(pool, key=lambda c: city_score[c])
    shortlist = list(dict.fromkeys(must_include + ranked))[:max(top_k_cities, len(must_include))]

    # ---- 3. screening por conjunto (com spillover no donor pool) ---------------
    set_rows = []
    for n in n_treated:
        for combo in combinations(shortlist, n):
            if must_include and not set(must_include) <= set(combo):
                continue
            donors, _ = spillover_exclusions(
                list(combo), [c for c in pool if c not in combo], coords,
                radius_km=radius_km, adjacency=adjacency,
                require_complete_coverage=require_spillover_coverage,
            )
            if len(donors) < min_donors:
                continue
            y = panel.aggregate(list(combo)).to_numpy(float)[sl]
            y_train = panel.aggregate(list(combo)).to_numpy(float)[correlation_sl]
            donors = _sort_donors_by_corr(
                y_train, panel, donors, correlation_sl,
            )  # melhores primeiro, usando somente o treino
            Yd = panel.outcome[donors[:screen_donor_cap]].to_numpy(float)[sl]
            rmspe = _holdout_rmspe(
                y, Yd, screen_fn, fit_kwargs=screen_fit_kwargs,
            )
            fit_ok = max_pre_rmspe is None or rmspe <= max_pre_rmspe
            if np.isfinite(rmspe) and fit_ok:
                set_rows.append((list(combo), donors, rmspe))
    if not set_rows:
        raise ValueError("Nenhum conjunto viável (spillover/min_donors muito restritivos?)")
    set_rows.sort(key=lambda r: r[2])

    # ---- 4. power analysis nos finalistas ---------------------------------------
    out: List[TreatedRecommendation] = []
    for cities, donors, rmspe in set_rows[:top_sets]:
        selected_donors = donors[:20]
        spacing = 1 if window_spacing_days is None else window_spacing_days
        candidate_spec = DesignSpec(
            treated_units=tuple(cities), donor_units=tuple(selected_donors),
            estimator_names=tuple(estimator_map), pre_days=pre_days,
            post_days=post_days, alpha=alpha,
            experiment_id=experiment_id, start_date=start_date, end_date=end_date,
            max_group_placebos=max_group_placebos, effect_grid=tuple(effect_grid),
            decision_rule=rule, primary_kpi=primary_kpi,
            assignment_mechanism=assignment_mechanism,
            selection_procedure_id=selection_procedure_id,
            eligible_units=tuple(pool),
            selector_config_json=selector_config_json,
            max_pre_rmspe=max_pre_rmspe, anticipation_days=anticipation_days,
            require_calibration=require_calibration,
            calibration_scope="exact_design" if require_calibration else "none",
            calibration_fingerprint=calibration_fingerprint,
            power_target=target_power,
            power_n_sims_per_point=n_sims_per_point,
            power_min_valid_sims_per_point=power_min_valid,
            power_max_invalid_rate=max_invalid_rate,
            seed=seed, permutation_seed=permutation_seed,
            window_spacing_days=spacing,
            estimator_config_json=json.dumps(estimator_kwargs),
        )
        # This MDE is used only to rank candidates. It conditions on the set
        # already selected above and therefore cannot authorize the final
        # design. Confirmatory power must replay ``selection_procedure_id``.
        screening_spec = replace(
            candidate_spec,
            selection_procedure_id="fixed_prespecified",
            require_calibration=False,
            calibration_scope="none",
            calibration_fingerprint=None,
        )
        uses_deployed_estimators = all(
            name in ESTIMATORS and fit is ESTIMATORS[name]
            for name, fit in estimator_map.items()
        )
        pw = power_analysis(
            panel, cities, selected_donors, power_fn,
            pre_days=pre_days, post_days=post_days,
            effect_grid=effect_grid, n_sims_per_point=n_sims_per_point,
            alpha=alpha, seed=seed, max_group_placebos=max_group_placebos,
            permutation_seed=permutation_seed, fit_fns=estimator_map,
            fit_kwargs_by_estimator=estimator_kwargs, decision_rule=rule,
            design_spec=screening_spec if uses_deployed_estimators else None,
            window_spacing_days=spacing,
            target_power=target_power,
            min_valid_sims_per_point=power_min_valid,
            max_invalid_rate=max_invalid_rate,
        )
        out.append(TreatedRecommendation(
            cities=cities, n_donors=len(selected_donors), screen_rmspe=rmspe,
            mde_80=pw.mde_80, fpr=pw.fpr, power_curve=pw.power,
            gmv_share=sum(share.get(c, 0.0) for c in cities), donors=selected_donors,
            power_ci=pw.power_ci, invalid_rate=pw.invalid_rate,
            design_fingerprint=candidate_spec.fingerprint,
            design_spec_json=candidate_spec.to_json(),
            power_design_fingerprint=pw.design_fingerprint,
        ))
    out.sort(key=lambda r: (r.mde_80 is None, r.mde_80 if r.mde_80 is not None else np.inf,
                            r.screen_rmspe))
    return out


def recommendations_frame(recs: List[TreatedRecommendation]) -> pd.DataFrame:
    """Tabela amigável para stakeholders (uma linha por conjunto recomendado)."""
    return pd.DataFrame([{
        "rank": i + 1,
        "cidades_tratadas": ", ".join(r.cities),
        "mde_80_screening_only": r.mde_80,
        "n_doadoras": r.n_donors,
        "fpr_design_simulado": r.fpr,
        "invalid_rate_max": max(r.invalid_rate.values(), default=np.nan),
        "share_gmv_nacional": r.gmv_share,
        "screen_rmspe": r.screen_rmspe,
        "design_fingerprint": r.design_fingerprint,
        "power_scope": r.power_scope,
    } for i, r in enumerate(recs)])
