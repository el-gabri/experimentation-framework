"""Calibração A/A: o certificado de validade do framework.

Roda N experimentos nulos (tratadas e janelas sorteadas do histórico, sem
tratamento) por todo o pipeline de inferência e verifica:
  - FPR ≈ α (taxa de falsos positivos);
  - distribuição de p-values ~ Uniforme(0,1) (KS test);
  - viés mediano do ATT ≈ 0.

Deve rodar a cada mudança de estimador (teste de regressão estatístico).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from supply_experiments.design.power import simulate_once
from supply_experiments.panel import CityPanel


@dataclass
class AACalibration:
    n_runs: int
    alpha: float
    fpr: float
    fpr_ci: tuple            # IC binomial 95% do FPR
    ks_p_value: float        # H0: p-values ~ U(0,1)
    median_att_bias: float
    passed: bool
    details: pd.DataFrame


def run_aa_calibration(
    panel: CityPanel,
    eligible: Sequence[str],
    fit_fn: Callable,
    pre_days: int,
    post_days: int,
    n_runs: int = 200,
    n_treated: int = 2,
    n_donors: int = 15,
    alpha: float = 0.10,
    seed: int = 123,
    fit_kwargs: Optional[dict] = None,
    min_valid_runs: int = 100,
    min_ks_p_value: float = 0.01,
    max_group_placebos: Optional[int] = 30,
    permutation_seed: Optional[int] = 123,
) -> AACalibration:
    from scipy.stats import beta as beta_dist
    from scipy.stats import kstest

    rng = np.random.default_rng(seed)
    eligible = [c for c in eligible if c in panel.cities]
    T = len(panel.index)
    if n_treated < 1:
        raise ValueError("n_treated deve ser >= 1")
    if n_donors < 2:
        raise ValueError("n_donors deve ser >= 2")
    if len(eligible) < n_treated + n_donors:
        raise ValueError(
            f"Elegíveis insuficientes: precisa de {n_treated + n_donors}, "
            f"recebeu {len(eligible)}"
        )
    if T < pre_days + post_days:
        raise ValueError(f"Histórico insuficiente: precisa de >= {pre_days + post_days} dias")

    rows: List[dict] = []
    for run in range(n_runs):
        cities = rng.choice(eligible, size=n_treated + n_donors, replace=False).tolist()
        treated, donors = cities[:n_treated], cities[n_treated:]
        start = int(rng.integers(pre_days, T - post_days))
        try:
            r = simulate_once(panel, treated, donors, fit_fn, start,
                              pre_days, post_days, delta=0.0, alpha=alpha,
                              fit_kwargs=fit_kwargs,
                              max_group_placebos=max_group_placebos,
                              permutation_seed=permutation_seed)
        except Exception as e:  # pragma: no cover
            rows.append({"run": run, "p_value": np.nan, "reject": False, "error": str(e)})
            continue
        r["run"] = run
        rows.append(r)

    df = pd.DataFrame(rows)
    valid = df[np.isfinite(df["p_value"])]
    n = len(valid)
    fpr = float(valid["reject"].mean()) if n else np.nan
    k = int(valid["reject"].sum())
    # IC binomial exato (Clopper-Pearson)
    lo = float(beta_dist.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta_dist.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    ks_p = float(kstest(valid["p_value"], "uniform").pvalue) if n >= 20 else np.nan
    med_bias = float(valid["att_pct_placebo_med"].median()) if n else np.nan

    # com poucas runs o IC binomial é largo e "passa" mesmo com FPR mal
    # calibrado; exige-se também n mínimo e uniformidade dos p-values (KS).
    passed = bool(
        n >= min_valid_runs
        and lo <= alpha <= hi
        and np.isfinite(ks_p)
        and ks_p >= min_ks_p_value
    )
    return AACalibration(n_runs=n, alpha=alpha, fpr=fpr, fpr_ci=(lo, hi),
                         ks_p_value=ks_p, median_att_bias=med_bias,
                         passed=passed, details=df)
