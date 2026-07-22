"""Gerador de painel sintético — DGP com fator comum + sazonalidade + AR(1).

Promovido de `tests/test_core.py` para módulo de primeira classe: é o dataset
de exemplo do repositório (usado por `calibration_certificate.py`, pelos
testes de calibração estatística, e por qualquer tutorial/notebook que queira
rodar o pipeline ponta-a-ponta sem depender de dados reais).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from supply_experiments.panel import CityPanel


def make_synthetic_panel(
    n_cities: int = 30,
    n_days: int = 400,
    seed: int = 0,
    treat_effect: float = 0.0,
    treated: list | None = None,
    treat_start_idx: int | None = None,
) -> CityPanel:
    """Painel cidade x dia sintético: fator comum + DOW + heterogeneidade + AR(1).

    Se `treat_effect != 0`, aplica um choque multiplicativo constante nas
    cidades `treated` a partir de `treat_start_idx` — usado para testar
    recuperação de efeito conhecido (ver tests/test_core.py) e para o
    certificado de calibração A/A (`calibration_certificate.py`).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)

    common = 1.0 + 0.001 * t + 0.10 * np.sin(2 * np.pi * t / 365)
    dow = np.array([1.0, 0.95, 0.95, 1.0, 1.1, 1.3, 1.25])[idx.dayofweek]

    data = {}
    for i in range(n_cities):
        base = np.exp(rng.normal(10.5, 0.8))
        loading = rng.uniform(0.7, 1.3)
        # AR(1) multiplicativo
        eps = np.zeros(n_days)
        for k in range(1, n_days):
            eps[k] = 0.55 * eps[k - 1] + rng.normal(0, 0.05)
        y = base * (common**loading) * dow * np.exp(eps)
        data[f"CITY_{i:02d}"] = y
    df = pd.DataFrame(data, index=idx)

    if treat_effect != 0.0 and treated and treat_start_idx is not None:
        for c in treated:
            df.iloc[treat_start_idx:, df.columns.get_loc(c)] *= 1.0 + treat_effect

    orders = (df / 50.0).round()
    rupt = orders * 0.04 + rng.normal(0, 0.5, size=df.shape).clip(0)
    return CityPanel(
        outcome=df,
        numerators={"rupture_rate_order": rupt},
        denominators={"rupture_rate_order": orders},
    )
