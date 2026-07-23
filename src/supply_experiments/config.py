"""Configuração centralizada — um só lugar para os defaults que hoje ficam
espalhados como kwargs individuais em `design.power`, `design.spillover`,
`reporting.analyze_experiment` e `spark_io`.

`RunConfig` é aditivo: nenhuma função passa a EXIGIR um `RunConfig` — os
kwargs individuais continuam funcionando (e são a fonte de verdade dos
defaults). O objetivo é permitir que um projeto declare a configuração uma
vez e a reutilize entre notebooks/scripts, em vez de repetir `alpha=0.10`,
`radius_km=40.0` etc. em cada chamada.

    cfg = RunConfig(alpha=0.05, radius_km=30.0)
    analyze_experiment(panel, ..., alpha=cfg.alpha, agreement_tol=cfg.agreement_tol)
    spillover_exclusions(treated, candidates, coords, radius_km=cfg.radius_km)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from supply_experiments.design.spillover import EligibilityCriteria


@dataclass
class RunConfig:
    # inferência / decisão
    alpha: float = 0.10             # ver README "Key statistical design decisions"
    fdr_q: float = 0.10             # FDR dos guardrails (Benjamini-Hochberg)
    agreement_tol: float = 0.05     # spread máximo de ATT para veredito CONCORDANTE
    min_donors: int = 9             # mínimo de doadoras exigido no pré-registro a α=0.10

    # design geográfico
    radius_km: float = 40.0         # raio de exclusão por spillover
    eligibility: EligibilityCriteria = field(default_factory=EligibilityCriteria)

    # calendário
    holidays: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha deve estar em (0, 1): {self.alpha}")
        if not (0.0 < self.fdr_q < 1.0):
            raise ValueError(f"fdr_q deve estar em (0, 1): {self.fdr_q}")
