"""Compatibility container for reusable non-design defaults.

``RunConfig`` is mutable and is not a preregistration contract. Use the frozen
``DesignSpec`` for every input that must be bound across power, calibration,
registry approval, and final analysis.

    cfg = RunConfig(alpha=0.05, radius_km=30.0)
    analyze_experiment(panel, ..., alpha=cfg.alpha, agreement_tol=cfg.agreement_tol)
    spillover_exclusions(treated, candidates, coords, radius_km=cfg.radius_km)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from supply_experiments.design.spillover import EligibilityCriteria


@dataclass
class RunConfig:
    # inferência / decisão
    alpha: float = 0.10             # ver README "Key statistical design decisions"
    fdr_q: float = 0.10             # FDR dos guardrails (Benjamini-Hochberg)
    agreement_tol: float = 0.05     # spread máximo de ATT para veredito CONCORDANTE
    min_donors: int = 9             # mínimo de doadoras exigido no pré-registro a α=0.10
    max_group_placebos: Optional[int] = 30
    max_pre_rmspe: Optional[float] = 0.10

    # design geográfico
    radius_km: float = 40.0         # raio de exclusão por spillover
    eligibility: EligibilityCriteria = field(default_factory=EligibilityCriteria)

    # calendário
    holidays: List[str] = field(default_factory=list)
    anticipation_days: int = 0

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha deve estar em (0, 1): {self.alpha}")
        if not (0.0 < self.fdr_q < 1.0):
            raise ValueError(f"fdr_q deve estar em (0, 1): {self.fdr_q}")
        if self.agreement_tol <= 0:
            raise ValueError("agreement_tol deve ser > 0")
        if self.min_donors < 2:
            raise ValueError("min_donors deve ser >= 2")
        if self.max_group_placebos is not None and self.max_group_placebos < 1:
            raise ValueError("max_group_placebos deve ser >= 1 ou None")
        if self.max_pre_rmspe is not None and self.max_pre_rmspe <= 0:
            raise ValueError("max_pre_rmspe deve ser > 0 ou None")
        if self.radius_km < 0:
            raise ValueError("radius_km deve ser >= 0")
        if self.anticipation_days < 0:
            raise ValueError("anticipation_days deve ser >= 0")
