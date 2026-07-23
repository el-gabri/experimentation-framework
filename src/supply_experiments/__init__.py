"""
supply_experiments — Framework de experimentação geográfica em nível de cidade.

Princípios:
  1. Validade mensurável: todo estimador passa por calibração A/A antes de uso.
  2. Design != estimação != decisão, com contratos explícitos (registry v2).
  3. Inferência por permutação/conformal como padrão (poucas unidades tratadas).
  4. Biblioteca testada; notebooks são clientes finos.
  5. Triangulação: DiD + SCM + ASCM + SDID lado a lado.

Camadas:
  - panel:        estruturas de painel (pandas puro, testável offline); PanelSlice
  - estimators:   did, scm, ascm, sdid (implementações fiéis aos papers); ESTIMATORS
  - inference:    permutation (Abadie), conformal (CWZ 2021), wild cluster bootstrap
  - design:       power (simulação), control_selection (train/holdout), treated_selection
                  (ranking por MDE), spillover
  - calibration:  runner A/A e curvas de poder
  - config:       RunConfig — defaults centralizados (alpha, fdr_q, spillover, elegibilidade)
  - synthetic:    gerador de painel sintético (tutoriais, testes, certificado de calibração)
  - metrics:      método delta para métricas de razão
  - reporting:    relatório multi-estimador + correção BH
  - spark_io:     adapters Databricks (orders base, registry v2, results) — import guardado
"""

from supply_experiments.calibration.aa import run_aa_calibration
from supply_experiments.config import RunConfig
from supply_experiments.design.power import minimum_detectable_effect, power_analysis
from supply_experiments.design.treated_selection import (
    TreatedRecommendation,
    recommend_treated_sets,
    recommendations_frame,
)
from supply_experiments.estimators import ESTIMATORS
from supply_experiments.estimators.ascm import fit_ascm
from supply_experiments.estimators.did import fit_did_panel
from supply_experiments.estimators.scm import fit_scm
from supply_experiments.estimators.sdid import fit_sdid
from supply_experiments.inference.bootstrap import wild_cluster_bootstrap
from supply_experiments.inference.conformal import conformal_inference
from supply_experiments.inference.permutation import placebo_inference
from supply_experiments.panel import CityPanel, ExperimentWindow, PanelSlice
from supply_experiments.reporting import ExperimentReport, analyze_experiment
from supply_experiments.synthetic import make_synthetic_panel

__version__ = "1.0.3"

__all__ = [
    "CityPanel", "ExperimentWindow", "PanelSlice",
    "ESTIMATORS", "fit_scm", "fit_ascm", "fit_sdid", "fit_did_panel",
    "placebo_inference", "conformal_inference", "wild_cluster_bootstrap",
    "power_analysis", "minimum_detectable_effect",
    "recommend_treated_sets", "recommendations_frame", "TreatedRecommendation",
    "run_aa_calibration", "RunConfig",
    "analyze_experiment", "ExperimentReport",
    "make_synthetic_panel",
]
