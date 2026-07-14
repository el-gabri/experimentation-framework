"""
supply_experiments — Framework de experimentação geográfica de supply (iFood Groceries).

Princípios:
  1. Validade mensurável: todo estimador passa por calibração A/A antes de uso.
  2. Design != estimação != decisão, com contratos explícitos (registry v2).
  3. Inferência por permutação/conformal como padrão (poucas unidades tratadas).
  4. Biblioteca testada; notebooks são clientes finos.
  5. Triangulação: DiD + SCM + ASCM + SDID lado a lado.

Camadas:
  - panel:        estruturas de painel (pandas puro, testável offline)
  - estimators:   did, scm, ascm, sdid (implementações fiéis aos papers)
  - inference:    permutation (Abadie), conformal (CWZ 2021), wild cluster bootstrap
  - design:       power (simulação), control_selection (train/holdout), spillover
  - calibration:  runner A/A e curvas de poder
  - metrics:      método delta para métricas de razão
  - reporting:    relatório multi-estimador + correção BH
  - spark_io:     adapters Databricks (orders base, registry v2, results) — import guardado
"""

from supply_experiments.panel import CityPanel, ExperimentWindow
from supply_experiments.estimators.did import fit_did_panel
from supply_experiments.estimators.scm import fit_scm
from supply_experiments.estimators.ascm import fit_ascm
from supply_experiments.estimators.sdid import fit_sdid
from supply_experiments.inference.permutation import placebo_inference
from supply_experiments.inference.conformal import conformal_inference
from supply_experiments.inference.bootstrap import wild_cluster_bootstrap
from supply_experiments.design.power import power_analysis, minimum_detectable_effect
from supply_experiments.calibration.aa import run_aa_calibration
from supply_experiments.reporting import analyze_experiment, ExperimentReport

__version__ = "1.0.2"
