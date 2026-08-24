"""Design and audit components for small-market geo experiments.

The package provides estimators and diagnostics; it does not infer causal
identification from an arbitrary panel. See ``docs/VALIDITY.md`` for each
method's estimand, assumptions, deviations, and unsupported regimes.
"""

from supply_experiments._version import IMPLEMENTATION_VERSION
from supply_experiments.calibration.aa import AACalibration, run_aa_calibration
from supply_experiments.config import RunConfig
from supply_experiments.design.power import minimum_detectable_effect, power_analysis
from supply_experiments.design.spec import DecisionRule, DesignSpec
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
from supply_experiments.inference.bootstrap import (
    UnsupportedFewTreatedClustersError,
    wild_cluster_bootstrap,
)
from supply_experiments.inference.conformal import conformal_inference
from supply_experiments.inference.permutation import placebo_inference
from supply_experiments.panel import CityPanel, ExperimentWindow, PanelSlice
from supply_experiments.reporting import ExperimentReport, analyze_experiment
from supply_experiments.synthetic import make_synthetic_panel

__version__ = IMPLEMENTATION_VERSION

__all__ = [
    "CityPanel", "ExperimentWindow", "PanelSlice",
    "DesignSpec", "DecisionRule",
    "ESTIMATORS", "fit_scm", "fit_ascm", "fit_sdid", "fit_did_panel",
    "placebo_inference", "conformal_inference", "wild_cluster_bootstrap",
    "power_analysis", "minimum_detectable_effect",
    "recommend_treated_sets", "recommendations_frame", "TreatedRecommendation",
    "run_aa_calibration", "AACalibration", "RunConfig",
    "UnsupportedFewTreatedClustersError",
    "analyze_experiment", "ExperimentReport",
    "make_synthetic_panel",
]
