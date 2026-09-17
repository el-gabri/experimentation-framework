# `supply-experiments`

`supply-experiments` is an opinionated Python toolkit for designing and auditing
small-market geo experiments. It combines historical-panel design tools,
spillover-aware donor filtering, simulation-based power analysis, synthetic-control
estimators, placebo diagnostics, A/A calibration, and optional Spark/Delta adapters.

> **Project status: alpha / research prototype.** The package can support a
> disciplined workflow, but it does not make an observational geo experiment valid by
> itself. Causal interpretation still depends on design-specific assumptions such as
> unaffected donors, no anticipation, stable untreated relationships, and a defensible
> treatment-assignment or exchangeability argument. Read
> [the validity contract](https://github.com/el-gabri/experimentation-framework/blob/main/docs/VALIDITY.md)
> before using an inferential result.

## What the package provides

- A pandas/numpy core for balanced city-by-day panels.
- SCM, ridge-augmented SCM, and synthetic DiD point estimates for sensitivity
  analysis.
- In-space placebo ranks and an experimental conformal-inference path.
- Historical-window power/MDE simulation and treated-market screening.
- Radius/adjacency donor exclusions and data-eligibility checks.
- A serializable `DesignSpec`, stable design fingerprints, registry approval checks,
  and an analysis-window guard.
- An estimator-pluggable A/A runner with optional selector replay and a reproducible
  synthetic calibration artifact.
- Optional Spark/Delta adapters and thin Databricks notebook templates.

The estimator implementations are not presented as new statistical methods. The
project's intended contribution is the Python design-and-governance workflow around
them.

## Installation

Python 3.10 or newer is required.

From a source checkout:

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

The alpha release is versioned as `2.0.0a2`. After it has been published to
PyPI, install it explicitly while it remains a pre-release:

```bash
python -m pip install --pre supply-experiments==2.0.0a2
```

The core library requires only NumPy, pandas, and SciPy. Spark is optional:

```bash
python -m pip install ".[spark]"
```

## Minimal plain-Python example

This example fits a single-treated-geo SCM on deterministic synthetic data and
computes an in-space placebo rank:

```python
from supply_experiments import ExperimentWindow, fit_scm, make_synthetic_panel
from supply_experiments.inference.permutation import placebo_inference

panel = make_synthetic_panel(
    n_cities=16,
    n_days=330,
    seed=42,
    treated=["CITY_03"],
    treat_start_idx=260,
    treat_effect=0.08,
)
window = ExperimentWindow(
    start_date=panel.index[260].date(),
    end_date=panel.index[287].date(),
    pre_window_days=120,
)
treated = ["CITY_03"]
donors = [city for city in panel.cities if city not in treated][:12]
panel_slice = panel.slice_for(treated, donors, window)

fit = fit_scm(*panel_slice.as_args())
placebos = placebo_inference(
    fit_scm,
    *panel_slice.as_args(),
    n_treated_units=1,
    max_group_placebos=None,
    seed=123,
)

print(f"ATT: {fit.att_pct:+.1%}")
print(f"pre-period RMSPE: {fit.pre_rmspe:.1%}")
print(f"in-space placebo rank: {placebos.p_value:.3f}")
print(f"decision-valid inference: {placebos.valid_for_decision}")
```

The placebo rank is an exact p-value only when the treatment assignment or an
appropriate exchangeability argument justifies relabeling geographies. Otherwise,
treat it as a falsification/sensitivity diagnostic. See the
[full quickstart](https://github.com/el-gabri/experimentation-framework/blob/main/docs/QUICKSTART.md)
for the real-data contract and interpretation.

## Intended workflow

1. Define the intervention, outcome, treatment window, decision rule, and plausible
   spillover mechanism before looking at post-treatment results.
2. Build a complete daily `CityPanel`; make every missing-data decision explicit.
3. Freeze eligible treated and donor geographies after spillover exclusions.
4. Serialize a draft `DesignSpec` containing the exact estimator set, inference
   configuration, validity gates, calibration acceptance thresholds, and executable
   decision rule.
5. Estimate power and run A/A calibration on historical data by replaying that complete
   design procedure, including treatment-market selection when it was optimized. MDEs
   returned while ranking candidate markets are screening quantities, not approval gates.
6. Bind the passing calibration fingerprint to the `DesignSpec`, approve it, and store
   both before treatment starts.
7. After the registered window ends, analyze through the bound `DesignSpec`; inspect
   pre-fit, weights, placebo behavior, and estimator sensitivity.

The high-level power, registry, and analysis paths compare explicit inputs with the
stored design fingerprint and fail closed on a mismatch. External storage still has to
preserve and load the exact registry row and calibration artifact; a bare low-level
estimator call intentionally bypasses that governance layer.

## Statistical scope

The most important current boundaries are:

- SCM is an outcome-only, equal-pre-period-weight special case of classical SCM.
- ASCM tunes ridge regularization with leave-one-pre-period-out prediction and exposes
  effective-weight/extrapolation diagnostics, but has no official-package replication
  artifact yet.
- SDID implements the paper's treated-mean and regularization formulas through a
  project-specific optimizer; numerical equivalence to the official implementation is
  not claimed.
- Conformal inference is restricted to SCM and inverts a sharp constant post-effect
  hypothesis; it is not a generic interval for an unrestricted ATT. See the
  [validity contract](https://github.com/el-gabri/experimentation-framework/blob/main/docs/VALIDITY.md)
  for the method-by-method scope.
- In-space placebo ranks do not automatically become randomization p-values after
  treatment-market optimization.
- The exact randomization path permutes labels over the full treated-plus-donor
  universe. Multi-treated calls therefore require the individual treated trajectories;
  an aggregate treated series is insufficient. A failed alternative-assignment fit
  invalidates the reference distribution instead of being silently dropped.
- The ordinary panel-DiD wild-cluster bootstrap fails closed below four treated
  clusters; above that misuse threshold it remains diagnostic-only because the package
  has no design-matched size certificate. Webb weights do not solve the
  few-treated-cluster problem.
- The public A/A artifact covers 400 seeded pseudo-experiments under an explicitly
  randomized assignment over the treated-plus-donor universe. It is regression
  evidence for that randomized configuration, not evidence that observational placebo
  ranks or optimized market selection are calibrated.

## Reproducible synthetic calibration

The repository includes a deterministic synthetic calibration job:

```bash
python calibration_certificate.py --output-dir calibration-artifacts
```

The command reports randomized-label null rejection frequencies, exact binomial
uncertainty, invalid-fit rates, rank diagnostics, and a grid-based power curve for its
declared seeded configuration. It deliberately does not certify the observational
placebo path: that path requires A/A calibration of the user's complete selected
design. A null rejection frequency estimates type-I error under the simulated
generator; it is not the fraction of significant real-world findings that are false.
The reported MDE is the smallest tested effect-grid value meeting the target power,
not a precise continuous threshold. Regenerate the artifact for each release instead
of copying numbers across estimator or design changes.

CI regenerates the JSON artifact from fixed seeds and records its design parameters
and software versions. A passing artifact means that this seeded regression job met
its declared checks; it does not certify other estimators, data-generating regimes,
market-selection procedures, or decision rules.

## Development and verification

```bash
python -m pytest tests/ -q -m "not slow"
python -m pytest tests/ -q -m slow
python -m ruff check src tests calibration_certificate.py make_notebooks.py
python -m mypy src
python -m build
python -m twine check dist/*
```

CI exercises supported Python versions, checks generated-notebook determinism, runs
the slow statistical regression tests, and regenerates the synthetic calibration
artifact. Spark/Delta integration requires an external Spark environment.

## Documentation

- [Plain-Python quickstart](https://github.com/el-gabri/experimentation-framework/blob/main/docs/QUICKSTART.md)
- [Validity, assumptions, implementation deviations, and unsupported regimes](https://github.com/el-gabri/experimentation-framework/blob/main/docs/VALIDITY.md)
- [Migration notes for the breaking 2.0 alpha](https://github.com/el-gabri/experimentation-framework/blob/main/docs/MIGRATION_V2.md)
- [Maintainer release procedure](https://github.com/el-gabri/experimentation-framework/blob/main/docs/RELEASING.md)
- [Changelog](https://github.com/el-gabri/experimentation-framework/blob/main/CHANGELOG.md)
- [`calibration_certificate.py`](https://github.com/el-gabri/experimentation-framework/blob/main/calibration_certificate.py)
  for the reproducible
  synthetic artifact

## Positioning and non-goals

The closest end-to-end comparator is
[GeoLift](https://github.com/facebookincubator/GeoLift). Other relevant Python or
time-series libraries include [CausalPy](https://github.com/pymc-labs/CausalPy),
[pysyncon](https://github.com/sdfordham/pysyncon),
[SparseSC](https://github.com/microsoft/SparseSC),
[CausalImpact](https://github.com/google/CausalImpact), and
[DoubleML](https://github.com/DoubleML/doubleml-for-py).

This project is not intended to replace those estimator libraries, provide automated
causal identification, support multi-cell winner selection, or guarantee valid
inference from arbitrary observational panels. The detailed comparison is in
[VALIDITY.md](https://github.com/el-gabri/experimentation-framework/blob/main/docs/VALIDITY.md#positioning-and-non-goals).

## Primary references

- Abadie, Diamond, and Hainmueller,
  [Synthetic Control Methods for Comparative Case Studies](https://j-hai.github.io/assets/pdf/scm.pdf).
- Ben-Michael, Feller, and Rothstein,
  [The Augmented Synthetic Control Method](https://jesse-rothstein.com/wp-content/uploads/2021/07/Ben-Michael_Feller_Rothstein_Augsynth_JASA_2021.pdf).
- Arkhangelsky et al.,
  [Synthetic Difference-in-Differences](https://www.nber.org/papers/w25532).
- Chernozhukov, Wüthrich, and Zhu,
  [An Exact and Robust Conformal Inference Method for Counterfactual and Synthetic Controls](https://doi.org/10.1080/01621459.2021.1920957).
- Cameron, Gelbach, and Miller,
  [Bootstrap-Based Improvements for Inference with Clustered Errors](https://doi.org/10.1162/rest.90.3.414).

MIT licensed. See [LICENSE](https://github.com/el-gabri/experimentation-framework/blob/main/LICENSE).
