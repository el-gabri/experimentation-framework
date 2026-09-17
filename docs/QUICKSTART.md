# Plain-Python quickstart

This guide exercises the pandas/numpy core. It does not require Spark or Databricks.
The example is deliberately single-treated and does not use conformal inference; see
[VALIDITY.md](VALIDITY.md) before extending it to a decision workflow.

## 1. Install

From a source checkout:

```bash
python -m pip install -e ".[dev]"
```

## 2. Run a deterministic synthetic example

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

print(f"fit succeeded: {fit.success}")
print(f"estimated ATT: {fit.att_pct:+.1%}")
print(f"pre-period RMSPE: {fit.pre_rmspe:.1%}")
print(f"pre-period correlation: {fit.pre_correlation:.3f}")
print(f"valid placebo fits: {placebos.n_placebos}")
print(f"in-space placebo rank: {placebos.p_value:.3f}")
print(f"inference basis: {placebos.inference_basis}")
print(f"decision-valid inference: {placebos.valid_for_decision}")
```

This is a software demonstration on a known synthetic effect. It is not evidence that
the same estimator or rank is valid for a real intervention.

## 3. Build a real `CityPanel`

The primary outcome is a wide `pandas.DataFrame`:

- index: unique, sorted daily `DatetimeIndex`, one observation per local calendar day
  at a consistent local time (timezone preserved);
- columns: unique geography identifiers;
- values: finite numeric outcomes on a common scale;
- coverage: every date needed by the pre/post window.

```python
import pandas as pd

from supply_experiments import CityPanel

outcome = pd.read_parquet("city_day_outcome.parquet")
outcome["date"] = pd.to_datetime(outcome["date"])
outcome = outcome.pivot(index="date", columns="city", values="outcome").sort_index()

panel = CityPanel(outcome=outcome)
```

Do not silently replace unknown missing observations with zero. Zero filling is
appropriate only when the data producer guarantees that an absent row means zero
activity rather than data loss.

The optional Spark loader follows the same rule. Its default rejects absent city-days;
use `load_city_panel(..., assume_missing_city_days_zero=True)` only after confirming
that source completeness makes absence equivalent to zero activity. Invalid observed
GMV is always rejected. Ambiguous city names spanning multiple states must be resolved
upstream. Optional rupture telemetry with missing flags is marked unavailable; request
`require_rupture_telemetry=True` when the guardrail is required. A missing flag does not
mean an observed absence of rupture.

Ratio outcomes must be supplied as numerator/denominator pairs rather than daily
precomputed ratios:

```python
panel = CityPanel(
    outcome=outcome,
    numerators={"conversion_rate": conversions},
    denominators={"conversion_rate": eligible_visits},
)
```

## 4. Freeze the design before analysis

At minimum, record:

- treated geographies and their aggregation/weights;
- donor geographies after spillover exclusions;
- outcome and guardrails;
- treatment start/end and any anticipation window;
- estimator, inference procedure, alpha, placebo count, and seed;
- expected effect and executable decision rule;
- a versioned treatment-market selection procedure, when units were optimized;
- package version and data snapshot.

`supply_experiments.design.DesignSpec` serializes and fingerprints these design inputs,
including the minimum valid A/A runs and the KS, FPR-inflation, and invalid-run gates;
the power API checks its explicit arguments against that contract, and the Spark/Delta
registry can bind a record to the fingerprint and a calibration artifact.
`analyze_experiment(..., design_spec=spec, calibration=aa)` checks the final units,
dates, KPI, alpha, placebo cap, and anticipation window against that contract. Calls
without `design_spec` remain available for exploration but always return
`DIAGNOSTIC_ONLY`.

`recommend_treated_sets` uses conditional MDEs to rank measurable candidates. Do not
copy that screening MDE into an approval record. Confirm it with `power_analysis` and a
`selection_fn` that replays the versioned selector on each historical window.
Schema-2 power samples feasible historical starts iid with replacement. Its Wilson
interval is conditional Monte Carlo uncertainty over that empirical start distribution,
not uncertainty about future shocks or the representativeness of the historical panel.
Selector callbacks must apply one frozen policy on every call; use the supplied random
generator for stochastic tie-breaking rather than changing policy by simulation index.

## 5. Interpret outputs conservatively

- `att_pct` is a point estimate relative to the fitted counterfactual.
- `pre_rmspe` and `pre_correlation` describe pre-period fit; neither proves the
  counterfactual is valid after treatment.
- donor weights reveal concentration but not absence of hidden bias.
- an observational in-space placebo rank compares the treated statistic with donor
  placebos and is diagnostic. The randomization path is exact only for the declared
  randomized assignment over the full treated-plus-donor universe; selected designs
  require complete-procedure calibration.
- agreement among SCM, ASCM, and SDID is a sensitivity check, not independent causal
  replication.

Before a real decision, inspect the assumptions and unsupported regimes in
[VALIDITY.md](VALIDITY.md), run design-matched A/A calibration, and document any
deviation from the registered plan.
