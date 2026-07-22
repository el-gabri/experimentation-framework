# supply_experiments — Geo-Experimentation Framework

A city-level (geo) experimentation framework with measurable statistical rigor:
a tested Python library, thin Databricks notebook clients, an experiment
registry with pre-registration as a contract, and A/A calibration as a
permanent validity certificate.

> **Status:** internal proof of concept, made public for reference. Data
> adapters (`spark_io.py`) use placeholder table names — point them to your own
> environment. Everything else runs offline on plain pandas/numpy.

## Why this exists

Measuring the impact of an intervention rolled out in a handful of cities is
statistically hostile territory: few treated units, strong day-of-week and
seasonal structure, autocorrelated daily series, and donor pools contaminated
by spillover. Naive approaches — e.g. a classical t-test on an OLS regression
over aggregated treated-vs-control series — ignore autocorrelation and deliver
false-positive rates far above the nominal α (43% at α = 0.05 in our A/A
benchmark below). This framework is built so that every reported p-value means
what it claims to mean.

Core principles:

1. **Measurable validity** — every estimator must pass A/A calibration before use.
2. **Design ≠ estimation ≠ decision**, with explicit contracts (registry + pre-registration).
3. **Permutation / conformal inference by default** — the only honest options with 1–3 treated units.
4. **Tested library, thin notebooks** — one implementation, no copy-paste divergence.
5. **Triangulation** — DiD, SCM, ASCM and SDID side by side, with an explicit agreement verdict.

## Calibration certificate (A/A, synthetic panel: 40 cities × 430 days)

```
NAIVE BASELINE (aggregated OLS, classical t-test):
  FPR @ α=0.05: 43.3%   (expected: 5%)   ← nearly half of all "significant" results are false
  FPR @ α=0.10: 49.2%   (expected: 10%)

THIS FRAMEWORK (SCM + in-space permutation):
  FPR @ α=0.10:  7.5%   95% CI [3.5%, 13.8%]   ← calibrated
  KS p-value (p-value uniformity): 0.486
  Median placebo ATT bias: +0.04%

POWER CURVE (2 treated, 18 donors, 35 days, SCM):
  δ=3% → 36% | δ=5% → 76% | δ=8% → 88% | δ=12% → 100%   (MDE@80% = 8%)
```

Reproduce with `python calibration_certificate.py`. The `05_aa_calibration`
notebook runs the same procedure **on real data** and persists the certificate —
run it after any estimator change (a statistical regression test).

## Architecture

```
src/supply_experiments/
├── panel.py                  # CityPanel (outcome + num/den for ratio KPIs), ExperimentWindow
├── estimators/
│   ├── scm.py                # Abadie SCM; simplex via FISTA + exact projection (Duchi 2008)
│   ├── ascm.py               # Ridge-augmented SCM (Ben-Michael, Feller & Rothstein 2021)
│   ├── sdid.py               # Synthetic DiD (Arkhangelsky et al. 2021), ζ + Frank-Wolfe
│   └── did.py                # City-level panel DiD, within-FE, normalized scale
├── inference/
│   ├── permutation.py        # post/pre RMSPE-ratio p-value (Abadie) — primary inference
│   ├── conformal.py          # CIs by test inversion (Chernozhukov, Wüthrich & Zhu 2021)
│   └── bootstrap.py          # Restricted wild cluster bootstrap-t; Webb weights if G<12
├── design/
│   ├── power.py              # simulation-based power on historical windows; MDE@80%
│   ├── control_selection.py  # greedy+swaps on train split; F-test only on holdout
│   ├── treated_selection.py  # ranks candidate treated sets by MDE (measurability-first design)
│   └── spillover.py          # radius/adjacency exclusion; eligibility (zero-runs, CV)
├── calibration/aa.py         # A/A runner: FPR + Clopper-Pearson CI + KS + bias
├── synthetic.py              # synthetic panel generator (tests + calibration_certificate.py)
├── config.py                 # RunConfig: centralized defaults (alpha, spillover, eligibility)
├── metrics.py                # ratio DiD with delta method (weekly blocks)
├── reporting.py              # analyze_experiment: triangulation + BH + anti-peeking
├── io/                        # Spark/Databricks adapters, split by concern
│   ├── tables.py             # table names + holiday calendar
│   ├── etl.py                # build_orders_base, load_city_panel
│   └── registry.py           # ExperimentRecord, pre-registration gate, registry queries
└── spark_io.py               # back-compat re-export of `io.*` (existing notebooks import this)

notebooks/  (thin clients, ~30 lines of logic each)
├── 01_etl.ipynb                       # panel + eligibility
├── 02_fixed_control_selection.ipynb   # fixed control with temporal holdout
├── 03_design_experiment.ipynb         # spillover + POWER GATE + pre-registration
├── 04_analyze_experiment.ipynb        # triangulation + persistence
└── 05_aa_calibration.ipynb            # recurring certificate on real data

tests/test_core.py            # 20 tests, incl. effect recovery and statistical calibration
```

## Experiment lifecycle

1. **Design** (`03_design_experiment`): proposed treated cities → donor pool
   cleaned of spillover → `power_analysis` estimates the MDE → `save_experiment`
   **refuses** the design if MDE > expected effect, the hypothesis is empty, the
   decision rule is empty, or there are fewer than 8 donors.
2. **Execution**: the intervention runs; nobody analyzes (anti-peeking raises
   `ValueError`).
3. **Analysis** (`04_analyze_experiment`): SCM + ASCM + SDID with permutation
   inference, conformal CIs, optional panel DiD with wild cluster bootstrap,
   delta-method guardrails with BH correction, and a triangulation verdict
   (CONCORDANT / PARTIAL / DIVERGENT).
4. **Decision**: made against the pre-registered `decision_rule` — never ad hoc.

## Key statistical design decisions

- **Default α = 0.10 for permutation inference**: with J donors the minimum
  attainable p-value is 1/(J+1); with 15 donors that is 0.0625 — α = 0.05 would
  require ≥ 20 donors. Reports warn when the donor pool limits granularity.
- **RMSPE-ratio as the primary statistic** (robust to placebos with poor
  pre-period fit); p(|ATT|) is reported for transparency.
- **City-level DiD with outcomes normalized by the pre-period mean**: without
  this, within-FE removes level but not scale, and τ is dominated by the
  largest cities.
- **FISTA with exact simplex projection instead of SLSQP**: SLSQP declares
  success with an objective ~20× worse than optimal on instances with donors of
  very different scales (a failure mode caught by the test suite).
- **Ratio KPIs via ratio-of-sums + delta method on weekly blocks** — never OLS
  on the daily ratio, whose variance explodes on small-denominator days.
- **Control selection with temporal holdout**: the parallel-trends test is an
  acceptance criterion on held-out data, never part of the optimization score
  (no pre-testing contamination).

## Getting started

```bash
pip install -e ".[dev]"
pytest tests/ -q -m "not slow"         # fast unit tests
pytest tests/ -q -m slow               # statistical calibration (effect recovery, A/A FPR)
python calibration_certificate.py      # reproduce the A/A certificate
```

Optional extras: `pip install -e ".[spark]"` for the Databricks/Spark adapters
in `spark_io.py` (not needed to run the core library or the test suite).

The library core (`panel`, `estimators`, `inference`, `design`, `calibration`,
`metrics`, `reporting`) has no Spark dependency and runs anywhere. Spark is
imported lazily inside `spark_io.py` only.

## References

Abadie, Diamond & Hainmueller (2010, JASA); Abadie (2021, JEL); Arkhangelsky et
al. (2021, AER); Ben-Michael, Feller & Rothstein (2021, JASA); Chernozhukov,
Wüthrich & Zhu (2021, JASA); Cameron, Gelbach & Miller (2008, REStat);
MacKinnon & Webb (2018); Duchi et al. (2008, ICML).
