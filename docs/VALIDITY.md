# Statistical validity contract

This document states what the current package computes, how that relates to source
methods, what the repository has tested, and where causal or inferential claims remain
unsupported. It is a disclosure document, not a certificate of validity for a user
dataset.

## Cross-cutting causal requirements

Every method in this package needs substantive assumptions that code cannot verify:

1. **No relevant interference.** Treatment in a selected geography must not change
   outcomes in donors through media leakage, commuting, migration, competition, or
   supply displacement. Radius and adjacency filters are design aids, not proof.
2. **No anticipation or concurrent differential shock.** The untreated potential
   outcome relationship learned before treatment must remain stable after treatment.
   `anticipation_days` can exclude a declared pre-start interval; it does not detect or
   repair anticipatory behavior.
3. **Usable donor support.** The donor pool must reproduce the treated pre-period well
   without relying on a fragile or contaminated combination.
4. **Frozen design.** Outcomes, windows, treated markets, donors, estimators,
   specifications, and decision rules must be fixed before inspecting post-treatment
   results.
5. **Inference-specific assignment assumptions.** Preserving serial dependence by
   moving whole geography trajectories is not enough; a placebo rank needs a
   randomized assignment or a credible exchangeability/conditioning argument.

Missing any one of these can invalidate a causal interpretation even when the software
returns finite estimates.

## SCM

### Estimand and implemented equation

For a treated trajectory \(y\) and donor matrix \(Y\), the package solves

\[
\hat w=\arg\min_{w\ge0,\;\mathbf 1^\top w=1}
\lVert y_{pre}-Y_{pre}w\rVert_2^2,
\qquad
\hat\tau=\frac{1}{T_{post}}\sum_t(y_t-Y_tw).
\]

`att_pct` divides the post-period mean gap by the mean synthetic post-period level.

### Source

[Abadie, Diamond, and Hainmueller (2010)](https://j-hai.github.io/assets/pdf/scm.pdf).

### Implementation scope and deviations

- This is the outcome-history special case \(X=Y_{pre}\), with equal pre-period
  importance (effectively \(V=I\)). It does not implement general predictors or
  data-driven predictor weights \(V\).
- The optimizer uses an active-set quadratic-program solve over the simplex rather than
  the nested optimizer used by `Synth`. It declares success only when the unscaled
  Frank-Wolfe/KKT gap certifies the stated objective. That changes computation, not the
  stated outcome-only objective.
- Multiple treated geographies must be combined on a scale comparable with individual
  donor trajectories. The intended paper-style default is an equal or explicitly
  weighted treated mean; an aggregate-volume sum requires separately constructed,
  comparably scaled donor composites.

### Required assumptions

Good pre-fit/convex-hull support, unaffected donors, stable factor loadings, no
anticipation, no differential concurrent shocks, and a substantively meaningful
treated aggregation.

### Repository validation

Tests cover simplex constraints, the KKT convergence certificate under heterogeneous
donor scales, point-effect recovery, null-effect behavior, failed fits, and synthetic
A/A behavior. The public calibration artifact exercises SCM on one seeded
common-factor/seasonality/AR(1) generator.

### Not validated or supported as a general claim

General covariate SCM; source-paper weight replication; structural breaks; spatial
spillovers; selected-treatment exact inference; heavy tails; arbitrary missingness;
and external validity beyond treated geographies.

## Augmented SCM (ASCM)

### Estimand and implemented equation

The package augments the SCM counterfactual with a ridge outcome model:

\[
\hat Y^{aug}_{1t}(0)=Y_{0t}^{\top}\hat w+
(x_1-X_0^{\top}\hat w)^{\top}\hat\beta_t,
\qquad
\hat\tau_t=Y_{1t}-\hat Y^{aug}_{1t}(0).
\]

### Source

[Ben-Michael, Feller, and Rothstein (2021)](https://jesse-rothstein.com/wp-content/uploads/2021/07/Ben-Michael_Feller_Rothstein_Augsynth_JASA_2021.pdf).

### Implementation scope and deviations

- Pre-treatment outcomes are the ridge features.
- The project tunes the ridge penalty by leaving out each pre-treatment period,
  refitting SCM and the outcome model without that period, and predicting the held-out
  treated outcome. No post-treatment outcome enters tuning. This implements the
  paper's time-validation idea, but numerical equivalence to `augsynth` is not claimed.
- The primary `weights` and `pre_rmspe` fields describe the base simplex SCM. Effective
  augmented weights, cross-fit augmented pre-RMSPE, negative-weight mass, and an
  extrapolation norm are exposed in `extras` and can differ materially.
- The conformal wrapper is intentionally not presented as supported for ASCM.

### Required assumptions

The SCM requirements plus a stable cross-unit outcome model, adequate regularization,
and tolerance for extrapolation introduced by augmentation.

### Repository validation

Tests cover effect recovery on the synthetic generator, null behavior, and that the
augmented counterfactual differs from base SCM under imbalance.

### Not validated or supported as a general claim

Golden equivalence with `augsynth`; a universal cutoff for the supplied extrapolation
diagnostics; repeated coverage of any ASCM interval; or robustness to weak cross-unit
stability.

## Synthetic difference-in-differences (SDID)

### Estimand and source equations

For treated means \(\bar Y_{tr,t}\), the target is the weighted difference in
differences

\[
\hat\tau=
\left(\bar Y_{tr,post}-\sum_{t\le T_0}\hat\lambda_t\bar Y_{tr,t}\right)
-\sum_i\hat\omega_i
\left(\bar Y_{i,post}-\sum_{t\le T_0}\hat\lambda_tY_{it}\right),
\]

with non-negative unit and time weights that each sum to one.

### Source

[Arkhangelsky et al. (2021)](https://www.nber.org/papers/w25532) and the authors'
[official `synthdid` implementation](https://github.com/synth-inference/synthdid).

### Implementation scope and deviations

- The package works with an aggregated treated mean rather than the full panel of
  treated-unit observations. An explicit legacy `sum` mode divides by the treated-unit
  count before fitting and reports the implied total separately. That legacy mode is
  available only from the low-level estimator; schema-2 design, power, placebo, and
  reporting paths enforce the registered treated count and `mean` aggregation.
- Unit and time weights are optimized with the same active-set simplex QP solver used
  by SCM. Both solves must satisfy the Frank-Wolfe/KKT gap certificate or the SDID fit
  fails closed.
- The unit penalty uses first-difference noise and the paper's \(\zeta\) formula; the
  time-weight ridge uses \(\zeta_\lambda=10^{-6}\hat\sigma\) and scales by the number
  of controls, as in Algorithm 1. The optimizer remains project-specific.
- Conformal test inversion changes the pre/post partition used to fit SDID and should
  not be treated as a supported SDID confidence interval without independent evidence.

### Required assumptions

Stable low-rank untreated outcome structure, unaffected donors, no anticipation,
adequate pre/post length, and a treated aggregation matching the ATT being reported.

### Repository validation

Tests cover synthetic effect recovery, treated-mean/sum scale equivalence, the stated
regularization formulas, null behavior, simplex time weights, and fail-closed solver
exhaustion. There is no release-level numeric replication against the official
`synthdid` package.

### Not validated or supported as a general claim

Arbitrary time-varying treatment effects, conformal SDID intervals, selected-treatment
exact inference, interference, structural breaks, or equivalence to every option in
the official R implementation.

## Panel difference-in-differences

### Estimand and implemented equation

The package estimates a two-way fixed-effects coefficient on treated-by-post after
optionally normalizing each city's outcome by its own pre-period mean:

\[
y_{it}=\alpha_i+\gamma_t+\tau(D_i\times Post_t)+\varepsilon_{it}.
\]

With normalization, \(\tau\) is an unweighted average relative effect across included
cities, not a market-size-weighted aggregate lift.

### Source and implementation status

This is a conventional two-group/two-period-style panel DiD specialization, not a
staggered-adoption estimator. The optional restricted wild-cluster bootstrap follows
the broad procedure in
[Cameron, Gelbach, and Miller (2008)](https://doi.org/10.1162/rest.90.3.414).

### Required assumptions

Parallel untreated trends, no anticipation, no spillovers, stable composition, a
correct common treatment date, and an inference method appropriate to the number and
balance of treated/control clusters.

### Repository validation

Tests cover point-effect recovery on the synthetic generator and basic bootstrap
execution. Ordinary WCR inference raises a typed error below four treated clusters;
that threshold is a misuse guard, not evidence that four treated clusters are enough.
At or above the threshold, the result remains explicitly diagnostic-only because the
package has no design-matched null-size certificate for this path. The null-size test
uses too few replications to establish nominal size.

### Not validated or supported as a general claim

Staggered adoption, heterogeneous dynamic effects, spatially correlated cities,
ordinary wild-cluster inference with one to three treated clusters, or a general claim
of calibrated WCR inference immediately above the four-treated threshold. For richer
many-unit DiD designs, use a purpose-built implementation such as
[DoubleML's documented DiD models](https://docs.doubleml.org/stable/guide/models.html#difference-in-differences-models-did).

## In-space placebo ranks

### Implemented statistic

The primary rank uses the post/pre RMSPE ratio:

\[
S_g=RMSPE_{g,post}/RMSPE_{g,pre},\qquad
\hat p=\frac{1+\sum_{g=1}^{M}\mathbf 1\{S_g\ge S_{obs}\}}{M+1}.
\]

For multiple treated geographies, placebo groups have the same cardinality as the
treated group; large combination spaces can be sampled without replacement.

### Source

The diagnostic follows the in-space placebo logic used by
[Abadie, Diamond, and Hainmueller](https://j-hai.github.io/assets/pdf/scm.pdf).

### Required assumptions for a p-value interpretation

The observed treatment assignment must be randomized over the enumerated assignments,
or units/sets must be exchangeable after conditioning on every design and selection
step. Optimizing the treated set for pre-fit or MDE and then comparing it with ordinary
donor groups does not automatically satisfy this condition.

### Repository validation

Tests cover strong-effect detection, null ranks, same-size grouped placebos,
deterministic sampling, failure handling, and the explicit
`valid_for_decision`/`inference_basis` distinction.

### Unsupported regimes

Exact randomization claims for observationally selected markets; contaminated donors;
post hoc donor/specification changes; and p-values whose attainable rank support is too
coarse for the chosen alpha.

## Conformal test inversion

### Implemented hypothesis

The implementation inverts tests of the sharp constant post-period path

\[
H_0:\tau_t=\tau_0\quad\text{for every post-treatment }t,
\]

using cyclic shifts of fitted residuals and a grid of candidate relative effects.

### Source

[Chernozhukov, Wüthrich, and Zhu (2021)](https://doi.org/10.1080/01621459.2021.1920957).

### Required assumptions

Residual exchangeability for exact finite-sample inference, or the paper's stability
and stationarity/mixing conditions for dependent-data approximations; a correctly
implemented sharp-null refit; and a grid wide/fine enough to represent the acceptance
set.

### Repository validation

One synthetic test checks that a known effect lies inside an SCM acceptance interval.
That is a regression test, not a repeated-coverage study.

### Unsupported regimes

Automatic stationarity validation; time-varying effects interpreted as an ATT
interval; ASCM; SDID without independent algorithmic validation; disconnected or
grid-truncated acceptance sets; and generic 95% labeling when a different alpha is
used.

## Ratio-outcome DiD with paired circular-block bootstrap

### Implemented estimand

For numerator \(N\) and denominator \(D\), each group-period rate is a ratio of sums,
\(R=\sum N/\sum D\), and the effect is

\[
\Delta=(R_{tr,post}-R_{tr,pre})-(R_{co,post}-R_{co,pre}).
\]

The package jointly resamples temporal blocks within the pre and post periods, using
the same sampled dates for treated/control numerators and denominators. It reports the
bootstrap standard deviation, a centered two-sided bootstrap p-value, and percentile
endpoints.

### Required assumptions and deviations

Joint resampling preserves contemporaneous treated/control and numerator/denominator
covariance and short-range serial structure inside a block. Validity still requires a
defensible circular-block-bootstrap regime: locally stable dependence, a suitable block
length, enough effectively independent blocks, and no unmodeled cross-period or
cross-geo interference. Pre and post periods are resampled separately.

### Repository validation and unsupported regimes

Tests cover one simulated rate shift, deterministic seeding, and failure to report
inference when too few blocks are available. Repeated-coverage and null-size behavior
have not been established for strong spatial dependence, long memory, few blocks, or
arbitrary nonstationarity.

## Power, MDE, and A/A calibration

### What is computed

Power simulation samples historical pseudo-treatment windows, optionally replays the
versioned market selector, injects a signed multiplicative effect, and evaluates the
estimator set, pre-fit gate, assignment mechanism, and executable `DecisionRule` in a
`DesignSpec`. A non-default `selection_procedure_id` requires a matching `selection_fn`.
It reports Wilson Monte Carlo intervals and invalid-fit rates. The
MDE is a non-negative magnitude for the smallest tested signed effect-grid value whose
estimated rejection rate reaches the target power; the signed crossing is retained.

Schema-2 power draws feasible historical start indices independently and uniformly
with replacement. Its Wilson interval therefore quantifies Monte Carlo error for the
empirical distribution over those starts, conditional on the supplied panel and frozen
procedure. It does **not** account for panel estimation error, future regime change,
spillover, or uncertainty about whether the historical period represents launch-time
conditions. Overlapping windows share observations but remain iid draws from this
finite empirical start distribution; legacy spacing values above one are exploratory
and are not eligible for a schema-2 authoritative artifact.

The A/A runner samples null treated/donor sets and windows, or calls a supplied
`selection_fn` to replay selection. It reports rejection frequency, an exact binomial
interval and one-sided upper bound, invalid-fit rate, a randomized-PIT KS diagnostic
for a single estimator's discrete ranks, and median ATT diagnostics. The joint-rule
path does not apply that marginal-uniform KS test to the minimum of several p-values;
its empirical rule-level FPR gate is the relevant calibration check. Results record
design and calibration fingerprints, but the caller must supply and persist the correct
design fingerprint.

### Required matching condition

Power and A/A apply to a deployed decision only when they replay the complete procedure:
treatment selection, spillover exclusions, treated aggregation, donor selection,
estimator versions/parameters, inference cap and seed policy, multiplicity handling,
validity gates, calibration acceptance thresholds, and executable decision rule.

### Current evidence and deviations

- The public artifact checks SCM over 400 pseudo-experiments on one benign seeded
  synthetic generator under explicitly randomized labels. It does not calibrate the
  observational placebo path or any optimized treatment-market selector.
- The current power API can execute a method-keyed estimator set and serializable
  minimum-rejections rule and replay a supplied selector; the legacy single-estimator
  call remains available.
- `recommend_treated_sets` reports a conditional screening MDE for ranking only. It
  returns a candidate design with a non-default selector ID, forcing confirmatory power
  to replay that selector before approval.
- Treatment-set selection is not replayed by default. A supplied `selection_fn` can
  replay it, but the library cannot verify that callback matches an external selector.
- For the binomial power and A/A intervals to retain their stated conditional meaning,
  `selection_fn` must implement one frozen policy and be stateless across simulation
  calls except for randomness drawn from the supplied generator. The callback receives
  a run index for reproducible bookkeeping; changing the policy by run number destroys
  the identical-trial condition and is unsupported.
- An observational fixed-set A/A must explicitly replay that fixed selection through a
  `selection_fn` (which may return the same registered set for every draw). A random-set
  A/A without selector replay is intentionally labeled `random_assignment` and cannot
  satisfy an observational design's exact-design match.
- For a single estimator, randomized PIT removes the mechanical tie problem before the
  continuous-uniform KS diagnostic; validity still depends on the discrete rank null
  and adequate valid runs. Joint rules intentionally skip this marginal-uniform check.
- Grid MDE and power estimates retain Monte Carlo uncertainty and invalid rates.
- `analyze_experiment` checks explicit inputs against a supplied `DesignSpec`; calls
  without one are retained for exploration but cannot emit an eligible decision.
- The randomized path refits every assignment with a symmetric treated-plus-donor
  universe. For multiple treated units, individual treated trajectories are mandatory;
  an aggregate series cannot reconstruct alternative assignments. If any sampled or
  enumerated assignment lacks a defined fit/statistic, the randomization p-value is
  suppressed rather than computed on the surviving subset.

### Unsupported regimes

Universal false-positive control, extrapolation from the bundled generator to a user's
panel, power for an unmatched decision rule, or post-selection inference without
replaying/conditioning on the selection algorithm.

## Evidence levels

Use these labels in issues, documentation, and downstream reports:

- **Unit tested:** a deterministic invariant or example is covered by repository tests.
- **Synthetic regression tested:** behavior is checked on one or more stated simulated
  generators.
- **Reference replicated:** outputs match a source-paper example or official package
  within a declared tolerance.
- **Design calibrated:** the complete deployed design/selection/decision procedure
  meets a preregistered operating-characteristic target on representative null and
  alternative regimes.

The current project contains unit tests and synthetic regression tests. Do not claim
reference replication or design calibration unless a release artifact explicitly
provides that evidence.

## Positioning and non-goals

### Closest comparators

- [GeoLift](https://github.com/facebookincubator/GeoLift) is the closest end-to-end
  comparator: its official project covers geo market selection, power, inference, and
  plotting, with a separate
  [multi-cell workflow](https://github.com/facebookincubator/GeoLift/blob/main/vignettes/GeoLift_MultiCell_Walkthrough.md).
- [CausalPy](https://github.com/pymc-labs/CausalPy) is a broader Python quasi-experiment
  library with Bayesian/OLS estimators, reports, plots, synthetic control, geo-lift
  examples, and
  [multi-treated analysis](https://causalpy.readthedocs.io/en/stable/notebooks/multi_cell_geolift.html).
- [pysyncon](https://github.com/sdfordham/pysyncon) focuses on classic, robust,
  augmented, and penalized SCM plus placebo tests and paper-replication notebooks.
- [SparseSC](https://github.com/microsoft/SparseSC) focuses on regularized feature/unit
  weights, covariates, cross-validation, and larger-dimensional synthetic controls.
- [CausalImpact](https://github.com/google/CausalImpact) estimates time-series
  intervention effects using Bayesian structural time-series models and unaffected
  control series.
- [DoubleML](https://github.com/DoubleML/doubleml-for-py) targets cross-fitted
  orthogonal causal ML and many-unit designs, including documented DiD variants.

### Defensible project position

The project's distinctive scope is an opinionated **Python geo-design and governance
layer**: spillover-aware design inputs, treated-set/MDE screening, preregistration
metadata, an analysis-window guard, multi-estimator sensitivity output, A/A tooling,
and optional Spark/Delta persistence.

### Non-goals

The project does not currently aim to provide:

- a novel SCM/ASCM/SDID estimator;
- automatic causal identification or automatic validation of SUTVA;
- general covariate-rich/high-dimensional synthetic controls;
- multi-cell allocation and winner selection;
- budget/CPIC optimization;
- a replacement for CausalImpact's Bayesian time-series models;
- a replacement for DoubleML's many-unit orthogonal-learning estimators;
- guaranteed valid p-values for arbitrary observational market selection;
- automatic generalization from treated geographies to a national rollout.

## Release checklist for a statistical claim

Before strengthening any public claim, require all of the following:

1. Name the estimand and treatment aggregation.
2. Cite the source equation and list every intentional deviation.
3. State assignment, interference, anticipation, stability, and overlap assumptions.
4. Add a source-paper or official-package golden comparison when claiming fidelity.
5. Calibrate the complete selection-to-decision procedure, not only the estimator.
6. Report invalid-fit rates, Monte Carlo uncertainty, and attainable p-value support.
7. Name untested regimes and keep them out of default decision paths.
