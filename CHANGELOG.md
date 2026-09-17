# Changelog

## 2.0.0a2

- Reject ambiguous city-name/state mappings before Spark aggregates distinct geographies.
- Require an explicit zero-activity assumption to fill absent city-days, reject invalid
  observed GMV, and keep incomplete rupture telemetry unavailable rather than zero.
- Validate daily calendar coverage and consistent local timestamps, including ratio
  frames; explicit filling no longer replaces non-finite observed values.
- Execute supplied directional decision rules in single-estimator A/A calibration.
- Reject altered fixed-control/DiD membership and incomplete or unidentified DiD windows.
- Reabsorb city/date fixed effects in each wild-cluster bootstrap draw while retaining
  the diagnostic-only status and few-treated-cluster guard.
- Advance the implementation binding: designs and calibration artifacts from `2.0.0a1`
  must be regenerated for this version.

## 2.0.0a1

- Corrected multi-treated estimands to compare treated means with donor counterfactuals.
- Prevented schema-2 SDID configuration from overriding the registered treated count or
  re-dividing an already averaged treated trajectory through the legacy `sum` mode.
- Aligned ASCM cross-validation and SDID penalty/aggregation with the documented source
  equations; estimator failures now fail closed.
- Separated observational placebo diagnostics from decision-valid randomized or
  calibrated inference, including symmetric full-universe randomization permutations;
  randomized assignment-dependent fit failures now invalidate the reference set.
- Added immutable `DesignSpec` and executable `DecisionRule` contracts across power,
  calibration, registry, and analysis.
- Added joint-procedure A/A calibration fingerprints, one-sided FPR inflation checks,
  invalid-run accounting, and randomized-PIT diagnostics for single-estimator discrete
  ranks; joint rules use the empirical rule-level FPR gate instead.
- Bound eligible universes, selector configuration, runtime version, simulation budgets,
  and acceptance thresholds into design, power, calibration, and registry artifacts;
  authoritative power now samples historical starts iid with replacement.
- Scoped the public calibration certificate to a 400-run symmetric randomized-label
  fixture; observational designs still require their own selected-procedure A/A.
- Added anticipation gaps, pre-fit gates, strict panel validation, immutable registry
  transitions, and guardrail inference via a paired circular-block bootstrap.
- Narrowed conformal inference to its implemented SCM sharp-null scope and made ordinary
  wild-cluster bootstrap fail closed for fewer than four treated clusters and remain
  diagnostic-only above that unsupported count threshold.
- Reworked packaging metadata, release documentation, CI build checks, and statistical
  regression tests for the alpha release.
