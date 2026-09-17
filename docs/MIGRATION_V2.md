# Migrating to the 2.0 alpha

`2.0.0a1` intentionally breaks several prototype defaults that could produce a
statistically misleading result.

## Updating from 2.0.0a1 to 2.0.0a2

- Recreate the `DesignSpec` and rerun its power and A/A procedure under the new
  implementation version. Old artifacts intentionally fail the runtime-version check.
- Supply one observation per local calendar day at a consistent local time. Timezones
  are preserved across daylight-saving transitions; timestamps are not silently
  normalized or merged. Ratio frames must use the outcome's timezone and local time.
- `load_city_panel` rejects ambiguous city names spanning multiple states. Resolve the
  geography identity upstream instead of combining distinct places under one name.
- The Spark loader now rejects missing city-days by default. Set
  `assume_missing_city_days_zero=True` only when source completeness guarantees absent
  rows mean no activity. Invalid observed monetary values are rejected in either mode.
- Incomplete rupture telemetry makes the optional guardrail unavailable. Set
  `require_rupture_telemetry=True` when it is required for the design or selection score.
- Fixed-control revalidation and panel DiD reject missing/duplicate members, overlapping
  groups, and incomplete requested windows rather than analyzing a modified group or
  period. Low-level DiD requires an identified, finite relative-effect estimate.
- Single-estimator A/A now executes its supplied directional rule. Bootstrap diagnostics
  can change because fixed effects are reabsorbed in every draw; their inferential
  status remains diagnostic-only.

## Required changes

- Python 3.9 is no longer supported; the 2.0 alpha requires Python 3.10 or newer.
- Multi-treated SCM, ASCM, and SDID now use the mean treated trajectory. The old SDID
  sum behavior is available only through an explicit low-level estimator call;
  authoritative design, power, placebo, and reporting paths reject or override it.
- `CityPanel` now rejects missing calendar days, non-finite values, unknown cities,
  duplicates, and treated/donor overlap. Pass `fill_value=0` only when a missing row is
  known to mean zero activity.
- Observational placebo ranks are diagnostics. A decision requires either a declared
  randomized assignment with symmetric label permutations or a passing selected-design,
  joint-procedure A/A calibration bound to the same `DesignSpec`.
- `analyze_experiment` without `design_spec` always returns `DIAGNOSTIC_ONLY`. Interim
  mode returns guardrails only and suppresses point estimates and decisions.
- Ordinary wild-cluster bootstrap inference now raises
  `UnsupportedFewTreatedClustersError` below four treated clusters.
- Fixed-control validation uses a TOST equivalence test. Passing means the equivalence
  p-value is at or below alpha and the correlation floor is met; failure to reject an
  ordinary zero-slope test is no longer treated as evidence of parallel trends.
- Schema-2 power uses iid historical starts with replacement (`window_spacing_days=1`)
  so its Wilson intervals have a conditional Monte Carlo interpretation. Larger spacing
  values are legacy/exploratory and cannot be bound to an authoritative v2 design.
- Design contracts now bind the eligible-unit universe, canonical selector settings,
  implementation version, power simulation budget, calibration budget, and their
  acceptance thresholds. Old serialized designs must be recreated rather than silently
  upgraded.

## Recommended migration sequence

1. Construct and serialize a frozen `DesignSpec` before launch.
2. Run power with the exact estimator set and `DecisionRule` in that spec.
   If market outcomes influenced treatment selection, supply the same versioned
   `selection_fn`; candidate-ranking MDEs are explicitly screening-only.
3. Run A/A with the deployed selector callback, estimator set, validity gates, and rule.
   The callback must be stateless across runs apart from the supplied random generator.
4. Bind the passing calibration fingerprint to the spec and registry record.
5. Pass the stored spec and calibration artifact to final analysis.

Low-level estimator functions remain useful for research and sensitivity analysis, but
they do not enforce this lifecycle.
