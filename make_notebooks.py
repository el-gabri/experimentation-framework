"""Gera os 5 notebooks Databricks finos (clientes do pacote supply_experiments)."""

import hashlib

import nbformat as nbf


def nb(cells, path):
    n = nbf.v4.new_notebook()
    # nbformat gera IDs aleatórios por padrão; IDs derivados do conteúdo tornam
    # a geração reprodutível e permitem que a CI detecte deriva real via diff.
    for position, cell in enumerate(cells):
        identity = f"{path}:{position}:{cell.cell_type}:{cell.source}".encode("utf-8")
        cell["id"] = hashlib.sha256(identity).hexdigest()[:12]
    n.cells = cells
    n.metadata = {"language_info": {"name": "python"},
                  "application/vnd.databricks.v1+notebook": {"language": "python"}}
    nbf.write(n, path)
    print("wrote", path)


md = nbf.v4.new_markdown_cell
py = nbf.v4.new_code_cell

# =============================================================================
# 01 — ETL
# =============================================================================
nb([
md("""# 01 — ETL: painel de cidades e snapshots
Cliente fino do pacote `supply_experiments`. Toda a lógica (filtros da base de
pedidos, normalização de cidades, estatísticas de elegibilidade) vive no pacote
— **uma única implementação**, testada, em vez das 4 cópias divergentes anteriores."""),
    py("""%pip install /Workspace/Shared/supply_experiments/supply_experiments-2.0.0a1-py3-none-any.whl --quiet --force-reinstall --no-deps
dbutils.library.restartPython()"""),
py("""from supply_experiments.spark_io import load_city_panel, TABLES
from supply_experiments.design.spillover import EligibilityCriteria, eligible_cities

# ---- parâmetros -----------------------------------------------------------
dbutils.widgets.text("target_env", "sandbox")     # sandbox | prod
dbutils.widgets.text("date_start", "2025-09-01")
dbutils.widgets.text("date_end",   "2026-06-30")

TARGET_ENV = dbutils.widgets.get("target_env")
DATE_START = dbutils.widgets.get("date_start")
DATE_END   = dbutils.widgets.get("date_end")
tables = TABLES[TARGET_ENV]"""),
py("""# painel diário (GMV + num/den de ruptura) e stats por cidade — 1 chamada
panel, stats = load_city_panel(spark, DATE_START, DATE_END)
print(f"Painel: {len(panel.cities)} cidades x {len(panel.index)} dias")
stats.head(10)"""),
py("""# elegibilidade: filtros de volume + NOVOS critérios de qualidade de série
crit = EligibilityCriteria(
    min_nonzero_days=92,
    min_avg_daily_gmv=10_000.0,
    max_zero_run_days=7,   # buracos de dados longos => fora
    max_cv=1.5,            # séries erráticas => fora
)
elig = eligible_cities(stats, crit)
print(f"{len(elig)} cidades elegíveis de {len(stats)}")
display(elig.sort_values("avg_daily_gmv", ascending=False).head(30))"""),
py("""# persiste snapshot de elegibilidade p/ os notebooks de design
elig_sdf = spark.createDataFrame(elig.reset_index())
elig_sdf.write.format("delta").mode("overwrite") \\
    .saveAsTable(f"{tables['registry'].rsplit('.',1)[0]}.eligible_cities_snapshot")"""),
], "notebooks/01_etl.ipynb")

# =============================================================================
# 02 — Seleção de controle fixo (DiD)
# =============================================================================
nb([
md("""# 02 — Controle fixo para DiD (com holdout temporal)
Princípios de design:
1. **Holdout temporal**: otimização roda nos primeiros 70% da janela; tendências
   paralelas são testadas **apenas nos 30% finais** (critério de aceite, nunca
   de otimização) — elimina pre-testing contamination.
2. **p-value fora do score**: score = correlação + forma + região + ruptura.
3. **Target limpo**: Brasil excluindo o próprio controle e tratadas ativas."""),
    py("""%pip install /Workspace/Shared/supply_experiments/supply_experiments-2.0.0a1-py3-none-any.whl --quiet --force-reinstall --no-deps
dbutils.library.restartPython()"""),
py("""from supply_experiments.spark_io import (load_city_panel, TABLES,
                                          active_blocked_cities, NATIONAL_HOLIDAYS)
from supply_experiments.design.control_selection import select_fixed_control

TARGET_ENV = "sandbox"
tables = TABLES[TARGET_ENV]
panel, stats = load_city_panel(spark, "2025-09-01", "2026-06-30")

blocked = active_blocked_cities(spark, tables["registry"])
print("Bloqueadas (experimentos ativos):", blocked)"""),
py("""STATE_TO_REGION = {
    "SP":"Sudeste","RJ":"Sudeste","MG":"Sudeste","ES":"Sudeste",
    "PR":"Sul","SC":"Sul","RS":"Sul",
    "BA":"Nordeste","PE":"Nordeste","CE":"Nordeste","MA":"Nordeste","PB":"Nordeste",
    "RN":"Nordeste","AL":"Nordeste","SE":"Nordeste","PI":"Nordeste",
    "DF":"Centro-Oeste","GO":"Centro-Oeste","MT":"Centro-Oeste","MS":"Centro-Oeste",
    "AM":"Norte","PA":"Norte","RO":"Norte","RR":"Norte","AC":"Norte","AP":"Norte","TO":"Norte",
}
candidates = [c for c in stats.index
              if c in panel.cities
              and c not in blocked["treated_active"]]
city_states  = stats["state_address"].to_dict()
city_rupture = stats["rupture_rate"].to_dict()

res = select_fixed_control(
    panel, candidates, city_states, STATE_TO_REGION,
    max_cities=9, train_frac=0.7, alpha=0.05,
    holidays=NATIONAL_HOLIDAYS,
    city_rupture=city_rupture,
    target_exclude=set(blocked["treated_active"]),
)
print("Cidades:", res.cities)
print(f"Score treino: {res.train_score:.3f}")
print(f"HOLDOUT — p-value de equivalência: {res.holdout_p_value:.3f} "
      f"({'ACEITO' if res.holdout_passed else 'REPROVADO'}) | corr: {res.holdout_corr:.3f}")"""),
py("""# persiste no registry v2 apenas se aprovado no holdout
from supply_experiments.spark_io import ExperimentRecord, ensure_registry, save_experiment
import uuid
from datetime import date

assert res.holdout_passed, "Controle reprovado no holdout — refaça a seleção (mais candidatas / outra janela)"

rec = ExperimentRecord(
    experiment_id=str(uuid.uuid4()), name="fixed_control_groceries",
    hypothesis="Conjunto de controle fixo representativo do Brasil (uso geral DiD)",
    status="approved", design_method="did", primary_kpi="gmv",
    guardrail_kpis=["rupture_rate_order"],
    start_date=date.today(), end_date=date.today(), pre_window_days=0,
    treated_units=[],
    control_units=[{"city_norm": c, "city_address": c,
                    "state_address": city_states.get(c, ""), "weight": None}
                   for c in res.cities],
    mde_estimated=0.0, expected_effect=None,
    decision_rule="n/a (controle fixo, não é experimento)",
    created_by=spark.sql("SELECT current_user()").first()[0],
)
ensure_registry(spark, tables["registry"])
save_experiment(spark, tables["registry"], rec, require_approval_checks=False)
print("Controle fixo registrado:", rec.experiment_id)"""),
], "notebooks/02_fixed_control_selection.ipynb")

# =============================================================================
# 03 — Design de experimento com contrato, power e calibração conjunta
# =============================================================================
nb([
    md("""# 03 — Design, power e aprovação
O registro só vira `approved` depois que **o mesmo `DesignSpec`** passa pelo power
gate e por A/A do procedimento completo. Para seleção observacional de mercados,
o callback de A/A deve reproduzir seleção, exclusões, estimadores e regra de decisão;
um A/A com cidades sorteadas não autoriza o desenho selecionado."""),
    py("""%pip install /Workspace/Shared/supply_experiments/supply_experiments-2.0.0a1-py3-none-any.whl --quiet --force-reinstall --no-deps
dbutils.library.restartPython()"""),
    py("""from dataclasses import replace
from datetime import date, timedelta
import json, uuid

from supply_experiments.calibration.aa import run_aa_calibration
from supply_experiments.design.power import power_analysis
from supply_experiments.design.spec import DecisionRule, DesignSpec
from supply_experiments.design.treated_selection import (recommend_treated_sets,
                                                         recommendations_frame)
from supply_experiments.estimators import fit_ascm, fit_scm, fit_sdid
from supply_experiments.io import (TABLES, ExperimentRecord, active_blocked_cities,
                                   ensure_registry, load_city_panel, save_experiment)

POST_DAYS, PRE_DAYS = 35, 122
ALPHA, MAX_PLACEBOS = 0.10, 30
EFFECT_GRID = (0.0, 0.02, 0.03, 0.05, 0.08)
FIT_FNS = {"scm": fit_scm, "ascm": fit_ascm, "sdid": fit_sdid}
RULE = DecisionRule(min_rejections=2, direction="positive",
                    methods=("scm", "ascm", "sdid"))
SELECTION_PROCEDURE_ID = "recommend-treated-sets-v1"
SELECTOR_CONFIG = {
    "selector": "recommend_treated_sets",
    "n_treated": [2, 3],
    "max_gmv_share_per_city": 0.05,
    "min_donors": 10,
    "top_k_cities": 12,
    "screen_donor_cap": 40,
    "spillover_radius_km": 40.0,
}

tables = TABLES["sandbox"]
panel, stats = load_city_panel(spark, "2025-06-01", "2026-06-30")
blocked = active_blocked_cities(spark, tables["registry"])
candidates = [c for c in stats.index if c in panel.cities
              and c not in blocked["treated_active"] | blocked["control_active"]]"""),
    md("""## 1. Propose a design without post-treatment data
The ranking is a selection algorithm, not random assignment. Its exact deployed
version must therefore be replayed in A/A below."""),
    py("""coords = {}  # supply coordinates/adjacency in production; empty means no radius screen
recs = recommend_treated_sets(
    panel, stats, candidates, fit_scm,
    pre_days=PRE_DAYS, post_days=POST_DAYS, n_treated=(2, 3),
    coords=coords, max_gmv_share_per_city=0.05,
)
display(recommendations_frame(recs))"""),
    py("""CHOSEN_RANK = 1
TREATED = tuple(recs[CHOSEN_RANK - 1].cities)
DONORS = tuple(recs[CHOSEN_RANK - 1].donors[:20])
HYPOTHESIS = "State the directional causal hypothesis and mechanism here"
EXPECTED_EFFECT = 0.05
EXPERIMENT_ID = str(uuid.uuid4())
START = date.today() + timedelta(days=7)
END = START + timedelta(days=POST_DAYS - 1)

draft_spec = DesignSpec(
    experiment_id=EXPERIMENT_ID, treated_units=TREATED, donor_units=DONORS,
    estimator_names=tuple(FIT_FNS), pre_days=PRE_DAYS, post_days=POST_DAYS,
    start_date=START, end_date=END, alpha=ALPHA,
    max_group_placebos=MAX_PLACEBOS, effect_grid=EFFECT_GRID,
    decision_rule=RULE, primary_kpi="gmv",
    assignment_mechanism="observational",
    selection_procedure_id=SELECTION_PROCEDURE_ID,
    eligible_units=tuple(candidates),
    selector_config_json=json.dumps(SELECTOR_CONFIG),
    max_pre_rmspe=0.10, require_calibration=True, calibration_scope="exact_design",
)
print("Draft design fingerprint:", draft_spec.fingerprint)

SELECTION_REPLAY_IMPLEMENTED = False
def replay_deployed_selection(panel_for_run, eligible, rng, run, start_idx):
    # Re-run SELECTION_PROCEDURE_ID using only information available before start_idx.
    # It must return exactly (n_treated, n_donors) cities after the same exclusions.
    # The version ID is the code-identity boundary: keep that implementation and
    # SELECTOR_CONFIG versioned together because Python callback bytecode is not hashed.
    raise NotImplementedError("Implement the deployed selector replay before approval")

assert SELECTION_REPLAY_IMPLEMENTED, 'selection replay is required for power and A/A'"""),
    md("""## 2. Power the executable rule
This runs all three estimators and the registered two-of-three rule. Invalid fits and
the pre-fit gate are reported separately; they are not counted as non-rejections."""),
    py("""pw = power_analysis(
    panel, TREATED, DONORS, fit_scm, pre_days=PRE_DAYS, post_days=POST_DAYS,
    effect_grid=EFFECT_GRID, n_sims_per_point=30, alpha=ALPHA,
    max_group_placebos=MAX_PLACEBOS, fit_fns=FIT_FNS,
    decision_rule=RULE, design_spec=draft_spec,
    selection_fn=replay_deployed_selection, eligible=candidates,
    selection_procedure_id=SELECTION_PROCEDURE_ID,
)
for effect in pw.effect_grid:
    print(effect, pw.power[effect], pw.power_ci[effect], pw.invalid_rate[effect])
assert pw.design_fingerprint == draft_spec.fingerprint
assert pw.mde_80 is not None and pw.mde_80 <= EXPECTED_EFFECT"""),
    md("""## 3. Replay market selection under the null
Implement this callback from the versioned production selector. Returning the already
chosen cities is **not** a replay when outcomes influenced their selection. The explicit
assertion prevents accidental approval of the placeholder."""),
    py("""assert pw.selection_scope == "replayed_selection"
assert pw.selection_procedure_id == SELECTION_PROCEDURE_ID"""),
    py("""aa = run_aa_calibration(
    panel, candidates, fit_scm, pre_days=PRE_DAYS, post_days=POST_DAYS,
    n_runs=200, n_treated=len(TREATED), n_donors=len(DONORS), alpha=ALPHA,
    max_group_placebos=MAX_PLACEBOS,
    min_valid_runs=draft_spec.calibration_min_valid_runs,
    min_ks_p_value=draft_spec.calibration_min_ks_p_value,
    max_fpr_inflation=draft_spec.calibration_max_fpr_inflation,
    max_invalid_rate=draft_spec.calibration_max_invalid_rate,
    fit_fns=FIT_FNS, decision_rule=RULE,
    selection_fn=replay_deployed_selection,
    selection_procedure_id=SELECTION_PROCEDURE_ID,
    design_fingerprint=draft_spec.fingerprint,
    assignment_mechanism=draft_spec.assignment_mechanism,
    max_pre_rmspe=draft_spec.max_pre_rmspe,
    seed=draft_spec.calibration_seed,
    design_spec=draft_spec,
)
assert aa.passed, aa.failure_reasons
spec = replace(draft_spec, calibration_fingerprint=aa.calibration_fingerprint)
assert aa.matches_design(spec)
from datetime import datetime
spark.createDataFrame([{
    "run_at": datetime.utcnow(),
    "calibration_fingerprint": aa.calibration_fingerprint,
    "design_fingerprint": aa.design_fingerprint,
    "passed": aa.passed,
    "selection_scope": aa.selection_scope,
    "procedure_scope": aa.procedure_scope,
    "artifact_json": aa.to_json(include_details=False),
}]).write.format("delta").mode("append").option("mergeSchema", "true") \
  .saveAsTable(tables["calibration"])
print("Calibration fingerprint:", aa.calibration_fingerprint)"""),
    md("""## 4. Approve the immutable bound record
`save_experiment` rejects an unbound design, a failed power gate, a missing calibration
digest, late approval, or a mutation after approval."""),
    py("""rec = ExperimentRecord(
    experiment_id=EXPERIMENT_ID, name="geo_experiment",
    hypothesis=HYPOTHESIS, status="approved", design_method="ascm",
    primary_kpi=spec.primary_kpi, guardrail_kpis=["rupture_rate_order"],
    start_date=START, end_date=END, pre_window_days=PRE_DAYS,
    treated_units=[{"city_norm": c, "city_address": c, "state_address": "",
                    "weight": None} for c in TREATED],
    control_units=[{"city_norm": c, "city_address": c,
                    "state_address": str(stats.loc[c, "state_address"]),
                    "weight": None} for c in DONORS],
    mde_estimated=pw.mde_80, expected_effect=EXPECTED_EFFECT,
    decision_rule=json.dumps(RULE.to_dict(), sort_keys=True),
    created_by=spark.sql("SELECT current_user()").first()[0],
).with_design_contract(spec, aa.calibration_fingerprint, power_result=pw)
ensure_registry(spark, tables["registry"])
save_experiment(spark, tables["registry"], rec, alpha=ALPHA)
print("Approved experiment:", rec.experiment_id)"""),
], "notebooks/03_design_experiment.ipynb")

# =============================================================================
# 04 — Análise
# =============================================================================
nb([
md("""# 04 — Final analysis bound to the approved design
The notebook loads the immutable `DesignSpec` and exact calibration artifact from
storage. Analysis before the registered end is blocked; explicit interim mode returns
guardrails only. Estimator agreement is a sensitivity check, not independent evidence."""),
py("""%pip install /Workspace/Shared/supply_experiments/supply_experiments-2.0.0a1-py3-none-any.whl --quiet --force-reinstall --no-deps
dbutils.library.restartPython()"""),
py("""from datetime import timedelta
from pyspark.sql import functions as F

from supply_experiments.calibration.aa import AACalibration
from supply_experiments.design.spec import DesignSpec
from supply_experiments.io import (NATIONAL_HOLIDAYS, TABLES, load_city_panel,
                                   load_experiment, load_fixed_control)
from supply_experiments.panel import ExperimentWindow
from supply_experiments.reporting import analyze_experiment

dbutils.widgets.text("experiment_id", "")
EXPERIMENT_ID = dbutils.widgets.get("experiment_id")

tables = TABLES["sandbox"]
rec = load_experiment(spark, tables["registry"], EXPERIMENT_ID)
assert rec.status in {"completed", "analyzed"}, f"unexpected lifecycle state: {rec.status}"
spec = DesignSpec.from_json(rec.design_spec_json)
assert spec.fingerprint == rec.design_fingerprint

artifact_rows = (spark.table(tables["calibration"])
    .where(F.col("calibration_fingerprint") == rec.calibration_fingerprint)
    .where(F.col("passed") == F.lit(True))
    .orderBy(F.col("run_at").desc()).limit(1).collect())
assert artifact_rows, "matching passing calibration artifact not found"
aa = AACalibration.from_json(artifact_rows[0]["artifact_json"])
assert aa.matches_design(spec), 'calibration artifact does not match DesignSpec'"""),
py("""date_start = str(
    rec.start_date - timedelta(days=rec.pre_window_days + spec.anticipation_days)
)
panel, _ = load_city_panel(spark, date_start, str(rec.end_date))
treated = list(spec.treated_units)
donors = list(spec.donor_units)
window = ExperimentWindow(
    start_date=spec.start_date, end_date=spec.end_date,
    pre_window_days=spec.pre_days, anticipation_days=spec.anticipation_days,
)"""),
py("""# Optional complementary DiD: equivalence plus correlation on recent pre-data.
from supply_experiments.design.control_selection import revalidate_fixed_control

did_control = None
fixed = load_fixed_control(spark, tables["registry"])
if fixed:
    val = revalidate_fixed_control(
        panel, fixed, window_days=90, holidays=NATIONAL_HOLIDAYS,
        target_exclude=set(treated), end_date=window.pre_end,
        equivalence_margin=0.10, min_corr=0.80,
    )
    print(val.summary())
    if val.passed:
        did_control = [city for city in fixed if city not in set(treated)]
else:
    print("No current fixed control; complementary DiD omitted.")"""),
py("""report = analyze_experiment(
    panel, rec.experiment_id, treated, donors, window,
    primary_kpi=spec.primary_kpi, guardrail_kpis=rec.guardrail_kpis,
    alpha=spec.alpha, max_group_placebos=spec.max_group_placebos,
    holidays=NATIONAL_HOLIDAYS, did_control=did_control,
    run_conformal=True, design_spec=spec, calibration=aa,
)
assert report.decision_eligible, report.warnings
print(report.summary())"""),
py("""display(report.to_frame())
if not report.guardrails.empty:
    display(report.guardrails)"""),
py("""import json
from datetime import datetime
rows = report.to_frame().assign(
    experiment_id=rec.experiment_id, kpi=rec.primary_kpi,
    decision=report.decision, decision_eligible=report.decision_eligible,
    design_fingerprint=report.design_fingerprint,
    calibration_fingerprint=report.calibration_fingerprint,
    sensitivity_agreement=report.triangulation, analyzed_at=datetime.utcnow(),
    warnings=json.dumps(report.warnings, ensure_ascii=False),
)
spark.createDataFrame(rows).write.format("delta").mode("append") \\
    .option("mergeSchema", "true").saveAsTable(tables["results"])
print("Results persisted in", tables["results"])"""),
], "notebooks/04_analyze_experiment.ipynb")

# =============================================================================
# 05 — Calibração A/A (job recorrente)
# =============================================================================
nb([
md("""# 05 — Recurring estimator stress calibration
This job monitors single-estimator behavior across random historical pseudo-assignments.
It is useful regression evidence, but it does **not** authorize the selected multi-
estimator design in notebook 03 because it does not replay that selector or decision rule."""),
py("""%pip install /Workspace/Shared/supply_experiments/supply_experiments-2.0.0a1-py3-none-any.whl --quiet --force-reinstall --no-deps
dbutils.library.restartPython()"""),
py("""from supply_experiments.io import load_city_panel, TABLES
from supply_experiments.calibration.aa import run_aa_calibration
from supply_experiments.estimators.scm import fit_scm
from supply_experiments.estimators.ascm import fit_ascm
from supply_experiments.estimators.sdid import fit_sdid
import supply_experiments

tables = TABLES["sandbox"]
panel, stats = load_city_panel(spark, "2025-06-01", "2026-06-30")
eligible = [c for c in stats.index if c in panel.cities][:60]"""),
py("""from datetime import datetime
rows = []
for name, fn in [("scm", fit_scm), ("ascm", fit_ascm), ("sdid", fit_sdid)]:
    aa = run_aa_calibration(panel, eligible, fn, pre_days=120, post_days=35,
                            n_runs=200, n_treated=2, n_donors=15, alpha=0.10)
    print(f"{name}: FPR={aa.fpr:.1%}, upper95={aa.fpr_upper_bound:.1%}, "
          f"invalid={aa.invalid_rate:.1%}, rank-PIT KS={aa.ks_p_value:.3f}, "
          f"passed={aa.passed}")
    rows.append({"run_at": datetime.utcnow(), "estimator": name,
                 "package_version": supply_experiments.__version__,
                 "n_runs": aa.n_runs, "alpha": aa.alpha, "fpr": aa.fpr,
                 "fpr_ci_low": aa.fpr_ci[0], "fpr_ci_high": aa.fpr_ci[1],
                 "ks_p_value": aa.ks_p_value, "median_bias": aa.median_att_bias,
                 "invalid_rate": aa.invalid_rate,
                 "calibration_fingerprint": aa.calibration_fingerprint,
                 "design_fingerprint": aa.design_fingerprint,
                 "selection_scope": aa.selection_scope,
                 "procedure_scope": aa.procedure_scope,
                 "artifact_json": aa.to_json(include_details=False),
                 "passed": aa.passed})
spark.createDataFrame(rows).write.format("delta").mode("append") \\
    .option("mergeSchema", "true").saveAsTable(tables["calibration"])"""),
], "notebooks/05_aa_calibration.ipynb")
