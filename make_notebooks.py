"""Gera os 4 notebooks Databricks finos (clientes do pacote supply_experiments)."""

import nbformat as nbf


def nb(cells, path):
    n = nbf.v4.new_notebook()
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
py("""%pip install /Workspace/Shared/supply_experiments/supply_experiments-1.0.2-py3-none-any.whl --quiet --force-reinstall --no-deps
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
py("""%pip install /Workspace/Shared/supply_experiments/supply_experiments-1.0.2-py3-none-any.whl --quiet --force-reinstall --no-deps
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
print(f"HOLDOUT — p-value tendências paralelas: {res.holdout_p_value:.3f} "
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
# 03 — Design de experimento (SC) com power gate
# =============================================================================
nb([
md("""# 03 — Design de experimento com power gate
Fluxo: tratadas propostas → exclusão de spillover no donor pool → **análise de
poder por simulação** → só registra como `approved` se MDE ≤ efeito esperado e
pré-registro completo (hipótese + regra de decisão). O `save_experiment` **recusa**
registros que não cumpram o contrato."""),
py("""%pip install /Workspace/Shared/supply_experiments/supply_experiments-1.0.2-py3-none-any.whl --quiet --force-reinstall --no-deps
dbutils.library.restartPython()"""),
py("""from supply_experiments.spark_io import (load_city_panel, TABLES,
                                          active_blocked_cities, ExperimentRecord,
                                          ensure_registry, save_experiment)
from supply_experiments.design.spillover import spillover_exclusions
from supply_experiments.design.treated_selection import (recommend_treated_sets,
                                                         recommendations_frame)
from supply_experiments.design.power import power_analysis
from supply_experiments.estimators.scm import fit_scm
from supply_experiments.estimators.ascm import fit_ascm

POST_DAYS = 35
PRE_DAYS  = 122
ESTIMATOR = fit_ascm             # scm | ascm | sdid

TARGET_ENV = "sandbox"
tables = TABLES[TARGET_ENV]
panel, stats = load_city_panel(spark, "2025-09-01", "2026-06-30")
blocked = active_blocked_cities(spark, tables["registry"])
candidates = [c for c in stats.index if c in panel.cities
              and c not in blocked["treated_active"] | blocked["control_active"]]"""),
md("""## Passo 1 — Recomendação de tratadas (market selection)
O framework ranqueia os conjuntos de cidades **mais mensuráveis** (menor MDE) e
o stakeholder só escolhe uma linha da tabela. Restrições da operação entram como
parâmetros: `states=["SP"]`, `must_include=["GUARULHOS"]`, `exclude=[...]`."""),
py("""# ---- RECOMENDAÇÃO DE TRATADAS ------------------------------------------------
# Roda inteiramente com SCM (rápido). O ESTIMATOR pré-registrado (ex.: ascm,
# ~100x mais caro por causa do LOO-CV) entra só no power gate confirmatório.
coords = {}   # opcional: dict city -> (lat, lon) p/ exclusão de spillover por raio
recs = recommend_treated_sets(
    panel, stats, candidates, fit_scm,
    pre_days=PRE_DAYS, post_days=POST_DAYS,
    n_treated=(2, 3),            # tamanhos de conjunto a considerar
    states=None,                 # ex.: ["SP", "MG"] se a operação for regional
    must_include=[],             # cidades que a operação exige tratar
    exclude=[],                  # cidades vetadas pela operação
    coords=coords,
    max_gmv_share_per_city=0.05, # evita tratar cidades grandes demais
)
display(recommendations_frame(recs))"""),
md("""## Passo 2 — Pré-registro
Escolha o `CHOSEN_RANK` da tabela acima. Se as tratadas forem impostas pelo
negócio, defina `TREATED` manualmente e derive `donors` com
`spillover_exclusions(TREATED, candidates, coords, radius_km=40, adjacency=...)`."""),
py("""# ---- PRÉ-REGISTRO (preencher ANTES de olhar qualquer resultado) -------------
CHOSEN_RANK      = 1
TREATED          = recs[CHOSEN_RANK - 1].cities
donors           = recs[CHOSEN_RANK - 1].donors
HYPOTHESIS       = "Tratativa de ruptura via chat reduz rupture_rate e protege GMV"
EXPECTED_EFFECT  = 0.05          # efeito esperado no KPI primário (fração)
DECISION_RULE    = "ship se ATT>0 e p<=0.10 (permutação) em >=2 de 3 estimadores"
print("Tratadas:", TREATED, "| doadoras:", len(donors))"""),
py("""# ---- POWER GATE (confirmação no design final) --------------------------------
pw = power_analysis(panel, TREATED, donors[:20], ESTIMATOR,
                    pre_days=PRE_DAYS, post_days=POST_DAYS,
                    effect_grid=(0.0, 0.02, 0.03, 0.05, 0.08),
                    n_sims_per_point=30, alpha=0.10)
for d, p in sorted(pw.power.items()):
    print(f"  δ={d:5.1%} -> poder {p:6.1%}")
print(f"FPR(δ=0)={pw.fpr:.1%} | MDE@80% = {pw.mde_80}")
assert pw.mde_80 is not None and pw.mde_80 <= EXPECTED_EFFECT, (
    f"SEM PODER: MDE={pw.mde_80} > efeito esperado {EXPECTED_EFFECT}. "
    "Aumente cidades tratadas, duração, ou escolha cidades menos ruidosas.")"""),
py("""# ---- registro (validado) -----------------------------------------------------
import uuid
from datetime import date, timedelta

start = date.today() + timedelta(days=7)   # início planejado
rec = ExperimentRecord(
    experiment_id=str(uuid.uuid4()), name="tratativa_chat_ruptura",
    hypothesis=HYPOTHESIS, status="approved", design_method="ascm",
    primary_kpi="gmv", guardrail_kpis=["rupture_rate_order"],
    start_date=start, end_date=start + timedelta(days=POST_DAYS - 1),
    pre_window_days=PRE_DAYS,
    treated_units=[{"city_norm": c, "city_address": c, "state_address": "SP",
                    "weight": None} for c in TREATED],
    control_units=[{"city_norm": c, "city_address": c,
                    "state_address": str(stats.loc[c, "state_address"]),
                    "weight": None} for c in donors[:20]],
    mde_estimated=pw.mde_80, expected_effect=EXPECTED_EFFECT,
    decision_rule=DECISION_RULE,
    created_by=spark.sql("SELECT current_user()").first()[0],
)
ensure_registry(spark, tables["registry"])
save_experiment(spark, tables["registry"], rec)   # levanta erro se pré-registro incompleto
print("Experimento registrado:", rec.experiment_id)"""),
], "notebooks/03_design_experiment.ipynb")

# =============================================================================
# 04 — Análise
# =============================================================================
nb([
md("""# 04 — Análise de experimento (triangulação)
- **Anti-peeking estrutural**: `analyze_experiment` recusa análise antes de `end_date`.
- SCM + ASCM + SDID com p-values de **permutação**; IC **conformal** no ASCM;
  DiD painel com **wild cluster bootstrap** se houver controle fixo.
- Guardrails de razão via **método delta** + correção **Benjamini-Hochberg**.
- Veredito de triangulação: CONCORDANTE / PARCIAL / DIVERGENTE."""),
py("""%pip install /Workspace/Shared/supply_experiments/supply_experiments-1.0.2-py3-none-any.whl --quiet --force-reinstall --no-deps
dbutils.library.restartPython()"""),
py("""from supply_experiments.spark_io import (load_city_panel, TABLES, load_experiment,
                                          NATIONAL_HOLIDAYS)
from supply_experiments.panel import ExperimentWindow
from supply_experiments.reporting import analyze_experiment

dbutils.widgets.text("experiment_id", "")
EXPERIMENT_ID = dbutils.widgets.get("experiment_id")

TARGET_ENV = "sandbox"
tables = TABLES[TARGET_ENV]
rec = load_experiment(spark, tables["registry"], EXPERIMENT_ID)
print(rec.name, "|", rec.status, "|", rec.hypothesis)
print("Regra de decisão pré-registrada:", rec.decision_rule)"""),
py("""from datetime import timedelta
date_start = str(rec.start_date - timedelta(days=rec.pre_window_days + 7))
date_end   = str(rec.end_date)
panel, _ = load_city_panel(spark, date_start, date_end)

treated = [u["city_norm"] for u in rec.treated_units]
donors  = [u["city_norm"] for u in rec.control_units]
window  = ExperimentWindow(start_date=rec.start_date, end_date=rec.end_date,
                           pre_window_days=rec.pre_window_days)"""),
py("""# ---- revalidação do controle fixo (habilita o DiD complementar) -------------
# O certificado do notebook 02 só atesta o passado; aqui checamos tendências
# paralelas na janela PRÉ recente deste experimento. Reprovou -> DiD é omitido
# da triangulação (SCM/ASCM/SDID não dependem do controle fixo).
from supply_experiments.spark_io import load_fixed_control
from supply_experiments.design.control_selection import revalidate_fixed_control

did_control = None
fixed = load_fixed_control(spark, tables["registry"])
if fixed:
    val = revalidate_fixed_control(panel, fixed, window_days=90,
                                   holidays=NATIONAL_HOLIDAYS,
                                   target_exclude=set(treated),
                                   end_date=window.pre_end)  # só o pré-período
    print(val.summary())
    if val.passed:
        did_control = [c for c in fixed if c not in set(treated)]
else:
    print("Sem controle fixo no registry — DiD complementar será omitido.")"""),
py("""report = analyze_experiment(
    panel, rec.experiment_id, treated, donors, window,
    primary_kpi=rec.primary_kpi, guardrail_kpis=rec.guardrail_kpis,
    alpha=0.10, holidays=NATIONAL_HOLIDAYS,
    did_control=did_control,
    run_conformal=True,
)   # levanta ValueError (anti-peeking) se o experimento ainda não terminou
print(report.summary())"""),
py("""display(report.to_frame())
if not report.guardrails.empty:
    display(report.guardrails)"""),
py("""# persiste resultados
import json
from datetime import datetime
rows = report.to_frame().assign(
    experiment_id=rec.experiment_id, kpi=rec.primary_kpi,
    triangulation=report.triangulation, analyzed_at=datetime.utcnow(),
    warnings=json.dumps(report.warnings, ensure_ascii=False),
)
spark.createDataFrame(rows).write.format("delta").mode("append") \\
    .saveAsTable(tables["results"])
print("Resultados persistidos em", tables["results"])"""),
], "notebooks/04_analyze_experiment.ipynb")

# =============================================================================
# 05 — Calibração A/A (job recorrente)
# =============================================================================
nb([
md("""# 05 — Calibração A/A (certificado do framework)
Job recorrente (e obrigatório após qualquer mudança de estimador): roda N
experimentos nulos em dados históricos reais e verifica FPR ≈ α, p-values
uniformes e viés ≈ 0. Persiste o certificado em tabela auditável."""),
py("""%pip install /Workspace/Shared/supply_experiments/supply_experiments-1.0.2-py3-none-any.whl --quiet --force-reinstall --no-deps
dbutils.library.restartPython()"""),
py("""from supply_experiments.spark_io import load_city_panel, TABLES
from supply_experiments.calibration.aa import run_aa_calibration
from supply_experiments.estimators.scm import fit_scm
from supply_experiments.estimators.ascm import fit_ascm
from supply_experiments.estimators.sdid import fit_sdid
import supply_experiments

TARGET_ENV = "sandbox"
tables = TABLES[TARGET_ENV]
panel, stats = load_city_panel(spark, "2025-06-01", "2026-06-30")
eligible = [c for c in stats.index if c in panel.cities][:60]"""),
py("""from datetime import datetime
rows = []
for name, fn in [("scm", fit_scm), ("ascm", fit_ascm), ("sdid", fit_sdid)]:
    aa = run_aa_calibration(panel, eligible, fn, pre_days=120, post_days=35,
                            n_runs=200, n_treated=2, n_donors=15, alpha=0.10)
    print(f"{name}: FPR={aa.fpr:.1%} IC[{aa.fpr_ci[0]:.1%},{aa.fpr_ci[1]:.1%}] "
          f"KS p={aa.ks_p_value:.3f} viés={aa.median_att_bias:+.2%} "
          f"-> {'CALIBRADO' if aa.passed else 'REPROVADO'}")
    rows.append({"run_at": datetime.utcnow(), "estimator": name,
                 "package_version": supply_experiments.__version__,
                 "n_runs": aa.n_runs, "alpha": aa.alpha, "fpr": aa.fpr,
                 "fpr_ci_low": aa.fpr_ci[0], "fpr_ci_high": aa.fpr_ci[1],
                 "ks_p_value": aa.ks_p_value, "median_bias": aa.median_att_bias,
                 "passed": aa.passed})
spark.createDataFrame(rows).write.format("delta").mode("append") \\
    .saveAsTable(tables["calibration"])"""),
], "notebooks/05_aa_calibration.ipynb")
