# supply_experiments — Framework de Experimentação Geográfica (iFood Groceries)

Framework de experimentação por cidades com rigor estatístico mensurável e escalável:
biblioteca Python testada + notebooks Databricks finos + registry v2 com pré-registro
como contrato + calibração A/A como certificado permanente.

## Por que reescrever

Diagnóstico do framework anterior (4 notebooks):

| # | Problema | Consequência | Solução aqui |
|---|----------|--------------|--------------|
| P0 | t-test OLS em séries agregadas (autocorrelação ignorada, n efetivo ≈ 2 unidades) | **FPR medido em A/A: 43% a α=5%** | Inferência por permutação (Abadie), conformal (CWZ 2021), wild cluster bootstrap |
| P0 | p-value de tendências paralelas otimizado na seleção do controle (pre-testing) | Controle overfitado ao período pré | Holdout temporal: otimiza no treino, aceita no holdout; p-value fora do score |
| P0 | Target "Brasil" inclui tratadas e o próprio controle | Efeito vaza para o benchmark | Target exclui controle + tratadas ativas (registry) |
| P1 | "ASCM" somava resíduo médio (zera gap pré por construção) | Fit aparente inflado, sem correção real | ASCM ridge fiel (Ben-Michael 2021), λ por LOO-CV |
| P1 | "SDID" com pesos temporais lineares arbitrários | Sem as garantias do paper | SDID fiel (Arkhangelsky 2021): ζ, pesos unitários e temporais otimizados |
| P1 | Sem análise de poder | Experimentos natimortos (MDE > efeito esperado) | Power gate por simulação; `save_experiment` recusa design sem poder |
| P2 | Razões (rupture rate) via OLS na taxa diária | Variância errada | Razão de somas + método delta em blocos semanais |
| P2 | Pesos SC em JSON no `experiment_description` | Frágil, quebra silenciosa | `weight DOUBLE` no struct de unidades do registry v2 |
| P2 | Código duplicado 4x, sem testes | Divergências silenciosas | 1 pacote, 20 testes (incluindo testes de calibração estatística) |
| P2 | Múltiplos KPIs sem correção; peeking possível | Inflação de falsos positivos | BH nos guardrails; anti-peeking estrutural em `analyze_experiment` |
| P2 | Sem tratamento de spillover | Metropolitana contamina doadoras | Exclusão por raio (haversine) + adjacência explícita |

## Certificado de calibração (A/A, painel sintético 40 cidades × 430 dias)

```
MÉTODO ANTIGO (OLS agregado, t clássico):
  FPR @ α=0.05: 43.3%   (esperado: 5%)   ← quase metade dos "significativos" é falsa
  FPR @ α=0.10: 49.2%   (esperado: 10%)

MÉTODO NOVO (SCM + permutação in-space):
  FPR @ α=0.10:  7.5%   IC95% [3.5%, 13.8%]   ← calibrado
  KS p-value (uniformidade dos p-values): 0.486
  Viés mediano do ATT placebo: +0.04%

CURVA DE PODER (2 tratadas, 18 doadoras, 35 dias, SCM):
  δ=3% → 36% | δ=5% → 76% | δ=8% → 88% | δ=12% → 100%   (MDE@80% = 8%)
```

Reproduza com `python calibration_certificate.py`. O notebook `05_aa_calibration`
roda o mesmo procedimento **nos dados reais** e persiste o certificado — rode-o
após qualquer mudança de estimador (teste de regressão estatístico).

## Arquitetura

```
supply_experiments/
├── panel.py                  # CityPanel (outcome + num/den p/ razões), ExperimentWindow
├── estimators/
│   ├── scm.py                # SCM Abadie; simplex via FISTA + projeção exata (Duchi 2008)
│   ├── ascm.py               # Ridge-ASCM fiel (Ben-Michael, Feller & Rothstein 2021)
│   ├── sdid.py               # SDID fiel (Arkhangelsky et al. 2021), ζ + Frank-Wolfe
│   └── did.py                # DiD painel nível-cidade, within-FE, escala normalizada
├── inference/
│   ├── permutation.py        # p-value razão RMSPE pós/pré (Abadie) — inferência primária
│   ├── conformal.py          # ICs por inversão de teste (Chernozhukov, Wüthrich & Zhu 2021)
│   └── bootstrap.py          # Wild cluster bootstrap-t restrito; pesos Webb se G<12
├── design/
│   ├── power.py              # poder por simulação em janelas históricas; MDE@80%
│   ├── control_selection.py  # greedy+swaps no treino; teste F só no holdout
│   └── spillover.py          # exclusão por raio/adjacência; elegibilidade (zero-runs, CV)
├── calibration/aa.py         # runner A/A: FPR + IC Clopper-Pearson + KS + viés
├── metrics.py                # ratio DiD com método delta (blocos semanais)
├── reporting.py              # analyze_experiment: triangulação + BH + anti-peeking
└── spark_io.py               # build_orders_base ÚNICO, load_city_panel, registry v2

notebooks/  (clientes finos, ~30 linhas de lógica cada)
├── 01_etl.ipynb                       # painel + elegibilidade
├── 02_fixed_control_selection.ipynb   # controle fixo com holdout
├── 03_design_experiment.ipynb         # spillover + POWER GATE + pré-registro
├── 04_analyze_experiment.ipynb        # triangulação + persistência
└── 05_aa_calibration.ipynb            # certificado recorrente nos dados reais

tests/test_core.py            # 20 testes, incluindo recuperação de efeito e calibração
```

## Fluxo de um experimento

1. **Design** (`03_design_experiment`): tratadas propostas → donor pool limpo de
   spillover → `power_analysis` estima MDE → `save_experiment` **recusa** se
   MDE > efeito esperado, hipótese vazia, regra de decisão vazia ou <8 doadoras.
2. **Execução**: intervenção roda; ninguém analisa (anti-peeking é `ValueError`).
3. **Análise** (`04_analyze_experiment`): SCM+ASCM+SDID com permutação, conformal,
   DiD/WCB opcional, guardrails delta+BH, veredito de triangulação
   (CONCORDANTE / PARCIAL / DIVERGENTE).
4. **Decisão**: contra a `decision_rule` pré-registrada — nunca ad hoc.

## Decisões estatísticas importantes

- **α padrão 0.10 para permutação**: com J doadoras o p-value mínimo é 1/(J+1);
  com 15 doadoras, 0.0625 — α=0.05 exigiria ≥20 doadoras. O relatório avisa
  quando o donor pool limita a granularidade.
- **Estatística RMSPE-ratio** como primária (robusta a placebos com fit pré ruim);
  p(|ATT|) reportado para transparência.
- **DiD em nível de cidade com y normalizado pela média pré**: sem isso, within-FE
  remove nível mas não escala e o τ é dominado pelas cidades grandes.
- **FISTA no simplex, não SLSQP**: SLSQP declarava sucesso com objetivo ~20x pior
  em instâncias com doadoras de escalas díspares (bug encontrado nos testes).

## Migração

1. Publicar o wheel: `pip wheel .` → `/Workspace/Shared/supply_experiments/`.
2. Rodar `01_etl` e `05_aa_calibration` (sandbox) — obter o certificado nos dados reais.
3. Reproduzir o último experimento real com `04_analyze_experiment` e comparar com
   o resultado antigo (esperar p-values bem mais conservadores).
4. Congelar os notebooks antigos; novos experimentos só via registry v2.

## Referências

Abadie, Diamond & Hainmueller (2010, JASA); Abadie (2021, JEL); Arkhangelsky et al.
(2021, AER); Ben-Michael, Feller & Rothstein (2021, JASA); Chernozhukov, Wüthrich &
Zhu (2021, JASA); Cameron, Gelbach & Miller (2008, REStat); MacKinnon & Webb (2018);
Duchi et al. (2008, ICML).
