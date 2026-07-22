# Plano de Melhorias e Refatoração — supply_experiments

Objetivo: evoluir a POC para uma biblioteca pública de experimentação geográfica
de referência — o melhor "modelo de previsão do contrafactual" possível, com
inferência calibrada, código limpo e reprodutível.

Prioridades: **P0** = bloqueia publicação pública · **P1** = correção/qualidade ·
**P2** = performance · **P3** = novas capacidades.

---

## 1. P0 — Publicação segura (higiene do repositório)

- [x] Remover menções ao nome da empresa (`setup.py`, `__init__.py`, README).
- [x] Remover artefatos de build versionados (`build/`, `dist/`, `*.egg-info`,
      `__pycache__`, `.DS_Store`) — os wheels e o `PKG-INFO` continham os
      metadados antigos.
- [x] Substituir nomes de tabelas internas em `spark_io.py` por placeholders
      (`catalog.schema.orders` etc.).
- [x] Adicionar `.gitignore`.
- [ ] **Recriar o histórico git antes de publicar**: o commit inicial ainda
      contém os metadados antigos (wheels, PKG-INFO, table names). Como há um
      único commit, o mais simples é iniciar um repositório novo a partir da
      árvore atual (`git init` + commit único), em vez de reescrever histórico.
- [ ] Adicionar `LICENSE` (MIT ou Apache-2.0) — obrigatório para repo público.
- [ ] Varredura final de strings sensíveis nos notebooks (`notebooks/*.ipynb`
      têm outputs? limpar outputs com `nbstripout` antes de publicar).
- [ ] Limpar referências a "versão anterior do framework" nos docstrings
      (`scm.py`, `ascm.py`, `sdid.py`, `permutation.py`, `control_selection.py`,
      `make_notebooks.py`, `calibration_certificate.py`): reformular como
      comparação com "baseline ingênuo", como já feito no README. Não expõem a
      empresa, mas são contexto interno sem valor para o leitor externo.
- [ ] `calibration_certificate.py:16` tem `sys.path.insert(0, "/home/claude/framework")`
      — caminho de máquina alheia; remover (o pacote instalado via `pip install -e .`
      já resolve os imports).

## 2. P1 — Correções e riscos identificados na leitura do código

### 2.1 Bugs / dívidas pontuais

| Onde | Problema | Correção proposta |
|---|---|---|
| `estimators/did.py:94-95` | Loop morto `for arr in [...]: pass` | Remover. |
| `estimators/did.py:111` | `att_abs = tau * mean(scale) * len(treated)` assume escala média — aproximação que erra quando as tratadas têm tamanhos díspares | Reagregar: `att_abs = tau * Σ_c scale[c]` (soma das médias pré das tratadas). |
| `spark_io.py:load_experiment` | `where(f"experiment_id = '{id}'")` — injeção de SQL se o id vier de input externo | Usar `F.col("experiment_id") == F.lit(experiment_id)`. |
| `spark_io.py:save_experiment` | `datetime.utcnow()` deprecado (Python 3.12+) | `datetime.now(timezone.utc)`. |
| `panel.py:__post_init__` | Dias faltantes preenchidos com `0.0` silenciosamente — zero é indistinguível de buraco de dado e vaza para estimadores e elegibilidade | Preencher com `NaN` + validação explícita (`max_zero_run` já existe); oferecer `fill_value` opcional. |
| `metrics.py:_ratio_and_var` | Dias além de `B*block_days` entram em `R` mas ficam de fora da variância em blocos | Incluir resto no último bloco ou truncar ambos consistentemente. |
| `inference/conformal.py` | Para o ASCM, `y_synth_pre` é o SCM puro (correção ridge só no pós) — os resíduos do conformal ignoram a correção, descalibrando o IC do ASCM | Expor trajetória ajustada completa no fit ou restringir conformal ao SCM/SDID e documentar. |
| `reporting.py:analyze_experiment` | Cada estimador é ajustado 2× (dentro de `placebo_inference` e de novo para o fit real) | `placebo_inference` deve retornar o fit real; reaproveitar. |
| `reporting.py:_triangulate` | Limiar de concordância `spread < 0.05` hardcoded | Parametrizar (`agreement_tol`), reportar no verdict. |
| `design/power.py:_draw_windows` | Comentário promete "espaçados + jitter", mas o código faz amostragem uniforme sem espaçamento — janelas placebo sobrepostas correlacionam as simulações e subestimam a variância do poder | Implementar amostragem com espaçamento mínimo (ex.: `post_days // 2`) ou documentar a limitação. |
| `calibration/aa.py` | `passed` exige apenas `IC ∋ α`; com poucas runs o IC é largo e "passa" fácil | Exigir também `n ≥ 100` e KS p-value > 0.01; tornar critérios configuráveis. |
| `estimators/scm.py:_package` | `mean_pre == 0` vira `1.0` via `or` — mascara painéis degenerados | Validar e falhar explicitamente (`success=False`). |

### 2.2 Empacotamento e qualidade de engenharia

- Migrar `setup.py` → `pyproject.toml` (PEP 621) com extras:
  `pip install supply-experiments[spark]` (pyspark opcional), `[dev]`
  (pytest, ruff, mypy).
- Adotar **src layout** (`src/supply_experiments/`) para evitar imports
  acidentais do diretório de trabalho.
- **CI (GitHub Actions)**: lint (ruff) + type-check (mypy, gradual) + pytest em
  3.9–3.13 + job noturno com os testes estatísticos lentos (calibração A/A).
- **Type hints completos** e `py.typed` marker.
- Congelar seeds e tolerâncias dos testes estatísticos; marcar com
  `@pytest.mark.slow` os de calibração.
- `make_notebooks.py` como única fonte dos notebooks: gerar em CI e falhar se
  os `.ipynb` versionados divergirem (ou migrar para `jupytext` e versionar só
  `.py`).

## 3. P1 — Refatoração estrutural (API)

### 3.1 Interface unificada de estimadores

Hoje `SCMFit` serve SCM/ASCM/SDID e `DiDFit` é outra coisa; `_package`/`_failed`
são semiprivados importados entre módulos. Proposta:

```python
class Estimator(Protocol):
    name: str
    def fit(self, data: PanelSlice) -> EstimatorFit: ...

@dataclass
class EstimatorFit:          # substitui SCMFit e DiDFit
    att: float; att_pct: float
    counterfactual: TrajectoryPair   # pré/pós
    diagnostics: Diagnostics         # pre_rmspe, corr, extras
    success: bool
```

- `PanelSlice` encapsula `(y_pre, Yd_pre, y_post, Yd_post, donor_names)` — hoje
  essa 5-tupla é repetida em ~10 assinaturas.
- `placebo_inference`, `conformal_inference`, `power_analysis` e
  `analyze_experiment` passam a receber `Estimator`, eliminando `fit_fn` +
  `fit_kwargs` soltos.
- Registrar estimadores num dict (`ESTIMATORS = {"scm": ..., ...}`) para que
  `analyze_experiment` e o registry usem a mesma nomenclatura.

### 3.2 Configuração

- `Config` dataclass (ou pydantic-settings) para: tabelas, feriados, critérios
  de elegibilidade, α, q do BH, raios de spillover. Hoje há constantes
  espalhadas (`TABLES`, `NATIONAL_HOLIDAYS`, defaults duplicados).
- Feriados: substituir a lista hardcoded por integração opcional com o pacote
  `holidays` (parametrizado por país) com override manual.

### 3.3 Separação domínio × infraestrutura

- `spark_io.py` mistura três responsabilidades: ETL de pedidos, registry e
  resultados. Dividir em `io/etl.py`, `io/registry.py`, `io/results.py`, com o
  schema do orders base **configurável** (mapa de colunas), já que os nomes de
  colunas atuais são específicos de um warehouse particular.
- Publicar um **gerador de dados sintéticos** como módulo de primeira classe
  (`supply_experiments.synthetic`), promovendo `make_synthetic_panel` de
  `tests/` — vira o dataset de exemplo do repo público e desacopla
  `calibration_certificate.py` dos testes.

## 4. P2 — Performance

- **`ascm._loo_cv_lambda`**: hoje resolve um ridge por (doadora × período × λ) —
  O(J·T·|grid|) solves. Usar a identidade fechada do LOO para ridge
  (via SVD de X, computada 1×: erro LOO = resíduo/(1−h_jj)), reduzindo a
  1 SVD + operações vetorizadas. Ganho estimado: 10–100×.
- **`placebo_inference`**: paraleliza trivialmente por placebo (`joblib`);
  para o ASCM, fixar o λ escolhido no fit real em vez de re-otimizar por
  placebo (além de rápido, é o procedimento correto de permutação — mesma
  pipeline para tratada e placebos).
- **`conformal_inference`**: warm-start do FISTA com os pesos do fit anterior
  na busca em grade (os τ0 vizinhos têm soluções próximas); busca binária nas
  bordas do IC em vez de varredura linear + refinamento.
- **`power_analysis`**: reutilizar janelas e fits entre pontos do grid de δ
  (o fit dos placebos não depende de δ — só a série tratada muda). Isso corta
  o custo por ~|grid|×.
- Benchmark suite (`asv` ou pytest-benchmark) para não regredir.

## 5. P3 — Capacidades novas (o "melhor contrafactual possível")

Em ordem de retorno/custo:

1. **Placebo-in-time como diagnóstico padrão** do `analyze_experiment`:
   backdating do início do tratamento no pré — barato e altamente convincente.
2. **Sensibilidade leave-one-donor-out**: reestimar o ATT removendo cada
   doadora com peso > 5%; reportar intervalo de estabilidade (Abadie 2021
   recomenda). Detecta contrafactuais apoiados numa única cidade.
3. **Conformal para todos os estimadores** (hoje só ASCM ganha IC) + IC também
   por permutação (inversão do teste de RMSPE-ratio).
4. **Covariáveis no SCM** (V-weights de Abadie ou demeaned SCM de Ferman &
   Pinto 2021) — clima, feriados regionais, população.
5. **Matrix completion / interactive fixed effects** (Athey et al. 2021, JASA;
   Xu 2017 gsynth) como 4º estimador da triangulação — costuma dominar SCM em
   painéis longos.
6. **Adoção escalonada (staggered)**: hoje o desenho supõe um único bloco de
   tratamento; suportar múltiplas datas de início (SDID já tem extensão no
   paper; para DiD usar Callaway & Sant'Anna 2021).
7. **Inferência SDID fiel ao paper**: variância por placebo/bootstrap do
   Arkhangelsky et al. (§4) além da permutação genérica.
8. **Tratadas individuais em vez de agregadas**: rodar por cidade tratada e
   combinar (média ponderada + p-values combinados por Fisher/Stouffer) —
   ganha diagnóstico de heterogeneidade.
9. **Seleção de doadoras por similaridade** (clustering/DTW sobre séries
   normalizadas) como pré-filtro do donor pool, antes do fit.
10. **Relatório HTML** (`ExperimentReport.to_html()`): trajetórias observado ×
    contrafactual, distribuição de placebos, curva de poder — hoje o output é
    texto puro.
11. **CLI** (`supx design ...`, `supx analyze ...`, `supx calibrate`) para uso
    fora de notebooks.
12. **Docs**: mkdocs-material com tutorial ponta-a-ponta usando o gerador
    sintético (design → power gate → análise → decisão).

## 6. Sequenciamento sugerido

| Fase | Conteúdo | Resultado |
|---|---|---|
| 1 (agora) | Itens P0 restantes: LICENSE, histórico novo, limpeza de docstrings, nbstripout | Repo publicável |
| 2 | §2.1 (bugs) + §2.2 (pyproject, CI, ruff/mypy) | Base confiável |
| 3 | §3 (API unificada + synthetic como módulo) | Biblioteca coesa; breaking change → v2.0 |
| 4 | §4 (performance) + §5.1–5.3 (diagnósticos) | Análises rápidas e defensáveis |
| 5 | §5.4+ (novos estimadores, staggered, docs, CLI) | Ferramenta de referência |

## 7. Pontos fortes a preservar

- Inferência por permutação como primária e A/A como "certificado" recorrente —
  é o diferencial do projeto; nenhuma refatoração deve quebrar
  `calibration_certificate.py` (mantê-lo como teste de regressão estatística).
- Anti-peeking estrutural e power gate no `save_experiment` (design como
  contrato) — raro até em ferramentas maduras.
- Núcleo 100% pandas/numpy testável offline, Spark isolado em adapters.
- Holdout temporal na seleção de controle (sem pre-testing contamination).
