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
- [x] Substituir nomes de tabelas internas em `spark_io.py` (agora `io/tables.py`)
      por placeholders (`catalog.schema.orders` etc.).
- [x] Adicionar `.gitignore`.
- [x] Adicionar `LICENSE` (MIT).
- [x] Notebooks já estavam sem outputs versionados (verificado via `nbformat`) —
      nada para o `nbstripout` limpar.
- [x] Limpar referências a "versão anterior do framework" nos docstrings
      (`scm.py`, `ascm.py`, `sdid.py`, `permutation.py`, `control_selection.py`,
      `make_notebooks.py`, `calibration_certificate.py`) — reformuladas como
      comparação com "baseline ingênuo"/"implementações simplificadas".
- [x] Removido `sys.path.insert(0, "/home/claude/framework")` de
      `calibration_certificate.py` (caminho de outra máquina).
- [ ] **Recriar o histórico git antes de publicar**: o commit inicial ainda
      contém os metadados antigos (wheels, PKG-INFO, table names). Como há um
      único commit, o mais simples é iniciar um repositório novo a partir da
      árvore atual (`git init` + commit único), em vez de reescrever histórico.
      **Não executado automaticamente** — é uma operação destrutiva de git
      (reescreve/descarta histórico) fora do escopo de edições de arquivo; faça
      manualmente antes do primeiro push público.

## 2. P1 — Correções e riscos identificados na leitura do código

### 2.1 Bugs / dívidas pontuais — todos corrigidos

| Onde | Problema | Correção aplicada |
|---|---|---|
| `estimators/did.py:94-95` | Loop morto `for arr in [...]: pass` | [x] Removido. |
| `estimators/did.py:111` | `att_abs = tau * mean(scale) * len(treated)` assume escala média — aproximação que erra quando as tratadas têm tamanhos díspares | [x] Reagregado para `att_abs = tau * Σ_c scale[c]` (soma, não média, das escalas pré). |
| `spark_io.py:load_experiment` | `where(f"experiment_id = '{id}'")` — injeção de SQL se o id vier de input externo | [x] `F.col("experiment_id") == F.lit(experiment_id)` (agora em `io/registry.py`). |
| `spark_io.py:save_experiment` | `datetime.utcnow()` deprecado (Python 3.12+) | [x] `datetime.now(timezone.utc)`. |
| `panel.py:__post_init__` | Dias faltantes preenchidos com `0.0` silenciosamente | [x] Mantido `fill_value=0.0` como default (mudar para NaN propagaria por toda a pipeline numérica — risco desproporcional ao ganho), mas agora emite `warnings.warn` explícito com a contagem de dias e expõe `fill_value` como parâmetro (`np.nan` disponível para quem quiser tratar buracos explicitamente). |
| `metrics.py:_ratio_and_var` | Dias além de `B*block_days` entravam em `R` mas ficavam de fora da variância em blocos | [x] Último bloco absorve o resto — `R` e a variância agora cobrem exatamente os mesmos dias. |
| `inference/conformal.py` | Para o ASCM, `y_synth_pre` é o SCM puro (correção ridge só no pós) — IC descalibrado | [x] Documentado explicitamente no docstring (incompatibilidade real, não só cosmética) + `reporting.py` passou a rodar conformal só em SCM/SDID (`conformal_method` parametrizável). |
| `reporting.py:analyze_experiment` | Cada estimador era ajustado 2× | [x] `PlaceboInference.real_fit` carrega o fit já computado; `analyze_experiment` reaproveita em vez de rechamar `fit_fn`. |
| `reporting.py:_triangulate` | Limiar de concordância `spread < 0.05` hardcoded | [x] Parametrizado como `agreement_tol` (default 0.05), reportado no texto do veredito. |
| `design/power.py:_draw_windows` | Amostragem uniforme sem espaçamento apesar do comentário prometer isso | [x] Amostragem estratificada (uma janela por faixa de `[lo,hi)`) — reduz sobreposição entre simulações. |
| `calibration/aa.py` | `passed` exigia só `IC ∋ α`; com poucas runs "passa" fácil | [x] Agora exige também `n >= min_valid_runs` (100) e `KS p-value >= min_ks_p_value` (0.01), ambos configuráveis. |
| `estimators/scm.py:_package` | `mean_pre == 0` virava `1.0` via `or` — mascarava painéis degenerados | [x] Falha explicitamente (`success=False`) quando `|mean_pre| < 1e-9`. |

### 2.2 Empacotamento e qualidade de engenharia — concluído

- [x] Migrado `setup.py` → `pyproject.toml` (PEP 621) com extras `[spark]`
  (pyspark) e `[dev]` (pytest, ruff, mypy, nbformat).
- [x] Adotado **src layout** (`src/supply_experiments/`), reinstalado via
  `pip install -e .`.
- [x] **CI (GitHub Actions)** em `.github/workflows/ci.yml`: lint (ruff) +
  type-check (mypy) + pytest em 5 versões (3.9–3.13, testes rápidos) + job
  separado para os testes lentos e para reproduzir o certificado A/A.
- [x] `py.typed` marker adicionado; `mypy src` limpo (27 arquivos).
- [x] Testes estatísticos pesados marcados com `@pytest.mark.slow`
  (recuperação de efeito em loop, bootstrap null-calibrado, seleção de
  controle, pipeline fim-a-fim, recomendação de tratadas).
- [x] `make_notebooks.py` como fonte única verificada em CI (gerar e diffar
  contra os `.ipynb` versionados).

## 3. P1 — Refatoração estrutural (API) — implementado com escopo reduzido

**Decisão de escopo (tomada durante a execução, registrada aqui para o
próximo leitor):** a proposta original de §3.1 substituiria as assinaturas de
`fit_scm`/`fit_ascm`/`fit_sdid`/`fit_did_panel` por um `Protocol` único
recebendo `PanelSlice`. Isso tocaria ~10 arquivos do núcleo estatístico
(estimadores + inferência + design + testes) que são validados por
recuperação de efeito conhecido — o risco de introduzir um erro sutil de
sinal/eixo sem re-derivar a matemática manualmente era desproporcional ao
ganho de estilo. Optou-se por uma versão **aditiva**: as funções
`fit_scm`/`fit_ascm`/`fit_sdid`/`fit_did_panel` mantêm suas assinaturas
originais (continuam testadas e chamáveis exatamente como antes); a
unificação acontece na camada de orquestração, que é onde a duplicação
realmente doía. Não houve bump para v2.0 — a superfície pública antiga
continua válida.

### 3.1 Interface unificada — versão aditiva entregue

- [x] `PanelSlice` (em `panel.py`): dataclass com `y_pre, Yd_pre, y_post,
  Yd_post, donor_names` + `.as_args()` (5-tupla posicional) e `.fit(fit_fn)`.
  `CityPanel.slice_for(treated, donors, window)` constrói uma a partir de uma
  `ExperimentWindow`, substituindo o boilerplate que existia em
  `analyze_experiment` (máscaras pré/pós + agregação + indexação manual).
- [x] `ESTIMATORS` (`dict[str, Callable[..., SCMFit]]`) — **já existia** em
  `estimators/__init__.py`, criado mas nunca usado; agora `reporting.py`
  itera `ESTIMATORS.items()` em vez de uma lista hardcoded duplicando os
  mesmos três nomes.
- [ ] `Estimator` Protocol / `EstimatorFit` unificando `SCMFit` e `DiDFit`:
  não implementado (é a parte que foi conscientemente deixada de fora —
  ver decisão de escopo acima). Fica como follow-up caso o projeto quera
  investir em reescrever e re-validar a suite de recuperação de efeito.

### 3.2 Configuração — entregue como agregador opcional

- [x] `RunConfig` (`config.py`): dataclass bundlando `alpha`, `fdr_q`,
  `agreement_tol`, `min_donors`, `radius_km`, `eligibility`
  (`EligibilityCriteria`, que já existia) e `holidays`. É **aditivo**: nenhuma
  função passou a exigir `RunConfig`; os kwargs individuais continuam sendo a
  fonte de verdade dos defaults. Serve para um projeto declarar a config uma
  vez em vez de repetir `alpha=0.10` em cada chamada.
- [ ] Integração com o pacote `holidays` (calendário por país): não feita —
  `NATIONAL_HOLIDAYS` continua uma lista hardcoded em `io/tables.py`.

### 3.3 Separação domínio × infraestrutura — concluído

- [x] `spark_io.py` dividido em `io/tables.py` (nomes de tabela + feriados),
  `io/etl.py` (`build_orders_base`, `load_city_panel`) e `io/registry.py`
  (`ExperimentRecord`, DDL, `save_experiment`/`load_experiment`/
  `load_fixed_control`/`active_blocked_cities`). `spark_io.py` virou um shim
  de reexport (`from supply_experiments.io import ...`) para não quebrar os
  7 notebooks/`make_notebooks.py` que importam dele.
- [ ] Schema do orders base configurável (mapa de colunas): não feito —
  `build_orders_base` ainda assume os nomes de coluna do warehouse original
  (`merchant_city`, `groceries_classification` etc.). Fica para quando o
  framework for de fato reutilizado fora do contexto Groceries original.
- [x] `supply_experiments.synthetic` — `make_synthetic_panel` promovido de
  `tests/test_core.py`; `calibration_certificate.py` e os testes agora
  importam do mesmo lugar (zero duplicação do gerador sintético).

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
3. **Conformal para todos os estimadores** (hoje SCM/SDID ganham IC) + IC também
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
