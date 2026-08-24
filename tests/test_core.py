"""Testes do core estatístico.

Além de testes unitários, inclui testes de CALIBRAÇÃO: os estimadores devem
recuperar efeitos conhecidos em DGPs sintéticos, e a inferência deve controlar
o erro tipo I. Se um refactor quebrar a matemática, esses testes pegam.
"""

import numpy as np
import pandas as pd
import pytest

from supply_experiments.calibration.aa import run_aa_calibration
from supply_experiments.design.control_selection import select_fixed_control
from supply_experiments.design.spillover import (
    EligibilityCriteria,
    eligible_cities,
    max_zero_run,
    spillover_exclusions,
)
from supply_experiments.estimators.ascm import fit_ascm
from supply_experiments.estimators.did import fit_did_panel
from supply_experiments.estimators.scm import fit_scm
from supply_experiments.estimators.sdid import fit_sdid
from supply_experiments.inference.bootstrap import (
    UnsupportedFewTreatedClustersError,
    wild_cluster_bootstrap,
)
from supply_experiments.inference.conformal import conformal_inference
from supply_experiments.inference.permutation import placebo_inference
from supply_experiments.io.registry import ExperimentRecord, minimum_donors_for_alpha
from supply_experiments.metrics import _ratio_and_var, ratio_did
from supply_experiments.panel import CityPanel, ExperimentWindow
from supply_experiments.reporting import analyze_experiment, benjamini_hochberg
from supply_experiments.synthetic import make_synthetic_panel


def split_panel(panel, treated, pre_days=120, post_days=35, start_idx=300):
    donors = [c for c in panel.cities if c not in set(treated)]
    y = panel.aggregate(treated).to_numpy()
    Yd = panel.outcome[donors].to_numpy()
    pre = slice(start_idx - pre_days, start_idx)
    post = slice(start_idx, start_idx + post_days)
    return y[pre], Yd[pre], y[post], Yd[post], donors


# ---------------------------------------------------------------------------
# Estimadores: recuperação de efeito conhecido
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fit_fn,tol", [(fit_scm, 0.025), (fit_ascm, 0.025), (fit_sdid, 0.025)])
def test_estimator_recovers_effect(fit_fn, tol):
    true_eff = 0.08
    panel = make_synthetic_panel(seed=1, treat_effect=true_eff,
                                 treated=["CITY_03", "CITY_07"], treat_start_idx=300)
    args = split_panel(panel, ["CITY_03", "CITY_07"])
    fit = fit_fn(*args)
    assert fit.success
    assert abs(fit.att_pct - true_eff) < tol, f"{fit.method}: {fit.att_pct:.3f} vs {true_eff}"


@pytest.mark.parametrize("fit_fn", [fit_scm, fit_ascm, fit_sdid])
def test_estimator_null_effect_near_zero(fit_fn):
    atts = []
    for seed in (2, 22, 42):
        panel = make_synthetic_panel(seed=seed)
        args = split_panel(panel, ["CITY_05"])
        fit = fit_fn(*args)
        assert fit.success
        atts.append(fit.att_pct)
    assert abs(float(np.median(atts))) < 0.05


def test_scm_weights_simplex():
    panel = make_synthetic_panel(seed=3)
    args = split_panel(panel, ["CITY_00"])
    fit = fit_scm(*args)
    w = fit.w
    assert np.all(w >= -1e-9)
    assert abs(w.sum() - 1.0) < 1e-6


def test_sdid_has_regularized_time_weights():
    panel = make_synthetic_panel(seed=4)
    args = split_panel(panel, ["CITY_01"])
    fit = fit_sdid(*args)
    lam = np.array(fit.extras["time_weights"])
    assert abs(lam.sum() - 1.0) < 1e-6 and np.all(lam >= -1e-9)
    assert fit.extras["zeta"] > 0


def test_ascm_bias_correction_differs_from_scm():
    """ASCM deve divergir do SCM quando há desbalanceamento pré (fit imperfeito)."""
    panel = make_synthetic_panel(seed=5, n_cities=12)
    args = split_panel(panel, ["CITY_00"])
    scm = fit_scm(*args)
    ascm = fit_ascm(*args)
    # correção não pode ser a média do resíduo pré (bug antigo zerava gap pré)
    assert not np.allclose(ascm.y_synth_post, scm.y_synth_post)


# ---------------------------------------------------------------------------
# Inferência
# ---------------------------------------------------------------------------

def test_placebo_pvalue_small_under_effect():
    panel = make_synthetic_panel(seed=6, treat_effect=0.15,
                                 treated=["CITY_02"], treat_start_idx=300, n_cities=25)
    args = split_panel(panel, ["CITY_02"])
    inf = placebo_inference(fit_scm, *args)
    assert inf.p_value <= 0.10


def test_group_placebos_match_number_of_treated_cities():
    panel = make_synthetic_panel(seed=61, n_cities=14, treat_effect=0.10,
                                 treated=["CITY_02", "CITY_05"], treat_start_idx=300)
    args = split_panel(panel, ["CITY_02", "CITY_05"])
    inf = placebo_inference(fit_scm, *args, n_treated_units=2,
                            max_group_placebos=20, seed=7)
    assert 8 <= inf.n_placebos <= 20
    assert "grupos de 2" in inf.note
    assert np.isfinite(inf.p_value)


def test_group_placebos_enforce_the_mean_sdid_estimand():
    panel = make_synthetic_panel(seed=62, n_cities=12, treat_effect=0.05,
                                 treated=["CITY_01", "CITY_03"], treat_start_idx=300)
    args = split_panel(panel, ["CITY_01", "CITY_03"])
    seen = []

    def recording_sdid(
        *fit_args, n_treated_units=1, treated_aggregation="sum"
    ):
        seen.append((n_treated_units, treated_aggregation))
        return fit_sdid(
            *fit_args,
            n_treated_units=n_treated_units,
            treated_aggregation=treated_aggregation,
        )

    placebo_inference(
        recording_sdid,
        *args,
        fit_kwargs={"n_treated_units": 999, "treated_aggregation": "sum"},
        n_treated_units=2,
        max_group_placebos=5,
        seed=7,
    )
    assert seen and set(seen) == {(2, "mean")}


def test_failed_real_fit_does_not_report_significant_permutation_pvalue():
    inf = placebo_inference(
        fit_scm,
        np.zeros(12), np.ones((12, 3)),
        np.ones(4), np.ones((4, 3)), ["a", "b", "c"],
    )
    assert np.isnan(inf.p_value)
    assert "Ajuste real falhou" in inf.note


@pytest.mark.slow
def test_aa_calibration_supports_multiple_treated_cities():
    panel = make_synthetic_panel(seed=63, n_cities=16, n_days=380)
    aa = run_aa_calibration(
        panel, panel.cities, fit_scm, pre_days=120, post_days=30,
        n_runs=12, n_treated=2, n_donors=8, min_valid_runs=10,
        max_group_placebos=10,
    )
    assert aa.requested_runs == 12
    assert len(aa.details) == 12
    assert aa.n_runs >= 10
    valid = aa.details.loc[aa.details["valid"]]
    assert len(valid) == aa.n_runs
    assert valid["n_placebos"].between(8, 10).all()
    assert aa.invalid_rate == pytest.approx((12 - aa.n_runs) / 12)


@pytest.mark.slow
def test_placebo_pvalue_large_under_null():
    ps = []
    for seed in range(5):
        panel = make_synthetic_panel(seed=100 + seed, n_cities=25)
        args = split_panel(panel, ["CITY_02"])
        ps.append(placebo_inference(fit_scm, *args).p_value)
    assert np.median(ps) > 0.15  # sob nulo, p-values não devem concentrar perto de 0


def test_conformal_covers_truth():
    true_eff = 0.10
    panel = make_synthetic_panel(seed=7, treat_effect=true_eff,
                                 treated=["CITY_04"], treat_start_idx=300, n_cities=15)
    args = split_panel(panel, ["CITY_04"])
    conf = conformal_inference(fit_scm, *args[:5], alpha=0.10,
                               rel_grid=np.linspace(-0.1, 0.3, 21))
    assert conf.ci_lower_pct <= true_eff <= conf.ci_upper_pct
    assert conf.p_value <= 0.20  # efeito de 10% deve ser detectável


def test_wild_cluster_bootstrap_fails_closed_with_few_treated_clusters():
    panel = make_synthetic_panel(seed=8, treat_effect=0.12,
                                 treated=["CITY_01", "CITY_02"], treat_start_idx=300)
    w = ExperimentWindow(start_date=panel.index[300].date(),
                         end_date=panel.index[334].date(), pre_window_days=120)
    controls = [c for c in panel.cities if c not in ("CITY_01", "CITY_02")][:12]
    fit = fit_did_panel(panel, ["CITY_01", "CITY_02"], controls, w)
    assert fit.success
    assert abs(fit.att_pct - 0.12) < 0.04
    with pytest.raises(UnsupportedFewTreatedClustersError, match="found 2"):
        wild_cluster_bootstrap(fit, n_boot=399, seed=1)


@pytest.mark.slow
def test_wild_cluster_bootstrap_diagnostic_is_deterministic_above_misuse_cutoff():
    panel = make_synthetic_panel(seed=200, n_cities=20)
    w = ExperimentWindow(start_date=panel.index[300].date(),
                         end_date=panel.index[334].date(), pre_window_days=120)
    treated = ["CITY_01", "CITY_02", "CITY_03", "CITY_04"]
    fit = fit_did_panel(
        panel, treated, [c for c in panel.cities if c not in treated][:12], w,
    )
    first = wild_cluster_bootstrap(fit, n_boot=199, seed=7)
    second = wild_cluster_bootstrap(fit, n_boot=199, seed=7)
    assert not first.valid_for_inference and first.n_treated_clusters == 4
    assert "diagnostic only" in first.diagnostic
    assert first.p_value == second.p_value
    assert np.array_equal(first.boot_t, second.boot_t)


# ---------------------------------------------------------------------------
# Métricas de razão / BH / spillover / seleção de controle
# ---------------------------------------------------------------------------

def test_ratio_did_detects_rate_shift():
    rng = np.random.default_rng(9)
    T = 200
    idx = pd.date_range("2025-01-01", periods=T)
    den_t = pd.Series(rng.poisson(2000, T).astype(float), index=idx)
    den_c = pd.Series(rng.poisson(2000, T).astype(float), index=idx)
    pre = np.arange(T) < 150
    post = ~pre
    rate_t = np.where(pre, 0.05, 0.07)  # +2pp no pós
    num_t = pd.Series(rng.binomial(den_t.astype(int), rate_t).astype(float), index=idx)
    num_c = pd.Series(rng.binomial(den_c.astype(int), 0.05).astype(float), index=idx)
    r = ratio_did(num_t, den_t, num_c, den_c, pre, post)
    assert abs(r.effect_abs - 0.02) < 0.006
    assert r.p_value < 0.01


def test_ratio_variance_requires_two_nonempty_blocks():
    ratio, variance = _ratio_and_var(pd.Series([1.0] * 5), pd.Series([10.0] * 5), block_days=7)
    assert ratio == 0.1
    assert np.isnan(variance)


def test_bh_correction():
    sig = benjamini_hochberg({"a": 0.001, "b": 0.04, "c": 0.90}, q=0.10)
    assert sig["a"] and not sig["c"]


def test_spillover_radius_and_adjacency():
    coords = {"GUARULHOS": (-23.46, -46.53), "OSASCO": (-23.53, -46.79),
              "SAO PAULO": (-23.55, -46.63), "CURITIBA": (-25.43, -49.27)}
    clean, excl = spillover_exclusions(
        ["GUARULHOS"], ["SAO PAULO", "CURITIBA", "OSASCO"], coords, radius_km=40,
        adjacency={"GUARULHOS": {"OSASCO"}})
    assert "CURITIBA" in clean
    assert "SAO PAULO" not in clean          # ~13km de Guarulhos
    reasons = {c: r for c, r, _ in excl}
    assert "OSASCO" in reasons


def test_spillover_adjacency_is_symmetric():
    clean, _ = spillover_exclusions(
        ["GUARULHOS"], ["OSASCO"], {},
        adjacency={"OSASCO": {"GUARULHOS"}},
    )
    assert clean == []


def test_eligibility_requires_quality_columns():
    stats = pd.DataFrame({"nonzero_days": [100], "avg_daily_gmv": [20_000.0]})
    with pytest.raises(ValueError, match="colunas obrigatórias"):
        eligible_cities(stats, EligibilityCriteria())


def test_max_zero_run():
    assert max_zero_run(np.array([1, 0, 0, 0, 2, 0])) == 3


def test_registry_gate_matches_permutation_granularity():
    record = ExperimentRecord(
        experiment_id="exp", name="teste", hypothesis="hipótese", status="approved",
        design_method="scm", primary_kpi="gmv", guardrail_kpis=[],
        start_date=pd.Timestamp("2026-01-10").date(), end_date=pd.Timestamp("2026-01-20").date(),
        pre_window_days=30,
        treated_units=[{"city_norm": "A"}],
        control_units=[{"city_norm": f"C{i}"} for i in range(8)],
        mde_estimated=0.01, expected_effect=0.05, decision_rule="decidir",
    )
    assert minimum_donors_for_alpha(0.10) == 9
    assert any("mínimo 9" in p for p in record.validate_for_approval())


@pytest.mark.slow
def test_control_selection_holdout_is_not_optimized():
    panel = make_synthetic_panel(seed=10, n_cities=25, n_days=365)
    states = {c: ["SP", "RJ", "MG", "BA", "RS"][i % 5] for i, c in enumerate(panel.cities)}
    s2r = {"SP": "Sudeste", "RJ": "Sudeste", "MG": "Sudeste", "BA": "Nordeste", "RS": "Sul"}
    res = select_fixed_control(panel, panel.cities, states, s2r, max_cities=6)
    assert 3 <= len(res.cities) <= 6
    assert np.isfinite(res.holdout_p_value)
    assert res.holdout_corr > 0.8  # DGP com fator comum: controle deve rastrear o alvo


# ---------------------------------------------------------------------------
# Pipeline fim-a-fim
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_analyze_experiment_end_to_end_and_antipeeking():
    true_eff = 0.10
    panel = make_synthetic_panel(seed=11, n_cities=25, treat_effect=true_eff,
                                 treated=["CITY_03", "CITY_06"], treat_start_idx=300)
    w = ExperimentWindow(start_date=panel.index[300].date(),
                         end_date=panel.index[334].date(), pre_window_days=120)
    donors = [c for c in panel.cities if c not in ("CITY_03", "CITY_06")][:18]

    # anti-peeking: análise antes do fim deve falhar
    with pytest.raises(ValueError, match="anti-peeking"):
        analyze_experiment(panel, "exp-1", ["CITY_03", "CITY_06"], donors, w,
                           today=w.start_date, run_conformal=False)

    rep = analyze_experiment(panel, "exp-1", ["CITY_03", "CITY_06"], donors, w,
                             guardrail_kpis=["rupture_rate_order"],
                             today=pd.Timestamp("2026-06-01").date(),
                             run_conformal=False)
    frame = rep.to_frame()
    assert set(frame["method"]) >= {"scm", "ascm", "sdid"}
    atts = frame.set_index("method")["att_pct"]
    for m in ("scm", "ascm", "sdid"):
        assert abs(atts[m] - true_eff) < 0.04
    assert "DESCRIPTIVE AGREEMENT ONLY" in rep.triangulation
    assert rep.decision == "DIAGNOSTIC_ONLY"
    assert not rep.guardrails.empty


@pytest.mark.slow
def test_recommend_treated_sets_ranks_by_mde():
    from supply_experiments.design.treated_selection import (
        recommend_treated_sets,
        recommendations_frame,
    )

    panel = make_synthetic_panel(seed=12, n_cities=25, n_days=400)
    stats = pd.DataFrame({
        "sum_gmv": panel.outcome.sum(),
        "state_address": ["SP"] * len(panel.cities),
    })
    stats.index.name = "city_norm"

    recs = recommend_treated_sets(
        panel, stats, panel.cities, fit_scm,
        pre_days=200, post_days=35,
        n_treated=(2,), top_k_cities=6, top_sets=2,
        effect_grid=(0.0, 0.10, 0.15),
        n_sims_per_point=5, max_gmv_share_per_city=1.0,
    )
    assert 1 <= len(recs) <= 2
    for r in recs:
        assert len(r.cities) == 2
        assert r.n_donors >= 10
        assert np.isfinite(r.screen_rmspe)
        # doadoras não podem conter tratadas
        assert not set(r.cities) & set(r.donors)
    # ordenação: MDE definido vem antes de indefinido; crescente entre definidos
    mdes = [r.mde_80 for r in recs if r.mde_80 is not None]
    assert mdes == sorted(mdes)

    df = recommendations_frame(recs)
    assert list(df["rank"]) == list(range(1, len(recs) + 1))


@pytest.mark.slow
def test_recommend_treated_sets_must_include_and_states():
    from supply_experiments.design.treated_selection import recommend_treated_sets

    panel = make_synthetic_panel(seed=13, n_cities=20, n_days=380)
    states = (["SP"] * 14) + (["RJ"] * 6)
    stats = pd.DataFrame({"sum_gmv": panel.outcome.sum(), "state_address": states})

    recs = recommend_treated_sets(
        panel, stats, panel.cities, fit_scm,
        pre_days=180, post_days=30,
        n_treated=(2,), must_include=["CITY_00"], states=["SP"],
        top_k_cities=5, top_sets=1,
        effect_grid=(0.0, 0.10), n_sims_per_point=4,
        max_gmv_share_per_city=1.0,
    )
    sp = set(f"CITY_{i:02d}" for i in range(14))
    for r in recs:
        assert "CITY_00" in r.cities
        assert set(r.cities) <= sp

    # must_include bloqueada deve falhar explicitamente
    with pytest.raises(ValueError, match="must_include"):
        recommend_treated_sets(
            panel, stats, panel.cities, fit_scm,
            pre_days=180, post_days=30, n_treated=(2,),
            must_include=["CITY_19"], states=["SP"],
            max_gmv_share_per_city=1.0)


def test_revalidate_fixed_control_pass_and_fail():
    from supply_experiments.design.control_selection import revalidate_fixed_control

    panel = make_synthetic_panel(seed=14, n_cities=25, n_days=380)
    control = ["CITY_01", "CITY_04", "CITY_07", "CITY_10", "CITY_13"]

    val = revalidate_fixed_control(panel, control, window_days=90)
    assert val.passed and val.p_value <= 0.05
    assert val.corr > 0.5

    # injeta tendência divergente no controle nos últimos 90 dias -> reprova
    bad = panel.outcome.copy()
    T = len(bad)
    ramp = 1.0 + 0.6 * np.arange(90) / 90.0
    for c in control:
        bad.iloc[T - 90:, bad.columns.get_loc(c)] *= ramp
    panel_bad = CityPanel(outcome=bad)
    val_bad = revalidate_fixed_control(panel_bad, control, window_days=90)
    assert not val_bad.passed

    # end_date corta o pós: com a divergência toda depois do corte, o painel
    # contaminado deve dar resultado idêntico ao painel limpo
    cut = panel.index[T - 91].date()
    val_pre_bad = revalidate_fixed_control(panel_bad, control, window_days=90,
                                           end_date=cut)
    val_pre_ok = revalidate_fixed_control(panel, control, window_days=90,
                                          end_date=cut)
    assert np.isclose(val_pre_bad.p_value, val_pre_ok.p_value)
    assert np.isclose(val_pre_bad.corr, val_pre_ok.corr)
