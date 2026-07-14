"""Certificado de calibração: método ANTIGO vs NOVO sob A/A (efeito nulo).

ANTIGO: OLS y ~ 1 + group + post + group*post + DOW nas séries agregadas
        (tratado-soma vs controle-soma), p-value do t clássico — o que o
        experiments_results.ipynb original fazia.
NOVO:   inferência por permutação in-space (Abadie) sobre SCM.

Sob o nulo, a taxa de rejeição deve ser ~α. Rodamos 120 experimentos A/A
num painel sintético realista (fator comum + DOW + AR(1) diário).
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/framework")
from tests.test_core import make_synthetic_panel
from supply_experiments.calibration.aa import run_aa_calibration
from supply_experiments.estimators.scm import fit_scm
from supply_experiments.design.power import power_analysis


# ---------------------------------------------------------------------------
# Método ANTIGO (reprodução fiel do estimate_effect_ols do notebook original)
# ---------------------------------------------------------------------------
def old_ols_pvalue(dates, y_control, y_treated, start_date):
    from scipy.stats import t as t_dist
    y0, y1 = np.asarray(y_control, float), np.asarray(y_treated, float)
    T = len(y0)
    dt = pd.to_datetime(dates)
    post = (dt.date >= start_date).astype(float)
    y = np.concatenate([y0, y1])
    group = np.concatenate([np.zeros(T), np.ones(T)])
    post2 = np.concatenate([post, post])
    inter = group * post2
    cols = [np.ones(2 * T), group, post2, inter]
    dow = pd.Series(dt).dt.dayofweek.to_numpy()
    dow = np.concatenate([dow, dow])
    for k in range(1, 7):
        cols.append((dow == k).astype(float))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    df2 = n - k
    sigma2 = float(resid @ resid) / df2
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(sigma2 * XtX_inv[3, 3])
    t_stat = beta[3] / se
    return float(2 * t_dist.sf(abs(t_stat), df=df2))


def main():
    rng = np.random.default_rng(2026)
    panel = make_synthetic_panel(n_cities=40, n_days=430, seed=99)
    cities = panel.cities
    PRE, POST = 120, 35
    N_RUNS = 120
    T = len(panel.index)

    # ---------------- A/A: método antigo -----------------------------------
    old_rejections_05, old_rejections_10 = 0, 0
    for _ in range(N_RUNS):
        pick = rng.choice(cities, size=12, replace=False).tolist()
        treated, control = pick[:2], pick[2:]
        start = int(rng.integers(PRE, T - POST))
        sl = slice(start - PRE, start + POST)
        dates = panel.index[sl]
        y_t = panel.aggregate(treated).to_numpy()[sl]
        y_c = panel.aggregate(control).to_numpy()[sl]
        p = old_ols_pvalue(dates, y_c, y_t, panel.index[start].date())
        old_rejections_05 += p <= 0.05
        old_rejections_10 += p <= 0.10

    print("=" * 72)
    print("CERTIFICADO DE CALIBRAÇÃO — A/A (efeito verdadeiro = 0)")
    print("=" * 72)
    print(f"\nPainel sintético: 40 cidades, {T} dias, fator comum + DOW + AR(1)")
    print(f"Runs A/A: {N_RUNS} | pré={PRE}d, pós={POST}d, 2 tratadas\n")
    print("MÉTODO ANTIGO (OLS agregado, t clássico — experiments_results.ipynb):")
    print(f"  FPR @ α=0.05: {old_rejections_05 / N_RUNS:6.1%}   (esperado: 5%)")
    print(f"  FPR @ α=0.10: {old_rejections_10 / N_RUNS:6.1%}   (esperado: 10%)")

    # ---------------- A/A: método novo --------------------------------------
    aa = run_aa_calibration(panel, cities, fit_scm, PRE, POST,
                            n_runs=N_RUNS, n_treated=2, n_donors=15,
                            alpha=0.10, seed=7)
    print("\nMÉTODO NOVO (SCM + permutação in-space):")
    print(f"  FPR @ α=0.10: {aa.fpr:6.1%}   IC95% binomial [{aa.fpr_ci[0]:.1%}, {aa.fpr_ci[1]:.1%}]")
    print(f"  KS p-value (p-values ~ U(0,1)): {aa.ks_p_value:.3f}")
    print(f"  Viés mediano do ATT placebo: {aa.median_att_bias:+.2%}")
    print(f"  CALIBRADO: {'SIM ✓' if aa.passed else 'NÃO ✗'}")

    # ---------------- Curva de poder -----------------------------------------
    treated = ["CITY_03", "CITY_07"]
    donors = [c for c in cities if c not in treated][:18]
    pw = power_analysis(panel, treated, donors, fit_scm, PRE, POST,
                        effect_grid=(0.0, 0.03, 0.05, 0.08, 0.12),
                        n_sims_per_point=25, alpha=0.10, seed=3)
    print("\n" + "=" * 72)
    print("CURVA DE PODER (design: 2 tratadas, 18 doadoras, 35 dias, SCM)")
    print("=" * 72)
    for d in pw.effect_grid:
        bar = "█" * int(pw.power[d] * 40)
        print(f"  δ={d:5.1%} | poder={pw.power[d]:6.1%} {bar}")
    print(f"\n  FPR (δ=0): {pw.fpr:.1%}  |  MDE@80%: "
          f"{pw.mde_80:.1%}" if pw.mde_80 else
          f"\n  FPR (δ=0): {pw.fpr:.1%}  |  MDE@80%: > {max(pw.effect_grid):.0%}")
    print("\n  Regra do framework: experimento só é aprovado se MDE <= efeito esperado.")


if __name__ == "__main__":
    main()
