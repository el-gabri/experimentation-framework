"""Análise de experimento: triangulação multi-estimador + regras de decisão.

analyze_experiment() é o ponto de entrada da medição. Ele:
  1. Exige que o experimento tenha terminado (anti-peeking estrutural).
  2. Roda SCM, ASCM e SDID com inferência por permutação; conformal p/ o principal.
  3. Roda DiD em painel com wild cluster bootstrap se houver controle não-doador.
  4. KPIs de razão via método delta (guardrails), com correção BH.
  5. Emite veredito de triangulação: CONCORDANTE / DIVERGENTE / INCONCLUSIVO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from supply_experiments.estimators import ESTIMATORS, fit_did_panel
from supply_experiments.inference.bootstrap import wild_cluster_bootstrap
from supply_experiments.inference.conformal import conformal_inference
from supply_experiments.inference.permutation import placebo_inference
from supply_experiments.metrics import ratio_did
from supply_experiments.panel import CityPanel, ExperimentWindow


def benjamini_hochberg(p_values: Dict[str, float], q: float = 0.10) -> Dict[str, bool]:
    """Retorna, por KPI, se é significativo sob FDR q."""
    items = [(k, v) for k, v in p_values.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out = {k: False for k in p_values}
    threshold_idx = -1
    for i, (_, p) in enumerate(items, start=1):
        if p <= q * i / m:
            threshold_idx = i
    for i, (k, _) in enumerate(items, start=1):
        out[k] = i <= threshold_idx
    return out


@dataclass
class EstimatorRow:
    method: str
    att_pct: float
    p_value: float
    p_value_att: float
    ci: Optional[tuple] = None
    pre_rmspe: float = np.nan
    n_placebos: int = 0
    note: str = ""


@dataclass
class ExperimentReport:
    experiment_id: str
    primary_kpi: str
    window: ExperimentWindow
    estimators: List[EstimatorRow]
    guardrails: pd.DataFrame
    triangulation: str
    decision_inputs: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "method": e.method, "att_pct": e.att_pct, "p_value": e.p_value,
            "p_value_att": e.p_value_att,
            "ci_low": e.ci[0] if e.ci else np.nan,
            "ci_high": e.ci[1] if e.ci else np.nan,
            "pre_rmspe": e.pre_rmspe, "n_placebos": e.n_placebos, "note": e.note,
        } for e in self.estimators])

    def summary(self) -> str:
        lines = [f"=== Experimento {self.experiment_id} — KPI primário: {self.primary_kpi} ==="]
        lines.append(f"Janela: pré {self.window.pre_start}..{self.window.pre_end} | "
                     f"pós {self.window.start_date}..{self.window.end_date}")
        for e in self.estimators:
            ci = f" IC95%[{e.ci[0]:+.1%}, {e.ci[1]:+.1%}]" if e.ci and np.isfinite(e.ci[0]) else ""
            lines.append(f"  {e.method:>10}: ATT={e.att_pct:+.2%} | p(RMSPE-ratio)={e.p_value:.3f} "
                         f"| p(|ATT|)={e.p_value_att:.3f}{ci} | fit pré RMSPE={e.pre_rmspe:.1%}")
        lines.append(f"Triangulação: {self.triangulation}")
        for w in self.warnings:
            lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


def _triangulate(rows: List[EstimatorRow], alpha: float, agreement_tol: float = 0.05) -> str:
    ok = [r for r in rows if np.isfinite(r.att_pct)]
    if len(ok) < 2:
        return "INCONCLUSIVO: menos de 2 estimadores válidos"
    atts = np.array([r.att_pct for r in ok])
    sig = [r.p_value <= alpha for r in ok if np.isfinite(r.p_value)]
    same_sign = np.all(atts > 0) or np.all(atts < 0)
    spread = float(atts.max() - atts.min())
    if same_sign and spread < agreement_tol and (all(sig) or not any(sig)):
        verdict = "CONCORDANTE"
    elif same_sign and spread < agreement_tol:
        verdict = "PARCIAL: mesmo sinal e magnitude, significância divergente"
    else:
        verdict = "DIVERGENTE: investigar antes de decidir (fit? spillover? outlier?)"
    return (f"{verdict} (tolerância de spread={agreement_tol:.0%}) | ATTs: "
            + ", ".join(f"{r.method}={r.att_pct:+.1%}" for r in ok))


def analyze_experiment(
    panel: CityPanel,
    experiment_id: str,
    treated: Sequence[str],
    donors: Sequence[str],
    window: ExperimentWindow,
    primary_kpi: str = "gmv",
    guardrail_kpis: Sequence[str] = (),
    alpha: float = 0.10,
    fdr_q: float = 0.10,
    holidays: Optional[Sequence[str]] = None,
    did_control: Optional[Sequence[str]] = None,   # controle fixo p/ DiD (opcional)
    run_conformal: bool = True,
    conformal_method: str = "scm",   # scm|sdid — ASCM não é compatível (ver inference.conformal)
    agreement_tol: float = 0.05,
    today: Optional[date] = None,
    allow_interim: bool = False,
    max_group_placebos: Optional[int] = 30,
) -> ExperimentReport:
    warnings: List[str] = []
    today = today or date.today()
    if today <= window.end_date and not allow_interim:
        raise ValueError(
            f"Experimento termina em {window.end_date}; análise bloqueada até lá "
            f"(anti-peeking). Use allow_interim=True apenas para monitoramento de "
            f"guardrails — nunca para decisão de negócio."
        )

    pslice = panel.slice_for(treated, donors, window)
    donors = pslice.donor_names
    pre_mask, post_mask = window.masks(panel.index)

    if len(donors) < 10:
        warnings.append(f"Só {len(donors)} doadoras: p-value mínimo por permutação = "
                        f"{1 / (len(donors) + 1):.3f}. Considere ampliar o donor pool.")

    rows: List[EstimatorRow] = []
    # ESTIMATORS têm assinaturas extras distintas (ex.: n_treated_units do SDID)
    # além dos 5 args posicionais comuns — daí o Callable genérico.
    for name, fn in ESTIMATORS.items():
        fit_kwargs = {"n_treated_units": len(treated)} if name == "sdid" else {}
        inf = placebo_inference(
            fn, *pslice.as_args(), fit_kwargs=fit_kwargs,
            n_treated_units=len(treated), max_group_placebos=max_group_placebos,
        )
        fit = inf.real_fit  # já ajustado dentro de placebo_inference — evita refit
        assert fit is not None, f"placebo_inference não retornou real_fit para {name}"
        ci = None
        if run_conformal and name == conformal_method:
            try:
                y_pre, Yd_pre, y_post, Yd_post, names = pslice.as_args()
                conf = conformal_inference(fn, y_pre, Yd_pre, y_post, Yd_post, names,
                                           alpha=alpha, fit_kwargs=fit_kwargs)
                ci = (conf.ci_lower_pct, conf.ci_upper_pct)
            except Exception as e:
                warnings.append(f"Conformal falhou p/ {name}: {e}")
        rows.append(EstimatorRow(
            method=name, att_pct=fit.att_pct, p_value=inf.p_value,
            p_value_att=inf.p_value_att, ci=ci, pre_rmspe=fit.pre_rmspe,
            n_placebos=inf.n_placebos, note=inf.note,
        ))
        if inf.note:
            warnings.append(f"{name}: {inf.note}")

    if did_control:
        didf = fit_did_panel(panel, treated, did_control, window, holidays)
        if didf.success:
            wc = wild_cluster_bootstrap(didf, n_boot=999, seed=11)
            rows.append(EstimatorRow(
                method="did_panel", att_pct=didf.att_pct, p_value=wc.p_value,
                p_value_att=wc.p_value,
                note=f"wild cluster bootstrap ({wc.weight_type}, G={wc.n_clusters})",
            ))

    # guardrails de razão (método delta) + BH
    g_rows = []
    for kpi in guardrail_kpis:
        if kpi not in panel.numerators or kpi not in panel.denominators:
            warnings.append(f"Guardrail '{kpi}' sem numerador/denominador no painel — pulado")
            continue
        num, den = panel.numerators[kpi], panel.denominators[kpi]
        missing = [c for c in list(treated) + list(donors)
                   if c not in num.columns or c not in den.columns]
        if missing:
            warnings.append(
                f"Guardrail '{kpi}' sem dados para cidades {sorted(set(missing))} — pulado"
            )
            continue
        nt = num[list(treated)].sum(axis=1)
        dt = den[list(treated)].sum(axis=1)
        nc = num[donors].sum(axis=1)
        dc = den[donors].sum(axis=1)
        r = ratio_did(nt, dt, nc, dc, pre_mask, post_mask)
        g_rows.append({"kpi": kpi, "effect_abs": r.effect_abs, "effect_rel": r.effect_rel,
                       "se": r.se, "p_value": r.p_value})
    guardrails = pd.DataFrame(g_rows)
    if not guardrails.empty:
        sig = benjamini_hochberg(dict(zip(guardrails["kpi"], guardrails["p_value"])), q=fdr_q)
        guardrails["significant_bh"] = guardrails["kpi"].map(sig)

    return ExperimentReport(
        experiment_id=experiment_id, primary_kpi=primary_kpi, window=window,
        estimators=rows, guardrails=guardrails,
        triangulation=_triangulate([r for r in rows if r.method in ("scm", "ascm", "sdid")],
                                   alpha, agreement_tol),
        decision_inputs={"alpha": alpha, "fdr_q": fdr_q, "n_donors": len(donors)},
        warnings=warnings,
    )
