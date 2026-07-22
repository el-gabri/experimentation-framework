from typing import Callable, Dict

from supply_experiments.estimators.ascm import fit_ascm
from supply_experiments.estimators.did import DiDFit, fit_did_panel
from supply_experiments.estimators.scm import SCMFit, fit_scm
from supply_experiments.estimators.sdid import fit_sdid

# fit_scm/fit_ascm/fit_sdid têm assinaturas extras distintas além dos 5 args
# posicionais comuns (y_pre, Yd_pre, y_post, Yd_post, donor_names) — daí o
# Callable[..., SCMFit] genérico em vez de um Protocol estrito por parâmetro.
ESTIMATORS: Dict[str, Callable[..., SCMFit]] = {
    "scm": fit_scm, "ascm": fit_ascm, "sdid": fit_sdid,
}
