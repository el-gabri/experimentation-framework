from supply_experiments.estimators.scm import fit_scm, SCMFit
from supply_experiments.estimators.ascm import fit_ascm
from supply_experiments.estimators.sdid import fit_sdid
from supply_experiments.estimators.did import fit_did_panel, DiDFit

ESTIMATORS = {"scm": fit_scm, "ascm": fit_ascm, "sdid": fit_sdid}
