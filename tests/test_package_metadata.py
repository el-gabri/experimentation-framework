"""Release metadata invariants."""

from importlib.metadata import version

import supply_experiments


def test_runtime_and_distribution_versions_match():
    assert supply_experiments.__version__ == "2.0.0a1"
    assert version("supply-experiments") == supply_experiments.__version__
