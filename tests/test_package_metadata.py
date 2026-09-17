"""Release metadata invariants."""

from importlib.metadata import version

import pytest

import supply_experiments
from supply_experiments._version import require_runtime_version


def test_runtime_and_distribution_versions_match():
    assert supply_experiments.__version__ == "2.0.0a2"
    assert version("supply-experiments") == supply_experiments.__version__


def test_previous_implementation_artifacts_require_recalibration():
    with pytest.raises(ValueError, match="implementation_version diverge"):
        require_runtime_version("2.0.0a1")
