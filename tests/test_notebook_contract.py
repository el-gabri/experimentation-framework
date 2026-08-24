"""Static checks for generated Databricks notebook templates."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_generated_notebook_python_cells_compile():
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 5
    for path in notebooks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(payload["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = cell["source"]
            if isinstance(source, list):
                source = "".join(source)
            if source.lstrip().startswith("%"):
                continue  # Databricks/IPython magic is not CPython syntax.
            compile(source, f"{path.name}:cell-{index}", "exec")


def test_notebook_templates_use_current_prerelease_and_bound_analysis():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "notebooks").glob("*.ipynb"))
    )
    assert "supply_experiments-1.0.3" not in text
    assert "supply_experiments-2.0.0a1" in text
    assert "design_spec=spec" in text
    assert "calibration=aa" in text
    assert "SELECTION_REPLAY_IMPLEMENTED = False" in text
