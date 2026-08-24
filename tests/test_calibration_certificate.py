"""Fast contract tests for the public calibration-certificate artifacts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

import calibration_certificate as certificate_module
import supply_experiments.design.power as power_module
from supply_experiments.calibration.aa import AACalibration
from supply_experiments.design.power import PowerResult
from supply_experiments.synthetic import make_synthetic_panel


def _sample_results():
    rejects = [True] * 9 + [False] * 111
    aa = AACalibration(
        n_runs=120,
        alpha=0.10,
        fpr=0.075,
        fpr_ci=(0.035, 0.138),
        ks_p_value=0.486,
        median_att_bias=0.0004,
        passed=True,
        details=pd.DataFrame(
            {
                "p_value": [0.05 if rejected else 0.50 for rejected in rejects],
                "reject": rejects,
            }
        ),
    )
    power_values = {0.0: 0.08, 0.03: 0.20, 0.05: 0.68, 0.08: 0.92, 0.12: 1.0}
    power_rows = []
    for effect, rejection_rate in power_values.items():
        n_rejections = round(rejection_rate * 25)
        power_rows.extend(
            {
                "delta": effect,
                "p_value": 0.05 if simulation < n_rejections else 0.50,
                "reject": simulation < n_rejections,
            }
            for simulation in range(25)
        )
    power = PowerResult(
        effect_grid=[0.0, 0.03, 0.05, 0.08, 0.12],
        power=power_values,
        fpr=0.08,
        mde_80=0.08,
        n_sims_per_point=25,
        alpha=0.10,
        details=pd.DataFrame(power_rows),
    )
    return aa, power


def _sample_certificate():
    aa, power = _sample_results()
    naive_p_values = [0.01] * 52 + [0.08] * 7 + [0.50] * 61
    return certificate_module.build_certificate(naive_p_values, aa, power)


def test_machine_certificate_contract_is_explicit_and_synthetic_only():
    certificate = _sample_certificate()

    assert certificate["schema_version"] == "1.0.0"
    assert certificate["certificate_type"] == "synthetic_statistical_calibration"
    assert certificate["data_scope"]["kind"] == "synthetic_only"
    assert certificate["data_scope"]["contains_real_data"] is False
    assert certificate["reproducibility"]["random_seeds"] == {
        "panel": 99,
        "naive_aa": 2026,
        "framework_aa": 7,
        "power": 3,
        "permutation_group_sampling": 123,
    }

    framework = certificate["results"]["framework_aa"]
    assert framework["requested_runs"] == 120
    assert framework["valid_runs"] == 120
    assert framework["failed_runs"] == 0
    assert framework["median_placebo_att_bias_role"] == "diagnostic_only"
    assert framework["pass_criteria"]["nominal_alpha_inside_fpr_interval"] is True
    baseline = certificate["results"]["naive_baseline"]
    assert baseline["complete"] is True
    assert baseline["failed_runs"] == 0
    assert [point["rejections"] for point in baseline["false_positive_rates"]] == [
        52,
        59,
    ]

    power_points = certificate["results"]["power_analysis"]["points"]
    assert [point["effect"] for point in power_points] == [0.0, 0.03, 0.05, 0.08, 0.12]
    assert certificate["results"]["power_analysis"]["mde_status"] == (
        "estimated_on_tested_grid"
    )
    assert certificate["results"]["power_analysis"]["complete"] is True
    assert certificate["verdict"]["passed"] is True
    assert set(certificate["software"]) == {
        "python",
        "supply_experiments",
        "numpy",
        "pandas",
        "scipy",
    }

    # A strict JSON encoder must accept the whole public payload.
    json.dumps(certificate, allow_nan=False)


def test_text_and_json_artifacts_share_one_payload_and_are_repeatable(tmp_path):
    certificate = _sample_certificate()

    text_path, json_path = certificate_module.write_certificate(certificate, tmp_path)
    first_text = text_path.read_bytes()
    first_json = json_path.read_bytes()
    certificate_module.write_certificate(certificate, tmp_path)

    assert text_path.name == "calibration_certificate.txt"
    assert json_path.name == "calibration_certificate.json"
    assert text_path.read_bytes() == first_text
    assert json_path.read_bytes() == first_json
    assert first_text.endswith(b"\n")
    assert first_json.endswith(b"\n")

    loaded = json.loads(first_json.decode("utf-8"))
    assert loaded == certificate
    human_report = first_text.decode("utf-8")
    assert "Synthetic-only panel" in human_report
    assert f"{loaded['results']['framework_aa']['false_positive_rate']:.1%}" in human_report


def test_unavailable_diagnostics_are_portable_json_nulls(tmp_path):
    aa = AACalibration(
        n_runs=0,
        alpha=0.10,
        fpr=np.nan,
        fpr_ci=(np.nan, np.nan),
        ks_p_value=np.nan,
        median_att_bias=np.nan,
        passed=False,
        details=pd.DataFrame(
            {"p_value": [np.nan] * 120, "reject": [False] * 120}
        ),
    )
    power_rows = [
        {"delta": effect, "p_value": np.nan, "reject": False}
        for effect in (0.0, 0.03, 0.05, 0.08, 0.12)
        for _ in range(25)
    ]
    power = PowerResult(
        effect_grid=[0.0, 0.03, 0.05, 0.08, 0.12],
        power={effect: np.nan for effect in (0.0, 0.03, 0.05, 0.08, 0.12)},
        fpr=np.nan,
        mde_80=None,
        n_sims_per_point=25,
        alpha=0.10,
        details=pd.DataFrame(power_rows),
    )
    certificate = certificate_module.build_certificate([np.nan] * 120, aa, power)

    framework = certificate["results"]["framework_aa"]
    assert certificate["results"]["naive_baseline"]["complete"] is False
    assert certificate["results"]["naive_baseline"]["failed_runs"] == 120
    assert framework["false_positive_rate"] is None
    assert framework["ks_uniformity_p_value"] is None
    assert certificate["results"]["power_analysis"]["mde_grid_at_80pct_power"] is None
    assert certificate["results"]["power_analysis"]["mde_status"] == (
        "unavailable_incomplete_simulations"
    )
    assert certificate["results"]["power_analysis"]["complete"] is False
    assert certificate["verdict"]["passed"] is False
    text_path, json_path = certificate_module.write_certificate(certificate, tmp_path)
    assert "unavailable" in text_path.read_text(encoding="utf-8")
    assert "MDE@80%: >" not in text_path.read_text(encoding="utf-8")
    assert "NaN" not in json_path.read_text(encoding="utf-8")
    json.loads(json_path.read_text(encoding="utf-8"))


def test_cli_writes_requested_directory_without_running_slow_calibration(
    tmp_path, monkeypatch, capsys
):
    certificate = _sample_certificate()
    monkeypatch.setattr(certificate_module, "run_calibration", lambda: certificate)

    exit_code = certificate_module.main(["--output-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "calibration_certificate.txt").is_file()
    assert (tmp_path / "calibration_certificate.json").is_file()
    stdout = capsys.readouterr().out
    assert "STATISTICAL CALIBRATION CERTIFICATE" in stdout
    assert "calibration_certificate.json" in stdout


def test_cli_returns_failure_after_writing_a_failed_certificate(
    tmp_path, monkeypatch
):
    certificate = _sample_certificate()
    certificate["verdict"]["passed"] = False
    monkeypatch.setattr(certificate_module, "run_calibration", lambda: certificate)

    exit_code = certificate_module.main(["--output-dir", str(tmp_path)])

    assert exit_code == 1
    assert (tmp_path / "calibration_certificate.txt").is_file()
    assert (tmp_path / "calibration_certificate.json").is_file()


def test_simulation_passes_the_published_permutation_seed(monkeypatch):
    seen = []

    def fake_placebo_inference(*args, seed=None, **kwargs):
        seen.append(seed)
        return SimpleNamespace(p_value=0.5, placebo_atts=[0.0], n_placebos=2)

    monkeypatch.setattr(power_module, "placebo_inference", fake_placebo_inference)
    panel = make_synthetic_panel(n_cities=4, n_days=30, seed=1)

    power_module.simulate_once(
        panel,
        treated=["CITY_00"],
        donors=["CITY_01", "CITY_02", "CITY_03"],
        fit_fn=lambda *args: None,
        start_idx=15,
        pre_days=10,
        post_days=5,
        delta=0.0,
        alpha=0.10,
        permutation_seed=456,
    )

    assert seen == [456]
