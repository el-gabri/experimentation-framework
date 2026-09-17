"""Actual Spark transformations; optional locally and mandatory in the Spark CI job."""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("pyspark")
from pyspark.sql import SparkSession  # noqa: E402

from supply_experiments.io.etl import load_city_panel  # noqa: E402


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    warehouse = tmp_path_factory.mktemp("spark-warehouse").as_uri()
    session = (SparkSession.builder.master("local[2]").appName("geo-etl-contract-tests")
               .config("spark.ui.enabled", "false")
               .config("spark.sql.shuffle.partitions", "2")
               .config("spark.sql.session.timeZone", "UTC")
               .config("spark.sql.warehouse.dir", warehouse).getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def _source(spark, rows, *, telemetry=True):
    # SQL values exercise JVM transformations without Python-worker startup.
    flag = ", named_struct('has_rupture_ticket', flag) AS otif_struct" if telemetry else ""
    spark.sql(f"""
        SELECT dt, order_id, order_id AS merchant_id, merchant_city, merchant_state,
               order_total, 'CONCLUDED' AS last_status,
               'Mercado' AS groceries_classification,
               'Marketplace' AS marketplace_classification {flag}
        FROM VALUES {rows}
        AS input(dt, order_id, merchant_city, merchant_state, order_total, flag)
    """).createOrReplaceTempView("geo_etl_fixture")


def _load(spark, **kwargs):
    return load_city_panel(spark, "2025-01-01", "2025-01-03",
                           source_table="geo_etl_fixture", **kwargs)


def test_spark_does_not_merge_same_named_cities_across_states(spark):
    _source(spark, """
        ('2025-01-01', '1', 'Santa Helena', 'PR', 100., false),
        ('2025-01-01', '2', ' SANTA HELENA ', 'ma', 200., false),
        ('2025-01-01', '3', 'Santa Helena', CAST(NULL AS STRING), 50., false)
    """)
    with pytest.raises(ValueError, match="ambíguo entre estados"):
        _load(spark, assume_missing_city_days_zero=True)


def test_spark_preserves_known_zeros_and_only_explicitly_fills_absent_days(spark):
    _source(spark, """
        ('2025-01-01', '1', 'A', 'SP', 100., false),
        ('2025-01-03', '2', 'A', 'SP', 120., false)
    """)
    with pytest.raises(ValueError, match="city-days ausentes"):
        _load(spark)
    panel, stats = _load(spark, assume_missing_city_days_zero=True, require_rupture_telemetry=True)
    np.testing.assert_array_equal(panel.outcome.A, [100., 0., 120.])
    assert stats.loc["A", "sum_gmv"] == 220
    assert stats.loc["A", "state_address"] == "SP"
    assert stats.loc["A", "rupture_rate"] == 0


@pytest.mark.parametrize("value", ["CAST(NULL AS DOUBLE)", "CAST('NaN' AS DOUBLE)",
                                  "CAST('Infinity' AS DOUBLE)"])
def test_spark_rejects_invalid_money_before_sum_can_hide_it(spark, value):
    _source(spark, f"""
        ('2025-01-01', '1', 'A', 'SP', 100., false),
        ('2025-01-01', '2', 'A', 'SP', {value}, false)
    """)
    with pytest.raises(ValueError, match="order_total observado"):
        _load(spark, assume_missing_city_days_zero=True)


@pytest.mark.parametrize("telemetry,second_flag", [(False, "false"),
                                                  (True, "CAST(NULL AS BOOLEAN)"),
                                                  (True, "false")])
def test_spark_missing_or_partial_rupture_cannot_become_observed_zero(spark, telemetry, second_flag):
    _source(spark, f"""
        ('2025-01-01', '1', 'A', 'SP', 100., CAST(NULL AS BOOLEAN)),
        ('2025-01-03', '2', 'A', 'SP', 120., {second_flag})
    """, telemetry=telemetry)
    with pytest.warns(UserWarning, match="indisponível"):
        panel, stats = _load(spark, assume_missing_city_days_zero=True)
    assert not panel.numerators
    assert np.isnan(stats.loc["A", "rupture_rate"])
    with pytest.raises(ValueError, match="telemetria de ruptura incompleta"):
        _load(spark, assume_missing_city_days_zero=True, require_rupture_telemetry=True)
