import numpy as np
import pandas as pd

from stratum.tools.tool_audit import run_audit


def test_detects_leak_and_named_id(binary_csv):
    result = run_audit(str(binary_csv), "target")
    kinds = {(flag["flag_type"], flag["column"]) for flag in result["flags"]}
    assert ("leakage", "leak") in kinds
    assert ("id_col", "customer_id") in kinds


def test_null_threshold_and_clean_continuous_not_id(tmp_path):
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "continuous": rng.normal(size=100),
        "nullable": [np.nan] * 25 + list(rng.normal(size=75)),
        "target": [0, 1] * 50,
    })
    path = tmp_path / "audit.csv"
    df.to_csv(path, index=False)
    result = run_audit(str(path), "target")
    assert any(f["flag_type"] == "null_pattern" and f["column"] == "nullable" for f in result["flags"])
    assert not any(f["flag_type"] == "id_col" and f["column"] == "continuous" for f in result["flags"])


def test_audit_detects_string_datetime_column(tmp_path):
    df = pd.DataFrame({
        "event": pd.date_range("2025-01-01", periods=20).astype(str),
        "feature": range(20),
        "target": [0, 1] * 10,
    })
    path = tmp_path / "audit_dates.csv"
    df.to_csv(path, index=False)
    result = run_audit(str(path), "target")
    assert any(f["flag_type"] == "temporal" and f["column"] == "event" for f in result["flags"])
    assert result["dataset_stats"]["datetime_columns"] == ["event"]
