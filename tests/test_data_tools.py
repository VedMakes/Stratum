import numpy as np
import pandas as pd

from stratum.data.drift import compute_psi, run_drift
from stratum.data.profiler import run_profile
from stratum.data.splitter import run_split
from stratum.session import get_artifacts


def test_profile(binary_csv):
    result = run_profile(str(binary_csv), target="target")
    assert result["shape"] == [60, 6]
    assert result["columns"]
    assert "missing_patterns" in result
    assert "quasi_constant_columns" in result
    assert result["target_relationships"][0]["column"] in {"leak", "segment"}


def test_profile_detects_datetime_missing_groups_and_quasi_constants(tmp_path):
    rows = 100
    df = pd.DataFrame({
        "when": pd.date_range("2025-01-01", periods=rows).astype(str),
        "mostly_zero": [0] * 99 + [1],
        "a": [np.nan, np.nan, np.nan] + list(range(rows - 3)),
        "b": [np.nan, np.nan, np.nan] + list(range(rows - 3)),
        "c": [np.nan, np.nan, np.nan] + list(range(rows - 3)),
        "target": [0, 1] * (rows // 2),
    })
    path = tmp_path / "profile.csv"
    df.to_csv(path, index=False)
    result = run_profile(str(path), target="target")
    assert result["datetime_columns"] == ["when"]
    assert result["missing_patterns"]["structural_null_groups"]
    assert result["quasi_constant_columns"][0]["column"] == "mostly_zero"


def test_split_modes(binary_csv, tmp_path):
    stratified = run_split(str(binary_csv), "target")
    assert stratified["train_rows"] == 48
    temporal_df = pd.read_csv(binary_csv)
    temporal_df["when"] = pd.date_range("2025-01-01", periods=len(temporal_df))
    path = tmp_path / "temporal.csv"
    temporal_df.to_csv(path, index=False)
    temporal = run_split(str(path), "target", mode="temporal", time_col="when")
    train = pd.read_csv(temporal["train_path"])
    test = pd.read_csv(temporal["test_path"])
    assert pd.to_datetime(test["when"]).min() >= pd.to_datetime(train["when"]).max()
    assert temporal["split_id"]


def test_group_split_keeps_groups_disjoint_and_persists_manifest(tmp_path):
    df = pd.DataFrame({
        "entity": [f"g{i}" for i in range(12) for _ in range(5)],
        "feature": range(60),
        "target": [0, 1, 0, 1, 0] * 12,
    })
    path = tmp_path / "groups.csv"
    df.to_csv(path, index=False)

    result = run_split(str(path), "target", mode="group", group_col="entity", seed=7)

    train, test = pd.read_csv(result["train_path"]), pd.read_csv(result["test_path"])
    assert set(train["entity"]).isdisjoint(test["entity"])
    assert result["excluded_feature"] == "entity"
    manifest = next(
        artifact for artifact in get_artifacts(result["sid"], "split_manifest")
        if artifact["metadata"]["split_id"] == result["split_id"]
    )
    assert manifest["metadata"]["parameters"]["seed"] == 7
    assert manifest["metadata"]["train_row_fingerprint"]


def test_predefined_split_uses_and_removes_assignment_column(tmp_path):
    df = pd.DataFrame({
        "feature": range(10),
        "partition": ["train"] * 7 + ["test"] * 3,
        "target": [0, 1] * 5,
    })
    path = tmp_path / "predefined.csv"
    df.to_csv(path, index=False)

    result = run_split(
        str(path), "target", mode="predefined", split_col="partition", test_size=1.5
    )

    train, test = pd.read_csv(result["train_path"]), pd.read_csv(result["test_path"])
    assert len(train) == 7 and len(test) == 3
    assert "partition" not in train and "partition" not in test
    assert result["parameters"]["train_value"] == "train"


def test_predefined_split_rejects_duplicate_rows_across_partitions(tmp_path):
    path = tmp_path / "overlap.csv"
    pd.DataFrame({
        "feature": [1, 1, 2, 3],
        "partition": ["train", "test", "train", "test"],
        "target": [0, 0, 1, 1],
    }).to_csv(path, index=False)

    result = run_split(str(path), "target", mode="predefined", split_col="partition")

    assert result["error"] == "Predefined partitions contain 1 identical row value(s) across train and test."


def test_repeated_splits_keep_prior_artifacts_usable(binary_csv):
    from stratum.tools.tool_train import _load_saved_split

    first = run_split(str(binary_csv), "target", mode="stratified", seed=1)
    second = run_split(str(binary_csv), "target", mode="stratified", seed=2)

    assert first["train_path"] != second["train_path"]
    assert first["test_path"] != second["test_path"]
    train_df, test_df, manifest, error = _load_saved_split(
        str(binary_csv), "target", first["sid"], first["split_id"]
    )
    assert error is None
    assert len(train_df) == manifest["train_rows"]
    assert len(test_df) == manifest["test_rows"]


def test_temporal_gap_rows_are_excluded_between_partitions(tmp_path):
    df = pd.DataFrame({
        "when": pd.date_range("2025-01-01", periods=20).astype(str),
        "row_id": range(20),
        "target": [0, 1] * 10,
    })
    path = tmp_path / "temporal_gap.csv"
    df.to_csv(path, index=False)

    result = run_split(str(path), "target", mode="temporal", time_col="when", gap_rows=2)

    train, test = pd.read_csv(result["train_path"]), pd.read_csv(result["test_path"])
    assert train["row_id"].max() == 13
    assert test["row_id"].min() == 16
    assert result["parameters"]["gap_rows"] == 2


def test_stratified_split_warns_on_disguised_datetime(binary_csv, tmp_path):
    df = pd.read_csv(binary_csv)
    df["event"] = pd.date_range("2025-01-01", periods=len(df)).astype(str)
    path = tmp_path / "with_event.csv"
    df.to_csv(path, index=False)
    result = run_split(str(path), "target")
    assert "Datetime column 'event' found. Consider mode='temporal'." in result["warnings"]


def test_split_accumulates_stratification_datetime_and_size_warnings(tmp_path):
    df = pd.DataFrame({
        "event": pd.date_range("2025-01-01", periods=8).astype(str),
        "feature": range(8),
        "target": [0, 0, 0, 0, 0, 0, 0, 1],
    })
    path = tmp_path / "small_imbalanced.csv"
    df.to_csv(path, index=False)
    result = run_split(str(path), "target")
    assert "Stratification unavailable; used a deterministic random split." in result["warnings"]
    assert "Datetime column 'event' found. Consider mode='temporal'." in result["warnings"]
    assert any("Test set has only" in warning for warning in result["warnings"])
    assert any("Training set has only" in warning for warning in result["warnings"])


def test_temporal_split_allows_null_times_with_warning(tmp_path):
    df = pd.DataFrame({
        "when": list(pd.date_range("2025-01-01", periods=11).astype(str)) + [None],
        "feature": range(12),
        "target": [0, 1] * 6,
    })
    path = tmp_path / "temporal_nulls.csv"
    df.to_csv(path, index=False)
    result = run_split(str(path), "target", mode="temporal", time_col="when")
    test = pd.read_csv(result["test_path"])
    assert any("time_col has 1 null values" in warning for warning in result["warnings"])
    assert test["when"].isna().any()


def test_temporal_split_rejects_non_null_unparseable_times(tmp_path):
    df = pd.DataFrame({
        "when": ["2025-01-01", "not-a-date", "2025-01-03", None],
        "feature": range(4),
        "target": [0, 1, 0, 1],
    })
    path = tmp_path / "temporal_bad.csv"
    df.to_csv(path, index=False)
    result = run_split(str(path), "target", mode="temporal", time_col="when")
    assert result == {"error": "Column 'when' contains 1 unparseable values"}


def test_drift(binary_csv, tmp_path):
    current = pd.read_csv(binary_csv)
    current["age"] += 50
    path = tmp_path / "current.csv"
    current.to_csv(path, index=False)
    result = run_drift(str(binary_csv), str(path), target="target")
    assert result["overall_verdict"] in {"mild_drift", "significant_drift"}
    assert result["top_drifted"][0] == "age"
    assert compute_psi(np.arange(100), np.arange(100) + 100) > .2


def test_drift_handles_single_category_chi2(tmp_path):
    reference = pd.DataFrame({"constant": ["same"] * 20, "target": [0, 1] * 10})
    current = pd.DataFrame({"constant": ["same"] * 20, "target": [0, 1] * 10})
    ref_path = tmp_path / "single_ref.csv"
    cur_path = tmp_path / "single_cur.csv"
    reference.to_csv(ref_path, index=False)
    current.to_csv(cur_path, index=False)
    result = run_drift(str(ref_path), str(cur_path), target="target")
    column = result["columns"][0]
    assert result["overall_verdict"] == "no_drift"
    assert column["test"] == "chi2"
    assert column["statistic"] == 0.0
    assert column["p_value"] == 1.0
    assert column["cramers_v"] == 0.0


def test_drift_uses_max_psi_for_verdict(tmp_path):
    rows = 100
    reference = pd.DataFrame({
        f"stable_{i}": ["a", "b"] * (rows // 2)
        for i in range(19)
    })
    current = reference.copy()
    reference["shifted"] = ["a"] * 90 + ["b"] * 10
    current["shifted"] = ["a"] * 10 + ["b"] * 90
    reference["target"] = [0, 1] * (rows // 2)
    current["target"] = [0, 1] * (rows // 2)
    ref_path = tmp_path / "max_ref.csv"
    cur_path = tmp_path / "max_cur.csv"
    reference.to_csv(ref_path, index=False)
    current.to_csv(cur_path, index=False)
    result = run_drift(str(ref_path), str(cur_path), target="target")
    assert result["psi_overall"] < .2
    assert result["psi_max"] > .5
    assert result["psi_max_column"] == "shifted"
    assert result["overall_verdict"] == "significant_drift"


def test_drift_warns_on_dataset_size_disparity(tmp_path):
    reference = pd.DataFrame({"feature": range(100), "target": [0, 1] * 50})
    current = pd.DataFrame({"feature": range(9), "target": [0, 1, 0, 1, 0, 1, 0, 1, 0]})
    ref_path = tmp_path / "large_ref.csv"
    cur_path = tmp_path / "small_cur.csv"
    reference.to_csv(ref_path, index=False)
    current.to_csv(cur_path, index=False)
    result = run_drift(str(ref_path), str(cur_path), target="target")
    assert result["warnings"]
    assert "Dataset sizes differ significantly" in result["warnings"][0]
    assert result["note"] == "Target column excluded. Label shift detection requires separate analysis."
