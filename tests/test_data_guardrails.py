import pandas as pd

from stratum.data.profiler import run_profile
from stratum.session import create_session, save_run
from stratum.provenance import build_drift_reference
from stratum.tools.tool_contamination import run_contamination
from stratum.tools.tool_drift_current import run_drift_current
from stratum.tools.tool_join_check import run_join_check
from stratum.tools.tool_label_quality import run_label_quality
from stratum.tools.tool_leakage_deep import run_leakage_deep
from stratum.tools.tool_schema_check import run_schema_check


class Result:
    model_name = "random_forest"
    task = "classification"
    best_params = {}


def test_profile_persists_contract_and_schema_check_uses_it(tmp_path):
    reference = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "plan": ["free", "pro", "free", "pro"],
        "target": [0, 1, 0, 1],
    })
    ref_path = tmp_path / "reference.csv"
    reference.to_csv(ref_path, index=False)
    profiled = run_profile(str(ref_path), target="target")

    current = pd.DataFrame({
        "age": [25, 90],
        "plan": ["enterprise", "free"],
        "target": [0, 1],
    })
    cur_path = tmp_path / "current.csv"
    current.to_csv(cur_path, index=False)

    result = run_schema_check(str(cur_path), sid=profiled["sid"])

    issue_types = {item["issue_type"] for item in result["issues"]}
    assert result["readiness"] == "risky"
    assert "unseen_categories" in issue_types
    assert "numeric_range_violation" in issue_types
    assert "target_present" in issue_types


def test_drift_current_uses_model_training_path(tmp_path):
    reference = pd.DataFrame({"age": range(50), "target": [0, 1] * 25})
    current = pd.DataFrame({"age": range(100, 150), "target": [0, 1] * 25})
    ref_path = tmp_path / "train.csv"
    cur_path = tmp_path / "current.csv"
    reference.to_csv(ref_path, index=False)
    current.to_csv(cur_path, index=False)
    sid = create_session(str(ref_path), "target")
    save_run(sid, "run_drift_ref", Result(), {"accuracy": .8}, metadata={
        "dataset_path": str(ref_path),
        "target_col": "target",
        "feature_names": ["age"],
        "feature_dtypes": {"age": "int64"},
    })

    result = run_drift_current("run_drift_ref", str(cur_path), sid=sid)

    assert result["model_id"] == "run_drift_ref"
    assert result["top_drifted"][0] == "age"


def test_drift_current_uses_immutable_training_profile_after_source_changes(tmp_path):
    original = pd.DataFrame({"age": range(60), "target": [0, 1] * 30})
    source = tmp_path / "training.csv"
    current = tmp_path / "current.csv"
    original.to_csv(source, index=False)
    original[["age"]].to_csv(current, index=False)
    sid = create_session(str(source), "target")
    save_run(sid, "run_immutable_drift", Result(), {"accuracy": .8}, metadata={
        "dataset_path": str(source),
        "target_col": "target",
        "feature_names": ["age"],
        "provenance": {
            "schema_version": 2,
            "dataset": {
                "drift_reference": {
                    "rows": len(original),
                    "features": build_drift_reference(original[["age"]], ["age"]),
                },
            },
        },
    })
    pd.DataFrame({"age": range(1000, 1060), "target": [0, 1] * 30}).to_csv(source, index=False)

    result = run_drift_current("run_immutable_drift", str(current), sid=sid)

    assert result["reference_kind"] == "training_time_feature_summary"
    assert result["overall_verdict"] == "no_drift"


def test_contamination_detects_overlap_id_and_time(tmp_path):
    train = pd.DataFrame({
        "id": [1, 2, 3],
        "x": [10, 20, 30],
        "when": ["2025-01-02", "2025-01-03", "2025-01-04"],
        "target": [0, 0, 1],
    })
    test = pd.DataFrame({
        "id": [3, 4],
        "x": [30, 40],
        "when": ["2025-01-03", "2025-01-05"],
        "target": [1, 1],
    })
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    result = run_contamination(str(train_path), str(test_path), target="target", id_col="id", time_col="when")

    kinds = {flag["flag_type"] for flag in result["flags"]}
    assert result["verdict"] == "contaminated"
    assert "id_overlap" in kinds
    assert "temporal_overlap" in kinds


def test_deep_leakage_detects_duplicate_target_and_name_signal(tmp_path):
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "paid_after_signup": [0, 1, 0, 1],
        "target_copy": [0, 1, 0, 1],
        "target": [0, 1, 0, 1],
    })
    path = tmp_path / "leak.csv"
    df.to_csv(path, index=False)

    result = run_leakage_deep(str(path), "target")

    kinds = {flag["flag_type"] for flag in result["flags"]}
    assert result["verdict"] == "critical"
    assert "duplicate_target" in kinds
    assert "post_outcome_name_signal" in kinds


def test_label_quality_detects_conflicts_and_rare_classes(tmp_path):
    df = pd.DataFrame({
        "x": [1, 1, 2, 3, 4, 5],
        "target": [0, 1, 1, 1, 1, None],
    })
    path = tmp_path / "labels.csv"
    df.to_csv(path, index=False)

    result = run_label_quality(str(path), "target", min_class_count=3)

    kinds = {issue["issue_type"] for issue in result["issues"]}
    assert "null_labels" in kinds
    assert "conflicting_labels_for_duplicate_features" in kinds
    assert "rare_classes" in kinds


def test_join_check_detects_row_explosion_and_target_risk(tmp_path):
    left = pd.DataFrame({"id": [1, 1, 2], "x": [10, 11, 20]})
    right = pd.DataFrame({"id": [1, 1, 3], "target_value": [0, 1, 1]})
    left_path = tmp_path / "left.csv"
    right_path = tmp_path / "right.csv"
    left.to_csv(left_path, index=False)
    right.to_csv(right_path, index=False)

    result = run_join_check(str(left_path), str(right_path), key="id", target="target")

    kinds = {issue["issue_type"] for issue in result["issues"]}
    assert result["verdict"] == "risky"
    assert "join_row_explosion" in kinds
    assert "right_table_target_leakage_risk" in kinds
