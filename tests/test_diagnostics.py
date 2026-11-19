import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from stratum.session import create_session, get_artifacts, save_run
from stratum.tools.tool_report import run_report
from stratum.tools.tool_segments import run_segments
from stratum.tools.tool_group_evaluation import run_group_evaluation
from stratum.tools.tool_stability import run_stability
from stratum.tools.tool_validate import run_validate
from stratum.tools.eval_common import classification_metrics
from stratum.tools.eval_common import evaluation_warnings, residual_summary


class Result:
    model_name = "logistic_regression"
    task = "classification"
    best_params = {}


def _training_frame(rows=80):
    score = np.linspace(0, 1, rows)
    return pd.DataFrame({
        "score": score,
        "target": (score > .5).astype(int),
    })


def _save_classifier(tmp_path, sid, model_id="diagnostic_model"):
    train = _training_frame()
    model = LogisticRegression().fit(train[["score"]], train["target"])
    artifact_path = tmp_path / f"{model_id}.joblib"
    joblib.dump(model, artifact_path)
    save_run(sid, model_id, Result(), {"accuracy": .9}, metadata={
        "model_name": "logistic_regression",
        "task": "classification",
        "feature_names": ["score"],
        "feature_dtypes": {"score": "float64"},
        "target_col": "target",
        "artifact_path": str(artifact_path),
    })
    return model_id


def _diagnostic_csv(tmp_path):
    df = _training_frame()
    df["group"] = ["good", "bad"] * 40
    df.loc[df["group"] == "bad", "target"] = 1 - df.loc[df["group"] == "bad", "target"]
    path = tmp_path / "diagnostic.csv"
    df.to_csv(path, index=False)
    return path


def test_validate_returns_metrics_and_confusion_matrix(tmp_path):
    sid = create_session(str(tmp_path / "train.csv"), "target")
    model_id = _save_classifier(tmp_path, sid)
    data = _diagnostic_csv(tmp_path)

    result = run_validate(model_id, str(data), sid=sid)

    assert result["rows_evaluated"] == 80
    assert "accuracy" in result["metrics"]
    assert set(result["metrics"]["per_class"]) == {"0", "1"}
    assert "pr_auc" in result["metrics"]
    assert "brier_score" in result["metrics"]
    assert result["confusion_matrix"]["labels"] == ["0", "1"]
    assert result["calibration_bins"]


def test_probability_metrics_support_string_classes():
    metrics = classification_metrics(
        ["no", "yes", "no", "yes"],
        ["no", "yes", "no", "no"],
        np.array([[.9, .1], [.1, .9], [.8, .2], [.4, .6]]),
        class_labels=["no", "yes"],
    )

    assert metrics["per_class"]["yes"]["support"] == 2
    assert metrics["positive_class"] == "yes"
    assert metrics["pr_auc"] > .9
    assert metrics["brier_score"] < .1


def test_evaluation_warnings_flag_imbalance_and_missing_probabilities():
    warnings = evaluation_warnings(
        "classification", [0] * 40 + [1] * 3,
        {"accuracy": .95}, y_score=None,
    )

    assert any("highly imbalanced" in warning for warning in warnings)
    assert any("Probability-based" in warning for warning in warnings)


def test_residual_summary_includes_robust_and_tail_diagnostics():
    summary = residual_summary([1, 2, 3, 4, 100], [1, 2, 3, 4, 5])

    assert summary["median_residual"] == 0
    assert summary["median_abs_residual"] == 0
    assert summary["p95_abs_residual"] > summary["p90_abs_residual"]
    assert summary["robust_outlier_fraction"] == .2


def test_segments_finds_bad_group(tmp_path):
    sid = create_session(str(tmp_path / "train.csv"), "target")
    model_id = _save_classifier(tmp_path, sid)
    data = _diagnostic_csv(tmp_path)

    result = run_segments(model_id, str(data), sid=sid, min_support=10)

    assert result["segments"]
    assert result["segments"][0]["column"] == "group"
    assert result["segments"][0]["value"] == "bad"
    assert result["segments"][0]["delta"] > 0


def test_group_evaluation_reports_metrics_and_excludes_low_support(tmp_path):
    sid = create_session(str(tmp_path / "train.csv"), "target")
    model_id = _save_classifier(tmp_path, sid)
    data = _diagnostic_csv(tmp_path)
    frame = pd.read_csv(data)
    frame.loc[:2, "group"] = "rare"
    frame.to_csv(data, index=False)

    result = run_group_evaluation(model_id, str(data), "group", sid=sid, min_support=10)

    assert result["rows_evaluated"] == 80
    assert result["overall_metrics"]["accuracy"] < 1
    assert [group["value"] for group in result["groups"]] == ["bad", "good"]
    assert result["excluded_groups"] == 1
    assert result["groups"][0]["metrics"]["accuracy"] < result["groups"][1]["metrics"]["accuracy"]
    assert "not a fairness" in result["note"]
    artifact = get_artifacts(sid, "group_evaluation")[-1]
    assert artifact["metadata"]["min_support"] == 10


def test_group_evaluation_counts_missing_values_as_a_group(tmp_path):
    sid = create_session(str(tmp_path / "train.csv"), "target")
    model_id = _save_classifier(tmp_path, sid)
    frame = _diagnostic_csv(tmp_path)
    data = pd.read_csv(frame)
    data.loc[:19, "group"] = None
    data.to_csv(frame, index=False)

    result = run_group_evaluation(model_id, str(frame), "group", sid=sid, min_support=10)

    missing_group = next(group for group in result["groups"] if group["value"] == "<missing>")
    assert missing_group["support"] == 20


def test_group_evaluation_validates_group_column_and_support(tmp_path):
    sid = create_session(str(tmp_path / "train.csv"), "target")
    model_id = _save_classifier(tmp_path, sid)
    data = _diagnostic_csv(tmp_path)

    missing = run_group_evaluation(model_id, str(data), "unknown", sid=sid)
    support = run_group_evaluation(model_id, str(data), "group", sid=sid, min_support=1)

    assert "not found" in missing["error"]
    assert "at least 2" in support["error"]


def test_stability_reports_fold_variance(tmp_path):
    sid = create_session(str(tmp_path / "train.csv"), "target")
    model_id = _save_classifier(tmp_path, sid)
    data = tmp_path / "stable.csv"
    _training_frame(60).to_csv(data, index=False)

    result = run_stability(model_id, str(data), sid=sid, folds=4, repeats=1)

    assert result["folds"] == 4
    assert result["metric"] == "f1_weighted"
    assert len(result["fold_scores"]) == 4
    assert result["stability"] in {"stable", "moderate", "unstable"}
    assert result["protocol"]["method"] == "repeated_stratified_kfold"
    assert result["protocol"]["source_sha256"]
    artifact = [item for item in get_artifacts(sid, "stability")][-1]
    assert artifact["metadata"]["protocol"] == result["protocol"]


def test_stability_repeats_iid_cv_and_reports_descriptive_spread(tmp_path):
    sid = create_session(str(tmp_path / "train.csv"), "target")
    model_id = _save_classifier(tmp_path, sid)
    data = tmp_path / "repeated.csv"
    _training_frame(60).to_csv(data, index=False)

    result = run_stability(model_id, str(data), sid=sid, folds=3, repeats=4, seed=9)

    assert result["repeats"] == 4
    assert len(result["fold_scores"]) == 12
    assert len(result["repeat_scores"]) == 4
    assert len(result["score_range_90pct"]) == 2
    assert result["protocol"]["seed"] == 9
    assert any("not independent confidence intervals" in warning for warning in result["warnings"])


def test_group_stability_uses_disjoint_groups_in_each_repeat(tmp_path, monkeypatch):
    from stratum.tools import tool_stability

    sid = create_session(str(tmp_path / "train.csv"), "target")
    model_id = _save_classifier(tmp_path, sid)
    save_run(sid, model_id, Result(), {"accuracy": .9}, metadata={
        "model_name": "logistic_regression", "task": "classification",
        "feature_names": ["score"], "target_col": "target",
        "artifact_path": str(tmp_path / f"{model_id}.joblib"),
        "split_metadata": {"strategy": "group", "parameters": {"group_col": "entity"}},
    })
    data = tmp_path / "group_eval.csv"
    frame = pd.DataFrame({
        "entity": [f"g{i}" for i in range(12) for _ in range(5)],
        "score": np.tile(np.linspace(0, 1, 5), 12),
        "target": [0, 1, 0, 1, 0] * 12,
    })
    frame.to_csv(data, index=False)
    real_score = tool_stability.cross_val_score
    observed = []

    def checked_score(estimator, X, y, *, cv, groups, **kwargs):
        fold_rows = list(cv.split(X, y, groups))
        for train_indices, test_indices in fold_rows:
            assert set(groups[train_indices]).isdisjoint(groups[test_indices])
        observed.append(len(fold_rows))
        return real_score(estimator, X, y, cv=cv, groups=groups, **kwargs)

    monkeypatch.setattr(tool_stability, "cross_val_score", checked_score)
    result = tool_stability.run_stability(
        model_id, str(data), sid=sid, folds=4, repeats=3
    )

    assert "error" not in result
    assert observed == [4, 4, 4]
    assert result["protocol"]["method"] == "repeated_group_kfold"
    assert result["protocol"]["group_col"] == "entity"


def test_temporal_stability_preserves_order_and_gap(tmp_path, monkeypatch):
    from stratum.tools import tool_stability

    sid = create_session(str(tmp_path / "train.csv"), "target")
    model_id = _save_classifier(tmp_path, sid)
    save_run(sid, model_id, Result(), {"accuracy": .9}, metadata={
        "model_name": "logistic_regression", "task": "classification",
        "feature_names": ["score"], "target_col": "target",
        "artifact_path": str(tmp_path / f"{model_id}.joblib"),
        "split_metadata": {
            "strategy": "temporal",
            "parameters": {"time_col": "when", "gap_rows": 2},
        },
    })
    data = tmp_path / "temporal_eval.csv"
    frame = _training_frame(60)
    frame["target"] = [0, 1] * 30
    frame["when"] = pd.date_range("2025-01-01", periods=60).astype(str)
    frame = frame.sample(frac=1, random_state=21)
    frame.to_csv(data, index=False)
    real_score = tool_stability.cross_val_score
    observed = []

    def checked_score(estimator, X, y, *, cv, groups, **kwargs):
        folds = list(cv.split(X, y))
        for train_indices, test_indices in folds:
            assert train_indices.max() + 2 < test_indices.min()
        observed.append(len(folds))
        return real_score(estimator, X, y, cv=cv, groups=groups, **kwargs)

    monkeypatch.setattr(tool_stability, "cross_val_score", checked_score)
    result = tool_stability.run_stability(
        model_id, str(data), sid=sid, folds=4, repeats=3
    )

    assert "error" not in result
    assert observed == [4]
    assert result["repeats"] == 1
    assert result["protocol"]["method"] == "time_series"
    assert result["protocol"]["gap_rows"] == 2
    assert any("walk-forward schedule" in warning for warning in result["warnings"])


def test_predefined_stability_requires_validation_not_resplitting(tmp_path):
    sid = create_session(str(tmp_path / "train.csv"), "target")
    model_id = _save_classifier(tmp_path, sid)
    save_run(sid, model_id, Result(), {"accuracy": .9}, metadata={
        "model_name": "logistic_regression", "task": "classification",
        "feature_names": ["score"], "target_col": "target",
        "artifact_path": str(tmp_path / f"{model_id}.joblib"),
        "split_metadata": {"strategy": "predefined", "parameters": {}},
    })
    data = tmp_path / "predefined.csv"
    _training_frame(60).to_csv(data, index=False)

    result = run_stability(model_id, str(data), sid=sid)

    assert "predefined partition" in result["error"]


def test_report_includes_latest_diagnostics(tmp_path):
    sid = create_session(str(tmp_path / "train.csv"), "target")
    model_id = _save_classifier(tmp_path, sid)
    data = _diagnostic_csv(tmp_path)
    run_validate(model_id, str(data), sid=sid)
    run_segments(model_id, str(data), sid=sid, min_support=10)
    run_group_evaluation(model_id, str(data), "group", sid=sid, min_support=10)
    run_stability(model_id, str(data), sid=sid, folds=3)

    report = run_report(sid=sid)

    assert report["best_model"] == model_id
    assert set(report["diagnostics"]) >= {
        "validation", "segment_analysis", "group_evaluation", "stability"
    }
    assert "Run stratum_validate" not in " ".join(report["recommended_next_actions"])
