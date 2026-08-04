import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from stratum.session import create_session, save_run
from stratum.tools.tool_report import run_report
from stratum.tools.tool_segments import run_segments
from stratum.tools.tool_stability import run_stability
from stratum.tools.tool_validate import run_validate


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
    assert result["confusion_matrix"]["labels"] == ["0", "1"]
    assert result["calibration_bins"]


def test_segments_finds_bad_group(tmp_path):
    sid = create_session(str(tmp_path / "train.csv"), "target")
    model_id = _save_classifier(tmp_path, sid)
    data = _diagnostic_csv(tmp_path)

    result = run_segments(model_id, str(data), sid=sid, min_support=10)

    assert result["segments"]
    assert result["segments"][0]["column"] == "group"
    assert result["segments"][0]["value"] == "bad"
    assert result["segments"][0]["delta"] > 0


def test_stability_reports_fold_variance(tmp_path):
    sid = create_session(str(tmp_path / "train.csv"), "target")
    model_id = _save_classifier(tmp_path, sid)
    data = tmp_path / "stable.csv"
    _training_frame(60).to_csv(data, index=False)

    result = run_stability(model_id, str(data), sid=sid, folds=4)

    assert result["folds"] == 4
    assert result["metric"] == "f1_weighted"
    assert len(result["fold_scores"]) == 4
    assert result["stability"] in {"stable", "moderate", "unstable"}


def test_report_includes_latest_diagnostics(tmp_path):
    sid = create_session(str(tmp_path / "train.csv"), "target")
    model_id = _save_classifier(tmp_path, sid)
    data = _diagnostic_csv(tmp_path)
    run_validate(model_id, str(data), sid=sid)
    run_segments(model_id, str(data), sid=sid, min_support=10)
    run_stability(model_id, str(data), sid=sid, folds=3)

    report = run_report(sid=sid)

    assert report["best_model"] == model_id
    assert set(report["diagnostics"]) >= {"validation", "segment_analysis", "stability"}
    assert "Run stratum_validate" not in " ".join(report["recommended_next_actions"])
