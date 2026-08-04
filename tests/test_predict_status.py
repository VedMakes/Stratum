import joblib
import numpy as np
import pandas as pd

from stratum.session import (
    create_session,
    resolve_flag,
    save_artifact,
    save_flag,
    save_run,
)
from stratum.tools.tool_predict import run_predict
from stratum.tools.tool_status import run_status


class TinyClassifier:
    classes_ = np.array([0, 1])

    def predict(self, X):
        return (np.asarray(X["score"]) > .5).astype(int)

    def predict_proba(self, X):
        positive = np.asarray(X["score"], dtype=float)
        return np.column_stack([1 - positive, positive])


class Result:
    model_name = "tiny"
    task = "classification"
    best_params = {}


def _save_model_run(tmp_path, sid, model_id, accuracy):
    artifact_path = tmp_path / f"{model_id}.joblib"
    joblib.dump(TinyClassifier(), artifact_path)
    save_run(sid, model_id, Result(), {"accuracy": accuracy}, metadata={
        "model_name": "tiny",
        "task": "classification",
        "feature_names": ["score"],
        "feature_dtypes": {"score": "float64"},
        "artifact_path": str(artifact_path),
    })
    return artifact_path


def test_predict_scores_rows_and_writes_output(tmp_path):
    sid = create_session(str(tmp_path / "train.csv"), "target")
    _save_model_run(tmp_path, sid, "run_predict", .8)
    data = tmp_path / "new.csv"
    pd.DataFrame({"score": [.2, .91, .73]}).to_csv(data, index=False)
    output = tmp_path / "predictions.csv"

    result = run_predict("run_predict", str(data), output_path=str(output), sid=sid)

    assert result["rows_scored"] == 3
    assert result["class_distribution"] == {"1": 2, "0": 1}
    assert result["high_confidence_positives"] == 1
    assert result["predictions"][1] == {
        "row_index": 1, "prediction": 1, "confidence": .91
    }
    assert output.exists()
    written = pd.read_csv(output)
    assert list(written["prediction"]) == [0, 1, 1]


def test_predict_requires_trained_features(tmp_path):
    sid = create_session(str(tmp_path / "train.csv"), "target")
    _save_model_run(tmp_path, sid, "run_missing", .8)
    data = tmp_path / "bad.csv"
    pd.DataFrame({"other": [1]}).to_csv(data, index=False)

    result = run_predict("run_missing", str(data))

    assert "missing required feature columns" in result["error"]


def test_status_summarises_session_by_sid_or_file(tmp_path):
    dataset = tmp_path / "train.csv"
    dataset.write_text("score,target\n.1,0\n.9,1\n", encoding="utf-8")
    sid = create_session(str(dataset.resolve()), "target")
    flag_id = save_flag(sid, {
        "flag_type": "leakage", "column": "paid_date",
        "confidence": .9, "reasoning": "Looks post-outcome.",
    })
    assert resolve_flag(flag_id, "rejected", "recorded at signup")
    _save_model_run(tmp_path, sid, "baseline", .71)
    _save_model_run(tmp_path, sid, "best", .84)
    save_artifact(
        sid,
        "engineered_csv",
        str(tmp_path / "engineered.csv"),
        metadata={"features": ["support_tickets_per_month", "days_since_last_login"]},
    )

    by_sid = run_status(sid=sid)
    by_file = run_status(file=str(dataset))

    assert by_file["sid"] == sid
    assert by_sid["best_model"] == "best"
    assert by_sid["current_metric"] == "accuracy"
    assert by_sid["current_score"] == .84
    assert round(by_sid["improvement"], 2) == .13
    assert by_sid["engineered_features"] == [
        "support_tickets_per_month", "days_since_last_login"
    ]
    assert by_sid["resolved_flags"][0]["user_comment"] == "recorded at signup"
