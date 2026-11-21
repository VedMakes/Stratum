import asyncio

import joblib
import numpy as np
import pandas as pd

from stratum.mcp_server import dispatch_tool
from stratum.tools import tool_train


class WorkflowPredictor:
    def predict(self, X):
        return (X["signal"].to_numpy() >= 30).astype(int)

    def predict_proba(self, X):
        positive = self.predict(X) * 0.9 + 0.05
        return np.column_stack([1 - positive, positive])


class WorkflowResult:
    task = "classification"
    model_name = "logistic_regression"
    best_params = {"C": 1.0}
    study = None

    def predict(self, X):
        return WorkflowPredictor().predict(X)

    def evaluate(self, X, y):
        return {"accuracy": float(np.mean(self.predict(X) == np.asarray(y)))}

    def get_feature_importance(self, X, y):
        return [{"name": "signal", "importance": 1.0}]

    def save(self, path):
        joblib.dump(WorkflowPredictor(), path)
        return path

    def summarize_trials(self):
        return []

    def get_pipeline_spec(self):
        return {"steps": [{"name": "model", "class": "WorkflowPredictor"}]}


def test_agent_workflow_and_direct_guardrail_bypasses(tmp_path, monkeypatch):
    data_path = tmp_path / "workflow.csv"
    pd.DataFrame({
        "signal": range(60),
        "outcome": [0] * 30 + [1] * 30,
    }).to_csv(data_path, index=False)

    def call(name, arguments):
        return asyncio.run(dispatch_tool(name, arguments))

    profile = call("stratum_profile", {"file": str(data_path), "target": "outcome"})
    sid = profile["sid"]
    assert call("stratum_status", {"sid": sid})["runs"] == []

    split = call("stratum_split", {
        "file": str(data_path), "target": "outcome", "sid": sid,
        "mode": "stratified", "seed": 19,
    })
    assert split["split_mode"] == "stratified"

    first_decision = call("stratum_create_decision", {
        "sid": sid,
        "decision_type": "target_definition",
        "question": "Does outcome represent the event we want to predict?",
        "why_it_matters": "The target definition controls what the model learns.",
        "options": ["confirmed", "incorrect", "unknown"],
        "safe_answers": ["confirmed"],
        "blocking": True,
    })["decision_id"]
    train_calls = []
    monkeypatch.setattr(
        tool_train, "core_train",
        lambda: lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("guarded training must not reach MLProj")
        ),
    )

    blocked = call("stratum_train", {
        "file": str(data_path), "target": "outcome", "sid": sid,
        "split_id": split["split_id"],
    })
    assert blocked["recommended_action"] == "ask_user"
    assert not train_calls
    assert call("stratum_next_action", {"sid": sid})["recommended_action"] == "ask_user"
    call("stratum_resolve_decision", {"decision_id": first_decision, "answer": "unknown"})
    assert call("stratum_next_action", {"sid": sid})["recommended_action"] == "ask_user"
    assert "error" in call("stratum_train", {
        "file": str(data_path), "target": "outcome", "sid": sid,
        "split_id": split["split_id"],
    })

    call("stratum_resolve_decision", {
        "decision_id": first_decision, "answer": "confirmed",
        "comment": "Confirmed by the domain owner.",
    })

    class Policy:
        task = "classification"
        model = "logistic_regression"
        preset = "fast"
        n_trials = 1
        timeout = 10
        crossval = False

        def to_dict(self):
            return {"task": self.task, "model": self.model, "preset": self.preset}

    def fake_train(task, model, X, y, **kwargs):
        train_calls.append({"rows": len(X), "kwargs": kwargs})
        return WorkflowResult()

    monkeypatch.setattr(tool_train, "core_policy", lambda: lambda *a, **k: Policy())
    monkeypatch.setattr(tool_train, "core_train", lambda: fake_train)
    trained = call("stratum_train", {
        "file": str(data_path), "target": "outcome", "sid": sid,
        "split_id": split["split_id"],
    })
    assert "error" not in trained
    assert train_calls[0]["rows"] == split["train_rows"]
    model_id = trained["model_id"]
    validation = call("stratum_validate", {
        "model_id": model_id, "file": str(data_path), "sid": sid,
    })
    segments = call("stratum_segments", {
        "model_id": model_id, "file": str(data_path), "sid": sid,
        "min_support": 5,
    })
    assert validation["rows_evaluated"] == 60
    assert validation["metrics"]["accuracy"] == 1.0
    assert segments["rows_evaluated"] == 60

    deployment_decision = call("stratum_create_decision", {
        "sid": sid,
        "decision_type": "deployment_authorization",
        "question": "Has this model been approved for use?",
        "why_it_matters": "Prediction and export require authorization.",
        "options": ["approved", "not_approved"],
        "safe_answers": ["approved"],
        "blocking": True,
    })["decision_id"]
    scoring_path = tmp_path / "scoring.csv"
    pd.DataFrame({"signal": [1, 29, 30, 59]}).to_csv(scoring_path, index=False)
    prediction_path = tmp_path / "predictions.csv"
    export_path = tmp_path / "model_export.py"

    for tool_name, arguments in (
        ("stratum_train", {
            "file": str(data_path), "target": "outcome", "sid": sid,
            "split_id": split["split_id"],
        }),
        ("stratum_predict", {
            "model_id": model_id, "file": str(scoring_path),
            "output_path": str(prediction_path),
        }),
        ("stratum_export", {
            "model_id": model_id, "output_path": str(export_path),
        }),
    ):
        result = call(tool_name, arguments)
        assert result["recommended_action"] == "ask_user"
        assert result["blocked_decisions"][0]["decision_type"] == "deployment_authorization"
    assert len(train_calls) == 1
    assert not prediction_path.exists()
    assert not export_path.exists()

    call("stratum_resolve_decision", {
        "decision_id": deployment_decision, "answer": "approved",
    })
    prediction = call("stratum_predict", {
        "model_id": model_id, "file": str(scoring_path),
        "output_path": str(prediction_path),
    })
    exported = call("stratum_export", {
        "model_id": model_id, "output_path": str(export_path),
    })
    status = call("stratum_status", {"sid": sid})
    report = call("stratum_report", {"sid": sid})

    assert prediction["rows_scored"] == 4
    assert prediction["class_distribution"] == {"0": 2, "1": 2}
    assert prediction_path.exists()
    assert exported["script_path"] == str(export_path)
    assert report["best_model"] == model_id
    assert {"validation", "segment_analysis", "export_script"} <= set(report["diagnostics"])
    assert status["runs"][0]["split_id"] == split["split_id"]
    assert status["runs"][0]["baseline_comparison"]["metric"] == "accuracy"
    assert status["blocking_decisions"] == []
    assert status["recent_journal"]


def test_association_is_available_through_mcp_dispatch(tmp_path):
    data_path = tmp_path / "association.csv"
    pd.DataFrame({
        "signal": range(20),
        "outcome": [0] * 10 + [1] * 10,
    }).to_csv(data_path, index=False)

    result = asyncio.run(dispatch_tool("stratum_association", {
        "file": str(data_path),
        "outcome": "outcome",
        "positive_label": "1",
        "features": ["signal"],
        "bootstrap_samples": 100,
    }))

    assert "error" not in result
    assert result["results"][0]["effect_name"] == "cliffs_delta"
