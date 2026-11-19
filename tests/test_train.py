import numpy as np
import pandas as pd

from stratum.session import create_session, get_run
from stratum.data.splitter import run_split
from stratum.tools import tool_train


class FakeModel:
    def final_estimator(self, estimator):
        return estimator


class FakeEstimator:
    feature_importances_ = np.array([.6, .2, .2, 0, 0])

    def predict(self, X):
        return (np.asarray(X.iloc[:, 0]) > np.median(X.iloc[:, 0])).astype(int)


class FakeResult:
    task = "classification"
    model_name = "random_forest"
    best_params = {"n_estimators": 10}
    study = None
    model = FakeModel()
    estimator = FakeEstimator()
    feature_names = ["age", "spend", "segment", "customer_id", "leak"]

    def predict(self, X):
        return self.estimator.predict(X)

    def evaluate(self, X, y):
        return {"accuracy": float(np.mean(self.predict(X) == np.asarray(y)))}

    def get_feature_importance(self, X, y):
        return [
            {"name": name, "importance": float(value)}
            for name, value in zip(self.feature_names, self.estimator.feature_importances_)
        ]

    def save(self, path):
        path.write_bytes(b"model")
        return path

    def summarize_trials(self):
        return []

    def get_pipeline_spec(self):
        return {"steps": [{"name": "model", "class": "FakeEstimator"}]}


def test_train_uses_holdout_and_persists(binary_csv, monkeypatch):
    seen = {}

    def fake_train(task, model, X, y, **kwargs):
        seen.update(task=task, model=model, rows=len(X), kwargs=kwargs)
        return FakeResult()

    policy = type("Policy", (), {
        "task": "classification",
        "model": "random_forest",
        "preset": "fast",
        "n_trials": 10,
        "timeout": 180,
        "crossval": False,
        "to_dict": lambda self: {
            "task": self.task, "model": self.model, "preset": self.preset,
            "n_trials": self.n_trials, "timeout": self.timeout, "crossval": self.crossval,
        },
    })()
    monkeypatch.setattr(tool_train, "core_train", lambda: fake_train)
    monkeypatch.setattr(tool_train, "core_policy", lambda: lambda *args, **kwargs: policy)
    split = run_split(str(binary_csv), "target", mode="stratified")
    result = tool_train.run_train(
        str(binary_csv), "target", model="rf", task="auto",
        sid=split["sid"], split_id=split["split_id"],
    )
    assert "error" not in result
    assert seen["rows"] == split["train_rows"] == 48
    assert seen["kwargs"]["n_trials"] == 10
    assert result["decision_tree_summary"]
    saved_run = get_run(result["model_id"])
    assert saved_run["metrics"] == result["metrics"]
    assert saved_run["split_metadata"]["split_id"] == split["split_id"]
    assert saved_run["split_metadata"]["strategy"] == "stratified"

    internal = tool_train.run_train(
        str(binary_csv), "target", model="rf", task="auto", sid=split["sid"]
    )
    assert "error" not in internal
    assert internal["split_metadata"]["split_id"] is None
    assert internal["split_metadata"]["strategy"] == "stratified"


def test_group_split_training_excludes_group_feature(tmp_path, monkeypatch):
    df = pd.DataFrame({
        "entity": [f"g{i}" for i in range(20) for _ in range(5)],
        "signal": [0, 1, 0, 1, 0] * 20,
        "target": [0, 1, 0, 1, 0] * 20,
    })
    path = tmp_path / "group_training.csv"
    df.to_csv(path, index=False)
    split = run_split(str(path), "target", mode="group", group_col="entity")
    observed = {}

    def fake_train(task, model, X, y, **kwargs):
        observed["columns"] = list(X.columns)
        observed["kwargs"] = kwargs
        return FakeResult()

    policy = type("Policy", (), {
        "task": "classification", "model": "random_forest", "preset": "fast",
        "n_trials": 10, "timeout": 180, "crossval": False,
        "to_dict": lambda self: {"task": self.task, "model": self.model},
    })()
    monkeypatch.setattr(tool_train, "core_train", lambda: fake_train)
    monkeypatch.setattr(tool_train, "core_policy", lambda: lambda *args, **kwargs: policy)

    result = tool_train.run_train(
        str(path), "target", sid=split["sid"], split_id=split["split_id"]
    )

    assert "error" not in result
    assert observed["columns"] == ["signal"]
    assert observed["kwargs"]["crossval"] is False
    assert observed["kwargs"]["cv_splitter"].n_splits == 5
    assert len(observed["kwargs"]["cv_groups"]) == split["train_rows"]
    assert result["split_metadata"]["cv_strategy"] == "group_kfold"
    assert result["split_metadata"]["strategy"] == "group"


def test_temporal_training_uses_time_series_cv(tmp_path, monkeypatch):
    df = pd.DataFrame({
        "signal": range(60),
        "when": pd.date_range("2025-01-01", periods=60).astype(str),
        "target": [0, 1] * 30,
    })
    path = tmp_path / "temporal_training.csv"
    df.to_csv(path, index=False)
    split = run_split(str(path), "target", mode="temporal", time_col="when")
    observed = {}

    def fake_train(task, model, X, y, **kwargs):
        observed.update(kwargs)
        return FakeResult()

    policy = type("Policy", (), {
        "task": "classification", "model": "random_forest", "preset": "fast",
        "n_trials": 10, "timeout": 180, "crossval": False,
        "to_dict": lambda self: {"task": self.task, "model": self.model},
    })()
    monkeypatch.setattr(tool_train, "core_train", lambda: fake_train)
    monkeypatch.setattr(tool_train, "core_policy", lambda: lambda *args, **kwargs: policy)

    result = tool_train.run_train(
        str(path), "target", sid=split["sid"], split_id=split["split_id"]
    )

    assert "error" not in result
    assert observed["cv_splitter"].n_splits == 5
    assert observed["cv_groups"] is None
    assert result["split_metadata"]["cv_strategy"] == "time_series"


def test_core_policy_defaults():
    import pandas as pd
    from MLProj.policy import resolve_training_policy

    classification = resolve_training_policy(pd.Series([0, 1, 0]), 100)
    regression = resolve_training_policy(pd.Series(np.linspace(0, 1, 30)), 100)
    assert classification.model == "logistic_regression"
    assert regression.model == "random_forest"


def test_direct_training_call_is_blocked_by_pending_domain_decision(binary_csv, monkeypatch):
    from stratum.session import create_decision, resolve_decision

    sid = create_session(str(binary_csv), "target")
    decision_id = create_decision(
        sid,
        "temporal_availability",
        "Was the date available before prediction?",
        "Post-outcome values can leak the target.",
        options=["available_before_prediction", "post_outcome", "unknown"],
        safe_answers=["available_before_prediction"],
        blocking=True,
    )
    monkeypatch.setattr(
        tool_train,
        "core_policy",
        lambda: (_ for _ in ()).throw(AssertionError("training policy must not run")),
    )

    result = tool_train.run_train(str(binary_csv), "target", sid=sid)

    assert result["recommended_action"] == "ask_user"
    assert result["blocked_decisions"][0]["status"] == "pending"

    resolve_decision(decision_id, "unknown")
    result = tool_train.run_train(str(binary_csv), "target", sid=sid)

    assert result["recommended_action"] == "ask_user"
    assert result["blocked_decisions"][0]["answer"] == "unknown"


def test_training_rejects_modified_saved_split(binary_csv):
    split = run_split(str(binary_csv), "target", mode="stratified")
    with open(split["train_path"], "a", encoding="utf-8") as output:
        output.write("0,0,changed,changed,0,0\n")
    result = tool_train.run_train(
        str(binary_csv), "target", sid=split["sid"], split_id=split["split_id"]
    )

    assert "Saved training split changed" in result["error"]
