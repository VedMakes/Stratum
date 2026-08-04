import numpy as np

from stratum.session import get_run
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
    result = tool_train.run_train(str(binary_csv), "target", model="rf", task="auto")
    assert "error" not in result
    assert seen["rows"] == 48
    assert seen["kwargs"]["n_trials"] == 10
    assert result["decision_tree_summary"]
    assert get_run(result["model_id"])["metrics"] == result["metrics"]


def test_core_policy_defaults():
    import pandas as pd
    from MLProj.policy import resolve_training_policy

    classification = resolve_training_policy(pd.Series([0, 1, 0]), 100)
    regression = resolve_training_policy(pd.Series(np.linspace(0, 1, 30)), 100)
    assert classification.model == "logistic_regression"
    assert regression.model == "random_forest"
