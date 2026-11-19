import numpy as np
import pandas as pd

from MLProj import train


def test_core_pipeline_owns_imputation_encoding_and_scaling():
    X = pd.DataFrame({
        "amount": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "segment": ["a", "b", None, "a", "b", "a", "b", "a"],
    })
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    result = train(
        "classification",
        "logistic_regression",
        X,
        y,
        tune=False,
        params={"max_iter": 200},
    )
    assert [name for name, _ in result.estimator.steps] == [
        "imputer", "encoder", "scaler", "model"
    ]
    assert len(result.predict(X)) == len(X)
    assert result.get_pipeline_spec()["steps"][0]["class"] == "DataFrameImputer"
    assert result.get_feature_importance(X, y)
    metrics = result.evaluate(X, y)
    assert set(metrics["per_class"]) == {"0", "1"}
    assert "pr_auc" in metrics
    assert "brier_score" in metrics


def test_core_policy_rejects_unsupported_model_task():
    from MLProj.policy import resolve_training_policy

    try:
        resolve_training_policy(pd.Series([1.0, 2.1, 3.4]), 3, task="regression", model="logreg")
    except ValueError as exc:
        assert "does not support regression" in str(exc)
    else:
        raise AssertionError("unsupported model/task pair was accepted")


def test_catboost_uses_core_encoder_for_categorical_data():
    X = pd.DataFrame({
        "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "segment": ["a", "b", None, "a", "b", "a"],
    })
    y = pd.Series([0, 0, 0, 1, 1, 1])
    result = train(
        "classification",
        "catboost",
        X,
        y,
        tune=False,
        params={"iterations": 3, "depth": 2},
    )
    assert [name for name, _ in result.estimator.steps][:2] == ["imputer", "encoder"]
    assert len(result.predict(X)) == len(X)


def test_train_routes_custom_cv_splitter_and_groups(monkeypatch):
    from sklearn.model_selection import GroupKFold
    import MLProj.api as api

    class Wrapper:
        def __init__(self):
            self.opt_crossval = False

        def optimize(self, X, y, **kwargs):
            self.optimized = True

        def get_best_model(self):
            return object()

        def get_best_params(self):
            return {}

        def get_study(self):
            return None

        def feature_names(self):
            return []

    wrapper = Wrapper()
    def build_model(*args, **kwargs):
        wrapper.opt_crossval = kwargs["crossval"]
        return wrapper

    monkeypatch.setattr(api, "create_model", build_model)
    groups = np.repeat(np.arange(4), 3)
    splitter = GroupKFold(n_splits=4)

    result = api.train(
        "classification", "logistic_regression", pd.DataFrame({"x": range(12)}),
        pd.Series([0, 1] * 6), tune=True, n_trials=1, cv_splitter=splitter,
        cv_groups=groups, crossval=False,
    )

    assert wrapper.opt_crossval is True
    assert wrapper.cv is splitter
    assert np.array_equal(wrapper.cv_groups, groups)
    assert result.metadata["crossval"] is True


def test_model_cv_passes_group_labels_to_sklearn(monkeypatch):
    from sklearn.model_selection import GroupKFold
    import MLProj.base_model as base_model
    from MLProj.statistical.logreg import LogRegModel

    groups = np.repeat(np.arange(4), 3)
    splitter = GroupKFold(n_splits=4)
    wrapper = LogRegModel(mode="Classifier", crossval=True)
    wrapper.cv = splitter
    wrapper.cv_groups = groups
    wrapper.set_loss_function("accuracy")
    observed = {}

    def score(model, X, y, **kwargs):
        observed.update(kwargs)
        return np.array([0.5])

    monkeypatch.setattr(base_model, "cross_val_score", score)
    X = pd.DataFrame({"x": range(12)})
    y = pd.Series([0, 1] * 6)

    assert wrapper._cpu_cross_val_score(X, y, {}, splitter) == 0.5
    assert observed["cv"] is splitter
    assert np.array_equal(observed["groups"], groups)


def test_smote_resamples_only_each_cv_training_fold(monkeypatch):
    from sklearn.base import BaseEstimator
    import imblearn.over_sampling
    from MLProj.statistical.logreg import LogRegModel

    observed_rows = []

    class RecordingSampler(BaseEstimator):
        def __init__(self, random_state=None):
            self.random_state = random_state

        def fit_resample(self, X, y):
            observed_rows.append(X.iloc[:, 0].astype(int).tolist())
            return X, y

    monkeypatch.setattr(imblearn.over_sampling, "SMOTE", RecordingSampler)
    X = pd.DataFrame({"row_id": range(30), "signal": np.arange(30) % 3})
    y = pd.Series([0] * 24 + [1] * 6)
    wrapper = LogRegModel(mode="Classifier", crossval=True, use_smote=True)
    X_prepared, y_prepared = wrapper._prepare_training_data(X, y)
    wrapper.set_loss_function("accuracy")
    wrapper.n_jobs = 1

    assert len(X_prepared) == len(X)
    assert len(y_prepared) == len(y)
    wrapper._cpu_cross_val_score(X_prepared, y_prepared, {}, 3)

    assert len(observed_rows) == 3
    assert all(len(rows) == 20 for rows in observed_rows)
    assert all(set(rows) < set(range(30)) for rows in observed_rows)


def test_smote_pipeline_fits_and_predicts_with_categorical_features():
    X = pd.DataFrame({
        "amount": np.arange(40, dtype=float),
        "segment": ["a", "b"] * 20,
    })
    y = pd.Series([0] * 32 + [1] * 8)

    result = train(
        "classification", "logistic_regression", X, y, tune=False,
        use_smote=True, params={"max_iter": 200},
    )

    assert [name for name, _ in result.estimator.steps] == [
        "imputer", "encoder", "smote", "scaler", "model",
    ]
    assert len(result.predict(X)) == len(X)
    importance = result.get_feature_importance(X, y)
    assert {item["name"] for item in importance} == {"amount", "segment_b"}
