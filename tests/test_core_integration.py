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
