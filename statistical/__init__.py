from importlib import import_module


_EXPORTS = {
    "CatBoostModel": ".catboost_model",
    "KNNModel": ".knn",
    "LightGBMModel": ".lightgbm_model",
    "LogRegModel": ".logreg",
    "MultiRegModel": ".multireg",
    "RandomForestModel": ".random_forest",
    "SVMModel": ".svm",
    "XGBoostModel": ".xgboost_model",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
