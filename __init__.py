__version__ = "0.1.0"

_EXPORTS = {
    "train": ".api",
    "PRESETS": ".api",
    "TrainingResult": ".result",
    "ModelSpec": ".registry",
    "available_models": ".registry",
    "create_model": ".registry",
    "get_model_spec": ".registry",
    "model_info": ".registry",
    "normalize_model_name": ".registry",
    "normalize_task": ".registry",
    "BaseModel": ".base_model",
    "DummyEncoder": ".base_model",
    "RandomForestModel": ".RandomForest",
    "MultiRegModel": ".MultiReg",
    "LightGBMModel": ".LightGBM",
    "CatBoostModel": ".CatBoost",
    "XGBoostModel": ".XGBoost",
    "SVMModel": ".SVM",
    "KNNModel": ".KNN",
    "LogRegModel": ".LogReg",
    "Ensemble": ".ensemble",
    "EnsembleCandidate": ".ensemble",
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
