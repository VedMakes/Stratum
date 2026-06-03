# MLProj

MLProj is a small tabular machine-learning wrapper library. It gives the common model families a consistent interface for fitting, tuning, scoring, explaining, CUDA opt-in, and simple ensemble search.

The project is intentionally lightweight: use the high-level `train()` facade for quick workflows, or instantiate wrappers directly when you need model-specific control.

## Supported Models

Classification and regression:

- `random_forest` aliases: `rf`, `forest`
- `xgboost` aliases: `xgb`
- `lightgbm` aliases: `lgbm`, `lgb`
- `catboost` aliases: `cat`
- `svm` aliases: `svc`, `svr`
- `knn` aliases: `nearest_neighbors`, `k_neighbors`

Classification only:

- `logistic_regression` aliases: `logreg`, `logistic`, `lr`

Regression only:

- `polynomial_regression` aliases: `multireg`, `multi_reg`, `elastic_poly`

## Installation Notes

This repo does not yet include packaging metadata. For now, keep the parent directory of `MLProj` on your `PYTHONPATH`, or add packaging later with `pyproject.toml`.

Core dependencies depend on which models you use:

```bash
pip install numpy pandas scikit-learn scikit-optimize optuna imbalanced-learn
pip install xgboost lightgbm catboost
```

Optional explanation/plotting dependencies:

```bash
pip install shap yellowbrick seaborn matplotlib dtreeviz statsmodels
```

CUDA support is explicit opt-in and requires RAPIDS packages such as `cudf`, `cupy`, and `cuml`. Install RAPIDS using the official instructions for your CUDA/Python version.

## Quick Start

```python
from MLProj import train

result = train(
    task="classification",
    model="random_forest",
    X=X_train,
    y=y_train,
)

predictions = result.predict(X_test)
metrics = result.evaluate(X_test, y_test)
print(metrics)
```

## Hyperparameter Tuning

Set `tune=True` to use Optuna. Presets control trial count, timeout, and whether cross-validation is used.

```python
result = train(
    task="regression",
    model="xgboost",
    X=X_train,
    y=y_train,
    tune=True,
    preset="balanced",
)

print(result.best_params)
print(result.score(X_test, y_test))
```

Available presets:

- `fast`: 10 trials, shorter timeout, no cross-validation
- `balanced`: 30 trials, moderate timeout, no cross-validation
- `accurate`: 60 trials, longer timeout, cross-validation enabled

You can override the preset:

```python
result = train(
    "classification",
    "lightgbm",
    X_train,
    y_train,
    tune=True,
    n_trials=100,
    timeout=1800,
    crossval=True,
)
```

## Built-In Metrics

Classification:

- `accuracy`
- `f1`
- `roc_auc`

Regression:

- `rmse`
- `mse`
- `mae`
- `r2`

```python
result = train(
    "classification",
    "logreg",
    X_train,
    y_train,
    tune=True,
    metric="f1",
)
```

## Custom Losses and Scorers

Custom loss functions are lower-is-better by default:

```python
def weighted_error(y_true, y_pred):
    return ((y_true != y_pred) * 2.0).mean()

result = train(
    "classification",
    "random_forest",
    X_train,
    y_train,
    tune=True,
    metric=weighted_error,
    metric_direction="minimize",
)
```

Score-like functions should be marked as higher-is-better:

```python
def custom_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

result = train(
    "classification",
    "svm",
    X_train,
    y_train,
    tune=True,
    metric=custom_accuracy,
    metric_direction="maximize",
)
```

For probability or decision-score metrics, use `metric_response="score"`:

```python
from sklearn.metrics import roc_auc_score

result = train(
    "classification",
    "xgboost",
    X_train,
    y_train,
    tune=True,
    metric=roc_auc_score,
    metric_direction="maximize",
    metric_response="score",
)
```

The direct wrapper API exposes the same route:

```python
from MLProj import RandomForestModel

model = RandomForestModel(mode="Classifier")
model.set_loss_function(weighted_error, direction="minimize")
model.optimize(X_train, y_train)
model.fit()
```

## CUDA Usage

CUDA is disabled by default. You must explicitly request it:

```python
result = train(
    "classification",
    "random_forest",
    X_train,
    y_train,
    supports_cuda=True,
)
```

When `supports_cuda=True`, the base pipeline checks for RAPIDS dependencies and ports training data through cuDF/cuML where the model supports it. If RAPIDS is missing, construction fails early instead of silently falling back to CPU.

Current CUDA notes:

- `random_forest`, `knn`, `logistic_regression`, and `svm` use cuML routes where supported.
- `xgboost`, `lightgbm`, and `catboost` use their native GPU options when requested.
- `polynomial_regression` rejects CUDA for now.
- SMOTE and explanation plots are CPU-only and are blocked in CUDA mode.
- Ensemble stacking is CPU-only; CUDA-backed models should use hard or soft voting.

## Ensembles

`Ensemble` accepts fitted models or `TrainingResult` objects, evaluates model subsets, and chooses the best voting strategy.

```python
from MLProj import Ensemble

rf = train("classification", "random_forest", X_train, y_train)
xgb = train("classification", "xgboost", X_train, y_train)
lgb = train("classification", "lightgbm", X_train, y_train)

ensemble = Ensemble(
    models=[rf, xgb, lgb],
    X_train=X_train,
    y_train=y_train,
    X_test=X_valid,
    y_test=y_valid,
)

ensemble.ensemble_auto("auto")
print(ensemble.summary())
predictions = ensemble.predict(X_test)
```

Hard, soft, and stacking are supported for CPU-backed models. Soft voting and stacking require models with `predict_proba`.

Custom ensemble losses work the same way:

```python
ensemble.set_loss_function(weighted_error, direction="minimize")
ensemble.ensemble_auto("hard")
```

## Result Object

`train()` returns a `TrainingResult`:

```python
result.predict(X_test)
result.predict_proba(X_test)
result.score(X_test, y_test)
result.evaluate(X_test, y_test)
result.best_params
result.study
result.feature_names
result.unwrap()
```

Use `unwrap()` when you need direct access to the underlying wrapper.

## Discovering Models

```python
from MLProj import available_models, model_info

print(available_models())
print(available_models("classification"))
print(model_info("regression"))
```

## Development

Run a syntax check:

```bash
python -m py_compile *.py
```

The neural-network files under `Neural/` are placeholders for now.
