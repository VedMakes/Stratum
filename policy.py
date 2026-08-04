from dataclasses import asdict, dataclass

import pandas as pd

from .registry import get_model_spec, normalize_model_name, normalize_task


@dataclass(frozen=True)
class TrainingPolicy:
    task: str
    model: str
    preset: str
    n_trials: int
    timeout: int
    crossval: bool = False

    def to_dict(self):
        return asdict(self)


def infer_task(target) -> str:
    series = target if isinstance(target, pd.Series) else pd.Series(target)
    unique = series.nunique(dropna=True)
    if unique <= 20 and (
        pd.api.types.is_integer_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or not pd.api.types.is_numeric_dtype(series)
    ):
        return "classification"
    return "regression"


def select_auto_model(task: str, rows: int) -> str:
    normalized_task = normalize_task(task)
    if rows < 1000:
        return "logistic_regression" if normalized_task == "classification" else "random_forest"
    return "lightgbm"


def tuning_budget(rows: int) -> tuple[str, int, int]:
    if rows < 500:
        return "fast", 10, 180
    if rows <= 5000:
        return "balanced", 30, 500
    return "balanced", 50, 1200


def resolve_training_policy(target, rows: int, task: str = "auto", model: str = "auto") -> TrainingPolicy:
    selected_task = infer_task(target) if task == "auto" else normalize_task(task)
    selected_model = (
        select_auto_model(selected_task, rows)
        if model == "auto"
        else normalize_model_name(model)
    )
    get_model_spec(selected_model, selected_task)
    preset, n_trials, timeout = tuning_budget(rows)
    return TrainingPolicy(
        task=selected_task,
        model=selected_model,
        preset=preset,
        n_trials=n_trials,
        timeout=timeout,
    )
