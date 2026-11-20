from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    GroupKFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
)

from ..common import json_safe
from ..data.splitter import file_sha256, frame_fingerprint
from ..session import save_artifact
from .eval_common import load_labeled_frame, load_run_model


def _scoring(task: str, metric: str | None) -> str:
    if metric:
        return metric
    return "neg_root_mean_squared_error" if task == "regression" else "f1_weighted"


def run_stability(
    model_id: str,
    file: str,
    target: str | None = None,
    sid: str | None = None,
    folds: int = 5,
    metric: str | None = None,
    repeats: int = 5,
    seed: int = 42,
) -> dict:
    run, estimator, error = load_run_model(model_id, sid)
    if error:
        return error
    df, X, y, error = load_labeled_frame(file, run, target)
    if error:
        return error
    if folds < 2:
        return {"error": "folds must be at least 2"}
    if not 1 <= repeats <= 20:
        return {"error": "repeats must be between 1 and 20"}
    if not isinstance(seed, int):
        return {"error": "seed must be an integer"}

    split_metadata = run.get("split_metadata") or {}
    strategy = split_metadata.get("strategy", "stratified")
    parameters = split_metadata.get("parameters") or {}
    warnings = list(run.get("integrity_warnings", []))
    groups = None
    temporal_gap = 0

    if strategy == "predefined":
        return {
            "error": "Repeated CV cannot preserve a predefined partition. Use stratum_validate on its designated test data."
        }
    if strategy == "group":
        group_col = parameters.get("group_col")
        if not group_col or group_col not in df:
            return {"error": "Group-aware stability requires the saved group column in the evaluation CSV."}
        if df[group_col].isna().any():
            return {"error": f"Group column '{group_col}' contains null values."}
        groups = df[group_col].to_numpy()
        group_count = int(pd.Series(groups).nunique())
        if group_count < 2:
            return {"error": "Group-aware stability requires at least two distinct groups."}
        if folds > group_count:
            warnings.append(f"Reduced folds from {folds} to {group_count} because only {group_count} groups are available.")
            folds = group_count
        method = "repeated_group_kfold"
        if run["task"] == "classification":
            warnings.append("Group folds are not stratified; inspect per-fold scores and class support.")
    elif strategy == "temporal":
        time_col = parameters.get("time_col")
        if not time_col or time_col not in df:
            return {"error": "Time-aware stability requires the saved time column in the evaluation CSV."}
        parsed = pd.to_datetime(df[time_col], errors="coerce")
        failures = parsed.isna() & df[time_col].notna()
        if failures.any():
            return {"error": f"Time column '{time_col}' contains {int(failures.sum())} unparseable values."}
        if parsed.isna().any():
            return {"error": f"Time column '{time_col}' contains null values; ordered evaluation is undefined."}
        temporal_gap = int(parameters.get("gap_rows", 0))
        order = parsed.sort_values(kind="stable").index
        df = df.loc[order].reset_index(drop=True)
        X = df[list(run.get("feature_names") or [])].copy()
        y = df[target or run.get("target_col")].reset_index(drop=True)
        if repeats > 1:
            warnings.append("Temporal backtesting uses one ordered walk-forward schedule; repeats are not applicable.")
        repeats = 1
        method = "time_series"
    elif strategy in {"random", "stratified"}:
        method = "repeated_stratified_kfold" if run["task"] == "classification" else "repeated_kfold"
    else:
        return {"error": f"Unsupported saved split strategy '{strategy}'."}

    if folds > len(X):
        return {"error": "folds cannot exceed the number of evaluation rows"}
    if method == "repeated_stratified_kfold":
        counts = y.value_counts()
        if len(counts) > 1:
            max_folds = int(counts.min())
            if max_folds < 2:
                return {"error": "Each class needs at least 2 rows for stability analysis."}
            if folds > max_folds:
                warnings.append(f"Reduced folds from {folds} to {max_folds} because of class support.")
                folds = max_folds
    if method == "time_series" and folds > len(X) - temporal_gap - 1:
        return {"error": "folds and temporal gap leave insufficient rows for ordered evaluation."}

    scorer = _scoring(run["task"], metric)
    fingerprint_columns = list(run.get("feature_names") or [])
    fingerprint_columns.append(target or run.get("target_col"))
    if strategy == "group":
        fingerprint_columns.append(parameters["group_col"])
    elif strategy == "temporal":
        fingerprint_columns.append(parameters["time_col"])
    fingerprint_columns = list(dict.fromkeys(fingerprint_columns))
    protocol = {
        "method": method,
        "folds": folds,
        "repeats": repeats,
        "seed": seed,
        "metric": scorer,
        "split_strategy": strategy,
        "time_col": parameters.get("time_col") if strategy == "temporal" else None,
        "gap_rows": temporal_gap if strategy == "temporal" else None,
        "group_col": parameters.get("group_col") if strategy == "group" else None,
        "source_sha256": file_sha256(file),
        "data_fingerprint": frame_fingerprint(df[fingerprint_columns]),
    }

    scores_by_repeat = []
    try:
        for repeat in range(repeats):
            if method == "repeated_stratified_kfold":
                cv = RepeatedStratifiedKFold(
                    n_splits=folds, n_repeats=1, random_state=seed + repeat
                )
            elif method == "repeated_kfold":
                cv = RepeatedKFold(n_splits=folds, n_repeats=1, random_state=seed + repeat)
            elif method == "repeated_group_kfold":
                cv = GroupKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
            else:
                cv = TimeSeriesSplit(n_splits=folds, gap=temporal_gap)
            scores = cross_val_score(
                clone(estimator), X, y, cv=cv, groups=groups,
                scoring=scorer, n_jobs=None, error_score="raise",
            )
            signed = -scores if scorer.startswith("neg_") else scores
            scores_by_repeat.append([float(score) for score in signed])
    except Exception as exc:
        return {"error": f"Stability analysis failed: {type(exc).__name__}: {exc}"}

    fold_scores = [score for repeat_scores in scores_by_repeat for score in repeat_scores]
    score_array = np.asarray(fold_scores, dtype=float)
    mean = float(np.mean(score_array))
    std = float(np.std(score_array, ddof=1)) if len(score_array) > 1 else 0.0
    relative_std = float(std / abs(mean)) if mean else None
    stability = "stable" if relative_std is not None and relative_std < .05 else (
        "moderate" if relative_std is not None and relative_std < .15 else "unstable"
    )
    result = {
        "sid": run["sid"],
        "model_id": model_id,
        "rows_evaluated": len(df),
        "folds": folds,
        "repeats": repeats,
        "metric": scorer.removeprefix("neg_"),
        "fold_scores": fold_scores,
        "repeat_scores": [float(np.mean(values)) for values in scores_by_repeat],
        "mean": mean,
        "std": std,
        "score_range_90pct": [float(value) for value in np.percentile(score_array, [5, 95])],
        "min": float(np.min(score_array)),
        "max": float(np.max(score_array)),
        "relative_std": relative_std,
        "stability": stability,
        "protocol": protocol,
        "warnings": warnings + [
            "Fold-score spread is descriptive; overlapping CV training sets are not independent confidence intervals."
        ],
    }
    save_artifact(
        run["sid"], "stability", file, model_id=model_id, metadata=result,
    )
    return json_safe(result)
