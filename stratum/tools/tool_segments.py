from __future__ import annotations

import numpy as np
import pandas as pd

from ..common import json_safe
from ..session import save_artifact
from .eval_common import load_labeled_frame, load_run_model, predict_scores


def _segment_masks(df: pd.DataFrame, target: str, max_categories: int):
    for col in df.columns:
        if col == target:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) >= 4:
            try:
                buckets = pd.qcut(series, q=4, duplicates="drop")
            except ValueError:
                continue
            for bucket in buckets.dropna().unique():
                mask = buckets == bucket
                yield col, str(bucket), mask
        elif series.nunique(dropna=True) <= max_categories:
            for value in series.dropna().unique():
                mask = series == value
                yield col, str(value), mask


def _error_values(task: str, y, predictions) -> np.ndarray:
    if task == "regression":
        return np.abs(np.asarray(y, dtype=float) - np.asarray(predictions, dtype=float))
    return (np.asarray(y) != np.asarray(predictions)).astype(float)


def run_segments(
    model_id: str,
    file: str,
    target: str | None = None,
    sid: str | None = None,
    min_support: int = 20,
    max_categories: int = 12,
) -> dict:
    run, estimator, error = load_run_model(model_id, sid)
    if error:
        return error
    df, X, y, error = load_labeled_frame(file, run, target)
    if error:
        return error
    if min_support < 2:
        return {"error": "min_support must be at least 2"}

    try:
        predictions, _ = predict_scores(estimator, X)
        errors = _error_values(run["task"], y, predictions)
    except Exception as exc:
        return {"error": f"Segment analysis failed: {type(exc).__name__}: {exc}"}

    overall = float(np.mean(errors))
    segments = []
    target_col = target or run.get("target_col")
    for col, value, mask in _segment_masks(df, target_col, max_categories):
        count = int(mask.sum())
        if count < min_support or count >= len(df):
            continue
        segment_error = float(np.mean(errors[mask.to_numpy()]))
        segments.append({
            "column": str(col),
            "value": value,
            "rule": f"{col} == {value!r}",
            "support": count,
            "error_metric": "mae" if run["task"] == "regression" else "error_rate",
            "error": segment_error,
            "overall_error": overall,
            "delta": segment_error - overall,
        })
    segments.sort(key=lambda item: (item["delta"], item["support"]), reverse=True)
    worst = segments[:10]

    save_artifact(
        run["sid"],
        "segment_analysis",
        file,
        model_id=model_id,
        metadata={
            "overall_error": overall,
            "segments": worst,
            "min_support": min_support,
        },
    )
    return json_safe({
        "sid": run["sid"],
        "model_id": model_id,
        "rows_evaluated": len(df),
        "overall_error": overall,
        "segments": worst,
    })
