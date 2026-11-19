from __future__ import annotations

import pandas as pd

from ..common import json_safe
from ..session import save_artifact
from .eval_common import (
    evaluate_predictions,
    load_labeled_frame,
    load_run_model,
    predict_scores,
)


def run_group_evaluation(
    model_id: str,
    file: str,
    group_col: str,
    target: str | None = None,
    sid: str | None = None,
    min_support: int = 30,
) -> dict:
    run, estimator, error = load_run_model(model_id, sid)
    if error:
        return error
    df, X, y, error = load_labeled_frame(file, run, target)
    if error:
        return error
    if group_col not in df.columns:
        return {"error": f"Group column '{group_col}' was not found in the labeled CSV."}
    if group_col == (target or run.get("target_col")):
        return {"error": "group_col cannot be the target column."}
    if min_support < 2:
        return {"error": "min_support must be at least 2."}

    try:
        predictions, scores = predict_scores(estimator, X)
        class_labels = getattr(estimator, "classes_", None)
        overall = evaluate_predictions(run["task"], y, predictions, scores, class_labels)
    except Exception as exc:
        return {"error": f"Group evaluation failed: {type(exc).__name__}: {exc}"}

    grouped = df.groupby(group_col, dropna=False, sort=False).indices
    supported = [(value, positions) for value, positions in grouped.items() if len(positions) >= min_support]
    if len(supported) > 100:
        return {
            "error": "More than 100 groups meet min_support; increase min_support to keep the result interpretable."
        }

    groups = []
    for value, positions in supported:
        positions = list(positions)
        group_y = y.iloc[positions]
        group_predictions = predictions[positions]
        group_scores = scores[positions] if scores is not None else None
        label = "<missing>" if pd.isna(value) else str(value)
        try:
            metrics = evaluate_predictions(
                run["task"], group_y, group_predictions, group_scores, class_labels
            )
        except Exception as exc:
            return {"error": f"Metrics failed for group {label!r}: {type(exc).__name__}: {exc}"}
        groups.append({"value": label, "support": len(positions), "metrics": metrics})
    groups.sort(key=lambda item: (-item["support"], item["value"]))

    excluded = len(grouped) - len(supported)
    warnings = []
    if excluded:
        warnings.append(
            f"Excluded {excluded} group(s) with fewer than {min_support} rows; their metrics are not reported."
        )
    if not groups:
        warnings.append("No groups met the minimum support threshold.")
    result = {
        "sid": run["sid"],
        "model_id": model_id,
        "group_column": group_col,
        "target": target or run.get("target_col"),
        "task": run["task"],
        "min_support": min_support,
        "rows_evaluated": len(df),
        "overall_metrics": overall,
        "groups": groups,
        "excluded_groups": excluded,
        "warnings": warnings,
        "note": "Descriptive subgroup performance only; this is not a fairness or causal assessment. Interpret small groups cautiously.",
    }
    save_artifact(
        run["sid"],
        "group_evaluation",
        file,
        model_id=model_id,
        metadata={
            "group_column": group_col,
            "min_support": min_support,
            "rows_evaluated": len(df),
            "groups": groups,
            "excluded_groups": excluded,
            "warnings": warnings,
        },
    )
    return json_safe(result)
