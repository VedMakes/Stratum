from __future__ import annotations

from ..common import json_safe
from ..session import save_artifact
from .eval_common import (
    calibration_bins,
    confusion_summary,
    evaluate_predictions,
    evaluation_warnings,
    load_labeled_frame,
    load_run_model,
    predict_scores,
    residual_summary,
)


def run_validate(
    model_id: str, file: str, target: str | None = None, sid: str | None = None
) -> dict:
    run, estimator, error = load_run_model(model_id, sid)
    if error:
        return error
    df, X, y, error = load_labeled_frame(file, run, target)
    if error:
        return error
    try:
        predictions, scores = predict_scores(estimator, X)
        metrics = evaluate_predictions(
            run["task"], y, predictions, scores,
            class_labels=getattr(estimator, "classes_", None),
        )
    except Exception as exc:
        return {"error": f"Validation failed: {type(exc).__name__}: {exc}"}

    result = {
        "sid": run["sid"],
        "model_id": model_id,
        "rows_evaluated": len(df),
        "metrics": metrics,
        "target": target or run.get("target_col"),
        "warnings": evaluation_warnings(run["task"], y, metrics, scores),
    }
    if run["task"] == "regression":
        result["residuals"] = residual_summary(y, predictions)
        if result["residuals"]["abs_error_vs_prediction_spearman"] is not None and abs(
            result["residuals"]["abs_error_vs_prediction_spearman"]
        ) >= 0.5:
            result["warnings"].append(
                "Absolute error is strongly associated with prediction magnitude; inspect errors across the prediction range."
            )
    else:
        result["confusion_matrix"] = confusion_summary(y, predictions)
        result["calibration_bins"] = calibration_bins(y, scores)

    save_artifact(
        run["sid"],
        "validation",
        file,
        model_id=model_id,
        metadata={
            "rows_evaluated": len(df),
            "metrics": metrics,
            "target": result["target"],
        },
    )
    return json_safe(result)
