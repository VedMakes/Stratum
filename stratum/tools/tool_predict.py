from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..common import json_safe, load_csv
from ..provenance import verify_run_artifact
from ..session import get_run, get_workflow_blockers, save_artifact


def _confidence(estimator, X: pd.DataFrame, predictions: np.ndarray) -> list[float | None]:
    if not hasattr(estimator, "predict_proba"):
        return [None] * len(predictions)
    probabilities = np.asarray(estimator.predict_proba(X))
    if probabilities.ndim != 2 or probabilities.shape[0] != len(predictions):
        return [None] * len(predictions)
    return [float(value) for value in probabilities.max(axis=1)]


def _positive_count(predictions: np.ndarray, confidence: list[float | None]) -> int | None:
    try:
        labels = np.asarray(predictions)
        if set(pd.Series(labels).dropna().unique()) <= {0, 1}:
            confident = np.asarray([c or 0 for c in confidence], dtype=float) >= .8
            return int(np.sum((labels == 1) & confident))
    except Exception:
        return None
    return None


def run_predict(
    model_id: str, file: str, output_path: str | None = None, sid: str | None = None
) -> dict:
    run = get_run(model_id)
    if not run:
        return {"error": f"Model '{model_id}' not found. Run stratum_train first."}
    if sid and sid != run["sid"]:
        return {"error": f"Model '{model_id}' does not belong to session '{sid}'."}
    blockers = get_workflow_blockers(run["sid"])
    if blockers:
        return {
            "error": "Prediction is blocked by unresolved domain decisions.",
            "sid": run["sid"],
            "recommended_action": "ask_user",
            "blocked_decisions": blockers,
        }
    if not run.get("artifact_path"):
        return {"error": f"Model '{model_id}' has no fitted artifact path."}
    integrity = verify_run_artifact(run)
    if "error" in integrity:
        return integrity

    df, error = load_csv(file)
    if error:
        return error
    features = list(run.get("feature_names") or [])
    if not features:
        return {"error": f"Model '{model_id}' does not record feature names."}
    missing = [name for name in features if name not in df.columns]
    if missing:
        return {"error": f"Input CSV is missing required feature columns: {missing}"}

    try:
        estimator = joblib.load(run["artifact_path"])
        X = df[features].copy()
        predictions = np.asarray(estimator.predict(X))
        confidence = _confidence(estimator, X, predictions)
    except Exception as exc:
        return {"error": f"Prediction failed: {type(exc).__name__}: {exc}"}

    rows = [
        {
            "row_index": int(index),
            "prediction": prediction,
            "confidence": confidence[index],
        }
        for index, prediction in enumerate(predictions.tolist())
    ]
    distribution = {
        str(key): int(value)
        for key, value in pd.Series(predictions).value_counts(dropna=False).items()
    }

    saved_path = None
    if output_path:
        saved_path = Path(output_path).expanduser()
        if saved_path.exists():
            return {"error": f"Refusing to overwrite existing file: {saved_path}"}
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        output = df.copy()
        output["prediction"] = predictions
        if any(value is not None for value in confidence):
            output["confidence"] = confidence
        output.to_csv(saved_path, index=False)
        save_artifact(run["sid"], "predictions_csv", str(saved_path), model_id=model_id)

    return json_safe({
        "sid": run["sid"],
        "model_id": model_id,
        "rows_scored": len(df),
        "predictions": rows,
        "class_distribution": distribution if run["task"] == "classification" else None,
        "high_confidence_positives": (
            _positive_count(predictions, confidence)
            if run["task"] == "classification" else None
        ),
        "warnings": integrity["warnings"],
        "output_path": str(saved_path) if saved_path else None,
    })
