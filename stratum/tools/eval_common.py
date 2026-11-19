from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from ..common import load_csv
from ..session import get_run


def load_run_model(model_id: str, sid: str | None = None) -> tuple[dict | None, Any | None, dict | None]:
    run = get_run(model_id)
    if not run:
        return None, None, {"error": f"Model '{model_id}' not found. Run stratum_train first."}
    if sid and sid != run["sid"]:
        return None, None, {"error": f"Model '{model_id}' does not belong to session '{sid}'."}
    if not run.get("artifact_path"):
        return None, None, {"error": f"Model '{model_id}' has no fitted artifact path."}
    try:
        return run, joblib.load(run["artifact_path"]), None
    except Exception as exc:
        return None, None, {"error": f"Unable to load model artifact: {type(exc).__name__}: {exc}"}


def load_labeled_frame(file: str, run: dict, target: str | None = None) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.Series | None, dict | None]:
    target = target or run.get("target_col")
    if not target:
        return None, None, None, {"error": "Target column is required because the run does not record one."}
    df, error = load_csv(file, target)
    if error:
        return None, None, None, error
    features = list(run.get("feature_names") or [])
    missing = [name for name in features if name not in df.columns]
    if missing:
        return None, None, None, {"error": f"Input CSV is missing required feature columns: {missing}"}
    return df, df[features].copy(), df[target], None


def predict_scores(estimator, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    predictions = np.asarray(estimator.predict(X))
    scores = None
    if hasattr(estimator, "predict_proba"):
        try:
            scores = np.asarray(estimator.predict_proba(X))
        except Exception:
            scores = None
    return predictions, scores


def classification_metrics(y_true, y_pred, y_score=None, class_labels=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(class_labels) if class_labels is not None else list(pd.unique(y_true))
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    support = np.asarray([(y_true == label).sum() for label in labels])
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }
    if y_score is not None:
        y_score = np.asarray(y_score)
        try:
            if y_score.ndim != 2 or y_score.shape[1] != len(labels):
                return metrics
            if len(labels) == 2:
                positive = labels[1]
                metrics["positive_class"] = str(positive)
                binary_true = (y_true == positive).astype(int)
                metrics["pr_auc"] = float(average_precision_score(binary_true, y_score[:, 1]))
                metrics["brier_score"] = float(brier_score_loss(binary_true, y_score[:, 1]))
                try:
                    metrics["roc_auc"] = float(roc_auc_score(binary_true, y_score[:, 1]))
                except ValueError:
                    pass
            elif len(labels) > 2:
                one_hot = np.column_stack([y_true == label for label in labels]).astype(float)
                metrics["brier_score"] = float(np.mean(np.sum((y_score - one_hot) ** 2, axis=1)))
                try:
                    metrics["roc_auc_ovr_weighted"] = float(
                        roc_auc_score(y_true, y_score, labels=labels, multi_class="ovr", average="weighted")
                    )
                except ValueError:
                    pass
        except ValueError:
            pass
    return metrics


def regression_metrics(y_true, y_pred) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_predictions(task: str, y_true, y_pred, y_score=None, class_labels=None) -> dict:
    if task == "regression":
        return regression_metrics(y_true, y_pred)
    return classification_metrics(y_true, y_pred, y_score, class_labels)


def confusion_summary(y_true, y_pred) -> dict:
    labels = sorted(pd.Series(y_true).dropna().unique().tolist())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "labels": [str(label) for label in labels],
        "matrix": matrix.astype(int).tolist(),
    }


def residual_summary(y_true, y_pred) -> dict:
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    abs_residuals = np.abs(residuals)
    return {
        "mean_residual": float(np.mean(residuals)),
        "mean_abs_residual": float(np.mean(abs_residuals)),
        "p50_abs_residual": float(np.percentile(abs_residuals, 50)),
        "p90_abs_residual": float(np.percentile(abs_residuals, 90)),
        "max_abs_residual": float(np.max(abs_residuals)),
    }


def calibration_bins(y_true, y_score, bins: int = 10) -> list[dict]:
    if y_score is None or getattr(y_score, "ndim", 1) != 2 or y_score.shape[1] != 2:
        return []
    y = np.asarray(y_true)
    labels = sorted(pd.Series(y).dropna().unique().tolist())
    if len(labels) != 2:
        return []
    positive = labels[1]
    proba = np.asarray(y_score[:, 1], dtype=float)
    actual = (y == positive).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (proba >= start) & (proba < end if end < 1 else proba <= end)
        if not mask.any():
            continue
        rows.append({
            "bin_start": round(float(start), 4),
            "bin_end": round(float(end), 4),
            "count": int(mask.sum()),
            "avg_confidence": float(proba[mask].mean()),
            "positive_rate": float(actual[mask].mean()),
        })
    return rows
