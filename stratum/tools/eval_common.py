from __future__ import annotations

import time
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
from ..provenance import verify_run_artifact
from ..session import get_run


def load_run_model(model_id: str, sid: str | None = None) -> tuple[dict | None, Any | None, dict | None]:
    run = get_run(model_id)
    if not run:
        return None, None, {"error": f"Model '{model_id}' not found. Run stratum_train first."}
    if sid and sid != run["sid"]:
        return None, None, {"error": f"Model '{model_id}' does not belong to session '{sid}'."}
    if not run.get("artifact_path"):
        return None, None, {"error": f"Model '{model_id}' has no fitted artifact path."}
    integrity = verify_run_artifact(run)
    if "error" in integrity:
        return None, None, {"error": integrity["error"]}
    try:
        run["integrity_warnings"] = integrity["warnings"]
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


def inference_timing(predictor, X: pd.DataFrame, repeats: int = 5, max_rows: int = 100) -> dict:
    sample = X.iloc[:max_rows]
    if sample.empty:
        return {"rows_per_batch": 0, "repeats": 0, "median_ms_per_row": None}
    predictor(sample)
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        predictor(sample)
        durations.append(time.perf_counter() - started)
    median_seconds = float(np.median(durations))
    return {
        "rows_per_batch": len(sample),
        "repeats": repeats,
        "median_batch_seconds": median_seconds,
        "median_ms_per_row": median_seconds * 1000 / len(sample),
        "method": "warm local batch prediction; hardware and system load dependent",
    }


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
    residual_median = float(np.median(residuals))
    residual_mad = float(np.median(np.abs(residuals - residual_median)))
    robust_scale = 1.4826 * residual_mad
    if robust_scale:
        outlier_fraction = float(np.mean(np.abs(residuals - residual_median) > 3 * robust_scale))
    else:
        outlier_fraction = float(np.mean(residuals != residual_median))
    abs_error_rank_correlation = None
    if len(residuals) > 1 and np.ptp(y_pred) > 0 and np.ptp(abs_residuals) > 0:
        correlation = pd.Series(abs_residuals).corr(pd.Series(y_pred), method="spearman")
        if pd.notna(correlation):
            abs_error_rank_correlation = float(correlation)
    return {
        "mean_residual": float(np.mean(residuals)),
        "median_residual": residual_median,
        "mean_abs_residual": float(np.mean(abs_residuals)),
        "median_abs_residual": float(np.median(abs_residuals)),
        "residual_mad": residual_mad,
        "p50_abs_residual": float(np.percentile(abs_residuals, 50)),
        "p90_abs_residual": float(np.percentile(abs_residuals, 90)),
        "p95_abs_residual": float(np.percentile(abs_residuals, 95)),
        "max_abs_residual": float(np.max(abs_residuals)),
        "robust_outlier_fraction": outlier_fraction,
        "abs_error_vs_prediction_spearman": abs_error_rank_correlation,
    }


def evaluation_warnings(task: str, y_true, metrics: dict, y_score=None) -> list[str]:
    y = np.asarray(y_true)
    warnings = []
    if len(y) < 30:
        warnings.append(f"Evaluation has only {len(y)} rows; metrics may be highly uncertain.")
    if task == "regression":
        if np.ptp(y.astype(float)) == 0:
            warnings.append("Target is constant in this evaluation data; R-squared is not informative.")
        elif metrics.get("r2", 0) < 0:
            warnings.append(
                "R-squared is negative; predictions perform worse than the evaluation-set target mean baseline."
            )
        return warnings

    counts = pd.Series(y).value_counts(dropna=True)
    if len(counts) < 2:
        warnings.append("Evaluation contains only one target class; class-separation metrics are unavailable.")
    elif int(counts.min()) < 5:
        warnings.append("At least one evaluation class has fewer than 5 rows; per-class metrics are unstable.")
    if len(counts) > 1 and float(counts.min() / counts.max()) < 0.1:
        warnings.append(
            "Target classes are highly imbalanced; accuracy can conceal minority-class errors. "
            "Inspect balanced accuracy, per-class recall, and PR-AUC."
        )
    if y_score is None:
        warnings.append("Probability-based calibration and Brier diagnostics are unavailable for this estimator.")
    return warnings


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
