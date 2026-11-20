from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path

import pandas as pd

from .data.splitter import file_sha256, frame_fingerprint


PROVENANCE_SCHEMA_VERSION = 1

METRIC_DEFINITIONS = {
    "accuracy": "Fraction of evaluated rows whose predicted class matches the true class.",
    "balanced_accuracy": "Mean per-class recall; less dominated by the majority class than accuracy.",
    "f1_weighted": "Support-weighted mean of per-class F1 scores.",
    "precision_weighted": "Support-weighted mean of per-class precision.",
    "recall_weighted": "Support-weighted mean of per-class recall.",
    "pr_auc": "Average precision over the precision-recall curve for the recorded positive class.",
    "roc_auc": "Area under the binary receiver operating characteristic curve.",
    "brier_score": "Mean squared probability error; lower is better.",
    "mae": "Mean absolute prediction error; lower is better.",
    "mse": "Mean squared prediction error; lower is better.",
    "rmse": "Square root of mean squared prediction error; lower is better.",
    "r2": "Proportion of target variance explained relative to the evaluation-set mean baseline.",
}


def environment_snapshot() -> dict:
    distributions = {
        "mlproj": "mlproj",
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit_learn": "scikit-learn",
        "joblib": "joblib",
        "optuna": "optuna",
    }
    versions = {}
    for key, distribution in distributions.items():
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions}


def build_run_provenance(
    *,
    df: pd.DataFrame,
    source_path: str | Path,
    target: str,
    feature_names: list[str],
    split_metadata: dict,
    training_policy: dict,
    model_name: str,
    task: str,
    best_params: dict,
    preprocessing: dict,
    metrics: dict,
    artifact_path: str | Path,
    decision_ids: list[str],
) -> dict:
    source = Path(source_path).expanduser().resolve()
    artifact = Path(artifact_path)
    schema = {
        str(column): {
            "dtype": str(df[column].dtype),
            "nullable": bool(df[column].isna().any()),
            "null_count": int(df[column].isna().sum()),
            "n_unique": int(df[column].nunique(dropna=True)),
        }
        for column in df.columns
    }
    selection = training_policy.get("selection", {})
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "dataset": {
            "path": str(source),
            "sha256": file_sha256(source),
            "frame_fingerprint": frame_fingerprint(df),
            "rows": int(len(df)),
            "columns": schema,
        },
        "target": target,
        "feature_names": list(feature_names),
        "split": split_metadata,
        "selection": selection,
        "metric_definitions": {
            name: METRIC_DEFINITIONS.get(name, f"Optimization/evaluation metric '{name}'.")
            for name in set(metrics) | {selection.get("metric", "")}
            if name
        },
        "model": {
            "name": model_name,
            "task": task,
            "best_params": best_params,
            "preprocessing_pipeline": preprocessing,
        },
        "environment": environment_snapshot(),
        "artifact": {
            "path": str(artifact.resolve()),
            "format": "joblib",
            "sha256": file_sha256(artifact),
            "size_bytes": int(artifact.stat().st_size),
        },
        "lineage": {"parent_model_id": None, "decision_ids": sorted(set(decision_ids))},
    }
