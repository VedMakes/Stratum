from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from ..common import json_safe
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
) -> dict:
    run, estimator, error = load_run_model(model_id, sid)
    if error:
        return error
    df, X, y, error = load_labeled_frame(file, run, target)
    if error:
        return error
    if folds < 2:
        return {"error": "folds must be at least 2"}
    if folds > len(df):
        return {"error": "folds cannot exceed the number of rows"}

    warnings = []
    if run["task"] == "classification":
        counts = y.value_counts()
        if len(counts) > 1:
            max_folds = int(counts.min())
            if max_folds < 2:
                return {"error": "Each class needs at least 2 rows for stability analysis."}
            if folds > max_folds:
                warnings.append(f"Reduced folds from {folds} to {max_folds} because of class support.")
                folds = max_folds
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)

    scorer = _scoring(run["task"], metric)
    try:
        scores = cross_val_score(
            clone(estimator), X, y, cv=cv, scoring=scorer, n_jobs=None, error_score="raise"
        )
    except Exception as exc:
        return {"error": f"Stability analysis failed: {type(exc).__name__}: {exc}"}

    signed_scores = -scores if scorer.startswith("neg_") else scores
    mean = float(np.mean(signed_scores))
    std = float(np.std(signed_scores))
    relative_std = float(std / abs(mean)) if mean else None
    stability = "stable" if relative_std is not None and relative_std < .05 else (
        "moderate" if relative_std is not None and relative_std < .15 else "unstable"
    )
    result = {
        "sid": run["sid"],
        "model_id": model_id,
        "rows_evaluated": len(df),
        "folds": folds,
        "metric": scorer.removeprefix("neg_"),
        "fold_scores": [float(score) for score in signed_scores],
        "mean": mean,
        "std": std,
        "min": float(np.min(signed_scores)),
        "max": float(np.max(signed_scores)),
        "relative_std": relative_std,
        "stability": stability,
        "warnings": warnings,
    }
    save_artifact(
        run["sid"],
        "stability",
        file,
        model_id=model_id,
        metadata=result,
    )
    return json_safe(result)
