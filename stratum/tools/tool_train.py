from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from ..common import load_csv, stratum_home
from ..core import core_policy, core_train
from ..data.contracts import build_data_contract
from ..session import ensure_session, save_artifact, save_run


def _error_summary(result, X_test: pd.DataFrame, y_test: pd.Series, task: str) -> str:
    predictions = np.asarray(result.predict(X_test))
    actual = np.asarray(y_test)
    if task == "classification":
        error = predictions != actual
    else:
        residual = np.abs(predictions.astype(float) - actual.astype(float))
        error = residual >= np.quantile(residual, .75)
    if not error.any():
        return "No concentrated holdout error pattern was detected."
    usable = X_test.drop(columns=[
        col for col in X_test.select_dtypes(exclude=np.number).columns
        if X_test[col].nunique(dropna=True) > 50
    ])
    encoded = pd.get_dummies(usable, drop_first=True).astype(float)
    if encoded.shape[1] == 0:
        return "Errors exist, but no usable feature split was available."
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(encoded, error.astype(int))
    text = export_text(tree, feature_names=list(encoded.columns), max_depth=3, show_weights=True)
    return "Shallow error-segment tree (1 means high error):\n" + text


def run_train(
    file: str, target: str, model: str = "auto", task: str = "auto", sid: str | None = None
) -> dict:
    df, error = load_csv(file, target)
    if error:
        return error
    sid, error = ensure_session(sid, file, target)
    if error:
        return error
    try:
        policy = core_policy()(df[target], len(df), task=task, model=model)
    except ValueError as exc:
        return {"error": str(exc)}
    selected_task = policy.task
    selected_model = policy.model
    X = df.drop(columns=[target])
    y = df[target]
    counts = y.value_counts()
    stratify = y if selected_task == "classification" and len(counts) > 1 and counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=stratify
    )
    try:
        result = core_train()(
            selected_task, selected_model, X_train, y_train,
            tune=True,
            preset=policy.preset,
            n_trials=policy.n_trials,
            timeout=policy.timeout,
            crossval=policy.crossval,
        )
        metrics = result.evaluate(X_test, y_test)
        top_features = result.get_feature_importance(X_test, y_test)[:20]
        tree_summary = _error_summary(result, X_test, y_test, selected_task)
    except Exception as exc:
        return {"error": f"Training failed: {type(exc).__name__}: {exc}"}
    model_id = f"run_{uuid.uuid4().hex[:12]}"
    model_dir = stratum_home() / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / f"{model_id}.joblib"
    result.save(artifact_path)
    save_artifact(sid, "fitted_model", str(artifact_path), model_id=model_id)
    contract = build_data_contract(df, target, features=list(X.columns))
    save_artifact(
        sid,
        "data_contract",
        str(Path(file).expanduser().resolve()),
        model_id=model_id,
        metadata={"contract": contract},
    )
    top5 = result.summarize_trials()
    direction = (
        str(result.study.direction).lower().split(".")[-1]
        if result.study is not None else ("minimize" if selected_task == "regression" else "maximize")
    )
    metadata = {
        "model_name": selected_model, "task": selected_task, "best_params": result.best_params,
        "top_features": top_features, "decision_tree_summary": tree_summary, "optuna_top5": top5,
        "dataset_path": str(Path(file).expanduser().resolve()), "target_col": target,
        "feature_names": list(X.columns), "feature_dtypes": {c: str(t) for c, t in X.dtypes.items()},
        "preprocessing": result.get_pipeline_spec(), "metric_direction": direction,
        "artifact_path": str(artifact_path),
        "training_policy": policy.to_dict(),
    }
    save_run(sid, model_id, result, metrics, metadata=metadata)
    best = top5[0] if top5 else None
    optuna_summary = (
        f"Best trial #{best['number']} scored {best['value']:.5f} across {len(result.study.trials)} trials."
        if best else "No tuning trial history is available."
    )
    return {
        "model_id": model_id, "sid": sid, "model_name": selected_model, "task": selected_task,
        "metrics": metrics, "top_features": top_features, "best_params": result.best_params,
        "decision_tree_summary": tree_summary, "optuna_summary": optuna_summary,
        "model_path": str(artifact_path), "training_policy": policy.to_dict(),
    }
