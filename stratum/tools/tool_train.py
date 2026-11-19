from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from ..common import load_csv, stratum_home
from ..core import core_policy, core_train
from ..data.contracts import build_data_contract
from ..data.splitter import file_sha256, frame_fingerprint
from .eval_common import evaluate_predictions
from ..session import (
    ensure_session,
    get_artifacts,
    get_session_by_file,
    get_workflow_blockers,
    save_artifact,
    save_run,
)


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


def _find_split_manifest(sid: str, split_id: str) -> dict | None:
    for artifact in get_artifacts(sid, kind="split_manifest"):
        manifest = artifact.get("metadata") or {}
        if manifest.get("split_id") == split_id:
            return manifest
    return None


def _load_saved_split(file: str, target: str, sid: str, split_id: str):
    manifest = _find_split_manifest(sid, split_id)
    if not manifest:
        return None, None, None, {"error": f"Split '{split_id}' was not found in session '{sid}'."}
    if manifest.get("target") != target:
        return None, None, None, {"error": f"Split '{split_id}' was created for a different target."}
    try:
        if file_sha256(file) != manifest.get("source_sha256"):
            return None, None, None, {"error": "Source CSV changed after the split was created; create a new split."}
        train_path = manifest["train_path"]
        test_path = manifest["test_path"]
        if file_sha256(train_path) != manifest.get("train_sha256"):
            return None, None, None, {"error": "Saved training split changed after creation; create a new split."}
        if file_sha256(test_path) != manifest.get("test_sha256"):
            return None, None, None, {"error": "Saved test split changed after creation; create a new split."}
    except (KeyError, OSError) as exc:
        return None, None, None, {"error": f"Saved split files are unavailable: {exc}"}

    train_df, error = load_csv(train_path, target)
    if error:
        return None, None, None, {"error": f"Unable to load saved training split: {error['error']}"}
    test_df, error = load_csv(test_path, target)
    if error:
        return None, None, None, {"error": f"Unable to load saved test split: {error['error']}"}
    excluded_feature = manifest.get("excluded_feature")
    train_features = [
        col for col in train_df.columns if col != target and col != excluded_feature
    ]
    test_features = [
        col for col in test_df.columns if col != target and col != excluded_feature
    ]
    if train_features != test_features:
        return None, None, None, {"error": "Saved train and test splits have different feature schemas."}
    if not train_features:
        return None, None, None, {"error": "Saved split has no usable model features."}
    if train_df[target].isna().any() or test_df[target].isna().any():
        return None, None, None, {"error": "Training and test splits must not contain null target values."}
    return train_df, test_df, manifest, None


def _baseline_evaluation(task: str, y_train, y_test, model_metrics: dict) -> dict:
    dummy = DummyClassifier(strategy="prior") if task == "classification" else DummyRegressor(strategy="mean")
    dummy.fit(np.zeros((len(y_train), 1)), np.asarray(y_train))
    baseline_predictions = dummy.predict(np.zeros((len(y_test), 1)))
    baseline_scores = dummy.predict_proba(np.zeros((len(y_test), 1))) if task == "classification" else None
    baseline_metrics = evaluate_predictions(
        task, y_test, baseline_predictions, baseline_scores,
        class_labels=getattr(dummy, "classes_", None),
    )
    metric = "f1_weighted" if task == "classification" else "rmse"
    if metric not in model_metrics or metric not in baseline_metrics:
        metric = "accuracy" if task == "classification" else "mae"
    direction = "maximize" if task == "classification" else "minimize"
    model_score = float(model_metrics[metric])
    baseline_score = float(baseline_metrics[metric])
    raw_delta = model_score - baseline_score
    improvement = raw_delta if direction == "maximize" else -raw_delta

    warnings = []
    if len(y_test) < 30:
        warnings.append(
            f"Holdout has only {len(y_test)} rows; evaluation metrics may have high uncertainty."
        )
    if task == "classification":
        counts = pd.Series(y_test).value_counts()
        if len(counts) < 2:
            warnings.append("Holdout contains only one target class; class-separation metrics are unavailable.")
        elif int(counts.min()) < 5:
            warnings.append("At least one holdout class has fewer than 5 rows; per-class metrics are unstable.")
        train_classes = set(pd.Series(y_train).dropna().unique())
        test_classes = set(pd.Series(y_test).dropna().unique())
        if train_classes - test_classes:
            warnings.append("Holdout is missing target class(es) present in training.")
        if test_classes - train_classes:
            warnings.append("Holdout contains target class(es) absent from training; they cannot be learned.")
        train_counts = pd.Series(y_train).value_counts()
        if len(train_counts) > 1 and float(train_counts.min() / train_counts.max()) < .1:
            warnings.append("Training classes are highly imbalanced; inspect per-class recall and PR-AUC.")

    return {
        "baseline": {
            "name": "prior_dummy" if task == "classification" else "mean_dummy",
            "metrics": baseline_metrics,
        },
        "comparison": {
            "metric": metric,
            "direction": direction,
            "model_score": model_score,
            "baseline_score": baseline_score,
            "improvement": improvement,
            "model_beats_baseline": improvement > 0,
        },
        "warnings": warnings,
    }


def run_train(
    file: str,
    target: str,
    model: str = "auto",
    task: str = "auto",
    sid: str | None = None,
    split_id: str | None = None,
) -> dict:
    df, error = load_csv(file, target)
    if error:
        return error
    if split_id and not sid:
        existing_session = get_session_by_file(file)
        if not existing_session:
            return {"error": "Pass the sid returned by stratum_split when training with split_id."}
        sid = existing_session["sid"]
    sid, error = ensure_session(sid, file, target)
    if error:
        return error
    blockers = get_workflow_blockers(sid)
    if blockers:
        return {
            "error": "Training is blocked by unresolved domain decisions.",
            "sid": sid,
            "recommended_action": "ask_user",
            "blocked_decisions": blockers,
        }
    split_metadata = None
    train_df = test_df = None
    if split_id:
        train_df, test_df, split_metadata, error = _load_saved_split(
            file, target, sid, split_id
        )
        if error:
            return error
    try:
        initial_policy = core_policy()(df[target], len(df), task=task, model=model)
    except ValueError as exc:
        return {"error": str(exc)}
    selected_task = initial_policy.task
    selected_model = initial_policy.model

    if split_id:
        feature_columns = [
            col for col in train_df.columns
            if col != target and col != split_metadata.get("excluded_feature")
        ]
        X_train, X_test = train_df[feature_columns].copy(), test_df[feature_columns].copy()
        y_train, y_test = train_df[target], test_df[target]
        policy_target = y_train
        policy_rows = len(train_df)
        try:
            policy = core_policy()(
                policy_target, policy_rows, task=selected_task, model=selected_model
            )
        except ValueError as exc:
            return {"error": str(exc)}
    else:
        X = df.drop(columns=[target])
        y = df[target]
        counts = y.value_counts()
        stratify = (
            y if selected_task == "classification" and len(counts) > 1 and counts.min() >= 2
            else None
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, random_state=42, stratify=stratify
        )
        policy_target = y_train
        policy_rows = len(X_train)
        split_metadata = {
            "split_id": None,
            "strategy": "stratified" if stratify is not None else "random",
            "parameters": {"test_size": 0.2, "seed": 42, "stratify": stratify is not None},
            "source_path": str(Path(file).expanduser().resolve()),
            "source_sha256": file_sha256(file),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_row_fingerprint": frame_fingerprint(pd.concat([X_train, y_train], axis=1)),
            "test_row_fingerprint": frame_fingerprint(pd.concat([X_test, y_test], axis=1)),
            "excluded_feature": None,
        }
        try:
            policy = core_policy()(
                policy_target, policy_rows, task=selected_task, model=selected_model
            )
        except ValueError as exc:
            return {"error": str(exc)}
    selected_task = policy.task
    selected_model = policy.model
    feature_columns = list(X_train.columns)
    if selected_task == "classification" and y_train.nunique(dropna=True) < 2:
        return {"error": "Training split must contain at least two target classes."}
    cv_splitter = None
    cv_groups = None
    if split_metadata["strategy"] == "group":
        group_col = split_metadata.get("parameters", {}).get("group_col")
        groups = train_df[group_col]
        n_splits = min(5, int(groups.nunique()))
        if n_splits < 2:
            return {"error": "Group-aware tuning requires at least two training groups."}
        cv_splitter = GroupKFold(n_splits=n_splits)
        cv_groups = groups.to_numpy()
        split_metadata["cv_strategy"] = "group_kfold"
        split_metadata["cv_splits"] = n_splits
    elif split_metadata["strategy"] == "temporal":
        n_splits = min(5, len(X_train) - 1)
        if n_splits < 2:
            return {"error": "Time-aware tuning requires at least three training rows."}
        cv_splitter = TimeSeriesSplit(n_splits=n_splits)
        split_metadata["cv_strategy"] = "time_series"
        split_metadata["cv_splits"] = n_splits
    try:
        result = core_train()(
            selected_task, selected_model, X_train, y_train,
            tune=True,
            preset=policy.preset,
            n_trials=policy.n_trials,
            timeout=policy.timeout,
            crossval=policy.crossval,
            cv_splitter=cv_splitter,
            cv_groups=cv_groups,
        )
        metrics = result.evaluate(X_test, y_test)
        evaluation = _baseline_evaluation(selected_task, y_train, y_test, metrics)
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
    contract = build_data_contract(df, target, features=feature_columns)
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
        "feature_names": feature_columns,
        "feature_dtypes": {c: str(t) for c, t in X_train.dtypes.items()},
        "preprocessing": result.get_pipeline_spec(), "metric_direction": direction,
        "artifact_path": str(artifact_path),
        "training_policy": policy.to_dict(),
        "split_metadata": split_metadata,
        "evaluation": evaluation,
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
        "split_metadata": split_metadata,
        "evaluation": evaluation,
    }
