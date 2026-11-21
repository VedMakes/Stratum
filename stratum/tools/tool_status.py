from __future__ import annotations

from pathlib import Path

from ..common import json_safe
from ..session import (
    get_artifacts,
    get_decisions,
    get_journal,
    get_resolved_flags,
    get_session,
    get_session_by_file,
    get_session_runs,
    get_workflow_blockers,
)


def _primary_metric(run: dict) -> tuple[str | None, float | None, str]:
    metrics = run.get("metrics") or {}
    if not metrics:
        return None, None, run.get("metric_direction") or "maximize"
    selection = (run.get("training_policy") or {}).get("selection") or {}
    selected = selection.get("metric")
    selected_key = {"f1": "f1_weighted"}.get(selected, selected)
    if selected_key in metrics:
        return (
            selected_key,
            float(metrics[selected_key]),
            selection.get("direction") or run.get("metric_direction") or "maximize",
        )
    if run.get("task") == "regression":
        for name in ("rmse", "mae", "mse", "r2"):
            if name in metrics:
                direction = "maximize" if name == "r2" else "minimize"
                return name, float(metrics[name]), direction
    for name in ("f1_weighted", "f1", "roc_auc", "accuracy"):
        if name in metrics:
            return name, float(metrics[name]), "maximize"
    name = next(iter(metrics))
    return name, float(metrics[name]), run.get("metric_direction") or "maximize"


def _best_run(runs: list[dict]) -> dict | None:
    scored = []
    for run in runs:
        metric, value, direction = _primary_metric(run)
        if metric is not None and value is not None:
            scored.append((run, metric, value, direction))
    if not scored:
        return runs[-1] if runs else None
    latest_metric, _, latest_direction = _primary_metric(scored[-1][0])
    current_objective = (latest_metric, latest_direction)
    comparable = [
        (run, value, direction)
        for run, metric, value, direction in scored
        if (metric, direction) == current_objective
    ]
    if not comparable:
        comparable = [(scored[-1][0], scored[-1][2], scored[-1][3])]
    return sorted(
        comparable, key=lambda item: item[1], reverse=current_objective[1] == "maximize"
    )[0][0]


def _engineered_features(artifacts: list[dict]) -> list[str]:
    names = []
    for artifact in artifacts:
        if artifact.get("kind") != "engineered_csv":
            continue
        for name in artifact.get("metadata", {}).get("features", []):
            if name not in names:
                names.append(str(name))
    return names


def run_status(sid: str | None = None, file: str | None = None) -> dict:
    session = get_session(sid) if sid else None
    if session is None and file:
        session = get_session_by_file(file)
    if session is None:
        return {"error": "Session not found. Provide sid or a file that has an existing session."}

    sid = session["sid"]
    runs = get_session_runs(sid)
    artifacts = get_artifacts(sid)
    best = _best_run(runs)
    best_metric, best_value, best_direction = _primary_metric(best) if best else (None, None, "maximize")
    evaluation = (best or {}).get("evaluation") or {}
    baseline_metrics = (evaluation.get("baseline") or {}).get("metrics") or {}
    baseline_value = baseline_metrics.get(best_metric) if best_metric else None
    baseline_source = "evaluation_baseline" if baseline_value is not None else None

    if baseline_value is None:
        prior_runs = [run for run in runs if run.get("model_id") != (best or {}).get("model_id")]
        baseline = prior_runs[0] if prior_runs else None
        baseline_metric, baseline_value, _ = (
            _primary_metric(baseline) if baseline else (None, None, "maximize")
        )
        if baseline_metric != best_metric:
            baseline_value = None
        elif baseline is not None:
            baseline_source = "prior_run"
    comparable = best_metric is not None and best_value is not None and baseline_value is not None
    improvement = None
    if comparable:
        delta = best_value - baseline_value
        improvement = delta if best_direction == "maximize" else -delta

    compact_runs = []
    for run in runs:
        metric, value, _ = _primary_metric(run)
        compact = {
            "model_id": run["model_id"],
            "model": run["model_name"],
            "metric": metric,
            "score": value,
            "date": run["created_at"],
            "split_id": (run.get("split_metadata") or {}).get("split_id"),
            "split_strategy": (run.get("split_metadata") or {}).get("strategy"),
            "baseline_comparison": (run.get("evaluation") or {}).get("comparison"),
            "evaluation_warnings": (run.get("evaluation") or {}).get("warnings", []),
            "selection": (run.get("training_policy") or {}).get("selection"),
            "operational": (run.get("evaluation") or {}).get("operational"),
            "provenance": run.get("provenance") or {},
            "metrics": run.get("metrics") or {},
            "top_features": run.get("top_features") or [],
            "decision_tree_summary": run.get("decision_tree_summary") or "",
            "preprocessing": run.get("preprocessing") or {},
        }
        if metric:
            compact[metric] = value
        compact_runs.append(compact)

    return json_safe({
        "sid": sid,
        "dataset": Path(session["dataset_path"]).name if session.get("dataset_path") else "",
        "runs": compact_runs,
        "best_model": best["model_id"] if best else None,
        "resolved_flags": get_resolved_flags(sid),
        "pending_decisions": get_decisions(sid, status="pending"),
        "blocking_decisions": get_workflow_blockers(sid),
        "recent_journal": get_journal(sid, limit=10),
        "engineered_features": _engineered_features(artifacts),
        "current_metric": best_metric,
        "current_score": best_value,
        "baseline_score": baseline_value if comparable else None,
        "improvement": improvement,
        "baseline_source": baseline_source if comparable else None,
        "artifacts": artifacts,
    })
