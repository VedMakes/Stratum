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
)


def _primary_metric(run: dict) -> tuple[str | None, float | None, str]:
    metrics = run.get("metrics") or {}
    if not metrics:
        return None, None, run.get("metric_direction") or "maximize"
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
            scored.append((run, value, direction))
    if not scored:
        return runs[-1] if runs else None
    return sorted(scored, key=lambda item: item[1], reverse=scored[0][2] == "maximize")[0][0]


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
    baseline = runs[0] if runs else None
    best_metric, best_value, best_direction = _primary_metric(best) if best else (None, None, "maximize")
    baseline_metric, baseline_value, _ = (
        _primary_metric(baseline) if baseline else (None, None, "maximize")
    )
    comparable = (
        best_metric is not None
        and best_metric == baseline_metric
        and best_value is not None
        and baseline_value is not None
    )
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
        "recent_journal": get_journal(sid, limit=10),
        "engineered_features": _engineered_features(artifacts),
        "current_metric": best_metric,
        "current_score": best_value,
        "baseline_score": baseline_value if comparable else None,
        "improvement": improvement,
        "artifacts": artifacts,
    })
