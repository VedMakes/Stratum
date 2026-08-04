from __future__ import annotations

from ..common import json_safe
from ..session import get_artifacts
from .tool_status import run_status


def _latest_by_kind(artifacts: list[dict], kinds: set[str]) -> dict:
    latest = {}
    for artifact in artifacts:
        kind = artifact.get("kind")
        if kind in kinds:
            latest[kind] = artifact
    return latest


def run_report(sid: str | None = None, file: str | None = None) -> dict:
    status = run_status(sid=sid, file=file)
    if "error" in status:
        return status
    artifacts = get_artifacts(status["sid"])
    latest = _latest_by_kind(artifacts, {
        "schema_check", "validation", "segment_analysis", "stability",
        "predictions_csv", "export_script",
    })
    diagnostics = {
        kind: {
            "artifact_id": artifact["artifact_id"],
            "model_id": artifact["model_id"],
            "path": artifact["path"],
            "created_at": artifact["created_at"],
            "summary": artifact.get("metadata", {}),
        }
        for kind, artifact in latest.items()
    }
    recommended_next_actions = []
    if "validation" not in latest and status.get("best_model"):
        recommended_next_actions.append("Run stratum_validate on a labeled holdout or recent production sample.")
    if "segment_analysis" not in latest and status.get("best_model"):
        recommended_next_actions.append("Run stratum_segments to find where the model fails.")
    if "stability" not in latest and status.get("best_model"):
        recommended_next_actions.append("Run stratum_stability to estimate fold-to-fold metric variance.")
    if "schema_check" not in latest and status.get("best_model"):
        recommended_next_actions.append("Run stratum_schema_check before scoring new prediction data.")
    if status.get("best_model") and "export_script" not in latest:
        recommended_next_actions.append("Export the best model when you are ready to operationalize it.")

    return json_safe({
        "sid": status["sid"],
        "dataset": status["dataset"],
        "best_model": status["best_model"],
        "current_metric": status["current_metric"],
        "current_score": status["current_score"],
        "baseline_score": status["baseline_score"],
        "improvement": status["improvement"],
        "runs": status["runs"],
        "resolved_flags": status["resolved_flags"],
        "pending_decisions": status.get("pending_decisions", []),
        "recent_journal": status.get("recent_journal", []),
        "engineered_features": status["engineered_features"],
        "diagnostics": diagnostics,
        "recommended_next_actions": recommended_next_actions,
    })
