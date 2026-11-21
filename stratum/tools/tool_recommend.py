from __future__ import annotations

from ..common import json_safe
from ..session import log_event, save_artifact
from .tool_status import run_status


def _model_artifacts(status: dict, kind: str, model_id: str) -> list[dict]:
    return [
        artifact for artifact in status.get("artifacts", [])
        if artifact.get("kind") == kind and artifact.get("model_id") == model_id
    ]


def _recommendation(
    recommendation_id: str,
    title: str,
    next_tool: str,
    rationale: str,
    trade_off: str,
    would_change: str,
    evidence: list[dict],
    *,
    requires_user_choice: bool = False,
) -> dict:
    return {
        "recommendation_id": recommendation_id,
        "title": title,
        "next_tool": next_tool,
        "rationale": rationale,
        "trade_off": trade_off,
        "what_would_change_this_recommendation": would_change,
        "evidence": evidence,
        "requires_user_choice": requires_user_choice,
    }


def run_recommend(sid: str | None = None, file: str | None = None) -> dict:
    status = run_status(sid=sid, file=file)
    if "error" in status:
        return status

    model_id = status.get("best_model")
    recommendations = []
    if status.get("blocking_decisions"):
        blockers = status["blocking_decisions"]
        recommendations.append(_recommendation(
            "resolve_blocking_decisions",
            "Resolve the blocking domain questions before further modeling",
            "stratum_decisions",
            "The session contains unresolved or unsafe decisions that block protected operations.",
            "Waiting for domain input delays experiments but avoids silently assuming target or feature semantics.",
            "A user answer that exactly matches a declared safe answer clears the relevant gate; unknown or unsafe answers do not.",
            [{"source": "decision", "id": item["decision_id"], "type": item["decision_type"]} for item in blockers],
            requires_user_choice=True,
        ))
    elif not model_id:
        return json_safe({
            "sid": status["sid"],
            "best_model": None,
            "recommendations": [],
            "reason": "No saved model run is available to ground an experiment recommendation.",
            "recommendations_are_not_decisions": True,
        })
    else:
        best = next(run for run in status["runs"] if run["model_id"] == model_id)
        comparison = best.get("baseline_comparison")
        run_evidence = [{
            "source": "run",
            "id": model_id,
            "field": "baseline_comparison",
            "value": comparison,
        }] if comparison else [{
            "source": "run", "id": model_id, "field": "metrics", "value": best["metrics"],
        }]
        warnings = best.get("evaluation_warnings") or []
        small_holdout = any("only" in warning.casefold() and "rows" in warning.casefold() for warning in warnings)
        validations = _model_artifacts(status, "validation", model_id)
        independent_validations = [
            item for item in validations
            if (item.get("metadata") or {}).get("source_sha256")
            and (item.get("metadata") or {}).get("source_sha256") != best.get("dataset_sha256")
        ]
        adequate_validation = [
            item for item in independent_validations
            if (item.get("metadata") or {}).get("rows_evaluated", 0) >= 30
        ]
        if not adequate_validation:
            rationale = (
                "The saved evaluation warns that its holdout is small. An independent labeled sample is the strongest next check."
                if small_holdout else
                "No independent labeled validation artifact with at least 30 rows is saved for the current best model."
            )
            recommendations.append(_recommendation(
                "independent_validation",
                "Validate on a representative, independently labeled dataset",
                "stratum_validate",
                rationale,
                "Requires access to labeled data that represents the intended use; it costs time but is more informative than reusing the training holdout.",
                "A representative sample with adequate support and acceptable metrics would reduce the need for immediate model changes; a weak result or distribution mismatch would redirect investigation.",
                run_evidence + ([{
                    "source": "run", "id": model_id,
                    "field": "evaluation_warnings", "value": warnings,
                }] if warnings else []),
            ))

        stability = _model_artifacts(status, "stability", model_id)
        strategy = best.get("split_strategy")
        if not stability and strategy != "predefined":
            recommendations.append(_recommendation(
                "estimate_metric_stability",
                "Estimate fold-to-fold metric spread",
                "stratum_stability",
                "The session has no saved stability analysis for the best run.",
                "Repeated CV adds computation and measures sensitivity within this dataset; it is not an independent validation set or a confidence interval.",
                "Large fold spread would argue against fine-grained model ranking and favor more representative data or a simpler protocol; consistently narrow spread would make the selected metric more repeatable on this dataset.",
                [{"source": "run", "id": model_id, "field": "selection", "value": best.get("selection")},
                 {"source": "run", "id": model_id, "field": "split_strategy", "value": strategy}],
            ))

        if adequate_validation and not _model_artifacts(status, "segment_analysis", model_id):
            validation = adequate_validation[-1]
            validation_meta = validation.get("metadata") or {}
            if validation_meta.get("rows_evaluated", 0) >= 30:
                recommendations.append(_recommendation(
                    "inspect_validation_errors",
                    "Find supported slices with elevated validation error",
                    "stratum_segments",
                    "A labeled validation artifact is available, but no error-segment analysis is saved for this model.",
                    "Segment findings are descriptive and may be noisy across many candidate slices; inspect support and confirm slices before changing features.",
                    "A supported, actionable error pattern may justify a user-approved feature or data-collection experiment; no stable pattern favors gathering more evidence before changing the model.",
                    [{"source": "artifact", "id": validation["artifact_id"], "kind": "validation",
                      "rows_evaluated": validation_meta["rows_evaluated"]},
                     {"source": "run", "id": model_id, "field": "metrics", "value": best["metrics"]}],
                    requires_user_choice=True,
                ))

    dataset_path = ""
    if model_id:
        best = next(run for run in status["runs"] if run["model_id"] == model_id)
        dataset_path = best.get("dataset_path") or ""
    if not dataset_path:
        contracts = [a for a in status.get("artifacts", []) if a.get("kind") == "data_contract"]
        if contracts:
            dataset_path = contracts[-1].get("path") or ""
    result = {
        "sid": status["sid"],
        "best_model": model_id,
        "recommendations": recommendations[:3],
        "recommendations_are_not_decisions": True,
        "note": "Recommendations summarize saved evidence; they do not execute tools, change a protocol, or represent user approval.",
    }
    save_artifact(
        status["sid"], "experiment_recommendations", dataset_path,
        model_id=model_id,
        metadata={
            "recommendations": result["recommendations"],
            "recommendations_are_not_decisions": True,
        },
    )
    log_event(
        status["sid"], "tool", "recommendations_generated",
        "Generated evidence-linked experiment recommendations",
        {"model_id": model_id,
         "recommendation_ids": [item["recommendation_id"] for item in recommendations[:3]]},
    )
    return json_safe(result)
