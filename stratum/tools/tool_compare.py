from __future__ import annotations

from numbers import Real

from ..session import get_artifacts, get_run


def _primary(run: dict) -> tuple[str, float, str]:
    metrics = run["metrics"]
    selection = (run.get("training_policy") or {}).get("selection") or {}
    selected = selection.get("metric")
    selected_key = {"f1": "f1_weighted"}.get(selected, selected)
    if selected_key in metrics:
        return (
            selected_key,
            float(metrics[selected_key]),
            selection.get("direction") or run.get("metric_direction") or "maximize",
        )
    if run["task"] == "regression":
        name = "rmse" if "rmse" in metrics else next(iter(metrics))
        return name, float(metrics[name]), "minimize"
    for name in ("f1_weighted", "roc_auc", "accuracy"):
        if name in metrics:
            return name, float(metrics[name]), "maximize"
    name = next(iter(metrics))
    return name, float(metrics[name]), "maximize"


def run_compare(model_id_a: str, model_id_b: str, sid: str | None = None) -> dict:
    a, b = get_run(model_id_a), get_run(model_id_b)
    if not a:
        return {"error": f"Model '{model_id_a}' not found. Run stratum_train first."}
    if not b:
        return {"error": f"Model '{model_id_b}' not found. Run stratum_train first."}
    if sid and (a["sid"] != sid or b["sid"] != sid):
        return {"error": "Both models must belong to the supplied session."}
    metric_a, value_a, direction_a = _primary(a)
    metric_b, value_b, direction_b = _primary(b)
    common_metrics = set(a["metrics"]) & set(b["metrics"])
    deltas = {
        name: float(b["metrics"][name]) - float(a["metrics"][name])
        for name in common_metrics
        if isinstance(a["metrics"][name], Real) and isinstance(b["metrics"][name], Real)
    }
    if metric_a != metric_b or direction_a != direction_b:
        winner = "tie"
        reason = "The runs do not share a comparable primary metric."
    else:
        delta = value_b - value_a
        winner = "tie" if abs(delta) < 1e-12 else (
            "b" if (delta > 0) == (direction_a == "maximize") else "a"
        )
        reason = f"Run {winner.upper()} wins on {metric_a} by {abs(delta):.4f}." if winner != "tie" else f"The runs tie on {metric_a}."
    rank_a = {item["name"]: i + 1 for i, item in enumerate(a["top_features"])}
    rank_b = {item["name"]: i + 1 for i, item in enumerate(b["top_features"])}
    changes = [
        {"feature": name, "rank_a": rank_a.get(name), "rank_b": rank_b.get(name),
         "delta": (rank_a.get(name, len(rank_a) + 1) - rank_b.get(name, len(rank_b) + 1))}
        for name in set(rank_a) | set(rank_b)
        if abs(rank_a.get(name, len(rank_a) + 1) - rank_b.get(name, len(rank_b) + 1)) >= 2
    ]
    params = {
        key: {"a": a["best_params"].get(key), "b": b["best_params"].get(key)}
        for key in set(a["best_params"]) | set(b["best_params"])
        if a["best_params"].get(key) != b["best_params"].get(key)
    }
    narrative = f"{reason} {len(changes)} feature ranks moved materially, and {len(params)} tuned parameters differ."
    stability_by_model = {}
    for artifact in get_artifacts(a["sid"], "stability"):
        if artifact.get("model_id") in {model_id_a, model_id_b}:
            stability_by_model[artifact["model_id"]] = artifact.get("metadata", {})

    def tradeoff(run, model_id):
        operational = (run.get("evaluation") or {}).get("operational", {})
        stability = stability_by_model.get(model_id)
        return {
            "selection": (run.get("training_policy") or {}).get("selection"),
            "training_wall_seconds": operational.get("training_wall_seconds"),
            "inference_timing": operational.get("inference_timing"),
            "feature_count": len(run.get("feature_names") or []),
            "top_features": (run.get("top_features") or [])[:5],
            "stability": {
                key: stability.get(key)
                for key in ("metric", "mean", "std", "relative_std", "stability", "protocol")
            } if stability else None,
        }
    return {
        "sid": a["sid"], "winner": winner, "metric_delta": deltas,
        "feature_rank_changes": sorted(changes, key=lambda x: abs(x["delta"]), reverse=True),
        "param_differences": params, "narrative": narrative,
        "tradeoffs": {
            "a": tradeoff(a, model_id_a),
            "b": tradeoff(b, model_id_b),
            "note": "Predictive winner uses the primary holdout metric only. Runtime and feature summaries are context, not a combined score; latency is hardware/load dependent and monetary cost is not estimated.",
        },
    }
