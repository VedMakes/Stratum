from __future__ import annotations

import json

from ..explainer import call_llm_async
from ..privacy import redact_context, safe_flag_context
from ..session import get_resolved_flags, get_run


async def run_explain(model_id: str, question: str, sid: str | None = None) -> dict:
    run = get_run(model_id)
    if not run:
        return {"error": f"Model '{model_id}' not found. Run stratum_train first."}
    if sid and sid != run["sid"]:
        return {"error": f"Model '{model_id}' does not belong to session '{sid}'."}
    context = redact_context({
        "model": run["model_name"], "task": run["task"], "metrics": run["metrics"],
        "best_params": run["best_params"], "top_features": run["top_features"][:5],
        "optuna_top5": run["optuna_top5"],
        "domain_knowledge": safe_flag_context(
            get_resolved_flags(run["sid"]), run.get("feature_names") or []
        ),
    }, run.get("feature_names") or [])
    safe_question = redact_context(
        {"question": question}, run.get("feature_names") or []
    )["question"]
    system = (
        "Explain this model to its creator. Answer only the question, use the supplied "
        "run evidence, and avoid generic advice.\n" + json.dumps(context, default=str)
    )
    try:
        answer = await call_llm_async(system, safe_question, max_tokens=600)
        return {"sid": run["sid"], "model_id": model_id, "answer": answer}
    except Exception as exc:
        return {
            "sid": run["sid"], "model_id": model_id,
            "warning": f"AI explanation unavailable: {exc}", "metrics": run["metrics"],
            "top_features": run["top_features"][:5],
        }
