from __future__ import annotations

import json
from pathlib import Path

from ..common import load_csv
from ..engineer import evaluate_formula
from ..explainer import call_llm
from ..privacy import redact_context, sanitise_for_api
from ..session import ensure_session, get_resolved_flags, get_session_runs, save_artifact


SYSTEM = """You are a tabular feature engineering expert. Return only a JSON array of 3-5 objects with keys name, formula, reasoning, leakage_risk. Formulas may use column names as variables, arithmetic, and np.log1p, np.sqrt, np.abs, np.maximum, np.minimum, or np.clip. Never derive a feature from the target. Reject high leakage risk."""


def run_engineer(
    file: str, target: str, goal: str, sid: str | None = None, apply: bool = False
) -> dict:
    df, error = load_csv(file, target)
    if error:
        return error
    sid, error = ensure_session(sid, file, target, goal)
    if error:
        return error
    safe = sanitise_for_api(df, target)
    runs = get_session_runs(sid)
    prior = runs[-1] if runs else None
    context = redact_context({
        "goal": goal, "dataset": safe, "resolved_flags": get_resolved_flags(sid),
        "prior_run": ({
            "model": prior["model_name"], "metrics": prior["metrics"],
            "top_features": prior["top_features"][:5],
            "decision_tree_summary": prior["decision_tree_summary"],
        } if prior else None),
    }, list(df.columns) + (prior.get("feature_names", []) if prior else []))
    try:
        raw = call_llm(SYSTEM, json.dumps(context, default=str), max_tokens=600)
        proposals = json.loads(raw)
        if not isinstance(proposals, list):
            raise ValueError("LLM response was not a JSON array")
    except Exception as exc:
        return {
            "sid": sid, "features": [], "engineered_path": None,
            "warning": f"AI feature engineering unavailable: {exc}",
            "local_context": {"columns": [item["name"] for item in safe["columns"]]},
        }
    valid = []
    sample = df.drop(columns=[target]).head(5)
    for proposal in proposals:
        try:
            if proposal.get("leakage_risk") == "high":
                continue
            name = str(proposal["name"])
            formula = str(proposal["formula"])
            if name == target or name in df.columns:
                continue
            evaluate_formula(sample, formula)
            valid.append({
                "name": name, "formula": formula,
                "reasoning": str(proposal.get("reasoning", "")),
                "leakage_risk": str(proposal.get("leakage_risk", "low")),
                "code": f"df[{name!r}] = {formula}",
            })
        except (KeyError, SyntaxError, TypeError, ValueError):
            continue
    output = None
    if apply and valid:
        engineered = df.copy()
        features = engineered.drop(columns=[target])
        for item in valid:
            engineered[item["name"]] = evaluate_formula(features, item["formula"])
            features[item["name"]] = engineered[item["name"]]
        source = Path(file).expanduser().resolve()
        output = source.with_name(f"{source.stem}_engineered.csv")
        engineered.to_csv(output, index=False)
        save_artifact(
            sid,
            "engineered_csv",
            str(output),
            metadata={"source": str(source), "features": [item["name"] for item in valid]},
        )
    return {"sid": sid, "features": valid, "engineered_path": str(output) if output else None}
