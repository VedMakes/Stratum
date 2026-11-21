from __future__ import annotations

from pathlib import Path

from ..data.drift import run_drift, run_drift_profile
from ..session import get_artifacts, get_run


def _reference_path(run: dict) -> str | None:
    path = run.get("dataset_path")
    if path and Path(path).is_file():
        return path
    for artifact in reversed(get_artifacts(run["sid"], "data_contract")):
        if artifact.get("model_id") in {None, run["model_id"]} and Path(artifact["path"]).is_file():
            return artifact["path"]
    return None


def run_drift_current(
    model_id: str,
    file: str,
    sid: str | None = None,
    target: str | None = None,
    psi_mild_threshold: float = .1,
    psi_significant_threshold: float = .2,
    psi_high_threshold: float = .5,
    p_value_threshold: float = .05,
) -> dict:
    run = get_run(model_id)
    if not run:
        return {"error": f"Model '{model_id}' not found. Run stratum_train first."}
    if sid and sid != run["sid"]:
        return {"error": f"Model '{model_id}' does not belong to session '{sid}'."}
    profile = ((run.get("provenance") or {}).get("dataset") or {}).get("drift_reference")
    if profile and profile.get("features"):
        result = run_drift_profile(
            profile,
            file,
            sid=run["sid"],
            model_id=model_id,
            psi_mild_threshold=psi_mild_threshold,
            psi_significant_threshold=psi_significant_threshold,
            psi_high_threshold=psi_high_threshold,
            p_value_threshold=p_value_threshold,
        )
    else:
        reference = _reference_path(run)
        if not reference:
            return {"error": "No saved training reference profile or readable legacy reference file is available."}
        result = run_drift(
            reference,
            file,
            target=target or run.get("target_col"),
            sid=run["sid"],
            psi_mild_threshold=psi_mild_threshold,
            psi_significant_threshold=psi_significant_threshold,
            psi_high_threshold=psi_high_threshold,
            p_value_threshold=p_value_threshold,
        )
        result.setdefault("warnings", []).append(
            "This legacy run has no immutable training-time feature summary; the current reference file may have changed since training."
        )
        result["model_id"] = model_id
        result["reference_file"] = reference
    return result
