from __future__ import annotations

from pathlib import Path

from ..data.drift import run_drift
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
    model_id: str, file: str, sid: str | None = None, target: str | None = None
) -> dict:
    run = get_run(model_id)
    if not run:
        return {"error": f"Model '{model_id}' not found. Run stratum_train first."}
    if sid and sid != run["sid"]:
        return {"error": f"Model '{model_id}' does not belong to session '{sid}'."}
    reference = _reference_path(run)
    if not reference:
        return {"error": "No readable training/reference file is available for this model."}
    result = run_drift(reference, file, target=target or run.get("target_col"), sid=run["sid"])
    result["model_id"] = model_id
    result["reference_file"] = reference
    return result
