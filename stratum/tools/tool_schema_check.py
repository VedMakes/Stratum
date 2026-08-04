from __future__ import annotations

from ..common import json_safe, load_csv
from ..data.contracts import build_data_contract, compare_to_contract
from ..session import get_artifacts, get_run, save_artifact


def _latest_contract(sid: str, model_id: str | None = None) -> dict | None:
    artifacts = get_artifacts(sid, "data_contract")
    if model_id:
        artifacts = [item for item in artifacts if item.get("model_id") == model_id] or artifacts
    for artifact in reversed(artifacts):
        contract = artifact.get("metadata", {}).get("contract")
        if contract:
            return contract
    return None


def run_schema_check(
    file: str,
    model_id: str | None = None,
    sid: str | None = None,
    reference_file: str | None = None,
    target: str | None = None,
    prediction_data: bool = True,
) -> dict:
    run = get_run(model_id) if model_id else None
    if model_id and not run:
        return {"error": f"Model '{model_id}' not found. Run stratum_train first."}
    if sid and run and sid != run["sid"]:
        return {"error": f"Model '{model_id}' does not belong to session '{sid}'."}

    contract = None
    contract_source = None
    if reference_file:
        ref, error = load_csv(reference_file, target)
        if error:
            return error
        contract = build_data_contract(ref, target)
        contract_source = reference_file
    elif run:
        contract = _latest_contract(run["sid"], model_id)
        if contract is None:
            contract = {
                "target": run.get("target_col"),
                "features": list(run.get("feature_names") or []),
                "columns": {
                    name: {"dtype": dtype, "null_rate": 0.0, "n_unique": None}
                    for name, dtype in (run.get("feature_dtypes") or {}).items()
                },
            }
        contract_source = model_id
    elif sid:
        contract = _latest_contract(sid)
        contract_source = sid
    else:
        return {"error": "Provide model_id, sid, or reference_file to choose a schema contract."}

    df, error = load_csv(file)
    if error:
        return error
    result = compare_to_contract(
        contract,
        df,
        target=target or contract.get("target"),
        prediction_data=prediction_data,
    )
    resolved_sid = run["sid"] if run else sid
    if resolved_sid:
        save_artifact(
            resolved_sid,
            "schema_check",
            file,
            model_id=model_id,
            metadata={**result, "contract_source": contract_source},
        )
    return json_safe({
        "sid": resolved_sid,
        "model_id": model_id,
        "contract_source": contract_source,
        "rows_checked": len(df),
        **result,
    })
