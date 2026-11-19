import importlib.util
from pathlib import Path

import pandas as pd

from stratum.session import create_decision, create_session, save_run
from stratum.tools.tool_compare import run_compare
from stratum.tools.tool_export import run_export


class Result:
    model_name = "random_forest"
    task = "classification"
    best_params = {"n_estimators": 10}


def _save(sid, model_id, accuracy):
    save_run(sid, model_id, Result(), {"accuracy": accuracy}, metadata={
        "model_name": "random_forest", "task": "classification",
        "best_params": {"n_estimators": 10}, "top_features": [{"name": "x", "importance": 1}],
        "feature_names": ["x"], "feature_dtypes": {"x": "float64"},
        "preprocessing": {"steps": [{"name": "model", "class": "RandomForestClassifier"}]},
        "target_col": "target", "artifact_path": f"/tmp/{model_id}.joblib",
    })


def test_compare_and_export(tmp_path):
    sid = create_session("/tmp/data.csv", "target")
    _save(sid, "a", .7)
    _save(sid, "b", .8)
    assert run_compare("a", "b")["winner"] == "b"
    output = tmp_path / "export.py"
    exported = run_export("b", str(output))
    assert Path(exported["script_path"]).exists()
    assert "stratum" not in exported["script"].lower()
    assert "from MLProj import train" in exported["script"]

    data = tmp_path / "data.csv"
    pd.DataFrame({
        "x": list(range(20)),
        "target": [0] * 10 + [1] * 10,
    }).to_csv(data, index=False)
    spec = importlib.util.spec_from_file_location("generated_export", output)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_path = tmp_path / "retrained.joblib"
    module.main(str(data), str(model_path))
    assert model_path.exists()


def test_export_is_blocked_by_unresolved_domain_decision(tmp_path):
    sid = create_session("/tmp/data.csv", "target")
    _save(sid, "blocked_export", .8)
    create_decision(
        sid,
        "temporal_availability",
        "Was the date available before prediction?",
        "Post-outcome values can leak the target.",
        options=["available_before_prediction", "post_outcome", "unknown"],
        safe_answers=["available_before_prediction"],
        blocking=True,
    )

    result = run_export("blocked_export")

    assert result["recommended_action"] == "ask_user"
    assert result["blocked_decisions"]
