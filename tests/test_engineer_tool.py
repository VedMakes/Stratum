import json

from stratum.session import create_session, save_run
from stratum.tools import tool_engineer


class Result:
    model_name = "random_forest"
    task = "classification"
    best_params = {}


def test_engineer_includes_prior_error_context(binary_csv, monkeypatch):
    sid = create_session(str(binary_csv), "target", "predict target")
    save_run(sid, "run_1", Result(), {"accuracy": .8}, metadata={
        "decision_tree_summary": "Errors concentrate when age is low.",
        "top_features": [{"name": "age", "importance": .5}],
    })
    captured = {}

    def fake_call(system, user, max_tokens):
        captured["context"] = user
        return json.dumps([{
            "name": "spend_per_age",
            "formula": "spend / age",
            "reasoning": "Normalizes spend by age.",
            "leakage_risk": "low",
        }])

    monkeypatch.setattr(tool_engineer, "call_llm", fake_call)
    result = tool_engineer.run_engineer(
        str(binary_csv), "target", "predict target", sid=sid
    )
    assert result["features"][0]["name"] == "spend_per_age"
    assert "Errors concentrate when age is low." in captured["context"]
    assert "c0" not in captured["context"]
