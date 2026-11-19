import json

from stratum.session import create_session, resolve_flag, save_flag, save_run
from stratum.tools import tool_engineer


class Result:
    model_name = "random_forest"
    task = "classification"
    best_params = {}


def test_engineer_includes_minimized_prior_run_context(binary_csv, monkeypatch):
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
    assert '"accuracy": 0.8' in captured["context"]
    assert "decision_tree_summary" not in captured["context"]
    assert "c0" not in captured["context"]


def test_engineer_payload_excludes_raw_labels_identifiers_and_free_text(tmp_path, monkeypatch):
    import pandas as pd

    path = tmp_path / "private.csv"
    pd.DataFrame({
        "customer_id": ["cust-938271", "cust-654321", "cust-123456", "cust-345678"],
        "email": ["one@example.com", "two@example.com", "three@example.com", "four@example.com"],
        "signal": [0, 1, 0, 1],
        "outcome": ["churned", "retained", "churned", "retained"],
    }).to_csv(path, index=False)
    sid = create_session(str(path), "outcome")
    flag_id = save_flag(sid, {
        "flag_type": "domain_note", "column": "customer_id", "confidence": 1,
        "reasoning": "raw reasoning with cust-938271", "status": "pending",
    })
    resolve_flag(flag_id, "modified", "one@example.com must stay local")
    captured = {}

    def fake_call(system, user, max_tokens):
        captured["payload"] = user
        return json.dumps([])

    monkeypatch.setattr(tool_engineer, "call_llm", fake_call)
    result = tool_engineer.run_engineer(
        str(path), "outcome", "Review one@example.com", sid=sid
    )

    assert result["features"] == []
    for secret in (
        "cust-938271", "one@example.com", "churned", "retained",
        "raw reasoning", "must stay local",
    ):
        assert secret not in captured["payload"]
    assert "decision_tree_summary" not in captured["payload"]
