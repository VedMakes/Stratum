import pandas as pd
import pytest

from stratum.engineer import evaluate_formula
from stratum.privacy import redact_context, sanitise_for_api


def test_privacy_redacts_pii_names_and_values():
    df = pd.DataFrame({"email": ["a@example.com"], "amount": [10], "target": [1]})
    safe = sanitise_for_api(df, "target")
    assert safe["columns"][0]["name"] == "pii_column_0"
    assert "a@example.com" not in str(safe)
    context = redact_context(
        {"feature": "email", "summary": "Errors split on email > 2"}, ["email"]
    )
    assert "email" not in str(context)


def test_api_context_anonymizes_target_labels_and_identifiers():
    df = pd.DataFrame({
        "customer_id": ["cust-938271", "cust-654321"],
        "email": ["one@example.com", "two@example.com"],
        "outcome": ["churned", "retained"],
    })

    safe = sanitise_for_api(df, "outcome")
    encoded = str(safe)

    assert "cust-938271" not in encoded
    assert "one@example.com" not in encoded
    assert "churned" not in encoded
    assert safe["target"]["value_counts"] == {"class_0": 1, "class_1": 1}
    assert safe["columns"][0]["name"] == "pii_column_0"


def test_redact_context_is_case_insensitive_and_redacts_common_identifiers():
    context = redact_context(
        {
            "customer_id": "cust-938271",
            "summary": "Email ONE@example.com and SSN 123-45-6789 for UUID 123e4567-e89b-12d3-a456-426614174000; CUSTOMER_ID used.",
        },
        ["customer_id"],
    )
    encoded = str(context)

    for secret in ("cust-938271", "ONE@example.com", "123-45-6789", "123e4567-e89b-12d3-a456-426614174000", "CUSTOMER_ID"):
        assert secret not in encoded


def test_explainer_sanitizes_question_and_minimizes_domain_context(monkeypatch):
    import asyncio
    from stratum.tools import tool_explain

    monkeypatch.setattr(tool_explain, "get_run", lambda model_id: {
        "sid": "sid1", "model_name": "random_forest", "task": "classification",
        "metrics": {"accuracy": .8}, "best_params": {}, "top_features": [],
        "optuna_top5": [], "decision_tree_summary": "email=one@example.com",
        "feature_names": ["email", "customer_id"],
    })
    monkeypatch.setattr(tool_explain, "get_resolved_flags", lambda sid: [{
        "flag_type": "domain_note", "column_name": "customer_id", "status": "modified",
        "confidence": 1, "reasoning": "private reasoning", "user_comment": "one@example.com",
    }])
    captured = {}

    async def fake_call(system, user, max_tokens):
        captured["system"] = system
        captured["user"] = user
        return "ok"

    monkeypatch.setattr(tool_explain, "call_llm_async", fake_call)
    result = asyncio.run(tool_explain.run_explain(
        "run1", "Explain email one@example.com for customer_id cust-938271"
    ))

    assert result["answer"] == "ok"
    sent = captured["system"] + captured["user"]
    for secret in ("one@example.com", "cust-938271", "private reasoning", "decision_tree_summary"):
        assert secret not in sent


def test_formula_allowlist():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert evaluate_formula(df, "a / b").tolist() == [1 / 3, .5]
    with pytest.raises(ValueError):
        evaluate_formula(df, "__import__('os').system('id')")
