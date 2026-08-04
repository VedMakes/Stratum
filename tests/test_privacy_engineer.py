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


def test_formula_allowlist():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert evaluate_formula(df, "a / b").tolist() == [1 / 3, .5]
    with pytest.raises(ValueError):
        evaluate_formula(df, "__import__('os').system('id')")
