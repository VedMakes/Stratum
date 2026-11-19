import pandas as pd

from stratum.session import create_session
from stratum.tools.tool_decisions import (
    run_create_decision,
    run_decisions,
    run_journal,
    run_next_action,
    run_resolve_decision,
)
from stratum.tools.tool_leakage_deep import run_leakage_deep
from stratum.tools.tool_report import run_report
from stratum.tools.tool_status import run_status


def test_decision_lifecycle_blocks_and_records_journal(tmp_path):
    sid = create_session(str(tmp_path / "data.csv"), "target")
    created = run_create_decision(
        sid,
        "metric_preference",
        "Should this model optimize recall or precision?",
        "The choice changes thresholding and model selection.",
        options=["recall", "precision", "f1"],
        safe_answers=["recall", "precision", "f1"],
        recommended="recall",
        blocking=True,
    )
    decision_id = created["decision_id"]

    pending = run_decisions(sid=sid)
    gate = run_next_action(sid=sid)

    assert pending["decisions"][0]["decision_id"] == decision_id
    assert gate["recommended_action"] == "ask_user"
    assert "train" in gate["blocked_actions"]

    resolved = run_resolve_decision(decision_id, "recall", "Missed churn is costly.")
    journal = run_journal(sid=sid)

    assert resolved["decision"]["status"] == "resolved"
    assert resolved["decision"]["safe_to_proceed"] is True
    assert run_next_action(sid=sid)["recommended_action"] == "continue"
    assert journal["journal"][0]["event_type"] == "decision_resolved"


def test_status_and_report_include_decision_context(tmp_path):
    sid = create_session(str(tmp_path / "data.csv"), "target")
    run_create_decision(
        sid,
        "temporal_availability",
        "Was invoice_paid_date available before prediction?",
        "Post-outcome dates leak the target.",
        options=["available_before_prediction", "post_outcome", "unknown"],
        safe_answers=["available_before_prediction"],
        column="invoice_paid_date",
        blocking=True,
    )

    status = run_status(sid=sid)
    report = run_report(sid=sid)

    assert status["pending_decisions"][0]["column_name"] == "invoice_paid_date"
    assert report["pending_decisions"][0]["decision_type"] == "temporal_availability"
    assert report["recent_journal"]


def test_blocking_decision_requires_explicit_safe_answers(tmp_path):
    sid = create_session(str(tmp_path / "data.csv"), "target")
    result = run_create_decision(
        sid,
        "temporal_availability",
        "Was the feature available before prediction?",
        "Post-outcome values can leak the target.",
        blocking=True,
    )
    assert "safe_answer" in result["error"]


def test_unknown_resolution_does_not_clear_blocking_decision(tmp_path):
    sid = create_session(str(tmp_path / "data.csv"), "target")
    created = run_create_decision(
        sid,
        "temporal_availability",
        "Was the feature available before prediction?",
        "Post-outcome values can leak the target.",
        options=["available_before_prediction", "post_outcome", "unknown"],
        safe_answers=["available_before_prediction"],
        blocking=True,
    )

    resolved = run_resolve_decision(created["decision_id"], "unknown")

    assert resolved["decision"]["status"] == "resolved"
    assert resolved["decision"]["safe_to_proceed"] is False
    assert resolved["decision"]["resolution_state"] == "unknown"
    next_action = run_next_action(sid=sid)
    assert next_action["recommended_action"] == "ask_user"
    assert next_action["decisions"][0]["answer"] == "unknown"


def test_unknown_cannot_be_declared_a_safe_answer(tmp_path):
    sid = create_session(str(tmp_path / "data.csv"), "target")
    result = run_create_decision(
        sid,
        "temporal_availability",
        "Was the feature available before prediction?",
        "Post-outcome values can leak the target.",
        options=["available_before_prediction", "unknown"],
        safe_answers=["unknown"],
        blocking=True,
    )
    assert "cannot be declared safe" in result["error"]


def test_leakage_deep_creates_temporal_decision(tmp_path):
    df = pd.DataFrame({
        "invoice_paid_date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
        "feature": [1, 2, 3, 4],
        "target": [0, 1, 0, 1],
    })
    path = tmp_path / "leakage.csv"
    df.to_csv(path, index=False)
    sid = create_session(str(path), "target")

    result = run_leakage_deep(str(path), "target", sid=sid)
    decisions = run_decisions(sid=sid)

    assert result["sid"] == sid
    assert any(flag.get("decision_id") for flag in result["flags"])
    assert decisions["decisions"][0]["decision_type"] == "temporal_availability"
