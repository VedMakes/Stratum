from __future__ import annotations

from ..common import json_safe
from ..session import (
    create_decision,
    get_decisions,
    get_journal,
    get_session,
    get_session_by_file,
    log_event,
    resolve_decision,
)


def _resolve_sid(sid: str | None = None, file: str | None = None) -> tuple[str | None, dict | None]:
    if sid:
        return (sid, None) if get_session(sid) else (None, {"error": f"Session '{sid}' not found."})
    if file:
        session = get_session_by_file(file)
        if session:
            return session["sid"], None
    return None, {"error": "Provide sid or a file with an existing session."}


def run_create_decision(
    sid: str,
    decision_type: str,
    question: str,
    why_it_matters: str,
    column: str | None = None,
    options: list[str] | None = None,
    recommended: str | None = None,
    blocking: bool = False,
    source: str = "agent",
) -> dict:
    if not get_session(sid):
        return {"error": f"Session '{sid}' not found."}
    decision_id = create_decision(
        sid,
        decision_type,
        question,
        why_it_matters,
        column=column,
        options=options,
        recommended=recommended,
        blocking=blocking,
        source=source,
    )
    return json_safe({"sid": sid, "decision_id": decision_id})


def run_decisions(
    sid: str | None = None,
    file: str | None = None,
    status: str | None = "pending",
) -> dict:
    resolved_sid, error = _resolve_sid(sid, file)
    if error:
        return error
    return json_safe({
        "sid": resolved_sid,
        "decisions": get_decisions(resolved_sid, status=status),
    })


def run_resolve_decision(decision_id: str, answer: str, comment: str = "") -> dict:
    try:
        decision = resolve_decision(decision_id, answer, comment)
    except ValueError as exc:
        return {"error": str(exc)}
    if decision is None:
        return {"error": f"Decision '{decision_id}' not found."}
    return json_safe({"sid": decision["sid"], "decision": decision})


def run_log_event(
    sid: str,
    actor: str,
    event_type: str,
    summary: str,
    payload: dict | None = None,
) -> dict:
    if not get_session(sid):
        return {"error": f"Session '{sid}' not found."}
    try:
        event_id = log_event(sid, actor, event_type, summary, payload)
    except ValueError as exc:
        return {"error": str(exc)}
    return json_safe({"sid": sid, "event_id": event_id})


def run_journal(
    sid: str | None = None,
    file: str | None = None,
    event_type: str | None = None,
    limit: int = 30,
) -> dict:
    resolved_sid, error = _resolve_sid(sid, file)
    if error:
        return error
    return json_safe({
        "sid": resolved_sid,
        "journal": get_journal(resolved_sid, event_type=event_type, limit=limit),
    })


def run_next_action(sid: str | None = None, file: str | None = None) -> dict:
    resolved_sid, error = _resolve_sid(sid, file)
    if error:
        return error
    decisions = get_decisions(resolved_sid, status="pending")
    blocking = [item for item in decisions if item["blocking"]]
    if blocking:
        return json_safe({
            "sid": resolved_sid,
            "recommended_action": "ask_user",
            "reason": f"{len(blocking)} blocking decision(s) require domain input.",
            "decisions": blocking,
            "safe_actions": ["profile", "schema_check", "label_quality", "report"],
            "blocked_actions": ["train", "export", "predict"],
        })
    if decisions:
        return json_safe({
            "sid": resolved_sid,
            "recommended_action": "continue_with_caution",
            "reason": f"{len(decisions)} non-blocking decision(s) remain unresolved.",
            "decisions": decisions,
            "safe_actions": ["profile", "audit", "schema_check", "validate", "segments", "stability"],
            "blocked_actions": [],
        })
    return json_safe({
        "sid": resolved_sid,
        "recommended_action": "continue",
        "reason": "No pending decisions require user input.",
        "decisions": [],
        "safe_actions": ["profile", "audit", "split", "train", "validate", "predict", "export"],
        "blocked_actions": [],
    })
