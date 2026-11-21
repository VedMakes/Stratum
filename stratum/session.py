from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import json_safe, stratum_home


CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    sid TEXT PRIMARY KEY, created_at TEXT, dataset_path TEXT,
    target_col TEXT, goal TEXT
);"""
CREATE_FLAGS = """
CREATE TABLE IF NOT EXISTS flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT, sid TEXT, flag_id TEXT UNIQUE,
    flag_type TEXT, column_name TEXT, confidence REAL, reasoning TEXT,
    status TEXT DEFAULT 'pending', user_comment TEXT, created_at TEXT,
    resolved_at TEXT
);"""
CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    model_id TEXT PRIMARY KEY, sid TEXT, model_name TEXT, task TEXT,
    best_params TEXT, metrics TEXT, top_features TEXT,
    decision_tree_summary TEXT, optuna_top5 TEXT, created_at TEXT,
    dataset_path TEXT, target_col TEXT, feature_names TEXT,
    feature_dtypes TEXT, preprocessing TEXT, metric_direction TEXT,
    artifact_path TEXT, training_policy TEXT, split_metadata TEXT, evaluation TEXT,
    provenance TEXT
);"""
CREATE_ARTIFACTS = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY, sid TEXT, model_id TEXT, kind TEXT,
    path TEXT, metadata TEXT, created_at TEXT
);"""
CREATE_DECISIONS = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, sid TEXT, decision_id TEXT UNIQUE,
    decision_type TEXT, column_name TEXT, question TEXT, why_it_matters TEXT,
    options TEXT, safe_answers TEXT DEFAULT '[]', recommended TEXT, blocking INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending', answer TEXT, user_comment TEXT,
    source TEXT, created_at TEXT, resolved_at TEXT
);"""
CREATE_JOURNAL = """
CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT, sid TEXT, event_id TEXT UNIQUE,
    actor TEXT, event_type TEXT, summary TEXT, payload TEXT, created_at TEXT
);"""

RUN_COLUMNS = {
    "dataset_path": "TEXT",
    "target_col": "TEXT",
    "feature_names": "TEXT",
    "feature_dtypes": "TEXT",
    "preprocessing": "TEXT",
    "metric_direction": "TEXT",
    "artifact_path": "TEXT",
    "training_policy": "TEXT",
    "split_metadata": "TEXT",
    "evaluation": "TEXT",
    "provenance": "TEXT",
}
UNSAFE_DECISION_ANSWERS = {"unknown", "deferred", "reject", "rejected"}


def _answer_is_safe(answer: str | None, safe_answers: list[str]) -> bool:
    if answer is None or answer not in safe_answers:
        return False
    return answer.strip().casefold() not in UNSAFE_DECISION_ANSWERS


def _resolution_state(status: str, answer: str | None, safe_answers: list[str]) -> str:
    if status != "resolved":
        return "pending" if status == "pending" else "invalidated"
    if _answer_is_safe(answer, safe_answers):
        return "confirmed"
    normalized = (answer or "").strip().casefold()
    if normalized in {"unknown", "deferred"}:
        return normalized
    if normalized in {"reject", "rejected"}:
        return "rejected"
    return "unsafe"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path(db_path: str | Path | None = None) -> Path:
    configured = db_path or os.environ.get("STRATUM_DB_PATH")
    return Path(configured).expanduser() if configured else stratum_home() / "sessions.db"


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    conn = sqlite3.connect(path)
    if os.name == "posix":
        try:
            path.chmod(0o600)
        except OSError:
            conn.close()
            raise
    conn.row_factory = sqlite3.Row
    _initialize(conn)
    return conn


def _initialize(conn: sqlite3.Connection) -> None:
    for statement in (
        CREATE_SESSIONS, CREATE_FLAGS, CREATE_RUNS, CREATE_ARTIFACTS,
        CREATE_DECISIONS, CREATE_JOURNAL,
    ):
        conn.execute(statement)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    for name, sql_type in RUN_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {name} {sql_type}")
    decision_columns = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}
    if "safe_answers" not in decision_columns:
        conn.execute("ALTER TABLE decisions ADD COLUMN safe_answers TEXT DEFAULT '[]'")
    conn.commit()


def create_session(
    dataset_path: str, target_col: str, goal: str = "", db_path: str | Path | None = None
) -> str:
    sid = uuid.uuid4().hex[:12]
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
            (sid, _now(), dataset_path, target_col, goal),
        )
    return sid


def get_session(sid: str, db_path: str | Path | None = None) -> dict | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM sessions WHERE sid = ?", (sid,)).fetchone()
    return dict(row) if row else None


def set_session_goal(sid: str, goal: str, db_path: str | Path | None = None) -> bool:
    """Set the original session goal once; later question changes belong in the journal."""
    goal = str(goal).strip()[:1000]
    if not goal:
        return False
    with _connect(db_path) as conn:
        row = conn.execute("SELECT goal FROM sessions WHERE sid=?", (sid,)).fetchone()
        if row is None or row["goal"]:
            return False
        conn.execute("UPDATE sessions SET goal=? WHERE sid=?", (goal, sid))
    return True


def get_session_by_file(file: str, db_path: str | Path | None = None) -> dict | None:
    path = str(Path(file).expanduser().resolve())
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE dataset_path = ? ORDER BY created_at DESC LIMIT 1",
            (path,),
        ).fetchone()
    return dict(row) if row else None


def ensure_session(
    sid: str | None, dataset_path: str = "", target_col: str = "", goal: str = ""
) -> tuple[str | None, dict | None]:
    if sid:
        if get_session(sid) is None:
            return None, {"error": f"Session '{sid}' not found. Omit sid to create a new session."}
        return sid, None
    return create_session(dataset_path, target_col, goal), None


def save_flag(sid: str, flag: dict, db_path: str | Path | None = None) -> str:
    flag_id = flag.get("flag_id") or f"flag_{uuid.uuid4().hex[:10]}"
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO flags
               (sid, flag_id, flag_type, column_name, confidence, reasoning,
                status, user_comment, created_at, resolved_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sid,
                flag_id,
                flag.get("flag_type"),
                flag.get("column") or flag.get("column_name"),
                float(flag.get("confidence", 0)),
                flag.get("reasoning", ""),
                flag.get("status", "pending"),
                flag.get("user_comment", ""),
                flag.get("created_at", _now()),
                flag.get("resolved_at"),
            ),
        )
    return flag_id


def resolve_flag(
    flag_id: str, status: str, comment: str = "", db_path: str | Path | None = None
) -> bool:
    if status not in {"approved", "rejected", "modified"}:
        raise ValueError("status must be approved, rejected, or modified")
    with _connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE flags SET status=?, user_comment=?, resolved_at=? WHERE flag_id=?",
            (status, comment, _now(), flag_id),
        )
    return cur.rowcount > 0


def get_resolved_flags(sid: str, db_path: str | Path | None = None) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM flags WHERE sid=? AND status != 'pending' ORDER BY created_at", (sid,)
        ).fetchall()
    return [dict(row) for row in rows]


def get_flags(sid: str, status: str | None = None, db_path: str | Path | None = None) -> list[dict]:
    query = "SELECT * FROM flags WHERE sid=?"
    params: list[Any] = [sid]
    if status:
        query += " AND status=?"
        params.append(status)
    query += " ORDER BY created_at"
    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def save_run(
    sid: str,
    model_id: str,
    result: Any,
    metrics: dict,
    *,
    metadata: dict | None = None,
    db_path: str | Path | None = None,
) -> None:
    metadata = metadata or {}
    payload = {
        "model_id": model_id,
        "sid": sid,
        "model_name": metadata.get("model_name", getattr(result, "model_name", "")),
        "task": metadata.get("task", getattr(result, "task", "")),
        "best_params": metadata.get("best_params", getattr(result, "best_params", {})),
        "metrics": metrics,
        "top_features": metadata.get("top_features", []),
        "decision_tree_summary": metadata.get("decision_tree_summary", ""),
        "optuna_top5": metadata.get("optuna_top5", []),
        "created_at": _now(),
        "dataset_path": metadata.get("dataset_path", ""),
        "target_col": metadata.get("target_col", ""),
        "feature_names": metadata.get("feature_names", []),
        "feature_dtypes": metadata.get("feature_dtypes", {}),
        "preprocessing": metadata.get("preprocessing", {}),
        "metric_direction": metadata.get("metric_direction", "maximize"),
        "artifact_path": metadata.get("artifact_path", ""),
        "training_policy": metadata.get("training_policy", {}),
        "split_metadata": metadata.get("split_metadata", {}),
        "evaluation": metadata.get("evaluation", {}),
        "provenance": metadata.get("provenance", {}),
    }
    json_fields = {
        "best_params", "metrics", "top_features", "optuna_top5", "feature_names",
        "feature_dtypes", "preprocessing", "training_policy", "split_metadata", "evaluation",
        "provenance",
    }
    values = [json.dumps(json_safe(payload[k])) if k in json_fields else payload[k] for k in payload]
    with _connect(db_path) as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO runs ({', '.join(payload)}) VALUES ({', '.join('?' for _ in payload)})",
            values,
        )


def _decode_run(row: sqlite3.Row | None) -> dict | None:
    if not row:
        return None
    result = dict(row)
    for key in (
        "best_params", "metrics", "top_features", "optuna_top5", "feature_names",
        "feature_dtypes", "preprocessing", "training_policy", "split_metadata", "evaluation",
        "provenance",
    ):
        try:
            result[key] = json.loads(result.get(key) or "null")
        except json.JSONDecodeError:
            pass
    return result


def get_run(model_id: str, db_path: str | Path | None = None) -> dict | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE model_id=?", (model_id,)).fetchone()
    return _decode_run(row)


def get_session_runs(sid: str, db_path: str | Path | None = None) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM runs WHERE sid=? ORDER BY created_at", (sid,)
        ).fetchall()
    return [_decode_run(row) for row in rows]


def save_artifact(
    sid: str,
    kind: str,
    path: str,
    model_id: str | None = None,
    metadata: dict | None = None,
    db_path: str | Path | None = None,
) -> str:
    artifact_id = f"artifact_{uuid.uuid4().hex[:10]}"
    artifact_metadata = dict(metadata or {})
    artifact_metadata.setdefault("artifact_schema_version", 1)
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?)",
            (artifact_id, sid, model_id, kind, path, json.dumps(json_safe(artifact_metadata)), _now()),
        )
    return artifact_id


def get_artifacts(
    sid: str, kind: str | None = None, db_path: str | Path | None = None
) -> list[dict]:
    query = "SELECT * FROM artifacts WHERE sid=?"
    params: list[Any] = [sid]
    if kind:
        query += " AND kind=?"
        params.append(kind)
    query += " ORDER BY created_at"
    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["metadata"] = json.loads(item["metadata"] or "{}")
        result.append(item)
    return result


def log_event(
    sid: str,
    actor: str,
    event_type: str,
    summary: str,
    payload: dict | None = None,
    db_path: str | Path | None = None,
) -> str:
    if actor not in {"user", "agent", "tool", "system"}:
        raise ValueError("actor must be user, agent, tool, or system")
    event_id = f"evt_{uuid.uuid4().hex[:10]}"
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO journal VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                None,
                sid,
                event_id,
                actor,
                event_type,
                str(summary)[:500],
                json.dumps(json_safe(payload or {})),
                _now(),
            ),
        )
    return event_id


def get_journal(
    sid: str,
    event_type: str | None = None,
    limit: int = 30,
    db_path: str | Path | None = None,
) -> list[dict]:
    query = "SELECT * FROM journal WHERE sid=?"
    params: list[Any] = [sid]
    if event_type:
        query += " AND event_type=?"
        params.append(event_type)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(max(1, min(int(limit), 100)))
    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["payload"] = json.loads(item["payload"] or "{}")
        result.append(item)
    return result


def create_decision(
    sid: str,
    decision_type: str,
    question: str,
    why_it_matters: str,
    *,
    column: str | None = None,
    options: list[str] | None = None,
    safe_answers: list[str] | None = None,
    recommended: str | None = None,
    blocking: bool = False,
    source: str = "",
    db_path: str | Path | None = None,
) -> str:
    options = options or ["approve", "reject", "unknown"]
    if blocking and not safe_answers:
        raise ValueError("Blocking decisions require at least one explicit safe_answer.")
    safe_answers = safe_answers or []
    invalid_safe_answers = [answer for answer in safe_answers if answer not in options]
    if invalid_safe_answers:
        raise ValueError(f"safe_answers must be selected from options: {options}")
    unsafe_answers = [
        answer for answer in safe_answers
        if answer.strip().casefold() in UNSAFE_DECISION_ANSWERS
    ]
    if unsafe_answers:
        raise ValueError("unknown, deferred, and rejected answers cannot be declared safe")
    with _connect(db_path) as conn:
        existing = conn.execute(
            """SELECT decision_id, options, safe_answers FROM decisions
               WHERE sid=? AND decision_type=? AND COALESCE(column_name, '')=?
               AND status='pending'""",
            (sid, decision_type, column or ""),
        ).fetchone()
        if existing:
            existing_options = json.loads(existing["options"] or "[]")
            stored_safe_answers = json.loads(existing["safe_answers"] or "[]")
            if blocking and not stored_safe_answers:
                if any(answer not in existing_options for answer in safe_answers):
                    raise ValueError(
                        f"safe_answers must be selected from existing options: {existing_options}"
                    )
                conn.execute(
                    "UPDATE decisions SET safe_answers=? WHERE decision_id=?",
                    (json.dumps(json_safe(safe_answers)), existing["decision_id"]),
                )
            return str(existing["decision_id"])
        decision_id = f"dec_{uuid.uuid4().hex[:10]}"
        conn.execute(
            """INSERT INTO decisions
               (sid, decision_id, decision_type, column_name, question,
                why_it_matters, options, safe_answers, recommended, blocking, status,
                answer, user_comment, source, created_at, resolved_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sid,
                decision_id,
                decision_type,
                column,
                str(question)[:500],
                str(why_it_matters)[:1000],
                json.dumps(json_safe(options)),
                json.dumps(json_safe(safe_answers)),
                recommended,
                int(bool(blocking)),
                "pending",
                None,
                "",
                source,
                _now(),
                None,
            ),
        )
    log_event(
        sid,
        "tool",
        "decision_created",
        f"Created decision {decision_id}: {question}",
        {"decision_id": decision_id, "decision_type": decision_type, "column": column},
        db_path=db_path,
    )
    return decision_id


def get_decisions(
    sid: str,
    status: str | None = "pending",
    db_path: str | Path | None = None,
) -> list[dict]:
    query = "SELECT * FROM decisions WHERE sid=?"
    params: list[Any] = [sid]
    if status:
        query += " AND status=?"
        params.append(status)
    query += " ORDER BY created_at"
    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["options"] = json.loads(item["options"] or "[]")
        item["safe_answers"] = json.loads(item.get("safe_answers") or "[]")
        item["blocking"] = bool(item["blocking"])
        item["resolution_state"] = _resolution_state(
            item["status"], item.get("answer"), item["safe_answers"]
        )
        item["safe_to_proceed"] = (
            not item["blocking"] or item["resolution_state"] == "confirmed"
        )
        result.append(item)
    return result


def get_workflow_blockers(sid: str, db_path: str | Path | None = None) -> list[dict]:
    """Return blocking decisions that have not been resolved with a safe answer."""
    return [
        decision
        for decision in get_decisions(sid, status=None, db_path=db_path)
        if decision["blocking"]
        and not decision["safe_to_proceed"]
    ]


def resolve_decision(
    decision_id: str,
    answer: str,
    comment: str = "",
    db_path: str | Path | None = None,
) -> dict | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM decisions WHERE decision_id=?", (decision_id,)
        ).fetchone()
        if not row:
            return None
        options = json.loads(row["options"] or "[]")
        safe_answers = json.loads(row["safe_answers"] or "[]")
        if options and answer not in options:
            raise ValueError(f"answer must be one of: {options}")
        conn.execute(
            """UPDATE decisions
               SET status='resolved', answer=?, user_comment=?, resolved_at=?
               WHERE decision_id=?""",
            (answer, comment, _now(), decision_id),
        )
    item = dict(row)
    item["options"] = options
    item["safe_answers"] = safe_answers
    item["blocking"] = bool(item["blocking"])
    item["status"] = "resolved"
    item["answer"] = answer
    item["user_comment"] = comment
    item["resolution_state"] = _resolution_state("resolved", answer, safe_answers)
    item["safe_to_proceed"] = not item["blocking"] or item["resolution_state"] == "confirmed"
    log_event(
        item["sid"],
        "user",
        "decision_resolved",
        f"Resolved decision {decision_id}: {answer}",
        {
            "decision_id": decision_id,
            "answer": answer,
            "comment": comment,
            "safe_to_proceed": item["safe_to_proceed"],
        },
        db_path=db_path,
    )
    return item
