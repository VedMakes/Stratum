import sqlite3
import stat
import os

import pytest

from stratum.session import (
    _connect, create_session, get_artifacts, get_resolved_flags, get_run, get_session,
    resolve_flag, save_artifact, save_flag, save_run,
)


class Result:
    model_name = "random_forest"
    task = "classification"
    best_params = {"n_estimators": 10}


def test_session_flag_run_and_artifact_roundtrip(tmp_path):
    sid = create_session("/tmp/data.csv", "target")
    assert get_session(sid)["target_col"] == "target"
    flag_id = save_flag(sid, {"flag_type": "id_col", "column": "id", "confidence": .9})
    assert resolve_flag(flag_id, "approved", "known key")
    assert get_resolved_flags(sid)[0]["user_comment"] == "known key"
    save_run(
        sid,
        "run_1",
        Result(),
        {"accuracy": .8},
        metadata={
            "feature_names": ["x"],
            "training_policy": {"preset": "fast"},
            "split_metadata": {"strategy": "group", "split_id": "split_1"},
            "evaluation": {"comparison": {"metric": "f1_weighted", "improvement": .1}},
            "provenance": {"schema_version": 1, "artifact": {"sha256": "abc"}},
        },
    )
    run = get_run("run_1")
    assert run["metrics"]["accuracy"] == .8
    assert run["training_policy"]["preset"] == "fast"
    assert run["split_metadata"]["split_id"] == "split_1"
    assert run["evaluation"]["comparison"]["improvement"] == .1
    assert run["provenance"]["artifact"]["sha256"] == "abc"
    save_artifact(sid, "split_train", str(tmp_path / "train.csv"))
    saved_artifact = get_artifacts(sid, "split_train")[0]
    assert saved_artifact["kind"] == "split_train"
    assert saved_artifact["metadata"]["artifact_schema_version"] == 1


def test_existing_decision_table_gets_safe_answers_column(tmp_path):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""CREATE TABLE decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, sid TEXT, decision_id TEXT UNIQUE,
            decision_type TEXT, column_name TEXT, question TEXT, why_it_matters TEXT,
            options TEXT, recommended TEXT, blocking INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending', answer TEXT, user_comment TEXT,
            source TEXT, created_at TEXT, resolved_at TEXT
        )""")
        conn.execute("""CREATE TABLE runs (
            model_id TEXT PRIMARY KEY, sid TEXT, model_name TEXT, task TEXT,
            best_params TEXT, metrics TEXT, top_features TEXT,
            decision_tree_summary TEXT, optuna_top5 TEXT, created_at TEXT,
            dataset_path TEXT, target_col TEXT, feature_names TEXT,
            feature_dtypes TEXT, preprocessing TEXT, metric_direction TEXT,
            artifact_path TEXT, training_policy TEXT
        )""")

    with _connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}
        run_columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}

    assert "safe_answers" in columns
    assert "split_metadata" in run_columns
    assert "evaluation" in run_columns
    assert "provenance" in run_columns


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes only")
def test_session_database_is_owner_only(tmp_path):
    db_path = tmp_path / "private" / "sessions.db"

    with _connect(db_path):
        pass

    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
