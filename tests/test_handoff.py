import pandas as pd
import pytest

from stratum.html_report import render_report_html
from stratum.session import create_session, get_journal, get_session, save_run
from stratum.tools.tool_report import run_report
from stratum.tools.tool_status import run_status


class _Run:
    model_name = "demo"
    task = "classification"
    best_params = {}


def _save_run(sid, model_id, score, dataset_hash, split, features, decision_ids):
    save_run(sid, model_id, _Run(), {"accuracy": score}, metadata={
        "model_name": "demo",
        "task": "classification",
        "target_col": "target",
        "feature_names": features,
        "training_policy": {"selection": {"metric": "accuracy", "direction": "maximize"}},
        "split_metadata": split,
        "provenance": {
            "dataset": {"sha256": dataset_hash, "rows": 80},
            "split": split,
            "lineage": {"decision_ids": decision_ids},
        },
    })


def test_profile_goal_and_question_changes_survive_report_and_html(tmp_path):
    from stratum.data.profiler import run_profile

    data = tmp_path / "customers.csv"
    pd.DataFrame({"x": range(40), "target": [0, 1] * 20}).to_csv(data, index=False)
    first = run_profile(str(data), target="target", goal="Estimate which customers may churn.")
    second = run_profile(
        str(data), target="target", sid=first["sid"],
        goal="Now compare the result with a later customer cohort.",
    )

    assert second["sid"] == first["sid"]
    assert get_session(first["sid"])["goal"] == "Estimate which customers may churn."
    questions = get_journal(first["sid"], event_type="analytical_question", limit=100)
    assert [item["summary"] for item in reversed(questions)] == [
        "Estimate which customers may churn.",
        "Now compare the result with a later customer cohort.",
    ]
    report = run_report(sid=first["sid"])
    assert report["session_goal"] == "Estimate which customers may churn."
    assert len(report["question_history"]) == 2
    html = render_report_html(report)
    assert "Question, assumptions, and journal" in html
    assert "Estimate which customers may churn." in html


def test_run_changes_compare_only_identical_dataset_target_and_split(tmp_path):
    data = tmp_path / "customers.csv"
    data.write_text("x,target\n1,0\n2,1\n", encoding="utf-8")
    sid = create_session(str(data), "target", goal="Estimate churn.")
    same_split = {
        "strategy": "stratified",
        "train_row_fingerprint": "train-fingerprint",
        "test_row_fingerprint": "test-fingerprint",
    }
    _save_run(sid, "run_a", .70, "same-data", same_split, ["x"], ["dec_1"])
    _save_run(sid, "run_b", .85, "same-data", same_split, ["x", "new_feature"], ["dec_1", "dec_2"])

    status = run_status(sid=sid)
    change = status["runs"][1]["change_from_previous"]

    assert change["metric_comparable"] is True
    assert change["metric_delta"] == pytest.approx(.15)
    assert change["added_features"] == ["new_feature"]
    assert change["new_decision_ids"] == ["dec_2"]

    _save_run(sid, "run_c", .99, "different-data", same_split, ["x"], ["dec_1", "dec_2"])
    different_data = run_status(sid=sid)["runs"][2]["change_from_previous"]
    assert different_data["same_dataset"] is False
    assert different_data["metric_comparable"] is False
    assert different_data["metric_delta"] is None
    assert "not compared" in different_data["note"]


def test_status_does_not_compare_runs_with_different_split_fingerprints(tmp_path):
    data = tmp_path / "customers.csv"
    data.write_text("x,target\n1,0\n2,1\n", encoding="utf-8")
    sid = create_session(str(data), "target")
    split_a = {"strategy": "stratified", "train_row_fingerprint": "a", "test_row_fingerprint": "b"}
    split_b = {"strategy": "stratified", "train_row_fingerprint": "c", "test_row_fingerprint": "d"}
    _save_run(sid, "run_a", .70, "same-data", split_a, ["x"], [])
    _save_run(sid, "run_b", .85, "same-data", split_b, ["x"], [])

    change = run_status(sid=sid)["runs"][1]["change_from_previous"]

    assert change["same_dataset"] is True
    assert change["same_split"] is False
    assert change["metric_comparable"] is False
    assert change["metric_delta"] is None
