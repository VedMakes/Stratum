from stratum.schemas import ALL_TOOLS
from stratum.session import create_decision, create_session, get_artifacts, get_decisions, get_session_runs, save_artifact, save_run
from stratum.tools.tool_recommend import run_recommend
from stratum.tools.tool_report import run_report


class _Run:
    model_name = "demo"
    task = "classification"
    best_params = {}


def _save_run(tmp_path, sid, model_id="run_recommend"):
    dataset = tmp_path / "train.csv"
    dataset.write_text("x,target\n1,0\n2,1\n", encoding="utf-8")
    save_run(sid, model_id, _Run(), {"accuracy": .86}, metadata={
        "model_name": "demo",
        "task": "classification",
        "dataset_path": str(dataset),
        "target_col": "target",
        "training_policy": {"selection": {"metric": "accuracy", "direction": "maximize"}},
        "split_metadata": {"strategy": "stratified", "split_id": "split_test"},
        "evaluation": {
            "baseline": {"name": "prior_dummy", "metrics": {"accuracy": .57}},
            "comparison": {
                "metric": "accuracy", "model_score": .86, "baseline_score": .57,
                "improvement": .29, "model_beats_baseline": True,
            },
            "warnings": ["Holdout has only 7 rows; evaluation metrics may have high uncertainty."],
        },
        "provenance": {"dataset": {"rows": 32}},
    })
    return dataset


def test_recommendations_cite_saved_evidence_and_do_not_choose_or_train(tmp_path):
    dataset = tmp_path / "source.csv"
    dataset.write_text("x,target\n1,0\n2,1\n", encoding="utf-8")
    sid = create_session(str(dataset), "target")
    _save_run(tmp_path, sid)

    result = run_recommend(sid=sid)

    assert result["best_model"] == "run_recommend"
    assert result["recommendations_are_not_decisions"] is True
    ids = [item["recommendation_id"] for item in result["recommendations"]]
    assert ids == ["independent_validation", "estimate_metric_stability"]
    validation = result["recommendations"][0]
    assert validation["next_tool"] == "stratum_validate"
    assert any(e.get("id") == "run_recommend" for e in validation["evidence"])
    assert "small" in validation["rationale"]
    assert "not an independent validation" in result["recommendations"][1]["trade_off"]
    assert len(get_session_runs(sid)) == 1
    assert get_decisions(sid, status=None) == []
    saved = get_artifacts(sid, "experiment_recommendations")[-1]
    assert saved["metadata"]["recommendations"] == result["recommendations"]
    assert "experiment_recommendations" in run_report(sid=sid)["diagnostics"]


def test_blocking_decisions_take_priority_over_experiments(tmp_path):
    data = tmp_path / "source.csv"
    data.write_text("x,target\n1,0\n2,1\n", encoding="utf-8")
    sid = create_session(str(data), "target")
    create_decision(
        sid, "target_definition", "What does the label mean?", "The label changes the question.",
        options=["defined", "unknown"], safe_answers=["defined"], blocking=True,
    )
    _save_run(tmp_path, sid)

    result = run_recommend(sid=sid)

    assert [item["recommendation_id"] for item in result["recommendations"]] == [
        "resolve_blocking_decisions"
    ]
    assert result["recommendations"][0]["evidence"][0]["source"] == "decision"
    assert get_decisions(sid, status="pending")


def test_recommendation_tool_schema_and_available_artifacts(tmp_path):
    schema = next(tool for tool in ALL_TOOLS if tool["name"] == "stratum_recommend")
    assert schema["inputSchema"]["properties"].keys() == {"sid", "file"}

    data = tmp_path / "source.csv"
    data.write_text("x,target\n1,0\n2,1\n", encoding="utf-8")
    sid = create_session(str(data), "target")
    _save_run(tmp_path, sid)
    save_artifact(sid, "stability", str(data), model_id="run_recommend", metadata={"stability": "moderate"})
    validation_id = save_artifact(
        sid, "validation", str(data), model_id="run_recommend",
        metadata={"rows_evaluated": 40, "metrics": {"accuracy": .8}, "source_sha256": "independent"},
    )

    result = run_recommend(sid=sid)

    segment = next(item for item in result["recommendations"]
                   if item["recommendation_id"] == "inspect_validation_errors")
    assert segment["requires_user_choice"] is True
    assert any(item.get("id") == validation_id for item in segment["evidence"])
