from stratum.session import (
    create_session, get_artifacts, get_resolved_flags, get_run, get_session,
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
        metadata={"feature_names": ["x"], "training_policy": {"preset": "fast"}},
    )
    run = get_run("run_1")
    assert run["metrics"]["accuracy"] == .8
    assert run["training_policy"]["preset"] == "fast"
    save_artifact(sid, "split_train", str(tmp_path / "train.csv"))
    assert get_artifacts(sid, "split_train")[0]["kind"] == "split_train"
