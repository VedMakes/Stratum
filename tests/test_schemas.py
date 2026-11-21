import json

from stratum.schemas import ALL_TOOLS


def test_schemas_are_compressed():
    encoded = json.dumps(ALL_TOOLS)
    assert len(ALL_TOOLS) == 30
    assert len(encoded) / 4 < 8500


def test_split_and_train_schemas_expose_saved_split_workflow():
    schemas = {tool["name"]: tool for tool in ALL_TOOLS}
    split_properties = schemas["stratum_split"]["inputSchema"]["properties"]
    train_properties = schemas["stratum_train"]["inputSchema"]["properties"]
    assert {"group_col", "split_col", "gap_rows", "seed"} <= set(split_properties)
    assert "split_id" in train_properties
    assert {"metric", "metric_direction"} <= set(train_properties)


def test_stability_schema_exposes_reproducible_repeats():
    schema = next(tool for tool in ALL_TOOLS if tool["name"] == "stratum_stability")
    properties = schema["inputSchema"]["properties"]
    assert {"folds", "repeats", "seed", "metric"} <= set(properties)


def test_group_evaluation_schema_requires_selected_group_column():
    schema = next(tool for tool in ALL_TOOLS if tool["name"] == "stratum_group_evaluation")
    assert schema["inputSchema"]["required"] == ["model_id", "file", "group_col"]
    assert "min_support" in schema["inputSchema"]["properties"]


def test_drift_schemas_expose_monitoring_thresholds():
    schemas = {tool["name"]: tool for tool in ALL_TOOLS}
    for name in ("stratum_drift", "stratum_drift_current"):
        properties = schemas[name]["inputSchema"]["properties"]
        assert {"psi_mild_threshold", "psi_significant_threshold", "psi_high_threshold", "p_value_threshold"} <= set(properties)


def test_profile_schema_can_capture_concise_local_question_context():
    schema = next(tool for tool in ALL_TOOLS if tool["name"] == "stratum_profile")
    assert "goal" in schema["inputSchema"]["properties"]
    assert "conversation transcript" in schema["description"]
