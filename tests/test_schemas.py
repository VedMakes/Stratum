import json

from stratum.schemas import ALL_TOOLS


def test_schemas_are_compressed():
    encoded = json.dumps(ALL_TOOLS)
    assert len(ALL_TOOLS) == 28
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
