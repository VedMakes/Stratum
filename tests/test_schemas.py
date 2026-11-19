import json

from stratum.schemas import ALL_TOOLS


def test_schemas_are_compressed():
    encoded = json.dumps(ALL_TOOLS)
    assert len(ALL_TOOLS) == 27
    assert len(encoded) / 4 < 8500


def test_split_and_train_schemas_expose_saved_split_workflow():
    schemas = {tool["name"]: tool for tool in ALL_TOOLS}
    split_properties = schemas["stratum_split"]["inputSchema"]["properties"]
    train_properties = schemas["stratum_train"]["inputSchema"]["properties"]
    assert {"group_col", "split_col", "gap_rows", "seed"} <= set(split_properties)
    assert "split_id" in train_properties
