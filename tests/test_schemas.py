import json

from stratum.schemas import ALL_TOOLS


def test_schemas_are_compressed():
    encoded = json.dumps(ALL_TOOLS)
    assert len(ALL_TOOLS) == 27
    assert len(encoded) / 4 < 8500
