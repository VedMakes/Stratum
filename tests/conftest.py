import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def isolated_stratum(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATUM_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STRATUM_DB_PATH", str(tmp_path / "sessions.db"))


@pytest.fixture
def binary_csv(tmp_path):
    rng = np.random.default_rng(42)
    target = np.array([0, 1] * 30)
    df = pd.DataFrame({
        "age": rng.normal(40, 8, 60),
        "spend": rng.normal(100, 20, 60),
        "segment": np.where(target, "b", "a"),
        "customer_id": [f"c{i}" for i in range(60)],
        "leak": target + rng.normal(0, .01, 60),
        "target": target,
    })
    path = tmp_path / "binary.csv"
    df.to_csv(path, index=False)
    return path
