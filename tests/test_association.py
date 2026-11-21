import numpy as np
import pandas as pd

from stratum.schemas import ALL_TOOLS
from stratum.session import get_artifacts
from stratum.tools.tool_association import run_association
from stratum.tools.tool_report import run_report


def _association_data(path):
    rng = np.random.default_rng(9)
    outcome = np.array([1] * 40 + [0] * 40)
    df = pd.DataFrame({
        "outcome": outcome,
        "signal": np.r_[rng.normal(2, .2, 40), rng.normal(0, .2, 40)],
        "noise": rng.normal(size=80),
        "channel": ["web", "store"] * 40,
    })
    df.loc[0, "signal"] = np.nan
    df.loc[2, "signal"] = np.inf
    df.loc[1, "channel"] = np.nan
    df.to_csv(path, index=False)
    return df


def test_association_reports_oriented_effects_uncertainty_fdr_and_artifact(tmp_path):
    path = tmp_path / "association.csv"
    _association_data(path)
    result = run_association(
        str(path), "outcome", "1", ["signal", "noise", "channel"],
        bootstrap_samples=200,
    )

    signal = result["results"][0]
    categorical = result["results"][2]
    assert result["sid"]
    assert signal["effect_name"] == "cliffs_delta"
    assert signal["effect_size"] > .9
    assert signal["ci_95"][0] <= signal["effect_size"] <= signal["ci_95"][1]
    assert signal["missing_rows_excluded"] == 2
    assert categorical["effect_name"] == "cramers_v"
    assert categorical["effect_size"] >= 0
    assert categorical["missing_rows_excluded"] == 1
    assert all(item["p_value_bh"] >= item["p_value"] for item in result["results"])
    assert "do not establish causality" in result["warnings"][0]
    artifacts = get_artifacts(result["sid"], "statistical_analysis")
    assert artifacts[0]["metadata"]["features"] == ["signal", "noise", "channel"]
    assert "statistical_analysis" in run_report(sid=result["sid"])["diagnostics"]


def test_association_is_reproducible_and_requires_explicit_binary_outcome(tmp_path):
    path = tmp_path / "association.csv"
    _association_data(path)
    first = run_association(str(path), "outcome", "1", ["signal"], bootstrap_samples=100, seed=7)
    second = run_association(str(path), "outcome", "1", ["signal"], bootstrap_samples=100, seed=7)
    assert first["results"][0]["ci_95"] == second["results"][0]["ci_95"]
    assert "exactly two" in run_association(str(path), "signal", "1", ["noise"])["error"]
    assert "positive_label" in run_association(str(path), "outcome", "other", ["noise"])["error"]
    assert "between 1 and 10" in run_association(str(path), "outcome", "1", [])["error"].lower()


def test_association_rejects_outcome_reuse_and_high_cardinality_category(tmp_path):
    path = tmp_path / "association.csv"
    df = _association_data(path)
    df["many_categories"] = [f"category_{i}" for i in range(len(df))]
    df.to_csv(path, index=False)

    assert "cannot also be analyzed" in run_association(
        str(path), "outcome", "1", ["outcome"]
    )["error"]
    assert "more than 20 categories" in run_association(
        str(path), "outcome", "1", ["many_categories"]
    )["error"]


def test_association_counts_rows_with_missing_outcome_as_excluded(tmp_path):
    path = tmp_path / "missing-outcome.csv"
    df = _association_data(path)
    df.loc[len(df)] = [np.nan, 0.5, 0.0, "web"]
    df.to_csv(path, index=False)

    result = run_association(str(path), "outcome", "1", ["signal"], bootstrap_samples=100)

    assert result["rows"] == 81
    assert result["outcome_rows_excluded"] == 1
    assert result["outcome_support"] == {"1": 40, "other": 40}


def test_association_accepts_boolean_outcomes_as_labels(tmp_path):
    path = tmp_path / "boolean.csv"
    pd.DataFrame({
        "outcome": [True, False] * 20,
        "feature": ["a", "b"] * 20,
    }).to_csv(path, index=False)

    result = run_association(str(path), "outcome", "True", ["feature"], bootstrap_samples=100)

    assert "error" not in result
    assert result["outcome_support"] == {"True": 20, "other": 20}
    assert result["results"][0]["effect_name"] == "cramers_v"


def test_association_warns_when_group_support_is_low(tmp_path):
    path = tmp_path / "small.csv"
    pd.DataFrame({
        "outcome": [1, 1, 1, 1, 0, 0, 0, 0],
        "feature": [3, 4, 5, 6, 0, 1, 2, 3],
    }).to_csv(path, index=False)

    result = run_association(str(path), "outcome", "1", ["feature"], bootstrap_samples=100)

    assert result["results"][0]["support"] == {"positive": 4, "other": 4}
    assert any("fewer than 10" in warning for warning in result["results"][0]["warnings"])
    assert any("Only 8 rows" in warning for warning in result["warnings"])


def test_association_schema_requires_a_reviewable_hypothesis():
    schema = next(tool for tool in ALL_TOOLS if tool["name"] == "stratum_association")
    assert schema["inputSchema"]["required"] == ["file", "outcome", "positive_label", "features"]

