import json

import pandas as pd

from stratum.cli import main
from stratum.session import save_run


class _ReportRun:
    model_name = "demo"
    task = "classification"
    best_params = {}


def test_cli_profiles_and_inspects_session_artifacts(tmp_path, capsys):
    data = tmp_path / "customers.csv"
    pd.DataFrame({"spend": range(40), "target": [0, 1] * 20}).to_csv(data, index=False)

    assert main(["profile", str(data), "--target", "target"]) == 0
    profile = json.loads(capsys.readouterr().out)
    assert profile["shape"] == [40, 2]

    assert main(["artifacts", "--sid", profile["sid"]]) == 0
    artifacts = json.loads(capsys.readouterr().out)
    assert artifacts["sid"] == profile["sid"]
    assert artifacts["artifacts"]


def test_cli_split_forwards_split_design(tmp_path, capsys):
    data = tmp_path / "customers.csv"
    pd.DataFrame({"spend": range(60), "target": [0, 1] * 30}).to_csv(data, index=False)

    result_code = main([
        "split", str(data), "--target", "target", "--mode", "stratified", "--seed", "11",
    ])
    result = json.loads(capsys.readouterr().out)

    assert result_code == 0
    assert result["split_mode"] == "stratified"
    assert result["parameters"]["seed"] == 11


def test_cli_returns_nonzero_for_tool_errors(tmp_path, capsys):
    result = main(["validate", "missing-model", str(tmp_path / "missing.csv")])

    assert result == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_cli_writes_self_contained_html_report(tmp_path, capsys):
    data = tmp_path / "customers.csv"
    pd.DataFrame({"spend": range(40), "target": [0, 1] * 20}).to_csv(data, index=False)
    profile = __import__("stratum.data.profiler", fromlist=["run_profile"]).run_profile(
        str(data), target="target"
    )
    save_run(profile["sid"], "run-html", _ReportRun(), {"accuracy": .75}, metadata={
        "model_name": "demo", "task": "classification", "feature_names": ["spend"],
        "provenance": {"schema_version": 1, "dataset": {"frame_fingerprint": "demo"}},
    })
    output = tmp_path / "session-report.html"

    result = main(["report", "--sid", profile["sid"], "--format", "html", "--output", str(output)])

    assert result == 0
    assert json.loads(capsys.readouterr().out)["output_path"] == str(output)
    html_report = output.read_text(encoding="utf-8")
    assert "Stratum session report" in html_report
    assert "Data risks and assumptions" in html_report
    assert "Provenance" in html_report
