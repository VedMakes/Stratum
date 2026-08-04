from __future__ import annotations

from pathlib import Path

from ..session import get_run, save_artifact


def _script(run: dict) -> str:
    params = repr(run["best_params"])
    features = repr(run["feature_names"])
    stratify = "y" if run["task"] == "classification" else "None"
    return f'''"""Standalone retraining script generated from a saved ML run."""
import argparse
import joblib
import pandas as pd
from MLProj import train
from sklearn.model_selection import train_test_split

FEATURES = {features}

def main(csv_path, output_path):
    df = pd.read_csv(csv_path)
    X = df[FEATURES].copy()
    y = df[{run["target_col"]!r}]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify={stratify}
    )
    result = train(
        {run["task"]!r},
        {run["model_name"]!r},
        X_train,
        y_train,
        tune=False,
        params={params},
    )
    print(result.evaluate(X_test, y_test))
    joblib.dump(result.estimator, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--output", default="trained_model.joblib")
    args = parser.parse_args()
    main(args.csv_path, args.output)
'''


def run_export(model_id: str, output_path: str | None = None, sid: str | None = None) -> dict:
    run = get_run(model_id)
    if not run:
        return {"error": f"Model '{model_id}' not found. Run stratum_train first."}
    if sid and sid != run["sid"]:
        return {"error": f"Model '{model_id}' does not belong to session '{sid}'."}
    try:
        code = _script(run)
    except Exception as exc:
        return {"error": f"Unable to generate export: {exc}"}
    script_path = None
    if output_path:
        script_path = Path(output_path).expanduser()
        if script_path.suffix != ".py":
            return {"error": "output_path must end in .py"}
        if script_path.exists():
            return {"error": f"Refusing to overwrite existing file: {script_path}"}
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(code, encoding="utf-8")
        save_artifact(run["sid"], "export_script", str(script_path), model_id=model_id)
    return {
        "sid": run["sid"], "model_id": model_id, "script": code,
        "script_path": str(script_path) if script_path else None,
        "model_path": run["artifact_path"],
        "warning": "Only load fitted joblib artifacts from trusted sources and compatible library versions.",
    }
