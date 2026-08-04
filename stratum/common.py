from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_csv(file: str, target: str | None = None) -> tuple[pd.DataFrame | None, dict | None]:
    path = Path(file).expanduser()
    if not path.is_file():
        return None, {"error": f"File not found: {path}"}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return None, {"error": f"Unable to read CSV '{path}': {exc}"}
    if target is not None and target not in df.columns:
        return None, {
            "error": f"Column '{target}' not found. Available columns: {list(df.columns)}"
        }
    if df.empty:
        return None, {"error": f"CSV file is empty: {path}"}
    return df, None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def stratum_home() -> Path:
    return Path(os.environ.get("STRATUM_HOME", "~/.stratum")).expanduser()
