from __future__ import annotations

import re
from typing import Any

import pandas as pd


PII_PATTERNS = [
    r"email", r"phone", r"mobile", r"ssn", r"social.?security", r"dob", r"birth",
    r"address", r"street", r"zip", r"postcode", r"passport", r"license",
    r"national.?id", r"\bname\b", r"first.?name", r"last.?name", r"surname",
]
PII_RE = re.compile("|".join(PII_PATTERNS), re.I)


def pii_name_map(columns) -> dict[str, str]:
    mapping = {}
    index = 0
    for name in columns:
        text = str(name)
        if PII_RE.search(text):
            mapping[text] = f"pii_column_{index}"
            index += 1
    return mapping


def redact_context(value: Any, columns) -> Any:
    mapping = pii_name_map(columns)

    def redact(item):
        if isinstance(item, dict):
            return {redact(key): redact(child) for key, child in item.items()}
        if isinstance(item, list):
            return [redact(child) for child in item]
        if isinstance(item, tuple):
            return [redact(child) for child in item]
        if isinstance(item, str):
            result = item
            for original, replacement in mapping.items():
                result = result.replace(original, replacement)
            return result
        return item

    return redact(value)


def sanitise_for_api(
    df: pd.DataFrame, target: str, privacy_mode: bool = True
) -> dict:
    columns = []
    redacted_names = pii_name_map(df.columns) if privacy_mode else {}
    for name in df.columns:
        safe_name = redacted_names.get(str(name), str(name))
        series = df[name]
        columns.append({
            "name": safe_name,
            "dtype": str(series.dtype),
            "n_unique": int(series.nunique(dropna=True)),
            "null_pct": round(float(series.isna().mean() * 100), 4),
            "sample_values": ["{redacted}"] * min(3, int(series.notna().sum())),
        })
    target_series = df[target]
    return {
        "columns": columns,
        "shape": [len(df), len(df.columns)],
        "target": {
            "name": redacted_names.get(target, target),
            "dtype": str(target_series.dtype),
            "value_counts": (
                {str(k): int(v) for k, v in target_series.value_counts().head(20).items()}
                if target_series.nunique(dropna=True) <= 20 else {}
            ),
        },
    }
