from __future__ import annotations

import re
from typing import Any

import pandas as pd


PII_PATTERNS = [
    r"email", r"phone", r"mobile", r"ssn", r"social.?security", r"dob", r"birth",
    r"address", r"street", r"zip", r"postcode", r"passport", r"license",
    r"national.?id", r"\bname\b", r"first.?name", r"last.?name", r"surname",
    r"\b(?:customer|client|patient|member|user|account|employee|person)[_ -]?(?:id|number|no)\b",
    r"\b(?:id|uuid|guid|identifier|record[_ -]?id)\b",
]
PII_RE = re.compile("|".join(PII_PATTERNS), re.I)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", re.I)
LABELED_ID_RE = re.compile(
    r"\b(?:customer|client|patient|member|user|account|employee|record)[_ -]?(?:id|number|no)\b"
    r"\s*[:=#]?\s*[A-Z0-9][A-Z0-9_-]{3,}",
    re.I,
)


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
    sensitive_names = {name.casefold() for name in mapping}

    def redact(item):
        if isinstance(item, dict):
            result = {}
            for key, child in item.items():
                if isinstance(key, str) and key.casefold() in sensitive_names:
                    result[redact(key)] = "{redacted}"
                else:
                    result[redact(key)] = redact(child)
            return result
        if isinstance(item, list):
            return [redact(child) for child in item]
        if isinstance(item, tuple):
            return [redact(child) for child in item]
        if isinstance(item, str):
            result = LABELED_ID_RE.sub("{redacted_id}", item)
            for original, replacement in mapping.items():
                result = re.sub(re.escape(original), replacement, result, flags=re.I)
            result = EMAIL_RE.sub("{redacted_email}", result)
            result = SSN_RE.sub("{redacted_id}", result)
            result = UUID_RE.sub("{redacted_id}", result)
            return result
        return item

    return redact(value)


def safe_flag_context(flags: list[dict], columns) -> list[dict]:
    safe = [
        {
            "flag_type": flag.get("flag_type"),
            "column": flag.get("column_name"),
            "status": flag.get("status"),
            "confidence": flag.get("confidence"),
        }
        for flag in flags
    ]
    return redact_context(safe, columns)


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
    target_counts = target_series.value_counts().head(20)
    return {
        "columns": columns,
        "shape": [len(df), len(df.columns)],
        "target": {
            "name": redacted_names.get(target, target),
            "dtype": str(target_series.dtype),
            "value_counts": (
                {f"class_{index}": int(count) for index, count in enumerate(target_counts)}
                if target_series.nunique(dropna=True) <= 20 else {}
            ),
        },
    }
