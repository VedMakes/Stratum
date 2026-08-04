from __future__ import annotations

import re

import pandas as pd

from .data.profiler import detect_datetime_columns, profile_target_relationship

ID_PATTERN = re.compile(r"(^id$|_id$|uuid|(^|_)key$|(^|_)index$|(^|_)idx$|no\.?$)", re.I)
LEAK_NAMES = ("date", "time", "paid", "received", "completed", "final", "result", "outcome", "after", "post", "end_")


def infer_task(target: pd.Series) -> str:
    unique = target.nunique(dropna=True)
    if unique == 2:
        return "binary"
    if unique <= 20 and (
        pd.api.types.is_integer_dtype(target)
        or pd.api.types.is_bool_dtype(target)
        or not pd.api.types.is_numeric_dtype(target)
    ):
        return "multiclass"
    return "regression"


def _encoded_target(target: pd.Series) -> pd.Series | None:
    clean = target.dropna()
    if clean.nunique() == 2:
        mapping = {value: i for i, value in enumerate(clean.unique())}
        return target.map(mapping).astype(float)
    if pd.api.types.is_numeric_dtype(target):
        return pd.to_numeric(target, errors="coerce")
    return None


def inspect_dataframe(df: pd.DataFrame, target: str) -> tuple[list[dict], dict]:
    y = df[target]
    task = infer_task(y)
    y_numeric = _encoded_target(y)
    target_relationships = {
        item["column"]: item
        for item in profile_target_relationship(df, target)
        if item["abs_statistic"] is not None
    }
    flags = []
    for col in df.columns:
        if col == target:
            continue
        series = df[col]
        relationship = target_relationships.get(str(col), {})
        correlation = float(relationship.get("abs_statistic") or 0.0)
        threshold = .95 if task == "regression" else .85
        medium = .70
        name_signal = any(signal in col.lower() for signal in LEAK_NAMES)
        if correlation > threshold or (task != "regression" and correlation >= medium) or (
            name_signal and correlation > .5
        ):
            confidence = min(1.0, correlation + (.1 if name_signal else 0))
            flags.append({
                "flag_type": "leakage", "column": str(col), "confidence": confidence,
                "reasoning": f"Absolute target correlation is {correlation:.3f}"
                + (" and the column name suggests post-outcome data." if name_signal else "."),
                "recommendation": "drop" if confidence >= .85 else "investigate",
            })
        if ID_PATTERN.search(str(col)):
            unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
            flags.append({
                "flag_type": "id_col", "column": str(col),
                "confidence": min(1.0, .7 + .3 * unique_ratio),
                "reasoning": f"ID-like name with {unique_ratio:.1%} unique values.",
                "recommendation": "drop" if unique_ratio > .95 else "investigate",
            })
        null_rate = float(series.isna().mean())
        null_corr = 0.0
        if y_numeric is not None and 0 < null_rate < 1:
            null_corr = abs(float(pd.Series(series.isna().astype(float)).corr(y_numeric) or 0))
        if null_rate > .2 or null_corr > .15:
            predictive = null_corr > .15
            flags.append({
                "flag_type": "null_pattern", "column": str(col),
                "confidence": min(1.0, max(null_rate, null_corr + .4)),
                "reasoning": (
                    f"{null_rate:.1%} nulls; null presence correlates {null_corr:.3f} with target. "
                    "Nulls may be predictive; keep an indicator."
                    if predictive else f"{null_rate:.1%} nulls; nulls appear to be noise."
                ),
                "recommendation": "keep_with_caution" if predictive else "investigate",
            })
    if task in {"binary", "multiclass"}:
        counts = y.value_counts(normalize=True)
        if len(counts) > 1 and counts.min() < .15:
            ratio = counts.max() / counts.min()
            flags.append({
                "flag_type": "imbalance", "column": target,
                "confidence": min(1.0, 1 - float(counts.min())),
                "reasoning": f"Minority class is {counts.min():.1%}; imbalance ratio is {ratio:.2f}:1.",
                "recommendation": "investigate",
            })
    datetime_columns = set(detect_datetime_columns(df))
    datetime_columns.update(
        str(col) for col in df.columns
        if col != target and pd.api.types.is_datetime64_any_dtype(df[col])
    )
    for col in sorted(datetime_columns):
        if col == target:
            continue
        flags.append({
            "flag_type": "temporal", "column": str(col), "confidence": .8,
            "reasoning": "Datetime column detected; random splitting may leak future information.",
            "recommendation": "investigate",
        })
    balance = (
        {str(k): int(v) for k, v in y.value_counts(dropna=False).items()}
        if task != "regression" else None
    )
    stats = {
        "rows": len(df), "cols": len(df.columns), "target_type": task,
        "class_balance": balance,
        "null_counts": {str(k): int(v) for k, v in df.isna().sum().items()},
        "datetime_columns": sorted(datetime_columns),
    }
    return flags, stats
