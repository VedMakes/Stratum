from __future__ import annotations

import pandas as pd

from ..common import json_safe, load_csv


def run_join_check(
    left_file: str,
    right_file: str,
    key: str,
    how: str = "left",
    target: str | None = None,
) -> dict:
    if how not in {"left", "inner", "outer"}:
        return {"error": "how must be one of: left, inner, outer"}
    left, error = load_csv(left_file)
    if error:
        return error
    right, error = load_csv(right_file)
    if error:
        return error
    if key not in left.columns or key not in right.columns:
        return {"error": f"Key '{key}' must exist in both CSVs."}

    issues = []
    left_counts = left[key].value_counts(dropna=False)
    right_counts = right[key].value_counts(dropna=False)
    left_many = int((left_counts > 1).sum())
    right_many = int((right_counts > 1).sum())
    if left_many:
        issues.append({"issue_type": "left_key_not_unique", "severity": "medium", "duplicate_keys": left_many})
    if right_many:
        issues.append({"issue_type": "right_key_not_unique", "severity": "high", "duplicate_keys": right_many})

    common_keys = set(left_counts.index) & set(right_counts.index)
    estimated_inner_rows = int(sum(int(left_counts[k]) * int(right_counts[k]) for k in common_keys))
    if how == "left":
        estimated_output_rows = int(sum(
            int(left_counts[k]) * max(int(right_counts.get(k, 0)), 1)
            for k in left_counts.index
        ))
        baseline = len(left)
    elif how == "outer":
        estimated_output_rows = estimated_inner_rows + int((~left[key].isin(right[key])).sum()) + int((~right[key].isin(left[key])).sum())
        baseline = max(len(left), len(right))
    else:
        estimated_output_rows = estimated_inner_rows
        baseline = min(len(left), len(right))
    if baseline and estimated_output_rows / baseline > 1.5:
        issues.append({
            "issue_type": "join_row_explosion",
            "severity": "high",
            "estimated_output_rows": estimated_output_rows,
            "baseline_rows": baseline,
        })

    unmatched_left = int((~left[key].isin(right[key])).sum())
    unmatched_right = int((~right[key].isin(left[key])).sum())
    if unmatched_left and how in {"left", "outer"}:
        issues.append({
            "issue_type": "unmatched_left_rows",
            "severity": "medium",
            "count": unmatched_left,
        })
    if unmatched_right and how in {"inner", "outer"}:
        issues.append({
            "issue_type": "unmatched_right_rows",
            "severity": "medium",
            "count": unmatched_right,
        })

    target_leak_columns = []
    if target:
        for col in right.columns:
            if col == target or target.lower() in str(col).lower():
                target_leak_columns.append(str(col))
        if target_leak_columns:
            issues.append({
                "issue_type": "right_table_target_leakage_risk",
                "severity": "high",
                "columns": target_leak_columns,
            })

    return json_safe({
        "left_rows": len(left),
        "right_rows": len(right),
        "key": key,
        "how": how,
        "estimated_inner_rows": estimated_inner_rows,
        "estimated_output_rows": estimated_output_rows,
        "unmatched_left_rows": unmatched_left,
        "unmatched_right_rows": unmatched_right,
        "issues": issues,
        "verdict": "risky" if issues else "safe",
    })
