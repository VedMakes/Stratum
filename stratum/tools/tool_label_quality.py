from __future__ import annotations

import numpy as np
import pandas as pd

from ..audit import infer_task
from ..common import json_safe, load_csv


def run_label_quality(
    file: str,
    target: str,
    time_col: str | None = None,
    min_class_count: int = 5,
) -> dict:
    df, error = load_csv(file, target)
    if error:
        return error
    y = df[target]
    task = infer_task(y)
    issues = []

    null_count = int(y.isna().sum())
    if null_count:
        issues.append({
            "issue_type": "null_labels",
            "severity": "high",
            "count": null_count,
            "reasoning": "Rows with missing labels cannot be used directly for supervised training.",
        })

    feature_cols = [col for col in df.columns if col != target]
    if feature_cols:
        conflicts = int((df.groupby(feature_cols, dropna=False)[target].nunique(dropna=True) > 1).sum())
        if conflicts:
            issues.append({
                "issue_type": "conflicting_labels_for_duplicate_features",
                "severity": "high",
                "count": conflicts,
            })

    if task in {"binary", "multiclass"}:
        counts = y.value_counts(dropna=False)
        rare = {str(k): int(v) for k, v in counts.items() if v < min_class_count}
        if rare:
            issues.append({
                "issue_type": "rare_classes",
                "severity": "medium",
                "classes": rare,
                "reasoning": "Rare classes make splitting, validation, and calibration unstable.",
            })
        if len(counts) > 1:
            minority = float(counts.min() / counts.sum())
            if minority < .05:
                issues.append({
                    "issue_type": "extreme_class_imbalance",
                    "severity": "medium",
                    "minority_fraction": minority,
                })

        if time_col and time_col in df.columns:
            parsed = pd.to_datetime(df[time_col], errors="coerce")
            valid = df.loc[parsed.notna()].assign(__time=parsed[parsed.notna()])
            if len(valid) >= 10:
                ordered = valid.sort_values("__time")
                chunks = np.array_split(ordered, min(4, len(ordered)))
                rates = []
                for chunk in chunks:
                    dist = chunk[target].value_counts(normalize=True)
                    rates.append({str(k): float(v) for k, v in dist.items()})
                classes = {key for row in rates for key in row}
                max_shift = max(
                    max(row.get(cls, 0.0) for row in rates) - min(row.get(cls, 0.0) for row in rates)
                    for cls in classes
                ) if classes else 0.0
                if max_shift > .25:
                    issues.append({
                        "issue_type": "label_distribution_shift_over_time",
                        "severity": "medium",
                        "max_shift": max_shift,
                        "time_col": time_col,
                    })
    else:
        numeric = pd.to_numeric(y, errors="coerce")
        invalid = int(numeric.isna().sum() - y.isna().sum())
        if invalid:
            issues.append({
                "issue_type": "non_numeric_regression_labels",
                "severity": "high",
                "count": invalid,
            })
        clean = numeric.dropna()
        if len(clean) > 10:
            z = ((clean - clean.mean()) / clean.std()).abs() if clean.std() else pd.Series([], dtype=float)
            outliers = int((z > 4).sum()) if len(z) else 0
            if outliers:
                issues.append({
                    "issue_type": "extreme_target_outliers",
                    "severity": "medium",
                    "count": outliers,
                })

    return json_safe({
        "rows": len(df),
        "target": target,
        "target_type": task,
        "class_balance": (
            {str(k): int(v) for k, v in y.value_counts(dropna=False).items()}
            if task in {"binary", "multiclass"} else None
        ),
        "issues": issues,
        "verdict": "risky" if issues else "clean",
    })
