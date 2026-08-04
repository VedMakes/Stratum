from __future__ import annotations

import pandas as pd

from ..common import json_safe, load_csv


def _hash_rows(df: pd.DataFrame, columns: list[str]) -> set[int]:
    if not columns:
        return set()
    return set(pd.util.hash_pandas_object(df[columns].astype(str), index=False).astype("uint64").tolist())


def run_contamination(
    train_file: str,
    test_file: str,
    target: str | None = None,
    id_col: str | None = None,
    time_col: str | None = None,
) -> dict:
    train, error = load_csv(train_file)
    if error:
        return error
    test, error = load_csv(test_file)
    if error:
        return error

    common = [col for col in train.columns if col in test.columns]
    feature_cols = [col for col in common if col != target]
    flags = []
    exact_overlap = len(_hash_rows(train, common) & _hash_rows(test, common))
    feature_overlap = len(_hash_rows(train, feature_cols) & _hash_rows(test, feature_cols))
    if exact_overlap:
        flags.append({
            "flag_type": "duplicate_rows_across_splits",
            "severity": "high",
            "count": exact_overlap,
            "reasoning": "Exact rows appear in both train and test.",
        })
    if feature_overlap:
        flags.append({
            "flag_type": "duplicate_features_across_splits",
            "severity": "medium",
            "count": feature_overlap,
            "reasoning": "Feature-identical rows appear in both train and test.",
        })

    if id_col and id_col in train.columns and id_col in test.columns:
        overlap = sorted(set(train[id_col].dropna().astype(str)) & set(test[id_col].dropna().astype(str)))
        if overlap:
            flags.append({
                "flag_type": "id_overlap",
                "severity": "high",
                "column": id_col,
                "count": len(overlap),
                "examples": overlap[:10],
            })

    if time_col and time_col in train.columns and time_col in test.columns:
        train_time = pd.to_datetime(train[time_col], errors="coerce")
        test_time = pd.to_datetime(test[time_col], errors="coerce")
        if train_time.notna().any() and test_time.notna().any() and test_time.min() < train_time.max():
            flags.append({
                "flag_type": "temporal_overlap",
                "severity": "high",
                "column": time_col,
                "reasoning": "Test timestamps begin before the latest train timestamp.",
                "train_max": train_time.max().isoformat(),
                "test_min": test_time.min().isoformat(),
            })

    if target and target in train.columns and target in test.columns:
        train_dist = train[target].astype(str).value_counts(normalize=True, dropna=False)
        test_dist = test[target].astype(str).value_counts(normalize=True, dropna=False)
        classes = sorted(set(train_dist.index) | set(test_dist.index))
        max_delta = max(
            abs(float(train_dist.get(cls, 0)) - float(test_dist.get(cls, 0)))
            for cls in classes
        ) if classes else 0.0
        if max_delta > .2:
            flags.append({
                "flag_type": "target_distribution_mismatch",
                "severity": "medium",
                "max_delta": max_delta,
                "reasoning": "Target distribution differs materially between train and test.",
            })

    return json_safe({
        "train_rows": len(train),
        "test_rows": len(test),
        "flags": flags,
        "verdict": "contaminated" if any(f["severity"] == "high" for f in flags) else (
            "risky" if flags else "clean"
        ),
    })
