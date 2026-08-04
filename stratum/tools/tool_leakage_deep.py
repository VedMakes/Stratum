from __future__ import annotations

import pandas as pd

from ..audit import LEAK_NAMES
from ..common import json_safe, load_csv
from ..data.profiler import detect_datetime_columns, profile_target_relationship
from ..session import create_decision, ensure_session


def _cramers_v_all_categories(x: pd.Series, y: pd.Series) -> float:
    from scipy.stats import chi2_contingency
    import numpy as np

    table = pd.crosstab(x.astype(str), y.astype(str))
    if table.empty or min(table.shape) <= 1:
        return 0.0
    chi2, _, _, _ = chi2_contingency(table)
    n = table.sum().sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))) if n else 0.0


def run_leakage_deep(
    file: str, target: str, time_col: str | None = None, sid: str | None = None
) -> dict:
    df, error = load_csv(file, target)
    if error:
        return error
    sid, error = ensure_session(sid, file, target)
    if error:
        return error
    y = df[target]
    flags = []

    for col in df.columns:
        if col == target:
            continue
        same = df[col].astype(str).fillna("<null>").equals(y.astype(str).fillna("<null>"))
        if same:
            flags.append({
                "flag_type": "duplicate_target",
                "severity": "critical",
                "column": str(col),
                "reasoning": "Column values exactly duplicate the target.",
                "recommendation": "drop",
            })

    for item in profile_target_relationship(df, target):
        stat = item.get("abs_statistic") or 0.0
        if stat >= .95:
            flags.append({
                "flag_type": "near_perfect_target_relationship",
                "severity": "critical",
                "column": item["column"],
                "method": item["method"],
                "statistic": stat,
                "recommendation": "drop_or_investigate_lineage",
            })

    for col in df.select_dtypes(exclude=["number", "bool"]).columns:
        if col == target:
            continue
        n_unique = df[col].nunique(dropna=True)
        if 20 < n_unique <= max(2000, len(df) // 2):
            stat = _cramers_v_all_categories(df[col], y)
            if stat >= .95:
                flags.append({
                    "flag_type": "high_cardinality_target_encoding_smell",
                    "severity": "high",
                    "column": str(col),
                    "cramers_v": round(stat, 6),
                    "reasoning": "High-cardinality categorical column is almost perfectly associated with the target.",
                })

    for col in df.columns:
        if col == target:
            continue
        lowered = str(col).lower()
        if any(token in lowered for token in LEAK_NAMES):
            decision_id = create_decision(
                sid,
                "temporal_availability",
                f"Was column '{col}' available before prediction time?",
                "If this column is recorded after the outcome, it can leak the target.",
                column=str(col),
                options=["available_before_prediction", "post_outcome", "unknown"],
                recommended="unknown",
                blocking=True,
                source="stratum_leakage_deep",
            )
            flags.append({
                "flag_type": "post_outcome_name_signal",
                "severity": "medium",
                "column": str(col),
                "decision_id": decision_id,
                "reasoning": "Column name suggests data that may be recorded after the outcome.",
            })

    feature_cols = [col for col in df.columns if col != target]
    if feature_cols:
        grouped = df.groupby(feature_cols, dropna=False)[target].nunique(dropna=True)
        conflicts = int((grouped > 1).sum())
        if conflicts:
            flags.append({
                "flag_type": "conflicting_labels_for_duplicate_features",
                "severity": "high",
                "count": conflicts,
                "reasoning": "Identical feature rows map to multiple target labels.",
            })

    datetime_cols = detect_datetime_columns(df)
    for col in datetime_cols:
        if col != target and (time_col is None or col != time_col):
            decision_id = create_decision(
                sid,
                "temporal_availability",
                f"Was datetime column '{col}' known before prediction time?",
                "Datetime-like features are safe only if they are available at scoring time.",
                column=col,
                options=["available_before_prediction", "post_outcome", "unknown"],
                recommended="unknown",
                blocking=True,
                source="stratum_leakage_deep",
            )
            flags.append({
                "flag_type": "temporal_feature_requires_lineage_check",
                "severity": "medium",
                "column": col,
                "decision_id": decision_id,
                "reasoning": "Datetime-like feature should be confirmed as available before prediction time.",
            })

    flags.sort(key=lambda item: {"critical": 3, "high": 2, "medium": 1}.get(item["severity"], 0), reverse=True)
    return json_safe({
        "rows": len(df),
        "sid": sid,
        "target": target,
        "flags": flags,
        "verdict": "critical" if any(f["severity"] == "critical" for f in flags) else (
            "risky" if flags else "clean"
        ),
    })
