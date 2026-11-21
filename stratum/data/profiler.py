from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr

from ..common import json_safe, load_csv
from ..session import ensure_session, get_journal, log_event, save_artifact, set_session_goal
from .contracts import build_data_contract


def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    candidates = df.select_dtypes(include=["object", "string"]).columns
    datetime_cols = []
    for col in candidates:
        sample = df[col].dropna().head(50)
        if sample.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= .8:
            datetime_cols.append(str(col))
    return datetime_cols


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    if table.empty:
        return 0.0
    chi2, _, _, _ = chi2_contingency(table)
    n = table.sum().sum()
    r, k = table.shape
    if n == 0 or min(r, k) <= 1:
        return 0.0
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def _encoded_binary(series: pd.Series) -> pd.Series:
    codes, _ = pd.factorize(series, sort=True)
    return pd.Series(codes, index=series.index, dtype=float)


def profile_target_relationship(df: pd.DataFrame, target: str) -> list[dict]:
    results = []
    target_series = df[target]
    is_binary_target = target_series.nunique(dropna=True) == 2

    for col in df.columns:
        if col == target:
            continue
        paired = pd.concat([df[col], target_series], axis=1).dropna()
        if len(paired) < 3:
            continue
        feature = paired.iloc[:, 0]
        aligned_target = paired.iloc[:, 1]

        try:
            if pd.api.types.is_numeric_dtype(feature):
                clean_feature = pd.to_numeric(feature, errors="coerce")
                valid = clean_feature.notna()
                clean_feature = clean_feature[valid]
                clean_target = aligned_target[valid]
                if len(clean_feature) < 3 or clean_feature.nunique(dropna=True) <= 1:
                    continue
                if is_binary_target:
                    stat, _ = pointbiserialr(_encoded_binary(clean_target), clean_feature)
                    method = "point_biserial"
                elif pd.api.types.is_numeric_dtype(clean_target):
                    stat = clean_feature.corr(pd.to_numeric(clean_target, errors="coerce"))
                    method = "pearson"
                else:
                    continue
            else:
                if feature.nunique(dropna=True) > 50:
                    continue
                stat = _cramers_v(feature.astype(str), aligned_target.astype(str))
                method = "cramers_v"
            results.append({
                "column": str(col),
                "method": method,
                "statistic": round(float(stat), 6) if pd.notna(stat) else None,
                "abs_statistic": round(abs(float(stat)), 6) if pd.notna(stat) else None,
            })
        except Exception:
            continue

    results.sort(key=lambda item: item["abs_statistic"] or 0, reverse=True)
    return results


def profile_missing_patterns(df: pd.DataFrame, max_cols: int = 20) -> dict:
    null_mask = df.isnull()
    null_cols = [col for col in df.columns if null_mask[col].any()][:max_cols]
    if not null_cols:
        return {
            "rows_with_any_null": 0,
            "rows_with_all_null": 0,
            "null_correlation_pairs": [],
            "structural_null_groups": [],
        }

    sub = null_mask[null_cols].astype(int)
    corr_matrix = sub.corr()
    pairs = []
    for a, b in itertools.combinations(null_cols, 2):
        value = corr_matrix.loc[a, b]
        if pd.notna(value) and abs(value) > .3:
            pairs.append({"col_a": str(a), "col_b": str(b), "r": round(float(value), 4)})
    pairs.sort(key=lambda item: abs(item["r"]), reverse=True)

    groups = []
    null_rows = sub[sub.sum(axis=1) >= 3]
    if not null_rows.empty:
        group_counts = null_rows.apply(tuple, axis=1).value_counts()
        for pattern, count in group_counts.head(5).items():
            cols_in_group = [str(null_cols[i]) for i, value in enumerate(pattern) if value]
            if len(cols_in_group) >= 2:
                groups.append({"columns": cols_in_group, "co_null_count": int(count)})

    return {
        "rows_with_any_null": int(null_mask.any(axis=1).sum()),
        "rows_with_all_null": int(null_mask.all(axis=1).sum()),
        "null_correlation_pairs": pairs[:10],
        "structural_null_groups": groups,
    }


def detect_constant_and_quasi_constant(
    df: pd.DataFrame, threshold: float = .99
) -> list[dict]:
    results = []
    for col in df.columns:
        counts = df[col].value_counts(normalize=True, dropna=False)
        if counts.empty:
            continue
        top_freq = float(counts.iloc[0])
        if top_freq >= threshold:
            top_val = df[col].value_counts(dropna=False).index[0]
            results.append({
                "column": str(col),
                "dominant_value": str(top_val),
                "frequency": round(top_freq, 4),
                "is_constant": top_freq == 1.0,
            })
    return results


def run_profile(
    file: str, target: str | None = None, sid: str | None = None, goal: str | None = None
) -> dict:
    df, error = load_csv(file, target)
    if error:
        return error
    question = goal.strip()[:1000] if goal and goal.strip() else ""
    sid, error = ensure_session(sid, file, target or "", question)
    if error:
        return error
    if question:
        set_session_goal(sid, question)
        prior_questions = get_journal(sid, event_type="analytical_question", limit=100)
        if not any(item["summary"] == question for item in prior_questions):
            log_event(
                sid, "user", "analytical_question", question,
                {"question": question},
            )
    columns = []
    for name in df.columns:
        series = df[name]
        item = {
            "name": str(name),
            "dtype": str(series.dtype),
            "null_pct": round(float(series.isna().mean() * 100), 4),
            "n_unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            std = float(clean.std()) if len(clean) > 1 else 0.0
            item.update({
                "mean": float(clean.mean()) if len(clean) else None,
                "std": std,
                "min": float(clean.min()) if len(clean) else None,
                "p25": float(clean.quantile(.25)) if len(clean) else None,
                "p50": float(clean.quantile(.5)) if len(clean) else None,
                "p75": float(clean.quantile(.75)) if len(clean) else None,
                "max": float(clean.max()) if len(clean) else None,
                "outlier_count": int((((clean - clean.mean()) / std).abs() > 3).sum()) if std else 0,
            })
        else:
            counts = series.fillna("<null>").astype(str).value_counts().head(5)
            item["top_values"] = [{"value": value, "count": int(count)} for value, count in counts.items()]
            if item["n_unique"] > 50:
                item["high_cardinality"] = True
        columns.append(item)
    numeric = df.select_dtypes(include=np.number)
    correlations = []
    if numeric.shape[1] >= 2:
        matrix = numeric.corr()
        for a, b in itertools.combinations(matrix.columns, 2):
            value = matrix.loc[a, b]
            if pd.notna(value):
                correlations.append({"col_a": str(a), "col_b": str(b), "r": float(value)})
        correlations.sort(key=lambda item: abs(item["r"]), reverse=True)
    contract = build_data_contract(df, target)
    save_artifact(sid, "data_contract", file, metadata={"contract": contract})
    result = {
        "sid": sid,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1e6), 4),
        "columns": columns,
        "top_correlations": correlations[:10],
        "datetime_columns": detect_datetime_columns(df),
        "missing_patterns": profile_missing_patterns(df),
        "quasi_constant_columns": detect_constant_and_quasi_constant(df),
        "data_contract": contract,
    }
    if target and target in df.columns:
        result["target_relationships"] = profile_target_relationship(df, target)
    return json_safe(result)
