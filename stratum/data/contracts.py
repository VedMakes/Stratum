from __future__ import annotations

import math
import warnings

import pandas as pd


MAX_CATEGORIES = 50


def _detect_datetime_columns(df: pd.DataFrame) -> set[str]:
    columns = set()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        sample = df[col].dropna().head(50)
        if sample.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= .8:
            columns.add(str(col))
    columns.update(
        str(col) for col in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[col])
    )
    return columns


def build_data_contract(
    df: pd.DataFrame, target: str | None = None, features: list[str] | None = None
) -> dict:
    selected = features or [col for col in df.columns if col != target]
    datetime_cols = _detect_datetime_columns(df)
    columns = {}
    for col in selected:
        if col not in df.columns:
            continue
        series = df[col]
        item = {
            "dtype": str(series.dtype),
            "null_rate": float(series.isna().mean()),
            "n_unique": int(series.nunique(dropna=True)),
            "is_datetime": str(col) in datetime_cols or pd.api.types.is_datetime64_any_dtype(series),
        }
        if pd.api.types.is_numeric_dtype(series):
            clean = pd.to_numeric(series, errors="coerce").dropna()
            item["numeric"] = {
                "min": float(clean.min()) if len(clean) else None,
                "max": float(clean.max()) if len(clean) else None,
                "mean": float(clean.mean()) if len(clean) else None,
                "std": float(clean.std()) if len(clean) > 1 else 0.0,
                "p01": float(clean.quantile(.01)) if len(clean) else None,
                "p99": float(clean.quantile(.99)) if len(clean) else None,
            }
        elif item["n_unique"] <= MAX_CATEGORIES:
            item["categories"] = sorted(series.dropna().astype(str).unique().tolist())
        columns[str(col)] = item
    return {
        "rows": int(len(df)),
        "target": target,
        "features": [str(col) for col in selected if col in df.columns],
        "columns": columns,
    }


def _dtype_family(dtype: str) -> str:
    text = dtype.lower()
    if any(token in text for token in ("int", "float", "double", "number")):
        return "numeric"
    if "bool" in text:
        return "bool"
    if "datetime" in text or "date" in text:
        return "datetime"
    return "categorical"


def _issue(severity: str, issue_type: str, column: str | None, message: str, **extra) -> dict:
    return {
        "severity": severity,
        "issue_type": issue_type,
        "column": column,
        "message": message,
        **extra,
    }


def compare_to_contract(
    contract: dict,
    df: pd.DataFrame,
    *,
    target: str | None = None,
    prediction_data: bool = True,
) -> dict:
    issues = []
    expected = contract.get("columns", {})
    expected_features = list(contract.get("features") or expected)
    missing = [col for col in expected_features if col not in df.columns]
    for col in missing:
        issues.append(_issue("blocker", "missing_column", col, f"Required column '{col}' is missing."))

    extra = [col for col in df.columns if col not in expected and col != target]
    for col in extra:
        issues.append(_issue("info", "extra_column", col, f"Column '{col}' was not in the reference contract."))

    contract_target = target or contract.get("target")
    if prediction_data and contract_target and contract_target in df.columns:
        issues.append(_issue(
            "warning", "target_present", contract_target,
            f"Target column '{contract_target}' is present in prediction data."
        ))

    for col, spec in expected.items():
        if col not in df.columns:
            continue
        series = df[col]
        actual_dtype = str(series.dtype)
        expected_family = _dtype_family(str(spec.get("dtype", "")))
        actual_family = _dtype_family(actual_dtype)
        if expected_family != actual_family:
            issues.append(_issue(
                "warning", "dtype_change", col,
                f"Column '{col}' changed from {spec.get('dtype')} to {actual_dtype}.",
                expected_dtype=spec.get("dtype"), actual_dtype=actual_dtype,
            ))

        expected_null = float(spec.get("null_rate") or 0.0)
        actual_null = float(series.isna().mean())
        if actual_null >= .2 and actual_null > expected_null + .15:
            issues.append(_issue(
                "warning", "null_rate_spike", col,
                f"Null rate rose from {expected_null:.1%} to {actual_null:.1%}.",
                expected_null_rate=expected_null, actual_null_rate=actual_null,
            ))

        if spec.get("is_datetime"):
            parsed = pd.to_datetime(series.dropna().head(100), errors="coerce")
            parse_rate = float(parsed.notna().mean()) if len(parsed) else 1.0
            if parse_rate < .8:
                issues.append(_issue(
                    "warning", "datetime_parse_change", col,
                    f"Datetime parse rate fell to {parse_rate:.1%}.",
                    parse_rate=parse_rate,
                ))

        numeric = spec.get("numeric")
        if numeric and pd.api.types.is_numeric_dtype(series):
            clean = pd.to_numeric(series, errors="coerce").dropna()
            if len(clean):
                expected_min, expected_max = numeric.get("min"), numeric.get("max")
                below = int((clean < expected_min).sum()) if expected_min is not None else 0
                above = int((clean > expected_max).sum()) if expected_max is not None else 0
                if below or above:
                    issues.append(_issue(
                        "warning", "numeric_range_violation", col,
                        f"{below + above} value(s) fall outside the reference min/max range.",
                        below_min=below, above_max=above,
                    ))

        categories = spec.get("categories")
        if categories is not None:
            seen = set(series.dropna().astype(str).unique())
            unseen = sorted(seen - set(categories))
            if unseen:
                issues.append(_issue(
                    "warning", "unseen_categories", col,
                    f"Column '{col}' has {len(unseen)} unseen categor(ies).",
                    unseen_categories=unseen[:20],
                ))

    blockers = sum(item["severity"] == "blocker" for item in issues)
    warnings = sum(item["severity"] == "warning" for item in issues)
    readiness = "blocked" if blockers else ("risky" if warnings else "pass")
    score = max(0, 100 - blockers * 30 - warnings * 10)
    if math.isnan(score):
        score = 0
    return {
        "readiness": readiness,
        "score": int(score),
        "issues": issues,
        "blocking_issues": [item for item in issues if item["severity"] == "blocker"],
        "warnings": [item for item in issues if item["severity"] == "warning"],
    }
