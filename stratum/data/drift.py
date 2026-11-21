from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency, ks_2samp

from ..common import json_safe, load_csv
from ..provenance import category_fingerprint
from ..session import ensure_session, save_artifact


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if not len(expected) or not len(actual) or np.all(expected == expected[0]):
        return 0.0
    edges = np.unique(np.percentile(expected, np.linspace(0, 100, buckets + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected_pct = np.histogram(expected, edges)[0] / len(expected)
    actual_pct = np.histogram(actual, edges)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    categories = sorted(set(expected.astype(str)) | set(actual.astype(str)))
    exp = expected.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    act = actual.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    exp, act = np.clip(exp, 1e-6, None), np.clip(act, 1e-6, None)
    return float(np.sum((act - exp) * np.log(act / exp)))


def _cramers_v_from_table(table: np.ndarray) -> float:
    chi2, _, _, _ = chi2_contingency(table)
    n = table.sum()
    r, k = table.shape
    if n == 0 or min(r, k) <= 1:
        return 0.0
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def _validate_thresholds(psi_mild: float, psi_significant: float, psi_high: float, p_value: float) -> str | None:
    if not 0 < psi_mild < psi_significant < psi_high:
        return "PSI thresholds must satisfy 0 < mild < significant < high."
    if not 0 < p_value < 1:
        return "p_value_threshold must be between 0 and 1."
    return None


def _finalize_drift(
    result: dict,
    *,
    psi_mild: float,
    psi_significant: float,
    psi_high: float,
    p_value_threshold: float,
) -> dict:
    actionable = []
    for item in result["columns"]:
        psi = float(item.get("psi") or 0)
        p_value = item.get("p_value")
        missing = bool(item.get("missing_feature"))
        item["drifted"] = missing or psi >= psi_mild or (
            p_value is not None and p_value < p_value_threshold
        )
        if missing:
            severity, action = "critical", "Restore the required feature or retrain with an explicitly revised feature set."
        elif psi >= psi_high:
            severity, action = "critical", "Investigate the shift and assess model validity before relying on predictions."
        elif psi >= psi_significant:
            severity, action = "high", "Review this feature's changed distribution and validate model performance on current labeled data."
        elif item["drifted"]:
            severity, action = "watch", "Review the shift; use labeled validation before changing or retraining the model."
        else:
            severity, action = "none", None
        item["severity"] = severity
        item["recommended_action"] = action
        if item["drifted"]:
            actionable.append({
                "column": item["name"],
                "severity": severity,
                "psi": psi,
                "p_value": p_value,
                "action": action,
            })
    psi_values = [float(item.get("psi") or 0) for item in result["columns"]]
    result["psi_overall"] = float(np.mean(psi_values)) if psi_values else 0.0
    result["psi_max"] = float(np.max(psi_values)) if psi_values else 0.0
    result["psi_max_column"] = (
        result["columns"][int(np.argmax(psi_values))]["name"] if psi_values else None
    )
    drift_fraction = (
        sum(bool(item["drifted"]) for item in result["columns"]) / len(result["columns"])
        if result["columns"] else 0.0
    )
    result["overall_verdict"] = (
        "significant_drift"
        if result["psi_overall"] >= psi_significant or result["psi_max"] >= psi_high or drift_fraction > .5
        else "mild_drift"
        if result["psi_overall"] >= psi_mild or result["psi_max"] >= psi_significant or drift_fraction >= .2
        else "no_drift"
    )
    result["top_drifted"] = [
        item["name"] for item in sorted(result["columns"], key=lambda item: item.get("psi", 0), reverse=True)[:5]
    ]
    result["actionable_findings"] = actionable
    result["thresholds"] = {
        "psi_mild": psi_mild,
        "psi_significant": psi_significant,
        "psi_high": psi_high,
        "p_value": p_value_threshold,
    }
    result.setdefault("warnings", []).append(
        "Per-feature statistical p-values are unadjusted for multiple comparisons; interpret them with effect sizes and domain context."
    )
    return result


def _metric_from_counts(expected: list[int], actual: list[int]) -> tuple[float, float | None, float | None]:
    expected_array = np.asarray(expected, dtype=float)
    actual_array = np.asarray(actual, dtype=float)
    keep = (expected_array + actual_array) > 0
    expected_array, actual_array = expected_array[keep], actual_array[keep]
    if not len(expected_array):
        return 0.0, None, 0.0
    exp_pct = np.clip(expected_array / expected_array.sum(), 1e-6, None)
    act_pct = np.clip(actual_array / actual_array.sum(), 1e-6, None)
    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    if len(expected_array) < 2 or np.any(expected_array == 0):
        return psi, None, None
    table = np.vstack([expected_array, actual_array])
    chi2, p_value, _, _ = chi2_contingency(table)
    n = table.sum()
    cramers_v = float(np.sqrt(chi2 / n)) if n else 0.0
    return psi, float(p_value), cramers_v


def _profile_column_drift(name: str, profile: dict, series: pd.Series) -> tuple[dict, list[str]]:
    warnings = []
    if profile.get("type") == "numeric":
        numeric = pd.to_numeric(series, errors="coerce")
        parse_failures = int((numeric.isna() & series.notna()).sum())
        if parse_failures:
            warnings.append(f"Column '{name}' has {parse_failures} non-numeric values in current data.")
        clean = numeric.dropna().to_numpy(dtype=float)
        boundaries = profile.get("boundaries", [])
        current_counts = np.bincount(
            np.searchsorted(boundaries, clean, side="right"),
            minlength=len(profile.get("bin_counts", [])),
        ).astype(int).tolist()
        current_counts.append(int(numeric.isna().sum()))
        expected_counts = list(profile.get("bin_counts", [])) + [int(profile.get("missing_count", 0))]
        psi, p_value, effect = _metric_from_counts(expected_counts, current_counts)
        test_name = "quantile_binned_chi2"
    else:
        values = series.dropna().astype(str)
        expected_map = profile.get("category_counts", {})
        other_key = profile.get("other_key", "__stratum_other__")
        current_map: dict[str, int] = {}
        unseen = 0
        for value, count in values.value_counts().items():
            key = category_fingerprint(value)
            if key in expected_map:
                current_map[key] = int(count)
            else:
                unseen += int(count)
        if other_key in expected_map:
            current_map[other_key] = current_map.get(other_key, 0) + unseen
        else:
            expected_map = {**expected_map, other_key: int(profile.get("other_count", 0))}
            current_map[other_key] = unseen
        keys = list(expected_map)
        expected_counts = [int(expected_map[key]) for key in keys] + [int(profile.get("missing_count", 0))]
        current_counts = [int(current_map.get(key, 0)) for key in keys] + [int(series.isna().sum())]
        psi, p_value, effect = _metric_from_counts(expected_counts, current_counts)
        test_name = "binned_chi2"
        if unseen:
            warnings.append(f"Column '{name}' contains {unseen} values not seen in the stored reference categories or rare-value bucket.")
    return ({
        "name": name,
        "dtype": str(series.dtype),
        "test": test_name,
        "statistic": None,
        "p_value": p_value,
        "cramers_v": effect,
        "psi": psi,
    }, warnings)


def run_drift_profile(
    reference_profile: dict,
    current: str,
    *,
    sid: str,
    model_id: str | None = None,
    psi_mild_threshold: float = .1,
    psi_significant_threshold: float = .2,
    psi_high_threshold: float = .5,
    p_value_threshold: float = .05,
) -> dict:
    error = _validate_thresholds(
        psi_mild_threshold, psi_significant_threshold, psi_high_threshold, p_value_threshold
    )
    if error:
        return {"error": error}
    cur, error = load_csv(current)
    if error:
        return error
    columns = []
    warnings = []
    reference_rows = int(reference_profile.get("rows", 0))
    size_ratio = min(reference_rows, len(cur)) / max(reference_rows, len(cur)) if reference_rows else 0
    if size_ratio < .1:
        warnings.append(
            f"Reference and current row counts differ substantially ({reference_rows} vs {len(cur)}); statistical power may differ."
        )
    profiles = reference_profile.get("features", {})
    for name, profile in profiles.items():
        if name not in cur:
            columns.append({
                "name": name, "dtype": None, "test": "schema", "statistic": None,
                "p_value": 0.0, "cramers_v": None, "psi": 1.0, "missing_feature": True,
            })
            continue
        result, feature_warnings = _profile_column_drift(name, profile, cur[name])
        columns.append(result)
        warnings.extend(feature_warnings)
    if not columns:
        return {"error": "The trained run has no saved feature distributions for drift monitoring."}
    result = _finalize_drift({
        "sid": sid,
        "model_id": model_id,
        "reference_kind": "training_time_feature_summary",
        "overall_verdict": "no_drift",
        "psi_overall": 0.0,
        "psi_max": 0.0,
        "psi_max_column": None,
        "columns": columns,
        "ref_rows": reference_rows,
        "cur_rows": len(cur),
        "warnings": warnings,
        "note": "Input feature drift only. Target/label shift is not assessed; use labeled validation for model-performance conclusions.",
    }, psi_mild=psi_mild_threshold, psi_significant=psi_significant_threshold,
        psi_high=psi_high_threshold, p_value_threshold=p_value_threshold)
    result["current_file"] = str(Path(current).expanduser().resolve())
    result["artifact_id"] = save_artifact(
        sid, "drift_monitoring", current, model_id=model_id,
        metadata={**result, "artifact_schema_version": 1},
    )
    return json_safe(result)


def run_drift(
    reference: str,
    current: str,
    target: str | None = None,
    sid: str | None = None,
    psi_mild_threshold: float = .1,
    psi_significant_threshold: float = .2,
    psi_high_threshold: float = .5,
    p_value_threshold: float = .05,
) -> dict:
    error = _validate_thresholds(
        psi_mild_threshold, psi_significant_threshold, psi_high_threshold, p_value_threshold
    )
    if error:
        return {"error": error}
    ref, error = load_csv(reference)
    if error:
        return error
    cur, error = load_csv(current)
    if error:
        return error
    sid, error = ensure_session(sid, reference, target or "")
    if error:
        return error
    columns = [col for col in ref.columns if col in cur.columns and col != target]
    if not columns:
        return {"error": "Reference and current CSVs have no comparable feature columns"}
    warnings = []
    size_ratio = min(len(ref), len(cur)) / max(len(ref), len(cur))
    if size_ratio < .1:
        warnings.append(
            f"Dataset sizes differ significantly ({len(ref)} vs {len(cur)} rows). "
            "Statistical tests are underpowered; drift may be underreported."
        )
    results = []
    for col in columns:
        numeric = pd.api.types.is_numeric_dtype(ref[col]) and pd.api.types.is_numeric_dtype(cur[col])
        cramers_v = None
        if numeric:
            a, b = ref[col].dropna().astype(float), cur[col].dropna().astype(float)
            if not len(a) or not len(b):
                statistic, p_value, psi = 0.0, 1.0, 0.0
            else:
                test = ks_2samp(a, b)
                statistic, p_value, psi = float(test.statistic), float(test.pvalue), compute_psi(a, b)
            test_name = "ks"
        else:
            a, b = ref[col].fillna("<null>").astype(str), cur[col].fillna("<null>").astype(str)
            categories = sorted(set(a) | set(b))
            psi = _categorical_psi(a, b)
            if len(categories) < 2:
                statistic, p_value = 0.0, 1.0
                cramers_v = 0.0
            else:
                table = np.vstack([
                    a.value_counts().reindex(categories, fill_value=0),
                    b.value_counts().reindex(categories, fill_value=0),
                ])
                statistic, p_value = chi2_contingency(table)[:2]
                statistic, p_value = float(statistic), float(p_value)
                cramers_v = _cramers_v_from_table(table)
            test_name = "chi2"
        results.append({
            "name": str(col), "dtype": str(ref[col].dtype), "test": test_name,
            "statistic": statistic, "p_value": p_value, "cramers_v": cramers_v,
            "psi": psi, "drifted": False,
        })
    result = _finalize_drift({
        "sid": sid,
        "columns": results,
        "ref_rows": len(ref), "cur_rows": len(cur), "warnings": warnings,
        "note": "Input feature drift only. Target/label shift is not assessed; use labeled validation for model-performance conclusions.",
    }, psi_mild=psi_mild_threshold, psi_significant=psi_significant_threshold,
        psi_high=psi_high_threshold, p_value_threshold=p_value_threshold)
    result["reference_file"] = str(Path(reference).expanduser().resolve())
    result["current_file"] = str(Path(current).expanduser().resolve())
    result["artifact_id"] = save_artifact(
        sid, "drift_monitoring", current,
        metadata={**result, "artifact_schema_version": 1},
    )
    return json_safe(result)
