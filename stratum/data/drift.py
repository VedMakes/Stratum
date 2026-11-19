from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from ..common import json_safe, load_csv
from ..session import ensure_session


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


def run_drift(
    reference: str, current: str, target: str | None = None, sid: str | None = None
) -> dict:
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
            "psi": psi, "drifted": p_value < .05,
        })
    psi_values = [item["psi"] for item in results]
    psi_overall = float(np.mean(psi_values))
    psi_max = float(np.max(psi_values))
    psi_max_column = results[int(np.argmax(psi_values))]["name"]
    drift_fraction = sum(item["drifted"] for item in results) / len(results)
    verdict = "significant_drift" if psi_overall > .2 or psi_max > .5 or drift_fraction > .5 else (
        "mild_drift" if psi_overall >= .1 or psi_max > .2 or drift_fraction >= .2 else "no_drift"
    )
    return json_safe({
        "sid": sid, "overall_verdict": verdict, "psi_overall": psi_overall,
        "psi_max": psi_max, "psi_max_column": psi_max_column,
        "columns": results,
        "top_drifted": [item["name"] for item in sorted(results, key=lambda x: x["psi"], reverse=True)[:5]],
        "ref_rows": len(ref), "cur_rows": len(cur), "warnings": warnings,
        "note": "Target column excluded. Label shift detection requires separate analysis.",
    })
