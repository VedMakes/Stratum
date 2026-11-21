from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from ..common import json_safe, load_csv
from ..session import ensure_session, log_event, save_artifact


MAX_FEATURES = 10
MAX_CATEGORIES = 20


def _cramers_v(table: np.ndarray) -> float:
    n = int(table.sum())
    if n == 0 or min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def _bh_adjust(p_values: list[float | None]) -> list[float | None]:
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    indexed.sort(key=lambda item: item[1])
    adjusted: list[float | None] = [None] * len(p_values)
    running = 1.0
    count = len(indexed)
    for rank in range(count, 0, -1):
        index, p_value = indexed[rank - 1]
        running = min(running, p_value * count / rank)
        adjusted[index] = float(min(1.0, running))
    return adjusted


def _cliffs_delta(positive: np.ndarray, other: np.ndarray) -> float:
    ranks = mannwhitneyu(positive, other, alternative="two-sided", method="asymptotic")
    return float(2 * ranks.statistic / (len(positive) * len(other)) - 1)


def _bootstrap_ci(values: np.ndarray, statistic, rng: np.random.Generator, samples: int) -> list[float] | None:
    if len(values) < 4:
        return None
    estimates = []
    for _ in range(samples):
        sample = values[rng.integers(0, len(values), len(values))]
        estimate = statistic(sample)
        if np.isfinite(estimate):
            estimates.append(float(estimate))
    if len(estimates) < samples * .8:
        return None
    return [float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))]


def _numeric_result(feature: pd.Series, positive: pd.Series,
                    rng: np.random.Generator, bootstrap_samples: int) -> tuple[dict, list[str]]:
    warnings = []
    x = pd.to_numeric(feature, errors="coerce")
    x = x.where(np.isfinite(x))
    valid = x.notna() & positive.notna()
    x = x[valid].to_numpy(dtype=float)
    y = positive[valid].to_numpy(dtype=bool)
    a, b = x[y], x[~y]
    if len(a) < 2 or len(b) < 2:
        return {"effect_size": None, "effect_name": "cliffs_delta", "ci_95": None,
                "p_value": None, "support": {"positive": len(a), "other": len(b)}}, [
                    "Fewer than two non-missing observations in an outcome group; inference omitted."
                ]
    delta = _cliffs_delta(a, b)
    p_value = float(mannwhitneyu(a, b, alternative="two-sided", method="auto").pvalue)

    def statistic(sample: np.ndarray) -> float:
        group = sample[:, 1].astype(bool)
        if not group.any() or group.all():
            return float("nan")
        return _cliffs_delta(sample[group, 0], sample[~group, 0])

    paired = np.column_stack([x, y.astype(float)])
    ci = _bootstrap_ci(paired, statistic, rng, bootstrap_samples)
    if ci is None:
        warnings.append("Bootstrap interval unavailable because the sample is too small or resamples lacked both groups.")
    if min(len(a), len(b)) < 10:
        warnings.append("At least one outcome group has fewer than 10 usable rows; estimates and p-values are unstable.")
    return {
        "effect_size": delta,
        "effect_name": "cliffs_delta",
        "effect_interpretation": "Probability that a positive-outcome row has a larger feature value, minus the reverse probability; positive means larger values are more common in the positive group.",
        "ci_95": ci,
        "p_value": p_value,
        "support": {"positive": len(a), "other": len(b)},
        "missing_rows_excluded": int((~valid).sum()),
    }, warnings


def _categorical_result(feature: pd.Series, positive: pd.Series,
                        rng: np.random.Generator, bootstrap_samples: int) -> tuple[dict, list[str]]:
    valid = feature.notna() & positive.notna()
    x = feature[valid].astype(str).to_numpy()
    y = positive[valid].to_numpy(dtype=bool)
    levels = list(pd.unique(x))
    table = np.array([[np.sum((x == level) & ~y), np.sum((x == level) & y)] for level in levels])
    warnings = []
    if not y.any() or y.all():
        return {
            "effect_size": None,
            "effect_name": "cramers_v",
            "effect_interpretation": "Association strength from 0 (none) to 1 (strong); it has no direction and does not imply causation.",
            "ci_95": None,
            "p_value": None,
            "support": {"positive": int(y.sum()), "other": int((~y).sum()), "categories": len(levels)},
            "missing_rows_excluded": int(feature.isna().sum() + positive.isna().sum()),
        }, ["Only one outcome group has non-missing feature values; inference omitted."]
    if len(levels) == 1:
        effect, p_value, ci = 0.0, 1.0, [0.0, 0.0]
    else:
        effect = _cramers_v(table)
        p_value = float(chi2_contingency(table, correction=False)[1])
        paired = np.column_stack([x, y.astype(int)])

        def statistic(sample: np.ndarray) -> float:
            sampled_levels = list(pd.unique(sample[:, 0]))
            sampled_y = sample[:, 1].astype(bool)
            if not sampled_y.any() or sampled_y.all():
                return float("nan")
            sampled_x = sample[:, 0]
            sampled_table = np.array([
                [np.sum((sampled_x == level) & ~sampled_y), np.sum((sampled_x == level) & sampled_y)]
                for level in sampled_levels
            ])
            return _cramers_v(sampled_table)

        ci = _bootstrap_ci(paired, statistic, rng, bootstrap_samples)
        expected = chi2_contingency(table, correction=False)[3]
        if (expected < 5).any():
            warnings.append("Some expected contingency counts are below 5; the chi-square p-value may be unreliable.")
        if min(np.bincount(y.astype(int), minlength=2)) < 10:
            warnings.append("At least one outcome group has fewer than 10 usable rows; estimates and p-values are unstable.")
    if ci is None:
        warnings.append("Bootstrap interval unavailable because the sample is too small.")
    return {
        "effect_size": float(effect),
        "effect_name": "cramers_v",
        "effect_interpretation": "Association strength from 0 (none) to 1 (strong); it has no direction and does not imply causation.",
        "ci_95": ci,
        "p_value": p_value,
        "support": {"positive": int(y.sum()), "other": int((~y).sum()), "categories": len(levels)},
        "missing_rows_excluded": int(feature.isna().sum() + positive.isna().sum()),
    }, warnings


def run_association(
    file: str,
    outcome: str,
    positive_label: str,
    features: list[str],
    sid: str | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict:
    if not isinstance(features, list) or not 1 <= len(features) <= MAX_FEATURES:
        return {"error": f"Select between 1 and {MAX_FEATURES} named features; broad automatic scans are not supported."}
    if len(set(features)) != len(features):
        return {"error": "features must not contain duplicates."}
    if bootstrap_samples < 100 or bootstrap_samples > 5000:
        return {"error": "bootstrap_samples must be between 100 and 5000."}
    df, error = load_csv(file, outcome)
    if error:
        return error
    missing_columns = [name for name in [*features] if name not in df.columns]
    if missing_columns:
        return {"error": f"Feature columns not found: {missing_columns}"}
    if outcome in features:
        return {"error": "The outcome column cannot also be analyzed as a feature."}
    classes = list(pd.unique(df[outcome].dropna()))
    if len(classes) != 2:
        return {"error": f"outcome must contain exactly two non-null values; found {len(classes)}."}
    numeric_outcome = (
        pd.api.types.is_numeric_dtype(df[outcome])
        and not pd.api.types.is_bool_dtype(df[outcome])
    )
    if numeric_outcome:
        try:
            numeric_label = float(positive_label)
        except (TypeError, ValueError):
            return {"error": f"positive_label '{positive_label}' does not match an outcome value."}
        label_matches = any(float(value) == numeric_label for value in classes)
    else:
        label_matches = any(str(value) == str(positive_label) for value in classes)
    if not label_matches:
        return {"error": f"positive_label '{positive_label}' does not match an outcome value."}
    input_rows = len(df)
    outcome_rows_excluded = int(df[outcome].isna().sum())
    df = df.loc[df[outcome].notna()].copy()
    positive_mask = (
        df[outcome] == numeric_label
        if numeric_outcome else df[outcome].astype("string") == str(positive_label)
    )
    if bootstrap_samples > 0 and int(positive_mask.sum()) < 2:
        return {"error": "Each outcome group needs at least two rows."}
    for name in features:
        if not pd.api.types.is_numeric_dtype(df[name]) and df[name].nunique(dropna=True) > MAX_CATEGORIES:
            return {"error": f"Feature '{name}' has more than {MAX_CATEGORIES} categories; select or group categories explicitly."}

    resolved_sid, error = ensure_session(sid, str(Path(file).expanduser().resolve()), outcome)
    if error:
        return error
    rng = np.random.default_rng(seed)
    positive = positive_mask
    results = []
    for name in features:
        if pd.api.types.is_numeric_dtype(df[name]) and not pd.api.types.is_bool_dtype(df[name]):
            stats, warnings = _numeric_result(df[name], positive, rng, bootstrap_samples)
            analysis = "numeric_vs_binary_outcome"
        else:
            stats, warnings = _categorical_result(df[name], positive, rng, bootstrap_samples)
            analysis = "categorical_vs_binary_outcome"
        results.append({"feature": name, "analysis": analysis, **stats, "warnings": warnings})
    adjusted = _bh_adjust([item["p_value"] for item in results])
    for item, value in zip(results, adjusted):
        item["p_value_bh"] = value
    warnings = [
        "Observational associations only; results do not establish causality or feature availability at prediction time.",
        "Intervals and tests are conditional on this CSV and the selected analysis; they do not account for study-design or sampling bias.",
    ]
    if len(df) < 50:
        warnings.append(f"Only {len(df)} rows are available; uncertainty and p-values should be treated as highly unstable.")
    metadata = {
        "outcome": outcome,
        "positive_label": str(positive_label),
        "features": features,
        "rows": input_rows,
        "outcome_rows_excluded": outcome_rows_excluded,
        "outcome_support": {
            str(positive_label): int(positive_mask.sum()),
            "other": int((~positive_mask).sum()),
        },
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "results": results,
        "warnings": warnings,
    }
    save_artifact(resolved_sid, "statistical_analysis", file, metadata=metadata)
    log_event(resolved_sid, "tool", "statistical_analysis", "Ran selected feature associations", {
        "outcome": outcome, "positive_label": str(positive_label), "features": features,
    })
    return json_safe({"sid": resolved_sid, **metadata})
