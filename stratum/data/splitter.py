from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ..common import json_safe, load_csv
from ..session import ensure_session, save_artifact
from .profiler import detect_datetime_columns


SPLIT_MODES = {"stratified", "temporal", "random", "group", "predefined"}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_fingerprint(df: pd.DataFrame) -> str:
    row_hashes = pd.util.hash_pandas_object(df, index=True, categorize=True).values
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def _balance(series: pd.Series) -> dict | None:
    if series.nunique(dropna=True) > 20:
        return None
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _warn_on_missing_classes(
    source: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, target: str,
) -> list[str]:
    if source[target].nunique(dropna=True) > 20:
        return []
    expected = set(source[target].dropna().unique())
    warnings = []
    for label, subset in (("training", train), ("test", test)):
        missing = expected - set(subset[target].dropna().unique())
        if missing:
            warnings.append(f"{label.title()} split is missing target value(s): {sorted(map(str, missing))}.")
    return warnings


def run_split(
    file: str,
    target: str,
    mode: str = "stratified",
    time_col: str | None = None,
    test_size: float = 0.2,
    sid: str | None = None,
    seed: int = 42,
    group_col: str | None = None,
    split_col: str | None = None,
    train_value: str = "train",
    test_value: str = "test",
    gap_rows: int = 0,
) -> dict:
    df, error = load_csv(file, target)
    if error:
        return error
    if mode not in SPLIT_MODES:
        return {"error": f"mode must be one of: {', '.join(sorted(SPLIT_MODES))}"}
    if mode != "predefined" and not 0 < test_size < 1:
        return {"error": "test_size must be between 0 and 1"}
    if not isinstance(seed, int):
        return {"error": "seed must be an integer"}
    if not isinstance(gap_rows, int) or gap_rows < 0:
        return {"error": "gap_rows must be a non-negative integer"}

    sid, error = ensure_session(sid, file, target)
    if error:
        return error

    warnings = []
    parameters = {}
    if mode != "predefined":
        parameters["test_size"] = test_size
    if mode in {"random", "stratified", "group"}:
        parameters["seed"] = seed
    excluded_feature = None

    if mode == "temporal":
        if not time_col or time_col not in df:
            return {"error": "temporal mode requires a valid time_col"}
        parsed = pd.to_datetime(df[time_col], errors="coerce")
        parse_failures = parsed.isna() & df[time_col].notna()
        if parse_failures.any():
            return {"error": f"Column '{time_col}' contains {int(parse_failures.sum())} unparseable values"}
        if parsed.isna().any():
            warnings.append(
                f"time_col has {int(parsed.isna().sum())} null values. "
                "These rows will sort last and land in the test set."
            )
        ordered = df.assign(__stratum_time=parsed).sort_values("__stratum_time").drop(columns="__stratum_time")
        split_idx = int(len(ordered) * (1 - test_size))
        train_end = split_idx - gap_rows
        if train_end <= 0 or split_idx >= len(ordered):
            return {"error": "test_size and gap_rows leave an empty train or test set"}
        train, test = ordered.iloc[:train_end].copy(), ordered.iloc[split_idx:].copy()
        if gap_rows:
            warnings.append(f"Excluded {gap_rows} row(s) between train and test as a temporal gap.")
        if pd.to_datetime(test[time_col]).min() == pd.to_datetime(train[time_col]).max():
            warnings.append("Time column has ties at the split boundary.")
        parameters.update({"time_col": time_col, "gap_rows": gap_rows})
    elif mode == "group":
        if not group_col or group_col not in df:
            return {"error": "group mode requires a valid group_col"}
        if group_col == target:
            return {"error": "group_col cannot be the target column"}
        if df[group_col].isna().any():
            return {"error": f"Group column '{group_col}' contains null values"}
        if df[group_col].nunique() < 2:
            return {"error": "group mode requires at least two distinct groups"}
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        try:
            train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
        except ValueError as exc:
            return {"error": f"Unable to create group split: {exc}"}
        train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        excluded_feature = group_col
        parameters.update({"group_col": group_col, "group_feature_excluded": True})
        warnings.extend(_warn_on_missing_classes(df, train, test, target))
    elif mode == "predefined":
        if not split_col or split_col not in df:
            return {"error": "predefined mode requires a valid split_col"}
        if split_col in {target, group_col, time_col}:
            return {"error": "split_col must be distinct from target, group_col, and time_col"}
        if train_value == test_value:
            return {"error": "train_value and test_value must differ"}
        labels = df[split_col].astype("string")
        allowed = {str(train_value), str(test_value)}
        unknown = sorted(set(labels.dropna().unique()) - allowed)
        if labels.isna().any() or unknown:
            return {
                "error": f"split_col '{split_col}' must contain only {sorted(allowed)}; "
                f"found nulls={bool(labels.isna().any())}, other_values={unknown[:10]}"
            }
        train_mask, test_mask = labels == str(train_value), labels == str(test_value)
        if not train_mask.any() or not test_mask.any():
            return {"error": "predefined split must contain at least one train and one test row"}
        train = df.loc[train_mask].drop(columns=[split_col]).copy()
        test = df.loc[test_mask].drop(columns=[split_col]).copy()
        train_hashes = set(pd.util.hash_pandas_object(train, index=False).tolist())
        test_hashes = set(pd.util.hash_pandas_object(test, index=False).tolist())
        overlap_count = len(train_hashes & test_hashes)
        if overlap_count:
            return {
                "error": f"Predefined partitions contain {overlap_count} identical row value(s) across train and test."
            }
        parameters.update({
            "split_col": split_col,
            "train_value": str(train_value),
            "test_value": str(test_value),
            "overlapping_rows": overlap_count,
        })
        warnings.extend(_warn_on_missing_classes(df, train, test, target))
    else:
        stratify = None
        if mode == "stratified":
            counts = df[target].value_counts()
            if len(counts) > 1 and counts.min() >= 2:
                stratify = df[target]
            else:
                warnings.append("Stratification unavailable; used a deterministic random split.")
        train, test = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=stratify
        )
        parameters["stratify"] = stratify is not None
        if mode == "stratified":
            datetime_cols = [
                str(col) for col in df.columns
                if col != target and pd.api.types.is_datetime64_any_dtype(df[col])
            ]
            datetime_cols.extend(
                col for col in detect_datetime_columns(df)
                if col != target and col not in datetime_cols
            )
            if datetime_cols:
                warnings.append(f"Datetime column '{datetime_cols[0]}' found. Consider mode='temporal'.")

    if len(test) < 10:
        warnings.append(
            f"Test set has only {len(test)} rows. "
            "Evaluation metrics will be unreliable at this size."
        )
    if len(train) < 30:
        warnings.append(
            f"Training set has only {len(train)} rows. "
            "Model generalisation will be limited."
        )

    split_id = f"split_{uuid.uuid4().hex[:12]}"
    source = Path(file).expanduser().resolve()
    train_path = source.with_name(f"{source.stem}_{split_id}_train.csv")
    test_path = source.with_name(f"{source.stem}_{split_id}_test.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    manifest = {
        "split_id": split_id,
        "sid": sid,
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "target": target,
        "strategy": mode,
        "parameters": parameters,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_sha256": file_sha256(train_path),
        "test_sha256": file_sha256(test_path),
        "train_row_fingerprint": frame_fingerprint(train),
        "test_row_fingerprint": frame_fingerprint(test),
        "train_rows": len(train),
        "test_rows": len(test),
        "excluded_feature": excluded_feature,
    }
    save_artifact(
        sid, "split_train", str(train_path), metadata={"split_id": split_id, "sha256": manifest["train_sha256"]}
    )
    save_artifact(
        sid, "split_test", str(test_path), metadata={"split_id": split_id, "sha256": manifest["test_sha256"]}
    )
    save_artifact(sid, "split_manifest", str(source), metadata=manifest)
    return json_safe({
        "sid": sid,
        "split_id": split_id,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_rows": len(train),
        "test_rows": len(test),
        "class_balance_train": _balance(train[target]),
        "class_balance_test": _balance(test[target]),
        "split_mode": mode,
        "parameters": parameters,
        "excluded_feature": excluded_feature,
        "warnings": warnings,
    })
