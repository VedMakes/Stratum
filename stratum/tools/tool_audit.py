from __future__ import annotations

from ..audit import inspect_dataframe
from ..common import load_csv
from ..session import ensure_session, save_flag


def run_audit(file: str, target: str, sid: str | None = None) -> dict:
    df, error = load_csv(file, target)
    if error:
        return error
    sid, error = ensure_session(sid, file, target)
    if error:
        return error
    flags, stats = inspect_dataframe(df, target)
    for flag in flags:
        flag["flag_id"] = save_flag(sid, flag)
    high = sum(flag["confidence"] >= .85 for flag in flags)
    summary = (
        f"Found {len(flags)} potential data-quality issue(s), including {high} high-confidence item(s). "
        "Review leakage and temporal flags before training."
        if flags else "No configured data-quality risks were detected. The dataset is ready for baseline training."
    )
    return {"sid": sid, "flags": flags, "summary": summary, "dataset_stats": stats}
