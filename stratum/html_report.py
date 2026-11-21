from __future__ import annotations

import html
import json
from typing import Any


def _json(value: Any) -> str:
    return html.escape(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _section(title: str, content: str) -> str:
    return f"<section><h2>{html.escape(title)}</h2>{content}</section>"


def _pre(value: Any) -> str:
    return f"<pre>{_json(value)}</pre>"


def render_report_html(report: dict) -> str:
    runs = report.get("runs") or []
    run_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(run.get('model_id', '')))}</td>"
        f"<td>{html.escape(str(run.get('model', '')))}</td>"
        f"<td>{html.escape(str(run.get('metric', '')))}</td>"
        f"<td>{html.escape(str(run.get('score', '')))}</td>"
        f"<td>{html.escape(str(run.get('split_strategy', '')))}</td>"
        "</tr>"
        for run in runs
    ) or '<tr><td colspan="5">No model runs recorded.</td></tr>'
    summary = {
        key: report.get(key)
        for key in ("sid", "dataset", "session_goal", "best_model", "current_metric", "current_score", "baseline_score", "improvement")
    }
    evidence = [
        f"<article><h3>{html.escape(str(run.get('model_id', 'Run')))}</h3>"
        f"<h4>Holdout metrics</h4>{_pre(run.get('metrics', {}))}"
        f"<h4>Feature importance and error evidence</h4>{_pre({'top_features': run.get('top_features', []), 'error_segments': run.get('decision_tree_summary', '')})}"
        f"<h4>Split and selection protocol</h4>{_pre({'split': (run.get('provenance') or {}).get('split', {}), 'selection': run.get('selection')})}"
        f"<h4>Change from previous run</h4>{_pre(run.get('change_from_previous'))}"
        f"<h4>Provenance</h4>{_pre(run.get('provenance', {}))}</article>"
        for run in runs
    ] or "<p>No run evidence recorded.</p>"
    table = (
        "<table><thead><tr><th>Run</th><th>Model</th><th>Primary metric</th>"
        "<th>Score</th><th>Split</th></tr></thead>"
        f"<tbody>{run_rows}</tbody></table>"
    )
    content = [
        _section("Session summary", _pre(summary)),
        _section("Question, assumptions, and journal", _pre({
            "session_goal": report.get("session_goal", ""),
            "question_history": report.get("question_history", []),
            "assumptions_and_caveats": report.get("assumptions_and_caveats", []),
            "decisions": report.get("decisions", []),
            "journal": report.get("journal", []),
        })),
        _section("Model comparison", table),
        _section("Data risks and assumptions", _pre({
            "flags": report.get("flags", []),
            "resolved_flags": report.get("resolved_flags", []),
            "decisions": report.get("decisions", []),
        })),
        _section("Diagnostic artifacts", _pre(report.get("diagnostics", {}))),
        _section("Run evaluation, explainability, and provenance", "".join(evidence)),
        _section("Recommended next actions", _pre(report.get("recommended_next_actions", []))),
    ]
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stratum session report</title><style>
body{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#172033;background:#f7f8fb}
h1,h2,h3{line-height:1.2}section,article{background:#fff;border:1px solid #dfe3eb;border-radius:8px;padding:1rem 1.25rem;margin:1rem 0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:.8rem;border-radius:5px}
table{width:100%;border-collapse:collapse}th,td{text-align:left;border-bottom:1px solid #dfe3eb;padding:.55rem}
th{background:#f3f5f8}article{margin:1rem 0}h4{margin-bottom:.3rem}
@media print{body{background:#fff;margin:0;max-width:none}section,article{break-inside:avoid}}
</style></head><body><h1>Stratum session report</h1>""" + "".join(content) + "</body></html>"
