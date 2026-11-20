from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stratum",
        description="Profile, audit, split, train, validate, report, and inspect local ML workflows.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser("profile", help="Profile a local CSV and save its data contract.")
    profile.add_argument("file")
    profile.add_argument("--target")
    profile.add_argument("--sid")

    audit = commands.add_parser("audit", help="Inspect a dataset for modelling risks.")
    audit.add_argument("file")
    audit.add_argument("--target", required=True)
    audit.add_argument("--sid")

    split = commands.add_parser("split", help="Create a persisted train/test split.")
    split.add_argument("file")
    split.add_argument("--target", required=True)
    split.add_argument("--mode", choices=["random", "stratified", "group", "temporal", "predefined"], default="stratified")
    split.add_argument("--time-col")
    split.add_argument("--group-col")
    split.add_argument("--split-col")
    split.add_argument("--test-size", type=float, default=.2)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--gap-rows", type=int, default=0)
    split.add_argument("--sid")

    train = commands.add_parser("train", help="Train a model and persist its run/provenance.")
    train.add_argument("file")
    train.add_argument("--target", required=True)
    train.add_argument("--model", default="auto")
    train.add_argument("--task", default="auto", choices=["auto", "classification", "regression"])
    train.add_argument("--sid")
    train.add_argument("--split-id")
    train.add_argument("--metric")
    train.add_argument("--metric-direction")

    validate = commands.add_parser("validate", help="Evaluate a saved model on labeled CSV data.")
    validate.add_argument("model_id")
    validate.add_argument("file")
    validate.add_argument("--target")
    validate.add_argument("--sid")

    report = commands.add_parser("report", help="Generate a JSON or HTML session report.")
    report.add_argument("--sid")
    report.add_argument("--file")
    report.add_argument("--format", choices=["json", "html"], default="json")
    report.add_argument("--output", help="Optional output file; HTML defaults to stdout.")

    artifacts = commands.add_parser("artifacts", help="List artifacts recorded for a session.")
    artifacts.add_argument("--sid")
    artifacts.add_argument("--file")
    return parser


def _execute(args: argparse.Namespace) -> dict:
    if args.command == "profile":
        from .data.profiler import run_profile

        return run_profile(args.file, target=args.target, sid=args.sid)
    if args.command == "audit":
        from .tools.tool_audit import run_audit

        return run_audit(args.file, args.target, sid=args.sid)
    if args.command == "split":
        from .data.splitter import run_split

        return run_split(
            args.file, args.target, mode=args.mode, time_col=args.time_col,
            group_col=args.group_col, split_col=args.split_col,
            test_size=args.test_size, seed=args.seed, gap_rows=args.gap_rows, sid=args.sid,
        )
    if args.command == "train":
        from .tools.tool_train import run_train

        return run_train(
            args.file, args.target, model=args.model, task=args.task, sid=args.sid,
            split_id=args.split_id, metric=args.metric, metric_direction=args.metric_direction,
        )
    if args.command == "validate":
        from .tools.tool_validate import run_validate

        return run_validate(args.model_id, args.file, target=args.target, sid=args.sid)
    if args.command == "report":
        from .tools.tool_report import run_report

        return run_report(sid=args.sid, file=args.file)
    if args.command == "artifacts":
        from .session import get_artifacts
        from .tools.tool_status import run_status

        status = run_status(sid=args.sid, file=args.file)
        if "error" in status:
            return status
        return {"sid": status["sid"], "artifacts": get_artifacts(status["sid"])}
    return {"error": f"Unknown command: {args.command}"}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _execute(args)
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}
    if args.command == "report" and args.format == "html" and "error" not in result:
        from .html_report import render_report_html

        output = render_report_html(result)
        if args.output:
            path = Path(args.output).expanduser()
            if path.exists():
                print(json.dumps({"error": f"Refusing to overwrite existing file: {path}"}))
                return 1
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(output, encoding="utf-8")
            print(json.dumps({"output_path": str(path)}))
        else:
            sys.stdout.write(output)
        return 0
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
