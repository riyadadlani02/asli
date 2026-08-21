"""Offline command-line interface for scoring TurnBench JSONL inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .report import score_inputs
from .schema import (
    DecisionLabel,
    ProviderTrace,
    Recording,
    _decode_strict_json,
    read_jsonl,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="asli-turnbench")
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="score local TurnBench JSONL files")
    score.add_argument("--recordings", type=Path, required=True)
    score.add_argument("--labels", type=Path, required=True)
    score.add_argument("--events", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--provider", required=True)
    score.add_argument("--config-json", default="{}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Score one provider run using local, versioned JSONL inputs only."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = _decode_strict_json(args.config_json)
        if not isinstance(config, dict):
            raise ValueError("--config-json must decode to an object")
        recordings = read_jsonl(args.recordings, Recording.from_dict)
        labels = read_jsonl(args.labels, DecisionLabel.from_dict)
        traces = read_jsonl(args.events, ProviderTrace.from_dict)
        report = score_inputs(
            recordings,
            labels,
            traces,
            provider=args.provider,
            config=config,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    try:
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0
