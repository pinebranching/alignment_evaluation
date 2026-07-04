from __future__ import annotations

import argparse
from pathlib import Path

from download_benchmark_data import (
    DEFAULT_SAVE_DIR,
    DEFAULT_VERIFICATION_MODE,
    download_and_slice_benchmark,
)

from .service import run_orchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Pipeline orchestrator from one master YAML config."
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--config",
        type=Path,
        help="Path to the master YAML config.",
    )
    mode_group.add_argument(
        "--dataset",
        help="Download one Hugging Face dataset split to a local JSONL file before pipeline execution.",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional Hugging Face dataset config/subset name used with --dataset.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help=(
            "Optional dataset split used with --dataset. "
            "If omitted, the downloader only auto-selects when the dataset exposes exactly one split."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional slice size used with --dataset. Omit or pass a negative value to save the full split.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help="Directory used with --dataset to store downloaded JSONL and manifest artifacts.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face dataset revision used with --dataset for reproducible downloads.",
    )
    parser.add_argument(
        "--verification-mode",
        default=DEFAULT_VERIFICATION_MODE,
        choices=("all_checks", "basic_checks", "no_checks"),
        help=(
            "Hugging Face dataset verification mode used with --dataset. "
            "Use 'no_checks' only as a workaround for broken upstream split metadata."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_mode_arguments(args)

    if args.dataset:
        download_and_slice_benchmark(
            dataset_name=args.dataset,
            config_name=args.dataset_config,
            split=args.split,
            num_samples=args.max_samples,
            save_dir=args.save_dir,
            revision=args.revision,
            verification_mode=args.verification_mode,
        )
        return 0

    state = run_orchestrator(args.config)
    print(f"Orchestrator run completed with status: {state['status']}")
    print(f"Pipeline state file: {Path(state['run_dir']) / 'pipeline_state.json'}")
    return 0


def _validate_mode_arguments(args: argparse.Namespace) -> None:
    if args.dataset:
        return

    if args.dataset_config is not None:
        raise SystemExit("--dataset-config can only be used together with --dataset.")
    if args.split is not None:
        raise SystemExit("--split can only be used together with --dataset.")
    if args.max_samples is not None:
        raise SystemExit("--max-samples can only be used together with --dataset.")
    if args.save_dir != DEFAULT_SAVE_DIR:
        raise SystemExit("--save-dir can only be used together with --dataset.")
    if args.revision is not None:
        raise SystemExit("--revision can only be used together with --dataset.")
    if args.verification_mode != DEFAULT_VERIFICATION_MODE:
        raise SystemExit("--verification-mode can only be used together with --dataset.")


if __name__ == "__main__":
    raise SystemExit(main())
