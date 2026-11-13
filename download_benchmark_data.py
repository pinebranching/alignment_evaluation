from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SAVE_DIR = Path("configs/benchmark_data")
DATASET_REPO_TYPE = "dataset"
SNAPSHOT_DOC_EXTENSIONS = {".md", ".pdf", ".rst", ".txt"}
SNAPSHOT_DOC_NAME_MARKERS = ("datasheet", "data_card", "datacard", "documentation", "readme")
# Hugging Face's basic checks validate split metadata without relying on
# dataset-card download URL keys, which can differ from the actual Hub URLs.
DEFAULT_VERIFICATION_MODE = "basic_checks"
LOCAL_DATASET_LOADERS = {
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "json",
    ".jsonl.gz": "json",
    ".parquet": "parquet",
}


def download_and_slice_benchmark(
    dataset_name: str,
    config_name: str | None = None,
    *,
    split: str | None = None,
    num_samples: int | None = None,
    save_dir: str | Path = DEFAULT_SAVE_DIR,
    revision: str | None = None,
    verification_mode: str = DEFAULT_VERIFICATION_MODE,
) -> tuple[Path, Path]:
    """Download one Hugging Face dataset split and persist it as local JSONL."""

    if not dataset_name.strip():
        raise ValueError("dataset_name must not be empty.")
    if split is not None and not split.strip():
        raise ValueError("split must not be empty.")
    if num_samples is not None and num_samples == 0:
        raise ValueError("num_samples must be > 0 when provided, or negative for full download.")

    print(f"[dataset] Processing: {dataset_name}")
    save_path_root = Path(save_dir).expanduser().resolve()
    save_path_root.mkdir(parents=True, exist_ok=True)
    print(f"[dataset] Saving artifacts under: {save_path_root}")

    builder_metadata: dict[str, Any] | None = None
    if split is None:
        builder_metadata = _collect_builder_metadata(
            dataset_name=dataset_name,
            config_name=config_name,
            revision=revision,
        )
        resolved_split = _resolve_split_selection(split, builder_metadata)
    else:
        resolved_split = split
    dataset = _load_dataset(
        dataset_name=dataset_name,
        config_name=config_name,
        split=resolved_split,
        revision=revision,
        verification_mode=verification_mode,
    )
    if builder_metadata is None:
        builder_metadata = _collect_builder_metadata_if_available(
            dataset_name=dataset_name,
            config_name=config_name,
            revision=revision,
        )
    if builder_metadata is None:
        builder_metadata = _build_runtime_builder_metadata(
            dataset_name=dataset_name,
            config_name=config_name,
            split=resolved_split,
            dataset=dataset,
        )

    if num_samples is None or num_samples < 0:
        selected_dataset = dataset
        sample_desc = "all"
    else:
        selected_count = min(num_samples, len(dataset))
        selected_dataset = dataset.select(range(selected_count))
        sample_desc = str(selected_count)

    record_count = len(selected_dataset)
    print(f"[dataset] Selected {record_count} samples from split '{resolved_split}'.")

    output_stem = _build_output_stem(
        dataset_name=dataset_name,
        config_name=config_name,
        split=resolved_split,
        sample_desc=sample_desc,
    )
    data_path = save_path_root / f"{output_stem}.jsonl"
    manifest_path = save_path_root / f"{output_stem}.manifest.json"
    docs_dir = save_path_root / f"{output_stem}.docs"
    documentation_snapshot = _snapshot_source_documentation(
        dataset_name=dataset_name,
        revision=revision,
        docs_dir=docs_dir,
    )
    datasheet_path = save_path_root / f"{output_stem}.datasheet.md"

    selected_dataset.to_json(str(data_path))
    manifest_payload = {
        "dataset_name": dataset_name,
        "config_name": config_name,
        "split": resolved_split,
        "revision": revision,
        "resolved_revision": documentation_snapshot.get("resolved_revision"),
        "num_samples": num_samples,
        "saved_records": record_count,
        "data_path": str(data_path),
        "manifest_path": str(manifest_path),
        "datasheet_path": str(datasheet_path),
        "docs_dir": str(docs_dir),
        "documentation": {
            "dataset_card_path": documentation_snapshot.get("dataset_card_path"),
            "supporting_doc_paths": documentation_snapshot.get("supporting_doc_paths", []),
            "card_metadata": documentation_snapshot.get("card_metadata"),
            "snapshot_warnings": documentation_snapshot.get("warnings", []),
        },
        "builder_metadata": builder_metadata,
        "saved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    datasheet_path.write_text(
        _build_local_datasheet(
            manifest_payload=manifest_payload,
            builder_metadata=builder_metadata,
            documentation_snapshot=documentation_snapshot,
        ),
        encoding="utf-8",
    )

    print(f"[dataset] Saved JSONL: {data_path}")
    print(f"[dataset] Saved manifest: {manifest_path}")
    print(f"[dataset] Saved local dataset card: {datasheet_path}")
    return data_path, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset split and persist it as local JSONL."
    )
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset id, e.g. TIGER-Lab/MMLU-Pro.")
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional dataset config/subset name, e.g. gpqa_diamond.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help=(
            "Optional dataset split to download. "
            "If omitted, the downloader only auto-selects when the dataset exposes exactly one split."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional slice size. Omit or pass a negative value to save the full split.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help=f"Directory for downloaded benchmark files. Defaults to '{DEFAULT_SAVE_DIR.as_posix()}'.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional dataset revision, tag, branch, or commit for reproducible downloads.",
    )
    parser.add_argument(
        "--verification-mode",
        default=DEFAULT_VERIFICATION_MODE,
        choices=("all_checks", "basic_checks", "no_checks"),
        help=(
            "Hugging Face dataset verification mode. "
            "Use 'no_checks' only as a workaround for broken upstream split metadata."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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


def _load_dataset(
    *,
    dataset_name: str,
    config_name: str | None,
    split: str,
    revision: str | None,
    verification_mode: str,
) -> Any:
    try:
        from datasets.exceptions import ExpectedMoreSplitsError
    except ImportError as exc:  # pragma: no cover - depends on optional runtime install
        raise RuntimeError(
            "The 'datasets' package is required for benchmark downloads. Install it in the active environment."
        ) from exc

    try:
        return _hf_load_dataset(
            dataset_name,
            name=config_name,
            split=split,
            revision=revision,
            verification_mode=verification_mode,
        )
    except ExpectedMoreSplitsError as exc:
        raise RuntimeError(
            f"Hugging Face dataset '{dataset_name}' reports split metadata that does not match the downloaded files. "
            f"You requested split '{split}', but the upstream builder verification failed: {exc}. "
            "If you trust the dataset and want to bypass this upstream inconsistency, retry with "
            "--verification-mode no_checks."
        ) from exc
    except RuntimeError as exc:
        if not _is_unsupported_dataset_script_error(exc):
            raise
        return _load_dataset_from_repo_files(
            dataset_name=dataset_name,
            config_name=config_name,
            split=split,
            revision=revision,
        )
    except ValueError as exc:
        if not _is_unknown_split_error(exc, split):
            raise
        return _load_dataset_from_repo_files(
            dataset_name=dataset_name,
            config_name=config_name,
            split=split,
            revision=revision,
        )


def _resolve_split_selection(
    requested_split: str | None,
    builder_metadata: dict[str, Any],
) -> str:
    """Resolve the effective split or raise a clear error when omission is ambiguous."""

    if requested_split is not None:
        return requested_split

    splits_payload = builder_metadata.get("splits")
    if not isinstance(splits_payload, dict) or not splits_payload:
        raise ValueError(
            "This dataset exposes no split metadata. Pass --split explicitly."
        )

    split_names = sorted(str(name) for name in splits_payload)
    if len(split_names) == 1:
        return split_names[0]

    raise ValueError(
        "This dataset exposes multiple splits. Pass --split explicitly. "
        f"Available splits: {', '.join(split_names)}"
    )


def _collect_builder_metadata(
    *,
    dataset_name: str,
    config_name: str | None,
    revision: str | None,
) -> dict[str, Any]:
    try:
        from datasets import load_dataset_builder
    except ImportError as exc:  # pragma: no cover - depends on optional runtime install
        raise RuntimeError(
            "The 'datasets' package is required for benchmark downloads. Install it in the active environment."
        ) from exc

    builder = load_dataset_builder(dataset_name, name=config_name, revision=revision)
    info = builder.info
    metadata: dict[str, Any] = {
        "builder_name": getattr(builder, "builder_name", None),
        "config_name": getattr(getattr(builder, "config", None), "name", None) or config_name,
        "version": str(getattr(info, "version", "")) or None,
        "description": getattr(info, "description", None),
        "homepage": getattr(info, "homepage", None),
        "license": getattr(info, "license", None),
        "citation": getattr(info, "citation", None),
        "features": _serialize_features(getattr(info, "features", None)),
        "splits": _serialize_splits(getattr(info, "splits", None)),
        "download_size": getattr(info, "download_size", None),
        "dataset_size": getattr(info, "dataset_size", None),
        "post_processing_size": getattr(info, "post_processing_size", None),
    }
    return metadata


def _collect_builder_metadata_if_available(
    *,
    dataset_name: str,
    config_name: str | None,
    revision: str | None,
) -> dict[str, Any] | None:
    try:
        return _collect_builder_metadata(
            dataset_name=dataset_name,
            config_name=config_name,
            revision=revision,
        )
    except RuntimeError as exc:
        if _is_unsupported_dataset_script_error(exc):
            return None
        raise


def _build_runtime_builder_metadata(
    *,
    dataset_name: str,
    config_name: str | None,
    split: str,
    dataset: Any,
) -> dict[str, Any]:
    return {
        "builder_name": "runtime_fallback",
        "config_name": config_name,
        "version": None,
        "description": (
            "Builder metadata was inferred from the loaded dataset at runtime because the "
            "upstream builder metadata was not resolved before loading."
        ),
        "homepage": f"https://huggingface.co/datasets/{dataset_name}",
        "license": None,
        "citation": None,
        "features": _serialize_features(getattr(dataset, "features", None)),
        "splits": {
            split: {
                "name": split,
                "num_examples": len(dataset),
                "num_bytes": None,
            }
        },
        "download_size": None,
        "dataset_size": None,
        "post_processing_size": None,
    }


def _hf_load_dataset(
    path: str,
    *,
    name: str | None = None,
    split: str,
    revision: str | None,
    verification_mode: str,
    data_files: dict[str, str] | None = None,
) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional runtime install
        raise RuntimeError(
            "The 'datasets' package is required for benchmark downloads. Install it in the active environment."
        ) from exc

    kwargs: dict[str, Any] = {
        "split": split,
        "verification_mode": verification_mode,
    }
    if data_files is not None:
        kwargs["data_files"] = data_files
    else:
        kwargs["name"] = name
        kwargs["revision"] = revision

    return load_dataset(path, **kwargs)


def _load_dataset_from_repo_files(
    *,
    dataset_name: str,
    config_name: str | None,
    split: str,
    revision: str | None,
) -> Any:
    downloaded_path = _download_repo_data_file(
        dataset_name=dataset_name,
        config_name=config_name,
        split=split,
        revision=revision,
    )
    loader_name = _infer_local_dataset_loader(downloaded_path)
    if loader_name is None:
        raise RuntimeError(
            f"Could not infer a local dataset loader for '{downloaded_path.name}'. "
            "Supported fallback extensions are: "
            + ", ".join(sorted(LOCAL_DATASET_LOADERS))
        )

    return _hf_load_dataset(
        loader_name,
        split=split,
        revision=None,
        verification_mode="no_checks",
        data_files={split: str(downloaded_path)},
    )


def _download_repo_data_file(
    *,
    dataset_name: str,
    config_name: str | None,
    split: str,
    revision: str | None,
) -> Path:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on optional runtime install
        raise RuntimeError(
            "The 'huggingface_hub' package is required for repo-file dataset fallback. "
            "Install it in the active environment."
        ) from exc

    api = HfApi()
    repo_files = api.list_repo_files(dataset_name, repo_type=DATASET_REPO_TYPE, revision=revision)
    repo_file = _select_repo_data_file(
        repo_files=repo_files,
        config_name=config_name,
        split=split,
    )
    downloaded = hf_hub_download(
        repo_id=dataset_name,
        filename=repo_file,
        repo_type=DATASET_REPO_TYPE,
        revision=revision,
    )
    return Path(downloaded)


def _infer_local_dataset_loader(path: Path) -> str | None:
    lowered_name = path.name.lower()
    for extension, loader_name in LOCAL_DATASET_LOADERS.items():
        if lowered_name.endswith(extension):
            return loader_name
    return None


def _select_repo_data_file(
    *,
    repo_files: list[str],
    config_name: str | None,
    split: str,
) -> str:
    normalized_split = split.lower()
    normalized_config = config_name.lower() if config_name else None
    candidates: list[tuple[int, str]] = []

    for file_path in repo_files:
        path = Path(file_path)
        loader_name = _infer_local_dataset_loader(path)
        if loader_name is None:
            continue

        score = 0
        lowered_parts = [part.lower() for part in path.parts]
        lowered_stem = path.stem.lower()

        if normalized_config:
            if lowered_parts and lowered_parts[0] == normalized_config:
                score += 4
            if lowered_parts and path.parent.name.lower() == normalized_config:
                score += 4
            if any(path.name.lower() == f"{normalized_config}{extension}" for extension in LOCAL_DATASET_LOADERS):
                score += 5
            if any(
                path.as_posix().lower() == f"data/{normalized_config}{extension}"
                for extension in LOCAL_DATASET_LOADERS
            ):
                score += 6
            if normalized_config not in lowered_parts and normalized_config not in lowered_stem:
                continue

        if normalized_split in lowered_stem or normalized_split in lowered_parts:
            score += 3
        elif normalized_split != "train":
            score -= 1

        if loader_name == "parquet":
            score += 2
        elif path.name.lower().endswith(".jsonl") or path.name.lower().endswith(".jsonl.gz"):
            score += 1

        candidates.append((score, file_path))

    if not candidates:
        config_display = config_name or "<default>"
        raise RuntimeError(
            f"Could not find a downloadable data file in '{candidates}' for config '{config_display}' "
            f"and split '{split}'."
        )

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def _is_unsupported_dataset_script_error(exc: RuntimeError) -> bool:
    return "Dataset scripts are no longer supported" in str(exc)


def _is_unknown_split_error(exc: ValueError, split: str) -> bool:
    message = str(exc)
    return f'Unknown split "{split}"' in message and "Should be one of" in message


def _snapshot_source_documentation(
    *,
    dataset_name: str,
    revision: str | None,
    docs_dir: Path,
) -> dict[str, Any]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    snapshot: dict[str, Any] = {
        "resolved_revision": None,
        "dataset_card_path": None,
        "supporting_doc_paths": [],
        "card_metadata": None,
        "card_markdown": None,
        "warnings": [],
    }

    try:
        from huggingface_hub import DatasetCard, HfApi, hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on optional runtime install
        snapshot["warnings"].append(
            "Could not import huggingface_hub; skipped dataset card snapshot."
        )
        return snapshot

    api = HfApi()
    try:
        repo_info = api.repo_info(dataset_name, repo_type=DATASET_REPO_TYPE, revision=revision)
        snapshot["resolved_revision"] = getattr(repo_info, "sha", None)
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        snapshot["warnings"].append(f"Could not resolve dataset revision: {exc}")

    try:
        readme_source_path = Path(
            hf_hub_download(
                repo_id=dataset_name,
                filename="README.md",
                repo_type=DATASET_REPO_TYPE,
                revision=revision,
            )
        )
        card = DatasetCard(readme_source_path.read_text(encoding="utf-8"))
        card_path = docs_dir / "README.md"
        card_path.write_text(card.content, encoding="utf-8")
        snapshot["dataset_card_path"] = str(card_path)
        snapshot["card_markdown"] = card.content
        snapshot["card_metadata"] = card.data.to_dict()
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        snapshot["warnings"].append(f"Could not snapshot dataset card README.md: {exc}")
        return snapshot

    try:
        repo_files = api.list_repo_files(
            dataset_name,
            repo_type=DATASET_REPO_TYPE,
            revision=revision,
        )
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        snapshot["warnings"].append(f"Could not list additional documentation files: {exc}")
        return snapshot

    for file_path in repo_files:
        if not _should_snapshot_supporting_doc(file_path):
            continue
        try:
            downloaded_path = Path(
                hf_hub_download(
                    repo_id=dataset_name,
                    filename=file_path,
                    repo_type=DATASET_REPO_TYPE,
                    revision=revision,
                )
            )
            target_path = docs_dir / file_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded_path, target_path)
            snapshot["supporting_doc_paths"].append(str(target_path))
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            snapshot["warnings"].append(f"Could not snapshot supporting doc '{file_path}': {exc}")

    return snapshot


def _serialize_features(features: Any) -> Any:
    if features is None:
        return None
    if hasattr(features, "to_dict"):
        return features.to_dict()
    return str(features)


def _serialize_splits(splits: Any) -> dict[str, Any] | None:
    if not splits:
        return None
    payload: dict[str, Any] = {}
    for split_name, split_info in dict(splits).items():
        payload[str(split_name)] = {
            "name": getattr(split_info, "name", None),
            "num_examples": getattr(split_info, "num_examples", None),
            "num_bytes": getattr(split_info, "num_bytes", None),
        }
    return payload


def _should_snapshot_supporting_doc(file_path: str) -> bool:
    path = Path(file_path)
    name = path.name.lower()
    if name == "readme.md":
        return False
    if path.suffix.lower() not in SNAPSHOT_DOC_EXTENSIONS:
        return False
    return any(marker in name for marker in SNAPSHOT_DOC_NAME_MARKERS)


def _build_local_datasheet(
    *,
    manifest_payload: dict[str, Any],
    builder_metadata: dict[str, Any],
    documentation_snapshot: dict[str, Any],
) -> str:
    source_lines = [
        "# Local Datasheet for Dataset",
        "",
        "This file is a local reproducibility artifact generated at download time.",
        "It snapshots the upstream Hugging Face dataset documentation when available,",
        "and adds local acquisition details for this exact saved benchmark file.",
        "",
        "## Acquisition",
        "",
        f"- Dataset id: `{manifest_payload['dataset_name']}`",
        f"- Dataset config: `{manifest_payload.get('config_name') or '<default>'}`",
        f"- Split: `{manifest_payload['split']}`",
        f"- Requested revision: `{manifest_payload.get('revision') or 'main/default'}`",
        f"- Resolved revision: `{manifest_payload.get('resolved_revision') or 'unknown'}`",
        f"- Requested sample limit: `{manifest_payload.get('num_samples')}`",
        f"- Saved records: `{manifest_payload['saved_records']}`",
        f"- Data path: `{manifest_payload['data_path']}`",
        f"- Manifest path: `{manifest_payload['manifest_path']}`",
        f"- Generated at (UTC): `{manifest_payload['saved_at_utc']}`",
        "",
        "## Builder Metadata",
        "",
        f"- Builder name: `{builder_metadata.get('builder_name') or 'unknown'}`",
        f"- Builder config: `{builder_metadata.get('config_name') or '<default>'}`",
        f"- Version: `{builder_metadata.get('version') or 'unknown'}`",
        f"- Homepage: `{builder_metadata.get('homepage') or 'unknown'}`",
        f"- License: `{builder_metadata.get('license') or 'unknown'}`",
        "",
    ]

    description = (builder_metadata.get("description") or "").strip()
    if description:
        source_lines.extend(
            [
                "### Description",
                "",
                description,
                "",
            ]
        )

    features_json = json.dumps(builder_metadata.get("features"), indent=2, ensure_ascii=True)
    source_lines.extend(
        [
            "### Features",
            "",
            "```json",
            features_json,
            "```",
            "",
        ]
    )

    splits_json = json.dumps(builder_metadata.get("splits"), indent=2, ensure_ascii=True)
    source_lines.extend(
        [
            "### Splits",
            "",
            "```json",
            splits_json,
            "```",
            "",
            "## Upstream Documentation Snapshot",
            "",
            f"- Dataset card path: `{documentation_snapshot.get('dataset_card_path') or 'not captured'}`",
        ]
    )

    supporting_docs = documentation_snapshot.get("supporting_doc_paths", [])
    if supporting_docs:
        source_lines.append("- Additional supporting docs:")
        source_lines.extend([f"  - `{path}`" for path in supporting_docs])
    else:
        source_lines.append("- Additional supporting docs: none captured")

    warnings = documentation_snapshot.get("warnings", [])
    if warnings:
        source_lines.extend(["", "## Snapshot Warnings", ""])
        source_lines.extend([f"- {warning}" for warning in warnings])
    
    # only need one of the two
    card_markdown = documentation_snapshot.get("card_markdown")
    card_metadata = documentation_snapshot.get("card_metadata")
    if card_markdown:
        formatted_card_markdown_lines: list[str] = []
        for line in card_markdown.strip().splitlines():
            stripped = line.strip()
            is_top_level_metadata_key = (
                bool(stripped)
                and line == stripped
                and stripped.endswith(":")
                and stripped[:-1].replace("_", "").replace("-", "").isalnum()
            )
            if (
                is_top_level_metadata_key
                and formatted_card_markdown_lines
                and formatted_card_markdown_lines[-1] != ""
            ):
                formatted_card_markdown_lines.append("")
            formatted_card_markdown_lines.append(line)

        source_lines.extend(
            [
                "",
                "## Dataset Card Metadata",
                "",
                "\n".join(formatted_card_markdown_lines),
                "",
            ]
        )
    elif card_metadata:
        source_lines.extend(
            [
                "",
                "### Dataset Card Metadata",
                "",
                "```json",
                json.dumps(card_metadata, indent=2, ensure_ascii=True),
                "```",
            ]
        )

    return "\n".join(source_lines).rstrip() + "\n \n"


def _build_output_stem(
    *,
    dataset_name: str,
    config_name: str | None,
    split: str,
    sample_desc: str,
) -> str:
    parts = [_sanitize_segment(dataset_name)]
    if config_name:
        parts.append(_sanitize_segment(config_name))
    parts.append(_sanitize_segment(split))
    parts.append(f"top_{sample_desc}")
    return "_".join(part for part in parts if part)


def _sanitize_segment(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    return sanitized.strip("_") or "dataset"


if __name__ == "__main__":
    raise SystemExit(main())
