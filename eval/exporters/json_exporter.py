from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_json_artifacts(output_dir: Path, summary: dict[str, Any], per_case_records: list[dict[str, Any]]) -> None:
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    per_case_path = output_dir / "per_case.jsonl"
    with per_case_path.open("w", encoding="utf-8") as handle:
        for record in per_case_records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
