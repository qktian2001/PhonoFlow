"""JSON report writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phonoflow.constants import PROJECT_NAME, VERSION


def build_stability_report(backend: str, stability: dict[str, Any], notes: str | None = None) -> dict[str, Any]:
    """Build a stability report dictionary."""

    return {
        "project": PROJECT_NAME,
        "version": VERSION,
        "backend": backend,
        **stability,
        "notes": notes or "Dummy backend result for workflow testing only.",
    }


def write_stability_json(report: dict[str, Any], path: Path) -> None:
    """Write a JSON stability report."""

    write_json(report, path)


def write_json(report: dict[str, Any], path: Path) -> None:
    """Write a JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
