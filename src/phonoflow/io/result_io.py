"""Generic result serialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def write_yaml(data: dict[str, Any], path: Path) -> None:
    """Write YAML data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def read_yaml(path: Path) -> dict[str, Any]:
    """Read YAML data."""

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
