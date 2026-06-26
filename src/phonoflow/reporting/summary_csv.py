"""Batch summary CSV writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


SUMMARY_COLUMNS = [
    "structure_name",
    "status",
    "outdir",
    "dynamically_stable",
    "minimum_frequency_THz",
    "error_message",
]


def write_summary_csv(results: list[dict[str, Any]], path: Path) -> None:
    """Write batch workflow summary CSV."""

    rows = [{column: result.get(column, "") for column in SUMMARY_COLUMNS} for result in results]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(path, index=False)
