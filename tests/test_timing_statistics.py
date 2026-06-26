from __future__ import annotations

import csv
import json
from pathlib import Path

from phonoflow.reporting.timing_statistics import write_calculation_time_statistics


def test_single_timing_statistics_writes_two_labeled_bars(tmp_path: Path) -> None:
    metadata = write_calculation_time_statistics(
        tmp_path,
        [
            {
                "model": "dpa31",
                "display_name": "DPA-3.1",
                "fc2_phonon_seconds": 12.5,
                "fc3_thermal_seconds": None,
                "thermal_reason": "Skipped because compute_kappa=false.",
            }
        ],
    )

    assert metadata["bar_count"] == 2
    assert metadata["value_label_count"] == 1
    assert metadata["thermal_reason"] == "Skipped because compute_kappa=false."
    assert (tmp_path / "calculation_time_statistics.png").exists()
    assert (tmp_path / "calculation_time_statistics.csv").exists()
    assert (tmp_path / "calculation_time_statistics.json").exists()
    rows = list(csv.DictReader((tmp_path / "calculation_time_statistics.csv").open(encoding="utf-8")))
    assert [row["stage"] for row in rows] == ["FC2 / phonon", "FC3 / thermal"]
    assert rows[1]["status"] == "skipped"


def test_compare_timing_statistics_writes_two_bars_per_model(tmp_path: Path) -> None:
    metadata = write_calculation_time_statistics(
        tmp_path,
        [
            {
                "model": "nep89",
                "display_name": "NEP89",
                "fc2_phonon_seconds": 10.0,
                "fc3_thermal_seconds": 20.0,
            },
            {
                "model": "dpa31",
                "display_name": "DPA-3.1",
                "fc2_phonon_seconds": 30.0,
                "fc3_thermal_seconds": 40.0,
            },
            {
                "model": "dpa4neo",
                "display_name": "DPA4-Neo",
                "fc2_phonon_seconds": 50.0,
                "fc3_thermal_seconds": 60.0,
            },
        ],
    )

    assert metadata["bar_count"] == 6
    assert metadata["value_label_count"] == 6
    payload = json.loads(
        (tmp_path / "calculation_time_statistics.json").read_text(encoding="utf-8")
    )
    assert len(payload["rows"]) == 6
