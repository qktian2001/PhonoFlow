"""Calculation-stage timing artifacts for single and comparison workflows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def write_calculation_time_statistics(
    outdir: Path,
    model_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write grouped FC2/phonon and FC3/thermal timing artifacts."""

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    thermal_reasons: list[str] = []
    for model in model_rows:
        model_id = str(model.get("model") or "model")
        display_name = str(model.get("display_name") or model_id)
        for stage, key in (
            ("FC2 / phonon", "fc2_phonon_seconds"),
            ("FC3 / thermal", "fc3_thermal_seconds"),
        ):
            raw_value = model.get(key)
            value = None if raw_value is None else max(0.0, float(raw_value))
            reason = ""
            status = "completed"
            if value is None:
                status = "skipped"
                reason = str(
                    model.get("thermal_reason")
                    or "Skipped because compute_kappa=false."
                )
                if stage == "FC3 / thermal":
                    thermal_reasons.append(reason)
            rows.append(
                {
                    "model": model_id,
                    "display_name": display_name,
                    "stage": stage,
                    "seconds": value,
                    "status": status,
                    "reason": reason,
                }
            )

    csv_path = outdir / "calculation_time_statistics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "display_name", "stage", "seconds", "status", "reason"],
        )
        writer.writeheader()
        writer.writerows(rows)

    png_path = outdir / "calculation_time_statistics.png"
    value_label_count = _write_timing_plot(png_path, model_rows)
    metadata = {
        "available": bool(model_rows),
        "path": png_path.name,
        "csv_path": csv_path.name,
        "json_path": "calculation_time_statistics.json",
        "unit": "seconds",
        "bar_count": len(rows),
        "value_label_count": value_label_count,
        "model_count": len(model_rows),
        "thermal_reason": thermal_reasons[0] if thermal_reasons else None,
        "rows": rows,
    }
    (outdir / "calculation_time_statistics.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


def timing_row_from_breakdown(
    *,
    model: str,
    display_name: str,
    timing_breakdown: dict[str, Any] | None,
    compute_kappa: bool,
) -> dict[str, Any]:
    """Build one chart row from a persisted workflow timing breakdown."""

    stages = (timing_breakdown or {}).get("stages") or {}
    fc2 = _stage_seconds(stages, "fc2_harmonic") + _stage_seconds(
        stages, "phonon_postprocess"
    )
    fc3_value: float | None
    if compute_kappa:
        fc3_value = _stage_seconds(stages, "fc3") + _stage_seconds(
            stages, "thermal_lifetime"
        )
    else:
        fc3_value = None
    return {
        "model": model,
        "display_name": display_name,
        "fc2_phonon_seconds": fc2,
        "fc3_thermal_seconds": fc3_value,
        "thermal_reason": (
            None if compute_kappa else "Skipped because compute_kappa=false."
        ),
    }


def _stage_seconds(stages: dict[str, Any], key: str) -> float:
    value = (stages.get(key) or {}).get("seconds")
    return max(0.0, float(value or 0.0))


def _write_timing_plot(path: Path, model_rows: list[dict[str, Any]]) -> int:
    model_names = [str(row.get("display_name") or row.get("model") or "model") for row in model_rows]
    centers = np.arange(len(model_rows), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(max(7.5, 2.8 * max(len(model_rows), 1)), 4.8))
    label_count = 0
    if not model_rows:
        ax.text(0.5, 0.5, "No timing metadata available", ha="center", va="center")
        ax.set_axis_off()
    else:
        fc2_values = [max(0.0, float(row.get("fc2_phonon_seconds") or 0.0)) for row in model_rows]
        fc3_values = [
            np.nan if row.get("fc3_thermal_seconds") is None
            else max(0.0, float(row["fc3_thermal_seconds"]))
            for row in model_rows
        ]
        bars_fc2 = ax.bar(
            centers - width / 2,
            fc2_values,
            width,
            label="FC2 / phonon",
            color="#176B87",
        )
        bars_fc3 = ax.bar(
            centers + width / 2,
            fc3_values,
            width,
            label="FC3 / thermal",
            color="#E67E22",
            hatch="//",
        )
        for bars in (bars_fc2, bars_fc3):
            for bar in bars:
                height = float(bar.get_height())
                if not np.isfinite(height):
                    continue
                ax.annotate(
                    f"{height:.2f}",
                    (bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
                label_count += 1
        ax.set_xticks(centers, model_names)
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Calculation Time Statistics")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return label_count
