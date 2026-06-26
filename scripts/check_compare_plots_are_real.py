"""Validate that compare-models plots are real data plots, not status placeholders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


PLOT_FILES = [
    "comparison_phonon_band.png",
    "comparison_dos.png",
    "comparison_thermal_conductivity.png",
    "comparison_kappa_bar.png",
    "comparison_kappa.png",
]

PLOT_KEY_BY_FILE = {
    "comparison_phonon_band.png": "phonon_band",
    "comparison_dos.png": "dos",
    "comparison_thermal_conductivity.png": "thermal_conductivity",
    "comparison_kappa_bar.png": "kappa_bar",
    "comparison_kappa.png": "legacy_kappa",
}

EXPECTED_DISPLAY_NAMES = {
    "nep89": "NEP89",
    "dpa3": "DPA3",
    "dpa4": "DPA4",
}


def check_compare_plots(compare_dir: Path) -> dict[str, Any]:
    result_path = compare_dir / "comparison_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    plot_metadata = result.get("comparison_plots") or {}
    components = result.get("kappa_bar_components") or []
    model_rows = result.get("models", [])
    model_statuses = {row.get("model"): row.get("status") for row in model_rows}
    model_data_statuses = {
        row.get("model"): row.get("plot_data_availability") or {}
        for row in model_rows
    }
    all_success = bool(model_statuses) and all(status == "success" for status in model_statuses.values())
    checks = []
    overall_pass = True
    for name in PLOT_FILES:
        path = compare_dir / name
        metadata = plot_metadata.get(PLOT_KEY_BY_FILE[name], {})
        image_check = _image_check(path)
        kind = metadata.get("kind")
        is_data = kind == "data"
        plotted_models = metadata.get("models") or []
        expected_legend_labels = [
            EXPECTED_DISPLAY_NAMES.get(str(model), str(model).upper())
            for model in plotted_models
        ]
        reasons: list[str] = []
        if not image_check["exists"]:
            reasons.append("missing file")
        if image_check["exists"] and image_check["bytes"] < 5000:
            reasons.append("file too small")
        if image_check["exists"] and image_check["pixel_stddev_mean"] < 1.0:
            reasons.append("near-blank image")
        if kind in {"dry-run/status", "missing-data/status"}:
            reasons.append(f"metadata marks placeholder: {kind}")
        if name in {
            "comparison_phonon_band.png",
            "comparison_dos.png",
            "comparison_thermal_conductivity.png",
        }:
            legend_labels = metadata.get("legend_labels") or []
            if legend_labels != expected_legend_labels:
                reasons.append(
                    "legend is not exactly one entry per plotted model: "
                    f"expected {expected_legend_labels}, found {legend_labels}"
                )
            if metadata.get("legend_entry_count", len(legend_labels)) != len(plotted_models):
                reasons.append("legend entry count does not match plotted model count")
        required_source = _required_source_for_plot(name)
        if required_source is not None:
            for model in plotted_models:
                source_status = (model_data_statuses.get(model) or {}).get(required_source)
                if source_status != "data":
                    reasons.append(
                        f"{model} {required_source} source is {source_status or 'unrecorded'}, not data"
                    )
        if name in {"comparison_kappa_bar.png", "comparison_kappa.png"} and all_success and len(components) != 12:
            reasons.append(f"expected 12 kappa components for three successful models, found {len(components)}")
        if name in {"comparison_kappa_bar.png", "comparison_kappa.png"}:
            expected_bars = 4 * sum(status == "success" for status in model_statuses.values())
            if len(components) != expected_bars:
                reasons.append(
                    f"expected {expected_bars} kappa components for successful models, found {len(components)}"
                )
        if name == "comparison_kappa.png":
            canonical = compare_dir / "comparison_kappa_bar.png"
            if path.exists() and canonical.exists() and path.read_bytes() != canonical.read_bytes():
                reasons.append("legacy kappa alias differs from comparison_kappa_bar.png")
        passed = image_check["exists"] and not reasons and is_data
        if not passed:
            overall_pass = False
        checks.append(
            {
                "file": name,
                "passed": passed,
                "kind": kind,
                "models": plotted_models,
                "legend_labels": metadata.get("legend_labels", []),
                "source_statuses": {
                    model: model_data_statuses.get(model, {})
                    for model in plotted_models
                },
                "image": image_check,
                "reasons": reasons,
            }
        )
    return {
        "compare_dir": str(compare_dir),
        "passed": overall_pass,
        "all_models_success": all_success,
        "model_statuses": model_statuses,
        "model_data_statuses": model_data_statuses,
        "kappa_bar_component_count": len(components),
        "checks": checks,
    }


def _required_source_for_plot(name: str) -> str | None:
    if name == "comparison_phonon_band.png":
        return "band"
    if name == "comparison_dos.png":
        return "dos"
    if name == "comparison_thermal_conductivity.png":
        return "thermal"
    return None


def _image_check(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "bytes": 0, "pixel_stddev_mean": 0.0}
    with Image.open(path) as image:
        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        stddev = float(stat.stddev[0]) if stat.stddev else 0.0
        return {
            "exists": True,
            "bytes": path.stat().st_size,
            "width": image.width,
            "height": image.height,
            "pixel_stddev_mean": stddev,
        }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Compare Plot Reality Check",
        "",
        f"- Compare dir: `{report['compare_dir']}`",
        f"- Passed: `{report['passed']}`",
        f"- All models success: `{report['all_models_success']}`",
        f"- Kappa component count: `{report['kappa_bar_component_count']}`",
        "",
        "| file | passed | kind | models | legend | reasons |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["checks"]:
        lines.append(
            "| {file} | {passed} | {kind} | {models} | {legend} | {reasons} |".format(
                file=item["file"],
                passed=item["passed"],
                kind=item.get("kind"),
                models=", ".join(item.get("models") or []),
                legend=", ".join(item.get("legend_labels") or []),
                reasons="; ".join(item.get("reasons") or []),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-dir", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    args = parser.parse_args()

    report = check_compare_plots(args.compare_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(_markdown(report), encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
