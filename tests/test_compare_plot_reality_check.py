from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_compare_plots_are_real.py"
SPEC = importlib.util.spec_from_file_location("check_compare_plots_are_real", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (600, 400), color)
    pixels = image.load()
    for y in range(400):
        for x in range(600):
            pixels[x, y] = (
                (color[0] + x * 7 + y * 3) % 256,
                (color[1] + x * 5 + y * 11) % 256,
                (color[2] + x * 13 + y * 2) % 256,
            )
    image.save(path)


def _write_result(compare_dir: Path, *, bad_legend: bool = False) -> None:
    models = ["nep89", "dpa3"]
    labels = ["NEP89", "DPA3"]
    plots = {
        "phonon_band": {
            "kind": "data",
            "models": models,
            "legend_labels": labels if not bad_legend else ["NEP89", "NEP89", "DPA3"],
            "legend_entry_count": 2 if not bad_legend else 3,
        },
        "dos": {
            "kind": "data",
            "models": models,
            "legend_labels": labels,
            "legend_entry_count": 2,
        },
        "thermal_conductivity": {
            "kind": "data",
            "models": models,
            "legend_labels": labels,
        },
        "kappa_bar": {
            "kind": "data",
            "models": models,
            "successful_model_count": 2,
            "bar_count": 8,
        },
        "legacy_kappa": {
            "kind": "data",
            "models": models,
            "successful_model_count": 2,
            "bar_count": 8,
        },
    }
    result = {
        "models": [
            {
                "model": model,
                "status": "success",
                "plot_data_availability": {"band": "data", "dos": "data", "thermal": "data"},
            }
            for model in models
        ]
        + [
            {
                "model": "dpa4",
                "status": "failed",
                "plot_data_availability": {"band": "failed", "dos": "failed", "thermal": "failed"},
            }
        ],
        "kappa_bar_components": [
            {"model": model, "component": component}
            for model in models
            for component in ("kxx", "kyy", "kzz", "kavg")
        ],
        "comparison_plots": plots,
    }
    (compare_dir / "comparison_result.json").write_text(json.dumps(result), encoding="utf-8")


def test_reality_check_accepts_real_two_model_plots_with_failed_dpa4(tmp_path: Path) -> None:
    _write_result(tmp_path)
    for index, name in enumerate(MODULE.PLOT_FILES):
        _write_image(tmp_path / name, (30 + index * 20, 80, 140))
    (tmp_path / "comparison_kappa.png").write_bytes(
        (tmp_path / "comparison_kappa_bar.png").read_bytes()
    )

    report = MODULE.check_compare_plots(tmp_path)

    assert report["passed"] is True
    assert report["all_models_success"] is False
    assert report["kappa_bar_component_count"] == 8
    assert report["model_data_statuses"]["dpa4"]["band"] == "failed"


def test_reality_check_rejects_branch_level_legend_duplication(tmp_path: Path) -> None:
    _write_result(tmp_path, bad_legend=True)
    for index, name in enumerate(MODULE.PLOT_FILES):
        _write_image(tmp_path / name, (30 + index * 20, 80, 140))
    (tmp_path / "comparison_kappa.png").write_bytes(
        (tmp_path / "comparison_kappa_bar.png").read_bytes()
    )

    report = MODULE.check_compare_plots(tmp_path)
    band = next(item for item in report["checks"] if item["file"] == "comparison_phonon_band.png")

    assert report["passed"] is False
    assert any("one entry per plotted model" in reason for reason in band["reasons"])
