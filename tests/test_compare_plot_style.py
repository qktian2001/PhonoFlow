from __future__ import annotations

import json
from pathlib import Path

from phonoflow.compare_models import compare_models


def _write_band(path: Path, offset: float, labels: tuple[str, str] = ("GAMMA", "X")) -> None:
    path.write_text(
        "\n".join(
            [
                "segment_nqpoint: [3]",
                f"labels: [[{labels[0]}, {labels[1]}]]",
                "phonon:",
                "- distance: 0.0",
                "  band:",
                f"  - frequency: {-0.2 + offset}",
                f"  - frequency: {1.0 + offset}",
                "- distance: 0.5",
                "  band:",
                f"  - frequency: {0.5 + offset}",
                f"  - frequency: {1.5 + offset}",
                "- distance: 1.0",
                "  band:",
                f"  - frequency: {1.0 + offset}",
                f"  - frequency: {2.0 + offset}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_dos(path: Path, scale: float) -> None:
    path.write_text(f"0 0\n1 {scale}\n2 {scale / 2}\n", encoding="utf-8")


def _result(config, kavg: float) -> dict:
    config.outdir.mkdir(parents=True, exist_ok=True)
    _write_band(config.outdir / "band.yaml", kavg / 100.0)
    _write_dos(config.outdir / "phonon_dos.dat", kavg)
    result = {
        "backend_requested": config.backend_alias,
        "backend_resolved": config.backend,
        "backend_alias": config.backend_alias,
        "model_path": str(config.model_path) if config.model_path else None,
        "dpa_model_name": config.dpa_model_name,
        "thermal_conductivity": {
            "available": True,
            "summary": [
                {
                    "temperature": 300.0,
                    "kxx": kavg - 1,
                    "kyy": kavg,
                    "kzz": kavg + 1,
                    "kavg": kavg,
                }
            ],
        },
    }
    (config.outdir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return {"status": "success", "outdir": str(config.outdir), "report": result}


def test_comparison_legends_are_model_level_and_colors_are_stable(tmp_path: Path, monkeypatch) -> None:
    values = {"nep89": 3.0, "dpa32": 4.0, "dpa4neo": 5.0}
    monkeypatch.setattr(
        "phonoflow.compare_models.run_single_workflow",
        lambda config: _result(config, values[str(config.backend_alias)]),
    )

    summary = compare_models(
        input_path=Path("examples/Si.vasp"),
        outdir=tmp_path / "compare",
        model_names=["nep89", "dpa32", "dpa4neo"],
        compute_kappa=True,
        overwrite=True,
    )

    band = summary["comparison_plots"]["phonon_band"]
    dos = summary["comparison_plots"]["dos"]
    thermal = summary["comparison_plots"]["thermal_conductivity"]
    assert band["legend_labels"] == [
        "NEP89",
        "DPA-3.2-5M.pt",
        "DPA4-Neo-OMat24-v20260528_rc.pt",
    ]
    assert band["legend_entry_count"] == 3
    assert dos["legend_labels"] == band["legend_labels"]
    assert thermal["legend_labels"] == band["legend_labels"]
    assert band["model_colors"] == dos["model_colors"] == thermal["model_colors"]


def test_comparison_band_uses_first_successful_model_ticks(tmp_path: Path, monkeypatch) -> None:
    def fake(config):
        response = _result(config, 3.0)
        if config.backend_alias == "dpa32":
            _write_band(config.outdir / "band.yaml", 0.1, labels=("GAMMA", "L"))
        return response

    monkeypatch.setattr("phonoflow.compare_models.run_single_workflow", fake)
    summary = compare_models(
        input_path=Path("examples/Si.vasp"),
        outdir=tmp_path / "compare",
        model_names=["nep89", "dpa32"],
        compute_kappa=True,
        overwrite=True,
    )

    band = summary["comparison_plots"]["phonon_band"]
    assert band["tick_labels"] == ["Γ", "X"]
    assert any("path labels differ" in warning for warning in band["warnings"])


def test_single_temperature_thermal_plot_reports_bar_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "phonoflow.compare_models.run_single_workflow",
        lambda config: _result(config, 3.0),
    )
    summary = compare_models(
        input_path=Path("examples/Si.vasp"),
        outdir=tmp_path / "compare",
        model_names=["nep89"],
        compute_kappa=True,
        overwrite=True,
    )

    assert summary["comparison_plots"]["thermal_conductivity"]["mode"] == "single-temperature-bars"
    assert summary["comparison_plots"]["kappa_bar"]["successful_model_count"] == 1
    assert summary["comparison_plots"]["kappa_bar"]["bar_count"] == 4


def test_kappa_bar_metadata_uses_grouped_short_labels_and_keeps_full_names(
    tmp_path: Path, monkeypatch
) -> None:
    values = {"nep89": 3.0, "dpa32": 4.0, "dpa4neo": 5.0}
    monkeypatch.setattr(
        "phonoflow.compare_models.run_single_workflow",
        lambda config: _result(config, values[str(config.backend_alias)]),
    )

    summary = compare_models(
        input_path=Path("examples/Si.vasp"),
        outdir=tmp_path / "compare",
        model_names=["nep89", "dpa32", "dpa4neo"],
        compute_kappa=True,
        overwrite=True,
    )

    metadata = summary["comparison_plots"]["kappa_bar"]
    assert metadata["bar_count"] == 12
    assert metadata["display_labels"] == ["NEP89", "DPA-3.2", "DPA4-Neo"]
    assert metadata["full_model_names"] == [
        "NEP89",
        "DPA-3.2-5M.pt",
        "DPA4-Neo-OMat24-v20260528_rc.pt",
    ]
