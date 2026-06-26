from __future__ import annotations

import json
from pathlib import Path

from phonoflow.compare_models import compare_models


def _write_fake_band_yaml(path: Path, offset: float) -> None:
    path.write_text(
        "\n".join(
            [
                "phonon:",
                "- distance: 0.0",
                "  band:",
                f"  - frequency: {0.0 + offset}",
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


def _write_fake_dos(path: Path, scale: float) -> None:
    path.write_text(
        "# frequency_THz total_DOS\n"
        f"0.0 {0.0 * scale}\n"
        f"1.0 {2.0 * scale}\n"
        f"2.0 {0.5 * scale}\n",
        encoding="utf-8",
    )


def _write_fake_thermal_csv(path: Path, kavg: float) -> None:
    path.write_text(
        "temperature_K,kxx,kyy,kzz,kappa_trace_over_3\n"
        f"300.0,{kavg - 1.0},{kavg},{kavg + 1.0},{kavg}\n",
        encoding="utf-8",
    )


def _write_fake_group_velocity_csv(path: Path, scale: float) -> None:
    path.write_text(
        "q_index,branch_index,frequency_THz,vg_x_km_s,vg_y_km_s,vg_z_km_s,vg_abs_km_s\n"
        f"0,0,1.0,{scale},0.0,0.0,{scale}\n"
        f"1,0,2.0,0.0,{scale},0.0,{scale}\n",
        encoding="utf-8",
    )


def _write_fake_lifetime_csv(path: Path, scale: float) -> None:
    path.write_text(
        "temperature_K,q_index,branch_index,frequency_THz,lifetime_ps,gamma_or_linewidth_raw\n"
        f"300.0,0,0,1.0,{scale},0.01\n"
        f"300.0,1,0,2.0,{scale * 2.0},0.02\n",
        encoding="utf-8",
    )


def test_compare_models_uses_real_band_dos_and_thermal_files(tmp_path: Path, monkeypatch) -> None:
    values = {"nep89": 2.0, "dpa32": 5.0, "dpa4neo": 8.0}

    def fake_run_single_workflow(config):
        model_outdir = config.outdir
        model_outdir.mkdir(parents=True, exist_ok=True)
        kavg = values[str(config.backend_alias)]
        _write_fake_band_yaml(model_outdir / "band.yaml", offset=kavg / 10.0)
        _write_fake_dos(model_outdir / "phonon_dos.dat", scale=kavg)
        _write_fake_thermal_csv(model_outdir / "thermal_conductivity.csv", kavg=kavg)
        _write_fake_group_velocity_csv(model_outdir / "phonon_group_velocity.csv", scale=kavg)
        _write_fake_lifetime_csv(model_outdir / "phonon_lifetime.csv", scale=kavg)
        result = {
            "success": True,
            "backend_requested": config.backend_alias,
            "backend_resolved": config.backend,
            "backend_alias": config.backend_alias,
            "dpa_model_name": config.dpa_model_name,
            "dynamically_stable": True,
            "minimum_frequency_THz": 0.0,
            "imaginary_mode_count": 0,
            "thermal_conductivity": {"available": True},
        }
        (model_outdir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return {"status": "success", "outdir": str(model_outdir), "report": result}

    monkeypatch.setattr("phonoflow.compare_models.run_single_workflow", fake_run_single_workflow)

    summary = compare_models(
        input_path=Path("examples/Si.vasp"),
        outdir=tmp_path / "compare",
        model_names=["nep89", "dpa32", "dpa4neo"],
        compute_kappa=True,
        overwrite=True,
        dry_run=False,
    )

    assert len(summary["kappa_bar_components"]) == 12
    assert summary["comparison_plots"]["phonon_band"]["kind"] == "data"
    assert summary["comparison_plots"]["phonon_band"]["models"] == ["nep89", "dpa32", "dpa4neo"]
    assert summary["comparison_plots"]["dos"]["kind"] == "data"
    assert summary["comparison_plots"]["dos"]["models"] == ["nep89", "dpa32", "dpa4neo"]
    assert summary["comparison_plots"]["thermal_conductivity"]["kind"] == "data"
    assert summary["comparison_plots"]["group_velocity"]["kind"] == "data"
    assert summary["comparison_plots"]["group_velocity"]["models"] == ["nep89", "dpa32", "dpa4neo"]
    assert summary["comparison_plots"]["phonon_lifetime"]["kind"] == "data"
    assert summary["comparison_plots"]["phonon_lifetime"]["models"] == ["nep89", "dpa32", "dpa4neo"]
    assert (tmp_path / "compare" / "comparison_group_velocity.png").exists()
    assert (tmp_path / "compare" / "comparison_group_velocity.csv").exists()
    assert (tmp_path / "compare" / "comparison_group_velocity_diagnostics.json").exists()
    assert (tmp_path / "compare" / "comparison_phonon_lifetime.png").exists()
    assert (tmp_path / "compare" / "comparison_phonon_lifetime.csv").exists()
    assert (tmp_path / "compare" / "comparison_phonon_lifetime_diagnostics.json").exists()
    result = json.loads((tmp_path / "compare" / "comparison_result.json").read_text(encoding="utf-8"))
    assert result["comparison_plots"]["phonon_band"]["kind"] == "data"


def test_compare_models_dry_run_marks_status_plots(tmp_path: Path, monkeypatch) -> None:
    def fake_run_single_workflow(config):
        config.outdir.mkdir(parents=True, exist_ok=True)
        result = {
            "success": True,
            "backend_requested": config.backend_alias,
            "backend_resolved": config.backend,
            "backend_alias": config.backend_alias,
            "thermal_conductivity": {"available": False},
        }
        (config.outdir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return {"status": "dry-run", "outdir": str(config.outdir), "report": result}

    monkeypatch.setattr("phonoflow.compare_models.run_single_workflow", fake_run_single_workflow)

    summary = compare_models(
        input_path=Path("examples/Si.vasp"),
        outdir=tmp_path / "compare",
        model_names=["nep89", "dpa3", "dpa4"],
        compute_kappa=True,
        overwrite=True,
        dry_run=True,
    )

    assert summary["comparison_plots"]["phonon_band"]["kind"] == "dry-run/status"
    assert summary["comparison_plots"]["dos"]["kind"] == "dry-run/status"


def test_compare_models_phonon_only_explains_lifetime_requirement(tmp_path: Path, monkeypatch) -> None:
    def fake_run_single_workflow(config):
        config.outdir.mkdir(parents=True, exist_ok=True)
        result = {
            "success": True,
            "backend_requested": config.backend_alias,
            "backend_resolved": config.backend,
            "backend_alias": config.backend_alias,
            "thermal_conductivity": {"enabled": False, "available": False},
        }
        (config.outdir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return {"status": "success", "outdir": str(config.outdir), "report": result}

    monkeypatch.setattr("phonoflow.compare_models.run_single_workflow", fake_run_single_workflow)

    summary = compare_models(
        input_path=Path("examples/Si.vasp"),
        outdir=tmp_path / "compare",
        model_names=["nep89"],
        compute_kappa=False,
        overwrite=True,
        dry_run=False,
    )

    lifetime = summary["comparison_plots"]["phonon_lifetime"]
    assert lifetime["available"] is False
    assert lifetime["reason"] == "Phonon lifetime requires thermal conductivity / FC3 calculation."
    assert not (tmp_path / "compare" / "comparison_phonon_lifetime.png").exists()
