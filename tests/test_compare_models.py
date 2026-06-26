from __future__ import annotations

import json
from pathlib import Path

from phonoflow.compare_models import _kappa_summary, _model_command, compare_models


def test_compare_models_writes_summary_and_continues_after_failure(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_single_workflow(config):
        calls.append({"backend": config.backend, "alias": config.backend_alias, "outdir": config.outdir})
        if config.backend_alias == "dpa32":
            raise RuntimeError("synthetic DPA3 failure")
        result = {
            "success": True,
            "backend_requested": config.backend_alias,
            "backend_resolved": config.backend,
            "backend_alias": config.backend_alias,
            "model_path": str(config.model_path) if config.model_path else None,
            "dpa_model_name": config.dpa_model_name,
            "force_backend": config.backend,
            "force_model_path": str(config.model_path) if config.model_path else None,
            "relax_backend": "calorine" if config.backend_alias in {"dpa32", "dpa4neo"} else config.backend,
            "relax_model_path": (
                "nep89_potential/nep89_20250409.txt"
                if config.backend_alias in {"dpa32", "dpa4neo"}
                else str(config.model_path) if config.model_path else None
            ),
            "relax_enabled": bool(config.relax),
            "dynamically_stable": True,
            "minimum_frequency_THz": 0.0,
            "imaginary_mode_count": 0,
            "thermal_conductivity": {
                "available": True,
                "summary": [{"temperature": 300.0, "kxx": 1.0, "kyy": 2.0, "kzz": 3.0, "kavg": 2.0}],
            },
        }
        config.outdir.mkdir(parents=True, exist_ok=True)
        (config.outdir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return {"status": "success", "outdir": str(config.outdir), "report": result}

    monkeypatch.setattr("phonoflow.compare_models.run_single_workflow", fake_run_single_workflow)

    summary = compare_models(
        input_path=Path("examples/Si.vasp"),
        outdir=tmp_path / "compare",
        model_names=["nep89", "dpa32", "dpa4neo"],
        compute_kappa=True,
        overwrite=True,
        dry_run=False,
    )

    assert [call["alias"] for call in calls] == ["nep89", "dpa32", "dpa4neo"]
    assert len(summary["models"]) == 3
    assert summary["models"][1]["status"] == "failed"
    assert "synthetic DPA3 failure" in summary["models"][1]["error_message"]
    assert (tmp_path / "compare" / "comparison_summary.json").exists()
    assert (tmp_path / "compare" / "comparison_summary.csv").exists()
    assert (tmp_path / "compare" / "comparison_summary.md").exists()
    assert (tmp_path / "compare" / "comparison_kappa.png").exists()
    assert (tmp_path / "compare" / "comparison_result.json").exists()
    assert (tmp_path / "compare" / "comparison_kappa_bar.png").exists()
    assert (tmp_path / "compare" / "comparison_phonon_band.png").exists()
    assert (tmp_path / "compare" / "comparison_dos.png").exists()
    assert (tmp_path / "compare" / "comparison_thermal_conductivity.png").exists()
    assert (tmp_path / "compare" / "report_en.md").exists()
    assert (tmp_path / "compare" / "report_zh.md").exists()
    assert (tmp_path / "compare" / "summary.json").exists()
    assert (tmp_path / "compare" / "commands.log").exists()
    assert (tmp_path / "compare" / "validation.log").exists()
    assert (tmp_path / "compare" / "artifacts_index.csv").exists()
    assert summary["models"][0]["force_model_path"]
    assert summary["models"][2]["relax_model_path"] == "nep89_potential/nep89_20250409.txt"
    parent_log = (tmp_path / "compare" / "run.log").read_text(encoding="utf-8")
    assert "[1/8] Reading input structure" in parent_log
    assert "[2/8] Resolving default settings" in parent_log
    assert "Resolved PhonoFlow settings" in parent_log
    assert all("run_log_path" in row for row in summary["models"])


def test_compare_models_isolated_subprocess_failure_writes_summary(tmp_path: Path, monkeypatch) -> None:
    captured_env: dict[str, str] = {}

    def fake_run(command, cwd, env, text, capture_output):
        captured_env.update(env)

        class Result:
            returncode = -9
            stdout = "synthetic stdout"
            stderr = (
                "implib-gen: libcudart.so.12: failed to resolve symbol "
                "'__cudaRegisterFatBinary'\n"
                "You can use the environment variable DP_INFER_BATCH_SIZE to control inference batch size."
            )

        return Result()

    monkeypatch.setattr("phonoflow.compare_models.subprocess.run", fake_run)

    summary = compare_models(
        input_path=Path("examples/Si.vasp"),
        outdir=tmp_path / "compare",
        model_names=["dpa4neo"],
        compute_kappa=True,
        overwrite=True,
        dry_run=False,
        isolate=True,
    )

    assert summary["models"][0]["status"] == "failed"
    assert "memory" in summary["models"][0]["failure_reason"].lower()
    assert "libcudart" not in summary["models"][0]["failure_reason"].lower()
    assert summary["models"][0]["return_code"] == -9
    assert summary["models"][0]["command"]
    assert summary["models"][0]["command_argv"]
    assert summary["models"][0]["backend_requested"] == "dpa4neo"
    assert summary["models"][0]["model_path"]
    assert summary["models"][0]["error_category"] == "out_of_memory"
    assert summary["models"][0]["failure_category"] == "out_of_memory"
    assert summary["models"][0]["deepmd_device"] == "cpu"
    assert summary["models"][0]["dp_infer_batch_size"] == 64
    assert captured_env["CUDA_VISIBLE_DEVICES"] == "-1"
    assert captured_env["DP_INFER_BATCH_SIZE"] == "64"
    assert "libcudart" not in (
        tmp_path / "compare" / "dpa4neo" / "compare_subprocess.stderr.log"
    ).read_text(encoding="utf-8")
    assert "libcudart" in (
        tmp_path / "compare" / "dpa4neo" / "compare_subprocess.stderr.raw.log"
    ).read_text(encoding="utf-8")
    assert "band" in summary["models"][0]["plot_data_availability"]
    assert (tmp_path / "compare" / "comparison_result.json").exists()


def test_compare_models_kappa_bar_has_four_components_per_successful_model(tmp_path: Path, monkeypatch) -> None:
    def fake_run_single_workflow(config):
        values = {
            "nep89": {"kxx": 1.0, "kyy": 2.0, "kzz": 3.0, "kavg": 2.0},
            "dpa32": {"kxx": 4.0, "kyy": 5.0, "kzz": 6.0, "kavg": 5.0},
            "dpa4neo": {"kxx": 7.0, "kyy": 8.0, "kzz": 9.0, "kavg": 8.0},
        }[config.backend_alias]
        result = {
            "success": True,
            "backend_requested": config.backend_alias,
            "backend_resolved": config.backend,
            "backend_alias": config.backend_alias,
            "model_path": str(config.model_path) if config.model_path else None,
            "dpa_model_name": config.dpa_model_name,
            "dynamically_stable": True,
            "minimum_frequency_THz": 0.0,
            "imaginary_mode_count": 0,
            "thermal_conductivity": {
                "available": True,
                "summary": [{"temperature": 300.0, **values}],
            },
        }
        config.outdir.mkdir(parents=True, exist_ok=True)
        (config.outdir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return {"status": "success", "outdir": str(config.outdir), "report": result}

    monkeypatch.setattr("phonoflow.compare_models.run_single_workflow", fake_run_single_workflow)

    summary = compare_models(
        input_path=Path("examples/Si.vasp"),
        outdir=tmp_path / "compare",
        model_names=["nep89", "dpa32", "dpa4neo"],
        compute_kappa=True,
        overwrite=True,
        dry_run=False,
    )

    components = summary["kappa_bar_components"]
    assert len(components) == 12
    assert {item["component"] for item in components} == {"kxx", "kyy", "kzz", "kavg"}
    assert {item["model"] for item in components} == {"nep89", "dpa32", "dpa4neo"}
    result = json.loads((tmp_path / "compare" / "comparison_result.json").read_text(encoding="utf-8"))
    assert len(result["kappa_bar_components"]) == 12
    assert result["comparison_plots"]["kappa_bar"]["successful_model_count"] == 3
    assert result["comparison_plots"]["kappa_bar"]["value_labels_enabled"] is True
    assert result["comparison_plots"]["kappa_bar"]["annotation_text"] == ""
    assert result["comparison_plots"]["kappa_bar"]["display_labels"] == ["NEP89", "DPA-3.2", "DPA4-Neo"]
    assert result["comparison_plots"]["kappa_bar"]["full_model_names"] == [
        "NEP89",
        "DPA-3.2-5M.pt",
        "DPA4-Neo-OMat24-v20260528_rc.pt",
    ]


def test_compare_models_uses_model_defaults_instead_of_smoke_test_overrides(tmp_path: Path, monkeypatch) -> None:
    seen: dict[str, dict[str, object]] = {}

    def fake_run_single_workflow(config):
        seen[str(config.backend_alias)] = {
            "supercell_dim": config.supercell_dim,
            "fc3_supercell_dim": config.fc3_supercell_dim,
            "kappa_mesh": config.kappa_mesh,
            "phonopy_symprec": config.phonopy_symprec,
            "phono3py_symprec": config.phono3py_symprec,
            "phono3py_cutoff_frequency": config.phono3py_cutoff_frequency,
            "phono3py_symmetrize_fc2": config.phono3py_symmetrize_fc2,
            "deepmd_deterministic": config.deepmd_deterministic,
            "deepmd_reuse_calculator": config.deepmd_reuse_calculator,
            "save_force_audit": config.save_force_audit,
        }
        config.outdir.mkdir(parents=True, exist_ok=True)
        result = {
            "success": True,
            "backend_requested": config.backend_alias,
            "backend_resolved": config.backend,
            "backend_alias": config.backend_alias,
            "model_path": str(config.model_path) if config.model_path else None,
            "dpa_model_name": config.dpa_model_name,
            "dynamically_stable": True,
            "minimum_frequency_THz": 0.0,
            "imaginary_mode_count": 0,
            "thermal_conductivity": {"available": False},
        }
        (config.outdir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return {"status": "success", "outdir": str(config.outdir), "report": result}

    monkeypatch.setattr("phonoflow.compare_models.run_single_workflow", fake_run_single_workflow)

    compare_models(
        input_path=Path("examples/Si.vasp"),
        outdir=tmp_path / "compare",
        model_names=["nep89", "dpa32", "dpa4neo"],
        compute_kappa=True,
        overwrite=True,
        dry_run=True,
    )

    assert seen["nep89"]["supercell_dim"] == "auto"
    assert seen["dpa32"]["supercell_dim"] == "auto"
    assert seen["dpa4neo"]["supercell_dim"] == "auto"
    assert seen["dpa32"]["fc3_supercell_dim"] == "auto"
    assert seen["dpa4neo"]["kappa_mesh"] == "auto"
    assert seen["nep89"]["phonopy_symprec"] == 1e-5
    assert seen["dpa32"]["phonopy_symprec"] == 1e-5
    assert seen["dpa32"]["phono3py_symprec"] == 1e-5
    assert seen["dpa4neo"]["phono3py_cutoff_frequency"] == 1e-4
    assert seen["nep89"]["phono3py_symmetrize_fc2"] is True
    assert seen["dpa32"]["phono3py_symmetrize_fc2"] is True
    assert seen["dpa4neo"]["phono3py_symmetrize_fc2"] is True
    assert seen["dpa32"]["deepmd_deterministic"] is True
    assert seen["dpa4neo"]["deepmd_reuse_calculator"] is True
    assert seen["dpa32"]["save_force_audit"] is True


def test_compare_model_commands_scope_deepmd_options_to_dpa_children() -> None:
    common = {
        "input_path": Path("examples/Si.vasp"),
        "compute_kappa": False,
        "relax": False,
        "dry_run": True,
        "overwrite": True,
        "deepmd_device": "cpu",
        "deepmd_deterministic": False,
        "deepmd_reuse_calculator": True,
        "save_force_audit": False,
    }
    nep_command = _model_command(
        outdir=Path("results/nep89"),
        backend="calorine",
        **common,
    )
    dpa_command = _model_command(
        outdir=Path("results/dpa31"),
        backend="dpa31",
        **common,
    )

    for option in (
        "--deepmd-device",
        "--deepmd-deterministic",
        "--no-deepmd-deterministic",
        "--deepmd-reuse-calculator",
        "--no-deepmd-reuse-calculator",
        "--save-force-audit",
        "--no-save-force-audit",
    ):
        assert option not in nep_command
    assert "--deepmd-device" in dpa_command
    assert "--no-deepmd-deterministic" in dpa_command
    assert "--deepmd-reuse-calculator" in dpa_command
    assert "--no-save-force-audit" in dpa_command


def test_compare_model_commands_forward_phono3py_options_to_nep_kappa_children() -> None:
    command = _model_command(
        input_path=Path("examples/Si.vasp"),
        outdir=Path("results/nep89"),
        backend="calorine",
        compute_kappa=True,
        relax=False,
        dry_run=True,
        overwrite=True,
        phono3py_symmetrize_fc2=False,
        phono3py_symmetrize_fc3=True,
        phono3py_symprec=2e-5,
        phono3py_cutoff_frequency=2e-4,
    )

    assert "--no-phono3py-symmetrize-fc2" in command
    assert "--phono3py-symmetrize-fc3" in command
    assert command[command.index("--phono3py-symprec") + 1] == "2e-05"
    assert command[command.index("--phono3py-cutoff-frequency") + 1] == "0.0002"
    assert "--deepmd-device" not in command


def test_compare_model_commands_forward_hiphive_options_to_kappa_children() -> None:
    command = _model_command(
        input_path=Path("examples/Si.vasp"),
        outdir=Path("results/nep89"),
        backend="calorine",
        compute_kappa=True,
        relax=False,
        dry_run=True,
        overwrite=True,
        fc3_method="hiphive",
        n_structures=7,
        rattle_std=0.011,
        cutoffs=[3.5, 2.8],
        min_dist=1.6,
    )

    assert command[command.index("--fc3-method") + 1] == "hiphive"
    assert command[command.index("--n-structures") + 1] == "7"
    assert command[command.index("--rattle-std") + 1] == "0.011"
    cutoff_index = command.index("--cutoffs")
    assert command[cutoff_index + 1 : cutoff_index + 3] == ["3.5", "2.8"]
    assert command[command.index("--min-dist") + 1] == "1.6"


def test_compare_model_command_forwards_phonopy_symprec() -> None:
    command = _model_command(
        input_path=Path("examples/Si.vasp"),
        outdir=Path("results/nep89"),
        backend="calorine",
        compute_kappa=False,
        relax=False,
        dry_run=True,
        overwrite=True,
        phonopy_symprec=1e-5,
    )

    assert command[command.index("--phonopy-symprec") + 1] == "1e-05"


def test_compare_model_command_forwards_lbte_and_shared_q_mesh() -> None:
    command = _model_command(
        input_path=Path("examples/Si.vasp"),
        outdir=Path("results/nep89"),
        backend="calorine",
        compute_kappa=True,
        relax=False,
        dry_run=True,
        overwrite=True,
        mesh=(3, 3, 3),
        kappa_method="lbte",
        temperatures=[100.0, 300.0],
        fc3_supercell_dim=(2, 2, 2),
        fc3_cutoff_pair_distance=4.5,
        max_fc3_displacements=5,
    )

    assert command[command.index("--method") + 1] == "lbte"
    temperature_indices = [index for index, item in enumerate(command) if item == "--temperatures"]
    assert [command[index + 1] for index in temperature_indices] == ["100.0", "300.0"]
    assert command[command.index("--mesh") + 1 : command.index("--mesh") + 4] == ["3", "3", "3"]
    assert command[command.index("--kappa-mesh") + 1 : command.index("--kappa-mesh") + 4] == ["3", "3", "3"]
    assert command[command.index("--fc3-supercell-dim") + 1 : command.index("--fc3-supercell-dim") + 4] == ["2", "2", "2"]
    assert command[command.index("--fc3-cutoff-pair-distance") + 1] == "4.5"


def test_compare_model_command_leaves_default_q_mesh_auto_for_child_structure_detection() -> None:
    command = _model_command(
        input_path=Path("examples/Si.vasp"),
        outdir=Path("results/nep89"),
        backend="calorine",
        compute_kappa=True,
        relax=False,
        dry_run=True,
        overwrite=True,
    )

    assert command[command.index("--mesh") + 1] == "auto"
    assert command[command.index("--kappa-mesh") + 1] == "auto"


def test_compare_models_propagates_lbte_to_child_configs(tmp_path: Path, monkeypatch) -> None:
    seen: dict[str, dict[str, object]] = {}

    def fake_run_single_workflow(config):
        seen[str(config.backend_alias)] = {
            "kappa_method": config.kappa_method,
            "temperatures": config.temperatures,
            "mesh": config.mesh,
            "kappa_mesh": config.kappa_mesh,
        }
        config.outdir.mkdir(parents=True, exist_ok=True)
        result = {
            "success": True,
            "backend_requested": config.backend_alias,
            "backend_resolved": config.backend,
            "backend_alias": config.backend_alias,
            "kappa_method": config.kappa_method,
            "temperatures": config.temperatures,
            "q_mesh": config.mesh,
            "thermal_conductivity": {"available": False, "kappa_method": config.kappa_method},
        }
        (config.outdir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return {"status": "success", "outdir": str(config.outdir), "report": result}

    monkeypatch.setattr("phonoflow.compare_models.run_single_workflow", fake_run_single_workflow)

    summary = compare_models(
        input_path=Path("examples/Si.vasp"),
        outdir=tmp_path / "compare",
        model_names=["nep89", "dpa32"],
        compute_kappa=True,
        overwrite=True,
        dry_run=True,
        mesh=(3, 3, 3),
        kappa_method="lbte",
        temperatures=[100.0, 300.0],
    )

    assert seen["nep89"]["kappa_method"] == "lbte"
    assert seen["dpa32"]["kappa_method"] == "lbte"
    assert seen["nep89"]["temperatures"] == [100.0, 300.0]
    assert seen["dpa32"]["mesh"] == [3, 3, 3]
    assert summary["kappa_method"] == "lbte"
    assert summary["solver_flags"] == ["--method", "lbte"]
    assert all(row["kappa_method"] == "lbte" for row in summary["models"])


def test_compare_models_propagates_hiphive_options_to_child_configs(tmp_path: Path, monkeypatch) -> None:
    seen: dict[str, dict[str, object]] = {}

    def fake_run_single_workflow(config):
        seen[str(config.backend_alias)] = {
            "fc3_method": config.fc3_method,
            "n_structures": config.n_structures,
            "rattle_std": config.rattle_std,
            "cutoffs": config.cutoffs,
            "min_dist": config.min_dist,
        }
        config.outdir.mkdir(parents=True, exist_ok=True)
        result = {
            "success": True,
            "backend_requested": config.backend_alias,
            "backend_resolved": config.backend,
            "backend_alias": config.backend_alias,
            "thermal_conductivity": {"available": False},
        }
        (config.outdir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return {"status": "success", "outdir": str(config.outdir), "report": result}

    monkeypatch.setattr("phonoflow.compare_models.run_single_workflow", fake_run_single_workflow)

    compare_models(
        input_path=Path("examples/Si.vasp"),
        outdir=tmp_path / "compare",
        model_names=["nep89", "dpa32"],
        compute_kappa=True,
        overwrite=True,
        dry_run=True,
        fc3_method="hiphive",
        n_structures=7,
        rattle_std=0.011,
        cutoffs=[3.5, 2.8],
        min_dist=1.6,
    )

    assert seen["nep89"]["fc3_method"] == "hiphive"
    assert seen["dpa32"]["fc3_method"] == "hiphive"
    assert seen["nep89"]["n_structures"] == 7
    assert seen["dpa32"]["rattle_std"] == 0.011
    assert seen["nep89"]["cutoffs"] == [3.5, 2.8]
    assert seen["dpa32"]["min_dist"] == 1.6


def test_compare_relax_uses_shared_nep89_structure_for_child_properties(tmp_path: Path, monkeypatch) -> None:
    seen: dict[str, dict[str, object]] = {}
    input_path = tmp_path / "Si.vasp"
    input_path.write_text(
        "\n".join(
            [
                "Si",
                "1.0",
                "3.8 0 0",
                "0 3.8 0",
                "0 0 3.8",
                "Si",
                "1",
                "Direct",
                "0 0 0",
            ]
        ),
        encoding="utf-8",
    )

    def fake_run_single_workflow(config):
        seen[str(config.backend_alias)] = {
            "input_path": config.input_path,
            "backend": config.backend,
            "relax": config.relax,
            "relax_cell": config.relax_cell,
        }
        config.outdir.mkdir(parents=True, exist_ok=True)
        result = {
            "success": True,
            "backend_requested": config.backend_alias,
            "backend_resolved": config.backend,
            "backend_alias": config.backend_alias,
            "model_path": str(config.model_path) if config.model_path else None,
            "force_backend": config.backend,
            "force_model_path": str(config.model_path) if config.model_path else None,
            "relax": config.relax,
            "relax_enabled": config.relax,
            "thermal_conductivity": {"available": False},
        }
        (config.outdir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return {"status": "success", "outdir": str(config.outdir), "report": result}

    monkeypatch.setattr("phonoflow.compare_models.run_single_workflow", fake_run_single_workflow)

    summary = compare_models(
        input_path=input_path,
        outdir=tmp_path / "compare",
        model_names=["nep89", "dpa32"],
        compute_kappa=False,
        overwrite=True,
        dry_run=True,
        relax=True,
        relax_cell=False,
    )

    shared_path = tmp_path / "compare" / "shared_nep89_relax" / "relaxed.vasp"
    assert shared_path.exists()
    assert seen["nep89"]["input_path"] == shared_path
    assert seen["dpa32"]["input_path"] == shared_path
    assert seen["nep89"]["relax"] is False
    assert seen["dpa32"]["backend"] == "dpa32"
    assert summary["relax_model"] == "NEP89"
    assert summary["relax_policy"] == "shared_nep89_pre_relax"
    assert summary["relax_cell"] is False
    assert summary["shared_relaxed_structure"] is True
    assert all(row["relax_enabled"] is True for row in summary["models"])
    assert all(row["shared_relaxed_structure"] is True for row in summary["models"])
    child_result = json.loads((tmp_path / "compare" / "dpa32" / "result.json").read_text(encoding="utf-8"))
    assert child_result["relax_model"] == "NEP89"
    assert child_result["property_model"] == "DPA-3.2-5M.pt"


def test_compare_model_command_forwards_relax_cell_flag() -> None:
    command = _model_command(
        input_path=Path("examples/Si.vasp"),
        outdir=Path("results/nep89"),
        backend="calorine",
        compute_kappa=False,
        relax=True,
        relax_cell=False,
        dry_run=True,
        overwrite=True,
    )

    assert "--relax" in command
    assert "--no-relax-cell" in command


def test_dpa4_safe_mode_command_applies_explicit_limits_without_common_target() -> None:
    command = _model_command(
        input_path=Path("examples/Si.vasp"),
        outdir=Path("results/dpa4-safe"),
        backend="dpa4neo",
        compute_kappa=False,
        relax=False,
        dry_run=True,
        overwrite=True,
        dpa_safe_mode=True,
    )

    assert command[command.index("--target-supercell-length") + 1] == "12.0"
    assert command[command.index("--max-supercell-atoms") + 1] == "256"


def test_kappa_summary_reads_result_dict_temperature_keys() -> None:
    thermal = {
        "summary": {
            "300": {
                "kxx": 1.0,
                "kyy": 2.0,
                "kzz": 3.0,
                "kappa_trace_over_3": 2.0,
            }
        }
    }

    assert _kappa_summary(thermal) == {"kxx": 1.0, "kyy": 2.0, "kzz": 3.0, "kavg": 2.0}


def test_kappa_summary_prefers_300k_row() -> None:
    thermal = {
        "summary": [
            {"temperature": 100.0, "kxx": 1.0, "kyy": 1.0, "kzz": 1.0, "kavg": 1.0},
            {"temperature": 300.0, "kxx": 3.0, "kyy": 4.0, "kzz": 5.0, "kavg": 4.0},
        ]
    }

    assert _kappa_summary(thermal) == {"kxx": 3.0, "kyy": 4.0, "kzz": 5.0, "kavg": 4.0}
