"""Single-structure workflow orchestration."""

from __future__ import annotations

import os
import platform
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

import ase
import matplotlib
import numpy as np
from rich.console import Console

from phonoflow.analysis.spacegroup import (
    analyze_spacegroup,
    build_spacegroup_report,
    write_spacegroup_report,
)
from phonoflow.analysis.stability import analyze_stability
from phonoflow.analysis.structure_type import classify_structure_type
from phonoflow.calculators import get_backend
from phonoflow.constants import PROJECT_NAME, VERSION
from phonoflow.config import WorkflowConfig, write_config
from phonoflow.defaults import DEFAULT_NEP89_MODEL_PATH, infer_default_config
from phonoflow.exceptions import BackendUnavailableError, ConfigError
from phonoflow.io.hash_utils import sha256_file
from phonoflow.io.path_utils import ensure_dir
from phonoflow.io.structure_io import read_structure, write_structure
from phonoflow.reporting.report_json import build_stability_report, write_json, write_stability_json
from phonoflow.reporting.report_text import write_stability_text
from phonoflow.reporting.run_report import build_default_audit_table
from phonoflow.reporting.summary_text import write_summary
from phonoflow.reporting.timing_statistics import (
    timing_row_from_breakdown,
    write_calculation_time_statistics,
)
from phonoflow.resolved_settings import ResolvedSettings, build_run_command
from phonoflow.thermal import disabled_thermal_result, run_thermal_conductivity_workflow
from phonoflow.thermal.wte_backend import get_wte_backend_capability
from phonoflow.workflow.phonon import run_phonon_calculation
from phonoflow.workflow.relax import relax_structure
from phonoflow.workflow.structure_provenance import build_structure_provenance, write_structure_provenance


def _q_mesh_metadata(config: WorkflowConfig) -> dict[str, Any]:
    used_for: list[str] = []
    if config.dos:
        used_for.append("dos")
    if config.compute_kappa:
        used_for.append("kappa")
    return {
        "q_mesh": [int(value) for value in config.mesh],
        "phono3py_mesh": [int(value) for value in config.mesh],
        "q_mesh_centering": "gamma",
        "q_mesh_used_for": used_for or ["dos"],
    }


def _thermal_solver_flags(config: WorkflowConfig) -> list[str]:
    return ["--method", str(config.kappa_method)]


def _relax_policy_metadata(
    config: WorkflowConfig,
    relax_info: dict[str, Any] | None = None,
    *,
    relaxed_structure_path: Path | str | None = None,
    shared_relaxed_structure: bool = False,
) -> dict[str, Any]:
    relax_info = relax_info or {}
    property_model = config.dpa_model_name or config.backend_alias or config.backend
    relax_backend = relax_info.get("relax_backend") or _resolved_relax_backend_name(config)
    relax_model_path = relax_info.get("relax_model_path") or _resolved_relax_model_path(config)
    relax_model = "NEP89" if config.relax and relax_backend == "calorine" else (property_model if config.relax else None)
    return {
        "relax_model": relax_model,
        "relax_policy": "shared_nep89_pre_relax" if config.relax and relax_model == "NEP89" else "input_structure_no_relax",
        "relaxed_structure_path": str(relaxed_structure_path) if relaxed_structure_path is not None else None,
        "shared_relaxed_structure": bool(shared_relaxed_structure),
        "property_model": property_model,
        "calculation_model": property_model,
        "relax_model_path": relax_model_path,
    }


def run_single_workflow(config: WorkflowConfig) -> dict[str, Any]:
    """Run or dry-run one single-structure workflow."""

    console = Console()
    start_time = time.perf_counter()
    last_stage_time = start_time
    timing_stages: dict[str, dict[str, Any]] = {}
    if config.input_path is None:
        raise ConfigError("single workflow requires input_path.")

    step_lines: list[str] = []

    def announce(
        step: int,
        message: str,
        log: Callable[[str], None] | None = None,
        details: list[str] | None = None,
        status: str | None = None,
        warning: str | None = None,
    ) -> None:
        nonlocal last_stage_time
        now = time.perf_counter()
        elapsed = now - last_stage_time
        last_stage_time = now
        suffix = f" | status={status}" if status else ""
        line = f"[{step}/8] {message}{suffix} | elapsed={elapsed:.2f}s"
        lines = [line]
        for detail in details or []:
            lines.append(f"  - {detail}")
        if warning:
            lines.append(f"  - warning: {warning}")
        text = "\n".join(lines)
        console.print(text)
        step_lines.append(text)
        if log is not None:
            log(text)

    requested_config = config
    structure_started = time.perf_counter()
    announce(
        1,
        f"Reading input structure ({config.input_path})",
        details=[
            f"input path: {config.input_path}",
        ],
    )
    atoms = read_structure(Path(config.input_path))
    timing_stages["structure_read"] = _timing_stage(
        "Reading input structure",
        time.perf_counter() - structure_started,
    )
    settings_started = time.perf_counter()
    structure_classification = classify_structure_type(atoms)
    initial_spacegroup = analyze_spacegroup(
        atoms,
        symprec=config.phonopy_symprec,
        angle_tolerance=config.angle_tolerance,
    )

    config = infer_default_config(atoms, Path(config.input_path), config.model_path, config)
    if config.fc_method == "hiphive":
        raise ConfigError(
            "fc_method='hiphive' is planned for a future release. "
            "Current supported method is finite-displacement."
        )
    outdir, outdir_note = _resolve_output_directory(config)
    config = config.model_copy(update={"outdir": outdir})
    outdir = ensure_dir(Path(config.outdir))

    run_log_path = outdir / "run.log"
    validation_log_path = outdir / "validation.log"
    if run_log_path.exists() and config.overwrite:
        run_log_path.unlink()
    if validation_log_path.exists() and config.overwrite:
        validation_log_path.unlink()

    def log(message: str) -> None:
        with run_log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")
        with validation_log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")

    for line in step_lines:
        log(line)
    announce(
        2,
        "Resolving default settings",
        log=log,
        details=_resolved_workflow_details(config, requested_config, outdir),
    )

    resolved_settings = _build_resolved_settings(
        requested_config=requested_config,
        config=config,
        atoms=atoms,
        outdir_note=outdir_note,
        structure_classification=structure_classification,
    )
    resolved_settings.print_table(console)
    relax_warnings = _relax_warnings(config, structure_classification)
    for warning in relax_warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
        log(f"Warning: {warning}")
    supercell_warnings = list(config.supercell_info.get("auto_supercell_warnings", []))
    for warning in supercell_warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
        log(f"Warning: {warning}")

    resolved_config = config.to_dict()
    resolved_config["backend_requested"] = requested_config.backend
    resolved_config["backend_resolved"] = config.backend
    resolved_config["supercell_info"] = config.supercell_info
    resolved_config["supercell_lengths_resolved"] = config.supercell_info.get("supercell_lengths_resolved")
    resolved_config["auto_supercell_warnings"] = config.supercell_info.get("auto_supercell_warnings", [])
    resolved_config["auto_supercell_notes"] = config.supercell_info.get("auto_supercell_notes", [])
    write_config(resolved_config, outdir / "resolved_config.yaml")
    resolved_settings.write_json(outdir / "resolved_settings.json")
    resolved_settings.write_yaml(outdir / "resolved_settings.yaml")
    settings_table = resolved_settings.write_table(outdir / "resolved_settings_table.txt")
    log(settings_table.rstrip())
    _write_run_commands(config, outdir, requested_config)
    timing_stages["settings_resolution"] = _timing_stage(
        "Resolving default settings",
        time.perf_counter() - settings_started,
    )

    if config.resume and _is_complete_result(outdir):
        result = _read_existing_result(outdir)
        console.print("[yellow]Existing complete result found; resume skipped calculation.[/yellow]")
        log("Resume requested and existing complete result was found; skipped calculation.")
        return {
            "structure_name": Path(config.input_path).name,
            "status": "skipped",
            "outdir": str(outdir),
            "backend": result.get("backend_resolved", config.backend),
            "report": result,
            "dynamically_stable": result.get("dynamically_stable"),
            "minimum_frequency_THz": result.get("minimum_frequency_THz"),
            "error_message": "",
        }

    if config.dry_run or config.print_config:
        for step, key, title, reason in (
            (3, "relaxation", "Relaxation", "skipped because dry-run=true"),
            (4, "fc2_harmonic", "Harmonic force constants / FC2", "skipped because dry-run=true"),
            (
                5,
                "phonon_postprocess",
                "Phonon band / DOS / group velocity",
                "skipped because dry-run=true",
            ),
            (
                6,
                "fc3",
                "Third-order force constants / FC3",
                "skipped because compute_kappa=false"
                if not config.compute_kappa
                else "skipped because dry-run=true",
            ),
            (
                7,
                "thermal_lifetime",
                "Thermal conductivity / lifetime",
                "skipped because compute_kappa=false"
                if not config.compute_kappa
                else "skipped because dry-run=true",
            ),
        ):
            timing_stages[key] = _timing_stage(title, 0.0, status="skipped", reason=reason)
            announce(step, title, log=log, status="skipped", details=[reason])
        reporting_started = time.perf_counter()
        timing_breakdown = _write_timing_breakdown(outdir, timing_stages, start_time)
        timing_statistics = write_calculation_time_statistics(
            outdir,
            [
                timing_row_from_breakdown(
                    model=config.backend_alias or config.backend,
                    display_name=config.dpa_model_name or (config.backend_alias or config.backend),
                    timing_breakdown=timing_breakdown,
                    compute_kappa=config.compute_kappa,
                )
            ],
        )
        spacegroup_report = _write_spacegroup_outputs(
            outdir,
            initial_spacegroup,
            None,
            config,
            dry_run=True,
            log=log,
        )
        result = _build_dry_run_result(
            config,
            requested_config,
            atoms,
            resolved_settings,
            start_time,
            structure_classification,
            relax_warnings,
            supercell_warnings,
            spacegroup_report,
            timing_breakdown,
            timing_statistics,
        )
        write_json(result, outdir / "result.json")
        write_summary(result, outdir / "summary.txt")
        timing_stages["reporting_packaging"] = _timing_stage(
            "Writing reports and artifacts",
            time.perf_counter() - reporting_started,
        )
        timing_breakdown = _write_timing_breakdown(outdir, timing_stages, start_time)
        timing_statistics = write_calculation_time_statistics(
            outdir,
            [
                timing_row_from_breakdown(
                    model=config.backend_alias or config.backend,
                    display_name=config.dpa_model_name or (config.backend_alias or config.backend),
                    timing_breakdown=timing_breakdown,
                    compute_kappa=config.compute_kappa,
                )
            ],
        )
        result["timing_breakdown"] = timing_breakdown
        result["calculation_time_statistics"] = timing_statistics
        result["elapsed_time_seconds"] = timing_breakdown["total_seconds"]
        write_json(result, outdir / "result.json")
        write_summary(result, outdir / "summary.txt")
        announce(
            8,
            "Writing reports and artifacts",
            log=log,
            details=[f"output directory: {outdir}", "result.json: written", "summary.txt: written"],
            status="dry-run",
        )
        log("Dry run completed; no relaxation or phonon calculation was executed.")
        return {
            "structure_name": Path(config.input_path).name,
            "status": "dry-run",
            "outdir": str(outdir),
            "backend": config.backend,
            "report": result,
            "dynamically_stable": None,
            "minimum_frequency_THz": None,
            "error_message": "",
        }

    backend_name = config.backend_alias if config.backend == "deepmd" and config.backend_alias else config.backend
    backend = get_backend(backend_name, model_path=config.model_path)
    backend.apply_config(config)
    if not backend.check_available():
        if backend.name == "calorine":
            raise BackendUnavailableError(
                "Calorine is required for real NEP/NEP89 calculations.\n"
                "Install it with:\n"
                "python -m pip install calorine"
            )
        if backend.name == "deepmd":
            raise BackendUnavailableError(
                "deepmd-kit is required for DeepMD/DPA calculations. For DPA4/SeZM, use a "
                "Linux/WSL environment with compatible deepmd-kit, torch, e3nn, and MPI builds."
            )
        raise BackendUnavailableError(
            f"Requested backend '{backend.name}' is not available. "
            "Use backend='dummy' for workflow tests, or install/configure the optional backend."
        )
    relax_backend = _select_relax_backend(config, backend)
    if relax_backend is not backend:
        relax_backend.apply_config(config)

    write_structure(atoms, outdir / "input_structure.vasp")

    relaxation_started = time.perf_counter()
    if config.relax:
        relaxed_atoms, relax_info = relax_structure(atoms, relax_backend, outdir, config)
        relaxation_status = "completed"
        relaxation_reason = None
    else:
        (outdir / "relax.log").write_text("Relaxation disabled; copied input structure.\n", encoding="utf-8")
        relaxed_atoms = atoms.copy()
        relax_info = _skipped_relax_info(atoms, config)
        relaxation_status = "skipped"
        relaxation_reason = "skipped because relax=false"
    timing_stages["relaxation"] = _timing_stage(
        "Relaxation",
        time.perf_counter() - relaxation_started,
        status=relaxation_status,
        reason=relaxation_reason,
    )
    announce(
        3,
        "Relaxation",
        log=log,
        details=[
            f"relax enabled: {config.relax}",
            f"relax backend resolved: {relax_backend.name if config.relax else 'none'}",
            f"force backend resolved: {backend.name}",
            f"relax cell: {config.relax_cell if config.relax else False}",
            *([relaxation_reason] if relaxation_reason else []),
        ],
        status=relaxation_status,
    )
    relax_info = _ensure_relax_schema(relax_info, atoms, relaxed_atoms, config)
    relax_info["relax_backend"] = relax_backend.name if config.relax else "none"
    relax_info["relax_model_path"] = (
        str(getattr(relax_backend, "model_path", None)) if config.relax else None
    )
    relax_info["force_model_path"] = str(config.model_path) if config.model_path else None
    relax_info["force_backend"] = backend.name
    relax_info["force_backend_alias"] = config.backend_alias or config.backend
    if relax_warnings:
        relax_info["warnings"] = list(relax_info.get("warnings") or []) + relax_warnings

    write_structure(relaxed_atoms, outdir / "relaxed.vasp")
    write_structure(relaxed_atoms, outdir / "fc_source_structure.vasp")
    structure_provenance = build_structure_provenance(
        input_atoms=atoms,
        relaxed_atoms=relaxed_atoms,
        fc2_atoms=relaxed_atoms,
        fc3_atoms=relaxed_atoms,
        input_structure_path=config.input_path,
        relaxed_structure_path=outdir / "relaxed.vasp",
        fc2_source_structure_path=outdir / "fc_source_structure.vasp",
        fc3_source_structure_path=outdir / "fc_source_structure.vasp",
        relax_backend=relax_backend.name if config.relax else "none",
        force_constants_backend=backend.name,
        structure_stage_mode=(
            "two_stage_nep_relax_dpa_force"
            if config.relax and relax_backend is not backend
            else "single_stage" if config.relax else "fixed_input"
        ),
    )
    write_structure_provenance(structure_provenance, outdir / "structure_provenance.json")
    final_spacegroup = analyze_spacegroup(
        relaxed_atoms,
        symprec=config.phonopy_symprec,
        angle_tolerance=config.angle_tolerance,
    )
    spacegroup_report = _write_spacegroup_outputs(
        outdir,
        initial_spacegroup,
        final_spacegroup,
        config,
        dry_run=False,
        log=log,
    )
    _print_spacegroup_summary(console, spacegroup_report)

    if backend.name in {"calorine", "deepmd"}:
        return _run_real_nep_single(
            config=config,
            requested_config=requested_config,
            outdir=outdir,
            backend=backend,
            relaxed_atoms=relaxed_atoms,
            relax_info=relax_info,
            log=log,
            announce=announce,
            start_time=start_time,
            resolved_settings=resolved_settings,
            structure_classification=structure_classification,
            relax_warnings=relax_warnings,
            supercell_warnings=supercell_warnings,
            spacegroup_report=spacegroup_report,
            structure_provenance=structure_provenance,
            timing_stages=timing_stages,
        )

    return _run_dummy_single(
        config=config,
        requested_config=requested_config,
        outdir=outdir,
        relaxed_atoms=relaxed_atoms,
            relax_info=relax_info,
        start_time=start_time,
        resolved_settings=resolved_settings,
        structure_classification=structure_classification,
        relax_warnings=relax_warnings,
        supercell_warnings=supercell_warnings,
        spacegroup_report=spacegroup_report,
        log=log,
        announce=announce,
        timing_stages=timing_stages,
    )


def _run_real_nep_single(
    config: WorkflowConfig,
    requested_config: WorkflowConfig,
    outdir: Path,
    backend: Any,
    relaxed_atoms: Any,
    relax_info: dict[str, Any],
    log: Callable[[str], None],
    announce: Callable[..., None],
    start_time: float,
    resolved_settings: ResolvedSettings,
    structure_classification: dict[str, Any],
    relax_warnings: list[str],
    supercell_warnings: list[str],
    spacegroup_report: dict[str, Any],
    structure_provenance: dict[str, Any],
    timing_stages: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Run the real ASE-calculator + Phonopy single workflow."""

    warnings: list[str] = [*supercell_warnings, *relax_warnings, *spacegroup_report.get("warnings", [])]
    if config.relax and not relax_info.get("relax_converged", False):
        warnings.append(
            "Structure relaxation did not reach fmax; phonon calculation continued with the final structure."
        )

    phonon_info = run_phonon_calculation(relaxed_atoms, backend, config, outdir, log=log)
    phonon_timing = phonon_info.get("timing_breakdown") or {}
    timing_stages["fc2_harmonic"] = _timing_stage(
        "Harmonic force constants / FC2",
        float(phonon_timing.get("fc2_harmonic_seconds") or 0.0),
    )
    timing_stages["phonon_postprocess"] = _timing_stage(
        "Phonon band / DOS / group velocity",
        float(phonon_timing.get("phonon_postprocess_seconds") or 0.0),
    )
    announce(
        4,
        "Harmonic force constants / FC2",
        log=log,
        details=[
            f"backend: {config.backend_alias or config.backend} -> {backend.name}",
            f"supercell: {config.supercell_dim}",
            f"displacement: {config.displacement}",
            f"stage time: {timing_stages['fc2_harmonic']['seconds']:.3f} s",
        ],
        status="completed",
    )
    announce(
        5,
        "Phonon band / DOS / group velocity",
        log=log,
        details=[
            f"DOS requested: {config.dos}",
            f"group velocity available: {(phonon_info.get('group_velocity') or {}).get('available')}",
            f"stage time: {timing_stages['phonon_postprocess']['seconds']:.3f} s",
        ],
        status="completed",
    )
    thermal_info = run_thermal_conductivity_workflow(relaxed_atoms, backend, config, outdir, log=log)
    thermal_timing = thermal_info.get("timing_breakdown") or {}
    if config.compute_kappa:
        timing_stages["fc3"] = _timing_stage(
            "Third-order force constants / FC3",
            float(thermal_timing.get("fc3_seconds") or 0.0),
            status="completed" if thermal_info.get("available") else "failed",
            reason=thermal_info.get("reason"),
        )
        timing_stages["thermal_lifetime"] = _timing_stage(
            "Thermal conductivity / lifetime",
            float(thermal_timing.get("thermal_lifetime_seconds") or 0.0),
            status="completed" if thermal_info.get("available") else "failed",
            reason=thermal_info.get("reason"),
        )
    else:
        reason = "skipped because compute_kappa=false"
        timing_stages["fc3"] = _timing_stage(
            "Third-order force constants / FC3",
            0.0,
            status="skipped",
            reason=reason,
        )
        timing_stages["thermal_lifetime"] = _timing_stage(
            "Thermal conductivity / lifetime",
            0.0,
            status="skipped",
            reason=reason,
        )
    announce(
        6,
        "Third-order force constants / FC3",
        log=log,
        details=[
            f"compute kappa: {config.compute_kappa}",
            f"fc3 supercell: {config.fc3_supercell_dim}",
            f"stage time: {timing_stages['fc3']['seconds']:.3f} s",
            *([str(timing_stages["fc3"]["reason"])] if timing_stages["fc3"].get("reason") else []),
        ],
        status=str(timing_stages["fc3"]["status"]),
    )
    announce(
        7,
        "Thermal conductivity / lifetime",
        log=log,
        details=[
            f"kappa method: {config.kappa_method}",
            f"WTE / Wigner: {config.wigner}",
            f"stage time: {timing_stages['thermal_lifetime']['seconds']:.3f} s",
            *(
                [str(timing_stages["thermal_lifetime"]["reason"])]
                if timing_stages["thermal_lifetime"].get("reason")
                else []
            ),
        ],
        status=str(timing_stages["thermal_lifetime"]["status"]),
    )
    reporting_started = time.perf_counter()
    timing_breakdown = _write_timing_breakdown(outdir, timing_stages, start_time)
    timing_statistics = write_calculation_time_statistics(
        outdir,
        [
            timing_row_from_breakdown(
                model=config.backend_alias or config.backend,
                display_name=config.dpa_model_name or (config.backend_alias or config.backend),
                timing_breakdown=timing_breakdown,
                compute_kappa=config.compute_kappa,
            )
        ],
    )
    stability = analyze_stability(phonon_info["frequencies_THz"], imag_threshold=config.imag_threshold)
    frequencies = np.asarray(phonon_info["frequencies_THz"], dtype=float)

    output_files = {
        "input_structure": "input_structure.vasp",
        "relaxed_structure": "relaxed.vasp",
        "fc_source_structure": "fc_source_structure.vasp",
        "structure_provenance": "structure_provenance.json",
        "relax_log": "relax.log",
        "resolved_config": "resolved_config.yaml",
        "resolved_settings_json": "resolved_settings.json",
        "resolved_settings_yaml": "resolved_settings.yaml",
        "resolved_settings_table": "resolved_settings_table.txt",
        "timing_breakdown": "timing_breakdown.json",
        "calculation_time_statistics_png": "calculation_time_statistics.png",
        "calculation_time_statistics_csv": "calculation_time_statistics.csv",
        "calculation_time_statistics_json": "calculation_time_statistics.json",
        "run_command": "run_command.txt",
        "spacegroup_report_json": "spacegroup_report.json",
        "spacegroup_report_txt": "spacegroup_report.txt",
        **phonon_info["output_files"],
        **_thermal_output_files(thermal_info),
    }
    output_files["summary"] = "summary.txt"
    warnings.extend(phonon_info.get("postprocess_warnings", []))
    warnings.extend(phonon_info.get("force_constants_text_warnings", []))
    structure_info = _structure_info(relaxed_atoms)
    relax_result = _relax_result_fields(config, relax_info)
    result = {
        "project": PROJECT_NAME,
        "version": VERSION,
        "success": True,
        "dry_run": False,
        "settings_summary": resolved_settings.to_dict(),
        "default_audit": build_default_audit_table(),
        "resolved_settings_file": "resolved_settings.json",
        "resolved_config_file": "resolved_config.yaml",
        "run_command_file": "run_command.txt",
        "backend": requested_config.backend,
        "backend_requested": requested_config.backend,
        "backend_resolved": backend.name,
        "backend_alias": config.backend_alias or getattr(backend, "backend_alias", requested_config.backend),
        "dpa_model_name": config.dpa_model_name,
        "model_backend_family": "deepmd" if backend.name == "deepmd" else "nep",
        "force_infer_backend": (
            "ase/deepmd.calculator.DP" if backend.name == "deepmd" else "ase/calorine.calculators.CPUNEP"
        ),
        "deepmd_reuse_calculator": bool(config.deepmd_reuse_calculator) if backend.name == "deepmd" else None,
        "deepmd_force_backend": config.deepmd_force_backend if backend.name == "deepmd" else None,
        "deepmd_device": config.deepmd_device if backend.name == "deepmd" else None,
        "deepmd_model_head": config.deepmd_model_head if backend.name == "deepmd" else None,
        "dp_infer_batch_size": _deepmd_infer_batch_size() if backend.name == "deepmd" else None,
        "deepmd_deterministic": bool(config.deepmd_deterministic) if backend.name == "deepmd" else None,
        "deepmd_deterministic_warnings": getattr(backend, "deterministic_warnings", []),
        "deepmd_version": _package_version("deepmd-kit") if backend.name == "deepmd" else None,
        "torch_version": _package_version("torch") if backend.name == "deepmd" else None,
        "save_force_audit": bool(config.save_force_audit) if backend.name == "deepmd" else None,
        "phono3py_symmetrize_fc2": bool(config.phono3py_symmetrize_fc2),
        "phono3py_symmetrize_fc3": bool(config.phono3py_symmetrize_fc3),
        "input_path": str(config.input_path),
        "model_path": str(config.model_path),
        "force_backend": backend.name,
        "force_model_path": str(config.model_path),
        "relax_backend": relax_info.get("relax_backend", "none"),
        "relax_model_path": relax_info.get("relax_model_path"),
        "relax_enabled": bool(config.relax),
        **_relax_policy_metadata(
            config,
            relax_info,
            relaxed_structure_path=outdir / "relaxed.vasp" if config.relax else None,
        ),
        "input_file_hash": sha256_file(config.input_path),
        "model_file_hash": sha256_file(config.model_path),
        "output_directory": str(outdir),
        **structure_info,
        **_classification_result_fields(structure_classification),
        **_spacegroup_result_fields(spacegroup_report),
        "relax": bool(config.relax),
        "relax_backend_requested": config.relax_backend,
        "relax_backend_resolved": relax_info.get("relax_backend", "none"),
        "force_backend_resolved": backend.name,
        "force_backend_alias": config.backend_alias or getattr(backend, "backend_alias", requested_config.backend),
        "relax_cell": bool(config.relax_cell) if config.relax else False,
        "relax_mode": _relax_mode(config),
        "constant_cell": _constant_cell(config),
        "relax_converged": relax_info.get("relax_converged"),
        "final_max_force_eV_per_A": relax_info.get("final_max_force_eV_per_A"),
        "final_stress_GPa": relax_info.get("final_stress_GPa"),
        "n_relax_steps": relax_info.get("n_steps"),
        "optimizer": relax_info.get("optimizer"),
        "fmax": config.fmax,
        "max_steps": config.max_steps,
        **relax_result,
        "relax_info": relax_info,
        "relax_warnings": relax_info.get("warnings", []),
        "supercell_dim": config.supercell_dim,
        "supercell_dim_requested": requested_config.supercell_dim,
        "supercell_dim_resolved": config.supercell_dim,
        "supercell_info": config.supercell_info,
        "target_supercell_length": config.target_supercell_length,
        "min_supercell_dim": config.min_supercell_dim,
        "max_supercell_dim": config.max_supercell_dim,
        "max_supercell_atoms": config.max_supercell_atoms,
        **_supercell_result_fields(config, relaxed_atoms),
        "displacement": config.displacement,
        "fc_method": config.fc_method,
        "n_displaced_supercells": phonon_info["n_displaced_supercells"],
        "primitive_matrix_requested": requested_config.primitive_matrix,
        "primitive_matrix_resolved": phonon_info.get("primitive_matrix_resolved"),
        "software_versions": _software_versions(backend.name, phonon_info.get("phonopy_version")),
        "phonopy_version": phonon_info.get("phonopy_version"),
        "calorine_version": _package_version("calorine") if backend.name == "calorine" else None,
        "ase_version": ase.__version__,
        "numpy_version": np.__version__,
        "band": config.band,
        "kpath_mode": config.kpath_mode,
        "kpath_mode_resolved": (phonon_info.get("kpath") or {}).get("resolved_mode"),
        "kpath_dimensionality": (phonon_info.get("kpath") or {}).get("dimensionality"),
        "kpath_source": (phonon_info.get("kpath") or {}).get("source"),
        "kpath_bravais": (phonon_info.get("kpath") or {}).get("bravais"),
        "vacuum_axis_name": (phonon_info.get("kpath") or {}).get("vacuum_axis_name"),
        "band_path_labels": phonon_info.get("band_labels"),
        "band_path_distances": phonon_info.get("band_distances"),
        "band_path_tick_labels": phonon_info.get("band_tick_labels"),
        "band_tick_labels": phonon_info.get("band_tick_labels"),
        "high_symmetry_path": phonon_info.get("high_symmetry_path"),
        "bandpath_symprec": phonon_info.get("bandpath_symprec"),
        "bandpath_with_time_reversal": phonon_info.get("bandpath_with_time_reversal"),
        "bandpath_structure_source": phonon_info.get("bandpath_structure_source"),
        "kpath": phonon_info.get("kpath"),
        "band_metadata": phonon_info.get("band_metadata"),
        "band_diagnostics": phonon_info.get("band_diagnostics"),
        "dos_diagnostics": phonon_info.get("dos_diagnostics"),
        "band_path_npoints": config.band_npoints,
        "minimum_frequency_THz": stability["minimum_frequency_THz"],
        "maximum_frequency_THz": float(np.max(frequencies)),
        "has_imaginary_frequency": stability["has_imaginary_frequency"],
        "dynamically_stable": stability["dynamically_stable"],
        "imag_threshold_THz": stability["imag_threshold_THz"],
        "imaginary_mode_count": stability["imaginary_mode_count"],
        "imaginary_mode_ratio": stability["imaginary_mode_ratio"],
        "dos": bool(config.dos and phonon_info.get("dos_generated", False)),
        "mesh_requested": requested_config.mesh,
        "mesh_resolved": config.mesh,
        **_q_mesh_metadata(config),
        "asr_requested": config.asr,
        "asr_applied": phonon_info.get("asr_applied"),
        "symmetrize_fc_requested": config.symmetrize_fc,
        "symmetrize_fc_applied": phonon_info.get("symmetrize_fc_applied"),
        "export_fc2_text": phonon_info.get("export_fc2_text"),
        "force_constants_text_exported": phonon_info.get("force_constants_text_exported"),
        "force_constants_text_format": phonon_info.get("force_constants_text_format"),
        "force_constants_text_shape": phonon_info.get("force_constants_text_shape"),
        "phonopy_force_constants_file": phonon_info.get("phonopy_force_constants_file"),
        "shengbte_fc2_file": phonon_info.get("shengbte_fc2_file"),
        "group_velocity": phonon_info.get("group_velocity"),
        "compute_kappa": config.compute_kappa,
        "fc3_method": config.fc3_method,
        "kappa_method": config.kappa_method,
        "solver_flags": _thermal_solver_flags(config),
        "method_flags": _thermal_solver_flags(config),
        "compare_mode": bool(config.backend_alias),
        "wigner": config.wigner,
        "temperatures": config.temperatures,
        "kappa_mesh": config.kappa_mesh,
        "fc3_supercell_dim": config.fc3_supercell_dim,
        "thermal_conductivity": thermal_info,
        "wte_capability": get_wte_backend_capability(),
        "timing_breakdown": timing_breakdown,
        "calculation_time_statistics": timing_statistics,
        "structure_provenance": structure_provenance,
        "band_source": phonon_info["band_source"],
        "spacegroup_symprec": config.phonopy_symprec,
        "overwrite": config.overwrite,
        "resume": config.resume,
        "output_files": output_files,
        "warnings": warnings,
        "elapsed_time_seconds": round(time.perf_counter() - start_time, 3),
        "notes": f"Real {backend.name} + Phonopy single workflow result.",
    }
    write_summary(result, outdir / "summary.txt")
    write_json(result, outdir / "result.json")
    write_stability_json(result, outdir / "stability_report.json")
    write_stability_text(result, outdir / "stability_report.txt")
    timing_stages["reporting_packaging"] = _timing_stage(
        "Writing reports and artifacts",
        time.perf_counter() - reporting_started,
    )
    timing_breakdown = _write_timing_breakdown(outdir, timing_stages, start_time)
    timing_statistics = write_calculation_time_statistics(
        outdir,
        [
            timing_row_from_breakdown(
                model=config.backend_alias or config.backend,
                display_name=config.dpa_model_name or (config.backend_alias or config.backend),
                timing_breakdown=timing_breakdown,
                compute_kappa=config.compute_kappa,
            )
        ],
    )
    result["timing_breakdown"] = timing_breakdown
    result["calculation_time_statistics"] = timing_statistics
    result["elapsed_time_seconds"] = timing_breakdown["total_seconds"]
    write_summary(result, outdir / "summary.txt")
    write_json(result, outdir / "result.json")
    announce(
        8,
        "Writing reports and artifacts",
        log=log,
        status="completed",
        details=[
            f"output directory: {outdir}",
            "result.json: written",
            "summary.txt: written",
            f"stage time: {timing_stages['reporting_packaging']['seconds']:.3f} s",
        ],
    )
    log("Workflow completed successfully")

    return {
        "structure_name": Path(config.input_path).name if config.input_path else "",
        "status": "success",
        "outdir": str(outdir),
        "backend": backend.name,
        "relax_info": relax_info,
        "frequencies_THz": np.asarray(phonon_info["frequencies_THz"], dtype=float).tolist(),
        "report": result,
        "dynamically_stable": result["dynamically_stable"],
        "minimum_frequency_THz": result["minimum_frequency_THz"],
        "error_message": "",
    }


def _run_dummy_single(
    config: WorkflowConfig,
    requested_config: WorkflowConfig,
    outdir: Path,
    relaxed_atoms: Any,
    relax_info: dict[str, Any],
    start_time: float,
    resolved_settings: ResolvedSettings,
    structure_classification: dict[str, Any],
    relax_warnings: list[str],
    supercell_warnings: list[str],
    spacegroup_report: dict[str, Any],
    log: Callable[[str], None],
    announce: Callable[..., None],
    timing_stages: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    timing_stages["fc2_harmonic"] = _timing_stage(
        "Harmonic force constants / FC2",
        0.0,
        status="skipped",
        reason="skipped because backend=dummy",
    )
    timing_stages["phonon_postprocess"] = _timing_stage(
        "Phonon band / DOS / group velocity",
        0.0,
        status="skipped",
        reason="skipped because backend=dummy",
    )
    thermal_reason = (
        "skipped because backend=dummy"
        if config.compute_kappa
        else "skipped because compute_kappa=false"
    )
    timing_stages["fc3"] = _timing_stage(
        "Third-order force constants / FC3",
        0.0,
        status="skipped",
        reason=thermal_reason,
    )
    timing_stages["thermal_lifetime"] = _timing_stage(
        "Thermal conductivity / lifetime",
        0.0,
        status="skipped",
        reason=thermal_reason,
    )
    for step, key in (
        (4, "fc2_harmonic"),
        (5, "phonon_postprocess"),
        (6, "fc3"),
        (7, "thermal_lifetime"),
    ):
        stage = timing_stages[key]
        announce(
            step,
            stage["label"],
            log=log,
            status=stage["status"],
            details=[str(stage["reason"])],
        )
    timing_breakdown = _write_timing_breakdown(outdir, timing_stages, start_time)
    timing_statistics = write_calculation_time_statistics(
        outdir,
        [
            timing_row_from_breakdown(
                model=config.backend_alias or config.backend,
                display_name=config.dpa_model_name or (config.backend_alias or config.backend),
                timing_breakdown=timing_breakdown,
                compute_kappa=config.compute_kappa,
            )
        ],
    )
    dummy_frequencies = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    stability = analyze_stability(dummy_frequencies, imag_threshold=config.imag_threshold)
    report = build_stability_report(
        backend=config.backend,
        stability=stability,
        notes="Dummy backend result for workflow testing only.",
    )
    output_files = {
        "input_structure": "input_structure.vasp",
        "relaxed_structure": "relaxed.vasp",
        "resolved_config": "resolved_config.yaml",
        "resolved_settings_json": "resolved_settings.json",
        "resolved_settings_yaml": "resolved_settings.yaml",
        "resolved_settings_table": "resolved_settings_table.txt",
        "timing_breakdown": "timing_breakdown.json",
        "calculation_time_statistics_png": "calculation_time_statistics.png",
        "calculation_time_statistics_csv": "calculation_time_statistics.csv",
        "calculation_time_statistics_json": "calculation_time_statistics.json",
        "run_command": "run_command.txt",
        "spacegroup_report_json": "spacegroup_report.json",
        "spacegroup_report_txt": "spacegroup_report.txt",
        "summary": "summary.txt",
    }
    if relax_warnings:
        relax_info["warnings"] = list(relax_info.get("warnings") or []) + relax_warnings
    result = {
        "project": PROJECT_NAME,
        "version": VERSION,
        "success": True,
        "dry_run": False,
        "settings_summary": resolved_settings.to_dict(),
        "default_audit": build_default_audit_table(),
        "resolved_settings_file": "resolved_settings.json",
        "resolved_config_file": "resolved_config.yaml",
        "run_command_file": "run_command.txt",
        "backend": requested_config.backend,
        "backend_requested": requested_config.backend,
        "backend_resolved": config.backend,
        "backend_alias": config.backend_alias or requested_config.backend,
        "dpa_model_name": config.dpa_model_name,
        "model_backend_family": "deepmd" if config.backend == "deepmd" else "nep" if config.backend == "calorine" else config.backend,
        "force_infer_backend": "ase/deepmd.calculator.DP" if config.backend == "deepmd" else None,
        "deepmd_reuse_calculator": bool(config.deepmd_reuse_calculator) if config.backend == "deepmd" else None,
        "deepmd_force_backend": config.deepmd_force_backend if config.backend == "deepmd" else None,
        "deepmd_device": config.deepmd_device if config.backend == "deepmd" else None,
        "deepmd_model_head": config.deepmd_model_head if config.backend == "deepmd" else None,
        "dp_infer_batch_size": _deepmd_infer_batch_size() if config.backend == "deepmd" else None,
        "deepmd_deterministic": bool(config.deepmd_deterministic) if config.backend == "deepmd" else None,
        "deepmd_version": _package_version("deepmd-kit") if config.backend == "deepmd" else None,
        "torch_version": _package_version("torch") if config.backend == "deepmd" else None,
        "save_force_audit": bool(config.save_force_audit) if config.backend == "deepmd" else None,
        "phono3py_symmetrize_fc2": bool(config.phono3py_symmetrize_fc2),
        "phono3py_symmetrize_fc3": bool(config.phono3py_symmetrize_fc3),
        "input_path": str(config.input_path),
        "model_path": str(config.model_path) if config.model_path else None,
        "force_backend": config.backend,
        "force_model_path": str(config.model_path) if config.model_path else None,
        "relax_backend": relax_info.get("relax_backend", "none"),
        "relax_model_path": relax_info.get("relax_model_path"),
        "relax_enabled": bool(config.relax),
        **_relax_policy_metadata(
            config,
            relax_info,
            relaxed_structure_path=outdir / "relaxed.vasp" if config.relax else None,
        ),
        "input_file_hash": sha256_file(config.input_path),
        "model_file_hash": sha256_file(config.model_path),
        "output_directory": str(outdir),
        **_structure_info(relaxed_atoms),
        **_classification_result_fields(structure_classification),
        **_spacegroup_result_fields(spacegroup_report),
        "relax": bool(config.relax),
        "relax_backend_requested": config.relax_backend,
        "relax_backend_resolved": relax_info.get("relax_backend", "none"),
        "force_backend_resolved": config.backend,
        "force_backend_alias": config.backend_alias or requested_config.backend,
        "relax_cell": bool(config.relax_cell) if config.relax else False,
        "relax_mode": _relax_mode(config),
        "constant_cell": _constant_cell(config),
        "relax_converged": relax_info.get("relax_converged"),
        "final_max_force_eV_per_A": relax_info.get("final_max_force_eV_per_A"),
        "final_stress_GPa": relax_info.get("final_stress_GPa"),
        "n_relax_steps": relax_info.get("n_steps"),
        "optimizer": config.optimizer,
        "fmax": config.fmax,
        "max_steps": config.max_steps,
        **_relax_result_fields(config, relax_info),
        "relax_info": relax_info,
        "relax_warnings": relax_info.get("warnings", []),
        "supercell_dim_requested": requested_config.supercell_dim,
        "supercell_dim_resolved": config.supercell_dim,
        "supercell_info": config.supercell_info,
        **_supercell_result_fields(config, relaxed_atoms),
        "mesh_requested": requested_config.mesh,
        "mesh_resolved": config.mesh,
        **_q_mesh_metadata(config),
        "primitive_matrix_requested": requested_config.primitive_matrix,
        "primitive_matrix_resolved": config.primitive_matrix,
        "kpath_mode": config.kpath_mode,
        "kpath_mode_resolved": None,
        "kpath_dimensionality": None,
        "kpath_source": None,
        "kpath_bravais": None,
        "vacuum_axis_name": None,
        "kpath": None,
        "bandpath_symprec": config.bandpath_symprec,
        "bandpath_with_time_reversal": config.bandpath_with_time_reversal,
        "bandpath_structure_source": "input_structure",
        "spacegroup_symprec": config.phonopy_symprec,
        "fc_method": config.fc_method,
        "compute_kappa": config.compute_kappa,
        "fc3_method": config.fc3_method,
        "kappa_method": config.kappa_method,
        "solver_flags": _thermal_solver_flags(config),
        "method_flags": _thermal_solver_flags(config),
        "compare_mode": bool(config.backend_alias),
        "wigner": config.wigner,
        "temperatures": config.temperatures,
        "kappa_mesh": config.kappa_mesh,
        "fc3_supercell_dim": config.fc3_supercell_dim,
        "export_fc2_text": config.export_fc2_text,
        "force_constants_text_exported": False,
        "group_velocity": {
            "available": False,
            "reason": "Group velocity is not generated for the dummy backend.",
            "data_file": None,
            "plot_file": None,
        },
        "thermal_conductivity": disabled_thermal_result(),
        "wte_capability": get_wte_backend_capability(),
        "timing_breakdown": timing_breakdown,
        "calculation_time_statistics": timing_statistics,
        "software_versions": _software_versions(config.backend, None),
        "frequencies_THz": dummy_frequencies.tolist(),
        "report": report,
        "dynamically_stable": report["dynamically_stable"],
        "minimum_frequency_THz": report["minimum_frequency_THz"],
        "maximum_frequency_THz": float(np.max(dummy_frequencies)),
        "has_imaginary_frequency": report["has_imaginary_frequency"],
        "output_files": output_files,
        "warnings": [*supercell_warnings, *relax_warnings, *spacegroup_report.get("warnings", [])],
        "elapsed_time_seconds": round(time.perf_counter() - start_time, 3),
    }
    write_summary(result, outdir / "summary.txt")
    write_json(result, outdir / "result.json")
    write_stability_json(result, outdir / "stability_report.json")
    write_stability_text(result, outdir / "stability_report.txt")
    announce(
        8,
        "Writing reports and artifacts",
        log=log,
        status="completed",
        details=[f"output directory: {outdir}"],
    )
    log("Dummy workflow completed successfully")
    return {
        "structure_name": Path(config.input_path).name,
        "status": "success",
        "outdir": str(outdir),
        "backend": config.backend,
        "relax_info": relax_info,
        "frequencies_THz": dummy_frequencies.tolist(),
        "report": result,
        "dynamically_stable": result["dynamically_stable"],
        "minimum_frequency_THz": result["minimum_frequency_THz"],
        "error_message": "",
    }


def _build_dry_run_result(
    config: WorkflowConfig,
    requested_config: WorkflowConfig,
    atoms: Any,
    resolved_settings: ResolvedSettings,
    start_time: float,
    structure_classification: dict[str, Any],
    relax_warnings: list[str],
    supercell_warnings: list[str],
    spacegroup_report: dict[str, Any],
    timing_breakdown: dict[str, Any],
    timing_statistics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "project": PROJECT_NAME,
        "version": VERSION,
        "success": True,
        "dry_run": True,
        "settings_summary": resolved_settings.to_dict(),
        "default_audit": build_default_audit_table(),
        "resolved_settings_file": "resolved_settings.json",
        "resolved_config_file": "resolved_config.yaml",
        "run_command_file": "run_command.txt",
        "backend": requested_config.backend,
        "backend_requested": requested_config.backend,
        "backend_resolved": config.backend,
        "backend_alias": config.backend_alias or requested_config.backend,
        "dpa_model_name": config.dpa_model_name,
        "model_backend_family": "deepmd" if config.backend == "deepmd" else "nep" if config.backend == "calorine" else config.backend,
        "force_infer_backend": "ase/deepmd.calculator.DP" if config.backend == "deepmd" else None,
        "deepmd_reuse_calculator": bool(config.deepmd_reuse_calculator) if config.backend == "deepmd" else None,
        "deepmd_force_backend": config.deepmd_force_backend if config.backend == "deepmd" else None,
        "deepmd_device": config.deepmd_device if config.backend == "deepmd" else None,
        "deepmd_model_head": config.deepmd_model_head if config.backend == "deepmd" else None,
        "dp_infer_batch_size": _deepmd_infer_batch_size() if config.backend == "deepmd" else None,
        "deepmd_deterministic": bool(config.deepmd_deterministic) if config.backend == "deepmd" else None,
        "deepmd_version": _package_version("deepmd-kit") if config.backend == "deepmd" else None,
        "torch_version": _package_version("torch") if config.backend == "deepmd" else None,
        "save_force_audit": bool(config.save_force_audit) if config.backend == "deepmd" else None,
        "phono3py_symmetrize_fc2": bool(config.phono3py_symmetrize_fc2),
        "phono3py_symmetrize_fc3": bool(config.phono3py_symmetrize_fc3),
        "input_path": str(config.input_path),
        "model_path": str(config.model_path) if config.model_path else None,
        "force_backend": config.backend,
        "force_model_path": str(config.model_path) if config.model_path else None,
        "relax_backend": _resolved_relax_backend_name(config),
        "relax_model_path": _resolved_relax_model_path(config),
        "relax_enabled": bool(config.relax),
        **_relax_policy_metadata(
            config,
            _skipped_relax_info(atoms, config),
            relaxed_structure_path=None,
        ),
        "input_file_hash": sha256_file(config.input_path),
        "model_file_hash": sha256_file(config.model_path),
        "output_directory": str(config.outdir),
        **_structure_info(atoms),
        **_classification_result_fields(structure_classification),
        **_spacegroup_result_fields(spacegroup_report),
        "relax": bool(config.relax),
        "relax_backend_requested": config.relax_backend,
        "relax_backend_resolved": "none" if not config.relax else config.relax_backend,
        "force_backend_resolved": config.backend,
        "force_backend_alias": config.backend_alias or requested_config.backend,
        "relax_cell": bool(config.relax_cell) if config.relax else False,
        "relax_mode": _relax_mode(config),
        "constant_cell": _constant_cell(config),
        "fmax": config.fmax,
        "max_steps": config.max_steps,
        "optimizer": config.optimizer,
        **_relax_result_fields(config, _skipped_relax_info(atoms, config)),
        "relax_warnings": list(relax_warnings),
        "supercell_dim_requested": requested_config.supercell_dim,
        "supercell_dim_resolved": config.supercell_dim,
        "supercell_info": config.supercell_info,
        **_supercell_result_fields(config, atoms),
        "mesh_requested": requested_config.mesh,
        "mesh_resolved": config.mesh,
        **_q_mesh_metadata(config),
        "primitive_matrix_requested": requested_config.primitive_matrix,
        "primitive_matrix_resolved": config.primitive_matrix,
        "fc_method": config.fc_method,
        "compute_kappa": config.compute_kappa,
        "fc3_method": config.fc3_method,
        "kappa_method": config.kappa_method,
        "solver_flags": _thermal_solver_flags(config),
        "method_flags": _thermal_solver_flags(config),
        "compare_mode": bool(config.backend_alias),
        "wigner": config.wigner,
        "temperatures": config.temperatures,
        "kappa_mesh": config.kappa_mesh,
        "fc3_supercell_dim": config.fc3_supercell_dim,
        "export_fc2_text": config.export_fc2_text,
        "force_constants_text_exported": False,
        "group_velocity": {
            "available": False,
            "reason": "Dry run only; no phonon group velocity calculation was executed.",
            "data_file": None,
            "plot_file": None,
        },
        "thermal_conductivity": disabled_thermal_result(
            "Dry run only; no thermal conductivity calculation was executed."
        ),
        "wte_capability": get_wte_backend_capability(),
        "timing_breakdown": timing_breakdown,
        "calculation_time_statistics": timing_statistics,
        "software_versions": _software_versions(config.backend, None),
        "overwrite": config.overwrite,
        "resume": config.resume,
        "output_files": {
            "resolved_config": "resolved_config.yaml",
            "resolved_settings_json": "resolved_settings.json",
            "resolved_settings_yaml": "resolved_settings.yaml",
            "resolved_settings_table": "resolved_settings_table.txt",
            "timing_breakdown": "timing_breakdown.json",
            "calculation_time_statistics_png": "calculation_time_statistics.png",
            "calculation_time_statistics_csv": "calculation_time_statistics.csv",
            "calculation_time_statistics_json": "calculation_time_statistics.json",
            "run_command": "run_command.txt",
            "spacegroup_report_json": "spacegroup_report.json",
            "spacegroup_report_txt": "spacegroup_report.txt",
            "summary": "summary.txt",
        },
        "warnings": [
            *supercell_warnings,
            *relax_warnings,
            *spacegroup_report.get("warnings", []),
            "Dry run only; no relaxation or phonon calculation was executed.",
        ],
        "elapsed_time_seconds": round(time.perf_counter() - start_time, 3),
    }


def _timing_stage(
    label: str,
    seconds: float,
    *,
    status: str = "completed",
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "label": label,
        "seconds": round(max(0.0, float(seconds)), 6),
        "status": status,
        "reason": reason,
    }


def _write_timing_breakdown(
    outdir: Path,
    stages: dict[str, dict[str, Any]],
    start_time: float,
) -> dict[str, Any]:
    payload = {
        "unit": "seconds",
        "total_seconds": round(time.perf_counter() - start_time, 6),
        "stages": dict(stages),
    }
    payload["stages"]["fc2_phonon"] = _timing_stage(
        "FC2 / phonon",
        float((stages.get("fc2_harmonic") or {}).get("seconds") or 0.0)
        + float((stages.get("phonon_postprocess") or {}).get("seconds") or 0.0),
        status=(
            "skipped"
            if all(
                (stages.get(key) or {}).get("status") == "skipped"
                for key in ("fc2_harmonic", "phonon_postprocess")
            )
            else "completed"
        ),
    )
    fc3_skipped = all(
        (stages.get(key) or {}).get("status") == "skipped"
        for key in ("fc3", "thermal_lifetime")
    )
    payload["stages"]["fc3_thermal"] = _timing_stage(
        "FC3 / thermal",
        float((stages.get("fc3") or {}).get("seconds") or 0.0)
        + float((stages.get("thermal_lifetime") or {}).get("seconds") or 0.0),
        status="skipped" if fc3_skipped else "completed",
        reason=(
            (stages.get("thermal_lifetime") or {}).get("reason")
            or (stages.get("fc3") or {}).get("reason")
        )
        if fc3_skipped
        else None,
    )
    write_json(payload, outdir / "timing_breakdown.json")
    return payload


def _build_resolved_settings(
    requested_config: WorkflowConfig,
    config: WorkflowConfig,
    atoms: Any,
    outdir_note: str,
    structure_classification: dict[str, Any],
) -> ResolvedSettings:
    explicit = set(requested_config.option_sources)

    def source(name: str, *, auto: bool = False) -> str:
        if name in explicit:
            return "user"
        if auto:
            return "auto"
        return "default"

    n_supercell_atoms = int(len(atoms) * np.prod(config.supercell_dim))
    supercell_info = _supercell_info(config, atoms)
    settings = ResolvedSettings()
    settings.add("input_path", config.input_path, source("input_path"))
    settings.add("model_path", config.model_path, source("model_path"))
    settings.add("output_directory", config.outdir, source("outdir", auto=requested_config.outdir is None), outdir_note)
    settings.add("backend_requested", requested_config.backend, source("backend"))
    settings.add(
        "backend_resolved",
        config.backend,
        "auto" if requested_config.backend == "auto" else source("backend"),
        "Calorine CPUNEP available" if config.backend == "calorine" else "",
    )
    settings.add("backend_alias", config.backend_alias or requested_config.backend, source("backend"))
    settings.add("dpa_model_name", config.dpa_model_name, "auto" if config.dpa_model_name else "default")
    settings.add(
        "selected_models",
        [config.backend_alias or requested_config.backend],
        "compare child" if requested_config.backend_alias else source("backend"),
    )
    settings.add("force_backend", config.backend, "auto")
    settings.add("force_model_path", config.model_path, source("model_path"))
    resolved_relax_backend = _resolved_relax_backend_name(config)
    resolved_relax_model_path = _resolved_relax_model_path(config)
    settings.add("relax_backend_requested", config.relax_backend, source("relax_backend"))
    settings.add(
        "relax_backend_resolved",
        resolved_relax_backend,
        source("relax_backend", auto=config.relax and config.relax_backend == "auto"),
        "DPA/DeepMD relaxation defaults to NEP89/Calorine"
        if config.backend == "deepmd" and config.relax and resolved_relax_backend == "calorine"
        else "",
    )
    settings.add(
        "relax_model_path",
        resolved_relax_model_path,
        source(
            "relax_model_path",
            auto=bool(config.relax and config.backend == "deepmd" and requested_config.relax_model_path is None),
        ),
    )
    settings.add("allow_dpa_relax", config.allow_dpa_relax, source("allow_dpa_relax"))
    settings.add("relax", config.relax, source("relax"))
    settings.add("relax_cell", config.relax_cell if config.relax else False, source("relax_cell"))
    settings.add("relax_mode", _relax_mode(config), source("relax_cell" if config.relax else "relax"))
    settings.add("constant_cell", _constant_cell(config), source("relax_cell"))
    settings.add("fmax", config.fmax, source("fmax"))
    settings.add("max_steps", config.max_steps, source("max_steps"))
    settings.add("optimizer", config.optimizer, source("optimizer"))
    structure_note = ""
    if structure_classification.get("vacuum_like_directions"):
        structure_note = "Detected vacuum-like direction. Consider --no-relax-cell for 2D/slab/interface systems."
    settings.add(
        "structure_type",
        structure_classification.get("structure_type"),
        "auto",
        structure_note,
    )
    settings.add(
        "vacuum_like_directions",
        structure_classification.get("vacuum_like_directions", []),
        "auto",
    )
    settings.add("supercell_dim_requested", requested_config.supercell_dim, source("supercell_dim"))
    settings.add(
        "supercell_dim_resolved",
        config.supercell_dim,
        source("supercell_dim", auto=requested_config.supercell_dim == "auto"),
        (
            f"target length {config.target_supercell_length} A, max atoms {config.max_supercell_atoms}; "
            + "; ".join(supercell_info.get("auto_supercell_notes", []))
        ).strip(),
    )
    settings.add(
        "target_supercell_length",
        config.target_supercell_length,
        source("target_supercell_length"),
        "auto supercell target",
    )
    settings.add("min_supercell_dim", config.min_supercell_dim, source("min_supercell_dim"))
    settings.add("max_supercell_dim", config.max_supercell_dim, source("max_supercell_dim"))
    settings.add("max_supercell_atoms", config.max_supercell_atoms, source("max_supercell_atoms"))
    settings.add("n_atoms_unitcell", len(atoms), "auto")
    settings.add("n_atoms_supercell", n_supercell_atoms, "auto")
    settings.add("supercell_lengths_resolved", supercell_info.get("supercell_lengths_resolved"), "auto")
    settings.add("auto_supercell_warnings", supercell_info.get("auto_supercell_warnings", []), "auto")
    settings.add("auto_supercell_notes", supercell_info.get("auto_supercell_notes", []), "auto")
    settings.add("displacement", config.displacement, source("displacement"))
    settings.add("primitive_matrix_requested", requested_config.primitive_matrix, source("primitive_matrix"))
    settings.add(
        "primitive_matrix_resolved",
        config.primitive_matrix,
        source("primitive_matrix"),
        "explicit to avoid Phonopy v4 implicit primitive-matrix changes",
    )
    settings.add("band", config.band, source("band"))
    settings.add(
        "kpath_mode",
        config.kpath_mode,
        source("kpath_mode"),
        "auto selects ASE 2D slab paths or SeekPath 3D bulk paths.",
    )
    settings.add("band_npoints", config.band_npoints, source("band_npoints"))
    settings.add(
        "bandpath_symprec",
        config.bandpath_symprec,
        source("bandpath_symprec"),
        "3D SeekPath tolerance; also used as the ASE 2D bandpath epsilon floor.",
    )
    settings.add(
        "bandpath_with_time_reversal",
        config.bandpath_with_time_reversal,
        source("bandpath_with_time_reversal"),
        "3D SeekPath time-reversal reduction setting; ignored by ASE 2D paths.",
    )
    settings.add("dos", config.dos, source("dos"))
    settings.add(
        "group_velocity",
        True,
        "default",
        "Generated as part of the harmonic phonon post-processing stage.",
    )
    settings.add("mesh_requested", requested_config.mesh, source("mesh"))
    settings.add(
        "mesh_resolved",
        config.mesh,
        source("mesh", auto=requested_config.mesh == "auto"),
        "fixed default mesh" if requested_config.mesh == "auto" else "",
    )
    settings.add(
        "q_mesh",
        config.mesh,
        source("mesh", auto=requested_config.mesh == "auto" and requested_config.kappa_mesh == "auto"),
        "gamma-centered for DOS and thermal conductivity",
    )
    settings.add("q_mesh_centering", "gamma", "fixed")
    settings.add("asr", config.asr, source("asr"))
    settings.add("symmetrize_fc", config.symmetrize_fc, source("symmetrize_fc"))
    settings.add("export_fc2_text", config.export_fc2_text, source("export_fc2_text"))
    settings.add("fc_method", config.fc_method, source("fc_method"))
    settings.add("compute_kappa", config.compute_kappa, source("compute_kappa"))
    settings.add("fc3_method", config.fc3_method, source("fc3_method"))
    settings.add("kappa_method", config.kappa_method, source("kappa_method"))
    settings.add("wigner", config.wigner, source("wigner"))
    settings.add("temperatures", config.temperatures, source("temperatures"))
    settings.add("kappa_mesh", config.kappa_mesh, source("kappa_mesh"), "compatibility alias for q_mesh")
    settings.add("fc3_supercell_dim", config.fc3_supercell_dim, source("fc3_supercell_dim"))
    settings.add("fc3_target_supercell_length", config.fc3_target_supercell_length, source("fc3_target_supercell_length"))
    settings.add("max_fc3_supercell_atoms", config.max_fc3_supercell_atoms, source("max_fc3_supercell_atoms"))
    settings.add("fc3_displacement", config.fc3_displacement, source("fc3_displacement"))
    settings.add("fc3_cutoff_pair_distance", config.fc3_cutoff_pair_distance, source("fc3_cutoff_pair_distance"))
    settings.add("max_fc3_displacements", config.max_fc3_displacements, source("max_fc3_displacements"))
    settings.add("phono3py_symprec", config.phono3py_symprec, source("phono3py_symprec"))
    settings.add("phono3py_cutoff_frequency", config.phono3py_cutoff_frequency, source("phono3py_cutoff_frequency"))
    settings.add("phono3py_plusminus", config.phono3py_plusminus, source("phono3py_plusminus"))
    settings.add("phono3py_diagonal", config.phono3py_diagonal, source("phono3py_diagonal"))
    settings.add("phono3py_symmetry", config.phono3py_symmetry, source("phono3py_symmetry"))
    settings.add("phono3py_mesh_symmetry", config.phono3py_mesh_symmetry, source("phono3py_mesh_symmetry"))
    settings.add("phono3py_isotope", config.phono3py_isotope, source("phono3py_isotope"))
    settings.add("boundary_mfp", config.boundary_mfp, source("boundary_mfp"))
    settings.add("cutoff_pair_distance", config.cutoff_pair_distance, source("cutoff_pair_distance"))
    settings.add("phono3py_symmetrize_fc2", config.phono3py_symmetrize_fc2, source("phono3py_symmetrize_fc2"))
    settings.add("phono3py_symmetrize_fc3", config.phono3py_symmetrize_fc3, source("phono3py_symmetrize_fc3"))
    if config.backend == "deepmd":
        settings.add("deepmd_reuse_calculator", config.deepmd_reuse_calculator, source("deepmd_reuse_calculator"))
        settings.add("deepmd_force_backend", config.deepmd_force_backend, source("deepmd_force_backend"))
        settings.add("deepmd_device", config.deepmd_device, source("deepmd_device"))
        settings.add("deepmd_model_head", config.deepmd_model_head, source("deepmd_model_head"))
        settings.add("deepmd_deterministic", config.deepmd_deterministic, source("deepmd_deterministic"))
    settings.add(
        "dpa_hidden_defaults",
        {
            "device": config.deepmd_device,
            "force_backend": config.deepmd_force_backend,
            "reuse_calculator": config.deepmd_reuse_calculator,
            "deterministic": config.deepmd_deterministic,
            "model_head": config.deepmd_model_head,
        }
        if config.backend == "deepmd"
        else {},
        "backend default" if config.backend == "deepmd" else "default",
    )
    if config.backend == "deepmd":
        settings.add("save_force_audit", config.save_force_audit, source("save_force_audit"))
    settings.add("n_structures", config.n_structures, source("n_structures"))
    settings.add("rattle_std", config.rattle_std, source("rattle_std"))
    settings.add("cutoffs", config.cutoffs, source("cutoffs"))
    settings.add("min_dist", config.min_dist, source("min_dist"))
    settings.add(
        "phonopy_symprec",
        config.phonopy_symprec,
        source("phonopy_symprec"),
        "phonopy SYMMETRY_TOLERANCE / Python API symprec; default 1e-5",
    )
    settings.add("spacegroup_symprec", config.phonopy_symprec, source("phonopy_symprec"), "alias for space-group diagnostics")
    settings.add("angle_tolerance", config.angle_tolerance, source("angle_tolerance"))
    settings.add("overwrite", config.overwrite, source("overwrite"))
    settings.add("resume", config.resume, source("resume"))
    settings.add("wte_capability", get_wte_backend_capability(), "auto")
    return settings


def _resolved_workflow_details(config: WorkflowConfig, requested_config: WorkflowConfig, outdir: Path) -> list[str]:
    """Return concise resolved settings for terminal and validation logs."""

    phono3py_symprec = config.phono3py_symprec if config.phono3py_symprec is not None else 1e-5
    details = [
        f"backend: {requested_config.backend} -> {config.backend}",
        f"backend alias: {config.backend_alias or requested_config.backend}",
        f"model path: {config.model_path}",
        f"dpa model: {config.dpa_model_name or 'not applicable'}",
        f"supercell: {config.supercell_dim} (requested: {requested_config.supercell_dim}, source: {config.supercell_info.get('source')})",
        f"target supercell length: {config.target_supercell_length}",
        f"q-mesh (DOS/kappa): {config.mesh}",
        "q-mesh centering: gamma",
        f"fc3 supercell: {config.fc3_supercell_dim}",
        f"phonopy symprec: {config.phonopy_symprec}",
        f"phono3py symprec: {phono3py_symprec}",
        f"cutoff frequency: {config.phono3py_cutoff_frequency}",
        f"phono3py_symmetrize_fc2: {config.phono3py_symmetrize_fc2}",
        f"phono3py_symmetrize_fc3: {config.phono3py_symmetrize_fc3}",
        f"relax requested: {config.relax}",
        f"relax backend requested: {config.relax_backend}",
        f"relax backend resolved: {_resolved_relax_backend_name(config)}",
        f"relax model path: {_resolved_relax_model_path(config) or 'none'}",
        f"output directory: {outdir}",
    ]
    if config.backend == "deepmd":
        details[12:12] = [
            f"deterministic: {config.deepmd_deterministic}",
            f"reuse calculator: {config.deepmd_reuse_calculator}",
            f"force audit: {config.save_force_audit}",
        ]
    return details


def _select_relax_backend(config: WorkflowConfig, force_backend: Any) -> Any:
    """Return the backend used for structure relaxation."""

    if not config.relax:
        return force_backend
    requested = str(config.relax_backend or "auto").lower()
    if force_backend.name != "deepmd":
        return force_backend
    if config.allow_dpa_relax or requested in {"dpa", "deepmd", "force"}:
        return force_backend
    if requested not in {"auto", "calorine", "nep89"}:
        raise ConfigError(
            f"Unsupported relax backend '{config.relax_backend}'. "
            "Use auto, calorine, nep89, dpa, deepmd, or force."
        )

    relax_model_path = Path(config.relax_model_path or DEFAULT_NEP89_MODEL_PATH)
    relax_backend = get_backend("calorine", model_path=relax_model_path)
    if not relax_model_path.exists():
        raise ConfigError(
            "DPA structure relaxation defaults to NEP89/Calorine, but the NEP89 relax model "
            f"was not found: {relax_model_path}. Provide --relax-model-path, use --no-relax, "
            "or explicitly allow DPA relaxation with --allow-dpa-relax."
        )
    if not relax_backend.check_available():
        raise BackendUnavailableError(
            "DPA structure relaxation defaults to NEP89/Calorine, but Calorine is not available. "
            "Install Calorine, use --no-relax, or explicitly allow DPA relaxation with --allow-dpa-relax."
        )
    return relax_backend


def _resolved_relax_backend_name(config: WorkflowConfig) -> str:
    if not config.relax:
        return "none"
    requested = str(config.relax_backend or "auto").lower()
    if config.backend != "deepmd":
        return config.backend
    if config.allow_dpa_relax or requested in {"dpa", "deepmd", "force"}:
        return "deepmd"
    return "calorine"


def _resolved_relax_model_path(config: WorkflowConfig) -> str | None:
    backend_name = _resolved_relax_backend_name(config)
    if backend_name == "none":
        return None
    if backend_name == "calorine" and config.backend == "deepmd":
        return str(config.relax_model_path or DEFAULT_NEP89_MODEL_PATH)
    return str(config.model_path) if config.model_path else None


def _resolve_output_directory(config: WorkflowConfig) -> tuple[Path, str]:
    outdir = Path(config.outdir) if config.outdir is not None else Path("results")
    note = ""
    if outdir.exists() and (outdir / "result.json").exists() and not config.overwrite and not config.resume:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_outdir = outdir.with_name(f"{outdir.name}_{timestamp}")
        note = f"Existing result.json found; using {new_outdir} to avoid overwrite."
        return new_outdir, note
    if config.overwrite:
        note = "overwrite=True; existing files may be replaced."
    return outdir, note


def _write_run_commands(config: WorkflowConfig, outdir: Path, requested_config: WorkflowConfig) -> None:
    fallback = [
        "python",
        "-m",
        "phonoflow",
        "run" if requested_config.backend == "auto" else "single",
        "--input-path",
        str(config.input_path),
        "--model-path",
        str(config.model_path),
    ]
    original = config.run_command or build_run_command(None, fallback)
    resolved = _resolved_command(config)
    (outdir / "run_command.txt").write_text(
        f"original_command: {original}\nresolved_command: {resolved}\n",
        encoding="utf-8",
    )


def _resolved_command(config: WorkflowConfig) -> str:
    command = [
        "python",
        "-m",
        "phonoflow",
        "single",
        "--input-path",
        str(config.input_path),
        "--model-path",
        str(config.model_path),
        "--outdir",
        str(config.outdir),
        "--backend",
        str(config.backend),
        "--supercell-dim",
        *[str(item) for item in config.supercell_dim],
        "--mesh",
        *[str(item) for item in config.mesh],
        "--primitive-matrix",
        config.primitive_matrix,
        "--displacement",
        str(config.displacement),
        "--target-supercell-length",
        str(config.target_supercell_length),
        "--max-supercell-atoms",
        str(config.max_supercell_atoms),
        "--min-supercell-dim",
        str(config.min_supercell_dim),
        "--max-supercell-dim",
        str(config.max_supercell_dim),
        "--fmax",
        str(config.fmax),
        "--max-steps",
        str(config.max_steps),
        "--phonopy-symprec",
        str(config.phonopy_symprec),
        "--kpath-mode",
        str(config.kpath_mode),
        "--bandpath-symprec",
        str(config.bandpath_symprec),
        "--bandpath-with-time-reversal"
        if config.bandpath_with_time_reversal
        else "--no-bandpath-with-time-reversal",
        "--angle-tolerance",
        str(config.angle_tolerance),
        "--band",
        config.band,
    ]
    command.append("--relax" if config.relax else "--no-relax")
    if config.relax:
        command.append("--relax-cell" if config.relax_cell else "--no-relax-cell")
    command.append("--dos" if config.dos else "--no-dos")
    command.append("--asr" if config.asr else "--no-asr")
    command.append("--symmetrize-fc" if config.symmetrize_fc else "--no-symmetrize-fc")
    command.append("--export-fc2-text" if config.export_fc2_text else "--no-export-fc2-text")
    command.append("--compute-kappa" if config.compute_kappa else "--no-compute-kappa")
    command.extend(["--fc3-method", config.fc3_method])
    command.extend(["--method", config.kappa_method])
    command.extend(["--wigner", "true" if config.wigner else "false"])
    command.append("--temperatures")
    command.extend(str(value) for value in config.temperatures)
    command.append("--kappa-mesh")
    if config.kappa_mesh == "auto":
        command.append("auto")
    else:
        command.extend(str(value) for value in config.kappa_mesh)
    command.append("--fc3-supercell-dim")
    if config.fc3_supercell_dim == "auto":
        command.append("auto")
    else:
        command.extend(str(value) for value in config.fc3_supercell_dim)
    command.extend(["--fc3-target-supercell-length", str(config.fc3_target_supercell_length)])
    command.extend(["--max-fc3-supercell-atoms", str(config.max_fc3_supercell_atoms)])
    command.extend(["--fc3-displacement", str(config.fc3_displacement)])
    if config.fc3_cutoff_pair_distance is not None:
        command.extend(["--fc3-cutoff-pair-distance", str(config.fc3_cutoff_pair_distance)])
    if config.max_fc3_displacements is not None:
        command.extend(["--max-fc3-displacements", str(config.max_fc3_displacements)])
    if config.phono3py_symprec is not None:
        command.extend(["--phono3py-symprec", str(config.phono3py_symprec)])
    if config.phono3py_cutoff_frequency is not None:
        command.extend(["--phono3py-cutoff-frequency", str(config.phono3py_cutoff_frequency)])
    command.extend(["--phono3py-plusminus", str(config.phono3py_plusminus)])
    command.append("--phono3py-diagonal" if config.phono3py_diagonal else "--no-phono3py-diagonal")
    command.append("--phono3py-symmetry" if config.phono3py_symmetry else "--no-phono3py-symmetry")
    command.append("--phono3py-mesh-symmetry" if config.phono3py_mesh_symmetry else "--no-phono3py-mesh-symmetry")
    command.append("--isotope" if config.phono3py_isotope else "--no-isotope")
    command.extend(["--boundary-mfp", str(config.boundary_mfp)])
    command.extend(["--cutoff-pair-distance", str(config.cutoff_pair_distance)])
    command.append("--phono3py-symmetrize-fc2" if config.phono3py_symmetrize_fc2 else "--no-phono3py-symmetrize-fc2")
    command.append("--phono3py-symmetrize-fc3" if config.phono3py_symmetrize_fc3 else "--no-phono3py-symmetrize-fc3")
    if config.backend == "deepmd":
        command.append("--deepmd-reuse-calculator" if config.deepmd_reuse_calculator else "--no-deepmd-reuse-calculator")
        command.extend(["--deepmd-force-backend", str(config.deepmd_force_backend)])
        command.extend(["--deepmd-device", str(config.deepmd_device)])
        if config.deepmd_model_head is not None:
            command.extend(["--deepmd-model-head", str(config.deepmd_model_head)])
        command.append("--deepmd-deterministic" if config.deepmd_deterministic else "--no-deepmd-deterministic")
        command.append("--save-force-audit" if config.save_force_audit else "--no-save-force-audit")
    command.extend(["--n-structures", str(config.n_structures)])
    command.extend(["--rattle-std", str(config.rattle_std)])
    command.append("--cutoffs")
    command.extend(str(value) for value in config.cutoffs)
    command.extend(["--min-dist", str(config.min_dist)])
    if config.overwrite:
        command.append("--overwrite")
    if config.resume:
        command.append("--resume")
    return build_run_command(command, command)


def _is_complete_result(outdir: Path) -> bool:
    result_path = outdir / "result.json"
    if not result_path.exists():
        return False
    result = _read_existing_result(outdir)
    if result.get("success") is not True:
        return False
    required = ["force_constants.hdf5", "band.yaml", "phonon_band.png", "summary.txt"]
    return all((outdir / filename).exists() and (outdir / filename).stat().st_size > 0 for filename in required)


def _read_existing_result(outdir: Path) -> dict[str, Any]:
    import json

    return json.loads((outdir / "result.json").read_text(encoding="utf-8"))


def _deepmd_infer_batch_size() -> int | None:
    value = os.environ.get("DP_INFER_BATCH_SIZE")
    try:
        return int(value) if value is not None and value.strip() else None
    except ValueError:
        return None


def _thermal_output_files(thermal: dict[str, Any]) -> dict[str, str]:
    files = thermal.get("files") if isinstance(thermal, dict) else None
    if not isinstance(files, dict):
        return {}
    labels = {
        "fc2_hdf5": "thermal_fc2_hdf5",
        "fc3_hdf5": "fc3_hdf5",
        "phono3py_params_yaml": "phono3py_params_yaml",
        "kappa_hdf5": "kappa_hdf5",
        "thermal_conductivity_csv": "thermal_conductivity_csv",
        "thermal_conductivity_png": "thermal_conductivity_png",
        "phonon_lifetime_csv": "phonon_lifetime_csv",
        "phonon_lifetime_png": "phonon_lifetime_png",
        "phonon_lifetime_diagnostics_json": "phonon_lifetime_diagnostics",
        "fd_fc2_diagnostics_json": "fd_fc2_diagnostics",
        "fd_fc3_diagnostics_json": "fd_fc3_diagnostics",
        "fd_phono3py_input_diagnostics_json": "fd_phono3py_input_diagnostics",
        "fd_fc2_forces_stats_csv": "fd_fc2_forces_stats",
        "fd_fc2_force_hashes_csv": "fd_fc2_force_hashes",
        "fd_fc2_forces_raw_npz": "fd_fc2_forces_raw",
        "fd_fc3_forces_stats_csv": "fd_fc3_forces_stats",
        "fd_fc3_force_hashes_csv": "fd_fc3_force_hashes",
        "fd_fc3_forces_raw_npz": "fd_fc3_forces_raw",
        "hiphive_fit_summary": "hiphive_fit_summary",
        "hiphive_fit_diagnostics_json": "hiphive_fit_diagnostics_json",
        "hiphive_fit_diagnostics_txt": "hiphive_fit_diagnostics_txt",
        "fc2_diagnostics_json": "thermal_fc2_diagnostics",
        "fc3_diagnostics_json": "thermal_fc3_diagnostics",
        "hiphive_force_fit_plot": "hiphive_force_fit_plot",
    }
    return {
        output_label: str(files[file_key])
        for file_key, output_label in labels.items()
        if files.get(file_key)
    }


def _structure_info(atoms: Any) -> dict[str, Any]:
    cell_parameters = atoms.cell.cellpar()
    return {
        "structure_formula": atoms.get_chemical_formula(),
        "n_atoms_unitcell": len(atoms),
        "cell_lengths": [float(value) for value in cell_parameters[:3]],
        "cell_angles": [float(value) for value in cell_parameters[3:]],
    }


def _supercell_info(config: WorkflowConfig, atoms: Any) -> dict[str, Any]:
    info = dict(config.supercell_info or {})
    if not info:
        lengths = np.asarray(atoms.cell.lengths(), dtype=float)
        dims = np.asarray(config.supercell_dim, dtype=int)
        info = {
            "supercell_dim": [int(value) for value in dims],
            "supercell_lengths_resolved": [float(value) for value in lengths * dims],
            "target_supercell_length": float(config.target_supercell_length),
            "min_supercell_dim": int(config.min_supercell_dim),
            "max_supercell_dim": int(config.max_supercell_dim),
            "max_supercell_atoms": int(config.max_supercell_atoms),
            "n_atoms_unitcell": int(len(atoms)),
            "n_atoms_supercell": int(len(atoms) * np.prod(dims)),
            "auto_supercell_warnings": [],
            "auto_supercell_notes": [],
        }
    return info


def _supercell_result_fields(config: WorkflowConfig, atoms: Any) -> dict[str, Any]:
    info = _supercell_info(config, atoms)
    return {
        "target_supercell_length": info.get("target_supercell_length", config.target_supercell_length),
        "min_supercell_dim": info.get("min_supercell_dim", config.min_supercell_dim),
        "max_supercell_dim": info.get("max_supercell_dim", config.max_supercell_dim),
        "max_supercell_atoms": info.get("max_supercell_atoms", config.max_supercell_atoms),
        "n_atoms_supercell": info.get("n_atoms_supercell", int(len(atoms) * np.prod(config.supercell_dim))),
        "supercell_lengths_resolved": info.get("supercell_lengths_resolved"),
        "auto_supercell_warnings": info.get("auto_supercell_warnings", []),
        "auto_supercell_notes": info.get("auto_supercell_notes", []),
    }


def _spacegroup_result_fields(spacegroup_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "initial_spacegroup": spacegroup_report.get("initial"),
        "final_spacegroup": spacegroup_report.get("final"),
        "spacegroup_changed": spacegroup_report.get("changed"),
        "spacegroup_change_summary": spacegroup_report.get("change_summary"),
        "spacegroup_report_json": "spacegroup_report.json",
        "spacegroup_report_txt": "spacegroup_report.txt",
        "phonopy_symprec": spacegroup_report.get("symprec"),
        "symprec": spacegroup_report.get("symprec"),
        "angle_tolerance": spacegroup_report.get("angle_tolerance"),
    }


def _write_spacegroup_outputs(
    outdir: Path,
    initial_spacegroup: dict[str, Any],
    final_spacegroup: dict[str, Any] | None,
    config: WorkflowConfig,
    dry_run: bool,
    log: Callable[[str], None],
) -> dict[str, Any]:
    report = build_spacegroup_report(
        initial_spacegroup,
        final_spacegroup,
        symprec=config.phonopy_symprec,
        angle_tolerance=config.angle_tolerance,
        dry_run=dry_run,
    )
    write_spacegroup_report(report, outdir / "spacegroup_report.json", outdir / "spacegroup_report.txt")
    log(f"Initial space group: {_format_spacegroup_for_log(report.get('initial'))}")
    log(f"Final space group: {_format_spacegroup_for_log(report.get('final'))}")
    log(f"Space group changed: {report.get('changed')}")
    log(f"Space group summary: {report.get('change_summary')}")
    return report


def _print_spacegroup_summary(console: Console, report: dict[str, Any]) -> None:
    console.print(f"Initial space group: {_format_spacegroup_for_log(report.get('initial'))}")
    console.print(f"Final space group: {_format_spacegroup_for_log(report.get('final'))}")
    console.print(f"Space group changed: {report.get('changed')}")


def _format_spacegroup_for_log(data: dict[str, Any] | None) -> str:
    data = data or {}
    symbol = data.get("international_symbol")
    number = data.get("spacegroup_number")
    if symbol and number:
        return f"{symbol} (No. {number})"
    return "unavailable"


def _classification_result_fields(structure_classification: dict[str, Any]) -> dict[str, Any]:
    return {
        "structure_type": structure_classification.get("structure_type"),
        "structure_classification": structure_classification,
        "vacuum_like_directions": structure_classification.get("vacuum_like_directions", []),
        "atom_extents": structure_classification.get("atom_extents", []),
        "classification_method": structure_classification.get("classification_method"),
    }


def _relax_mode(config: WorkflowConfig) -> str:
    if not config.relax:
        return "none"
    return "cell" if config.relax_cell else "positions"


def _constant_cell(config: WorkflowConfig) -> bool:
    return bool(config.relax and not config.relax_cell)


def _relax_warnings(config: WorkflowConfig, structure_classification: dict[str, Any]) -> list[str]:
    if not (config.relax and config.relax_cell):
        return []
    return list(structure_classification.get("warnings") or [])


def _skipped_relax_info(atoms: Any, config: WorkflowConfig) -> dict[str, Any]:
    cell_parameters = atoms.cell.cellpar()
    volume = float(atoms.get_volume())
    return {
        "relax_converged": False,
        "final_max_force_eV_per_A": None,
        "final_stress_GPa": None,
        "n_steps": 0,
        "fmax": float(config.fmax),
        "max_steps": int(config.max_steps),
        "optimizer": config.optimizer,
        "relax": bool(config.relax),
        "relax_cell": False,
        "relax_mode": "none",
        "constant_cell": False,
        "initial_cell": np.asarray(atoms.cell.array, dtype=float).tolist(),
        "final_cell": np.asarray(atoms.cell.array, dtype=float).tolist(),
        "initial_cell_lengths": [float(value) for value in cell_parameters[:3]],
        "final_cell_lengths": [float(value) for value in cell_parameters[:3]],
        "initial_cell_angles": [float(value) for value in cell_parameters[3:]],
        "final_cell_angles": [float(value) for value in cell_parameters[3:]],
        "initial_volume": volume,
        "final_volume": volume,
        "volume_change_percent": 0.0,
        "warnings": [],
        "notes": "Relaxation disabled; copied input structure.",
    }


def _ensure_relax_schema(
    relax_info: dict[str, Any],
    initial_atoms: Any,
    final_atoms: Any,
    config: WorkflowConfig,
) -> dict[str, Any]:
    initial_parameters = initial_atoms.cell.cellpar()
    final_parameters = final_atoms.cell.cellpar()
    initial_volume = float(initial_atoms.get_volume())
    final_volume = float(final_atoms.get_volume())
    enriched = dict(relax_info)
    enriched.setdefault("fmax", float(config.fmax))
    enriched.setdefault("max_steps", int(config.max_steps))
    enriched.setdefault("optimizer", config.optimizer)
    enriched.setdefault("relax", bool(config.relax))
    enriched.setdefault("relax_cell", bool(config.relax_cell) if config.relax else False)
    enriched.setdefault("relax_mode", _relax_mode(config))
    enriched.setdefault("constant_cell", _constant_cell(config))
    enriched.setdefault("initial_cell", np.asarray(initial_atoms.cell.array, dtype=float).tolist())
    enriched.setdefault("final_cell", np.asarray(final_atoms.cell.array, dtype=float).tolist())
    enriched.setdefault("initial_cell_lengths", [float(value) for value in initial_parameters[:3]])
    enriched.setdefault("final_cell_lengths", [float(value) for value in final_parameters[:3]])
    enriched.setdefault("initial_cell_angles", [float(value) for value in initial_parameters[3:]])
    enriched.setdefault("final_cell_angles", [float(value) for value in final_parameters[3:]])
    enriched.setdefault("initial_volume", initial_volume)
    enriched.setdefault("final_volume", final_volume)
    enriched.setdefault("volume_change_percent", _volume_change_percent(initial_volume, final_volume))
    enriched.setdefault("final_stress_GPa", None)
    enriched.setdefault("warnings", [])
    return enriched


def _volume_change_percent(initial_volume: float, final_volume: float) -> float | None:
    if abs(initial_volume) < 1e-12:
        return None
    return float((final_volume - initial_volume) / initial_volume * 100.0)


def _relax_result_fields(config: WorkflowConfig, relax_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "initial_cell_lengths": relax_info.get("initial_cell_lengths"),
        "final_cell_lengths": relax_info.get("final_cell_lengths"),
        "initial_cell_angles": relax_info.get("initial_cell_angles"),
        "final_cell_angles": relax_info.get("final_cell_angles"),
        "initial_volume": relax_info.get("initial_volume"),
        "final_volume": relax_info.get("final_volume"),
        "volume_change_percent": relax_info.get("volume_change_percent"),
        "final_max_force_eV_per_A": relax_info.get("final_max_force_eV_per_A"),
        "final_stress_GPa": relax_info.get("final_stress_GPa"),
        "relax_converged": relax_info.get("relax_converged"),
        "relax_cell": bool(config.relax_cell) if config.relax else False,
        "relax_mode": _relax_mode(config),
        "constant_cell": _constant_cell(config),
    }


def _software_versions(backend_name: str, phonopy_version: str | None) -> dict[str, str | None]:
    return {
        "PhonoFlow": VERSION,
        "Python": sys.version.split()[0],
        "numpy": np.__version__,
        "ASE": ase.__version__,
        "Phonopy": phonopy_version or _package_version("phonopy"),
        "phono3py": _package_version("phono3py"),
        "h5py": _package_version("h5py"),
        "hiphive": _package_version("hiphive"),
        "Calorine": _package_version("calorine") if backend_name == "calorine" else None,
        "deepmd": _package_version("deepmd-kit") if backend_name == "deepmd" else None,
        "torch": _package_version("torch") if backend_name == "deepmd" else None,
        "e3nn": _package_version("e3nn") if backend_name == "deepmd" else None,
        "seekpath": _package_version("seekpath"),
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }


def _git_commit() -> str:
    return "unknown"


def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except Exception:
        return "unknown"
