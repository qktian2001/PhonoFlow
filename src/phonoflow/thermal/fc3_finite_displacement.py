"""Finite-displacement FC3 and phono3py thermal workflow."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np

from phonoflow.calculators.base import CalculatorBackend
from phonoflow.config import WorkflowConfig, resolve_common_q_mesh
from phonoflow.defaults import infer_supercell_dim
from phonoflow.thermal.config import unavailable_thermal_result
from phonoflow.thermal.kappa_io import (
    extract_lifetime_from_hdf5,
    parse_kappa_hdf5,
    select_kappa_hdf5_path,
    summarize_kappa,
    write_thermal_conductivity_csv,
)
from phonoflow.thermal.plots import plot_thermal_conductivity
from phonoflow.thermal.wte_backend import get_wte_backend_capability
from phonoflow.workflow.force_audit import build_force_audit_record, write_force_audit_files
from phonoflow.workflow.displace import ase_atoms_to_phonopy_atoms, phonopy_atoms_to_ase_atoms


def run_finite_displacement_kappa_workflow(
    atoms: Any,
    backend: CalculatorBackend,
    config: WorkflowConfig,
    outdir: Path,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate FC3 with phono3py finite displacements and compute RTA/LBTE kappa."""

    if not config.compute_kappa:
        return unavailable_thermal_result(
            enabled=False,
            reason="Thermal conductivity calculation was not requested.",
            fc3_method=config.fc3_method,
            kappa_method=config.kappa_method,
        )

    temperatures = [float(value) for value in config.temperatures]
    kappa_mesh = _resolve_kappa_mesh(config)
    wte_capability = get_wte_backend_capability()
    if config.wigner and not wte_capability.get("available"):
        fc3_supercell_dim = (
            [int(value) for value in config.fc3_supercell_dim]
            if config.fc3_supercell_dim != "auto"
            else None
        )
        return unavailable_thermal_result(
            reason=wte_capability["reason"],
            warnings=[
                "Wigner transport did not run because the phono3py-wte WTE plugin could not be imported.",
                "Harmonic phonon, DOS, and group-velocity outputs are unaffected.",
            ],
            fc3_method="finite-displacement",
            kappa_method=config.kappa_method,
            wigner=True,
            wigner_requested=True,
            wigner_available=False,
            wigner_backend=wte_capability["backend"],
            wte_plugin_found=wte_capability["wte_module_found"],
            wte_module_found=wte_capability["wte_module_found"],
            phono3py_version=wte_capability["phono3py_version"],
            phonopy_version=wte_capability["phonopy_version"],
            transport_type=None,
            thermal_status="wte_plugin_unavailable",
            wigner_unavailable_reason=wte_capability["reason"],
            temperatures=temperatures,
            kappa_mesh=[int(value) for value in kappa_mesh],
            fc3_supercell_dim=fc3_supercell_dim,
            fc2_displacement=float(config.displacement),
            fc3_displacement=float(config.fc3_displacement),
            fc3_cutoff_pair_distance=config.fc3_cutoff_pair_distance,
            max_fc3_displacements=config.max_fc3_displacements,
            smoke_test=config.max_fc3_displacements is not None,
        )

    try:
        from phono3py import Phono3py
        from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5
    except Exception as exc:
        return unavailable_thermal_result(
            reason=f"phono3py is required for --compute-kappa but could not be imported: {exc}",
            fc3_method="finite-displacement",
            kappa_method=config.kappa_method,
        )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    fc3_started = time.perf_counter()
    fc3_seconds = 0.0
    thermal_started: float | None = None
    try:
        fc3_supercell_dim = _resolve_fc3_supercell_dim(atoms, config)
        transport_type = _resolve_transport_type(config, wte_capability)
        if config.wigner and transport_type is None:
            return unavailable_thermal_result(
                reason=wte_capability["reason"],
                warnings=[
                    "Wigner transport did not run because the phono3py-wte WTE plugin could not be imported.",
                    "Harmonic phonon, DOS, and group-velocity outputs are unaffected.",
                ],
                fc3_method="finite-displacement",
                kappa_method=config.kappa_method,
                wigner=True,
                wigner_requested=True,
                wigner_available=False,
                wigner_backend=wte_capability["backend"],
                wte_plugin_found=wte_capability["wte_module_found"],
                wte_module_found=wte_capability["wte_module_found"],
                phono3py_version=wte_capability["phono3py_version"],
                phonopy_version=wte_capability["phonopy_version"],
                transport_type=None,
                thermal_status="wte_plugin_unavailable",
                wigner_unavailable_reason=wte_capability["reason"],
                temperatures=temperatures,
                kappa_mesh=[int(value) for value in kappa_mesh],
                fc3_supercell_dim=[int(value) for value in fc3_supercell_dim],
                fc2_displacement=float(config.displacement),
                fc3_displacement=float(config.fc3_displacement),
                fc3_cutoff_pair_distance=config.fc3_cutoff_pair_distance,
                max_fc3_displacements=config.max_fc3_displacements,
                smoke_test=config.max_fc3_displacements is not None,
            )
        _log(log, f"Creating Phono3py object with fc3 supercell {fc3_supercell_dim}")
        phono3py = Phono3py(
            ase_atoms_to_phonopy_atoms(atoms),
            supercell_matrix=fc3_supercell_dim,
            phonon_supercell_matrix=fc3_supercell_dim,
            primitive_matrix=_primitive_matrix_argument(config.primitive_matrix),
            cutoff_frequency=_resolve_phono3py_cutoff_frequency(config),
            is_symmetry=bool(config.phono3py_symmetry),
            is_mesh_symmetry=bool(config.phono3py_mesh_symmetry),
            symprec=_resolve_phono3py_symprec(config),
            log_level=0,
        )
        displacement_kwargs = _displacement_kwargs(config)
        phono3py.generate_displacements(**displacement_kwargs)
        phono3py.generate_fc2_displacements(distance=config.displacement)
        fc3_supercells = list(phono3py.supercells_with_displacements or [])
        fc2_supercells = list(phono3py.phonon_supercells_with_displacements or [])
        if config.max_fc3_displacements is not None:
            warnings.append(
                "max_fc3_displacements was used for smoke testing. The resulting thermal "
                "conductivity is not a converged production value."
            )
            fc3_supercells = fc3_supercells[: int(config.max_fc3_displacements)]
        if not fc3_supercells:
            raise RuntimeError("phono3py did not generate FC3 displaced supercells.")
        if not fc2_supercells:
            raise RuntimeError("phono3py did not generate FC2 displaced supercells.")

        phono3py.forces = _evaluate_phono3py_forces(
            fc3_supercells,
            backend,
            log,
            label="FC3",
            audit_outdir=outdir if config.save_force_audit else None,
        )
        phono3py.phonon_forces = _evaluate_phono3py_forces(
            fc2_supercells,
            backend,
            log,
            label="FC2",
            audit_outdir=outdir if config.save_force_audit else None,
        )
        _log(log, "Producing phono3py FC3 and FC2")
        phono3py.produce_fc3()
        phono3py.produce_fc2()
        symmetrize_fc2_info = _apply_phono3py_symmetrize_fc2(phono3py, enabled=config.phono3py_symmetrize_fc2)
        symmetrize_fc3_info = _apply_phono3py_symmetrize_fc3(phono3py, enabled=config.phono3py_symmetrize_fc3)
        for symmetrization_info in (symmetrize_fc2_info, symmetrize_fc3_info):
            if symmetrization_info.get("warning"):
                warnings.append(str(symmetrization_info["warning"]))
        asr_info = {**symmetrize_fc2_info, **symmetrize_fc3_info}

        fc3_path = outdir / "fc3.hdf5"
        fc2_path = outdir / "fc2.hdf5"
        write_fc3_to_hdf5(phono3py.fc3, filename=str(fc3_path))
        write_fc2_to_hdf5(phono3py.fc2, filename=str(fc2_path))
        phono3py_params_path = outdir / "phono3py_params.yaml"
        _save_phono3py_params(phono3py, phono3py_params_path)
        fc3_seconds = time.perf_counter() - fc3_started

        thermal_started = time.perf_counter()
        phono3py.mesh_numbers = kappa_mesh
        phono3py.init_phph_interaction()
        is_lbte = config.kappa_method == "lbte"
        _log(
            log,
            "Running phono3py thermal conductivity "
            f"method={config.kappa_method}, mesh={kappa_mesh}, "
            f"transport_type={transport_type or 'standard'}",
        )
        with _working_directory(outdir):
            _run_thermal_conductivity_compat(
                phono3py,
                config=config,
                is_lbte=is_lbte,
                temperatures=temperatures,
                transport_type=transport_type,
            )
        try:
            kappa_path = select_kappa_hdf5_path(outdir, kappa_mesh)
        except FileNotFoundError:
            raise RuntimeError("phono3py completed without writing a kappa-m*.hdf5 file.")
        parsed = parse_kappa_hdf5(kappa_path)
        rows = parsed["rows"]
        thermal_csv = outdir / "thermal_conductivity.csv"
        thermal_png = outdir / "thermal_conductivity.png"
        write_thermal_conductivity_csv(rows, thermal_csv)
        plot_thermal_conductivity(rows, thermal_png, dpi=config.plot_dpi)
        lifetime = extract_lifetime_from_hdf5(kappa_path, outdir, dpi=config.plot_dpi)
        diagnostics = _write_fd_diagnostics(
            outdir=outdir,
            config=config,
            fc3_supercell_dim=fc3_supercell_dim,
            kappa_mesh=kappa_mesh,
            temperatures=temperatures,
            phono3py=phono3py,
            fc2=np.asarray(phono3py.fc2, dtype=float),
            fc3=np.asarray(phono3py.fc3, dtype=float),
            fc2_path=fc2_path,
            fc3_path=fc3_path,
            kappa_path=kappa_path,
            thermal_csv=thermal_csv,
            thermal_png=thermal_png,
            lifetime=lifetime,
            warnings=warnings,
            asr_info=asr_info,
        )
        thermal_seconds = time.perf_counter() - thermal_started
        files: dict[str, Any] = {
            "fc2_hdf5": fc2_path.name,
            "fc3_hdf5": fc3_path.name,
            "phono3py_params_yaml": phono3py_params_path.name,
            "kappa_hdf5": kappa_path.name,
            "thermal_conductivity_csv": thermal_csv.name,
            "thermal_conductivity_png": thermal_png.name,
            **diagnostics.get("files", {}),
        }
        if lifetime.get("available"):
            files["phonon_lifetime_csv"] = lifetime.get("data_file")
            files["phonon_lifetime_png"] = lifetime.get("plot_file")
            files["phonon_lifetime_diagnostics_json"] = lifetime.get("diagnostics_file")

        return {
            "enabled": True,
            "available": True,
            "fc3_method": "finite-displacement",
            "kappa_method": config.kappa_method,
            "solver_flags": ["--method", config.kappa_method],
            "method_flags": ["--method", config.kappa_method],
            "wigner": bool(config.wigner),
            "wigner_requested": bool(config.wigner),
            "wigner_available": bool(config.wigner and transport_type is not None),
            "wigner_backend": wte_capability["backend"] if config.wigner else None,
            "wte_plugin_found": wte_capability["wte_module_found"],
            "wte_module_found": wte_capability["wte_module_found"],
            "phono3py_version": wte_capability["phono3py_version"],
            "phonopy_version": wte_capability["phonopy_version"],
            "transport_type": transport_type,
            "thermal_status": "available",
            "wigner_unavailable_reason": None,
            "temperatures": temperatures,
            "kappa_mesh": [int(value) for value in kappa_mesh],
            "fc3_supercell_dim": [int(value) for value in fc3_supercell_dim],
            "fc2_displacement": float(config.displacement),
            "fc3_displacement": float(config.fc3_displacement),
            "fc3_cutoff_pair_distance": config.fc3_cutoff_pair_distance,
            "phono3py_symprec": _resolve_phono3py_symprec(config),
            "phono3py_cutoff_frequency": config.phono3py_cutoff_frequency,
            "phono3py_plusminus": config.phono3py_plusminus,
            "phono3py_diagonal": config.phono3py_diagonal,
            "phono3py_symmetry": config.phono3py_symmetry,
            "phono3py_mesh_symmetry": config.phono3py_mesh_symmetry,
            "phono3py_isotope": config.phono3py_isotope,
            "boundary_mfp": config.boundary_mfp,
            "cutoff_pair_distance": config.cutoff_pair_distance,
            **asr_info,
            "force_audit_saved": bool(config.save_force_audit),
            "n_fc3_displacements": len(fc3_supercells),
            "n_fc2_displacements": len(fc2_supercells),
            "finite_displacement_diagnostics": diagnostics,
            "max_fc3_displacements": config.max_fc3_displacements,
            "smoke_test": config.max_fc3_displacements is not None,
            "files": files,
            "kappa_unit": "W/m-K",
            "summary": summarize_kappa(rows),
            "lifetime": lifetime,
            "timing_breakdown": {
                "fc3_seconds": round(fc3_seconds, 6),
                "thermal_lifetime_seconds": round(thermal_seconds, 6),
            },
            "warnings": warnings,
            "reason": None,
        }
    except Exception as exc:
        _log(log, f"Thermal conductivity calculation failed: {exc}")
        wte_capability = get_wte_backend_capability()
        return unavailable_thermal_result(
            reason=str(exc),
            warnings=warnings,
            fc3_method="finite-displacement",
            kappa_method=config.kappa_method,
            wigner=bool(config.wigner),
            wigner_requested=bool(config.wigner),
            wigner_available=False,
            wigner_backend=wte_capability["backend"] if config.wigner else None,
            wte_plugin_found=wte_capability["wte_module_found"],
            wte_module_found=wte_capability["wte_module_found"],
            phono3py_version=wte_capability["phono3py_version"],
            phonopy_version=wte_capability["phonopy_version"],
            transport_type="WTE" if config.wigner and wte_capability["available"] else None,
            thermal_status="failed",
            wigner_unavailable_reason=str(exc) if config.wigner else None,
            timing_breakdown={
                "fc3_seconds": round(
                    fc3_seconds or (time.perf_counter() - fc3_started),
                    6,
                ),
                "thermal_lifetime_seconds": round(
                    time.perf_counter() - thermal_started,
                    6,
                )
                if thermal_started is not None
                else 0.0,
            },
        )


def _evaluate_phono3py_forces(
    supercells: list[Any],
    backend: CalculatorBackend,
    log: Callable[[str], None] | None,
    label: str,
    audit_outdir: Path | None = None,
) -> np.ndarray:
    forces = []
    audit_records: list[dict[str, Any]] = []
    total = len(supercells)
    for index, supercell in enumerate(supercells, start=1):
        _log(log, f"Evaluating {label} displaced supercell {index}/{total}")
        atoms = phonopy_atoms_to_ase_atoms(supercell)
        result = backend.calculate_energy_forces(atoms)
        force_array = np.asarray(result["forces"], dtype=float)
        forces.append(force_array)
        if audit_outdir is not None:
            audit_records.append(
                build_force_audit_record(
                    index - 1,
                    atoms,
                    energy=result.get("energy"),
                    forces=force_array,
                )
            )
    raw = np.asarray(forces, dtype=float)
    if audit_outdir is not None:
        write_force_audit_files(audit_outdir, label.lower(), audit_records, raw)
    return raw


def _write_fd_diagnostics(
    *,
    outdir: Path,
    config: WorkflowConfig,
    fc3_supercell_dim: list[int],
    kappa_mesh: list[int],
    temperatures: list[float],
    phono3py: Any,
    fc2: np.ndarray,
    fc3: np.ndarray,
    fc2_path: Path,
    fc3_path: Path,
    kappa_path: Path,
    thermal_csv: Path,
    thermal_png: Path,
    lifetime: dict[str, Any],
    warnings: list[str],
    asr_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    asr_info = dict(asr_info or {})
    files = {
        "fd_fc2_diagnostics_json": "fd_fc2_diagnostics.json",
        "fd_fc3_diagnostics_json": "fd_fc3_diagnostics.json",
        "fd_phono3py_input_diagnostics_json": "fd_phono3py_input_diagnostics.json",
    }
    n_atoms_supercell = len(phono3py.supercell) if getattr(phono3py, "supercell", None) is not None else None
    fc2_diagnostics = {
        **_force_constant_diagnostics(fc2, order=2),
        "method": "finite-displacement",
        "force_units_note": "Forces are evaluated by ASE/Calorine CPUNEP in eV/Angstrom; phono3py displacements are in Angstrom.",
        "supercell_dim": [int(value) for value in fc3_supercell_dim],
        "fc3_supercell_dim": [int(value) for value in fc3_supercell_dim],
        "n_atoms_supercell": n_atoms_supercell,
    }
    fc3_diagnostics = {
        **_force_constant_diagnostics(fc3, order=3),
        "method": "finite-displacement",
        "force_units_note": "Forces are evaluated by ASE/Calorine CPUNEP in eV/Angstrom; phono3py displacements are in Angstrom.",
        "supercell_dim": [int(value) for value in fc3_supercell_dim],
        "fc3_supercell_dim": [int(value) for value in fc3_supercell_dim],
        "n_atoms_supercell": n_atoms_supercell,
    }
    input_diagnostics = {
        "method": "finite-displacement",
        "supercell_dim": [int(value) for value in fc3_supercell_dim],
        "fc3_supercell_dim": [int(value) for value in fc3_supercell_dim],
        "n_atoms_supercell": n_atoms_supercell,
        "n_fc2_displacements": int(len(phono3py.phonon_supercells_with_displacements or [])),
        "n_fc3_displacements": int(len(phono3py.supercells_with_displacements or [])),
        "fc2_displacement": float(config.displacement),
        "fc3_displacement": float(config.fc3_displacement),
        "fc2_shape": [int(value) for value in fc2.shape],
        "fc3_shape": [int(value) for value in fc3.shape],
        "force_units_note": "ASE/Calorine forces are eV/Angstrom and are passed directly to phono3py.",
        "kappa_mesh": [int(value) for value in kappa_mesh],
        "temperatures": [float(value) for value in temperatures],
        "phono3py_method": config.kappa_method,
        "wigner": bool(config.wigner),
        "phono3py_symmetrize_fc2": bool(config.phono3py_symmetrize_fc2),
        "phono3py_symmetrize_fc3": bool(config.phono3py_symmetrize_fc3),
        **asr_info,
        "fc2_file": fc2_path.name,
        "fc3_file": fc3_path.name,
        "kappa_file": kappa_path.name,
        "thermal_conductivity_file": thermal_csv.name,
        "thermal_conductivity_plot": thermal_png.name,
        "phonon_lifetime_file": lifetime.get("data_file") if lifetime.get("available") else None,
        "phonon_lifetime_plot": lifetime.get("plot_file") if lifetime.get("available") else None,
        "warnings": warnings,
        "files": files,
    }
    (outdir / files["fd_fc2_diagnostics_json"]).write_text(json.dumps(fc2_diagnostics, indent=2), encoding="utf-8")
    (outdir / files["fd_fc3_diagnostics_json"]).write_text(json.dumps(fc3_diagnostics, indent=2), encoding="utf-8")
    (outdir / files["fd_phono3py_input_diagnostics_json"]).write_text(
        json.dumps(input_diagnostics, indent=2),
        encoding="utf-8",
    )
    return {
        **input_diagnostics,
        "fc2_diagnostics": fc2_diagnostics,
        "fc3_diagnostics": fc3_diagnostics,
    }


def _force_constant_diagnostics(fc: np.ndarray, *, order: int) -> dict[str, Any]:
    finite = np.asarray(fc, dtype=float)
    return {
        "order": order,
        "shape": [int(value) for value in finite.shape],
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "norm": float(np.linalg.norm(finite.ravel())) if finite.size else None,
        "mean_abs": float(np.mean(np.abs(finite))) if finite.size else None,
        "max_abs": float(np.max(np.abs(finite))) if finite.size else None,
        "n_nonzero": int(np.count_nonzero(np.abs(finite) > 0)),
    }


def _resolve_fc3_supercell_dim(atoms: Any, config: WorkflowConfig) -> list[int]:
    if config.fc3_supercell_dim != "auto":
        return [int(value) for value in config.fc3_supercell_dim]
    return infer_supercell_dim(
        atoms,
        target_supercell_length=config.fc3_target_supercell_length,
        min_dim=1,
        max_dim=6,
        max_supercell_atoms=config.max_fc3_supercell_atoms,
    )


def _resolve_kappa_mesh(config: WorkflowConfig) -> list[int]:
    return resolve_common_q_mesh(config.mesh, config.kappa_mesh)


def _resolve_phono3py_symprec(config: WorkflowConfig) -> float:
    return float(config.phono3py_symprec if config.phono3py_symprec is not None else 1e-5)


def _resolve_phono3py_cutoff_frequency(config: WorkflowConfig) -> float:
    return float(
        config.phono3py_cutoff_frequency
        if config.phono3py_cutoff_frequency is not None
        else 1e-4
    )


def _displacement_kwargs(config: WorkflowConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "distance": config.fc3_displacement,
        "cutoff_pair_distance": (
            config.fc3_cutoff_pair_distance
            if config.fc3_cutoff_pair_distance is not None
            else (config.cutoff_pair_distance or None)
        ),
    }
    plusminus = _plusminus_value(config.phono3py_plusminus)
    if plusminus is not None:
        kwargs["is_plusminus"] = plusminus
    kwargs["is_diagonal"] = bool(config.phono3py_diagonal)
    return kwargs


def _plusminus_value(value: str) -> bool | str | None:
    normalized = str(value).lower()
    if normalized == "auto":
        return "auto"
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _apply_phono3py_symmetrize_fc2(phono3py: Any, *, enabled: bool) -> dict[str, Any]:
    before = _fc2_residual(phono3py)
    if not enabled:
        return {
            "phono3py_symmetrize_fc2": False,
            "phono3py_symmetrize_fc2_applied": False,
            "fc2_asr_residual_before": before,
            "fc2_asr_residual_after": before,
        }
    try:
        phono3py.symmetrize_fc2()
        after = _fc2_residual(phono3py)
        return {
            "phono3py_symmetrize_fc2": True,
            "phono3py_symmetrize_fc2_applied": True,
            "fc2_asr_residual_before": before,
            "fc2_asr_residual_after": after,
        }
    except Exception as exc:
        return {
            "phono3py_symmetrize_fc2": True,
            "phono3py_symmetrize_fc2_applied": False,
            "fc2_asr_residual_before": before,
            "fc2_asr_residual_after": before,
            "warning": f"Could not apply phono3py FC2 force-constant symmetrization: {exc}",
        }


def _apply_phono3py_symmetrize_fc3(phono3py: Any, *, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {
            "phono3py_symmetrize_fc3": False,
            "phono3py_symmetrize_fc3_applied": False,
        }
    try:
        phono3py.symmetrize_fc3()
        return {
            "phono3py_symmetrize_fc3": True,
            "phono3py_symmetrize_fc3_applied": True,
        }
    except Exception as exc:
        return {
            "phono3py_symmetrize_fc3": True,
            "phono3py_symmetrize_fc3_applied": False,
            "warning": f"Could not apply phono3py FC3 force-constant symmetrization: {exc}",
        }


def _fc2_residual(phono3py: Any) -> float | None:
    fc2 = getattr(phono3py, "fc2", None)
    if fc2 is None:
        return None
    array = np.asarray(fc2, dtype=float)
    if array.size == 0:
        return 0.0
    try:
        drift = np.sum(array, axis=1)
        return float(np.max(np.abs(drift)))
    except Exception:
        return float(np.max(np.abs(array)))


def _save_phono3py_params(phono3py: Any, path: Path) -> None:
    try:
        phono3py.save(filename=str(path))
    except Exception:
        path.write_text("# phono3py parameter export unavailable for this phono3py version\n", encoding="utf-8")


def _run_thermal_conductivity_compat(
    phono3py: Any,
    *,
    config: WorkflowConfig,
    is_lbte: bool,
    temperatures: list[float],
    transport_type: str | None,
) -> None:
    kwargs: dict[str, Any] = {
        "is_LBTE": is_lbte,
        "temperatures": temperatures,
        "transport_type": transport_type,
        "write_kappa": True,
        "log_level": 0,
    }
    kwargs["is_isotope"] = bool(config.phono3py_isotope)
    if config.boundary_mfp > 0:
        kwargs["boundary_mfp"] = float(config.boundary_mfp)
    try:
        phono3py.run_thermal_conductivity(**kwargs)
    except TypeError:
        minimal = {
            "is_LBTE": is_lbte,
            "temperatures": temperatures,
            "transport_type": transport_type,
            "write_kappa": True,
            "log_level": 0,
        }
        phono3py.run_thermal_conductivity(**minimal)


def _resolve_transport_type(config: WorkflowConfig, wte_capability: dict[str, Any] | None = None) -> str | None:
    """Return the phono3py transport type for optional Wigner transport.

    phono3py v4 removed the old CLI ``--wigner`` flag. Its own argparse layer
    points users to the separate phono3py-wte plugin and the ``--tt wte``
    transport type. We mirror that API here: no plugin means a clear
    unavailable result instead of spending time on FC3 and failing later with a
    cryptic transport-type error.
    """

    if not config.wigner:
        return None
    wte_capability = wte_capability or get_wte_backend_capability()
    if not wte_capability.get("available"):
        return None
    return "WTE"


def _primitive_matrix_argument(value: str) -> Any:
    normalized = value.lower()
    if normalized == "p":
        return "P"
    if normalized == "auto":
        return "auto"
    if normalized == "identity":
        return np.eye(3)
    if normalized == "none":
        return None
    return value


@contextmanager
def _working_directory(path: Path) -> Any:
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _log(log: Callable[[str], None] | None, message: str) -> None:
    if log is not None:
        log(message)
