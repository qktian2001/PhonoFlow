"""HiPhive FC2/FC3 fitting route for phono3py thermal conductivity."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from phonopy import Phonopy

from phonoflow.calculators.base import CalculatorBackend
from phonoflow.config import WorkflowConfig, resolve_common_q_mesh
from phonoflow.defaults import infer_supercell_dim
from phonoflow.thermal.config import unavailable_thermal_result
from phonoflow.thermal.kappa_io import (
    extract_lifetime_from_hdf5,
    inspect_kappa_hdf5,
    parse_kappa_hdf5,
    select_kappa_hdf5_path,
    summarize_kappa,
    write_thermal_conductivity_csv,
)
from phonoflow.thermal.plots import plot_thermal_conductivity
from phonoflow.thermal.fc3_finite_displacement import (
    _resolve_phono3py_cutoff_frequency,
    _resolve_phono3py_symprec,
    _run_thermal_conductivity_compat,
)
from phonoflow.thermal.wte_backend import get_wte_backend_capability
from phonoflow.workflow.displace import ase_atoms_to_phonopy_atoms, phonopy_atoms_to_ase_atoms


def run_hiphive_kappa_workflow(
    atoms: Any,
    backend: CalculatorBackend,
    config: WorkflowConfig,
    outdir: Path,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fit FC2/FC3 with HiPhive and compute kappa with phono3py.

    HiPhive fits a force-constant potential from rattled supercells and exports
    dense FC2/FC3 arrays. phono3py then owns the thermal-transport calculation,
    keeping the kappa/lifetime parser shared with the finite-displacement route.
    """

    if not config.compute_kappa:
        return unavailable_thermal_result(
            enabled=False,
            reason="Thermal conductivity calculation was not requested.",
            fc3_method="hiphive",
            kappa_method=config.kappa_method,
        )

    try:
        from hiphive import ClusterSpace, ForceConstantPotential, StructureContainer, enforce_rotational_sum_rules
        from hiphive.structure_generation import generate_mc_rattled_structures, generate_rattled_structures
        from hiphive.utilities import prepare_structures
        from phono3py import Phono3py
        from trainstation import Optimizer
    except Exception as exc:
        return unavailable_thermal_result(
            reason=f"HiPhive/phono3py dependencies are required for --fc3-method hiphive: {exc}",
            fc3_method="hiphive",
            kappa_method=config.kappa_method,
            hiphive_available=False,
            hiphive_status="unavailable",
            **_hiphive_metadata(config),
        )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    fc3_started = time.perf_counter()
    fc3_seconds = 0.0
    thermal_started: float | None = None
    try:
        fc3_supercell_dim = _resolve_fc3_supercell_dim(atoms, config)
        kappa_mesh = _resolve_kappa_mesh(config)
        temperatures = [float(value) for value in config.temperatures]
        wte_capability = get_wte_backend_capability()
        transport_type = _resolve_transport_type(config, wte_capability)
        if config.wigner and transport_type is None:
            return unavailable_thermal_result(
                reason=wte_capability["reason"],
                warnings=[
                    "HiPhive fitting did not run because Wigner transport needs the phono3py-wte WTE plugin.",
                    "Run without --wigner true to use the standard phono3py thermal route.",
                ],
                fc3_method="hiphive",
                kappa_method=config.kappa_method,
                hiphive_available=False,
                hiphive_status="wte_plugin_unavailable",
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
                **_hiphive_metadata(config, fc3_supercell_dim=fc3_supercell_dim, kappa_mesh=kappa_mesh),
            )

        fit_atoms, canonicalization_warning = canonicalize_fractional_positions_for_hiphive(atoms)
        if canonicalization_warning:
            warnings.append(canonicalization_warning)
            _log(log, canonicalization_warning)

        _log(log, f"Creating HiPhive cluster space with cutoffs {config.cutoffs}")
        try:
            cluster_space = ClusterSpace(fit_atoms, [float(value) for value in config.cutoffs])
        except Exception as exc:
            raise RuntimeError(
                "HiPhive cluster-space construction failed. "
                "This usually indicates a symmetry or cutoff issue in the fitted structure; "
                f"cutoffs={list(config.cutoffs)}, reason={exc}"
            ) from exc
        reference_phonopy = Phonopy(
            ase_atoms_to_phonopy_atoms(fit_atoms),
            supercell_matrix=fc3_supercell_dim,
            primitive_matrix=_primitive_matrix_argument(config.primitive_matrix),
        )
        training_supercell = phonopy_atoms_to_ase_atoms(reference_phonopy.supercell)
        n_structures = int(config.n_structures)
        _log(
            log,
            "Generating HiPhive rattled structures "
            f"n={n_structures}, std={config.rattle_std}, min_dist={config.min_dist}",
        )
        try:
            training_structures = generate_mc_rattled_structures(
                training_supercell,
                n_structures,
                float(config.rattle_std),
                float(config.min_dist),
                seed=42,
            )
        except Exception as exc:
            warnings.append(f"MC rattling failed ({exc}); falling back to unconstrained rattling.")
            training_structures = generate_rattled_structures(
                training_supercell,
                n_structures,
                float(config.rattle_std),
                seed=42,
            )
        if not training_structures:
            raise RuntimeError("HiPhive did not generate any rattled training structures.")

        structure_container = StructureContainer(cluster_space)
        force_rmse_samples: list[float] = []
        for index, structure in enumerate(training_structures, start=1):
            _log(log, f"Evaluating HiPhive training structure {index}/{len(training_structures)}")
            force_result = backend.calculate_energy_forces(structure)
            forces = np.asarray(force_result["forces"], dtype=float)
            structure.arrays["forces"] = forces
            structure.calc = SinglePointCalculator(structure, forces=forces)
            force_rmse_samples.append(float(np.sqrt(np.mean(forces**2))))
        for prepared in prepare_structures(training_structures, training_supercell, check_permutation=False):
            structure_container.add_structure(prepared)

        fit_matrix, fit_targets = structure_container.get_fit_data()
        if fit_matrix.size == 0 or fit_targets.size == 0:
            raise RuntimeError("HiPhive produced empty fit data.")
        _log(log, f"Fitting HiPhive parameters from matrix {fit_matrix.shape}")
        optimizer = Optimizer(
            (fit_matrix, fit_targets),
            train_set=list(range(len(fit_targets))),
            test_set=[],
            check_condition=False,
        )
        optimizer.train()
        parameters = np.asarray(optimizer.parameters, dtype=float)
        try:
            parameters = np.asarray(
                enforce_rotational_sum_rules(cluster_space, parameters, ["Huang", "Born-Huang"]),
                dtype=float,
            )
        except Exception as exc:
            warnings.append(f"Rotational sum-rule enforcement failed; using unconstrained HiPhive parameters. Reason: {exc}")
        force_constant_potential = ForceConstantPotential(
            cluster_space,
            parameters,
            metadata={
                "fc3_method": "hiphive",
                "n_structures": n_structures,
                "rattle_std": float(config.rattle_std),
                "cutoffs": [float(value) for value in config.cutoffs],
                "min_dist": float(config.min_dist),
            },
        )
        force_constant_potential.write(str(outdir / "hiphive_model.fcp"))
        force_constants = force_constant_potential.get_force_constants(training_supercell)
        fc2 = np.asarray(force_constants.get_fc_array(order=2, format="phonopy"), dtype=float)
        fc3 = np.asarray(force_constants.get_fc_array(order=3, format="phonopy"), dtype=float)
        if fc2.ndim != 4 or fc3.ndim != 6:
            raise RuntimeError(f"Unexpected HiPhive FC shapes: fc2={fc2.shape}, fc3={fc3.shape}")

        fc2_path = outdir / "fc2.hdf5"
        fc3_path = outdir / "fc3.hdf5"
        force_constants.write_to_phonopy(str(fc2_path))
        force_constants.write_to_phono3py(str(fc3_path))
        diagnostics = _write_hiphive_diagnostics(
            outdir=outdir,
            config=config,
            fc3_supercell_dim=fc3_supercell_dim,
            kappa_mesh=kappa_mesh,
            training_supercell=training_supercell,
            cluster_space=cluster_space,
            fit_matrix=fit_matrix,
            fit_targets=fit_targets,
            parameters=parameters,
            optimizer=optimizer,
            fc2=fc2,
            fc3=fc3,
            force_rmse_samples=force_rmse_samples,
            warnings=warnings,
        )
        fit_summary_path = outdir / "hiphive_fit_summary.json"
        fit_summary = {
            "status": "available",
            "n_structures": n_structures,
            "rattle_std": float(config.rattle_std),
            "cutoffs": [float(value) for value in config.cutoffs],
            "min_dist": float(config.min_dist),
            "fc3_supercell_dim": [int(value) for value in fc3_supercell_dim],
            "fit_matrix_shape": [int(value) for value in fit_matrix.shape],
            "n_parameters": int(parameters.size),
            "force_rmse_input_eV_per_A": diagnostics.get("force_rmse_input_eV_per_A"),
            "force_rmse_train_eV_per_A": diagnostics.get("force_rmse_train_eV_per_A"),
            "max_force_error_train_eV_per_A": diagnostics.get("max_force_error_train_eV_per_A"),
            "number_of_force_components": diagnostics.get("number_of_force_components"),
            "number_of_fit_parameters": diagnostics.get("number_of_fit_parameters"),
            "underdetermined": diagnostics.get("underdetermined"),
            "fc2_shape": [int(value) for value in fc2.shape],
            "fc3_shape": [int(value) for value in fc3.shape],
            "diagnostics_files": diagnostics.get("files", {}),
            "warnings": warnings,
        }
        fit_summary_path.write_text(json.dumps(fit_summary, indent=2), encoding="utf-8")

        _log(log, "Creating phono3py object from HiPhive FC2/FC3")
        phono3py = Phono3py(
            ase_atoms_to_phonopy_atoms(fit_atoms),
            supercell_matrix=fc3_supercell_dim,
            phonon_supercell_matrix=fc3_supercell_dim,
            primitive_matrix=_primitive_matrix_argument(config.primitive_matrix),
            cutoff_frequency=_resolve_phono3py_cutoff_frequency(config),
            is_symmetry=bool(config.phono3py_symmetry),
            is_mesh_symmetry=bool(config.phono3py_mesh_symmetry),
            symprec=_resolve_phono3py_symprec(config),
            log_level=0,
        )
        phono3py.fc2 = fc2
        phono3py.fc3 = fc3
        hiphive_symmetrization_info = _hiphive_symmetrization_info(config)
        fc3_seconds = time.perf_counter() - fc3_started
        thermal_started = time.perf_counter()
        phono3py.mesh_numbers = kappa_mesh
        phono3py.init_phph_interaction()
        is_lbte = config.kappa_method == "lbte"
        _log(
            log,
            "Running phono3py thermal conductivity from HiPhive FCs "
            f"method={config.kappa_method}, mesh={kappa_mesh}, transport_type={transport_type or 'standard'}",
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
        kappa_diagnostics = inspect_kappa_hdf5(kappa_path)
        diagnostics = _annotate_transport_diagnostics(
            outdir=outdir,
            diagnostics=diagnostics,
            kappa_diagnostics=kappa_diagnostics,
            lifetime=lifetime,
        )
        thermal_seconds = time.perf_counter() - thermal_started

        files: dict[str, Any] = {
            "fc2_hdf5": fc2_path.name,
            "fc3_hdf5": fc3_path.name,
            "kappa_hdf5": kappa_path.name,
            "thermal_conductivity_csv": thermal_csv.name,
            "thermal_conductivity_png": thermal_png.name,
            "hiphive_fit_summary": fit_summary_path.name,
            **diagnostics.get("files", {}),
        }
        if lifetime.get("available"):
            files["phonon_lifetime_csv"] = lifetime.get("data_file")
            files["phonon_lifetime_png"] = lifetime.get("plot_file")
            files["phonon_lifetime_diagnostics_json"] = lifetime.get("diagnostics_file")

        return {
            "enabled": True,
            "available": True,
            "fc3_method": "hiphive",
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
            "hiphive_available": True,
            "hiphive_status": "available",
            "hiphive_n_structures": n_structures,
            "hiphive_rattle_std": float(config.rattle_std),
            "hiphive_cutoffs": [float(value) for value in config.cutoffs],
            "hiphive_min_dist": float(config.min_dist),
            "hiphive_fit_matrix_shape": [int(value) for value in fit_matrix.shape],
            "hiphive_n_parameters": int(parameters.size),
            "hiphive_force_rmse_input_eV_per_A": fit_summary["force_rmse_input_eV_per_A"],
            "hiphive_force_rmse_train_eV_per_A": diagnostics.get("force_rmse_train_eV_per_A"),
            "hiphive_max_force_error_train_eV_per_A": diagnostics.get("max_force_error_train_eV_per_A"),
            "hiphive_number_of_force_components": diagnostics.get("number_of_force_components"),
            "hiphive_number_of_fit_parameters": diagnostics.get("number_of_fit_parameters"),
            "hiphive_underdetermined": diagnostics.get("underdetermined"),
            "hiphive_diagnostics": diagnostics,
            "fc2_file": fc2_path.name,
            "fc3_file": fc3_path.name,
            "kappa_file": kappa_path.name,
            "thermal_conductivity_file": thermal_csv.name,
            "thermal_conductivity_plot": thermal_png.name,
            "phonon_lifetime_file": lifetime.get("data_file") if lifetime.get("available") else None,
            "phonon_lifetime_plot": lifetime.get("plot_file") if lifetime.get("available") else None,
            "temperatures": temperatures,
            "kappa_mesh": [int(value) for value in kappa_mesh],
            "fc3_supercell_dim": [int(value) for value in fc3_supercell_dim],
            "fc3_displacement": float(config.fc3_displacement),
            "fc3_cutoff_pair_distance": config.fc3_cutoff_pair_distance,
            "phono3py_symprec": _resolve_phono3py_symprec(config),
            "phono3py_cutoff_frequency": _resolve_phono3py_cutoff_frequency(config),
            "phono3py_symmetry": bool(config.phono3py_symmetry),
            "phono3py_mesh_symmetry": bool(config.phono3py_mesh_symmetry),
            "phono3py_isotope": bool(config.phono3py_isotope),
            "boundary_mfp": float(config.boundary_mfp),
            **hiphive_symmetrization_info,
            "n_structures": n_structures,
            "n_fc3_displacements": n_structures,
            "n_fc2_displacements": n_structures,
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
        reason = f"HiPhive thermal conductivity calculation failed during FC/kappa workflow: {exc}"
        _log(log, reason)
        wte_capability = get_wte_backend_capability()
        return unavailable_thermal_result(
            reason=reason,
            warnings=warnings,
            fc3_method="hiphive",
            kappa_method=config.kappa_method,
            hiphive_available=True,
            hiphive_status="failed",
            thermal_status="failed",
            error_message=reason,
            wigner=bool(config.wigner),
            wigner_requested=bool(config.wigner),
            wigner_available=False,
            wigner_backend=wte_capability["backend"] if config.wigner else None,
            wte_plugin_found=wte_capability["wte_module_found"],
            wte_module_found=wte_capability["wte_module_found"],
            phono3py_version=wte_capability["phono3py_version"],
            phonopy_version=wte_capability["phonopy_version"],
            transport_type="WTE" if config.wigner and wte_capability["available"] else None,
            wigner_unavailable_reason=reason if config.wigner else None,
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
            **_hiphive_metadata(config),
        )


def _write_hiphive_diagnostics(
    *,
    outdir: Path,
    config: WorkflowConfig,
    fc3_supercell_dim: list[int],
    kappa_mesh: list[int],
    training_supercell: Any,
    cluster_space: Any,
    fit_matrix: np.ndarray,
    fit_targets: np.ndarray,
    parameters: np.ndarray,
    optimizer: Any,
    fc2: np.ndarray,
    fc3: np.ndarray,
    force_rmse_samples: list[float],
    warnings: list[str],
) -> dict[str, Any]:
    """Write HiPhive fit and force-constant diagnostics.

    The fit matrix targets are Cartesian force components in eV/Angstrom from
    ASE/Calorine. HiPhive displacements are in Angstrom, so the resulting force
    constants are exported in the unit convention expected by phonopy/phono3py.
    """

    predicted = np.asarray(fit_matrix @ parameters, dtype=float)
    targets = np.asarray(fit_targets, dtype=float)
    residual = predicted - targets
    n_force_components = int(targets.size)
    n_parameters = int(parameters.size)
    force_rmse_train = float(np.sqrt(np.mean(residual**2))) if residual.size else None
    max_force_error = float(np.max(np.abs(residual))) if residual.size else None
    diagnostics_files = {
        "hiphive_fit_diagnostics_json": "hiphive_fit_diagnostics.json",
        "hiphive_fit_diagnostics_txt": "hiphive_fit_diagnostics.txt",
        "fc2_diagnostics_json": "fc2_diagnostics.json",
        "fc3_diagnostics_json": "fc3_diagnostics.json",
        "hiphive_force_fit_plot": "hiphive_force_fit.png",
    }
    cluster_counts = _cluster_counts_by_order(cluster_space)
    fc2_diagnostics = _force_constant_diagnostics(fc2, order=2)
    fc3_diagnostics = _force_constant_diagnostics(fc3, order=3)
    fc2_diagnostics["acoustic_sum_rule_residual_max_abs"] = _fc2_asr_residual(fc2)
    n_atoms_supercell = int(len(training_supercell))
    for item in (fc2_diagnostics, fc3_diagnostics):
        item.update(
            {
                "method": "hiphive",
                "supercell_dim": [int(value) for value in fc3_supercell_dim],
                "fc3_supercell_dim": [int(value) for value in fc3_supercell_dim],
                "n_atoms_supercell": n_atoms_supercell,
                "force_units_note": "Forces are evaluated by ASE/Calorine CPUNEP in eV/Angstrom; HiPhive uses Angstrom displacements.",
                "export_note": "Exported through HiPhive ForceConstants.write_to_phonopy/write_to_phono3py for phono3py readability.",
            }
        )
    fit_diagnostics: dict[str, Any] = {
        "method": "hiphive",
        "n_structures": int(config.n_structures),
        "rattle_std": float(config.rattle_std),
        "rattle_std_unit": "Angstrom",
        "cutoffs": [float(value) for value in config.cutoffs],
        "cutoffs_order_note": "cutoffs[0] is second-order and cutoffs[1] is third-order, matching HiPhive ClusterSpace convention.",
        "min_dist": float(config.min_dist),
        "min_dist_unit": "Angstrom",
        "supercell_dim": [int(value) for value in fc3_supercell_dim],
        "fc3_supercell_dim": [int(value) for value in fc3_supercell_dim],
        "kappa_mesh": [int(value) for value in kappa_mesh],
        "number_of_atoms_supercell": n_atoms_supercell,
        "training_supercell": _ase_atoms_diagnostics(training_supercell),
        "atom_order_note": "Rattled structures are generated from the Phonopy supercell and prepared against the same reference supercell before fitting.",
        "force_units_note": "Training forces are Cartesian ASE/Calorine CPUNEP forces in eV/Angstrom.",
        "fit_target_units_note": "StructureContainer targets are force components in eV/Angstrom.",
        "single_point_calculator": True,
        "optimizer_source": "StructureContainer.get_fit_data()",
        "rotational_sum_rules": ["Huang", "Born-Huang"],
        "fc2_export_method": "ForceConstants.write_to_phonopy",
        "fc3_export_method": "ForceConstants.write_to_phono3py",
        "number_of_force_components": n_force_components,
        "number_of_fit_parameters": n_parameters,
        "fit_matrix_shape": [int(value) for value in fit_matrix.shape],
        "force_rmse_input_eV_per_A": float(np.mean(force_rmse_samples)) if force_rmse_samples else None,
        "force_rmse_train_eV_per_A": force_rmse_train,
        "max_force_error_train_eV_per_A": max_force_error,
        "underdetermined": bool(n_parameters > n_force_components),
        "cluster_counts_by_order": cluster_counts,
        "fc2_shape": [int(value) for value in fc2.shape],
        "fc3_shape": [int(value) for value in fc3.shape],
        "fc2_diagnostics": fc2_diagnostics,
        "fc3_diagnostics": fc3_diagnostics,
        "warnings": warnings,
        "files": diagnostics_files,
    }
    (outdir / diagnostics_files["hiphive_fit_diagnostics_json"]).write_text(
        json.dumps(fit_diagnostics, indent=2),
        encoding="utf-8",
    )
    (outdir / diagnostics_files["fc2_diagnostics_json"]).write_text(
        json.dumps(fc2_diagnostics, indent=2),
        encoding="utf-8",
    )
    (outdir / diagnostics_files["fc3_diagnostics_json"]).write_text(
        json.dumps(fc3_diagnostics, indent=2),
        encoding="utf-8",
    )
    txt_lines = [
        "HiPhive fit diagnostics",
        f"n_structures: {fit_diagnostics['n_structures']}",
        f"rattle_std: {fit_diagnostics['rattle_std']}",
        f"cutoffs: {fit_diagnostics['cutoffs']}",
        f"cutoffs_order_note: {fit_diagnostics['cutoffs_order_note']}",
        f"min_dist: {fit_diagnostics['min_dist']}",
        f"supercell_dim: {fit_diagnostics['supercell_dim']}",
        f"number_of_atoms_supercell: {fit_diagnostics['number_of_atoms_supercell']}",
        f"force_units_note: {fit_diagnostics['force_units_note']}",
        f"single_point_calculator: {fit_diagnostics['single_point_calculator']}",
        f"optimizer_source: {fit_diagnostics['optimizer_source']}",
        f"fc2_export_method: {fit_diagnostics['fc2_export_method']}",
        f"fc3_export_method: {fit_diagnostics['fc3_export_method']}",
        f"number_of_force_components: {n_force_components}",
        f"number_of_fit_parameters: {n_parameters}",
        f"force_rmse_train_eV_per_A: {force_rmse_train}",
        f"max_force_error_train_eV_per_A: {max_force_error}",
        f"underdetermined: {fit_diagnostics['underdetermined']}",
        f"cluster_counts_by_order: {cluster_counts}",
        f"fc2_shape: {fit_diagnostics['fc2_shape']}",
        f"fc3_shape: {fit_diagnostics['fc3_shape']}",
        f"fc2_acoustic_sum_rule_residual_max_abs: {fc2_diagnostics['acoustic_sum_rule_residual_max_abs']}",
    ]
    (outdir / diagnostics_files["hiphive_fit_diagnostics_txt"]).write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    _plot_force_fit(targets, predicted, outdir / diagnostics_files["hiphive_force_fit_plot"])
    return fit_diagnostics


def canonicalize_fractional_positions_for_hiphive(atoms: Any, *, tolerance: float = 1e-6) -> tuple[Any, str | None]:
    """Return an equivalent cell with fractional positions canonicalized.

    HiPhive's orbit builder can fail for atoms represented infinitesimally
    outside the unit-cell boundary, for example fractional coordinates like
    0.999999998 that are physically equivalent to 0.0.  Relaxation can produce
    such values for non-orthogonal cells.  Canonicalizing the representation
    before HiPhive sees it preserves the structure while avoiding fragile image
    indexing such as ``(0, 68) is not in list``.
    """

    fit_atoms = atoms.copy()
    scaled = np.asarray(fit_atoms.get_scaled_positions(wrap=False), dtype=float)
    if scaled.size == 0:
        return fit_atoms, None
    canonical = np.mod(scaled, 1.0)
    canonical[np.isclose(canonical, 1.0, atol=tolerance, rtol=0.0)] = 0.0
    canonical[np.isclose(canonical, 0.0, atol=tolerance, rtol=0.0)] = 0.0
    changed = bool(np.max(np.abs(canonical - scaled)) > tolerance) if canonical.size else False
    fit_atoms.set_scaled_positions(canonical)
    warning = None
    if changed:
        warning = (
            "Canonicalized fractional coordinates before HiPhive fitting; "
            "positions within periodic boundaries were wrapped to [0, 1)."
        )
    return fit_atoms, warning


def _annotate_transport_diagnostics(
    *,
    outdir: Path,
    diagnostics: dict[str, Any],
    kappa_diagnostics: dict[str, Any],
    lifetime: dict[str, Any],
) -> dict[str, Any]:
    """Add post-phono3py transport metadata to HiPhive diagnostics."""

    diagnostics = dict(diagnostics)
    diagnostics["kappa_hdf5"] = kappa_diagnostics
    diagnostics["kappa_hdf5_fields_found"] = kappa_diagnostics.get("fields_found", [])
    diagnostics["lifetime_source"] = lifetime.get("source") if lifetime.get("available") else "unavailable"
    diagnostics["lifetime_available"] = bool(lifetime.get("available"))
    diagnostics["lifetime_reason"] = lifetime.get("reason")
    diagnostics["lifetime_warnings"] = lifetime.get("warnings", [])
    diagnostics_files = diagnostics.get("files", {})
    diagnostics_path = diagnostics_files.get("hiphive_fit_diagnostics_json")
    if diagnostics_path:
        (outdir / str(diagnostics_path)).write_text(
            json.dumps(diagnostics, indent=2),
            encoding="utf-8",
        )
    return diagnostics


def _force_constant_diagnostics(fc: np.ndarray, *, order: int) -> dict[str, Any]:
    finite = np.asarray(fc, dtype=float)
    nonzero = finite[np.abs(finite) > 0]
    return {
        "order": order,
        "shape": [int(value) for value in finite.shape],
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "mean_abs": float(np.mean(np.abs(finite))) if finite.size else None,
        "max_abs": float(np.max(np.abs(finite))) if finite.size else None,
        "n_nonzero": int(nonzero.size),
    }


def _fc2_asr_residual(fc2: np.ndarray) -> float | None:
    if fc2.ndim != 4:
        return None
    residual = np.sum(fc2, axis=1)
    return float(np.max(np.abs(residual))) if residual.size else None


def _ase_atoms_diagnostics(atoms: Any) -> dict[str, Any]:
    """Return JSON-safe ASE Atoms metadata for force-constant audits."""

    info: dict[str, Any] = {"n_atoms": int(len(atoms))}
    try:
        info["formula"] = atoms.get_chemical_formula()
    except Exception:
        info["formula"] = None
    try:
        info["cell"] = np.asarray(atoms.get_cell(), dtype=float).tolist()
    except Exception:
        info["cell"] = None
    try:
        info["pbc"] = [bool(value) for value in atoms.get_pbc()]
    except Exception:
        info["pbc"] = None
    try:
        symbols = list(atoms.get_chemical_symbols())
        info["symbols_sample"] = symbols[:12]
        info["n_symbols_sampled"] = min(len(symbols), 12)
    except Exception:
        info["symbols_sample"] = []
        info["n_symbols_sampled"] = 0
    return info


def _cluster_counts_by_order(cluster_space: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for orbit in getattr(cluster_space, "orbits", []):
        order = getattr(orbit, "order", None)
        if order is None and hasattr(orbit, "prototype_cluster"):
            order = len(getattr(orbit, "prototype_cluster"))
        key = str(order if order is not None else "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _plot_force_fit(targets: np.ndarray, predicted: np.ndarray, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if targets.size == 0 or predicted.size == 0:
        return
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=200)
    ax.scatter(targets, predicted, s=8, alpha=0.45, color="#1f7a8c", edgecolors="none")
    lower = float(min(np.min(targets), np.min(predicted)))
    upper = float(max(np.max(targets), np.max(predicted)))
    ax.plot([lower, upper], [lower, upper], color="#a23b72", linewidth=1.4, linestyle="--")
    ax.set_xlabel("Reference force component (eV/Angstrom)")
    ax.set_ylabel("Fitted force component (eV/Angstrom)")
    ax.set_title("HiPhive force fit")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _hiphive_metadata(
    config: WorkflowConfig,
    *,
    fc3_supercell_dim: list[int] | None = None,
    kappa_mesh: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "temperatures": [float(value) for value in config.temperatures],
        "kappa_mesh": kappa_mesh if kappa_mesh is not None else config.kappa_mesh,
        "fc3_supercell_dim": fc3_supercell_dim if fc3_supercell_dim is not None else config.fc3_supercell_dim,
        "fc3_displacement": float(config.fc3_displacement),
        "fc3_cutoff_pair_distance": config.fc3_cutoff_pair_distance,
        "phono3py_symprec": _resolve_phono3py_symprec(config),
        "phono3py_cutoff_frequency": _resolve_phono3py_cutoff_frequency(config),
        "phono3py_symmetry": bool(config.phono3py_symmetry),
        "phono3py_mesh_symmetry": bool(config.phono3py_mesh_symmetry),
        "phono3py_isotope": bool(config.phono3py_isotope),
        "boundary_mfp": float(config.boundary_mfp),
        **_hiphive_symmetrization_info(config),
        "max_fc3_displacements": config.max_fc3_displacements,
        "smoke_test": config.max_fc3_displacements is not None,
        "hiphive_n_structures": int(config.n_structures),
        "hiphive_rattle_std": float(config.rattle_std),
        "hiphive_cutoffs": [float(value) for value in config.cutoffs],
        "hiphive_min_dist": float(config.min_dist),
        "experimental_parameters": {
            "n_structures": int(config.n_structures),
            "rattle_std": float(config.rattle_std),
            "cutoffs": [float(value) for value in config.cutoffs],
            "min_dist": float(config.min_dist),
        },
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


def _hiphive_symmetrization_info(config: WorkflowConfig) -> dict[str, Any]:
    requested = bool(config.phono3py_symmetrize_fc2 or config.phono3py_symmetrize_fc3)
    return {
        "phono3py_symmetrize_fc2": False,
        "phono3py_symmetrize_fc2_requested": bool(config.phono3py_symmetrize_fc2),
        "phono3py_symmetrize_fc2_applied": False,
        "phono3py_symmetrize_fc3": False,
        "phono3py_symmetrize_fc3_requested": bool(config.phono3py_symmetrize_fc3),
        "phono3py_symmetrize_fc3_applied": False,
        "phono3py_symmetrization_policy": "not_used_for_hiphive_route",
        "hiphive_rotational_sum_rules": ["Huang", "Born-Huang"],
        "hiphive_uses_phono3py_symmetrize": False,
        "hiphive_symmetrization_note": (
            "HiPhive path fits force constants with HiPhive "
            "and rotational sum rules; phono3py.symmetrize_fc2/fc3 are not called."
        ),
        "phono3py_symmetrize_requested_but_ignored": requested,
    }


def _resolve_kappa_mesh(config: WorkflowConfig) -> list[int]:
    return resolve_common_q_mesh(config.mesh, config.kappa_mesh)


def _resolve_transport_type(config: WorkflowConfig, wte_capability: dict[str, Any] | None = None) -> str | None:
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
