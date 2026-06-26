"""Single-workflow summary writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def format_summary(result: dict[str, Any]) -> str:
    """Format a readable single-workflow summary."""

    output_files = result.get("output_files", {})
    kpath = result.get("kpath") if isinstance(result.get("kpath"), dict) else {}
    warnings = result.get("warnings") or []
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "None"
    return "\n".join(
        [
            "# PhonoFlow Summary",
            "",
            "## Input",
            f"Input structure: {result.get('input_path')}",
            f"Model: {result.get('model_path')}",
            f"Input SHA256: {result.get('input_file_hash')}",
            f"Model SHA256: {result.get('model_file_hash')}",
            "",
            "## Resolved settings",
            f"Backend: {result.get('backend_requested', result.get('backend'))} -> {result.get('backend_resolved')}",
            f"Backend requested: {result.get('backend_requested', result.get('backend'))}",
            f"Backend resolved: {result.get('backend_resolved')}",
            f"Backend alias: {result.get('backend_alias')}",
            f"Model backend family: {result.get('model_backend_family')}",
            f"Force inference backend: {result.get('force_infer_backend')}",
            f"DeepMD deterministic: {result.get('deepmd_deterministic')}",
            f"DeepMD reuse calculator: {result.get('deepmd_reuse_calculator')}",
            f"Save force audit: {result.get('save_force_audit')}",
            f"phono3py FC2 force-constant symmetrization: {result.get('phono3py_symmetrize_fc2')}",
            f"phono3py FC3 force-constant symmetrization: {result.get('phono3py_symmetrize_fc3')}",
            f"Output directory: {result.get('output_directory')}",
            f"Relax: {result.get('relax')}",
            f"Relax model: {result.get('relax_model')}",
            f"Relax policy: {result.get('relax_policy')}",
            f"Relax cell: {result.get('relax_cell')}",
            f"Relax mode: {result.get('relax_mode')}",
            f"Relaxed structure path: {result.get('relaxed_structure_path')}",
            f"Property model: {result.get('property_model')}",
            f"Calculation model: {result.get('calculation_model')}",
            f"constant_cell: {result.get('constant_cell')}",
            f"fmax: {result.get('fmax')}",
            f"max_steps: {result.get('max_steps')}",
            f"Optimizer: {result.get('optimizer')}",
            f"Supercell requested: {result.get('supercell_dim_requested')}",
            f"Supercell resolved: {result.get('supercell_dim_resolved')}",
            f"Supercell target length: {result.get('target_supercell_length')}",
            f"Supercell lengths resolved: {result.get('supercell_lengths_resolved')}",
            f"Supercell atoms: {result.get('n_atoms_supercell')}",
            f"Auto supercell warnings: {result.get('auto_supercell_warnings')}",
            f"Primitive matrix: {result.get('primitive_matrix_resolved')}",
            f"Displacement: {result.get('displacement')}",
            f"DOS: {result.get('dos')}",
            f"Q-mesh: {result.get('q_mesh', result.get('mesh_resolved'))}",
            f"Q-mesh centering: {result.get('q_mesh_centering')}",
            f"Q-mesh used for: {result.get('q_mesh_used_for')}",
            f"ASR: {result.get('asr_requested')}",
            f"Symmetrize FC: {result.get('symmetrize_fc_requested')}",
            f"Export FC2 text: {result.get('export_fc2_text')}",
            f"FC method: {result.get('fc_method')}",
            f"Kappa method: {result.get('kappa_method')}",
            f"Solver flags: {result.get('solver_flags')}",
            f"phono3py mesh: {result.get('phono3py_mesh')}",
            f"Compare mode: {result.get('compare_mode')}",
            "",
            "## Structure classification",
            f"Structure type: {result.get('structure_type')}",
            f"Classification method: {result.get('classification_method')}",
            f"Vacuum-like directions: {result.get('vacuum_like_directions')}",
            f"Cell lengths: {result.get('cell_lengths')}",
            f"Atom extents: {result.get('atom_extents')}",
            "",
            "## Structure",
            f"Formula: {result.get('structure_formula')}",
            f"Unit-cell atoms: {result.get('n_atoms_unitcell')}",
            f"Supercell atoms: {result.get('n_atoms_supercell')}",
            f"Cell lengths: {result.get('cell_lengths')}",
            f"Cell angles: {result.get('cell_angles')}",
            "",
            "## Relaxation",
            f"Relax: {result.get('relax')}",
            f"Relax model: {result.get('relax_model')}",
            f"Relax policy: {result.get('relax_policy')}",
            f"Relax cell: {result.get('relax_cell')}",
            f"Relax mode: {result.get('relax_mode')}",
            f"Relaxed structure path: {result.get('relaxed_structure_path')}",
            f"Shared relaxed structure: {result.get('shared_relaxed_structure')}",
            f"constant_cell: {result.get('constant_cell')}",
            f"fmax: {result.get('fmax')}",
            f"max_steps: {result.get('max_steps')}",
            f"Optimizer: {result.get('optimizer')}",
            f"Initial cell: {result.get('initial_cell_lengths')} / {result.get('initial_cell_angles')}",
            f"Final cell: {result.get('final_cell_lengths')} / {result.get('final_cell_angles')}",
            f"Initial volume: {result.get('initial_volume')}",
            f"Final volume: {result.get('final_volume')}",
            f"Volume change: {result.get('volume_change_percent')} %",
            f"Final max force: {result.get('final_max_force_eV_per_A')} eV/A",
            f"Final stress: {result.get('final_stress_GPa')} GPa",
            f"Relax converged: {result.get('relax_converged')}",
            f"Warnings: {result.get('relax_warnings')}",
            "",
            "## Space group",
            f"Phonopy symprec: {result.get('phonopy_symprec', result.get('symprec'))}, angle_tolerance={result.get('angle_tolerance')}",
            f"Initial space group: {_format_spacegroup(result.get('initial_spacegroup'))}",
            f"Final space group: {_format_spacegroup(result.get('final_spacegroup'))}",
            f"Space group changed: {result.get('spacegroup_changed')}",
            f"Change summary: {result.get('spacegroup_change_summary')}",
            f"Report files: {result.get('spacegroup_report_json')}, {result.get('spacegroup_report_txt')}",
            "",
            "## Results",
            f"Dry run: {result.get('dry_run')}",
            f"Relax converged: {result.get('relax_converged')}",
            f"Final max force: {result.get('final_max_force_eV_per_A')} eV/A",
            f"Displaced supercells: {result.get('n_displaced_supercells')}",
            f"Minimum frequency: {result.get('minimum_frequency_THz')} THz",
            f"Maximum frequency: {result.get('maximum_frequency_THz')} THz",
            f"Imaginary frequency: {result.get('has_imaginary_frequency')}",
            f"High-symmetry path: {_format_high_symmetry_path(result.get('high_symmetry_path'))}",
            f"K-path mode: requested={result.get('kpath_mode')}, resolved={result.get('kpath_mode_resolved')}",
            f"K-path dimensionality: {result.get('kpath_dimensionality')}",
            f"K-path generator: {result.get('kpath_source')}",
            f"ASE 2D Bravais lattice: {result.get('kpath_bravais')}",
            f"Vacuum axis: {result.get('vacuum_axis_name')}",
            f"Suggested path: {kpath.get('display_path')}",
            "Band path settings: "
            f"bandpath_symprec={result.get('bandpath_symprec')}, "
            f"bandpath_with_time_reversal={result.get('bandpath_with_time_reversal')}, "
            f"structure_source={result.get('bandpath_structure_source')}",
            f"Phonon dispersion plot: {output_files.get('band_plot')}",
            f"Phonon dispersion diagnostics: {output_files.get('band_diagnostics')}",
            f"DOS plot: {output_files.get('dos_plot')}",
            f"DOS diagnostics: {output_files.get('dos_diagnostics')}",
            f"Force constants HDF5: {output_files.get('force_constants')}",
            f"FORCE_CONSTANTS_2ND: {output_files.get('shengbte_force_constants_2nd')}",
            f"FC2 text export: {result.get('force_constants_text_exported')}",
            f"FC2 shape: {result.get('force_constants_text_shape')}",
            "Warning: This is second-order force constants only. ShengBTE thermal "
            "conductivity still requires third-order force constants.",
            "",
            "## Phonon group velocity",
            *_format_group_velocity(result.get("group_velocity")),
            "",
            "## Thermal conductivity",
            *_format_thermal_conductivity(result.get("thermal_conductivity")),
            "",
            "## Warnings",
            warning_text,
            "",
        ]
    )


def write_summary(result: dict[str, Any], path: Path) -> None:
    """Write a readable summary text file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_summary(result), encoding="utf-8")


def _format_spacegroup(data: Any) -> str:
    if not isinstance(data, dict):
        return "unavailable"
    symbol = data.get("international_symbol")
    number = data.get("spacegroup_number")
    if symbol and number:
        return f"{symbol} (No. {number})"
    return "unavailable"


def _format_group_velocity(data: Any) -> list[str]:
    if not isinstance(data, dict) or not data.get("available"):
        reason = data.get("reason") if isinstance(data, dict) else "not available"
        return [
            "Status: not available",
            f"Reason: {reason or 'not available'}",
        ]

    unit = data.get("unit", "km/s")
    return [
        "Status: available",
        f"Data file: {data.get('data_file')}",
        f"Plot file: {data.get('plot_file')}",
        f"Unit: {unit}",
        "X axis: Frequency (THz)",
        f"Y axis: Group velocity ({unit})",
        f"Max |v_g|: {data.get('max_abs_velocity')} {unit}",
        f"Mean |v_g|: {data.get('mean_abs_velocity')} {unit}",
        f"Diagnostics: {data.get('diagnostics_file')}",
    ]


def _format_high_symmetry_path(data: Any) -> str:
    if isinstance(data, dict):
        return str(data.get("display") or "not available")
    return "not available"


def _format_thermal_conductivity(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return [
            "Status: not available",
            "Reason: thermal_conductivity field is missing",
        ]
    if not data.get("enabled", False):
        return [
            "Status: disabled",
            f"Reason: {data.get('reason', 'Thermal conductivity calculation was not requested.')}",
        ]
    if not data.get("available", False):
        lines = [
            "Status: not available",
            f"FC3 method: {data.get('fc3_method')}",
            f"BTE method: {data.get('kappa_method')}",
            f"Solver flags: {data.get('solver_flags')}",
            f"Wigner: {data.get('wigner')}",
            f"Wigner requested: {data.get('wigner_requested')}",
            f"Wigner available: {data.get('wigner_available')}",
            f"Wigner backend: {data.get('wigner_backend')}",
            f"WTE plugin found: {data.get('wte_plugin_found')}",
            f"WTE module found: {data.get('wte_module_found')}",
            f"Transport type: {data.get('transport_type')}",
            f"phono3py version: {data.get('phono3py_version')}",
            f"phonopy version: {data.get('phonopy_version')}",
            f"Thermal status: {data.get('thermal_status')}",
            f"Temperatures: {data.get('temperatures')}",
            f"Kappa mesh: {data.get('kappa_mesh')}",
            f"FC3 supercell: {data.get('fc3_supercell_dim')}",
            f"FC3 displacement: {data.get('fc3_displacement')}",
            f"FC3 cutoff pair distance: {data.get('fc3_cutoff_pair_distance')}",
            f"Max FC3 displacements: {data.get('max_fc3_displacements')}",
            f"Smoke test: {data.get('smoke_test')}",
            f"HiPhive status: {data.get('hiphive_status')}",
            f"HiPhive available: {data.get('hiphive_available')}",
            f"HiPhive structures: {data.get('hiphive_n_structures')}",
            f"HiPhive rattle std: {data.get('hiphive_rattle_std')}",
            f"HiPhive cutoffs: {data.get('hiphive_cutoffs')}",
            f"HiPhive min distance: {data.get('hiphive_min_dist')}",
            f"Reason: {data.get('reason')}",
            f"Wigner unavailable reason: {data.get('wigner_unavailable_reason')}",
            f"Warnings: {data.get('warnings', [])}",
        ]
        experimental = data.get("experimental_parameters")
        if experimental:
            lines.append(f"Experimental parameters: {experimental}")
        return lines

    files = data.get("files") or {}
    lifetime = data.get("lifetime") or {}
    lines = [
        "Status: available",
        f"FC3 method: {data.get('fc3_method')}",
        f"BTE method: {data.get('kappa_method')}",
        f"Solver flags: {data.get('solver_flags')}",
        f"Wigner: {data.get('wigner')}",
        f"Wigner requested: {data.get('wigner_requested')}",
        f"Wigner available: {data.get('wigner_available')}",
        f"Wigner backend: {data.get('wigner_backend')}",
        f"WTE plugin found: {data.get('wte_plugin_found')}",
        f"WTE module found: {data.get('wte_module_found')}",
        f"Transport type: {data.get('transport_type')}",
        f"phono3py version: {data.get('phono3py_version')}",
        f"phonopy version: {data.get('phonopy_version')}",
        f"Thermal status: {data.get('thermal_status')}",
        f"Temperatures: {data.get('temperatures')}",
        f"Kappa mesh: {data.get('kappa_mesh')}",
        f"FC3 supercell: {data.get('fc3_supercell_dim')}",
        f"FC3 displacements: {data.get('n_fc3_displacements')}",
        f"FC2 displacements: {data.get('n_fc2_displacements')}",
        f"Smoke test: {data.get('smoke_test')}",
        f"HiPhive status: {data.get('hiphive_status')}",
        f"HiPhive available: {data.get('hiphive_available')}",
        f"HiPhive structures: {data.get('hiphive_n_structures')}",
        f"HiPhive rattle std: {data.get('hiphive_rattle_std')}",
        f"HiPhive cutoffs: {data.get('hiphive_cutoffs')}",
        f"HiPhive min distance: {data.get('hiphive_min_dist')}",
            f"HiPhive fit matrix: {data.get('hiphive_fit_matrix_shape')}",
            f"HiPhive parameters: {data.get('hiphive_n_parameters')}",
            f"HiPhive force RMSE train: {data.get('hiphive_force_rmse_train_eV_per_A')} eV/A",
            f"HiPhive max force error train: {data.get('hiphive_max_force_error_train_eV_per_A')} eV/A",
            f"HiPhive force components: {data.get('hiphive_number_of_force_components')}",
            f"HiPhive fit parameters: {data.get('hiphive_number_of_fit_parameters')}",
            f"HiPhive underdetermined: {data.get('hiphive_underdetermined')}",
            f"HiPhive fit summary: {files.get('hiphive_fit_summary')}",
            f"HiPhive fit diagnostics: {files.get('hiphive_fit_diagnostics_json')}",
            f"HiPhive force fit plot: {files.get('hiphive_force_fit_plot')}",
            f"FC2 diagnostics: {files.get('fc2_diagnostics_json')}",
            f"FC3 diagnostics: {files.get('fc3_diagnostics_json')}",
            f"fc3.hdf5: {files.get('fc3_hdf5')}",
        f"fc2.hdf5: {files.get('fc2_hdf5')}",
        f"Kappa HDF5: {files.get('kappa_hdf5')}",
        f"Thermal conductivity CSV: {files.get('thermal_conductivity_csv')}",
        f"Thermal conductivity plot: {files.get('thermal_conductivity_png')}",
        f"Kappa unit: {data.get('kappa_unit')}",
    ]
    if data.get("fc3_method") == "finite-displacement":
        lines.extend(
            [
                f"FD FC2 diagnostics: {files.get('fd_fc2_diagnostics_json')}",
                f"FD FC3 diagnostics: {files.get('fd_fc3_diagnostics_json')}",
                f"FD phono3py input diagnostics: {files.get('fd_phono3py_input_diagnostics_json')}",
            ]
        )
    if lifetime.get("available"):
        lines.extend(
            [
                "Lifetime: available",
                f"Lifetime CSV: {lifetime.get('data_file')}",
                f"Lifetime plot: {lifetime.get('plot_file')}",
                f"Lifetime unit: {lifetime.get('unit')}",
                f"Lifetime source: {lifetime.get('source')}",
                f"Mean lifetime: {lifetime.get('mean_lifetime_ps')} ps",
                f"Max lifetime: {lifetime.get('max_lifetime_ps')} ps",
                f"Lifetime diagnostics: {lifetime.get('diagnostics_file')}",
            ]
        )
        if lifetime.get("warnings"):
            lines.append(f"Lifetime warnings: {lifetime.get('warnings')}")
    else:
        lines.append(f"Lifetime: not available ({lifetime.get('reason', 'not available')})")
    lines.append(f"Warnings: {data.get('warnings', [])}")
    return lines
