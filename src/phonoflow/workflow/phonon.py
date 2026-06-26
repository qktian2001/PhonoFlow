"""Real Phonopy workflow helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import phonopy
from phonopy.file_IO import write_force_constants_to_hdf5

from phonoflow.analysis.bandpath import (
    band_path_from_kpath_result,
    generate_kpath,
    high_symmetry_path_metadata,
    write_band_yaml_path_metadata,
    write_band_path_json,
)
from phonoflow.analysis.group_velocity import compute_phonon_group_velocity
from phonoflow.analysis.postprocessing_diagnostics import (
    write_band_diagnostics,
    write_dos_diagnostics,
)
from phonoflow.calculators.base import CalculatorBackend
from phonoflow.config import WorkflowConfig
from phonoflow.kpath.schema import serialize_kpath_result
from phonoflow.io.force_constants_io import write_force_constants_text
from phonoflow.band import (
    band_data_from_phonopy_dict,
    band_data_to_metadata,
    export_phonon_band_data,
    plot_phonon_band,
)
from phonoflow.plotting.plot_dos import plot_phonon_dos
from phonoflow.workflow.displace import create_phonopy, generate_displacements
from phonoflow.workflow.force_eval import evaluate_forces


def run_phonon_calculation(
    relaxed_atoms: Any,
    backend: CalculatorBackend,
    config: WorkflowConfig,
    outdir: Path,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run finite-displacement Phonopy workflow for a relaxed structure."""

    outdir.mkdir(parents=True, exist_ok=True)
    fc2_started = time.perf_counter()
    _log(log, "Creating Phonopy object")
    try:
        phonon = create_phonopy(
            relaxed_atoms,
            config.supercell_dim,
            primitive_matrix=config.primitive_matrix,
            symprec=config.phonopy_symprec,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not create the Phonopy object with primitive_matrix="
            f"{config.primitive_matrix!r}. Try --primitive-matrix auto or "
            f"--primitive-matrix identity. Details: {exc}"
        ) from exc

    _log(log, f"Generating finite displacements with distance={config.displacement}")
    displaced_supercells = generate_displacements(phonon, config.displacement)
    n_displaced = len(displaced_supercells)
    _log(log, f"Generated {n_displaced} displaced supercells")

    force_sets = evaluate_forces(
        displaced_supercells,
        backend,
        config.model_path,
        log=log,
        audit_outdir=outdir if config.save_force_audit else None,
        audit_label="fc2",
    )
    phonon.forces = force_sets

    _log(log, "Producing second-order force constants")
    phonon.produce_force_constants(calculate_full_force_constants=True)
    postprocess_info = _postprocess_force_constants(phonon, config, log=log)
    force_constants_path = outdir / "force_constants.hdf5"
    write_force_constants_to_hdf5(phonon.force_constants, filename=str(force_constants_path))
    text_fc_info = _export_fc2_text_if_requested(phonon, config, outdir, log=log)

    phonopy_yaml_path = outdir / "phonopy.yaml"
    phonon.save(filename=phonopy_yaml_path)
    fc2_seconds = time.perf_counter() - fc2_started

    postprocess_started = time.perf_counter()
    _log(
        log,
        "Generating band path with "
        f"kpath_mode={config.kpath_mode}, band_selector={config.band}, "
        f"bandpath_symprec={config.bandpath_symprec}, "
        f"bandpath_with_time_reversal={config.bandpath_with_time_reversal}",
    )
    kpath = generate_kpath(
        relaxed_atoms,
        kpath_mode=config.kpath_mode,
        npoints=config.band_npoints,
        symprec=config.bandpath_symprec,
        with_time_reversal=config.bandpath_with_time_reversal,
    )
    band_path = band_path_from_kpath_result(kpath, npoints=config.band_npoints)
    high_symmetry_path = high_symmetry_path_metadata(
        band_path.segments,
        source=band_path.source,
        symprec=config.bandpath_symprec,
        with_time_reversal=config.bandpath_with_time_reversal,
        kpath=kpath,
    )
    phonon.run_band_structure(band_path.qpoints, labels=band_path.labels)
    band_yaml_path = outdir / "band.yaml"
    phonon.write_yaml_band_structure(filename=band_yaml_path)
    write_band_yaml_path_metadata(
        band_yaml_path,
        band_path,
        high_symmetry_path,
        kpath=kpath,
        requested_mode=config.kpath_mode,
    )
    band_dict = phonon.get_band_structure_dict()

    band_data = band_data_from_phonopy_dict(band_dict, band_path, imag_threshold=config.imag_threshold)
    band_export_files = export_phonon_band_data(band_data, outdir)
    band_path_json = outdir / "band_path.json"
    write_band_path_json(
        band_path,
        band_path_json,
        labels_for_plot=band_data.tick_labels,
        tick_positions=band_data.tick_positions,
        high_symmetry_path=high_symmetry_path,
        kpath=kpath,
        requested_mode=config.kpath_mode,
    )
    band_plot_path = outdir / "phonon_band.png"
    plot_phonon_band(
        band_data,
        band_plot_path,
        title="Phonon dispersion",
        dpi=config.plot_dpi,
    )
    band_diagnostics_path = outdir / "phonon_band_diagnostics.json"
    band_diagnostics = write_band_diagnostics(
        band_data=band_data,
        source_file=band_yaml_path,
        plot_file=band_plot_path,
        output_path=band_diagnostics_path,
        high_symmetry_path=high_symmetry_path,
    )
    band_dict["labels"] = band_path.labels

    dos_info: dict[str, Any] = {
        "dos_generated": False,
        "output_files": {},
    }
    if config.dos:
        dos_info = _run_total_dos(phonon, config, outdir, log=log)

    group_velocity_info = compute_phonon_group_velocity(
        phonon=phonon,
        output_dir=outdir,
        mesh=config.mesh,
        plot=True,
        dpi=config.plot_dpi,
        logger=log,
    )
    group_velocity_output_files = _group_velocity_output_files(group_velocity_info)

    frequencies = _flatten_band_frequencies(band_dict)
    primitive_matrix_resolved = getattr(phonon, "primitive_matrix", None)
    postprocess_seconds = time.perf_counter() - postprocess_started
    return {
        "phonon": phonon,
        "band_structure": band_dict,
        "frequencies_THz": frequencies,
        "n_displaced_supercells": n_displaced,
        "output_files": {
            "force_constants": force_constants_path.name,
            "phonopy_yaml": phonopy_yaml_path.name,
            "band_yaml": band_yaml_path.name,
            "band_plot": band_plot_path.name,
            "band_path": band_path_json.name,
            **text_fc_info["output_files"],
            **band_export_files,
            "band_diagnostics": band_diagnostics_path.name,
            **dos_info["output_files"],
            **group_velocity_output_files,
        },
        "band_labels": band_path.labels,
        "band_source": band_path.source,
        "band_distances": band_data.tick_positions,
        "band_tick_labels": band_data.tick_labels,
        "high_symmetry_path": high_symmetry_path,
        "kpath": serialize_kpath_result(kpath, requested_mode=config.kpath_mode),
        "kpath_mode_requested": config.kpath_mode,
        "bandpath_symprec": config.bandpath_symprec,
        "bandpath_with_time_reversal": config.bandpath_with_time_reversal,
        "bandpath_structure_source": "relaxed_structure" if config.relax else "input_structure",
        "band_metadata": band_data_to_metadata(band_data),
        "band_diagnostics": band_diagnostics,
        "dos_generated": dos_info["dos_generated"],
        "dos_diagnostics": dos_info.get("diagnostics"),
        "group_velocity": group_velocity_info,
        "primitive_matrix_resolved": _to_serializable_matrix(primitive_matrix_resolved),
        "phonopy_version": phonopy.__version__,
        "timing_breakdown": {
            "fc2_harmonic_seconds": round(fc2_seconds, 6),
            "phonon_postprocess_seconds": round(postprocess_seconds, 6),
        },
        **text_fc_info["result_fields"],
        **postprocess_info,
    }


def _flatten_band_frequencies(band_dict: dict[str, Any]) -> np.ndarray:
    frequencies = band_dict.get("frequencies")
    if frequencies is None:
        raise RuntimeError("Phonopy band structure did not contain frequencies.")
    flattened_segments: list[np.ndarray] = []
    for segment in frequencies:
        flattened_segments.append(np.asarray(segment, dtype=float).reshape(-1))
    if not flattened_segments:
        raise RuntimeError("Phonopy band structure contained no frequency values.")
    return np.concatenate(flattened_segments)


def _postprocess_force_constants(
    phonon: Any,
    config: WorkflowConfig,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    symmetrize_applied = False
    asr_applied = False

    if config.symmetrize_fc:
        try:
            phonon.symmetrize_force_constants(level=1, show_drift=False)
            symmetrize_applied = True
            asr_applied = True
            _log(log, "Applied Phonopy force-constant symmetrization")
        except Exception as exc:
            warnings.append(f"Could not symmetrize force constants: {exc}")

    if config.asr and not asr_applied:
        try:
            phonon.symmetrize_force_constants(level=1, show_drift=False)
            asr_applied = True
            _log(log, "Applied acoustic sum rule via force-constant symmetrization")
        except Exception as exc:
            warnings.append(f"Could not apply acoustic sum rule: {exc}")

    return {
        "asr_applied": asr_applied,
        "symmetrize_fc_applied": symmetrize_applied,
        "postprocess_warnings": warnings,
    }


def _export_fc2_text_if_requested(
    phonon: Any,
    config: WorkflowConfig,
    outdir: Path,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    result_fields: dict[str, Any] = {
        "export_fc2_text": bool(config.export_fc2_text),
        "force_constants_text_exported": False,
        "force_constants_text_format": "phonopy-text-fc2",
        "force_constants_text_shape": None,
        "phonopy_force_constants_file": None,
        "shengbte_fc2_file": None,
        "force_constants_text_warnings": [],
    }
    output_files: dict[str, str] = {}
    if not config.export_fc2_text:
        return {"result_fields": result_fields, "output_files": output_files}

    try:
        n_supercell_atoms = len(phonon.supercell)
        export_info = write_force_constants_text(
            phonon.force_constants,
            outdir,
            n_supercell_atoms=n_supercell_atoms,
            phonopy_filename=config.fc2_text_name,
            shengbte_filename=config.shengbte_fc2_name,
        )
    except Exception as exc:
        warning = (
            "Could not export FORCE_CONSTANTS text FC2. "
            f"{exc} Try --primitive-matrix identity or use force_constants.hdf5 only."
        )
        _log(log, warning)
        result_fields["force_constants_text_warnings"] = [warning]
        return {"result_fields": result_fields, "output_files": output_files}

    result_fields.update(export_info)
    duplicate_path = outdir / str(config.fc2_text_name)
    shengbte_path = outdir / str(config.shengbte_fc2_name)
    if duplicate_path.exists() and shengbte_path.exists() and duplicate_path != shengbte_path:
        duplicate_path.unlink()
        result_fields["force_constants_duplicate_removed"] = True
        result_fields["phonopy_force_constants_file"] = None
    else:
        result_fields["force_constants_duplicate_removed"] = False
    if export_info.get("shengbte_fc2_file"):
        output_files["shengbte_force_constants_2nd"] = str(export_info["shengbte_fc2_file"])
    _log(log, "Exported FORCE_CONSTANTS_2ND text FC2 file")
    return {"result_fields": result_fields, "output_files": output_files}


def _group_velocity_output_files(group_velocity: dict[str, Any]) -> dict[str, str]:
    output_files: dict[str, str] = {}
    data_file = group_velocity.get("data_file")
    plot_file = group_velocity.get("plot_file")
    diagnostics_file = group_velocity.get("diagnostics_file")
    if data_file:
        output_files["group_velocity_data"] = str(data_file)
    if plot_file:
        output_files["group_velocity_plot"] = str(plot_file)
    if diagnostics_file:
        output_files["group_velocity_diagnostics"] = str(diagnostics_file)
    return output_files


def _run_total_dos(
    phonon: Any,
    config: WorkflowConfig,
    outdir: Path,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    _log(log, f"Running total DOS on gamma-centered q-mesh={config.mesh}")
    phonon.run_mesh(config.mesh, is_gamma_center=True)
    phonon.run_total_dos()
    dos_dict = phonon.get_total_dos_dict()
    frequencies = np.asarray(dos_dict["frequency_points"], dtype=float)
    total_dos = np.asarray(dos_dict["total_dos"], dtype=float)

    dos_dat_path = outdir / "phonon_dos.dat"
    with dos_dat_path.open("w", encoding="utf-8") as handle:
        handle.write("# frequency_THz total_DOS\n")
        for frequency, dos_value in zip(frequencies, total_dos, strict=False):
            handle.write(f"{frequency:.10f} {dos_value:.10f}\n")

    dos_plot_path = outdir / "phonon_dos.png"
    plot_phonon_dos(frequencies, total_dos, dos_plot_path, dpi=config.plot_dpi)
    dos_diagnostics_path = outdir / "phonon_dos_diagnostics.json"
    dos_diagnostics = write_dos_diagnostics(
        frequencies=frequencies,
        total_dos=total_dos,
        source_file=dos_dat_path,
        plot_file=dos_plot_path,
        output_path=dos_diagnostics_path,
    )
    return {
        "dos_generated": True,
        "output_files": {
            "dos_data": dos_dat_path.name,
            "dos_plot": dos_plot_path.name,
            "dos_diagnostics": dos_diagnostics_path.name,
        },
        "diagnostics": dos_diagnostics,
    }


def _band_tick_distances(band_dict: dict[str, Any]) -> list[float]:
    distances = band_dict.get("distances") or []
    ticks: list[float] = []
    for segment in distances:
        values = np.asarray(segment, dtype=float)
        if values.size:
            ticks.extend([float(values[0]), float(values[-1])])
    return ticks


def _to_serializable_matrix(matrix: Any) -> Any:
    if matrix is None:
        return None
    try:
        return np.asarray(matrix, dtype=float).tolist()
    except Exception:
        return str(matrix)


def _log(log: Callable[[str], None] | None, message: str) -> None:
    if log is not None:
        log(message)
