"""Compare PhonoFlow results across NEP89, DPA3, and DPA4 models."""

from __future__ import annotations

import csv
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
import yaml

from phonoflow.analysis.bandpath import DEFAULT_SEEKPATH_WITH_TIME_REVERSAL
from phonoflow.config import WorkflowConfig, default_q_mesh
from phonoflow.band.io import load_band_yaml_segments
from phonoflow.calculators import get_backend
from phonoflow.defaults import (
    DEFAULT_NEP89_MODEL_PATH,
    DPA_BACKEND_ALIASES,
    DPA_MODEL_REGISTRY,
    canonical_dpa_alias,
    resolve_dpa_model_path,
)
from phonoflow.exceptions import ConfigError
from phonoflow.io.structure_io import read_structure, write_structure
from phonoflow.reporting.run_report import StepReporter, write_run_report
from phonoflow.reporting.timing_statistics import (
    timing_row_from_breakdown,
    write_calculation_time_statistics,
)
from phonoflow.workflow.pipeline import run_single_workflow
from phonoflow.workflow.relax import relax_structure


MODEL_SPECS: dict[str, dict[str, str | Path | None]] = {
    "nep89": {
        "model_id": "nep89",
        "backend": "calorine",
        "subdir": "nep89",
        "model_path": DEFAULT_NEP89_MODEL_PATH,
        "dpa_model_name": None,
        "display_name": "NEP89",
    },
}

MODEL_DISPLAY_NAMES = {
    "nep89": "NEP89",
    **{alias: str(spec["filename"]) for alias, spec in DPA_MODEL_REGISTRY.items()},
    "dpa3": str(DPA_MODEL_REGISTRY["dpa32"]["filename"]),
    "dpa4": str(DPA_MODEL_REGISTRY["dpa4neo"]["filename"]),
}
MODEL_COLORS = {
    "nep89": "#176B87",
    "dpa31": "#E67E22",
    "dpa32": "#C2410C",
    "dpa33": "#7C3AED",
    "dpa4neo": "#2E8B57",
    "dpa3": "#C2410C",
    "dpa4": "#2E8B57",
}
DPA4_DEFAULT_INFER_BATCH_SIZE = 64
LIFETIME_REQUIRES_THERMAL_REASON = "Phonon lifetime requires thermal conductivity / FC3 calculation."


def _resolve_compare_model_spec(selection: str) -> dict[str, str | Path | None]:
    """Resolve one explicit compare selection to an independent child workflow."""

    requested = str(selection).strip()
    normalized = requested.lower()
    if normalized == "nep89":
        return dict(MODEL_SPECS["nep89"])

    canonical = canonical_dpa_alias(normalized)
    if canonical not in DPA_MODEL_REGISTRY:
        for alias, registry in DPA_MODEL_REGISTRY.items():
            if normalized == str(registry["filename"]).lower():
                canonical = alias
                break
    if canonical in DPA_MODEL_REGISTRY:
        resolution = resolve_dpa_model_path(canonical, None)
        return {
            "model_id": canonical,
            "backend": canonical,
            "subdir": canonical,
            "model_path": resolution.model_path,
            "dpa_model_name": resolution.model_name,
            "model_head": resolution.model_head,
            "display_name": str(DPA_MODEL_REGISTRY[canonical]["filename"]),
        }

    custom_path = Path(requested).expanduser()
    if custom_path.is_file() or custom_path.is_dir():
        suffix = custom_path.suffix.lower()
        deepmd = suffix in {".pt", ".pth", ".pb"} or custom_path.is_dir()
        model_id = f"custom_{custom_path.stem.lower().replace(' ', '_')}"
        return {
            "model_id": model_id,
            "backend": "deepmd" if deepmd else "calorine",
            "subdir": model_id,
            "model_path": custom_path.resolve(),
            "dpa_model_name": custom_path.name if deepmd else None,
            "model_head": None,
            "display_name": custom_path.name,
        }

    raise ConfigError(
        f"Unknown compare model '{selection}'. Choose nep89, one of the four bundled DPA filenames, "
        "or an existing custom model path."
    )


class IsolatedModelError(RuntimeError):
    """Carry a child-process failure without discarding return-code evidence."""

    def __init__(
        self,
        return_code: int,
        stdout: str,
        stderr: str,
        runtime_env: dict[str, str] | None = None,
    ) -> None:
        detail = (stderr or stdout or "").strip().splitlines()
        tail = "; ".join(detail[-8:])
        super().__init__(f"isolated model command failed with exit code {return_code}: {tail}")
        self.return_code = int(return_code)
        self.stdout = stdout
        self.stderr = stderr
        self.runtime_env = dict(runtime_env or {})


def compare_models(
    *,
    input_path: Path,
    outdir: Path,
    model_names: list[str],
    compute_kappa: bool,
    overwrite: bool,
    dry_run: bool = False,
    relax: bool = False,
    relax_cell: bool = True,
    isolate: bool = False,
    supercell_dim: tuple[int, int, int] | None = None,
    mesh: tuple[int, int, int] | None = None,
    kappa_mesh: tuple[int, int, int] | None = None,
    target_supercell_length: float | None = None,
    displacement: float | None = None,
    fc3_target_supercell_length: float | None = None,
    fc3_supercell_dim: tuple[int, int, int] | None = None,
    fc3_displacement: float | None = None,
    fc3_cutoff_pair_distance: float | None = None,
    fc3_method: str | None = None,
    kappa_method: str | None = None,
    wigner: bool | None = None,
    temperatures: list[float] | None = None,
    max_fc3_displacements: int | None = None,
    fmax: float | None = None,
    max_steps: int | None = None,
    primitive_matrix: str | None = None,
    dos: bool | None = None,
    export_fc2_text: bool | None = None,
    kpath_mode: str | None = None,
    bandpath_with_time_reversal: bool | None = None,
    phonopy_symprec: float | None = None,
    phono3py_symmetrize_fc2: bool | None = None,
    phono3py_symmetrize_fc3: bool | None = None,
    phono3py_symprec: float | None = None,
    phono3py_cutoff_frequency: float | None = None,
    n_structures: int | None = None,
    rattle_std: float | None = None,
    cutoffs: list[float] | tuple[float, ...] | None = None,
    min_dist: float | None = None,
    deepmd_device: str | None = None,
    deepmd_deterministic: bool | None = None,
    deepmd_reuse_calculator: bool | None = None,
    save_force_audit: bool | None = None,
    dpa_safe_mode: bool = False,
) -> dict[str, Any]:
    """Run model workflows sequentially and write comparison artifacts."""

    model_names = [str(item).strip() for item in model_names if str(item).strip()]
    if not model_names:
        raise ConfigError("Compare-models requires at least one model.")
    if len(model_names) > 3:
        raise ConfigError("Compare-models accepts at most three models.")
    if len({item.lower() for item in model_names}) != len(model_names):
        raise ConfigError("Compare-models selections must be unique.")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    parent_log_path = outdir / "run.log"
    parent_log_path.write_text(
        "\n".join(
            [
                "[1/8] Reading input structure",
                f"  - input path: {input_path}",
                "[2/8] Resolving default settings",
                "Resolved PhonoFlow settings",
                f"  - selected model list: {', '.join(model_names)}",
                f"  - compute_kappa: {bool(compute_kappa)}",
                f"  - FC2 matrix: {supercell_dim if supercell_dim is not None else 'auto'}",
                f"  - phonon mesh: {mesh if mesh is not None else 'auto'}",
                f"  - FC2 target: {target_supercell_length if target_supercell_length is not None else 15.0} Å",
                f"  - FC3 target: {fc3_target_supercell_length if fc3_target_supercell_length is not None else 10.0} Å",
                f"  - kappa method: {kappa_method or 'rta'}",
                f"  - temperatures: {temperatures if temperatures is not None else [300.0]}",
                f"  - kappa mesh: {kappa_mesh if kappa_mesh is not None else (mesh if mesh is not None else default_q_mesh())}",
                f"  - relax: {bool(relax)}",
                f"  - relax_cell: {bool(relax_cell) if relax else False}",
                f"  - dry_run: {bool(dry_run)}",
                f"  - isolate children: {bool(isolate)}",
                f"  - output directory: {outdir}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    console = Console()
    reporter = StepReporter(total=max(len(model_names) * 2, 1), log_path=outdir / "validation.log", console=console)
    rows: list[dict[str, Any]] = []
    commands: list[str] = []
    step_no = 0
    child_input_path = input_path
    shared_relaxed_structure_path: Path | None = None
    shared_relax_info: dict[str, Any] = {}
    if relax:
        shared_relaxed_structure_path, shared_relax_info = _prepare_shared_relaxed_structure(
            input_path=input_path,
            outdir=outdir,
            relax_cell=relax_cell,
            fmax=fmax,
            max_steps=max_steps,
            dry_run=dry_run,
            overwrite=overwrite,
        )
        child_input_path = shared_relaxed_structure_path
        with parent_log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        "[3/8] Shared NEP89 pre-relaxation",
                        "  - relax_policy: shared_nep89_pre_relax",
                        "  - relax_model: NEP89",
                        f"  - relax_cell: {bool(relax_cell)}",
                        f"  - relaxed structure: {shared_relaxed_structure_path}",
                        "",
                    ]
                )
            )
    for index, model_name in enumerate(model_names, start=1):
        try:
            spec = _resolve_compare_model_spec(model_name)
            normalized = str(spec["model_id"])
            MODEL_DISPLAY_NAMES[normalized] = str(spec["display_name"])
        except ConfigError as exc:
            normalized = model_name.strip().lower()
            row = _failed_row(
                normalized,
                outdir / normalized,
                str(exc),
                command_argv=[],
                backend_requested=None,
                model_path=None,
            )
            rows.append(row)
            step_no += 1
            reporter.step(step_no, f"Compare model {index}/{len(model_names)}: {normalized}", status="failed", details=[row["error_message"]])
            continue
        model_outdir = outdir / str(spec["subdir"])
        command_argv = _model_command(
            input_path=child_input_path,
            outdir=model_outdir,
            backend=str(spec["backend"]),
            model_path=Path(spec["model_path"]) if spec.get("model_path") else None,
            model_head=str(spec["model_head"]) if spec.get("model_head") else None,
            compute_kappa=compute_kappa,
            relax=False if relax else relax,
            relax_cell=False if relax else relax_cell,
            dry_run=dry_run,
            overwrite=overwrite,
            supercell_dim=supercell_dim,
            mesh=mesh,
            kappa_mesh=kappa_mesh,
            target_supercell_length=target_supercell_length,
            displacement=displacement,
            fc3_target_supercell_length=fc3_target_supercell_length,
            fc3_supercell_dim=fc3_supercell_dim,
            fc3_displacement=fc3_displacement,
            fc3_cutoff_pair_distance=fc3_cutoff_pair_distance,
            fc3_method=fc3_method,
            kappa_method=kappa_method,
            wigner=wigner,
            temperatures=temperatures,
            max_fc3_displacements=max_fc3_displacements,
            fmax=fmax,
            max_steps=max_steps,
            primitive_matrix=primitive_matrix,
            dos=dos,
            export_fc2_text=export_fc2_text,
            kpath_mode=kpath_mode,
            bandpath_with_time_reversal=bandpath_with_time_reversal,
            phonopy_symprec=1e-5 if phonopy_symprec is None else phonopy_symprec,
            phono3py_symmetrize_fc2=phono3py_symmetrize_fc2,
            phono3py_symmetrize_fc3=phono3py_symmetrize_fc3,
            phono3py_symprec=1e-5 if phono3py_symprec is None else phono3py_symprec,
            phono3py_cutoff_frequency=1e-4 if phono3py_cutoff_frequency is None else phono3py_cutoff_frequency,
            n_structures=n_structures,
            rattle_std=rattle_std,
            cutoffs=cutoffs,
            min_dist=min_dist,
            deepmd_device=deepmd_device,
            deepmd_deterministic=deepmd_deterministic,
            deepmd_reuse_calculator=deepmd_reuse_calculator,
            save_force_audit=save_force_audit,
            dpa_safe_mode=dpa_safe_mode and canonical_dpa_alias(str(spec["backend"])) == "dpa4neo",
        )
        command_text = shlex.join(command_argv)
        resolved_model_path = _resolved_spec_model_path(normalized, spec)
        commands.append(command_text)
        with parent_log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        f"Child {index}/{len(model_names)}",
                        f"  - model: {normalized}",
                        f"  - command: {command_text}",
                        f"  - output directory: {model_outdir}",
                        "",
                    ]
                )
            )
        step_no += 1
        reporter.step(
            step_no,
            f"Compare model {index}/{len(model_names)}: {normalized}",
            details=[
                f"model name: {normalized}",
                f"resolved backend request: {spec['backend']}",
                f"model path: {resolved_model_path or 'auto / bundled'}",
                f"command: {command_text}",
                f"artifacts path: {model_outdir}",
            ],
        )
        is_dpa = str(spec["backend"]) in DPA_BACKEND_ALIASES or str(spec["backend"]) == "deepmd"
        config = WorkflowConfig(
            input_path=child_input_path,
            outdir=model_outdir,
            model_path=Path(spec["model_path"]) if spec.get("model_path") else None,
            backend=str(spec["backend"]),
            backend_alias=normalized,
            dpa_model_name=spec.get("dpa_model_name"),
            deepmd_model_head=str(spec["model_head"]) if spec.get("model_head") else None,
            compute_kappa=compute_kappa,
            overwrite=overwrite,
            dry_run=dry_run,
            relax=False if relax else relax,
            relax_cell=False if relax else relax_cell,
            supercell_dim=supercell_dim if supercell_dim is not None else "auto",
            mesh=mesh if mesh is not None else "auto",
            kappa_mesh=kappa_mesh if kappa_mesh is not None else "auto",
            target_supercell_length=(
                12.0
                if dpa_safe_mode and canonical_dpa_alias(str(spec["backend"])) == "dpa4neo"
                else (target_supercell_length if target_supercell_length is not None else 15.0)
            ),
            fc3_method=fc3_method or "finite-displacement",
            fc3_supercell_dim=fc3_supercell_dim if fc3_supercell_dim is not None else "auto",
            fc3_target_supercell_length=(
                fc3_target_supercell_length
                if fc3_target_supercell_length is not None
                else 10.0
            ),
            fc3_displacement=fc3_displacement if fc3_displacement is not None else 0.03,
            fc3_cutoff_pair_distance=fc3_cutoff_pair_distance,
            kappa_method=kappa_method or "rta",
            wigner=bool(wigner) if wigner is not None else False,
            temperatures=temperatures if temperatures is not None else [300.0],
            max_fc3_displacements=max_fc3_displacements,
            max_supercell_atoms=256 if dpa_safe_mode and canonical_dpa_alias(str(spec["backend"])) == "dpa4neo" else 1000,
            fmax=fmax if fmax is not None else 1e-5,
            max_steps=max_steps if max_steps is not None else 2000,
            primitive_matrix=primitive_matrix or "P",
            kpath_mode=kpath_mode or "auto",
            bandpath_with_time_reversal=(
                DEFAULT_SEEKPATH_WITH_TIME_REVERSAL
                if bandpath_with_time_reversal is None
                else bandpath_with_time_reversal
            ),
            phonopy_symprec=1e-5 if phonopy_symprec is None else phonopy_symprec,
            dos=True if dos is None else dos,
            export_fc2_text=True if export_fc2_text is None else export_fc2_text,
            phono3py_symmetrize_fc2=True if phono3py_symmetrize_fc2 is None else phono3py_symmetrize_fc2,
            phono3py_symmetrize_fc3=True if phono3py_symmetrize_fc3 is None else phono3py_symmetrize_fc3,
            phono3py_symprec=1e-5 if phono3py_symprec is None else phono3py_symprec,
            phono3py_cutoff_frequency=1e-4 if phono3py_cutoff_frequency is None else phono3py_cutoff_frequency,
            n_structures=int(n_structures) if n_structures is not None else 200,
            rattle_std=float(rattle_std) if rattle_std is not None else 0.02,
            cutoffs=[float(value) for value in cutoffs] if cutoffs is not None else [5.0, 4.0],
            min_dist=float(min_dist) if min_dist is not None else 1.8,
            deepmd_device=deepmd_device or "cpu",
            deepmd_deterministic=is_dpa if deepmd_deterministic is None else deepmd_deterministic,
            deepmd_reuse_calculator=True if deepmd_reuse_calculator is None else deepmd_reuse_calculator,
            save_force_audit=is_dpa if save_force_audit is None else save_force_audit,
        )
        try:
            if isolate:
                workflow_result = _run_isolated_model(
                    command=command_argv,
                    outdir=model_outdir,
                    model_name=normalized,
                )
            else:
                workflow_result = run_single_workflow(config)
            report = workflow_result.get("report") or _read_result(model_outdir)
            if relax:
                report = _with_shared_relax_metadata(
                    report,
                    shared_relaxed_structure_path=shared_relaxed_structure_path,
                    shared_relax_info=shared_relax_info,
                    relax_cell=relax_cell,
                    property_model=str(spec["display_name"] or normalized),
                    calculation_model=str(spec["display_name"] or normalized),
                )
                _write_json(model_outdir / "result.json", report)
            row = _summary_row(
                normalized,
                model_outdir,
                workflow_result,
                report,
                command_argv=command_argv,
            )
            if relax:
                row = _with_shared_relax_metadata(
                    row,
                    shared_relaxed_structure_path=shared_relaxed_structure_path,
                    shared_relax_info=shared_relax_info,
                    relax_cell=relax_cell,
                    property_model=str(spec["display_name"] or normalized),
                    calculation_model=str(spec["display_name"] or normalized),
                )
            rows.append(row)
            step_no += 1
            reporter.step(
                step_no,
                f"Compare model {index}/{len(model_names)} result: {normalized}",
                status=str(row["status"]),
                details=[
                    f"status: {row['status']}",
                    f"kavg: {row.get('kavg')}",
                    "resolved settings: "
                    f"backend={report.get('backend_resolved')}, "
                    f"force_model={report.get('force_model_path') or report.get('model_path')}, "
                    f"relax_backend={report.get('relax_backend') or report.get('relax_backend_resolved')}, "
                    f"FC2 symmetrization={report.get('phono3py_symmetrize_fc2')}",
                    f"artifacts path: {model_outdir}",
                ],
            )
        except Exception as exc:
            row = _failed_row(
                normalized,
                model_outdir,
                str(exc),
                command_argv=command_argv,
                backend_requested=str(spec["backend"]),
                model_path=resolved_model_path,
                dpa_model_name=str(spec.get("dpa_model_name") or "") or None,
                return_code=exc.return_code if isinstance(exc, IsolatedModelError) else None,
                stderr=exc.stderr if isinstance(exc, IsolatedModelError) else "",
                runtime_env=exc.runtime_env if isinstance(exc, IsolatedModelError) else None,
            )
            if relax:
                row = _with_shared_relax_metadata(
                    row,
                    shared_relaxed_structure_path=shared_relaxed_structure_path,
                    shared_relax_info=shared_relax_info,
                    relax_cell=relax_cell,
                    property_model=str(spec.get("display_name") or normalized),
                    calculation_model=str(spec.get("display_name") or normalized),
                )
            rows.append(row)
            step_no += 1
            reporter.step(
                step_no,
                f"Compare model {index}/{len(model_names)} result: {normalized}",
                status="failed",
                details=[
                    "status: failed",
                    f"failed reason: {row['error_message']}",
                    f"artifacts path: {model_outdir}",
                ],
            )

    resolved_q_mesh = list(mesh or kappa_mesh) if (mesh is not None or kappa_mesh is not None) else default_q_mesh()
    q_mesh_used_for = ["dos"] if dos is not False else []
    if compute_kappa:
        q_mesh_used_for.append("kappa")
    summary = {
        "success": any(row["status"] == "success" for row in rows),
        "input_path": str(input_path),
        "outdir": str(outdir),
        "compute_kappa": bool(compute_kappa),
        "relax": bool(relax),
        "relax_enabled": bool(relax),
        "relax_model": "NEP89" if relax else None,
        "relax_policy": "shared_nep89_pre_relax" if relax else "input_structure_no_relax",
        "relax_cell": bool(relax_cell) if relax else False,
        "relaxed_structure_path": str(shared_relaxed_structure_path) if shared_relaxed_structure_path else None,
        "shared_relaxed_structure": bool(relax),
        "dpa_safe_mode": bool(dpa_safe_mode),
        "compare_mode": True,
        "kappa_method": kappa_method or "rta",
        "temperatures": [float(value) for value in (temperatures if temperatures is not None else [300.0])],
        "solver_flags": ["--method", kappa_method or "rta"],
        "method_flags": ["--method", kappa_method or "rta"],
        "phono3py_mesh": [int(value) for value in resolved_q_mesh],
        "q_mesh": [int(value) for value in resolved_q_mesh],
        "q_mesh_centering": "gamma",
        "q_mesh_used_for": q_mesh_used_for or ["dos"],
        "elapsed_time_seconds": round(time.perf_counter() - started_at, 3),
        "models": rows,
    }
    for row in rows:
        child_outdir = Path(str(row.get("outdir") or ""))
        row["run_log_path"] = str(child_outdir / "run.log")
    summary["timing_breakdown"] = {
        "unit": "seconds",
        "children": {
            str(row.get("model")): row.get("timing_breakdown") or {}
            for row in rows
            if row.get("model")
        },
    }
    timing_rows = [
        timing_row_from_breakdown(
            model=str(row.get("model")),
            display_name=str(row.get("display_name") or row.get("model")),
            timing_breakdown=row.get("timing_breakdown"),
            compute_kappa=compute_kappa,
        )
        for row in rows
        if row.get("status") == "success"
    ]
    summary["calculation_time_statistics"] = write_calculation_time_statistics(
        outdir,
        timing_rows,
    )
    _write_json(outdir / "timing_breakdown.json", summary["timing_breakdown"])
    summary["kappa_bar_components"] = _kappa_bar_components(rows)
    kappa_plot = _write_kappa_bar_plot(outdir / "comparison_kappa_bar.png", summary["kappa_bar_components"])
    shutil.copyfile(outdir / "comparison_kappa_bar.png", outdir / "comparison_kappa.png")
    group_velocity_plot = _write_group_velocity_comparison(outdir, rows)
    lifetime_plot = _write_lifetime_comparison(outdir, rows, compute_kappa=compute_kappa)
    summary["comparison_plots"] = {
        "kappa_bar": kappa_plot,
        "legacy_kappa": {**kappa_plot, "path": "comparison_kappa.png", "alias_of": "comparison_kappa_bar.png"},
        "thermal_conductivity": _write_thermal_plot(outdir / "comparison_thermal_conductivity.png", rows, dry_run=dry_run),
        "phonon_band": _write_band_plot(outdir / "comparison_phonon_band.png", rows, dry_run=dry_run),
        "dos": _write_dos_plot(outdir / "comparison_dos.png", rows, dry_run=dry_run),
        "group_velocity": group_velocity_plot,
        "phonon_lifetime": lifetime_plot,
    }
    _write_json(outdir / "comparison_summary.json", summary)
    _write_json(outdir / "comparison_result.json", summary)
    _write_csv(outdir / "comparison_summary.csv", rows)
    _write_markdown(outdir / "comparison_summary.md", summary)
    validation_log = outdir / "validation.log"
    with parent_log_path.open("a", encoding="utf-8") as handle:
        if validation_log.exists():
            handle.write("\n[3/8] Running isolated model workflows\n")
            handle.write(validation_log.read_text(encoding="utf-8"))
        handle.write(
            "\n".join(
                [
                    "[8/8] Finalizing comparison artifacts",
                    f"  - elapsed time: {summary['elapsed_time_seconds']} seconds",
                    f"  - comparison result: {outdir / 'comparison_result.json'}",
                    "",
                ]
            )
        )
    write_run_report(
        outdir,
        title="DPA/DeepMD Compare Models Report",
        summary={
            **summary,
            "status": "success" if not any(row["status"] == "failed" for row in rows) else "completed_with_failures",
            "comparison_json": "comparison_result.json",
            "comparison_summary_json": "comparison_summary.json",
            "comparison_csv": "comparison_summary.csv",
            "comparison_markdown": "comparison_summary.md",
            "comparison_kappa_bar": "comparison_kappa_bar.png",
            "comparison_legacy_kappa_plot": "comparison_kappa.png",
            "comparison_thermal_conductivity": "comparison_thermal_conductivity.png",
            "comparison_phonon_band": "comparison_phonon_band.png",
            "comparison_dos": "comparison_dos.png",
            "comparison_group_velocity": (
                "comparison_group_velocity.png" if group_velocity_plot.get("available") else None
            ),
            "comparison_phonon_lifetime": (
                "comparison_phonon_lifetime.png" if lifetime_plot.get("available") else None
            ),
        },
        commands=commands,
        validation_lines=[],
    )
    return summary


def _model_command_text(
    *,
    input_path: Path,
    outdir: Path,
    backend: str,
    model_path: Path | None = None,
    model_head: str | None = None,
    compute_kappa: bool,
    relax: bool,
    dry_run: bool,
    overwrite: bool,
    relax_cell: bool = True,
    supercell_dim: tuple[int, int, int] | None = None,
    mesh: tuple[int, int, int] | None = None,
    kappa_mesh: tuple[int, int, int] | None = None,
    target_supercell_length: float | None = None,
    displacement: float | None = None,
    fc3_target_supercell_length: float | None = None,
    fc3_supercell_dim: tuple[int, int, int] | None = None,
    fc3_displacement: float | None = None,
    fc3_cutoff_pair_distance: float | None = None,
    fc3_method: str | None = None,
    kappa_method: str | None = None,
    wigner: bool | None = None,
    temperatures: list[float] | None = None,
    max_fc3_displacements: int | None = None,
    fmax: float | None = None,
    max_steps: int | None = None,
    primitive_matrix: str | None = None,
    dos: bool | None = None,
    export_fc2_text: bool | None = None,
    kpath_mode: str | None = None,
    bandpath_with_time_reversal: bool | None = None,
    phonopy_symprec: float | None = None,
    phono3py_symmetrize_fc2: bool | None = None,
    phono3py_symmetrize_fc3: bool | None = None,
    phono3py_symprec: float | None = None,
    phono3py_cutoff_frequency: float | None = None,
    n_structures: int | None = None,
    rattle_std: float | None = None,
    cutoffs: list[float] | tuple[float, ...] | None = None,
    min_dist: float | None = None,
    deepmd_device: str | None = None,
    deepmd_deterministic: bool | None = None,
    deepmd_reuse_calculator: bool | None = None,
    save_force_audit: bool | None = None,
    dpa_safe_mode: bool = False,
) -> str:
    return shlex.join(
        _model_command(
            input_path=input_path,
            outdir=outdir,
            backend=backend,
            model_path=model_path,
            model_head=model_head,
            compute_kappa=compute_kappa,
            relax=relax,
            relax_cell=relax_cell,
            dry_run=dry_run,
            overwrite=overwrite,
            supercell_dim=supercell_dim,
            mesh=mesh,
            kappa_mesh=kappa_mesh,
            target_supercell_length=target_supercell_length,
            displacement=displacement,
            fc3_target_supercell_length=fc3_target_supercell_length,
            fc3_supercell_dim=fc3_supercell_dim,
            fc3_displacement=fc3_displacement,
            fc3_cutoff_pair_distance=fc3_cutoff_pair_distance,
            fc3_method=fc3_method,
            kappa_method=kappa_method,
            wigner=wigner,
            temperatures=temperatures,
            max_fc3_displacements=max_fc3_displacements,
            fmax=fmax,
            max_steps=max_steps,
            primitive_matrix=primitive_matrix,
            dos=dos,
            export_fc2_text=export_fc2_text,
            kpath_mode=kpath_mode,
            bandpath_with_time_reversal=bandpath_with_time_reversal,
            phonopy_symprec=1e-5 if phonopy_symprec is None else phonopy_symprec,
            phono3py_symmetrize_fc2=phono3py_symmetrize_fc2,
            phono3py_symmetrize_fc3=phono3py_symmetrize_fc3,
            phono3py_symprec=1e-5 if phono3py_symprec is None else phono3py_symprec,
            phono3py_cutoff_frequency=1e-4 if phono3py_cutoff_frequency is None else phono3py_cutoff_frequency,
            n_structures=n_structures,
            rattle_std=rattle_std,
            cutoffs=cutoffs,
            min_dist=min_dist,
            deepmd_device=deepmd_device,
            deepmd_deterministic=deepmd_deterministic,
            deepmd_reuse_calculator=deepmd_reuse_calculator,
            save_force_audit=save_force_audit,
            dpa_safe_mode=dpa_safe_mode,
        )
    )


def _model_command(
    *,
    input_path: Path,
    outdir: Path,
    backend: str,
    model_path: Path | None = None,
    model_head: str | None = None,
    compute_kappa: bool,
    relax: bool,
    dry_run: bool,
    overwrite: bool,
    relax_cell: bool = True,
    supercell_dim: tuple[int, int, int] | None = None,
    mesh: tuple[int, int, int] | None = None,
    kappa_mesh: tuple[int, int, int] | None = None,
    target_supercell_length: float | None = None,
    displacement: float | None = None,
    fc3_target_supercell_length: float | None = None,
    fc3_supercell_dim: tuple[int, int, int] | None = None,
    fc3_displacement: float | None = None,
    fc3_cutoff_pair_distance: float | None = None,
    fc3_method: str | None = None,
    kappa_method: str | None = None,
    wigner: bool | None = None,
    temperatures: list[float] | None = None,
    max_fc3_displacements: int | None = None,
    fmax: float | None = None,
    max_steps: int | None = None,
    primitive_matrix: str | None = None,
    dos: bool | None = None,
    export_fc2_text: bool | None = None,
    phono3py_symmetrize_fc2: bool | None = None,
    phono3py_symmetrize_fc3: bool | None = None,
    kpath_mode: str | None = None,
    bandpath_with_time_reversal: bool | None = None,
    phonopy_symprec: float | None = None,
    phono3py_symprec: float | None = None,
    phono3py_cutoff_frequency: float | None = None,
    n_structures: int | None = None,
    rattle_std: float | None = None,
    cutoffs: list[float] | tuple[float, ...] | None = None,
    min_dist: float | None = None,
    deepmd_device: str | None = None,
    deepmd_deterministic: bool | None = None,
    deepmd_reuse_calculator: bool | None = None,
    save_force_audit: bool | None = None,
    dpa_safe_mode: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "phonoflow",
        "single",
        "--input-path",
        str(input_path),
        "--backend",
        backend,
        "--outdir",
        str(outdir),
        "--compute-kappa" if compute_kappa else "--no-compute-kappa",
        "--relax" if relax else "--no-relax",
    ]
    if relax:
        command.append("--relax-cell" if relax_cell else "--no-relax-cell")
    if model_path is not None:
        command.extend(["--model-path", str(model_path)])
    if model_head is not None:
        command.extend(["--deepmd-model-head", str(model_head)])
    if supercell_dim is not None:
        command.append("--supercell-dim")
        command.extend(str(item) for item in supercell_dim)
    else:
        command.extend(["--supercell-dim", "auto"])
    common_q_mesh = mesh if mesh is not None else kappa_mesh if kappa_mesh is not None else None
    command.append("--mesh")
    if common_q_mesh is None:
        command.append("auto")
    else:
        command.extend(str(item) for item in common_q_mesh)
    if dpa_safe_mode:
        command.extend(["--target-supercell-length", "12.0"])
    elif target_supercell_length is not None:
        command.extend(["--target-supercell-length", str(target_supercell_length)])
    if displacement is not None:
        command.extend(["--displacement", str(displacement)])
    if fc3_target_supercell_length is not None:
        command.extend(["--fc3-target-supercell-length", str(fc3_target_supercell_length)])
    if fc3_supercell_dim is not None:
        command.append("--fc3-supercell-dim")
        command.extend(str(item) for item in fc3_supercell_dim)
    if fc3_displacement is not None:
        command.extend(["--fc3-displacement", str(fc3_displacement)])
    if fc3_cutoff_pair_distance is not None:
        command.extend(["--fc3-cutoff-pair-distance", str(fc3_cutoff_pair_distance)])
    effective_fc3_method = str(fc3_method or "finite-displacement")
    if compute_kappa:
        command.extend(["--fc3-method", effective_fc3_method])
        command.extend(["--method", str(kappa_method or "rta")])
        command.extend(["--wigner", "true" if bool(wigner) else "false"])
        for temperature in temperatures if temperatures is not None else [300.0]:
            command.extend(["--temperatures", str(temperature)])
        command.append("--kappa-mesh")
        if common_q_mesh is None:
            command.append("auto")
        else:
            command.extend(str(item) for item in common_q_mesh)
        if effective_fc3_method == "hiphive":
            if n_structures is not None:
                command.extend(["--n-structures", str(n_structures)])
            if rattle_std is not None:
                command.extend(["--rattle-std", str(rattle_std)])
            if cutoffs is not None:
                command.append("--cutoffs")
                command.extend(str(item) for item in cutoffs)
            if min_dist is not None:
                command.extend(["--min-dist", str(min_dist)])
    if max_fc3_displacements is not None:
        command.extend(["--max-fc3-displacements", str(max_fc3_displacements)])
    if dpa_safe_mode:
        command.extend(["--max-supercell-atoms", "256"])
    if fmax is not None:
        command.extend(["--fmax", str(fmax)])
    if max_steps is not None:
        command.extend(["--max-steps", str(max_steps)])
    if primitive_matrix is not None:
        command.extend(["--primitive-matrix", str(primitive_matrix)])
    if kpath_mode is not None:
        command.extend(["--kpath-mode", str(kpath_mode)])
    if bandpath_with_time_reversal is not None:
        command.append("--bandpath-with-time-reversal" if bandpath_with_time_reversal else "--no-bandpath-with-time-reversal")
    if phonopy_symprec is not None:
        command.extend(["--phonopy-symprec", str(phonopy_symprec)])
    if dos is not None:
        command.append("--dos" if dos else "--no-dos")
    if export_fc2_text is not None:
        command.append("--export-fc2-text" if export_fc2_text else "--no-export-fc2-text")
    if compute_kappa:
        if phono3py_symmetrize_fc2 is not None:
            command.append("--phono3py-symmetrize-fc2" if phono3py_symmetrize_fc2 else "--no-phono3py-symmetrize-fc2")
        if phono3py_symmetrize_fc3 is not None:
            command.append("--phono3py-symmetrize-fc3" if phono3py_symmetrize_fc3 else "--no-phono3py-symmetrize-fc3")
        if phono3py_symprec is not None:
            command.extend(["--phono3py-symprec", str(phono3py_symprec)])
        if phono3py_cutoff_frequency is not None:
            command.extend(["--phono3py-cutoff-frequency", str(phono3py_cutoff_frequency)])
    if backend in DPA_BACKEND_ALIASES or backend == "deepmd":
        if deepmd_device is not None:
            command.extend(["--deepmd-device", str(deepmd_device)])
        if deepmd_deterministic is not None:
            command.append("--deepmd-deterministic" if deepmd_deterministic else "--no-deepmd-deterministic")
        if deepmd_reuse_calculator is not None:
            command.append("--deepmd-reuse-calculator" if deepmd_reuse_calculator else "--no-deepmd-reuse-calculator")
        if save_force_audit is not None:
            command.append("--save-force-audit" if save_force_audit else "--no-save-force-audit")
    if dry_run:
        command.append("--dry-run")
    if overwrite:
        command.append("--overwrite")
    return command


def _prepare_shared_relaxed_structure(
    *,
    input_path: Path,
    outdir: Path,
    relax_cell: bool,
    fmax: float | None,
    max_steps: int | None,
    dry_run: bool,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    shared_dir = outdir / "shared_nep89_relax"
    shared_dir.mkdir(parents=True, exist_ok=True)
    relaxed_path = shared_dir / "relaxed.vasp"
    if dry_run:
        shutil.copyfile(input_path, relaxed_path)
        info = {
            "relax": True,
            "relax_enabled": True,
            "relax_backend": "calorine",
            "relax_model": "NEP89",
            "relax_model_path": str(DEFAULT_NEP89_MODEL_PATH),
            "relax_policy": "shared_nep89_pre_relax",
            "relax_cell": bool(relax_cell),
            "relax_mode": "cell" if relax_cell else "positions",
            "relax_converged": None,
            "n_steps": 0,
            "fmax": float(fmax if fmax is not None else 1e-5),
            "max_steps": int(max_steps if max_steps is not None else 2000),
            "notes": "Dry run copied the input structure to represent the shared compare relax target.",
        }
        _write_json(shared_dir / "shared_relax_info.json", info)
        return relaxed_path, info

    atoms = read_structure(input_path)
    config = WorkflowConfig(
        input_path=input_path,
        outdir=shared_dir,
        backend="calorine",
        model_path=DEFAULT_NEP89_MODEL_PATH,
        relax=True,
        relax_cell=relax_cell,
        fmax=fmax if fmax is not None else 1e-5,
        max_steps=max_steps if max_steps is not None else 2000,
        overwrite=overwrite,
    )
    backend = get_backend("calorine", model_path=DEFAULT_NEP89_MODEL_PATH)
    backend.apply_config(config)
    if not backend.check_available():
        raise ConfigError(
            "Compare relaxation requires the shared NEP89/Calorine pre-relaxation backend. "
            "Install Calorine and ensure the bundled NEP89 model is available, or run compare with --no-relax."
        )
    relaxed_atoms, info = relax_structure(atoms, backend, shared_dir, config)
    write_structure(atoms, shared_dir / "input_structure.vasp")
    write_structure(relaxed_atoms, relaxed_path)
    info = {
        **dict(info),
        "relax": True,
        "relax_enabled": True,
        "relax_backend": "calorine",
        "relax_model": "NEP89",
        "relax_model_path": str(DEFAULT_NEP89_MODEL_PATH),
        "relax_policy": "shared_nep89_pre_relax",
        "relax_cell": bool(relax_cell),
        "relax_mode": "cell" if relax_cell else "positions",
        "relaxed_structure_path": str(relaxed_path),
    }
    _write_json(shared_dir / "shared_relax_info.json", info)
    return relaxed_path, info


def _with_shared_relax_metadata(
    data: dict[str, Any],
    *,
    shared_relaxed_structure_path: Path | None,
    shared_relax_info: dict[str, Any],
    relax_cell: bool,
    property_model: str,
    calculation_model: str,
) -> dict[str, Any]:
    updated = dict(data)
    updated.update(
        {
            "relax": True,
            "relax_enabled": True,
            "relax_model": "NEP89",
            "relax_policy": "shared_nep89_pre_relax",
            "relax_cell": bool(relax_cell),
            "relax_mode": "cell" if relax_cell else "positions",
            "relax_backend": "calorine",
            "relax_model_path": str(DEFAULT_NEP89_MODEL_PATH),
            "relaxed_structure_path": str(shared_relaxed_structure_path) if shared_relaxed_structure_path else None,
            "shared_relaxed_structure": True,
            "property_model": property_model,
            "calculation_model": calculation_model,
            "shared_relax_info": dict(shared_relax_info),
        }
    )
    return updated


def _run_isolated_model(command: list[str], outdir: Path, model_name: str) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    env, runtime_env = _isolated_model_env(command, model_name)
    _write_json(outdir / "deepmd_runtime_env.json", runtime_env)
    completed = subprocess.run(command, cwd=Path.cwd(), env=env, text=True, capture_output=True)
    (outdir / "compare_subprocess.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    raw_stderr = completed.stderr or ""
    filtered_stderr, suppressed_lines = _filter_optional_cuda_probe_warnings(
        raw_stderr,
        enabled=canonical_dpa_alias(model_name) == "dpa4neo" and runtime_env.get("deepmd_device") == "cpu",
    )
    if suppressed_lines:
        (outdir / "compare_subprocess.stderr.raw.log").write_text(raw_stderr, encoding="utf-8")
        runtime_env["cuda_probe_warnings_suppressed"] = str(len(suppressed_lines))
        runtime_env["cuda_probe_warning_audit_file"] = "compare_subprocess.stderr.raw.log"
        _write_json(outdir / "deepmd_runtime_env.json", runtime_env)
    (outdir / "compare_subprocess.stderr.log").write_text(filtered_stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise IsolatedModelError(
            completed.returncode,
            completed.stdout or "",
            filtered_stderr,
            runtime_env=runtime_env,
        )
    return {
        "status": "success",
        "outdir": str(outdir),
        "report": _read_result(outdir),
        "runtime_env": runtime_env,
    }


def _isolated_model_env(command: list[str], model_name: str) -> tuple[dict[str, str], dict[str, str]]:
    env = dict(os.environ)
    device = _command_option(command, "--deepmd-device") or ("cpu" if canonical_dpa_alias(model_name) in DPA_MODEL_REGISTRY else "")
    runtime_env: dict[str, str] = {"deepmd_device": device}
    if canonical_dpa_alias(model_name) == "dpa4neo" and device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env["HIP_VISIBLE_DEVICES"] = "-1"
        env["ROCR_VISIBLE_DEVICES"] = "-1"
        env.setdefault("DP_INFER_BATCH_SIZE", str(DPA4_DEFAULT_INFER_BATCH_SIZE))
        runtime_env.update(
            {
                "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
                "HIP_VISIBLE_DEVICES": env["HIP_VISIBLE_DEVICES"],
                "ROCR_VISIBLE_DEVICES": env["ROCR_VISIBLE_DEVICES"],
                "DP_INFER_BATCH_SIZE": env["DP_INFER_BATCH_SIZE"],
            }
        )
    return env, runtime_env


def _command_option(command: list[str], option: str) -> str | None:
    try:
        return str(command[command.index(option) + 1])
    except (ValueError, IndexError):
        return None


def _command_values(command: list[str], option: str) -> list[str]:
    try:
        start = command.index(option) + 1
    except ValueError:
        return []
    values: list[str] = []
    for item in command[start:]:
        if str(item).startswith("--"):
            break
        values.append(str(item))
    return values


def _filter_optional_cuda_probe_warnings(stderr: str, *, enabled: bool) -> tuple[str, list[str]]:
    if not enabled:
        return stderr, []
    kept: list[str] = []
    suppressed: list[str] = []
    for line in stderr.splitlines():
        lowered = line.lower()
        optional_cuda_probe = (
            ("implib-gen:" in lowered or "deepmd-kit:" in lowered)
            and ("libcudart" in lowered or "libcusparse" in lowered)
        )
        if optional_cuda_probe:
            suppressed.append(line)
        else:
            kept.append(line)
    filtered = "\n".join(kept)
    if stderr.endswith("\n") and filtered:
        filtered += "\n"
    return filtered, suppressed


def _resolved_spec_model_path(model_name: str, spec: dict[str, str | Path | None]) -> str | None:
    explicit = spec.get("model_path")
    if explicit:
        return str(Path(explicit))
    if canonical_dpa_alias(model_name) in DPA_MODEL_REGISTRY:
        try:
            return str(resolve_dpa_model_path(model_name, None).model_path)
        except Exception:
            return None
    return None


def _summary_row(
    model_name: str,
    outdir: Path,
    workflow_result: dict[str, Any],
    report: dict[str, Any],
    *,
    command_argv: list[str],
) -> dict[str, Any]:
    thermal = report.get("thermal_conductivity") or {}
    thermal_series = _thermal_series(thermal) or _read_thermal_csv(outdir / "thermal_conductivity.csv")
    kappa = _kappa_summary(thermal)
    if all(kappa.get(component) is None for component in ("kxx", "kyy", "kzz", "kavg")):
        kappa = _kappa_summary({"summary": thermal_series})
    deepmd_applicable = (
        report.get("backend_resolved") == "deepmd"
        or report.get("backend_requested") == "deepmd"
        or canonical_dpa_alias(model_name) in DPA_MODEL_REGISTRY
    )
    row = {
        "model": model_name,
        "display_name": MODEL_DISPLAY_NAMES.get(model_name, model_name.upper()),
        "status": workflow_result.get("status", "success"),
        "return_code": 0,
        "command": shlex.join(command_argv),
        "command_argv": list(command_argv),
        "outdir": str(outdir),
        "backend_requested": report.get("backend_requested"),
        "backend_resolved": report.get("backend_resolved"),
        "backend_alias": report.get("backend_alias"),
        "dpa_model_name": report.get("dpa_model_name"),
        "deepmd_model_head": report.get("deepmd_model_head"),
        "model_path": report.get("model_path"),
        "force_backend": report.get("force_backend", report.get("force_backend_resolved")),
        "force_model_path": report.get("force_model_path", report.get("model_path")),
        "relax_backend": report.get("relax_backend", report.get("relax_backend_resolved")),
        "relax_model_path": report.get("relax_model_path"),
        "relax_enabled": report.get("relax_enabled", report.get("relax")),
        "compare_child_model_name": model_name,
        "model_file_hash": report.get("model_file_hash"),
        "dynamically_stable": report.get("dynamically_stable"),
        "minimum_frequency_THz": report.get("minimum_frequency_THz"),
        "imaginary_mode_count": report.get("imaginary_mode_count"),
        "kappa_method": report.get("kappa_method") or thermal.get("kappa_method") or _command_option(command_argv, "--method"),
        "temperatures": report.get("temperatures") or thermal.get("temperatures"),
        "q_mesh": report.get("q_mesh"),
        "phono3py_mesh": report.get("phono3py_mesh") or report.get("q_mesh"),
        "solver_flags": report.get("solver_flags") or ["--method", _command_option(command_argv, "--method") or "rta"],
        "kxx": kappa.get("kxx"),
        "kyy": kappa.get("kyy"),
        "kzz": kappa.get("kzz"),
        "kavg": kappa.get("kavg"),
        "thermal_series": thermal_series,
        "band_yaml": str(outdir / "band.yaml") if (outdir / "band.yaml").exists() else None,
        "dos_file": str(_find_dos_file(outdir)) if _find_dos_file(outdir) is not None else None,
        "thermal_conductivity_csv": str(outdir / "thermal_conductivity.csv") if (outdir / "thermal_conductivity.csv").exists() else None,
        "deepmd_device": report.get("deepmd_device") or (workflow_result.get("runtime_env") or {}).get("deepmd_device"),
        "deepmd_parameters_applicable": deepmd_applicable,
        "deepmd_deterministic": report.get("deepmd_deterministic") if deepmd_applicable else "not_applicable",
        "deepmd_reuse_calculator": report.get("deepmd_reuse_calculator") if deepmd_applicable else "not_applicable",
        "save_force_audit": report.get("save_force_audit") if deepmd_applicable else "not_applicable",
        "dp_infer_batch_size": _int_or_none(
            report.get("dp_infer_batch_size")
            or (workflow_result.get("runtime_env") or {}).get("DP_INFER_BATCH_SIZE")
        ),
        "failure_category": None,
        "failure_reason": "",
        "error_category": None,
        "error_message": "",
        "timing_breakdown": report.get("timing_breakdown") or {},
    }
    row["plot_data_availability"] = _plot_data_availability(row)
    return row


def _failed_row(
    model_name: str,
    outdir: Path,
    error_message: str,
    *,
    command_argv: list[str],
    backend_requested: str | None,
    model_path: str | None,
    dpa_model_name: str | None = None,
    return_code: int | None = None,
    stderr: str = "",
    runtime_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    relax_enabled = "--relax" in command_argv
    is_dpa = canonical_dpa_alias(model_name) in DPA_MODEL_REGISTRY or backend_requested == "deepmd"
    failure_category = _classify_error(return_code, error_message, stderr)
    failure_reason = _clear_failure_reason(failure_category, return_code, error_message)
    runtime_env = runtime_env or {}
    row = {
        "model": model_name,
        "display_name": MODEL_DISPLAY_NAMES.get(model_name, model_name.upper()),
        "status": "failed",
        "return_code": return_code,
        "command": shlex.join(command_argv) if command_argv else "",
        "command_argv": list(command_argv),
        "outdir": str(outdir),
        "backend_requested": backend_requested,
        "backend_resolved": None,
        "backend_alias": model_name,
        "dpa_model_name": dpa_model_name,
        "deepmd_model_head": _command_option(command_argv, "--deepmd-model-head"),
        "model_path": model_path,
        "force_backend": "deepmd" if is_dpa else backend_requested,
        "force_model_path": model_path,
        "relax_backend": "calorine" if is_dpa and relax_enabled else ("none" if not relax_enabled else backend_requested),
        "relax_model_path": str(DEFAULT_NEP89_MODEL_PATH) if is_dpa and relax_enabled else (model_path if relax_enabled else None),
        "relax_enabled": relax_enabled,
        "compare_child_model_name": model_name,
        "model_file_hash": None,
        "dynamically_stable": None,
        "minimum_frequency_THz": None,
        "imaginary_mode_count": None,
        "kappa_method": _command_option(command_argv, "--method") or "rta",
        "temperatures": _command_values(command_argv, "--temperatures"),
        "q_mesh": _command_values(command_argv, "--mesh"),
        "phono3py_mesh": _command_values(command_argv, "--kappa-mesh") or _command_values(command_argv, "--mesh"),
        "solver_flags": ["--method", _command_option(command_argv, "--method") or "rta"],
        "kxx": None,
        "kyy": None,
        "kzz": None,
        "kavg": None,
        "thermal_series": [],
        "band_yaml": None,
        "dos_file": None,
        "thermal_conductivity_csv": None,
        "deepmd_device": runtime_env.get("deepmd_device") or _command_option(command_argv, "--deepmd-device"),
        "deepmd_parameters_applicable": is_dpa,
        "deepmd_deterministic": (
            "--deepmd-deterministic" in command_argv if is_dpa else "not_applicable"
        ),
        "deepmd_reuse_calculator": (
            "--deepmd-reuse-calculator" in command_argv if is_dpa else "not_applicable"
        ),
        "save_force_audit": (
            "--save-force-audit" in command_argv if is_dpa else "not_applicable"
        ),
        "dp_infer_batch_size": _int_or_none(runtime_env.get("DP_INFER_BATCH_SIZE")),
        "failure_category": failure_category,
        "failure_reason": failure_reason,
        "error_category": failure_category,
        "error_message": failure_reason,
        "raw_error_message": error_message,
    }
    row["plot_data_availability"] = _plot_data_availability(row)
    return row


def _plot_data_availability(row: dict[str, Any]) -> dict[str, str]:
    if row.get("status") != "success":
        return {
            "band": "failed",
            "dos": "failed",
            "thermal": "failed",
            "group_velocity": "failed",
            "phonon_lifetime": "failed",
        }
    outdir = Path(str(row.get("outdir") or ""))
    return {
        "band": "data" if row.get("band_yaml") else "missing",
        "dos": "data" if row.get("dos_file") else "missing",
        "thermal": "data" if row.get("thermal_series") else "missing",
        "group_velocity": "data" if (outdir / "phonon_group_velocity.csv").exists() else "missing",
        "phonon_lifetime": "data" if (outdir / "phonon_lifetime.csv").exists() else "missing",
    }


def _classify_error(return_code: int | None, message: str, stderr: str = "") -> str:
    text = f"{message}\n{stderr}".lower()
    if return_code in {-9, 9, 137} or "out of memory" in text or "oom" in text or "killed" in text:
        return "out_of_memory"
    if "no such option" in text or return_code == 2:
        return "cli_usage"
    if "model" in text and ("not found" in text or "initialize" in text or "compatib" in text):
        return "model_runtime"
    return "runtime_error"


def _clear_failure_reason(category: str, return_code: int | None, message: str) -> str:
    if category == "out_of_memory":
        return (
            f"Model subprocess was terminated with exit code {return_code}; "
            "this is consistent with memory/OOM exhaustion during model force evaluation."
        )
    return message


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None and str(value).strip() else None
    except (TypeError, ValueError):
        return None


def _kappa_summary(thermal: dict[str, Any]) -> dict[str, float | None]:
    for key in ("summary", "temperatures"):
        rows = thermal.get(key)
        if isinstance(rows, dict) and rows:
            row = _select_kappa_row(rows)
            return {
                "kxx": _float_or_none(row.get("kxx")),
                "kyy": _float_or_none(row.get("kyy")),
                "kzz": _float_or_none(row.get("kzz")),
                "kavg": _float_or_none(row.get("kavg", row.get("kappa_trace_over_3"))),
            }
        if isinstance(rows, list) and rows:
            row = _select_kappa_row(rows)
            return {
                "kxx": _float_or_none(row.get("kxx")),
                "kyy": _float_or_none(row.get("kyy")),
                "kzz": _float_or_none(row.get("kzz")),
                "kavg": _float_or_none(row.get("kavg")),
            }
    return {"kxx": None, "kyy": None, "kzz": None, "kavg": None}


def _select_kappa_row(rows: dict[str, Any] | list[Any]) -> dict[str, Any]:
    """Select the 300 K row for comparison bars, falling back to the first row."""

    if isinstance(rows, dict):
        for key, row in rows.items():
            if _temperature_matches_300(key) and isinstance(row, dict):
                return row
        for row in rows.values():
            if isinstance(row, dict) and _temperature_matches_300(row.get("temperature", row.get("T"))):
                return row
        first = next(iter(rows.values()))
        return first if isinstance(first, dict) else {}
    for row in rows:
        if isinstance(row, dict) and _temperature_matches_300(row.get("temperature", row.get("T"))):
            return row
    first = rows[0]
    return first if isinstance(first, dict) else {}


def _temperature_matches_300(value: Any) -> bool:
    try:
        return abs(float(value) - 300.0) < 1.0e-6
    except (TypeError, ValueError):
        return False


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _thermal_series(thermal: dict[str, Any]) -> list[dict[str, float]]:
    """Return per-temperature kappa rows for model comparison plots."""

    rows = thermal.get("summary") or thermal.get("temperatures") or []
    if isinstance(rows, dict):
        iterable = rows.values()
    else:
        iterable = rows if isinstance(rows, list) else []
    series: list[dict[str, float]] = []
    for row in iterable:
        if not isinstance(row, dict):
            continue
        temperature = _float_or_none(row.get("temperature", row.get("T")))
        if temperature is None:
            try:
                temperature = float(next(key for key, value in (rows.items() if isinstance(rows, dict) else []) if value is row))
            except Exception:
                temperature = 300.0
        item = {
            "temperature": temperature,
            "kxx": _float_or_none(row.get("kxx")),
            "kyy": _float_or_none(row.get("kyy")),
            "kzz": _float_or_none(row.get("kzz")),
            "kavg": _float_or_none(row.get("kavg", row.get("kappa_trace_over_3"))),
        }
        series.append({key: value for key, value in item.items() if value is not None})
    return series


def _read_thermal_csv(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            temperature = _float_or_none(row.get("temperature_K", row.get("temperature", row.get("T"))))
            kxx = _float_or_none(row.get("kxx"))
            kyy = _float_or_none(row.get("kyy"))
            kzz = _float_or_none(row.get("kzz"))
            kavg = _float_or_none(row.get("kavg", row.get("kappa_trace_over_3")))
            item = {
                "temperature": temperature,
                "kxx": kxx,
                "kyy": kyy,
                "kzz": kzz,
                "kavg": kavg,
            }
            rows.append({key: value for key, value in item.items() if value is not None})
    return rows


def _read_result(outdir: Path) -> dict[str, Any]:
    path = outdir / "result.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["model", "status", "error_message"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Compare Models Summary",
        "",
        f"- Input: `{summary['input_path']}`",
        f"- Compute kappa: `{summary['compute_kappa']}`",
        f"- Relax: `{summary['relax']}`",
        "",
        "| model | status | backend | stable | min freq THz | kavg W/m-K | error |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in summary["models"]:
        lines.append(
            "| {model} | {status} | {backend} | {stable} | {min_freq} | {kavg} | {error} |".format(
                model=row["model"],
                status=row["status"],
                backend=row.get("backend_alias") or "",
                stable=row.get("dynamically_stable"),
                min_freq=_fmt(row.get("minimum_frequency_THz")),
                kavg=_fmt(row.get("kavg")),
                error=str(row.get("error_message") or "").replace("|", "/"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _kappa_bar_components(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") not in {"success", "dry-run"}:
            continue
        for component in ("kxx", "kyy", "kzz", "kavg"):
            value = _float_or_none(row.get(component))
            if value is None:
                continue
            components.append(
                {
                    "model": row["model"],
                    "component": component,
                    "value": value,
                    "temperature": 300.0,
                    "unit": "W/m-K",
                }
            )
    return components


def _write_kappa_bar_plot(path: Path, components: list[dict[str, Any]]) -> dict[str, Any]:
    model_order = list(dict.fromkeys(str(item["model"]) for item in components))
    component_order = ["kxx", "kyy", "kzz", "kavg"]
    full_model_names = [MODEL_DISPLAY_NAMES.get(model, model.upper()) for model in model_order]
    display_labels = [_short_model_display_name(name) for name in full_model_names]
    fig_width = max(8.5, 3.2 * max(len(model_order), 1))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    if not components:
        _empty_axes(ax, "No finite 300 K kappa components available")
        kind = "missing-data/status"
    else:
        value_map = {
            (str(item["model"]), str(item["component"])): float(item["value"])
            for item in components
        }
        centers = np.arange(len(model_order), dtype=float)
        width = 0.18
        offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
        colors = {
            "kxx": "#176B87",
            "kyy": "#E67E22",
            "kzz": "#7C3AED",
            "kavg": "#2E8B57",
        }
        hatch_by_component = {"kxx": "", "kyy": "//", "kzz": "\\\\", "kavg": "xx"}
        for component, offset in zip(component_order, offsets):
            values = [value_map.get((model, component), np.nan) for model in model_order]
            bars = ax.bar(
                centers + offset,
                values,
                width=width,
                color=colors[component],
                hatch=hatch_by_component[component],
                edgecolor="#374151",
                linewidth=0.6,
                label=component,
            )
            ax.bar_label(
                bars,
                labels=["" if not np.isfinite(value) else f"{value:.2f}" for value in values],
                padding=3,
                fontsize=8,
            )
        ax.set_xticks(centers, display_labels)
        ax.set_ylabel("kappa component at 300 K (W/m-K)")
        ax.set_title("300 K Thermal Conductivity Components")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=0, labelsize=10)
        ax.legend(title="Component", ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01))
        kind = "data"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "path": path.name,
        "kind": kind,
        "bar_count": len(components),
        "successful_model_count": len({str(item["model"]) for item in components}),
        "models": sorted({str(item["model"]) for item in components}),
        "components": sorted({str(item["component"]) for item in components}),
        "display_labels": display_labels,
        "full_model_names": full_model_names,
        "value_labels_enabled": True,
        "value_label_format": ".2f",
        "annotation_text": "",
    }


def _short_model_display_name(full_name: str) -> str:
    name = str(full_name)
    if name == "NEP89":
        return name
    if name.startswith("DPA4-Neo"):
        return "DPA4-Neo"
    if name.startswith("DPA-3.1"):
        return "DPA-3.1"
    if name.startswith("DPA-3.2"):
        return "DPA-3.2"
    if name.startswith("DPA-3.3"):
        return "DPA-3.3"
    stem = Path(name).stem
    return stem if len(stem) <= 18 else f"{stem[:15]}…"


def _write_thermal_plot(path: Path, rows: list[dict[str, Any]], *, dry_run: bool = False) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    available: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    models: list[str] = []
    for row in rows:
        series = [item for item in row.get("thermal_series", []) if item.get("kavg") is not None]
        if row.get("status") != "success" or not series:
            continue
        series = sorted(series, key=lambda item: float(item.get("temperature", 300.0)))
        available.append((row, series))
        models.append(str(row["model"]))
    legend_labels = [MODEL_DISPLAY_NAMES.get(model, model.upper()) for model in models]
    if available and all(len(series) == 1 for _, series in available):
        values = [float(series[0]["kavg"]) for _, series in available]
        ax.bar(
            legend_labels,
            values,
            color=[MODEL_COLORS.get(str(row["model"]), "#6B7280") for row, _ in available],
        )
        temperature = float(available[0][1][0].get("temperature", 300.0))
        ax.set_xlabel("Model")
        ax.set_ylabel("kavg (W/m-K)")
        ax.set_title(f"Thermal Conductivity Comparison at {temperature:g} K")
        ax.grid(alpha=0.25)
        kind = "data"
        mode = "single-temperature-bars"
    elif available:
        for row, series in available:
            model = str(row["model"])
            ax.plot(
                [float(item.get("temperature", 300.0)) for item in series],
                [float(item["kavg"]) for item in series],
                marker="o",
                color=MODEL_COLORS.get(model, "#6B7280"),
                label=MODEL_DISPLAY_NAMES.get(model, model.upper()),
            )
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("kavg (W/m-K)")
        ax.set_title("Thermal Conductivity Comparison")
        ax.grid(alpha=0.25)
        ax.legend()
        kind = "data"
        mode = "temperature-curves"
    else:
        kind = "dry-run/status" if dry_run else "missing-data/status"
        mode = "status"
        _empty_axes(ax, "Dry-run/status only: no thermal conductivity series" if dry_run else "No thermal conductivity series available")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "path": path.name,
        "kind": kind,
        "mode": mode,
        "models": models,
        "legend_labels": legend_labels,
        "model_colors": {model: MODEL_COLORS.get(model, "#6B7280") for model in models},
    }


def _write_group_velocity_comparison(outdir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    combined: list[dict[str, Any]] = []
    models: list[str] = []
    model_diagnostics: list[dict[str, Any]] = []
    for row in rows:
        model = str(row["model"])
        source = Path(str(row["outdir"])) / "phonon_group_velocity.csv"
        if row.get("status") != "success":
            reason = f"{MODEL_DISPLAY_NAMES.get(model, model.upper())} child workflow failed."
            row["group_velocity_availability"] = {"available": False, "reason": reason}
            model_diagnostics.append({"model": model, "available": False, "reason": reason})
            continue
        parsed = _read_comparison_csv_rows(
            source,
            model=model,
            numeric_fields=("frequency_THz", "vg_abs_km_s"),
        )
        if not parsed:
            reason = f"{MODEL_DISPLAY_NAMES.get(model, model.upper())} produced no group velocity CSV data."
            row["group_velocity_availability"] = {"available": False, "reason": reason}
            model_diagnostics.append({"model": model, "available": False, "reason": reason})
            continue
        combined.extend(parsed)
        models.append(model)
        row["group_velocity_availability"] = {
            "available": True,
            "data_file": str(source),
            "n_points": len(parsed),
        }
        model_diagnostics.append({"model": model, "available": True, "n_points": len(parsed)})

    diagnostics_path = outdir / "comparison_group_velocity_diagnostics.json"
    if not combined:
        reason = "No successful compare child produced phonon group velocity data."
        diagnostics = {
            "available": False,
            "reason": reason,
            "models": model_diagnostics,
            "n_points": 0,
        }
        _write_json(diagnostics_path, diagnostics)
        return {
            "available": False,
            "kind": "missing-data/status",
            "reason": reason,
            "diagnostics_path": diagnostics_path.name,
            "models": [],
        }

    csv_path = outdir / "comparison_group_velocity.csv"
    png_path = outdir / "comparison_group_velocity.png"
    _write_combined_csv(csv_path, combined)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for model in models:
        model_rows = [item for item in combined if item["model"] == model]
        ax.scatter(
            [float(item["frequency_THz"]) for item in model_rows],
            [float(item["vg_abs_km_s"]) for item in model_rows],
            s=7,
            alpha=0.38,
            edgecolors="none",
            color=MODEL_COLORS.get(model, "#6B7280"),
            label=MODEL_DISPLAY_NAMES.get(model, model.upper()),
        )
    ax.set_title("Phonon Group Velocity Comparison")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Group velocity (km/s)")
    ax.grid(True, color="#d8e0eb", linewidth=0.55, alpha=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    diagnostics = {
        "available": True,
        "reason": None,
        "models": model_diagnostics,
        "n_points": len(combined),
        "data_file": csv_path.name,
        "plot_file": png_path.name,
        "legend_labels": [MODEL_DISPLAY_NAMES.get(model, model.upper()) for model in models],
    }
    _write_json(diagnostics_path, diagnostics)
    return {
        "available": True,
        "kind": "data",
        "reason": None,
        "path": png_path.name,
        "data_path": csv_path.name,
        "diagnostics_path": diagnostics_path.name,
        "models": models,
        "legend_labels": diagnostics["legend_labels"],
        "model_colors": {model: MODEL_COLORS.get(model, "#6B7280") for model in models},
        "n_points": len(combined),
    }


def _write_lifetime_comparison(
    outdir: Path,
    rows: list[dict[str, Any]],
    *,
    compute_kappa: bool,
) -> dict[str, Any]:
    diagnostics_path = outdir / "comparison_phonon_lifetime_diagnostics.json"
    if not compute_kappa:
        for row in rows:
            row["lifetime_availability"] = {
                "available": False,
                "reason": LIFETIME_REQUIRES_THERMAL_REASON,
            }
        diagnostics = {
            "available": False,
            "reason": LIFETIME_REQUIRES_THERMAL_REASON,
            "models": [
                {
                    "model": str(row["model"]),
                    "available": False,
                    "reason": LIFETIME_REQUIRES_THERMAL_REASON,
                }
                for row in rows
            ],
            "n_points": 0,
        }
        _write_json(diagnostics_path, diagnostics)
        return {
            "available": False,
            "kind": "not-requested",
            "reason": LIFETIME_REQUIRES_THERMAL_REASON,
            "diagnostics_path": diagnostics_path.name,
            "models": [],
        }

    combined: list[dict[str, Any]] = []
    models: list[str] = []
    model_diagnostics: list[dict[str, Any]] = []
    for row in rows:
        model = str(row["model"])
        source = Path(str(row["outdir"])) / "phonon_lifetime.csv"
        if row.get("status") != "success":
            reason = f"{MODEL_DISPLAY_NAMES.get(model, model.upper())} child workflow failed."
            row["lifetime_availability"] = {"available": False, "reason": reason}
            model_diagnostics.append({"model": model, "available": False, "reason": reason})
            continue
        parsed = _read_comparison_csv_rows(
            source,
            model=model,
            numeric_fields=("frequency_THz", "lifetime_ps"),
            positive_fields=("lifetime_ps",),
        )
        if not parsed:
            reason = f"{MODEL_DISPLAY_NAMES.get(model, model.upper())} produced no phonon lifetime data."
            row["lifetime_availability"] = {"available": False, "reason": reason}
            model_diagnostics.append({"model": model, "available": False, "reason": reason})
            continue
        combined.extend(parsed)
        models.append(model)
        row["lifetime_availability"] = {
            "available": True,
            "data_file": str(source),
            "n_points": len(parsed),
        }
        model_diagnostics.append({"model": model, "available": True, "n_points": len(parsed)})

    if not combined:
        reason = "No successful compare child produced phonon lifetime data."
        diagnostics = {
            "available": False,
            "reason": reason,
            "models": model_diagnostics,
            "n_points": 0,
        }
        _write_json(diagnostics_path, diagnostics)
        return {
            "available": False,
            "kind": "missing-data/status",
            "reason": reason,
            "diagnostics_path": diagnostics_path.name,
            "models": [],
        }

    csv_path = outdir / "comparison_phonon_lifetime.csv"
    png_path = outdir / "comparison_phonon_lifetime.png"
    _write_combined_csv(csv_path, combined)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for model in models:
        model_rows = [item for item in combined if item["model"] == model]
        ax.scatter(
            [float(item["frequency_THz"]) for item in model_rows],
            [float(item["lifetime_ps"]) for item in model_rows],
            s=8,
            alpha=0.42,
            edgecolors="none",
            color=MODEL_COLORS.get(model, "#6B7280"),
            label=MODEL_DISPLAY_NAMES.get(model, model.upper()),
        )
    ax.set_title("Phonon Lifetime Comparison")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Phonon lifetime (ps, log scale)")
    ax.set_yscale("log")
    ax.grid(True, color="#d8e0eb", linewidth=0.55, alpha=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    diagnostics = {
        "available": True,
        "reason": None,
        "models": model_diagnostics,
        "n_points": len(combined),
        "data_file": csv_path.name,
        "plot_file": png_path.name,
        "legend_labels": [MODEL_DISPLAY_NAMES.get(model, model.upper()) for model in models],
    }
    _write_json(diagnostics_path, diagnostics)
    return {
        "available": True,
        "kind": "data",
        "reason": None,
        "path": png_path.name,
        "data_path": csv_path.name,
        "diagnostics_path": diagnostics_path.name,
        "models": models,
        "legend_labels": diagnostics["legend_labels"],
        "model_colors": {model: MODEL_COLORS.get(model, "#6B7280") for model in models},
        "n_points": len(combined),
    }


def _read_comparison_csv_rows(
    path: Path,
    *,
    model: str,
    numeric_fields: tuple[str, ...],
    positive_fields: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    parsed: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source_row in csv.DictReader(handle):
            numeric: dict[str, float] = {}
            valid = True
            for field in numeric_fields:
                value = _float_or_none(source_row.get(field))
                if value is None or (field in positive_fields and value <= 0):
                    valid = False
                    break
                numeric[field] = value
            if not valid:
                continue
            parsed.append(
                {
                    "model": model,
                    "display_name": MODEL_DISPLAY_NAMES.get(model, model.upper()),
                    **source_row,
                    **numeric,
                }
            )
    return parsed


def _write_combined_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_band_plot(path: Path, rows: list[dict[str, Any]], *, dry_run: bool = False) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    plotted = False
    models: list[str] = []
    warnings: list[str] = []
    legend_labels: list[str] = []
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    all_frequencies: list[float] = []
    for row in rows:
        if row.get("status") != "success":
            continue
        model = str(row["model"])
        try:
            band_data = load_band_yaml_segments(Path(str(row["outdir"])) / "band.yaml")
        except Exception:
            warnings.append(f"{model}: band.yaml missing or unparsed")
            continue
        if not tick_positions:
            tick_positions = [float(value) for value in band_data.tick_positions]
            tick_labels = [str(value) for value in band_data.tick_labels]
        else:
            if [str(value) for value in band_data.tick_labels] != tick_labels:
                warnings.append(f"{model}: path labels differ from first successful model")
            if len(band_data.tick_positions) != len(tick_positions) or any(
                abs(float(left) - float(right)) > 1.0e-6
                for left, right in zip(band_data.tick_positions, tick_positions)
            ):
                warnings.append(f"{model}: path distances differ from first successful model")
        label_used = False
        for segment in band_data.segments:
            for branch_index in range(segment.n_branches):
                label = MODEL_DISPLAY_NAMES.get(model, model.upper()) if not label_used else None
                ax.plot(
                    segment.distances,
                    segment.frequencies[:, branch_index],
                    color=MODEL_COLORS.get(model, "#6B7280"),
                    linewidth=1.05,
                    alpha=0.85,
                    label=label,
                )
                label_used = True
                all_frequencies.extend(float(value) for value in segment.frequencies[:, branch_index])
        plotted = True
        models.append(model)
        legend_labels.append(MODEL_DISPLAY_NAMES.get(model, model.upper()))
    if plotted:
        for tick in tick_positions:
            ax.axvline(tick, color="#d8e0eb", linewidth=0.7, zorder=0)
        ax.axhline(0.0, color="#667085", linestyle="--", linewidth=0.8)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.tick_params(axis="x", which="both", length=0)
        ax.set_xlabel("Wave vector")
        ax.set_ylabel("Frequency (THz)")
        ax.set_title("Phonon Dispersion Comparison")
        ax.grid(True, axis="y", color="#e6edf5", linewidth=0.55, alpha=0.8)
        _set_frequency_limits(ax, all_frequencies)
        ax.legend()
        kind = "data"
    else:
        kind = "dry-run/status" if dry_run else "missing-data/status"
        _write_status_axes(ax, rows, "Dry-run/status only: phonon band" if dry_run else "No parsed phonon band data")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "path": path.name,
        "kind": kind,
        "models": models,
        "warnings": warnings,
        "legend_labels": legend_labels,
        "legend_entry_count": len(legend_labels),
        "model_colors": {model: MODEL_COLORS.get(model, "#6B7280") for model in models},
        "tick_positions": tick_positions,
        "tick_labels": tick_labels,
    }


def _write_dos_plot(path: Path, rows: list[dict[str, Any]], *, dry_run: bool = False) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    plotted = False
    models: list[str] = []
    warnings: list[str] = []
    legend_labels: list[str] = []
    for row in rows:
        if row.get("status") != "success":
            continue
        model = str(row["model"])
        dos_file = _find_dos_file(Path(str(row["outdir"])))
        if dos_file is None:
            warnings.append(f"{model}: DOS file missing")
            continue
        series = _read_dos_dat(dos_file)
        if not series:
            warnings.append(f"{model}: DOS file unparsed")
            continue
        ax.plot(
            [item[0] for item in series],
            [item[1] for item in series],
            color=MODEL_COLORS.get(model, "#6B7280"),
            linewidth=1.2,
            label=MODEL_DISPLAY_NAMES.get(model, model.upper()),
        )
        plotted = True
        models.append(model)
        legend_labels.append(MODEL_DISPLAY_NAMES.get(model, model.upper()))
    if plotted:
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel("DOS")
        ax.set_title("Phonon DOS Comparison")
        ax.axvline(0.0, color="#667085", linestyle="--", linewidth=0.8)
        ax.grid(alpha=0.25)
        ax.legend()
        kind = "data"
    else:
        kind = "dry-run/status" if dry_run else "missing-data/status"
        _write_status_axes(ax, rows, "Dry-run/status only: DOS" if dry_run else "No parsed DOS data")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "path": path.name,
        "kind": kind,
        "models": models,
        "warnings": warnings,
        "legend_labels": legend_labels,
        "legend_entry_count": len(legend_labels),
        "model_colors": {model: MODEL_COLORS.get(model, "#6B7280") for model in models},
    }


def _write_overlay_placeholder(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = [str(row["model"]) for row in rows]
    statuses = [1.0 if row.get("status") == "success" else 0.0 for row in rows]
    if labels:
        ax.bar(labels, statuses, color=["#4C78A8" if value else "#C44E52" for value in statuses])
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("available")
        ax.set_title(title)
        ax.set_yticks([0, 1], ["missing/failed", "available"])
        ax.grid(axis="y", alpha=0.25)
    else:
        _empty_axes(ax, "No model outputs available")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_status_axes(ax: Any, rows: list[dict[str, Any]], title: str) -> None:
    labels = [str(row["model"]) for row in rows]
    statuses = [1.0 if row.get("status") == "success" else 0.0 for row in rows]
    if labels:
        ax.bar(labels, statuses, color=["#4C78A8" if value else "#C44E52" for value in statuses])
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("status")
        ax.set_title(title)
        ax.set_yticks([0, 1], ["missing/failed", "available"])
        ax.grid(axis="y", alpha=0.25)
    else:
        _empty_axes(ax, title)


def _read_band_yaml(path: Path) -> list[dict[str, list[float]]]:
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    phonons = data.get("phonon")
    if not isinstance(phonons, list):
        return []
    distances: list[float] = []
    per_q_frequencies: list[list[float]] = []
    for index, point in enumerate(phonons):
        if not isinstance(point, dict):
            continue
        distance = _float_or_none(point.get("distance"))
        distances.append(float(distance if distance is not None else index))
        bands = point.get("band") or []
        frequencies: list[float] = []
        if isinstance(bands, list):
            for band in bands:
                if isinstance(band, dict):
                    frequency = _float_or_none(band.get("frequency"))
                    if frequency is not None:
                        frequencies.append(frequency)
        per_q_frequencies.append(frequencies)
    if not distances or not per_q_frequencies:
        return []
    n_branches = min((len(items) for items in per_q_frequencies if items), default=0)
    series: list[dict[str, list[float]]] = []
    for branch_index in range(n_branches):
        frequency = [items[branch_index] for items in per_q_frequencies if len(items) > branch_index]
        distance = [distances[i] for i, items in enumerate(per_q_frequencies) if len(items) > branch_index]
        if distance and frequency:
            series.append({"distance": distance, "frequency": frequency})
    return series


def _find_dos_file(outdir: Path) -> Path | None:
    for name in ("phonon_dos.dat", "total_dos.dat", "projected_dos.dat"):
        path = outdir / name
        if path.exists():
            return path
    return None


def _read_dos_dat(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < 2:
            continue
        frequency = _float_or_none(parts[0])
        dos = _float_or_none(parts[1])
        if frequency is not None and dos is not None:
            rows.append((frequency, dos))
    return rows


def _empty_axes(ax: Any, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(text)


def _set_frequency_limits(ax: Any, frequencies: list[float]) -> None:
    finite = [float(value) for value in frequencies if math.isfinite(float(value))]
    if not finite:
        return
    minimum = min(finite)
    maximum = max(finite)
    if abs(minimum) < 1.0e-6:
        minimum = 0.0
    span = max(maximum - minimum, 1.0)
    lower = minimum - 0.08 * span
    upper = maximum + 0.08 * span
    if minimum >= -1.0e-6:
        lower = min(lower, -0.05 * max(maximum, 1.0))
    ax.set_ylim(lower, upper)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
