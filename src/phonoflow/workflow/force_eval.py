"""Force evaluation helpers."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from phonoflow_scheduler.config import ForceScheduleConfig
from phonoflow_scheduler.force_tasks import ForceBatchResult, ForceResult, ForceTask, resolve_force_task_payload
from phonoflow_scheduler.process_pool import evaluate_force_tasks
from phonoflow_scheduler.worker_queue import evaluate_force_tasks_with_worker_queue
from phonopy.structure.atoms import PhonopyAtoms

from phonoflow.calculators import get_backend
from phonoflow.calculators.base import CalculatorBackend
from phonoflow.config import WorkflowConfig
from phonoflow.workflow.force_audit import build_force_audit_record, write_force_audit_files
from phonoflow.workflow.displace import phonopy_atoms_to_ase_atoms


@dataclass
class ForceEvaluationResult:
    """Forces and scheduling metadata for a displacement force loop."""

    forces: list[np.ndarray]
    metadata: dict[str, Any]
    warnings: list[str]
    audit_records: list[dict[str, Any]]


def evaluate_forces_for_displacements(
    displaced_atoms_list: list[Any],
    *,
    backend: CalculatorBackend | None = None,
    backend_name: str | None = None,
    model_path: Path | str | None = None,
    config: WorkflowConfig | None = None,
    force_workers: int | None = None,
    force_parallel_backend: str | None = None,
    deepmd_device: str | None = None,
    log: Callable[[str], None] | None = None,
    audit_outdir: Path | None = None,
    audit_label: str = "fc2",
    base_atoms: Any | None = None,
) -> ForceEvaluationResult:
    """Evaluate forces for ASE atoms while preserving displacement order."""

    started = time.perf_counter()
    warnings: list[str] = []
    requested_workers = _clean_workers(force_workers if force_workers is not None else getattr(config, "force_workers", 1))
    requested_backend = str(force_parallel_backend or getattr(config, "force_parallel_backend", "serial")).lower()
    device = str(deepmd_device or getattr(config, "deepmd_device", "cpu")).lower()
    effective_workers = requested_workers
    if requested_backend == "direct":
        effective_backend = "direct"
        effective_workers = 1
    elif requested_backend in {"process", "worker_queue"} and requested_workers > 1:
        effective_backend = requested_backend
    else:
        effective_backend = "serial"
    if device == "cuda" and requested_workers > 1:
        warnings.append("CUDA device requested; force parallelism downgraded to force_workers=1 to avoid CUDA context conflicts.")
        effective_workers = 1
        effective_backend = "serial"

    backend_label = backend_name or getattr(backend, "name", None)
    if backend_label is None:
        raise ValueError("backend or backend_name is required for force evaluation.")
    total = len(displaced_atoms_list)
    _log(log, f"{audit_label.upper()} force evaluation: n_displacements={total}, force_workers={effective_workers}, backend={effective_backend}")

    factory = _BackendFactory(
        backend_name=str(backend_label),
        model_path=str(model_path) if model_path is not None else None,
        config_payload=config.to_dict() if config is not None else None,
        existing_backend=backend if effective_backend in {"direct", "serial"} else None,
    )
    tasks = [ForceTask(index=index, payload=atoms, label=audit_label) for index, atoms in enumerate(displaced_atoms_list)]
    schedule_config = ForceScheduleConfig(
        force_workers=effective_workers,
        force_parallel_backend=effective_backend,
        deepmd_torch_threads=getattr(config, "deepmd_torch_threads", None) or 1,
        max_concurrent_jobs=getattr(config, "max_concurrent_jobs", 1),
        batch_workers=getattr(config, "batch_workers", 1),
        chunk_size=getattr(config, "force_chunk_size", None),
        max_pending_tasks=getattr(config, "force_max_pending_tasks", None),
        backend_name=str(backend_label),
        base_payload=base_atoms,
    )
    if effective_backend == "direct":
        scheduler_result = _evaluate_direct_force_tasks(
            tasks,
            calculator_factory=factory,
            base_payload=base_atoms,
            log=log,
        )
    else:
        evaluate_scheduler = evaluate_force_tasks_with_worker_queue if effective_backend == "worker_queue" else evaluate_force_tasks
        scheduler_result = evaluate_scheduler(
            tasks,
            calculator_factory=factory,
            calculate_task=_calculate_force_task,
            schedule_config=schedule_config,
            log=log,
        )
    ordered_results = scheduler_result.ordered_results()
    forces = [result.forces for result in ordered_results]
    audit_records = [result.audit_record or {} for result in ordered_results]

    wall_time = max(0.0, time.perf_counter() - started)
    dps = float(total / wall_time) if wall_time > 0 else None
    metadata = {
        **scheduler_result.metadata,
        "n_displacements": total,
        "requested_force_workers": requested_workers,
        "effective_force_workers": effective_workers,
        "force_parallel_backend": effective_backend,
        "wall_time": wall_time,
        "displacements_per_second": dps,
        "deepmd_torch_threads": getattr(config, "deepmd_torch_threads", None),
        "deepmd_deterministic": getattr(config, "deepmd_deterministic", None),
        "effective_cpu_parallelism": effective_workers * int(getattr(config, "deepmd_torch_threads", None) or 1),
    }
    dps_text = f"{dps:.6f}" if dps is not None else "unavailable"
    _log(
        log,
        f"{audit_label.upper()} force evaluation completed: wall_time={wall_time:.3f}s, "
        f"displacements_per_second={dps_text}",
    )
    if audit_outdir is not None:
        write_force_audit_files(audit_outdir, audit_label, audit_records, np.asarray(forces, dtype=float))
    return ForceEvaluationResult(
        forces=forces,
        metadata=metadata,
        warnings=[*warnings, *scheduler_result.warnings],
        audit_records=audit_records,
    )


def _evaluate_direct_force_tasks(
    tasks: list[ForceTask],
    *,
    calculator_factory: Callable[[], CalculatorBackend],
    base_payload: Any | None,
    log: Callable[[str], None] | None,
) -> ForceBatchResult:
    started = time.perf_counter()
    calculator = calculator_factory()
    results: list[ForceResult] = []
    total = len(tasks)
    label = tasks[0].label.upper() if tasks else "FORCE"
    for task in tasks:
        _log(log, f"Evaluating {label} displaced supercell {task.index + 1}/{total}")
        payload = resolve_force_task_payload(task.payload, base_payload)
        results.append(_calculate_force_task(calculator, ForceTask(index=task.index, payload=payload, label=task.label)))
    wall_time = max(0.0, time.perf_counter() - started)
    return ForceBatchResult(
        results=results,
        metadata={
            "n_displacements": total,
            "force_parallel_backend": "direct",
            "requested_force_workers": 1,
            "effective_force_workers": 1,
            "chunk_size": None,
            "max_pending_tasks": None,
            "max_pending_observed": 0,
            "submitted_chunks": 0,
            "wall_time": wall_time,
            "displacements_per_second": float(total / wall_time) if wall_time > 0 else None,
            "submit_time": 0.0,
            "force_evaluation_wall_time": wall_time,
            "result_collection_time": 0.0,
            "result_ordering_time": 0.0,
            "direct_no_scheduler": True,
        },
        warnings=[],
    )


def evaluate_forces(
    displaced_supercells: list[PhonopyAtoms],
    backend: CalculatorBackend,
    model_path: Path | None,
    log: Callable[[str], None] | None = None,
    audit_outdir: Path | None = None,
    audit_label: str = "fc2",
) -> list[np.ndarray]:
    """Evaluate forces for displaced Phonopy supercells."""

    if hasattr(backend, "set_model_path"):
        backend.set_model_path(model_path)

    atoms_list = [phonopy_atoms_to_ase_atoms(supercell) for supercell in displaced_supercells]
    result = evaluate_forces_for_displacements(
        atoms_list,
        backend=backend,
        model_path=model_path,
        force_workers=1,
        force_parallel_backend="serial",
        log=log,
        audit_outdir=audit_outdir,
        audit_label=audit_label,
    )
    return result.forces


@dataclass
class _BackendFactory:
    backend_name: str
    model_path: str | None
    config_payload: dict[str, Any] | None
    existing_backend: CalculatorBackend | None = None

    def __call__(self) -> CalculatorBackend:
        if self.existing_backend is not None:
            backend = self.existing_backend
        else:
            backend = get_backend(self.backend_name, model_path=Path(self.model_path) if self.model_path else None)
            if self.config_payload is not None:
                backend.apply_config(WorkflowConfig(**self.config_payload))
        if hasattr(backend, "set_model_path"):
            backend.set_model_path(Path(self.model_path) if self.model_path else None)
        return backend


def _calculate_force_task(backend: CalculatorBackend, task: ForceTask) -> ForceResult:
    try:
        result = _calculate_one(backend, task.payload, task.index, audit_label=task.label)
        return ForceResult(
            index=task.index,
            forces=result["forces"],
            audit_record=result["audit_record"],
        )
    except Exception as exc:
        details = traceback.format_exc()
        raise RuntimeError(f"displacement index {task.index}; error={exc}\n{details}") from exc


def _calculate_one(backend: CalculatorBackend, atoms: Any, index: int, *, audit_label: str) -> dict[str, Any]:
    if atoms.info.get("phonoflow_dummy_force_error"):
        raise RuntimeError(str(atoms.info["phonoflow_dummy_force_error"]))
    result: dict[str, Any] = backend.calculate_energy_forces(atoms)
    force_array = np.asarray(result["forces"], dtype=float)
    return {
        "forces": force_array,
        "audit_record": build_force_audit_record(
            index,
            atoms,
            energy=result.get("energy"),
            forces=force_array,
        ),
    }


def _clean_workers(value: object) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        parsed = 1
    return max(1, parsed)


def _log(log: Callable[[str], None] | None, message: str) -> None:
    if log is not None:
        log(message)
