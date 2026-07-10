"""Chunked serial/process force task scheduler."""

from __future__ import annotations

import math
import time
import traceback
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import replace
from typing import Any

import numpy as np

from phonoflow_scheduler.config import ForceScheduleConfig, clean_positive_int
from phonoflow_scheduler.force_tasks import ForceBatchResult, ForceResult, ForceTask, resolve_force_task_payload


CalculatorFactory = Callable[[], Any]
TaskEvaluator = Callable[[Any, ForceTask], ForceResult]

_WORKER_CALCULATOR: Any = None
_WORKER_EVALUATOR: TaskEvaluator | None = None
_WORKER_BASE_PAYLOAD: Any = None

_SINGLE_TASK_CHUNK_BACKENDS = {
    "calorine",
    "nep",
    "nep89",
    "cpunep",
    "deepmd",
    "dpa",
    "dpa3",
    "dpa4",
    "dpa31",
    "dpa32",
    "dpa33",
    "dpa4neo",
    "custom_deepmd",
}


def choose_chunk_size(
    *,
    num_tasks: int,
    workers: int,
    configured_chunk_size: int | None = None,
    backend_name: str | None = None,
) -> int:
    """Choose a conservative chunk size for large displacement queues."""

    if configured_chunk_size is not None:
        return clean_positive_int(configured_chunk_size)
    normalized_backend = _normalize_backend_name(backend_name)
    if normalized_backend in _SINGLE_TASK_CHUNK_BACKENDS:
        return 1
    tasks = clean_positive_int(num_tasks)
    worker_count = clean_positive_int(workers)
    return max(1, min(16, math.ceil(tasks / max(1, worker_count * 8))))


def choose_max_pending_tasks(*, workers: int, configured_max_pending_tasks: int | None = None) -> int:
    """Choose max pending futures/chunks to keep in the executor queue."""

    if configured_max_pending_tasks is not None:
        return clean_positive_int(configured_max_pending_tasks)
    return max(1, clean_positive_int(workers) * 4)


def evaluate_force_tasks(
    tasks: Iterable[ForceTask],
    *,
    calculator_factory: CalculatorFactory,
    calculate_task: TaskEvaluator | None = None,
    schedule_config: ForceScheduleConfig | None = None,
    log: Callable[[str], None] | None = None,
) -> ForceBatchResult:
    """Evaluate force tasks using serial or bounded chunked process scheduling."""

    task_list = list(tasks)
    config = schedule_config or ForceScheduleConfig()
    chunk_size = choose_chunk_size(
        num_tasks=len(task_list) or 1,
        workers=config.force_workers,
        configured_chunk_size=config.chunk_size,
        backend_name=config.backend_name,
    )
    chunks = _chunk_tasks(task_list, chunk_size)
    return evaluate_force_chunks(
        chunks,
        calculator_factory=calculator_factory,
        calculate_task=calculate_task,
        schedule_config=ForceScheduleConfig(
            force_workers=config.force_workers,
            force_parallel_backend=config.force_parallel_backend,
            deepmd_torch_threads=config.deepmd_torch_threads,
            max_concurrent_jobs=config.max_concurrent_jobs,
            batch_workers=config.batch_workers,
            chunk_size=chunk_size,
            max_pending_tasks=config.max_pending_tasks,
            calculator_initializer=config.calculator_initializer,
            backend_name=config.backend_name,
            base_payload=config.base_payload,
        ),
        log=log,
    )


def evaluate_force_chunks(
    chunks: list[list[ForceTask]],
    *,
    calculator_factory: CalculatorFactory,
    calculate_task: TaskEvaluator | None = None,
    schedule_config: ForceScheduleConfig | None = None,
    log: Callable[[str], None] | None = None,
) -> ForceBatchResult:
    """Evaluate already chunked force tasks."""

    config = schedule_config or ForceScheduleConfig()
    evaluator = calculate_task or _default_calculate_task
    started = time.perf_counter()
    chunk_size = config.chunk_size or (len(chunks[0]) if chunks else 1)
    max_pending = choose_max_pending_tasks(
        workers=config.force_workers,
        configured_max_pending_tasks=config.max_pending_tasks,
    )
    backend = "process" if config.force_parallel_backend == "process" and config.force_workers > 1 else "serial"
    if backend == "serial":
        results, profile = _evaluate_serial_chunks(
            chunks,
            calculator_factory=calculator_factory,
            calculate_task=evaluator,
            base_payload=config.base_payload,
            log=log,
        )
        max_pending_observed = 1 if chunks else 0
    else:
        results, profile, max_pending_observed = _evaluate_process_chunks(
            chunks,
            calculator_factory=calculator_factory,
            calculate_task=evaluator,
            base_payload=config.base_payload,
            max_workers=config.force_workers,
            max_pending_chunks=max_pending,
            log=log,
        )

    results.sort(key=lambda result: result.index)
    wall_time = max(0.0, time.perf_counter() - started)
    total_tasks = sum(len(chunk) for chunk in chunks)
    metadata = {
        "n_displacements": total_tasks,
        "force_parallel_backend": backend,
        "requested_force_workers": clean_positive_int(config.force_workers),
        "effective_force_workers": clean_positive_int(config.force_workers) if backend == "process" else 1,
        "chunk_size": chunk_size,
        "max_pending_tasks": max_pending,
        "max_pending_observed": max_pending_observed,
        "submitted_chunks": len(chunks),
        "wall_time": wall_time,
        "displacements_per_second": float(total_tasks / wall_time) if wall_time > 0 else None,
        **profile,
    }
    return ForceBatchResult(results=results, metadata=metadata, warnings=[])


def _evaluate_serial_chunks(
    chunks: list[list[ForceTask]],
    *,
    calculator_factory: CalculatorFactory,
    calculate_task: TaskEvaluator,
    base_payload: Any | None,
    log: Callable[[str], None] | None,
) -> tuple[list[ForceResult], dict[str, float]]:
    submit_started = time.perf_counter()
    calculator = calculator_factory()
    submit_seconds = max(0.0, time.perf_counter() - submit_started)
    force_started = time.perf_counter()
    results: list[ForceResult] = []
    for chunk in chunks:
        for task in chunk:
            _log(log, f"Evaluating displaced structure {task.index + 1}")
            results.append(calculate_task(calculator, _resolve_task(task, base_payload)))
    force_seconds = max(0.0, time.perf_counter() - force_started)
    return results, {
        "submit_time": submit_seconds,
        "force_evaluation_wall_time": force_seconds,
        "result_collection_time": 0.0,
        "result_ordering_time": 0.0,
    }


def _evaluate_process_chunks(
    chunks: list[list[ForceTask]],
    *,
    calculator_factory: CalculatorFactory,
    calculate_task: TaskEvaluator,
    base_payload: Any | None,
    max_workers: int,
    max_pending_chunks: int,
    log: Callable[[str], None] | None,
) -> tuple[list[ForceResult], dict[str, float], int]:
    submit_seconds = 0.0
    collection_seconds = 0.0
    results: list[ForceResult] = []
    chunk_iter = iter(enumerate(chunks))
    pending: dict[Future[list[ForceResult]], int] = {}
    max_pending_observed = 0
    force_started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(calculator_factory, calculate_task, base_payload),
    ) as executor:
        submit_started = time.perf_counter()
        _fill_pending(executor, pending, chunk_iter, max_pending_chunks)
        submit_seconds += max(0.0, time.perf_counter() - submit_started)
        max_pending_observed = max(max_pending_observed, len(pending))
        while pending:
            collect_started = time.perf_counter()
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            collection_seconds += max(0.0, time.perf_counter() - collect_started)
            for future in done:
                chunk_number = pending.pop(future)
                try:
                    chunk_results = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Force chunk {chunk_number} failed: {exc}") from exc
                results.extend(chunk_results)
                _log(log, f"Evaluated force chunk {chunk_number + 1}")
            submit_started = time.perf_counter()
            _fill_pending(executor, pending, chunk_iter, max_pending_chunks)
            submit_seconds += max(0.0, time.perf_counter() - submit_started)
            max_pending_observed = max(max_pending_observed, len(pending))
    force_seconds = max(0.0, time.perf_counter() - force_started)
    return results, {
        "submit_time": submit_seconds,
        "force_evaluation_wall_time": force_seconds,
        "result_collection_time": collection_seconds,
        "result_ordering_time": 0.0,
    }, max_pending_observed


def _fill_pending(
    executor: ProcessPoolExecutor,
    pending: dict[Future[list[ForceResult]], int],
    chunk_iter: Any,
    max_pending_chunks: int,
) -> None:
    while len(pending) < max_pending_chunks:
        try:
            chunk_number, chunk = next(chunk_iter)
        except StopIteration:
            return
        pending[executor.submit(_worker_calculate_chunk, chunk)] = chunk_number


def _init_worker(calculator_factory: CalculatorFactory, calculate_task: TaskEvaluator, base_payload: Any | None) -> None:
    global _WORKER_CALCULATOR, _WORKER_EVALUATOR, _WORKER_BASE_PAYLOAD
    _WORKER_CALCULATOR = calculator_factory()
    _WORKER_EVALUATOR = calculate_task
    _WORKER_BASE_PAYLOAD = base_payload


def _worker_calculate_chunk(chunk: list[ForceTask]) -> list[ForceResult]:
    if _WORKER_CALCULATOR is None or _WORKER_EVALUATOR is None:
        raise RuntimeError("worker calculator was not initialized")
    results: list[ForceResult] = []
    for task in chunk:
        try:
            results.append(_WORKER_EVALUATOR(_WORKER_CALCULATOR, _resolve_task(task, _WORKER_BASE_PAYLOAD)))
        except Exception as exc:
            details = traceback.format_exc()
            raise RuntimeError(f"displacement index {task.index} failed: {exc}\n{details}") from exc
    return results


def _default_calculate_task(calculator: Any, task: ForceTask) -> ForceResult:
    result = calculator.calculate_energy_forces(task.payload)
    forces = np.asarray(result["forces"], dtype=float)
    return ForceResult(
        index=task.index,
        forces=forces,
        energy=result.get("energy"),
    )


def _chunk_tasks(tasks: list[ForceTask], chunk_size: int) -> list[list[ForceTask]]:
    size = clean_positive_int(chunk_size)
    return [tasks[index : index + size] for index in range(0, len(tasks), size)]


def _resolve_task(task: ForceTask, base_payload: Any | None) -> ForceTask:
    payload = resolve_force_task_payload(task.payload, base_payload)
    if payload is task.payload:
        return task
    return replace(task, payload=payload)


def _normalize_backend_name(backend_name: str | None) -> str | None:
    if backend_name is None:
        return None
    return str(backend_name).strip().lower()


def _log(log: Callable[[str], None] | None, message: str) -> None:
    if log is not None:
        log(message)
