"""Worker-pull force scheduler prototype.

This backend keeps a fixed worker pool alive, initializes each worker
calculator once, and lets workers pull single-displacement tasks from a queue.
It is intended for FC3 force loops where chunk=1 gives better load balancing.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any

from phonoflow_scheduler.config import ForceScheduleConfig, clean_positive_int
from phonoflow_scheduler.force_tasks import ForceBatchResult, ForceResult, ForceTask, resolve_force_task_payload
from phonoflow_scheduler.process_pool import CalculatorFactory, TaskEvaluator, choose_max_pending_tasks


_STOP = "__phonoflow_worker_queue_stop__"


def evaluate_force_tasks_with_worker_queue(
    tasks: Iterable[ForceTask],
    *,
    calculator_factory: CalculatorFactory,
    calculate_task: TaskEvaluator,
    schedule_config: ForceScheduleConfig | None = None,
    log: Callable[[str], None] | None = None,
) -> ForceBatchResult:
    """Evaluate force tasks with fixed workers pulling single tasks."""

    task_list = list(tasks)
    config = schedule_config or ForceScheduleConfig(force_parallel_backend="worker_queue")
    workers = min(clean_positive_int(config.force_workers), max(1, len(task_list)))
    if not task_list:
        return ForceBatchResult(
            results=[],
            metadata=_metadata(
                total_tasks=0,
                workers=workers,
                max_pending=0,
                max_pending_observed=0,
                wall_time=0.0,
                submit_time=0.0,
                collection_time=0.0,
            ),
            warnings=[],
        )
    if workers <= 1:
        return _evaluate_serial_worker_queue(
            task_list,
            calculator_factory=calculator_factory,
            calculate_task=calculate_task,
            base_payload=config.base_payload,
            log=log,
        )

    max_pending = choose_max_pending_tasks(
        workers=workers,
        configured_max_pending_tasks=config.max_pending_tasks,
    )
    started = time.perf_counter()
    submit_seconds = 0.0
    collection_seconds = 0.0
    results: list[ForceResult] = []
    ctx = mp.get_context()
    task_queue = ctx.Queue(maxsize=max_pending)
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_worker_loop,
            args=(task_queue, result_queue, calculator_factory, calculate_task, config.base_payload),
        )
        for _ in range(workers)
    ]
    for process in processes:
        process.start()

    max_pending_observed = 0
    submitted = 0
    collected = 0
    completed_normally = False
    try:
        while collected < len(task_list):
            submit_started = time.perf_counter()
            while submitted < len(task_list) and (submitted - collected) < max_pending:
                task_queue.put(task_list[submitted])
                submitted += 1
                max_pending_observed = max(max_pending_observed, submitted - collected)
            submit_seconds += max(0.0, time.perf_counter() - submit_started)

            collect_started = time.perf_counter()
            try:
                ok, payload = result_queue.get(timeout=0.1)
            except queue.Empty:
                collection_seconds += max(0.0, time.perf_counter() - collect_started)
                _raise_if_worker_exited(processes)
                continue
            collection_seconds += max(0.0, time.perf_counter() - collect_started)
            if not ok:
                raise RuntimeError(str(payload))
            results.append(payload)
            collected += 1
            _log(log, f"Evaluated force task {payload.index + 1}")
        completed_normally = True
    finally:
        if completed_normally:
            for _ in processes:
                task_queue.put(_STOP)
            for process in processes:
                process.join(timeout=5.0)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

    results.sort(key=lambda result: result.index)
    wall_time = max(0.0, time.perf_counter() - started)
    return ForceBatchResult(
        results=results,
        metadata=_metadata(
            total_tasks=len(task_list),
            workers=workers,
            max_pending=max_pending,
            max_pending_observed=max_pending_observed,
            wall_time=wall_time,
            submit_time=submit_seconds,
            collection_time=collection_seconds,
        ),
        warnings=[],
    )


def _evaluate_serial_worker_queue(
    tasks: list[ForceTask],
    *,
    calculator_factory: CalculatorFactory,
    calculate_task: TaskEvaluator,
    base_payload: Any | None,
    log: Callable[[str], None] | None,
) -> ForceBatchResult:
    started = time.perf_counter()
    submit_started = time.perf_counter()
    calculator = calculator_factory()
    submit_seconds = max(0.0, time.perf_counter() - submit_started)
    results: list[ForceResult] = []
    for task in tasks:
        resolved_task = _resolve_task(task, base_payload)
        results.append(calculate_task(calculator, resolved_task))
        _log(log, f"Evaluated force task {task.index + 1}")
    wall_time = max(0.0, time.perf_counter() - started)
    return ForceBatchResult(
        results=sorted(results, key=lambda result: result.index),
        metadata=_metadata(
            total_tasks=len(tasks),
            workers=1,
            max_pending=1,
            max_pending_observed=1,
            wall_time=wall_time,
            submit_time=submit_seconds,
            collection_time=0.0,
        ),
        warnings=[],
    )


def _worker_loop(
    task_queue: Any,
    result_queue: Any,
    calculator_factory: CalculatorFactory,
    calculate_task: TaskEvaluator,
    base_payload: Any | None,
) -> None:
    try:
        calculator = calculator_factory()
        while True:
            task = task_queue.get()
            if task == _STOP:
                return
            resolved_task = _resolve_task(task, base_payload)
            result_queue.put((True, calculate_task(calculator, resolved_task)))
    except Exception as exc:
        details = traceback.format_exc()
        result_queue.put((False, f"worker_queue task failed: {exc}\n{details}"))


def _resolve_task(task: ForceTask, base_payload: Any | None) -> ForceTask:
    payload = resolve_force_task_payload(task.payload, base_payload)
    if payload is task.payload:
        return task
    return replace(task, payload=payload)


def _metadata(
    *,
    total_tasks: int,
    workers: int,
    max_pending: int,
    max_pending_observed: int,
    wall_time: float,
    submit_time: float,
    collection_time: float,
) -> dict[str, Any]:
    return {
        "n_displacements": total_tasks,
        "force_parallel_backend": "worker_queue" if workers > 1 else "serial",
        "requested_force_workers": workers,
        "effective_force_workers": workers,
        "chunk_size": 1,
        "max_pending_tasks": max_pending,
        "max_pending_observed": max_pending_observed,
        "submitted_chunks": total_tasks,
        "wall_time": wall_time,
        "displacements_per_second": float(total_tasks / wall_time) if wall_time > 0 else None,
        "submit_time": submit_time,
        "force_evaluation_wall_time": wall_time,
        "result_collection_time": collection_time,
        "result_ordering_time": 0.0,
        "worker_queue": True,
    }


def _raise_if_worker_exited(processes: list[Any]) -> None:
    failed = [process.exitcode for process in processes if process.exitcode not in (None, 0)]
    if failed:
        raise RuntimeError(f"worker_queue worker exited unexpectedly with code(s): {failed}")


def _log(log: Callable[[str], None] | None, message: str) -> None:
    if log is not None:
        log(message)
