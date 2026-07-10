"""Lightweight benchmark for PhonoFlow force-task scheduling.

The benchmark uses synthetic force tasks so it can compare scheduler overhead
without launching a real FC3 calculation or loading a production potential.
It writes a JSON result into a timestamped output directory.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from phonoflow_scheduler.config import ForceScheduleConfig
from phonoflow_scheduler.force_tasks import ForceResult, ForceTask
from phonoflow_scheduler.process_pool import evaluate_force_tasks


class FakeCalculator:
    """Small calculator with configurable init and force latency."""

    def __init__(self, init_sleep: float, force_sleep: float) -> None:
        self.init_sleep = init_sleep
        self.force_sleep = force_sleep
        if init_sleep > 0:
            time.sleep(init_sleep)

    def calculate(self, task: ForceTask) -> ForceResult:
        if self.force_sleep > 0:
            time.sleep(self.force_sleep)
        scale = float(task.index + 1)
        return ForceResult(index=task.index, forces=np.full((2, 3), scale, dtype=float))


def calculator_factory(init_sleep: float, force_sleep: float) -> FakeCalculator:
    return FakeCalculator(init_sleep=init_sleep, force_sleep=force_sleep)


def calculate_task(calculator: FakeCalculator, task: ForceTask) -> ForceResult:
    return calculator.calculate(task)


def legacy_calculate_one(task: ForceTask, init_sleep: float, force_sleep: float) -> ForceResult:
    calculator = FakeCalculator(init_sleep=init_sleep, force_sleep=force_sleep)
    return calculator.calculate(task)


def benchmark_legacy(tasks: list[ForceTask], *, workers: int, init_sleep: float, force_sleep: float) -> dict[str, Any]:
    started = time.perf_counter()
    results: list[ForceResult] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(legacy_calculate_one, task, init_sleep, force_sleep) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed = max(0.0, time.perf_counter() - started)
    results.sort(key=lambda item: item.index)
    return {
        "scheduler": "legacy_one_future_per_displacement",
        "wall_time": elapsed,
        "displacements_per_second": len(tasks) / elapsed if elapsed > 0 else None,
        "submitted_futures": len(tasks),
        "result_checksum": _checksum(results),
    }


def benchmark_scheduler(
    tasks: list[ForceTask],
    *,
    workers: int,
    chunk_size: int | None,
    max_pending_tasks: int | None,
    init_sleep: float,
    force_sleep: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = evaluate_force_tasks(
        tasks,
        calculator_factory=partial(calculator_factory, init_sleep, force_sleep),
        calculate_task=calculate_task,
        schedule_config=ForceScheduleConfig(
            force_workers=workers,
            force_parallel_backend="process" if workers > 1 else "serial",
            chunk_size=chunk_size,
            max_pending_tasks=max_pending_tasks,
        ),
    )
    elapsed = max(0.0, time.perf_counter() - started)
    return {
        "scheduler": "phonoflow_scheduler_chunked_worker_reuse",
        "wall_time": elapsed,
        "displacements_per_second": len(tasks) / elapsed if elapsed > 0 else None,
        "metadata": result.metadata,
        "result_checksum": _checksum(result.ordered_results()),
    }


def _checksum(results: list[ForceResult]) -> float:
    return float(sum(np.asarray(item.forces, dtype=float).sum() for item in results))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark synthetic force scheduling overhead.")
    parser.add_argument("--tasks", type=int, default=240)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--max-pending-tasks", type=int, default=None)
    parser.add_argument("--init-sleep", type=float, default=0.02)
    parser.add_argument("--force-sleep", type=float, default=0.005)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    outdir = args.outdir or Path("work") / "scheduler_benchmarks" / datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    tasks = [ForceTask(index=index, payload={"index": index}, label="benchmark") for index in range(max(1, args.tasks))]

    legacy = benchmark_legacy(
        tasks,
        workers=max(1, args.workers),
        init_sleep=max(0.0, args.init_sleep),
        force_sleep=max(0.0, args.force_sleep),
    )
    scheduler = benchmark_scheduler(
        tasks,
        workers=max(1, args.workers),
        chunk_size=args.chunk_size,
        max_pending_tasks=args.max_pending_tasks,
        init_sleep=max(0.0, args.init_sleep),
        force_sleep=max(0.0, args.force_sleep),
    )
    speedup = legacy["wall_time"] / scheduler["wall_time"] if scheduler["wall_time"] > 0 else None
    report = {
        "parameters": {
            "tasks": len(tasks),
            "workers": max(1, args.workers),
            "chunk_size": args.chunk_size,
            "max_pending_tasks": args.max_pending_tasks,
            "init_sleep": max(0.0, args.init_sleep),
            "force_sleep": max(0.0, args.force_sleep),
        },
        "legacy": legacy,
        "scheduler": scheduler,
        "speedup_legacy_over_scheduler": speedup,
    }
    output_path = outdir / "force_scheduler_benchmark.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote benchmark report: {output_path}")


if __name__ == "__main__":
    main()
