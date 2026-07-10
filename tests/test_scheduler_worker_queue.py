from __future__ import annotations

import numpy as np

from phonoflow_scheduler.config import ForceScheduleConfig
from phonoflow_scheduler.force_tasks import ForceResult, ForceTask
from phonoflow_scheduler.process_pool import evaluate_force_tasks
from phonoflow_scheduler.worker_queue import evaluate_force_tasks_with_worker_queue


class QueueTestCalculator:
    def calculate_energy_forces(self, payload):
        value = float(payload)
        return {"energy": value, "forces": np.full((2, 3), value)}


def queue_test_calculator_factory() -> QueueTestCalculator:
    return QueueTestCalculator()


def calculate_queue_test_task(calculator: QueueTestCalculator, task: ForceTask) -> ForceResult:
    result = calculator.calculate_energy_forces(task.payload)
    return ForceResult(index=task.index, forces=result["forces"], energy=result["energy"])


def _tasks(count: int) -> list[ForceTask]:
    return [ForceTask(index=index, payload=index) for index in range(count)]


def test_worker_queue_preserves_order_and_matches_process_pool() -> None:
    tasks = _tasks(12)
    process_batch = evaluate_force_tasks(
        tasks,
        calculator_factory=queue_test_calculator_factory,
        calculate_task=calculate_queue_test_task,
        schedule_config=ForceScheduleConfig(
            force_workers=3,
            force_parallel_backend="process",
            chunk_size=1,
            max_pending_tasks=3,
        ),
    )
    queue_batch = evaluate_force_tasks_with_worker_queue(
        tasks,
        calculator_factory=queue_test_calculator_factory,
        calculate_task=calculate_queue_test_task,
        schedule_config=ForceScheduleConfig(
            force_workers=3,
            force_parallel_backend="worker_queue",
            chunk_size=1,
            max_pending_tasks=3,
        ),
    )

    process_results = process_batch.ordered_results()
    queue_results = queue_batch.ordered_results()
    assert [result.index for result in queue_results] == list(range(12))
    for process_result, queue_result in zip(process_results, queue_results, strict=True):
        np.testing.assert_allclose(queue_result.forces, process_result.forces)
    assert queue_batch.metadata["force_parallel_backend"] == "worker_queue"
    assert queue_batch.metadata["chunk_size"] == 1
