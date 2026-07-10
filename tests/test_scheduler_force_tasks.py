from __future__ import annotations

import numpy as np

from phonoflow_scheduler.force_tasks import ForceBatchResult, ForceResult, ForceTask


def test_force_task_and_result_preserve_index() -> None:
    task = ForceTask(index=7, payload={"atoms": 2})
    result = ForceResult(index=task.index, forces=np.zeros((2, 3)), energy=0.0)

    assert task.index == 7
    assert result.index == 7
    assert result.forces.shape == (2, 3)


def test_force_batch_result_orders_by_index() -> None:
    first = ForceResult(index=2, forces=np.full((1, 3), 2.0))
    second = ForceResult(index=1, forces=np.full((1, 3), 1.0))
    batch = ForceBatchResult(results=[first, second], metadata={}, warnings=[])

    ordered = batch.ordered_results()

    assert [result.index for result in ordered] == [1, 2]
