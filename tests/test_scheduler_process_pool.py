from __future__ import annotations

import functools
import multiprocessing
import time

import numpy as np

from phonoflow_scheduler.config import ForceScheduleConfig
from ase import Atoms

from phonoflow_scheduler.force_tasks import ForceDisplacementSpec, ForceResult, ForceTask
from phonoflow_scheduler.process_pool import (
    choose_chunk_size,
    choose_max_pending_tasks,
    evaluate_force_tasks,
)


class FakeCalculator:
    def __init__(self, init_log=None, sleep: float = 0.0) -> None:
        self.init_log = init_log
        self.sleep = sleep
        if init_log is not None:
            init_log.append("init")

    def calculate_energy_forces(self, payload):
        if self.sleep:
            time.sleep(self.sleep)
        value = float(payload)
        return {"energy": value, "forces": np.full((1, 3), value)}


class PositionCalculator:
    def calculate_energy_forces(self, payload):
        positions = np.asarray(payload.get_positions(), dtype=float)
        return {"energy": float(positions.sum()), "forces": positions}


def fake_calculator_factory(init_log=None, sleep: float = 0.0) -> FakeCalculator:
    return FakeCalculator(init_log=init_log, sleep=sleep)


def position_calculator_factory() -> PositionCalculator:
    return PositionCalculator()


def calculate_fake_task(calculator: FakeCalculator, task: ForceTask) -> ForceResult:
    result = calculator.calculate_energy_forces(task.payload)
    return ForceResult(index=task.index, forces=result["forces"], energy=result["energy"])


def calculate_position_task(calculator: PositionCalculator, task: ForceTask) -> ForceResult:
    result = calculator.calculate_energy_forces(task.payload)
    return ForceResult(index=task.index, forces=result["forces"], energy=result["energy"])


def _tasks(count: int) -> list[ForceTask]:
    return [ForceTask(index=index, payload=index) for index in range(count)]


def test_choose_chunk_size_uses_conservative_default() -> None:
    assert choose_chunk_size(num_tasks=1000, workers=24) == 6
    assert choose_chunk_size(num_tasks=8, workers=24) == 1
    assert choose_chunk_size(num_tasks=1000, workers=24, configured_chunk_size=8) == 8


def test_choose_chunk_size_uses_nep_calorine_single_task_policy() -> None:
    assert choose_chunk_size(num_tasks=4428, workers=30, backend_name="calorine") == 1
    assert choose_chunk_size(num_tasks=4428, workers=30, backend_name="nep89") == 1
    assert choose_chunk_size(num_tasks=4428, workers=30, backend_name="cpunep") == 1
    assert choose_chunk_size(num_tasks=4428, workers=30, configured_chunk_size=4, backend_name="calorine") == 4


def test_choose_chunk_size_uses_deepmd_dpa_cpu_policy() -> None:
    assert choose_chunk_size(num_tasks=4428, workers=30, backend_name="dpa4neo") == 1
    assert choose_chunk_size(num_tasks=4428, workers=30, backend_name="deepmd") == 1
    assert choose_chunk_size(num_tasks=181, workers=36, backend_name="dpa4neo") == 1
    assert choose_chunk_size(num_tasks=4428, workers=30, configured_chunk_size=2, backend_name="dpa4neo") == 2


def test_choose_max_pending_tasks_defaults_to_four_chunks_per_worker() -> None:
    assert choose_max_pending_tasks(workers=6) == 24
    assert choose_max_pending_tasks(workers=6, configured_max_pending_tasks=3) == 3


def test_serial_scheduler_preserves_order_and_uses_one_calculator() -> None:
    init_log: list[str] = []
    batch = evaluate_force_tasks(
        _tasks(5),
        calculator_factory=functools.partial(fake_calculator_factory, init_log),
        calculate_task=calculate_fake_task,
        schedule_config=ForceScheduleConfig(
            force_parallel_backend="serial",
            force_workers=1,
            chunk_size=2,
        ),
    )

    assert [result.index for result in batch.ordered_results()] == [0, 1, 2, 3, 4]
    assert [float(result.forces[0, 0]) for result in batch.ordered_results()] == [0, 1, 2, 3, 4]
    assert init_log == ["init"]
    assert batch.metadata["chunk_size"] == 2


def test_process_scheduler_preserves_order_with_chunks() -> None:
    batch = evaluate_force_tasks(
        _tasks(9),
        calculator_factory=fake_calculator_factory,
        calculate_task=calculate_fake_task,
        schedule_config=ForceScheduleConfig(
            force_parallel_backend="process",
            force_workers=3,
            chunk_size=4,
            max_pending_tasks=2,
        ),
    )

    ordered = batch.ordered_results()

    assert [result.index for result in ordered] == list(range(9))
    assert [float(result.forces[0, 0]) for result in ordered] == list(range(9))
    assert batch.metadata["submitted_chunks"] == 3
    assert batch.metadata["max_pending_observed"] <= 2


def test_process_scheduler_reuses_worker_calculators_across_chunks() -> None:
    with multiprocessing.Manager() as manager:
        init_log = manager.list()
        batch = evaluate_force_tasks(
            _tasks(12),
            calculator_factory=functools.partial(fake_calculator_factory, init_log, 0.01),
            calculate_task=calculate_fake_task,
            schedule_config=ForceScheduleConfig(
                force_parallel_backend="process",
                force_workers=2,
                chunk_size=1,
                max_pending_tasks=2,
            ),
        )

        assert [result.index for result in batch.ordered_results()] == list(range(12))
        assert 1 <= len(init_log) <= 2
        assert len(init_log) < 12


def test_process_scheduler_resolves_compact_displacement_spec_payloads() -> None:
    base = Atoms("Si2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], cell=[5, 5, 5], pbc=True)
    displaced = base.copy()
    displaced.set_positions([[0.1, 0.0, 0.0], [1.0, 0.2, 0.0]])
    tasks = [ForceTask(index=0, payload=ForceDisplacementSpec.from_atoms(displaced))]

    batch = evaluate_force_tasks(
        tasks,
        calculator_factory=position_calculator_factory,
        calculate_task=calculate_position_task,
        schedule_config=ForceScheduleConfig(
            force_parallel_backend="process",
            force_workers=2,
            chunk_size=1,
            base_payload=base,
        ),
    )

    np.testing.assert_allclose(batch.ordered_results()[0].forces, displaced.get_positions())
