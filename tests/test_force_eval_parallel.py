from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from phonoflow_scheduler.force_tasks import ForceDisplacementSpec
from phonoflow.workflow.force_eval import evaluate_forces_for_displacements


def _atoms_with_count(count: int) -> Atoms:
    return Atoms("Si" * count, positions=[[float(i), 0.0, 0.0] for i in range(count)], cell=[10, 10, 10], pbc=True)


def test_parallel_force_workers_preserve_force_count_order_and_shape() -> None:
    atoms_list = [_atoms_with_count(1), _atoms_with_count(3), _atoms_with_count(2)]

    serial = evaluate_forces_for_displacements(
        atoms_list,
        backend_name="dummy",
        force_workers=1,
        force_parallel_backend="serial",
    )
    parallel = evaluate_forces_for_displacements(
        atoms_list,
        backend_name="dummy",
        force_workers=2,
        force_parallel_backend="process",
        deepmd_device="cpu",
    )

    assert len(serial.forces) == len(parallel.forces) == 3
    assert [forces.shape for forces in parallel.forces] == [(1, 3), (3, 3), (2, 3)]
    for serial_forces, parallel_forces in zip(serial.forces, parallel.forces, strict=True):
        np.testing.assert_allclose(parallel_forces, serial_forces)
    assert parallel.metadata["effective_force_workers"] == 2
    assert parallel.metadata["force_parallel_backend"] == "process"


def test_cuda_force_workers_downgrade_to_serial_with_warning() -> None:
    result = evaluate_forces_for_displacements(
        [_atoms_with_count(1), _atoms_with_count(2)],
        backend_name="dummy",
        force_workers=4,
        force_parallel_backend="process",
        deepmd_device="cuda",
    )

    assert result.metadata["effective_force_workers"] == 1
    assert result.metadata["force_parallel_backend"] == "serial"
    assert any("CUDA" in warning and "force_workers=1" in warning for warning in result.warnings)


def test_parallel_force_worker_failure_mentions_displacement_index() -> None:
    atoms_list = [_atoms_with_count(1), _atoms_with_count(2)]
    atoms_list[1].info["phonoflow_dummy_force_error"] = "intentional failure"

    with pytest.raises(RuntimeError, match="displacement index 1"):
        evaluate_forces_for_displacements(
            atoms_list,
            backend_name="dummy",
            force_workers=2,
            force_parallel_backend="process",
            deepmd_device="cpu",
        )


def test_worker_queue_matches_process_backend_for_force_eval() -> None:
    atoms_list = [_atoms_with_count(1), _atoms_with_count(3), _atoms_with_count(2)]

    process = evaluate_forces_for_displacements(
        atoms_list,
        backend_name="dummy",
        force_workers=2,
        force_parallel_backend="process",
        deepmd_device="cpu",
    )
    worker_queue = evaluate_forces_for_displacements(
        atoms_list,
        backend_name="dummy",
        force_workers=2,
        force_parallel_backend="worker_queue",
        deepmd_device="cpu",
    )

    for process_forces, queue_forces in zip(process.forces, worker_queue.forces, strict=True):
        np.testing.assert_allclose(queue_forces, process_forces)
    assert worker_queue.metadata["force_parallel_backend"] == "worker_queue"


def test_compact_displacement_specs_match_atoms_payloads() -> None:
    base = _atoms_with_count(2)
    atoms_list = [base.copy(), base.copy()]
    atoms_list[0].set_positions([[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms_list[1].set_positions([[0.0, 0.2, 0.0], [1.0, 0.0, 0.3]])
    specs = [ForceDisplacementSpec.from_atoms(atoms) for atoms in atoms_list]

    atoms_result = evaluate_forces_for_displacements(
        atoms_list,
        backend_name="dummy",
        force_workers=2,
        force_parallel_backend="process",
        deepmd_device="cpu",
    )
    spec_result = evaluate_forces_for_displacements(
        specs,
        backend_name="dummy",
        force_workers=2,
        force_parallel_backend="process",
        deepmd_device="cpu",
        base_atoms=base,
    )

    for atoms_forces, spec_forces in zip(atoms_result.forces, spec_result.forces, strict=True):
        np.testing.assert_allclose(spec_forces, atoms_forces)


def test_direct_force_backend_bypasses_scheduler_and_preserves_forces() -> None:
    atoms_list = [_atoms_with_count(1), _atoms_with_count(3), _atoms_with_count(2)]
    messages: list[str] = []

    serial = evaluate_forces_for_displacements(
        atoms_list,
        backend_name="dummy",
        force_workers=1,
        force_parallel_backend="serial",
    )
    direct = evaluate_forces_for_displacements(
        atoms_list,
        backend_name="dummy",
        force_workers=36,
        force_parallel_backend="direct",
        log=messages.append,
        audit_label="fc3",
    )

    for serial_forces, direct_forces in zip(serial.forces, direct.forces, strict=True):
        np.testing.assert_allclose(direct_forces, serial_forces)
    assert direct.metadata["force_parallel_backend"] == "direct"
    assert direct.metadata["effective_force_workers"] == 1
    assert direct.metadata["direct_no_scheduler"] is True
    assert direct.metadata["chunk_size"] is None
    assert any("Evaluating FC3 displaced supercell 1/3" in message for message in messages)


def test_direct_force_backend_resolves_compact_displacement_specs() -> None:
    base = _atoms_with_count(2)
    displaced = base.copy()
    displaced.set_positions([[0.2, 0.0, 0.0], [1.0, 0.3, 0.0]])

    atoms_result = evaluate_forces_for_displacements(
        [displaced],
        backend_name="dummy",
        force_workers=1,
        force_parallel_backend="direct",
    )
    spec_result = evaluate_forces_for_displacements(
        [ForceDisplacementSpec.from_atoms(displaced)],
        backend_name="dummy",
        force_workers=1,
        force_parallel_backend="direct",
        base_atoms=base,
    )

    np.testing.assert_allclose(spec_result.forces[0], atoms_result.forces[0])
