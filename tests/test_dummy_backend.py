import numpy as np
from ase import Atoms

from phonoflow.calculators.dummy import DummyBackend


def test_dummy_backend_energy_forces_shape():
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[5, 5, 5], pbc=True)
    backend = DummyBackend()
    result = backend.calculate_energy_forces(atoms)

    assert result["energy"] == 0.0
    assert result["forces"].shape == (2, 3)
    assert np.allclose(result["forces"], 0.0)
