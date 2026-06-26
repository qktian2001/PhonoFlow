import numpy as np
from ase.build import bulk

from phonoflow.workflow.displace import ase_atoms_to_phonopy_atoms, phonopy_atoms_to_ase_atoms


def test_ase_phonopy_round_trip():
    atoms = bulk("Si", "diamond", a=5.43)
    phonopy_atoms = ase_atoms_to_phonopy_atoms(atoms)
    round_trip = phonopy_atoms_to_ase_atoms(phonopy_atoms)

    assert round_trip.get_chemical_symbols() == atoms.get_chemical_symbols()
    assert np.allclose(round_trip.cell.array, atoms.cell.array)
    assert np.allclose(round_trip.get_scaled_positions(), atoms.get_scaled_positions())
