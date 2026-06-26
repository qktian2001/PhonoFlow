from __future__ import annotations

from pathlib import Path

import numpy as np

from phonoflow.config import WorkflowConfig
from phonoflow.thermal.fc3_finite_displacement import _write_fd_diagnostics


class _FakeSupercell:
    def __len__(self) -> int:
        return 4


class _FakePhono3py:
    supercell = _FakeSupercell()
    phonon_supercells_with_displacements = [object(), object()]
    supercells_with_displacements = [object(), object(), object()]


def test_fd_diagnostics_write_expected_fields(tmp_path: Path) -> None:
    config = WorkflowConfig(
        compute_kappa=True,
        displacement=0.012,
        fc3_method="finite-displacement",
        kappa_method="rta",
        temperatures=[300.0],
        kappa_mesh=[5, 5, 5],
        fc3_supercell_dim=[2, 2, 2],
        fc3_displacement=0.034,
    )
    fc2 = np.zeros((2, 4, 3, 3))
    fc3 = np.zeros((2, 4, 4, 3, 3, 3))

    diagnostics = _write_fd_diagnostics(
        outdir=tmp_path,
        config=config,
        fc3_supercell_dim=[2, 2, 2],
        kappa_mesh=[5, 5, 5],
        temperatures=[300.0],
        phono3py=_FakePhono3py(),
        fc2=fc2,
        fc3=fc3,
        fc2_path=tmp_path / "fc2.hdf5",
        fc3_path=tmp_path / "fc3.hdf5",
        kappa_path=tmp_path / "kappa-m555-g0.hdf5",
        thermal_csv=tmp_path / "thermal_conductivity.csv",
        thermal_png=tmp_path / "thermal_conductivity.png",
        lifetime={"available": True, "data_file": "phonon_lifetime.csv", "plot_file": "phonon_lifetime.png"},
        warnings=[],
    )

    assert diagnostics["method"] == "finite-displacement"
    assert diagnostics["supercell_dim"] == [2, 2, 2]
    assert diagnostics["kappa_mesh"] == [5, 5, 5]
    assert diagnostics["fc2_displacement"] == 0.012
    assert diagnostics["fc3_displacement"] == 0.034
    assert diagnostics["fc2_shape"] == [2, 4, 3, 3]
    assert diagnostics["fc3_shape"] == [2, 4, 4, 3, 3, 3]
    assert diagnostics["fc2_diagnostics"]["norm"] == 0.0
    assert diagnostics["fc3_diagnostics"]["norm"] == 0.0
    assert "eV/Angstrom" in diagnostics["force_units_note"]
    for filename in diagnostics["files"].values():
        assert (tmp_path / filename).exists()
