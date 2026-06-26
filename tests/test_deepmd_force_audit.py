from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms

from phonoflow.workflow.force_audit import build_force_audit_record, write_force_audit_files


def test_force_audit_writes_stats_hashes_and_raw_npz(tmp_path: Path) -> None:
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[3, 3, 3], pbc=True)
    forces = np.array([[0.1, 0.0, -0.1], [0.2, -0.2, 0.0]])
    records = [build_force_audit_record(0, atoms, energy=-2.5, forces=forces)]

    files = write_force_audit_files(tmp_path, "fc2", records, np.asarray([forces]))

    assert files["fd_fc2_forces_stats_csv"] == "fd_fc2_forces_stats.csv"
    assert files["fd_fc2_force_hashes_csv"] == "fd_fc2_force_hashes.csv"
    assert files["fd_fc2_forces_raw_npz"] == "fd_fc2_forces_raw.npz"
    stats_text = (tmp_path / "fd_fc2_forces_stats.csv").read_text(encoding="utf-8")
    hashes_text = (tmp_path / "fd_fc2_force_hashes.csv").read_text(encoding="utf-8")
    assert "index,natoms,energy,force_max_abs,force_mean_abs,force_norm" in stats_text
    assert "force_sha256,structure_hash,cell_hash,positions_hash" in stats_text
    assert "index,force_sha256,structure_hash,cell_hash,positions_hash" in hashes_text
    raw = np.load(tmp_path / "fd_fc2_forces_raw.npz")
    np.testing.assert_allclose(raw["forces"], np.asarray([forces]))
