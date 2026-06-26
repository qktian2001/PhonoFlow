from pathlib import Path

import numpy as np

from phonoflow.io.force_constants_io import write_force_constants_text


def test_force_constants_text_export_writes_phonopy_and_shengbte_files(tmp_path: Path):
    force_constants = np.zeros((2, 2, 3, 3), dtype=float)
    result = write_force_constants_text(force_constants, tmp_path, n_supercell_atoms=2)

    phonopy_path = tmp_path / "FORCE_CONSTANTS"
    shengbte_path = tmp_path / "FORCE_CONSTANTS_2ND"

    assert result["force_constants_text_exported"] is True
    assert result["force_constants_text_shape"] == [2, 2, 3, 3]
    assert phonopy_path.exists()
    assert shengbte_path.exists()
    assert phonopy_path.stat().st_size > 0
    assert shengbte_path.stat().st_size > 0
    assert phonopy_path.read_text(encoding="utf-8").splitlines()[0].split()[0] == "2"
    assert phonopy_path.read_text(encoding="utf-8") == shengbte_path.read_text(encoding="utf-8")
