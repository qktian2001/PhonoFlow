import numpy as np
import pytest

from phonoflow.io.force_constants_io import write_force_constants_text


def test_force_constants_text_export_rejects_non_full_shape(tmp_path):
    compact_force_constants = np.zeros((1, 2, 3, 3), dtype=float)

    with pytest.raises(ValueError, match="not full supercell FC2"):
        write_force_constants_text(compact_force_constants, tmp_path, n_supercell_atoms=2)
