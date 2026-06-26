import numpy as np
import pytest

pytest.importorskip("phonopy")
from phonoflow.workflow.phonon import _flatten_band_frequencies


def test_flatten_band_frequencies_accepts_ragged_segments():
    frequencies = [
        np.zeros((3, 2)),
        np.ones((2, 2)),
        np.full((4, 2), 2.0),
    ]

    flattened = _flatten_band_frequencies({"frequencies": frequencies})

    assert flattened.shape == (18,)
    assert np.isclose(flattened[-1], 2.0)
