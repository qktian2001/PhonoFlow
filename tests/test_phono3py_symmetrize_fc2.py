from __future__ import annotations

import numpy as np

from phonoflow.thermal.fc3_finite_displacement import _apply_phono3py_symmetrize_fc2


class _FakePhono3py:
    def __init__(self) -> None:
        self.fc2 = np.ones((2, 2, 3, 3))
        self.called = False

    def symmetrize_fc2(self) -> None:
        self.called = True
        self.fc2 = np.zeros((2, 2, 3, 3))


def test_phono3py_symmetrize_fc2_hook_records_before_and_after_residual() -> None:
    phono3py = _FakePhono3py()

    result = _apply_phono3py_symmetrize_fc2(phono3py, enabled=True)

    assert phono3py.called is True
    assert result["phono3py_symmetrize_fc2"] is True
    assert result["phono3py_symmetrize_fc2_applied"] is True
    assert result["fc2_asr_residual_before"] > result["fc2_asr_residual_after"]


def test_phono3py_symmetrize_fc2_hook_is_noop_when_disabled() -> None:
    phono3py = _FakePhono3py()

    result = _apply_phono3py_symmetrize_fc2(phono3py, enabled=False)

    assert phono3py.called is False
    assert result["phono3py_symmetrize_fc2"] is False
    assert result["phono3py_symmetrize_fc2_applied"] is False
