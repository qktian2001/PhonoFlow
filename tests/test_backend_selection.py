import pytest

from phonoflow.calculators import get_backend
from phonoflow.calculators.calorine_backend import CalorineBackend
from phonoflow.calculators.dummy import DummyBackend
from phonoflow.calculators.gpumd_backend import GPUMDBackend
from phonoflow.exceptions import ConfigError


def test_backend_selection_known_backends():
    assert isinstance(get_backend("dummy"), DummyBackend)
    assert isinstance(get_backend("calorine"), CalorineBackend)
    assert isinstance(get_backend("gpumd"), GPUMDBackend)


def test_backend_selection_is_case_insensitive():
    assert isinstance(get_backend("CALORINE"), CalorineBackend)


def test_backend_selection_invalid_backend_is_clear():
    with pytest.raises(ConfigError, match="auto, dummy, calorine, gpumd"):
        get_backend("not-a-backend")


def test_pynep_backend_removed_error_is_clear():
    with pytest.raises(ConfigError, match="PyNEP backend has been removed"):
        get_backend("pynep")
