from pathlib import Path

from phonoflow.config import WorkflowConfig, load_config, write_config


def test_default_config_values():
    config = WorkflowConfig()
    assert config.backend == "auto"
    assert config.supercell_dim == "auto"
    assert config.mesh == "auto"
    assert config.dos is True
    assert config.imag_threshold == -0.1
    assert config.compute_kappa is False
    assert config.fc3_method == "finite-displacement"
    assert config.kappa_method == "rta"
    assert config.temperatures == [300.0]
    assert config.target_supercell_length == 15.0
    assert config.fc3_target_supercell_length == 10.0
    assert config.max_fc3_supercell_atoms == 256


def test_yaml_round_trip(tmp_path: Path):
    path = tmp_path / "config.yaml"
    config = WorkflowConfig(input_path=Path("examples/Si.vasp"), outdir=tmp_path / "out")
    write_config(config, path)

    loaded = load_config(path)
    assert loaded.input_path == Path("examples/Si.vasp")
    assert loaded.outdir == tmp_path / "out"
    assert loaded.backend == "auto"
