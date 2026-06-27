from typer.testing import CliRunner

from phonoflow.cli import app


def test_root_version_option_reports_phonoflow_1_0_0() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "PhonoFlow 1.0.0" in result.stdout


def test_root_help_all_reports_parameter_reference() -> None:
    result = CliRunner().invoke(app, ["--help-all"])

    assert result.exit_code == 0
    assert "PhonoFlow workflow parameters" in result.stdout
    assert "input_path" in result.stdout
    assert "compute_kappa" in result.stdout
    assert "deepmd_device" in result.stdout


def test_root_show_parameters_alias_reports_parameter_reference() -> None:
    result = CliRunner().invoke(app, ["--show-parameters"])

    assert result.exit_code == 0
    assert "PhonoFlow workflow parameters" in result.stdout
    assert "phono3py_symmetrize_fc2" in result.stdout
