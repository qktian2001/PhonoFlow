from typer.testing import CliRunner

from phonoflow.cli import app


def test_root_version_option_reports_phonoflow_1_0_0() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "PhonoFlow 1.0.0" in result.stdout
