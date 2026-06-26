from pathlib import Path

from typer.testing import CliRunner

from phonoflow.cli import app


def test_doctor_no_longer_reports_pynep():
    runner = CliRunner()
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "PyNEP" not in result.stdout


def test_cli_help_no_longer_recommends_pynep():
    runner = CliRunner()
    result = runner.invoke(app, ["single", "--help"])
    assert result.exit_code == 0
    assert "pynep" not in result.stdout.lower()


def test_readme_mentions_pynep_only_as_history():
    readme = Path("README.md").read_text(encoding="utf-8")
    lowered = readme.lower()
    assert "install pynep" not in lowered
    assert "recommended backend is calorine cpunep" in lowered
