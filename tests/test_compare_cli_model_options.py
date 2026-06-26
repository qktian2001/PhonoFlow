from __future__ import annotations

from typer.testing import CliRunner

from phonoflow.cli import app


def test_compare_help_supports_repeatable_model_without_fixed_three_model_default() -> None:
    result = CliRunner().invoke(app, ["compare-models", "--help"])
    help_text = " ".join(result.output.replace("│", " ").split())

    assert result.exit_code == 0
    assert "--model" in help_text
    assert "Repeat for 1-3 models" in help_text
    assert "[default: nep89,dpa3,dpa4]" not in help_text
    assert "Compare one to three independent model workflows" in help_text
