from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from phonoflow.cli import app


def test_run_accepts_two_hihive_cutoffs_for_dry_run(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    outdir = tmp_path / "dry_run"
    model_path = tmp_path / "nep_model.txt"
    model_path.write_text("placeholder\n", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run",
            "--input-path",
            str(repo / "examples" / "Si.vasp"),
            "--model-path",
            str(model_path),
            "--outdir",
            str(outdir),
            "--compute-kappa",
            "--fc3-method",
            "hiphive",
            "--cutoffs",
            "4.5",
            "3.0",
            "--dry-run",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads((outdir / "result.json").read_text(encoding="utf-8"))
    assert data["thermal_conductivity"]["enabled"] is False
    settings = json.loads((outdir / "resolved_settings.json").read_text(encoding="utf-8"))
    assert settings["cutoffs"]["value"] == [4.5, 3.0]
