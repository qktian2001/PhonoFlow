from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from phonoflow.cli import app


def test_dpa4_cli_options_are_recorded_in_dry_run(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    model = tmp_path / "dpa4.pt"
    model.write_bytes(b"fake-model")
    outdir = tmp_path / "dry_run"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "single",
            "--input-path",
            str(repo / "examples" / "Si.vasp"),
            "--model-path",
            str(model),
            "--outdir",
            str(outdir),
            "--backend",
            "dpa4",
            "--deepmd-deterministic",
            "--deepmd-reuse-calculator",
            "--save-force-audit",
            "--phono3py-symmetrize-fc2",
            "--dry-run",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads((outdir / "result.json").read_text(encoding="utf-8"))
    assert data["backend_requested"] == "dpa4"
    assert data["backend_resolved"] == "deepmd"
    assert data["model_backend_family"] == "deepmd"
    assert data["deepmd_deterministic"] is True
    assert data["deepmd_reuse_calculator"] is True
    assert data["save_force_audit"] is True
    assert data["phono3py_symmetrize_fc2"] is True
