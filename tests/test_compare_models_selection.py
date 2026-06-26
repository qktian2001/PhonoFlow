from __future__ import annotations

from pathlib import Path

import pytest

from phonoflow.compare_models import _model_command, _resolve_compare_model_spec, compare_models
from phonoflow.exceptions import ConfigError


@pytest.mark.parametrize(
    "selection,alias,filename",
    [
        ("DPA-3.1-3M.pt", "dpa31", "DPA-3.1-3M.pt"),
        ("DPA-3.2-5M.pt", "dpa32", "DPA-3.2-5M.pt"),
        ("DPA-3.3-1M.pt", "dpa33", "DPA-3.3-1M.pt"),
        ("DPA4-Neo-OMat24-v20260528_rc.pt", "dpa4neo", "DPA4-Neo-OMat24-v20260528_rc.pt"),
    ],
)
def test_compare_resolves_exact_bundled_model_filename(
    selection: str,
    alias: str,
    filename: str,
) -> None:
    spec = _resolve_compare_model_spec(selection)

    assert spec["model_id"] == alias
    assert spec["backend"] == alias
    assert Path(spec["model_path"]).name == filename
    assert spec["display_name"] == filename
    expected_head = {
        "dpa31": "Omat24",
        "dpa32": "OMat24",
        "dpa33": "Omat24",
        "dpa4neo": None,
    }[alias]
    assert spec["model_head"] == expected_head


def test_compare_rejects_more_than_three_models(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="at most three"):
        compare_models(
            input_path=Path("examples/Si.vasp"),
            outdir=tmp_path / "compare",
            model_names=["nep89", "dpa31", "dpa32", "dpa33"],
            compute_kappa=False,
            overwrite=True,
            dry_run=True,
        )


def test_compare_rejects_empty_model_list(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="at least one"):
        compare_models(
            input_path=Path("examples/Si.vasp"),
            outdir=tmp_path / "compare",
            model_names=[],
            compute_kappa=False,
            overwrite=True,
            dry_run=True,
        )


def test_compare_child_command_passes_explicit_fc3_target() -> None:
    command = _model_command(
        input_path=Path("examples/Si.vasp"),
        outdir=Path("results/test"),
        backend="dpa31",
        compute_kappa=True,
        relax=False,
        dry_run=True,
        overwrite=True,
        fc3_target_supercell_length=9.5,
    )

    assert command[command.index("--fc3-target-supercell-length") + 1] == "9.5"


def test_compare_child_command_passes_explicit_fc2_displacement() -> None:
    command = _model_command(
        input_path=Path("examples/Si.vasp"),
        outdir=Path("results/test"),
        backend="dpa31",
        compute_kappa=True,
        relax=False,
        dry_run=True,
        overwrite=True,
        displacement=0.012,
    )

    assert command[command.index("--displacement") + 1] == "0.012"


def test_compare_child_command_passes_multitask_model_head() -> None:
    command = _model_command(
        input_path=Path("examples/Si.vasp"),
        outdir=Path("results/test"),
        backend="dpa31",
        model_path=Path("models/DPA-3.1-3M.pt"),
        model_head="Omat24",
        compute_kappa=False,
        relax=False,
        dry_run=True,
        overwrite=True,
    )

    assert command[command.index("--deepmd-model-head") + 1] == "Omat24"
