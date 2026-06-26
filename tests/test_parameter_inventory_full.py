from __future__ import annotations

from pathlib import Path


def test_inventory_contains_single_run_compare_and_required_fields() -> None:
    cli_text = Path("docs/cli.md").read_text(encoding="utf-8")
    config_text = Path("docs/configuration.md").read_text(encoding="utf-8")

    assert "phonoflow single" in cli_text
    assert "phonoflow run" in cli_text
    assert "phonoflow compare-models" in cli_text

    required_fields = {
        "input_path",
        "model_path",
        "backend",
        "supercell_dim",
        "mesh",
        "compute_kappa",
        "phono3py_symmetrize_fc2",
        "phono3py_symmetrize_fc3",
    }
    for field in required_fields:
        assert f"`{field}`" in config_text


def test_deprecated_phono3py_fc2_asr_is_not_formal_parameter() -> None:
    text = Path("docs/configuration.md").read_text(encoding="utf-8")
    inventory = text.split("## Accepted Aliases", 1)[0]
    assert "`phono3py_symmetrize_fc2`" in inventory
    assert "phono3py_fc2_asr" not in inventory
    assert "`phono3py_fc2_asr` is accepted as a deprecated alias" in text
