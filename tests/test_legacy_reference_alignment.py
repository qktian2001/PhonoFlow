from __future__ import annotations

import os
from pathlib import Path

import pytest


ROOT = Path(os.environ.get("PHONOFLOW_LEGACY_REFERENCE_TREE", "legacy_reference"))


@pytest.fixture()
def legacy_reference_tree() -> Path:
    if not ROOT.exists():
        pytest.skip("Set PHONOFLOW_LEGACY_REFERENCE_TREE to run legacy reference checks.")
    assert (ROOT / "workflow.py").exists()
    assert (ROOT / "nepkappa.py").exists()
    return ROOT


def test_legacy_reference_tree_is_present_and_read_only_fixture(legacy_reference_tree: Path) -> None:
    assert legacy_reference_tree.exists()


def test_legacy_reference_finite_displacement_uses_traditional_produce_fc_calls(
    legacy_reference_tree: Path,
) -> None:
    text = (legacy_reference_tree / "workflow.py").read_text()
    assert "ph3.produce_fc2()" in text
    assert "ph3.produce_fc3()" in text
    assert "symmetrize_fc2" not in text
    assert "symmetrize_fc3" not in text


def test_legacy_reference_hiphive_path_is_separate_from_phono3py_traditional_solver(
    legacy_reference_tree: Path,
) -> None:
    text = (legacy_reference_tree / "workflow.py").read_text()
    hiphive_index = text.index("def run_hiphive_fitting")
    finite_index = text.index("def run_finite_disp_fitting")
    hiphive_block = text[hiphive_index:finite_index]
    assert "ClusterSpace" in hiphive_block
    assert "ForceConstantPotential" in hiphive_block
    assert "fcs.write_to_phono3py" in hiphive_block
    assert "ph3.produce_fc2" not in hiphive_block
    assert "ph3.produce_fc3" not in hiphive_block


def test_legacy_reference_method_and_mesh_mapping_are_explicit(legacy_reference_tree: Path) -> None:
    text = (legacy_reference_tree / "workflow.py").read_text()
    assert 'method_flags = ["--lbte"]' in text
    assert 'method_flags = ["--br", "--nu"]' in text
    assert '"--mesh", str(mx), str(my), str(mz)' in text
