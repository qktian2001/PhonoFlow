from pathlib import Path

from ase import Atoms
from ase.build import bulk

from phonoflow.analysis.spacegroup import (
    analyze_spacegroup,
    build_spacegroup_report,
    write_spacegroup_report,
)


def test_spacegroup_bulk_si():
    result = analyze_spacegroup(bulk("Si", "diamond", a=5.43))
    assert result["dataset_available"] is True
    assert result["spacegroup_number"] is not None
    assert result["international_symbol"]


def test_spacegroup_failure_safe():
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=[0, 0, 0], pbc=True)
    result = analyze_spacegroup(atoms)
    assert result["dataset_available"] is False
    assert result["error"]


def test_spacegroup_report_writer(tmp_path: Path):
    initial = analyze_spacegroup(bulk("Si", "diamond", a=5.43))
    final = dict(initial)
    report = build_spacegroup_report(initial, final, symprec=1e-3, angle_tolerance=-1.0)
    json_path = tmp_path / "spacegroup_report.json"
    text_path = tmp_path / "spacegroup_report.txt"
    write_spacegroup_report(report, json_path, text_path)
    assert json_path.exists() and json_path.stat().st_size > 0
    assert text_path.exists() and "Space Group Report" in text_path.read_text(encoding="utf-8")
    assert report["changed"] is False
