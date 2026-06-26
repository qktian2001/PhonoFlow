from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from phonoflow.config import WorkflowConfig
from phonoflow.reporting.summary_text import format_summary
from phonoflow.thermal.fc3_finite_displacement import run_finite_displacement_kappa_workflow
from phonoflow.thermal.fc3_hiphive import (
    _annotate_transport_diagnostics,
    canonicalize_fractional_positions_for_hiphive,
    _write_hiphive_diagnostics,
    run_hiphive_kappa_workflow,
)


def test_wigner_missing_plugin_returns_clear_unavailable(monkeypatch, tmp_path: Path) -> None:
    import phonoflow.thermal.fc3_finite_displacement as finite_displacement

    monkeypatch.setattr(
        finite_displacement,
        "get_wte_backend_capability",
        lambda: {
            "available": False,
            "backend": None,
            "transport_type": None,
            "wte_module_found": False,
            "wte_importable": False,
            "phono3py_wte_distribution_version": "0.1.0",
            "phono3py_version": "4.1.0",
            "phonopy_version": "4.1.0",
            "reason": (
                "WTE plugin is not installed on this server. "
                "Ask the service owner to install phono3py-wte."
            ),
        },
    )
    config = WorkflowConfig(
        compute_kappa=True,
        wigner=True,
        fc3_supercell_dim=[1, 1, 1],
        kappa_mesh=[1, 1, 1],
        max_fc3_displacements=1,
    )

    result = run_finite_displacement_kappa_workflow(None, None, config, tmp_path)

    assert result["enabled"] is True
    assert result["available"] is False
    assert result["wigner"] is True
    assert result["kappa_method"] == "rta"
    assert result["kappa_mesh"] == [1, 1, 1]
    assert "phono3py-wte" in result["reason"]
    assert "FC3" not in "\n".join(tmp_path.iterdir().__repr__() for _ in [])


def test_finite_displacement_uses_core_displacement_for_fc2_generation(monkeypatch, tmp_path: Path) -> None:
    import phonoflow.thermal.fc3_finite_displacement as finite_displacement

    calls: dict[str, object] = {}

    class FakeSupercell:
        def __len__(self) -> int:
            return 1

    class FakePhono3py:
        def __init__(self, *args, **kwargs) -> None:
            calls["constructor_kwargs"] = kwargs
            self.supercell = FakeSupercell()
            self.supercells_with_displacements = [object()]
            self.phonon_supercells_with_displacements = [object()]
            self.fc2 = np.zeros((1, 1, 3, 3))
            self.fc3 = np.zeros((1, 1, 1, 3, 3, 3))
            self.mesh_numbers = None

        def generate_displacements(self, **kwargs) -> None:
            calls["fc3_kwargs"] = kwargs

        def generate_fc2_displacements(self, distance: float) -> None:
            calls["fc2_distance"] = distance

        def produce_fc3(self) -> None:
            return None

        def produce_fc2(self) -> None:
            return None

        def init_phph_interaction(self) -> None:
            return None

    fake_phono3py = types.ModuleType("phono3py")
    fake_phono3py.Phono3py = FakePhono3py
    fake_file_io = types.ModuleType("phono3py.file_IO")
    fake_file_io.write_fc2_to_hdf5 = lambda fc2, filename: Path(filename).write_text("fc2", encoding="utf-8")
    fake_file_io.write_fc3_to_hdf5 = lambda fc3, filename: Path(filename).write_text("fc3", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "phono3py", fake_phono3py)
    monkeypatch.setitem(sys.modules, "phono3py.file_IO", fake_file_io)

    monkeypatch.setattr(finite_displacement, "ase_atoms_to_phonopy_atoms", lambda atoms: atoms)
    monkeypatch.setattr(finite_displacement, "_resolve_fc3_supercell_dim", lambda atoms, config: [2, 2, 1])
    monkeypatch.setattr(finite_displacement, "_resolve_kappa_mesh", lambda config: [4, 4, 1])
    monkeypatch.setattr(
        finite_displacement,
        "get_wte_backend_capability",
        lambda: {
            "available": True,
            "backend": None,
            "transport_type": None,
            "wte_module_found": True,
            "phono3py_version": "0",
            "phonopy_version": "0",
            "reason": None,
        },
    )
    monkeypatch.setattr(
        finite_displacement,
        "_evaluate_phono3py_forces",
        lambda supercells, backend, log, label, audit_outdir=None: [np.zeros((1, 3)) for _ in supercells],
    )
    monkeypatch.setattr(
        finite_displacement,
        "_apply_phono3py_symmetrize_fc2",
        lambda phono3py, enabled: {"phono3py_symmetrize_fc2": bool(enabled)},
    )
    monkeypatch.setattr(
        finite_displacement,
        "_save_phono3py_params",
        lambda phono3py, path: path.write_text("params", encoding="utf-8"),
    )
    monkeypatch.setattr(
        finite_displacement,
        "_run_thermal_conductivity_compat",
        lambda phono3py, config, is_lbte, temperatures, transport_type: (
            tmp_path / "kappa-m441-g0.hdf5"
        ).write_text("kappa", encoding="utf-8"),
    )
    monkeypatch.setattr(
        finite_displacement,
        "select_kappa_hdf5_path",
        lambda outdir, mesh: tmp_path / "kappa-m441-g0.hdf5",
    )
    monkeypatch.setattr(
        finite_displacement,
        "parse_kappa_hdf5",
        lambda path: {
            "rows": [
                {
                    "temperature": 300.0,
                    "kappa_xx": 1.0,
                    "kappa_yy": 1.0,
                    "kappa_zz": 1.0,
                    "kappa_avg": 1.0,
                }
            ]
        },
    )
    monkeypatch.setattr(
        finite_displacement,
        "write_thermal_conductivity_csv",
        lambda rows, path: path.write_text("csv", encoding="utf-8"),
    )
    monkeypatch.setattr(
        finite_displacement,
        "plot_thermal_conductivity",
        lambda rows, path, dpi: path.write_text("png", encoding="utf-8"),
    )
    monkeypatch.setattr(
        finite_displacement,
        "extract_lifetime_from_hdf5",
        lambda path, outdir, dpi: {"available": False},
    )
    monkeypatch.setattr(
        finite_displacement,
        "summarize_kappa",
        lambda rows: {"kappa_avg": 1.0},
    )
    monkeypatch.setattr(
        finite_displacement,
        "_write_fd_diagnostics",
        lambda **kwargs: {"files": {}},
    )

    config = WorkflowConfig(
        compute_kappa=True,
        fc3_method="finite-displacement",
        kappa_method="rta",
        displacement=0.012,
        fc3_displacement=0.034,
        temperatures=[300.0],
        kappa_mesh=[4, 4, 1],
        fc3_supercell_dim=[2, 2, 1],
        phono3py_symprec=2e-5,
        phono3py_cutoff_frequency=2e-4,
        phono3py_symmetry=False,
        phono3py_mesh_symmetry=False,
        phono3py_plusminus=False,
        phono3py_diagonal=True,
    )

    run_finite_displacement_kappa_workflow(
        Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[1.0, 1.0, 1.0], pbc=True),
        None,
        config,
        tmp_path,
    )

    assert calls["constructor_kwargs"]["symprec"] == 2e-5
    assert calls["constructor_kwargs"]["cutoff_frequency"] == 2e-4
    assert calls["constructor_kwargs"]["is_symmetry"] is False
    assert calls["constructor_kwargs"]["is_mesh_symmetry"] is False
    assert calls["fc2_distance"] == 0.012
    assert calls["fc3_kwargs"]["distance"] == 0.034
    assert calls["fc3_kwargs"]["is_plusminus"] is False
    assert calls["fc3_kwargs"]["is_diagonal"] is True


def test_run_thermal_conductivity_compat_passes_only_official_runtime_kwargs() -> None:
    import phonoflow.thermal.fc3_finite_displacement as finite_displacement

    calls: dict[str, object] = {}

    class FakePhono3py:
        def run_thermal_conductivity(self, **kwargs) -> None:
            calls["kwargs"] = kwargs

    config = WorkflowConfig(
        compute_kappa=True,
        kappa_method="lbte",
        phono3py_cutoff_frequency=2e-4,
        phono3py_mesh_symmetry=False,
        phono3py_isotope=True,
        boundary_mfp=123.0,
    )

    finite_displacement._run_thermal_conductivity_compat(
        FakePhono3py(),
        config=config,
        is_lbte=True,
        temperatures=[300.0],
        transport_type="WTE",
    )

    kwargs = calls["kwargs"]
    assert kwargs["is_LBTE"] is True
    assert kwargs["temperatures"] == [300.0]
    assert kwargs["transport_type"] == "WTE"
    assert kwargs["is_isotope"] is True
    assert kwargs["boundary_mfp"] == 123.0
    assert kwargs["write_kappa"] is True
    assert "cutoff_frequency" not in kwargs
    assert "is_mesh_symmetry" not in kwargs


def test_hiphive_uses_phono3py_constructor_and_runtime_parameter_layers(monkeypatch, tmp_path: Path) -> None:
    import phonoflow.thermal.fc3_hiphive as hiphive_workflow

    calls: dict[str, object] = {}

    class FakeClusterSpace:
        def __init__(self, atoms, cutoffs) -> None:
            self.atoms = atoms
            self.cutoffs = cutoffs

    class FakeStructureContainer:
        def __init__(self, cluster_space) -> None:
            self.cluster_space = cluster_space

        def add_structure(self, structure) -> None:
            return None

        def get_fit_data(self):
            return np.eye(3), np.ones(3)

    class FakeOptimizer:
        def __init__(self, fit_data, train_set, test_set, check_condition) -> None:
            self.parameters = np.ones(3)

        def train(self) -> None:
            return None

    class FakeForceConstants:
        def get_fc_array(self, order: int, format: str):
            if order == 2:
                return np.zeros((1, 1, 3, 3))
            return np.zeros((1, 1, 1, 3, 3, 3))

        def write_to_phonopy(self, path: str) -> None:
            Path(path).write_text("fc2", encoding="utf-8")

        def write_to_phono3py(self, path: str) -> None:
            Path(path).write_text("fc3", encoding="utf-8")

    class FakeForceConstantPotential:
        def __init__(self, cluster_space, parameters, metadata) -> None:
            return None

        def write(self, path: str) -> None:
            Path(path).write_text("model", encoding="utf-8")

        def get_force_constants(self, training_supercell):
            return FakeForceConstants()

    class FakePhonopy:
        def __init__(self, *args, **kwargs) -> None:
            self.supercell = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[1.0, 1.0, 1.0], pbc=True)

    class FakePhono3py:
        def __init__(self, *args, **kwargs) -> None:
            calls["constructor_kwargs"] = kwargs
            self.mesh_numbers = None
            self.fc2 = None
            self.fc3 = None

        def symmetrize_fc2(self) -> None:
            raise AssertionError("HiPhive path must not call phono3py.symmetrize_fc2()")

        def symmetrize_fc3(self) -> None:
            raise AssertionError("HiPhive path must not call phono3py.symmetrize_fc3()")

        def init_phph_interaction(self) -> None:
            return None

        def run_thermal_conductivity(self, **kwargs) -> None:
            calls["runtime_kwargs"] = kwargs
            Path("kappa-m221.hdf5").write_text("kappa", encoding="utf-8")

    fake_hiphive = types.ModuleType("hiphive")
    fake_hiphive.ClusterSpace = FakeClusterSpace
    fake_hiphive.ForceConstantPotential = FakeForceConstantPotential
    fake_hiphive.StructureContainer = FakeStructureContainer
    fake_hiphive.enforce_rotational_sum_rules = lambda cluster_space, parameters, rules: parameters
    fake_structure_generation = types.ModuleType("hiphive.structure_generation")
    fake_structure_generation.generate_mc_rattled_structures = lambda atoms, n, std, min_dist, seed: [atoms]
    fake_structure_generation.generate_rattled_structures = lambda atoms, n, std, seed: [atoms]
    fake_utilities = types.ModuleType("hiphive.utilities")
    fake_utilities.prepare_structures = lambda structures, training_supercell, check_permutation=False: structures
    fake_phono3py = types.ModuleType("phono3py")
    fake_phono3py.Phono3py = FakePhono3py
    fake_trainstation = types.ModuleType("trainstation")
    fake_trainstation.Optimizer = FakeOptimizer
    monkeypatch.setitem(sys.modules, "hiphive", fake_hiphive)
    monkeypatch.setitem(sys.modules, "hiphive.structure_generation", fake_structure_generation)
    monkeypatch.setitem(sys.modules, "hiphive.utilities", fake_utilities)
    monkeypatch.setitem(sys.modules, "phono3py", fake_phono3py)
    monkeypatch.setitem(sys.modules, "trainstation", fake_trainstation)

    monkeypatch.setattr(hiphive_workflow, "Phonopy", FakePhonopy)
    monkeypatch.setattr(hiphive_workflow, "ase_atoms_to_phonopy_atoms", lambda atoms: atoms)
    monkeypatch.setattr(hiphive_workflow, "phonopy_atoms_to_ase_atoms", lambda atoms: atoms)
    monkeypatch.setattr(hiphive_workflow, "canonicalize_fractional_positions_for_hiphive", lambda atoms: (atoms, None))
    monkeypatch.setattr(hiphive_workflow, "_resolve_fc3_supercell_dim", lambda atoms, config: [1, 1, 1])
    monkeypatch.setattr(hiphive_workflow, "_resolve_kappa_mesh", lambda config: [2, 2, 1])
    monkeypatch.setattr(
        hiphive_workflow,
        "get_wte_backend_capability",
        lambda: {
            "available": True,
            "backend": None,
            "transport_type": None,
            "wte_module_found": True,
            "phono3py_version": "4.1.0",
            "phonopy_version": "4.1.0",
            "reason": None,
        },
    )
    monkeypatch.setattr(
        hiphive_workflow,
        "_write_hiphive_diagnostics",
        lambda **kwargs: {
            "files": {},
            "force_rmse_input_eV_per_A": 0.0,
            "force_rmse_train_eV_per_A": 0.0,
            "max_force_error_train_eV_per_A": 0.0,
            "number_of_force_components": 3,
            "number_of_fit_parameters": 3,
            "underdetermined": False,
        },
    )
    monkeypatch.setattr(hiphive_workflow, "inspect_kappa_hdf5", lambda path: {})
    monkeypatch.setattr(hiphive_workflow, "_annotate_transport_diagnostics", lambda **kwargs: kwargs["diagnostics"])
    monkeypatch.setattr(hiphive_workflow, "select_kappa_hdf5_path", lambda outdir, mesh: tmp_path / "kappa-m221.hdf5")
    monkeypatch.setattr(
        hiphive_workflow,
        "parse_kappa_hdf5",
        lambda path: {
            "rows": [
                {
                    "temperature": 300.0,
                    "kappa_xx": 2.0,
                    "kappa_yy": 2.0,
                    "kappa_zz": 2.0,
                    "kappa_avg": 2.0,
                }
            ]
        },
    )
    monkeypatch.setattr(hiphive_workflow, "write_thermal_conductivity_csv", lambda rows, path: path.write_text("csv", encoding="utf-8"))
    monkeypatch.setattr(hiphive_workflow, "plot_thermal_conductivity", lambda rows, path, dpi: path.write_text("png", encoding="utf-8"))
    monkeypatch.setattr(hiphive_workflow, "extract_lifetime_from_hdf5", lambda path, outdir, dpi: {"available": False})
    monkeypatch.setattr(hiphive_workflow, "summarize_kappa", lambda rows: {"kappa_avg": 2.0})

    class FakeBackend:
        def calculate_energy_forces(self, atoms):
            return {"forces": np.zeros((len(atoms), 3))}

    config = WorkflowConfig(
        compute_kappa=True,
        fc3_method="hiphive",
        kappa_method="lbte",
        temperatures=[300.0],
        phono3py_symprec=3e-5,
        phono3py_cutoff_frequency=4e-4,
        phono3py_symmetry=False,
        phono3py_mesh_symmetry=False,
        phono3py_isotope=True,
        boundary_mfp=321.0,
        n_structures=1,
        cutoffs=[3.0, 2.0],
    )

    result = run_hiphive_kappa_workflow(
        Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[1.0, 1.0, 1.0], pbc=True),
        FakeBackend(),
        config,
        tmp_path,
    )

    assert result["available"] is True
    assert calls["constructor_kwargs"]["symprec"] == 3e-5
    assert calls["constructor_kwargs"]["cutoff_frequency"] == 4e-4
    assert calls["constructor_kwargs"]["is_symmetry"] is False
    assert calls["constructor_kwargs"]["is_mesh_symmetry"] is False
    assert calls["runtime_kwargs"]["is_LBTE"] is True
    assert calls["runtime_kwargs"]["is_isotope"] is True
    assert calls["runtime_kwargs"]["boundary_mfp"] == 321.0
    assert "cutoff_frequency" not in calls["runtime_kwargs"]
    assert "is_mesh_symmetry" not in calls["runtime_kwargs"]
    assert result["phono3py_symprec"] == 3e-5
    assert result["phono3py_cutoff_frequency"] == 4e-4
    assert result["phono3py_symmetry"] is False
    assert result["phono3py_mesh_symmetry"] is False
    assert result["phono3py_symmetrize_fc2"] is False
    assert result["phono3py_symmetrize_fc2_requested"] is True
    assert result["phono3py_symmetrize_fc2_applied"] is False
    assert result["phono3py_symmetrize_fc3"] is False
    assert result["phono3py_symmetrize_fc3_requested"] is True
    assert result["phono3py_symmetrize_fc3_applied"] is False
    assert result["hiphive_uses_phono3py_symmetrize"] is False
    assert result["hiphive_rotational_sum_rules"] == ["Huang", "Born-Huang"]


def test_hiphive_experimental_parameters_are_recorded() -> None:
    config = WorkflowConfig(
        compute_kappa=True,
        fc3_method="hiphive",
        kappa_method="rta",
        wigner=False,
        temperatures=[300.0, 600.0],
        kappa_mesh=[3, 3, 3],
        fc3_supercell_dim=[2, 2, 2],
        fc3_displacement=0.02,
        max_fc3_displacements=4,
        n_structures=12,
        rattle_std=0.015,
        cutoffs=[4.5, 3.2],
        min_dist=1.7,
    )

    result = run_hiphive_kappa_workflow(None, None, config, Path("unused"))

    assert result["enabled"] is True
    assert result["available"] is False
    assert result["fc3_method"] == "hiphive"
    assert result["kappa_method"] == "rta"
    assert result["temperatures"] == [300.0, 600.0]
    assert result["kappa_mesh"] == [3, 3, 3]
    assert result["fc3_supercell_dim"] == [2, 2, 2]
    assert result["max_fc3_displacements"] == 4
    assert result["smoke_test"] is True
    assert result["experimental_parameters"] == {
        "n_structures": 12,
        "rattle_std": 0.015,
        "cutoffs": [4.5, 3.2],
        "min_dist": 1.7,
    }


def test_summary_reports_unavailable_thermal_options() -> None:
    result = {
        "output_files": {},
        "thermal_conductivity": {
            "enabled": True,
            "available": False,
            "fc3_method": "hiphive",
            "kappa_method": "rta",
            "wigner": False,
            "temperatures": [300.0],
            "kappa_mesh": [3, 3, 3],
            "fc3_supercell_dim": [2, 2, 2],
            "fc3_displacement": 0.03,
            "fc3_cutoff_pair_distance": None,
            "max_fc3_displacements": 2,
            "smoke_test": True,
            "reason": "HiPhive fit failed.",
            "warnings": ["fit failed"],
            "experimental_parameters": {
                "n_structures": 10,
                "rattle_std": 0.02,
                "cutoffs": [5.0, 4.0],
                "min_dist": 1.8,
            },
        },
    }

    summary = format_summary(result)

    assert "Wigner: False" in summary
    assert "Kappa mesh: [3, 3, 3]" in summary
    assert "Experimental parameters:" in summary
    assert "n_structures" in summary


def test_summary_reports_available_hiphive_outputs() -> None:
    result = {
        "output_files": {},
        "thermal_conductivity": {
            "enabled": True,
            "available": True,
            "fc3_method": "hiphive",
            "kappa_method": "rta",
            "wigner": False,
            "wigner_requested": False,
            "wigner_available": False,
            "wigner_backend": None,
            "wte_plugin_found": True,
            "wte_module_found": True,
            "transport_type": None,
            "phono3py_version": "4.1.0",
            "phonopy_version": "4.1.0",
            "thermal_status": "available",
            "temperatures": [300.0],
            "kappa_mesh": [1, 1, 1],
            "fc3_supercell_dim": [2, 2, 2],
            "n_fc3_displacements": 4,
            "n_fc2_displacements": 4,
            "smoke_test": False,
            "hiphive_status": "available",
            "hiphive_available": True,
            "hiphive_n_structures": 4,
            "hiphive_rattle_std": 0.01,
            "hiphive_cutoffs": [3.5, 2.8],
            "hiphive_min_dist": 1.8,
            "hiphive_fit_matrix_shape": [192, 5],
            "hiphive_n_parameters": 5,
            "hiphive_force_rmse_train_eV_per_A": 0.02,
            "hiphive_max_force_error_train_eV_per_A": 0.08,
            "hiphive_number_of_force_components": 192,
            "hiphive_number_of_fit_parameters": 5,
            "hiphive_underdetermined": False,
            "files": {
                "fc2_hdf5": "fc2.hdf5",
                "fc3_hdf5": "fc3.hdf5",
                "kappa_hdf5": "kappa-m111-g0.hdf5",
                "thermal_conductivity_csv": "thermal_conductivity.csv",
                "thermal_conductivity_png": "thermal_conductivity.png",
                "hiphive_fit_summary": "hiphive_fit_summary.json",
                "hiphive_fit_diagnostics_json": "hiphive_fit_diagnostics.json",
                "hiphive_force_fit_plot": "hiphive_force_fit.png",
                "fc2_diagnostics_json": "fc2_diagnostics.json",
                "fc3_diagnostics_json": "fc3_diagnostics.json",
                "phonon_lifetime_csv": "phonon_lifetime.csv",
                "phonon_lifetime_png": "phonon_lifetime.png",
            },
            "kappa_unit": "W/m-K",
            "lifetime": {
                "available": True,
                "data_file": "phonon_lifetime.csv",
                "plot_file": "phonon_lifetime.png",
                "unit": "ps",
                "source": "gamma",
            },
            "warnings": [],
        },
    }

    summary = format_summary(result)

    assert "FC3 method: hiphive" in summary
    assert "HiPhive status: available" in summary
    assert "HiPhive structures: 4" in summary
    assert "HiPhive fit summary: hiphive_fit_summary.json" in summary
    assert "HiPhive force RMSE train: 0.02 eV/A" in summary
    assert "HiPhive underdetermined: False" in summary
    assert "HiPhive fit diagnostics: hiphive_fit_diagnostics.json" in summary
    assert "FC2 diagnostics: fc2_diagnostics.json" in summary
    assert "Lifetime: available" in summary


def test_hiphive_diagnostics_detect_underdetermined_fit(tmp_path: Path) -> None:
    class FakeSupercell:
        def __len__(self) -> int:
            return 2

    class FakeOrbit:
        def __init__(self, order: int) -> None:
            self.order = order

    class FakeClusterSpace:
        orbits = [FakeOrbit(2), FakeOrbit(3), FakeOrbit(3)]

    class FakeOptimizer:
        rmse_train = None

    config = WorkflowConfig(
        compute_kappa=True,
        n_structures=2,
        rattle_std=0.01,
        cutoffs=[3.5, 2.8],
        min_dist=1.8,
        fc3_supercell_dim=[1, 1, 1],
        kappa_mesh=[1, 1, 1],
    )
    fit_matrix = __import__("numpy").eye(2, 3)
    fit_targets = __import__("numpy").array([1.0, -1.0])
    parameters = __import__("numpy").array([1.0, -1.0, 0.5])
    fc2 = __import__("numpy").zeros((2, 2, 3, 3))
    fc3 = __import__("numpy").zeros((2, 2, 2, 3, 3, 3))

    diagnostics = _write_hiphive_diagnostics(
        outdir=tmp_path,
        config=config,
        fc3_supercell_dim=[1, 1, 1],
        kappa_mesh=[1, 1, 1],
        training_supercell=FakeSupercell(),
        cluster_space=FakeClusterSpace(),
        fit_matrix=fit_matrix,
        fit_targets=fit_targets,
        parameters=parameters,
        optimizer=FakeOptimizer(),
        fc2=fc2,
        fc3=fc3,
        force_rmse_samples=[0.1, 0.2],
        warnings=[],
    )

    assert diagnostics["number_of_force_components"] == 2
    assert diagnostics["number_of_fit_parameters"] == 3
    assert diagnostics["underdetermined"] is True
    assert diagnostics["cluster_counts_by_order"] == {"2": 1, "3": 2}
    assert diagnostics["method"] == "hiphive"
    assert diagnostics["single_point_calculator"] is True
    assert diagnostics["optimizer_source"] == "StructureContainer.get_fit_data()"
    assert diagnostics["fc2_export_method"] == "ForceConstants.write_to_phonopy"
    assert diagnostics["fc3_export_method"] == "ForceConstants.write_to_phono3py"
    assert "cutoffs[0] is second-order" in diagnostics["cutoffs_order_note"]
    assert "eV/Angstrom" in diagnostics["force_units_note"]
    assert diagnostics["fc2_diagnostics"]["method"] == "hiphive"
    assert diagnostics["fc3_diagnostics"]["export_note"].startswith("Exported through HiPhive")
    for filename in diagnostics["files"].values():
        assert (tmp_path / filename).exists()

    annotated = _annotate_transport_diagnostics(
        outdir=tmp_path,
        diagnostics=diagnostics,
        kappa_diagnostics={
            "source_file": "kappa-m222.hdf5",
            "fields_found": ["frequency", "gamma", "grid_point", "kappa"],
            "fields": {"gamma": {"shape": [1, 2, 2], "dtype": "float64"}},
        },
        lifetime={"available": True, "source": "gamma", "warnings": ["converted"]},
    )

    assert annotated["kappa_hdf5_fields_found"] == ["frequency", "gamma", "grid_point", "kappa"]
    assert annotated["lifetime_source"] == "gamma"
    payload = json.loads((tmp_path / "hiphive_fit_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["kappa_hdf5"]["fields"]["gamma"]["shape"] == [1, 2, 2]


def test_hiphive_canonicalizes_fractional_boundary_positions() -> None:
    pytest.importorskip("hiphive")
    from hiphive import ClusterSpace

    atoms = Atoms(
        symbols=["Ta", "N"],
        cell=[
            [2.9804958517464923, 2.72579e-11, -1.0e-16],
            [-1.4902479259003223, 2.5811851234709993, 1.0e-16],
            [-1.0e-16, 1.0e-16, 2.825177081184042],
        ],
        scaled_positions=[
            [1.6665299999999998e-09, 0.9999999983334658, 0.5],
            [0.6666666683334675, 0.33333333166653223, 0.9999999999999987],
        ],
        pbc=True,
    )

    canonical, warning = canonicalize_fractional_positions_for_hiphive(atoms)

    scaled = canonical.get_scaled_positions(wrap=False)
    assert np.allclose(scaled[0], [0.0, 0.0, 0.5], atol=1.0e-6)
    assert np.allclose(scaled[1], [0.6666666683334675, 0.33333333166653223, 0.0], atol=1.0e-6)
    assert warning is not None
    ClusterSpace(canonical, [5.0, 4.0])
