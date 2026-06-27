"""Automatic default inference for user-friendly single workflows."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

from phonoflow.analysis.structure_type import classify_structure_type
from phonoflow.calculators.calorine_backend import CalorineBackend
from phonoflow.config import WorkflowConfig, resolve_common_q_mesh
from phonoflow.exceptions import BackendUnavailableError, ConfigError
from phonoflow.io.path_utils import safe_stem


DPA_DEFAULT_DISPLACEMENT = 0.03
DPA_DEFAULT_PHONO3PY_SYMPREC = 1e-5
DPA_PHONO3PY_SYMMETRIZE_FC2_DEFAULT = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DPA_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_NEP89_MODEL_PATH = PROJECT_ROOT / "nep89_potential" / "nep89_20250409.txt"


@dataclass(frozen=True)
class DPAModelResolution:
    """Resolved DPA model metadata."""

    backend_alias: str
    model_name: str | None
    model_path: Path
    model_head: str | None = None


DPA_MODEL_ALIASES = {
    "dpa3": "dpa32",
    "dpa4": "dpa4neo",
}
DPA_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "dpa31": {
        "model_name": "DPA-3.1-3M.pt",
        "filename": "DPA-3.1-3M.pt",
        "model_head": "Omat24",
    },
    "dpa32": {
        "model_name": "DPA-3.2-5M.pt",
        "filename": "DPA-3.2-5M.pt",
        "model_head": "OMat24",
    },
    "dpa33": {
        "model_name": "DPA-3.3-1M.pt",
        "filename": "DPA-3.3-1M.pt",
        "model_head": "Omat24",
    },
    "dpa4neo": {
        "model_name": "DPA4-Neo-OMat24-v20260528_rc.pt",
        "filename": "DPA4-Neo-OMat24-v20260528_rc.pt",
        "model_head": None,
    },
}
DPA_BACKEND_ALIASES = {"dpa", *DPA_MODEL_ALIASES, *DPA_MODEL_REGISTRY}
DEEPMD_MODEL_SUFFIXES = {".pb", ".pt", ".pth"}


def canonical_dpa_alias(alias: str) -> str:
    """Return the current canonical alias for a DPA backend name."""

    normalized = str(alias).strip().lower()
    return DPA_MODEL_ALIASES.get(normalized, normalized)


def dpa_alias_from_model_path(model_path: Path) -> str | None:
    """Return a registered DPA alias when a model filename is recognized."""

    filename = Path(model_path).name.lower()
    for alias, spec in DPA_MODEL_REGISTRY.items():
        if filename == str(spec["filename"]).lower():
            return alias
    return None


def is_deepmd_model_path(model_path: Path | None) -> bool:
    """Return True when a model path looks like a DeepMD/DPA model file."""

    if model_path is None:
        return False
    return Path(model_path).suffix.lower() in DEEPMD_MODEL_SUFFIXES


def discover_bundled_dpa_models() -> list[dict[str, Any]]:
    """Return availability and exact paths for the four supported bundled models."""

    return [
        {
            "alias": alias,
            "model_name": str(spec["model_name"]),
            "filename": str(spec["filename"]),
            "model_head": spec.get("model_head"),
            "path": str(DEFAULT_DPA_MODEL_DIR / str(spec["filename"])),
            "available": (DEFAULT_DPA_MODEL_DIR / str(spec["filename"])).is_file(),
        }
        for alias, spec in DPA_MODEL_REGISTRY.items()
    ]


def resolve_backend_name(requested: str) -> str:
    """Resolve backend='auto' without silently falling back to dummy."""

    normalized = requested.lower()
    if normalized == "pynep":
        raise ConfigError(
            "Unsupported backend: pynep. PyNEP backend has been removed. Please use "
            "backend=calorine for real NEP/NEP89 calculations or backend=dummy for tests."
        )
    if normalized != "auto":
        if normalized in DPA_BACKEND_ALIASES:
            return "deepmd"
        return normalized

    if CalorineBackend().check_available():
        return "calorine"
    raise BackendUnavailableError(
        "backend='auto' resolves to Calorine, but Calorine CPUNEP is not available. "
        "Calorine is required for real NEP/NEP89 calculations. Install it with:\n"
        "python -m pip install calorine\n"
        "Use --backend dummy only for workflow tests."
    )


def resolve_dpa_model_path(alias: str, model_path: Path | None) -> DPAModelResolution:
    """Resolve a DPA/DPA3/DPA4 alias to a model path and user-facing model name."""

    normalized = str(alias).lower()
    if normalized not in DPA_BACKEND_ALIASES:
        raise ConfigError(f"'{alias}' is not a DPA backend alias.")

    if model_path is not None:
        path = Path(model_path)
        canonical = canonical_dpa_alias(normalized)
        model_name = DPA_MODEL_REGISTRY.get(canonical, {}).get("model_name") or path.name
        return DPAModelResolution(
            backend_alias=normalized,
            model_name=model_name,
            model_path=path,
            model_head=DPA_MODEL_REGISTRY.get(canonical, {}).get("model_head"),
        )

    if normalized == "dpa":
        raise ConfigError(
            "backend='dpa' is generic and requires --model-path. "
            "Use --backend dpa31, dpa32, dpa33, or dpa4neo to select a bundled model."
        )

    canonical = canonical_dpa_alias(normalized)
    registry = DPA_MODEL_REGISTRY[canonical]
    filename = str(registry["filename"])
    candidate = DEFAULT_DPA_MODEL_DIR / filename
    if candidate.is_file():
        return DPAModelResolution(
            backend_alias=canonical if normalized not in DPA_MODEL_ALIASES else normalized,
            model_name=str(registry["model_name"]),
            model_path=candidate,
            model_head=registry.get("model_head"),
        )

    raise ConfigError(
        f"backend='{normalized}' could not find its bundled DPA model under {DEFAULT_DPA_MODEL_DIR}. "
        f"Expected: {filename}. Provide --model-path to use a custom model."
    )


def infer_default_config(
    atoms: Any,
    input_path: Path,
    model_path: Path | None,
    user_config: WorkflowConfig,
) -> WorkflowConfig:
    """Resolve auto/default workflow settings from structure and user config."""

    requested_backend = user_config.backend.lower()
    auto_dpa_resolution: DPAModelResolution | None = None
    if requested_backend == "auto" and is_deepmd_model_path(model_path):
        auto_dpa_alias = dpa_alias_from_model_path(Path(model_path)) or "dpa"
        auto_dpa_resolution = resolve_dpa_model_path(auto_dpa_alias, model_path)
        backend_resolved = "deepmd"
    else:
        backend_resolved = resolve_backend_name(requested_backend)
    backend_alias = user_config.backend_alias or requested_backend
    dpa_model_name = user_config.dpa_model_name
    deepmd_model_head = user_config.deepmd_model_head
    resolved_model_path = model_path
    if requested_backend in DPA_BACKEND_ALIASES or auto_dpa_resolution is not None:
        dpa_resolution = auto_dpa_resolution or resolve_dpa_model_path(requested_backend, model_path)
        backend_alias = dpa_resolution.backend_alias
        dpa_model_name = dpa_resolution.model_name
        resolved_model_path = dpa_resolution.model_path
        if deepmd_model_head is None:
            deepmd_model_head = dpa_resolution.model_head
    outdir = user_config.outdir
    if outdir is None:
        outdir_label = backend_alias if backend_resolved == "deepmd" else backend_resolved
        outdir = Path("results") / f"{safe_stem(input_path)}_{outdir_label}"

    option_sources = set(user_config.option_sources)
    dpa_requested = requested_backend in DPA_BACKEND_ALIASES or auto_dpa_resolution is not None
    structure_classification = classify_structure_type(atoms)
    supercell_dim = user_config.supercell_dim
    supercell_info: dict[str, Any] = {}
    if supercell_dim == "auto":
        supercell_info = infer_supercell_info(
            atoms,
            target_supercell_length=user_config.target_supercell_length,
            min_dim=user_config.min_supercell_dim,
            max_dim=user_config.max_supercell_dim,
            max_supercell_atoms=user_config.max_supercell_atoms,
            vacuum_like_directions=structure_classification.get("vacuum_like_directions", []),
        )
        supercell_dim = supercell_info["supercell_dim"]
    else:
        supercell_info = build_supercell_info(
            atoms,
            list(supercell_dim),
            target_supercell_length=user_config.target_supercell_length,
            min_dim=user_config.min_supercell_dim,
            max_dim=user_config.max_supercell_dim,
            max_supercell_atoms=user_config.max_supercell_atoms,
            source="user",
        )

    mesh = resolve_common_q_mesh(
        user_config.mesh,
        user_config.kappa_mesh,
        vacuum_like_directions=structure_classification.get("vacuum_like_directions", []),
    )

    if backend_resolved != "dummy" and resolved_model_path is None:
        raise ConfigError(f"backend='{backend_resolved}' requires --model-path.")

    updates: dict[str, Any] = {
        "backend": backend_resolved,
        "backend_alias": backend_alias,
        "dpa_model_name": dpa_model_name,
        "deepmd_model_head": deepmd_model_head,
        "model_path": resolved_model_path,
        "outdir": outdir,
        "supercell_dim": supercell_dim,
        "supercell_info": supercell_info,
        "mesh": mesh,
        "kappa_mesh": mesh,
    }
    if dpa_requested:
        if "relax" not in option_sources:
            updates["relax"] = False
        if "deepmd_reuse_calculator" not in option_sources:
            updates["deepmd_reuse_calculator"] = True
        if "deepmd_deterministic" not in option_sources:
            updates["deepmd_deterministic"] = True
        if "save_force_audit" not in option_sources:
            updates["save_force_audit"] = True
        if "phono3py_symmetrize_fc2" not in option_sources:
            updates["phono3py_symmetrize_fc2"] = DPA_PHONO3PY_SYMMETRIZE_FC2_DEFAULT
        effective_relax = bool(updates.get("relax", user_config.relax))
        if effective_relax and user_config.relax_backend == "auto" and user_config.relax_model_path is None:
            updates["relax_model_path"] = DEFAULT_NEP89_MODEL_PATH

    return user_config.model_copy(update=updates)


def infer_supercell_dim(
    atoms: Any,
    target_supercell_length: float = 15.0,
    min_dim: int = 1,
    max_dim: int = 6,
    max_supercell_atoms: int = 1000,
    vacuum_like_directions: list[str] | None = None,
) -> list[int]:
    """Infer a finite-displacement supercell dimension from cell lengths."""

    return infer_supercell_info(
        atoms,
        target_supercell_length=target_supercell_length,
        min_dim=min_dim,
        max_dim=max_dim,
        max_supercell_atoms=max_supercell_atoms,
        vacuum_like_directions=vacuum_like_directions,
    )["supercell_dim"]


def infer_supercell_info(
    atoms: Any,
    target_supercell_length: float = 15.0,
    min_dim: int = 1,
    max_dim: int = 6,
    max_supercell_atoms: int = 1000,
    vacuum_like_directions: list[str] | None = None,
) -> dict[str, Any]:
    """Infer supercell dimensions and return reproducibility diagnostics."""

    lengths = np.asarray(atoms.cell.lengths(), dtype=float)
    axis_map = {"a": 0, "b": 1, "c": 2}
    vacuum_axes = {axis_map[axis] for axis in (vacuum_like_directions or []) if axis in axis_map}
    notes: list[str] = []
    warnings: list[str] = []
    dims = []
    for axis, length in enumerate(lengths):
        if axis in vacuum_axes:
            dims.append(1)
            notes.append(
                "Vacuum-like direction detected; auto supercell dimension along "
                f"{['a', 'b', 'c'][axis]} was kept at 1."
            )
            continue
        if length <= 0:
            dims.append(min_dim)
        else:
            dims.append(int(np.clip(ceil(target_supercell_length / length), min_dim, max_dim)))

    n_atoms = len(atoms)
    initial_dims = list(dims)
    while int(np.prod(dims)) * n_atoms > max_supercell_atoms and max(dims) > min_dim:
        largest_index = int(np.argmax(dims))
        dims[largest_index] -= 1

    supercell_lengths = (lengths * np.asarray(dims, dtype=float)).tolist()
    if dims != initial_dims:
        short = any(
            length < 0.8 * target_supercell_length
            for axis, length in enumerate(supercell_lengths)
            if axis not in vacuum_axes
        )
        if short:
            warnings.append(
                "Auto supercell was capped by max_supercell_atoms. The final supercell "
                "length is smaller than target_supercell_length. Consider increasing "
                "--max-supercell-atoms or manually setting --supercell-dim."
            )

    return {
        "supercell_dim": [int(dim) for dim in dims],
        "initial_supercell_dim": [int(dim) for dim in initial_dims],
        "cell_lengths": [float(value) for value in lengths],
        "supercell_lengths_resolved": [float(value) for value in supercell_lengths],
        "target_supercell_length": float(target_supercell_length),
        "min_supercell_dim": int(min_dim),
        "max_supercell_dim": int(max_dim),
        "max_supercell_atoms": int(max_supercell_atoms),
        "n_atoms_unitcell": int(n_atoms),
        "n_atoms_supercell": int(n_atoms * np.prod(dims)),
        "vacuum_like_directions": list(vacuum_like_directions or []),
        "auto_supercell_warnings": warnings,
        "auto_supercell_notes": notes,
        "source": "auto",
    }


def build_supercell_info(
    atoms: Any,
    supercell_dim: list[int],
    target_supercell_length: float,
    min_dim: int,
    max_dim: int,
    max_supercell_atoms: int,
    source: str,
) -> dict[str, Any]:
    """Build diagnostics for a user-provided supercell."""

    lengths = np.asarray(atoms.cell.lengths(), dtype=float)
    dims = [int(value) for value in supercell_dim]
    return {
        "supercell_dim": dims,
        "initial_supercell_dim": dims,
        "cell_lengths": [float(value) for value in lengths],
        "supercell_lengths_resolved": [float(value) for value in lengths * np.asarray(dims, dtype=float)],
        "target_supercell_length": float(target_supercell_length),
        "min_supercell_dim": int(min_dim),
        "max_supercell_dim": int(max_dim),
        "max_supercell_atoms": int(max_supercell_atoms),
        "n_atoms_unitcell": int(len(atoms)),
        "n_atoms_supercell": int(len(atoms) * np.prod(dims)),
        "vacuum_like_directions": [],
        "auto_supercell_warnings": [],
        "auto_supercell_notes": [],
        "source": source,
    }
