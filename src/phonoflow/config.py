"""Workflow configuration models and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from phonoflow.analysis.bandpath import DEFAULT_SEEKPATH_SYMPREC, DEFAULT_SEEKPATH_WITH_TIME_REVERSAL
from phonoflow.exceptions import ConfigError


BackendName = Literal[
    "auto",
    "dummy",
    "calorine",
    "gpumd",
    "deepmd",
    "dpa",
    "dpa3",
    "dpa4",
    "dpa31",
    "dpa32",
    "dpa33",
    "dpa4neo",
]
AutoOrTriplet = list[int] | Literal["auto"]
PrimitiveMatrixName = Literal["auto", "P", "identity", "none"]
ForceConstantsMethod = Literal["finite-displacement", "hiphive"]
FC3Method = Literal["finite-displacement", "hiphive"]
KappaMethod = Literal["rta", "lbte"]
DeepMDForceBackend = Literal["ase", "deeppot"]
DeepMDDevice = Literal["auto", "cpu", "cuda"]
Phono3pyPlusMinus = Literal["auto", "true", "false"]
KPathMode = Literal["auto", "3d_seekpath", "2d_ase", "custom"]
DEFAULT_Q_MESH = [21, 21, 21]
DEFAULT_2D_Q_MESH_IN_PLANE = 51
_Q_MESH_AXIS_BY_VACUUM_DIRECTION = {"a": 0, "b": 1, "c": 2}


def default_q_mesh() -> list[int]:
    """Return the shared DOS/kappa gamma-centered q-mesh default."""

    return list(DEFAULT_Q_MESH)


def default_q_mesh_for_structure(vacuum_like_directions: list[str] | None = None) -> list[int]:
    """Return the structure-aware shared DOS/kappa q-mesh default.

    Bulk/3D systems keep the historical 21x21x21 default. A single
    vacuum-like direction is treated as a 2D slab: the periodic in-plane axes
    use a denser 51-point default and the vacuum axis is sampled once, so a
    common c-axis slab resolves to 51x51x1.
    """

    vacuum_directions = [
        str(direction).lower()
        for direction in (vacuum_like_directions or [])
        if str(direction).lower() in _Q_MESH_AXIS_BY_VACUUM_DIRECTION
    ]
    mesh = (
        [DEFAULT_2D_Q_MESH_IN_PLANE, DEFAULT_2D_Q_MESH_IN_PLANE, DEFAULT_2D_Q_MESH_IN_PLANE]
        if len(vacuum_directions) == 1
        else default_q_mesh()
    )
    for direction in vacuum_directions:
        axis = _Q_MESH_AXIS_BY_VACUUM_DIRECTION.get(str(direction).lower())
        if axis is not None:
            mesh[axis] = 1
    return mesh


def _is_explicit_triplet_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        return bool(normalized) and normalized != "auto"
    return True


def resolve_common_q_mesh(
    mesh: AutoOrTriplet | None,
    kappa_mesh: AutoOrTriplet | None,
    vacuum_like_directions: list[str] | None = None,
) -> list[int]:
    """Resolve one shared q-mesh from harmonic and thermal compatibility fields."""

    source = (
        mesh
        if isinstance(mesh, list)
        else kappa_mesh
        if isinstance(kappa_mesh, list)
        else default_q_mesh_for_structure(vacuum_like_directions)
    )
    return [int(value) for value in source]


class WorkflowConfig(BaseModel):
    """Configuration shared by single and batch workflow skeletons."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    input_path: Path | None = None
    input_dir: Path | None = None
    outdir: Path | None = None
    model_path: Path | None = None
    backend: BackendName = "auto"
    backend_alias: str | None = None
    dpa_model_name: str | None = None
    supercell_dim: AutoOrTriplet = "auto"
    mesh: AutoOrTriplet = "auto"
    target_supercell_length: float = 15.0
    min_supercell_dim: int = 1
    max_supercell_dim: int = 6
    max_supercell_atoms: int = 1000
    relax: bool = True
    relax_cell: bool = True
    displacement: float = 0.01
    fmax: float = 1e-5
    max_steps: int = 2000
    optimizer: str = "FIRE"
    relax_backend: str = "auto"
    relax_model_path: Path | None = None
    allow_dpa_relax: bool = False
    band: str = "auto"
    kpath_mode: KPathMode = "auto"
    band_npoints: int = 101
    bandpath_symprec: float = DEFAULT_SEEKPATH_SYMPREC
    bandpath_with_time_reversal: bool = DEFAULT_SEEKPATH_WITH_TIME_REVERSAL
    fc_method: ForceConstantsMethod = "finite-displacement"
    compute_kappa: bool = False
    fc3_method: FC3Method = "finite-displacement"
    kappa_method: KappaMethod = "rta"
    wigner: bool = False
    temperatures: list[float] = Field(default_factory=lambda: [300.0])
    kappa_mesh: AutoOrTriplet = "auto"
    fc3_supercell_dim: AutoOrTriplet = "auto"
    fc3_target_supercell_length: float = 10.0
    max_fc3_supercell_atoms: int = 256
    fc3_displacement: float = 0.03
    fc3_cutoff_pair_distance: float | None = None
    max_fc3_displacements: int | None = None
    phono3py_symprec: float | None = 1e-5
    phono3py_cutoff_frequency: float | None = 1e-4
    phono3py_plusminus: Phono3pyPlusMinus = "auto"
    phono3py_diagonal: bool = False
    phono3py_symmetry: bool = True
    phono3py_mesh_symmetry: bool = True
    phono3py_isotope: bool = False
    boundary_mfp: float = 0.0
    cutoff_pair_distance: float = 0.0
    phono3py_symmetrize_fc2: bool = True
    phono3py_symmetrize_fc3: bool = True
    deepmd_reuse_calculator: bool = True
    deepmd_force_backend: DeepMDForceBackend = "ase"
    deepmd_device: DeepMDDevice = "cpu"
    deepmd_model_head: str | None = None
    deepmd_deterministic: bool = False
    save_force_audit: bool = False
    n_structures: int = 200
    rattle_std: float = 0.02
    cutoffs: list[float] = Field(default_factory=lambda: [5.0, 4.0])
    min_dist: float = 1.8
    primitive_matrix: PrimitiveMatrixName = "P"
    dos: bool = True
    asr: bool = True
    symmetrize_fc: bool = True
    export_fc2_text: bool = True
    fc2_text_name: str = "FORCE_CONSTANTS"
    shengbte_fc2_name: str = "FORCE_CONSTANTS_2ND"
    plot_dpi: int = 300
    plot_format: str = "png"
    imag_threshold: float = -0.1
    phonopy_symprec: float = 1e-5
    angle_tolerance: float = -1.0
    max_workers: int = 1
    dry_run: bool = False
    print_config: bool = False
    overwrite: bool = False
    resume: bool = False
    log_level: str = "INFO"
    run_command: str | None = Field(default=None, exclude=True)
    option_sources: dict[str, str] = Field(default_factory=dict, exclude=True)
    supercell_info: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_q_mesh_aliases(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        data = dict(raw)
        legacy_fc2_asr = data.pop("phono3py_fc2_asr", None)
        if legacy_fc2_asr is not None and "phono3py_symmetrize_fc2" not in data:
            data["phono3py_symmetrize_fc2"] = legacy_fc2_asr
        legacy_symprec = data.pop("symprec", None)
        if legacy_symprec is not None and "phonopy_symprec" not in data:
            data["phonopy_symprec"] = legacy_symprec

        q_mesh = data.pop("q_mesh", None)
        dos_mesh = data.pop("dos_mesh", None)
        if q_mesh is not None:
            data["mesh"] = q_mesh
            data["kappa_mesh"] = q_mesh
            return data
        if dos_mesh is not None and not _is_explicit_triplet_value(data.get("mesh")):
            data["mesh"] = dos_mesh

        mesh = data.get("mesh")
        kappa_mesh = data.get("kappa_mesh")
        if _is_explicit_triplet_value(mesh):
            data["kappa_mesh"] = mesh
        elif _is_explicit_triplet_value(kappa_mesh):
            data["mesh"] = kappa_mesh
        return data

    @field_validator("backend", mode="before")
    @classmethod
    def normalize_backend(cls, value: str) -> str:
        normalized = value.lower()
        if normalized == "pynep":
            raise ValueError(
                "Unsupported backend: pynep. PyNEP backend has been removed. Please use "
                "backend=calorine for real NEP/NEP89 calculations or backend=dummy for tests."
            )
        return normalized

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("optimizer", mode="before")
    @classmethod
    def normalize_optimizer(cls, value: str) -> str:
        return value.upper()

    @field_validator("band", mode="before")
    @classmethod
    def normalize_band(cls, value: str) -> str:
        return value.lower()

    @field_validator("kpath_mode", mode="before")
    @classmethod
    def normalize_kpath_mode(cls, value: str) -> str:
        return str(value).lower()

    @field_validator("fc_method", mode="before")
    @classmethod
    def normalize_fc_method(cls, value: str) -> str:
        return str(value).lower()

    @field_validator("fc3_method", mode="before")
    @classmethod
    def normalize_fc3_method(cls, value: str) -> str:
        return str(value).lower()

    @field_validator("kappa_method", mode="before")
    @classmethod
    def normalize_kappa_method(cls, value: str) -> str:
        return str(value).lower()

    @field_validator("deepmd_force_backend", mode="before")
    @classmethod
    def normalize_deepmd_force_backend(cls, value: str) -> str:
        return str(value).lower()

    @field_validator("deepmd_device", mode="before")
    @classmethod
    def normalize_deepmd_device(cls, value: str) -> str:
        return str(value).lower()

    @field_validator("phono3py_plusminus", mode="before")
    @classmethod
    def normalize_phono3py_plusminus(cls, value: str | bool) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value).lower()

    @field_validator("primitive_matrix", mode="before")
    @classmethod
    def normalize_primitive_matrix(cls, value: str) -> str:
        normalized = value.lower()
        if normalized == "p":
            return "P"
        return normalized

    @field_validator("plot_format", mode="before")
    @classmethod
    def normalize_plot_format(cls, value: str) -> str:
        return value.lower()

    @field_validator("supercell_dim", "mesh", "kappa_mesh", "fc3_supercell_dim")
    @classmethod
    def validate_triplet(cls, value: AutoOrTriplet) -> AutoOrTriplet:
        if isinstance(value, str):
            if value.lower() != "auto":
                raise ValueError("must be 'auto' or three positive integers")
            return "auto"
        if len(value) != 3 or any(int(item) <= 0 for item in value):
            raise ValueError("must contain three positive integers")
        return [int(item) for item in value]

    @field_validator("max_workers")
    @classmethod
    def validate_workers(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_workers must be at least 1")
        return value

    @field_validator(
        "displacement",
        "fmax",
        "target_supercell_length",
        "phonopy_symprec",
        "bandpath_symprec",
        "fc3_target_supercell_length",
        "fc3_displacement",
        "rattle_std",
        "min_dist",
    )
    @classmethod
    def validate_positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be positive")
        return float(value)

    @field_validator("phono3py_symprec", "phono3py_cutoff_frequency")
    @classmethod
    def validate_optional_positive_phono3py_float(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("must be positive")
        return float(value) if value is not None else None

    @field_validator("boundary_mfp", "cutoff_pair_distance")
    @classmethod
    def validate_nonnegative_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("must be non-negative")
        return float(value)

    @field_validator(
        "max_steps",
        "band_npoints",
        "plot_dpi",
        "max_supercell_atoms",
        "min_supercell_dim",
        "max_supercell_dim",
        "max_fc3_supercell_atoms",
        "n_structures",
    )
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be at least 1")
        return int(value)

    @field_validator("max_fc3_displacements")
    @classmethod
    def validate_optional_positive_int(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("must be at least 1")
        return value

    @field_validator("fc3_cutoff_pair_distance")
    @classmethod
    def validate_optional_positive_float(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("temperatures", "cutoffs")
    @classmethod
    def validate_positive_float_list(cls, value: list[float]) -> list[float]:
        if not value or any(float(item) <= 0 for item in value):
            raise ValueError("must contain positive values")
        return [float(item) for item in value]

    @field_validator("angle_tolerance")
    @classmethod
    def validate_angle_tolerance(cls, value: float) -> float:
        value = float(value)
        if value == -1.0 or value > 0:
            return value
        raise ValueError("angle_tolerance must be -1.0 or a positive value")

    @model_validator(mode="after")
    def validate_supercell_bounds(self) -> "WorkflowConfig":
        if self.min_supercell_dim > self.max_supercell_dim:
            raise ValueError("min_supercell_dim must be <= max_supercell_dim")
        return self

    @property
    def symprec(self) -> float:
        """Deprecated compatibility alias for phonopy_symprec."""

        return self.phonopy_symprec

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly representation."""

        return self.model_dump(mode="json")


def default_config_dict() -> dict[str, Any]:
    """Return a complete example configuration dictionary."""

    return WorkflowConfig(
        input_path=Path("examples/Si.vasp"),
        input_dir=Path("examples"),
        outdir=Path("results"),
        model_path=Path("nep89_potential/nep89_20250409.txt"),
        backend="dummy",
    ).to_dict()


def load_config(path: Path) -> WorkflowConfig:
    """Load a workflow configuration from YAML."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise ConfigError(f"Could not read config file '{path}': {exc}") from exc

    try:
        return WorkflowConfig(**data)
    except Exception as exc:
        raise ConfigError(f"Invalid config file '{path}': {exc}") from exc


def write_config(config: WorkflowConfig | dict[str, Any], path: Path) -> None:
    """Write a workflow configuration to YAML."""

    data = config.to_dict() if isinstance(config, WorkflowConfig) else config
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def merge_overrides(config: WorkflowConfig, **overrides: Any) -> WorkflowConfig:
    """Return a new config with non-None overrides applied."""

    clean = {key: value for key, value in overrides.items() if value is not None}
    explicit = set(clean.pop("_explicit_options", []))
    if str(clean.get("backend", "")).lower() == "pynep":
        raise ConfigError(
            "Unsupported backend: pynep. PyNEP backend has been removed. Please use "
            "backend=calorine for real NEP/NEP89 calculations or backend=dummy for tests."
        )
    data = config.model_dump()
    option_sources = dict(config.option_sources)
    for key in clean:
        option_sources[key] = "user" if key in explicit or not explicit else "user"
    data.update(clean)
    data["option_sources"] = option_sources
    try:
        return WorkflowConfig(**data)
    except Exception as exc:
        raise ConfigError(f"Invalid workflow override: {exc}") from exc
