"""Typer command-line interface for PhonoFlow."""

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import typer
from rich.console import Console
from rich.table import Table

from phonoflow.config import WorkflowConfig, default_config_dict, load_config, merge_overrides, write_config
from phonoflow.constants import PROJECT_NAME, VERSION
from phonoflow.exceptions import PhonoFlowError
from phonoflow.runtime_warnings import install_optional_deepmd_cuda_probe_warning_filter

app = typer.Typer(
    help="PhonoFlow phonon and lattice thermal-transport workflow.",
    invoke_without_command=True,
    no_args_is_help=True,
)
console = Console()

_PARAMETER_PURPOSES: dict[str, str] = {
    "input_path": "Single input structure path.",
    "input_dir": "Batch input directory.",
    "outdir": "Output directory; command code resolves a default when omitted.",
    "model_path": "User-provided NEP, DeepMD, or compatible model path.",
    "backend": "Calculator backend selection.",
    "backend_alias": "Optional backend alias metadata.",
    "dpa_model_name": "Optional named DPA model selector.",
    "supercell_dim": "FC2 supercell dimensions or automatic inference.",
    "mesh": "Harmonic/DOS q mesh; q_mesh is accepted as an alias.",
    "target_supercell_length": "Target length for automatic FC2 supercell inference.",
    "min_supercell_dim": "Minimum automatic FC2 multiplier.",
    "max_supercell_dim": "Maximum automatic FC2 multiplier.",
    "max_supercell_atoms": "Maximum atoms in automatic FC2 supercell.",
    "relax": "Enable structure relaxation.",
    "relax_cell": "Relax cell and positions together.",
    "displacement": "Harmonic finite-displacement amplitude.",
    "fmax": "Relaxation force threshold in eV/A.",
    "max_steps": "Maximum relaxation optimizer steps.",
    "optimizer": "ASE optimizer name.",
    "relax_backend": "Backend used for relaxation.",
    "relax_model_path": "Optional relaxation-specific model path.",
    "allow_dpa_relax": "Explicitly permit DPA/DeepMD relaxation.",
    "band": "Legacy band selector.",
    "kpath_mode": "K-path generator: auto, 3d_seekpath, 2d_ase, or custom.",
    "band_npoints": "Points per band segment.",
    "bandpath_symprec": "SeekPath/2D ASE band-path precision.",
    "bandpath_with_time_reversal": "Use time-reversal reduction for 3D SeekPath.",
    "fc_method": "Harmonic force-constant method.",
    "compute_kappa": "Enable FC3 and thermal conductivity.",
    "fc3_method": "FC3 method: finite displacement or HiPhive.",
    "kappa_method": "Thermal solver method: RTA or LBTE.",
    "wigner": "Request Wigner transport when available.",
    "temperatures": "Thermal conductivity temperatures in K.",
    "kappa_mesh": "Phono3py kappa mesh.",
    "fc3_supercell_dim": "FC3 supercell dimensions or automatic inference.",
    "fc3_target_supercell_length": "Target length for automatic FC3 supercell inference.",
    "max_fc3_supercell_atoms": "Maximum atoms in automatic FC3 supercell.",
    "fc3_displacement": "FC3 displacement amplitude.",
    "fc3_cutoff_pair_distance": "Optional FC3 pair cutoff.",
    "max_fc3_displacements": "Optional smoke-test cap on FC3 displacements.",
    "phono3py_symprec": "Phono3py symmetry precision.",
    "phono3py_cutoff_frequency": "Phono3py cutoff frequency in THz.",
    "phono3py_plusminus": "Phono3py plus/minus displacement mode.",
    "phono3py_diagonal": "Use diagonal FC3 displacements.",
    "phono3py_symmetry": "Use Phono3py symmetry reduction.",
    "phono3py_mesh_symmetry": "Use mesh symmetry for kappa.",
    "phono3py_isotope": "Enable isotope scattering.",
    "boundary_mfp": "Boundary mean free path; zero disables it.",
    "cutoff_pair_distance": "Phono3py pair cutoff; zero disables it.",
    "phono3py_symmetrize_fc2": "Apply official Phono3py FC2 symmetrization.",
    "phono3py_symmetrize_fc3": "Apply official Phono3py FC3 symmetrization.",
    "deepmd_reuse_calculator": "Reuse one DeepMD calculator in force loops.",
    "deepmd_force_backend": "DeepMD force path: ASE or DeepPot direct.",
    "deepmd_device": "DeepMD runtime device.",
    "deepmd_model_head": "Optional multitask DeepMD model head.",
    "deepmd_deterministic": "Best-effort deterministic DeepMD environment.",
    "save_force_audit": "Save finite-displacement force diagnostics.",
    "n_structures": "HiPhive rattle structure count.",
    "rattle_std": "HiPhive rattle standard deviation.",
    "cutoffs": "HiPhive cutoff radii.",
    "min_dist": "HiPhive minimum interatomic distance.",
    "primitive_matrix": "Phonopy primitive matrix setting.",
    "dos": "Compute DOS outputs.",
    "asr": "Apply acoustic sum rule where possible.",
    "symmetrize_fc": "Symmetrize FC2 where possible.",
    "export_fc2_text": "Export Phonopy and ShengBTE-style FC2 text files.",
    "fc2_text_name": "Phonopy FC2 text filename.",
    "shengbte_fc2_name": "ShengBTE-style FC2 text filename.",
    "plot_dpi": "Plot resolution.",
    "plot_format": "Plot format; current release writes PNG.",
    "imag_threshold": "Imaginary-mode stability threshold in THz.",
    "phonopy_symprec": "Phonopy symmetry precision.",
    "angle_tolerance": "spglib angle tolerance; -1.0 means default.",
    "max_workers": "Reserved worker count field.",
    "dry_run": "Resolve settings without heavy calculation.",
    "print_config": "Print resolved settings.",
    "overwrite": "Allow replacing existing output directories.",
    "resume": "Reuse complete successful outputs.",
    "log_level": "Logging verbosity.",
}


def _format_parameter_default(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _print_parameter_reference() -> None:
    """Print the WorkflowConfig parameter reference used by docs/configuration.md."""

    config = WorkflowConfig()
    table = Table(title="PhonoFlow workflow parameters")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Default", style="green")
    table.add_column("Purpose")
    excluded = {"run_command", "option_sources", "supercell_info"}
    for name in WorkflowConfig.model_fields:
        if name in excluded:
            continue
        table.add_row(
            name,
            _format_parameter_default(getattr(config, name)),
            _PARAMETER_PURPOSES.get(name, "Workflow parameter."),
        )
    console.print(table)
    console.print(
        "Aliases: q_mesh -> mesh/kappa_mesh, dos_mesh -> mesh, "
        "symprec -> phonopy_symprec, phono3py_fc2_asr -> phono3py_symmetrize_fc2."
    )


@app.callback()
def root(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the installed PhonoFlow version and exit.",
        is_eager=True,
    ),
    help_all: bool = typer.Option(
        False,
        "--help-all",
        "--show-parameters",
        help="Show all workflow parameters, defaults, and purposes, then exit.",
        is_eager=True,
    ),
) -> None:
    """PhonoFlow command-line interface."""

    if version:
        console.print(f"{PROJECT_NAME} {VERSION}")
        raise typer.Exit()
    if help_all:
        _print_parameter_reference()
        raise typer.Exit()


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _status_text(available: bool, optional: bool = False) -> str:
    if available:
        return "[green]available[/green]"
    if optional:
        return "[yellow]optional dependency missing[/yellow]"
    return "[red]not found[/red]"


def _import_check(module_name: str, attr_path: str | None = None) -> tuple[bool, str]:
    try:
        module = __import__(module_name, fromlist=["*"])
        target: Any = module
        if attr_path is not None:
            for attr in attr_path.split("."):
                target = getattr(target, attr)
        return True, "available"
    except ImportError as exc:
        return False, f"optional dependency missing: {exc}"
    except Exception as exc:
        return False, f"API unavailable: {exc}"


def _calorine_status() -> tuple[str, str]:
    calorine_ok, calorine_detail = _import_check("calorine")
    if not calorine_ok:
        return "[yellow]optional dependency missing[/yellow]", calorine_detail
    cpunep_ok, cpunep_detail = _import_check("calorine.calculators", "CPUNEP")
    if cpunep_ok:
        return "[green]available[/green]", "CPUNEP importable"
    return "[red]API unavailable[/red]", cpunep_detail


def _load_or_default(config_path: Optional[Path]) -> WorkflowConfig:
    if config_path is None:
        return WorkflowConfig()
    return load_config(config_path)


def _optional_triplet(value: Tuple[int, int, int]) -> list[int] | None:
    return None if value == (-1, -1, -1) else list(value)


def _optional_float_pair(value: Tuple[float, float]) -> list[float] | None:
    return None if value == (-1.0, -1.0) else [float(item) for item in value]


def _optional_float_list(value: list[float] | None) -> list[float] | None:
    return list(value) if value else None


def _optional_bool_string(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise typer.BadParameter("must be true or false")


def _resolve_kappa_method_option(method: str | None, lbte_flag: bool | None) -> str | None:
    if lbte_flag is None:
        return method
    flag_method = "lbte" if lbte_flag else "rta"
    if method is not None and str(method).strip().lower() != flag_method:
        raise typer.BadParameter("--method conflicts with --lbte/--rta")
    return flag_method


def _explicit_options(**values: Any) -> list[str]:
    return [key for key, value in values.items() if value is not None]


def _warn_deprecated_phono3py_fc2_asr_alias() -> None:
    if any(arg in {"--phono3py-fc2-asr", "--no-phono3py-fc2-asr"} for arg in sys.argv):
        console.print(
            "[yellow]Warning: --phono3py-fc2-asr is deprecated; "
            "use --phono3py-symmetrize-fc2 instead.[/yellow]"
        )

def _warn_deprecated_symprec_alias() -> None:
    if "--symprec" in sys.argv:
        console.print(
            "[yellow]Warning: --symprec is deprecated; "
            "use --phonopy-symprec for phonopy SYMMETRY_TOLERANCE / API symprec instead.[/yellow]"
        )


def _preprocess_auto_triplet_options(argv: list[str]) -> list[str]:
    """Allow options like ``--supercell-dim auto`` for Typer tuple options."""

    output: list[str] = []
    index = 0
    triplet_options = {"--supercell-dim", "--mesh", "--q-mesh", "--kappa-mesh", "--fc3-supercell-dim"}
    while index < len(argv):
        item = argv[index]
        if item in triplet_options and index + 1 < len(argv) and argv[index + 1].lower() == "auto":
            output.extend([item, "-1", "-1", "-1"])
            index += 2
        else:
            output.append(item)
            index += 1
    return output


def _handle_error(exc: Exception) -> None:
    if isinstance(exc, PhonoFlowError):
        console.print(f"[red]Error:[/red] {exc}")
    else:
        console.print(f"[red]Unexpected error:[/red] {exc}")
    raise typer.Exit(code=1) from exc


@app.command()
def version() -> None:
    """Print the package version."""

    console.print(f"{PROJECT_NAME} version {VERSION}")


@app.command()
def doctor(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed import diagnostics."),
) -> None:
    """Check runtime and backend availability."""

    table = Table(title=f"{PROJECT_NAME} environment")
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Details")

    table.add_row("Python", _status_text(True), sys.version.split()[0])
    checks = [
        ("numpy", "numpy", False),
        ("ASE", "ase", False),
        ("Phonopy", "phonopy", False),
        ("seekpath", "seekpath", False),
        ("matplotlib", "matplotlib", False),
    ]
    for label, module, optional in checks:
        available = _module_available(module)
        detail = "importable" if available else "install package or use dummy backend"
        table.add_row(label, _status_text(available, optional=optional), detail)

    calorine_status, calorine_detail = _calorine_status()
    table.add_row("Calorine CPUNEP", calorine_status, calorine_detail)

    gpumd_path = shutil.which("gpumd")
    table.add_row(
        "GPUMD executable",
        _status_text(gpumd_path is not None, optional=True),
        gpumd_path or "gpumd command not on PATH",
    )
    console.print(table)

    if verbose:
        verbose_table = Table(title="Detailed diagnostics")
        verbose_table.add_column("Item")
        verbose_table.add_column("Value")
        verbose_table.add_row("Python executable", sys.executable)
        verbose_table.add_row("Python version", sys.version.replace("\n", " "))
        verbose_table.add_row("Platform", platform.platform())
        verbose_table.add_row("PhonoFlow version", VERSION)
        verbose_table.add_row("Calorine import", _import_check("calorine")[1])
        verbose_table.add_row("CPUNEP import", _import_check("calorine.calculators", "CPUNEP")[1])
        verbose_table.add_row("GPUMD executable", gpumd_path or "not found on PATH")
        console.print(verbose_table)


@app.command("init-config")
def init_config(
    out: Path = typer.Option(Path("config.yaml"), "--out", "-o", help="Path for the generated YAML config."),
) -> None:
    """Generate an example YAML configuration."""

    try:
        write_config(default_config_dict(), out)
    except Exception as exc:
        _handle_error(exc)
    console.print(f"[green]Wrote example config:[/green] {out}")


@app.command()
def single(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file."),
    input_path: Optional[Path] = typer.Option(None, "--input-path", help="Single structure input path."),
    outdir: Optional[Path] = typer.Option(None, "--outdir", help="Output directory."),
    model_path: Optional[Path] = typer.Option(None, "--model-path", help="NEP or DeepMD model path."),
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        help="Backend: auto, calorine, dummy, gpumd, deepmd, dpa31, dpa32, dpa33, or dpa4neo.",
    ),
    supercell_dim: Tuple[int, int, int] = typer.Option(
        (-1, -1, -1),
        "--supercell-dim",
        help="Phonopy supercell dimensions, e.g. --supercell-dim 2 2 2.",
        show_default=False,
    ),
    displacement: Optional[float] = typer.Option(None, "--displacement", help="Finite displacement distance."),
    target_supercell_length: Optional[float] = typer.Option(
        None, "--target-supercell-length", help="Target auto supercell length in Angstrom."
    ),
    max_supercell_atoms: Optional[int] = typer.Option(
        None, "--max-supercell-atoms", help="Maximum atoms allowed in an automatically inferred supercell."
    ),
    min_supercell_dim: Optional[int] = typer.Option(
        None, "--min-supercell-dim", help="Minimum multiplier for each automatic supercell direction."
    ),
    max_supercell_dim: Optional[int] = typer.Option(
        None, "--max-supercell-dim", help="Maximum multiplier for each automatic supercell direction."
    ),
    fmax: Optional[float] = typer.Option(None, "--fmax", help="Relaxation force threshold in eV/A."),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", help="Maximum relaxation optimizer steps."),
    optimizer: Optional[str] = typer.Option(None, "--optimizer", help="ASE optimizer: FIRE or LBFGS."),
    relax_backend: Optional[str] = typer.Option(
        None,
        "--relax-backend",
        help="Relax backend: auto, calorine/nep89, or dpa/deepmd/force for explicit DPA relaxation.",
    ),
    relax_model_path: Optional[Path] = typer.Option(
        None,
        "--relax-model-path",
        help="Optional NEP/NEP89 model path used for DPA default relaxation.",
    ),
    allow_dpa_relax: Optional[bool] = typer.Option(
        None,
        "--allow-dpa-relax/--no-allow-dpa-relax",
        help="Explicitly allow DPA/DeepMD to perform structure relaxation.",
        show_default=False,
    ),
    band: Optional[str] = typer.Option(None, "--band", help="Legacy band selector. Leave auto unless using custom path plumbing."),
    kpath_mode: Optional[str] = typer.Option(
        None,
        "--kpath-mode",
        help="K-path generator: auto, 3d_seekpath, 2d_ase, or custom.",
    ),
    bandpath_symprec: Optional[float] = typer.Option(
        None,
        "--bandpath-symprec",
        help="3D SeekPath symmetry precision; also used as the ASE 2D bandpath epsilon floor when slabs are auto-detected.",
    ),
    bandpath_with_time_reversal: Optional[bool] = typer.Option(
        None,
        "--bandpath-with-time-reversal/--no-bandpath-with-time-reversal",
        help="Use time-reversal reduction for 3D SeekPath band-path generation.",
        show_default=False,
    ),
    fc_method: Optional[str] = typer.Option(
        None,
        "--fc-method",
        help="Force-constant method: finite-displacement. hiphive is reserved for a future release.",
    ),
    mesh: Tuple[int, int, int] = typer.Option(
        (-1, -1, -1),
        "--mesh",
        "--q-mesh",
        help="Phonopy mesh for DOS, e.g. --mesh 20 20 20 or --mesh auto.",
        show_default=False,
    ),
    primitive_matrix: Optional[str] = typer.Option(
        None, "--primitive-matrix", help="Primitive matrix: auto, P, identity, or none."
    ),
    asr: Optional[bool] = typer.Option(
        None, "--asr/--no-asr", help="Apply acoustic sum rule if possible.", show_default=False
    ),
    symmetrize_fc: Optional[bool] = typer.Option(
        None,
        "--symmetrize-fc/--no-symmetrize-fc",
        help="Symmetrize force constants if possible.",
        show_default=False,
    ),
    export_fc2_text: Optional[bool] = typer.Option(
        None,
        "--export-fc2-text/--no-export-fc2-text",
        help="Export Phonopy FORCE_CONSTANTS and ShengBTE-style FORCE_CONSTANTS_2ND FC2 text files.",
        show_default=False,
    ),
    fc2_text_name: Optional[str] = typer.Option(None, "--fc2-text-name", help="Phonopy FC2 text filename."),
    shengbte_fc2_name: Optional[str] = typer.Option(
        None, "--shengbte-fc2-name", help="ShengBTE-style FC2 text filename."
    ),
    compute_kappa: Optional[bool] = typer.Option(
        None,
        "--compute-kappa/--no-compute-kappa",
        help="Enable third-order force constants and lattice thermal conductivity.",
        show_default=False,
    ),
    fc3_method: Optional[str] = typer.Option(
        None, "--fc3-method", help="FC3 method: finite-displacement or hiphive."
    ),
    kappa_method: Optional[str] = typer.Option(
        None, "--method", "--kappa-method", help="BTE solver method: rta or lbte."
    ),
    kappa_solver_flag: Optional[bool] = typer.Option(
        None,
        "--lbte/--rta",
        help="Shortcut for --method lbte or --method rta.",
        show_default=False,
    ),
    wigner: Optional[str] = typer.Option(
        None, "--wigner", help="Enable Wigner transport if supported: true or false."
    ),
    temperatures: Optional[list[float]] = typer.Option(
        None, "--temperatures", help="Thermal conductivity temperatures in K."
    ),
    kappa_mesh: Tuple[int, int, int] = typer.Option(
        (-1, -1, -1), "--kappa-mesh", help="phono3py kappa mesh, e.g. --kappa-mesh 11 11 11 or auto.", show_default=False
    ),
    fc3_supercell_dim: Tuple[int, int, int] = typer.Option(
        (-1, -1, -1), "--fc3-supercell-dim", help="FC3 supercell dimensions or auto.", show_default=False
    ),
    fc3_target_supercell_length: Optional[float] = typer.Option(
        None, "--fc3-target-supercell-length", help="Target auto FC3 supercell length in Angstrom."
    ),
    max_fc3_supercell_atoms: Optional[int] = typer.Option(
        None, "--max-fc3-supercell-atoms", help="Maximum atoms allowed in an automatic FC3 supercell."
    ),
    fc3_displacement: Optional[float] = typer.Option(None, "--fc3-displacement", help="FC3 displacement distance."),
    fc3_cutoff_pair_distance: Optional[float] = typer.Option(
        None, "--fc3-cutoff-pair-distance", help="Optional phono3py FC3 pair cutoff distance."
    ),
    max_fc3_displacements: Optional[int] = typer.Option(
        None, "--max-fc3-displacements", help="Smoke-test cap on FC3 displacements; not for production."
    ),
    phono3py_symprec: Optional[float] = typer.Option(None, "--phono3py-symprec", help="phono3py symmetry precision."),
    phono3py_cutoff_frequency: Optional[float] = typer.Option(
        None, "--phono3py-cutoff-frequency", help="phono3py cutoff frequency in THz."
    ),
    phono3py_plusminus: Optional[str] = typer.Option(
        None, "--phono3py-plusminus", help="phono3py plusminus mode: auto, true, or false."
    ),
    phono3py_diagonal: Optional[bool] = typer.Option(
        None, "--phono3py-diagonal/--no-phono3py-diagonal", help="Use diagonal FC3 displacements.", show_default=False
    ),
    phono3py_symmetry: Optional[bool] = typer.Option(
        None, "--phono3py-symmetry/--no-phono3py-symmetry", help="Use phono3py symmetry reduction.", show_default=False
    ),
    phono3py_mesh_symmetry: Optional[bool] = typer.Option(
        None,
        "--phono3py-mesh-symmetry/--no-phono3py-mesh-symmetry",
        help="Use mesh symmetry in phono3py thermal conductivity.",
        show_default=False,
    ),
    phono3py_isotope: Optional[bool] = typer.Option(
        None, "--isotope/--no-isotope", help="Enable or disable isotope scattering.", show_default=False
    ),
    boundary_mfp: Optional[float] = typer.Option(None, "--boundary-mfp", help="Boundary mean free path; 0 disables it."),
    cutoff_pair_distance: Optional[float] = typer.Option(
        None, "--cutoff-pair-distance", help="phono3py FC3 pair cutoff distance; 0 disables it."
    ),
    phono3py_symmetrize_fc2: Optional[bool] = typer.Option(
        None,
        "--phono3py-symmetrize-fc2/--no-phono3py-symmetrize-fc2",
        "--phono3py-fc2-asr/--no-phono3py-fc2-asr",
        help="Apply phono3py official FC2 force-constant symmetrization. Deprecated alias: --phono3py-fc2-asr.",
        show_default=False,
    ),
    phono3py_symmetrize_fc3: Optional[bool] = typer.Option(
        None,
        "--phono3py-symmetrize-fc3/--no-phono3py-symmetrize-fc3",
        help="Apply phono3py official FC3 force-constant symmetrization.",
        show_default=False,
    ),
    deepmd_reuse_calculator: Optional[bool] = typer.Option(
        None,
        "--deepmd-reuse-calculator/--no-deepmd-reuse-calculator",
        help="PhonoFlow performance policy: reuse one ASE/DeepMD DP calculator in force loops; not a scientific model parameter.",
        show_default=False,
    ),
    deepmd_force_backend: Optional[str] = typer.Option(
        None, "--deepmd-force-backend", help="DeepMD force path: ase or deeppot."
    ),
    deepmd_device: Optional[str] = typer.Option(
        None, "--deepmd-device", help="DeepMD runtime device: cpu, cuda, or auto."
    ),
    deepmd_model_head: Optional[str] = typer.Option(
        None, "--deepmd-model-head", help="DeepMD multitask model head, e.g. OMat24."
    ),
    deepmd_deterministic: Optional[bool] = typer.Option(
        None,
        "--deepmd-deterministic/--no-deepmd-deterministic",
        help="PhonoFlow reproducibility policy: best-effort deterministic thread/environment settings; not a DPA scientific parameter.",
        show_default=False,
    ),
    save_force_audit: Optional[bool] = typer.Option(
        None,
        "--save-force-audit/--no-save-force-audit",
        help="PhonoFlow diagnostic artifact: save finite-displacement force hashes, statistics, and raw arrays.",
        show_default=False,
    ),
    n_structures: Optional[int] = typer.Option(None, "--n-structures", help="HiPhive rattle structure count."),
    rattle_std: Optional[float] = typer.Option(None, "--rattle-std", help="HiPhive rattle standard deviation."),
    cutoffs: Tuple[float, float] = typer.Option(
        (-1.0, -1.0), "--cutoffs", help="HiPhive cutoffs, e.g. --cutoffs 5.0 4.0.", show_default=False
    ),
    min_dist: Optional[float] = typer.Option(None, "--min-dist", help="HiPhive minimum interatomic distance."),
    plot_dpi: Optional[int] = typer.Option(None, "--plot-dpi", help="Plot DPI."),
    plot_format: Optional[str] = typer.Option(None, "--plot-format", help="Plot format. Current release writes png."),
    relax_cell: Optional[bool] = typer.Option(
        None,
        "--relax-cell/--no-relax-cell",
        help="Relax atomic positions and cell together, or keep the cell fixed.",
        show_default=False,
    ),
    relax: Optional[bool] = typer.Option(
        None, "--relax/--no-relax", help="Enable or disable relaxation.", show_default=False
    ),
    dos: Optional[bool] = typer.Option(
        None, "--dos/--no-dos", help="Enable or disable total DOS output.", show_default=False
    ),
    imag_threshold: Optional[float] = typer.Option(None, "--imag-threshold", help="Imaginary-mode threshold in THz."),
    phonopy_symprec: Optional[float] = typer.Option(None, "--phonopy-symprec", "--symprec", help="phonopy SYMMETRY_TOLERANCE / API symprec. Deprecated alias: --symprec."),
    angle_tolerance: Optional[float] = typer.Option(
        None, "--angle-tolerance", help="spglib angle tolerance; -1.0 uses the spglib default."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Resolve settings and write metadata without running phonons."),
    print_config: bool = typer.Option(False, "--print-config", help="Print resolved settings without running phonons."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Allow replacing files in an existing output directory."),
    resume: bool = typer.Option(False, "--resume", help="Skip calculation if a complete successful result exists."),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="Log level."),
) -> None:
    """Run one single-structure phonon workflow."""

    _warn_deprecated_symprec_alias()

    try:
        workflow_config = merge_overrides(
            _load_or_default(config),
            input_path=input_path,
            outdir=outdir,
            model_path=model_path,
            backend=backend,
            supercell_dim=_optional_triplet(supercell_dim),
            displacement=displacement,
            target_supercell_length=target_supercell_length,
            max_supercell_atoms=max_supercell_atoms,
            min_supercell_dim=min_supercell_dim,
            max_supercell_dim=max_supercell_dim,
            fmax=fmax,
            max_steps=max_steps,
            optimizer=optimizer,
            relax_backend=relax_backend,
            relax_model_path=relax_model_path,
            allow_dpa_relax=allow_dpa_relax,
            band=band,
            kpath_mode=kpath_mode,
            bandpath_symprec=bandpath_symprec,
            bandpath_with_time_reversal=bandpath_with_time_reversal,
            fc_method=fc_method,
            mesh=_optional_triplet(mesh),
            primitive_matrix=primitive_matrix,
            asr=asr,
            symmetrize_fc=symmetrize_fc,
            export_fc2_text=export_fc2_text,
            fc2_text_name=fc2_text_name,
            shengbte_fc2_name=shengbte_fc2_name,
            compute_kappa=compute_kappa,
            fc3_method=fc3_method,
            kappa_method=_resolve_kappa_method_option(kappa_method, kappa_solver_flag),
            wigner=_optional_bool_string(wigner),
            temperatures=_optional_float_list(temperatures),
            kappa_mesh=_optional_triplet(kappa_mesh),
            fc3_supercell_dim=_optional_triplet(fc3_supercell_dim),
            fc3_target_supercell_length=fc3_target_supercell_length,
            max_fc3_supercell_atoms=max_fc3_supercell_atoms,
            fc3_displacement=fc3_displacement,
            fc3_cutoff_pair_distance=fc3_cutoff_pair_distance,
            max_fc3_displacements=max_fc3_displacements,
            phono3py_symprec=phono3py_symprec,
            phono3py_cutoff_frequency=phono3py_cutoff_frequency,
            phono3py_plusminus=phono3py_plusminus,
            phono3py_diagonal=phono3py_diagonal,
            phono3py_symmetry=phono3py_symmetry,
            phono3py_mesh_symmetry=phono3py_mesh_symmetry,
            phono3py_isotope=phono3py_isotope,
            boundary_mfp=boundary_mfp,
            cutoff_pair_distance=cutoff_pair_distance,
            phono3py_symmetrize_fc2=phono3py_symmetrize_fc2,
            phono3py_symmetrize_fc3=phono3py_symmetrize_fc3,
            deepmd_reuse_calculator=deepmd_reuse_calculator,
            deepmd_force_backend=deepmd_force_backend,
            deepmd_device=deepmd_device,
            deepmd_model_head=deepmd_model_head,
            deepmd_deterministic=deepmd_deterministic,
            save_force_audit=save_force_audit,
            n_structures=n_structures,
            rattle_std=rattle_std,
            cutoffs=_optional_float_pair(cutoffs),
            min_dist=min_dist,
            plot_dpi=plot_dpi,
            plot_format=plot_format,
            relax_cell=relax_cell,
            relax=relax,
            dos=dos,
            imag_threshold=imag_threshold,
            phonopy_symprec=phonopy_symprec,
            angle_tolerance=angle_tolerance,
            dry_run=(dry_run or print_config) or None,
            print_config=print_config or None,
            overwrite=overwrite or None,
            resume=resume or None,
            log_level=log_level,
            run_command=" ".join(sys.argv),
            _explicit_options=_explicit_options(
                input_path=input_path,
                outdir=outdir,
                model_path=model_path,
                backend=backend,
                supercell_dim=None if supercell_dim == (-1, -1, -1) else supercell_dim,
                displacement=displacement,
                target_supercell_length=target_supercell_length,
                max_supercell_atoms=max_supercell_atoms,
                min_supercell_dim=min_supercell_dim,
                max_supercell_dim=max_supercell_dim,
                fmax=fmax,
                max_steps=max_steps,
                optimizer=optimizer,
                relax_backend=relax_backend,
                relax_model_path=relax_model_path,
                allow_dpa_relax=allow_dpa_relax,
                band=band,
                kpath_mode=kpath_mode,
                bandpath_symprec=bandpath_symprec,
                bandpath_with_time_reversal=bandpath_with_time_reversal,
                fc_method=fc_method,
                mesh=None if mesh == (-1, -1, -1) else mesh,
                primitive_matrix=primitive_matrix,
                asr=asr,
                symmetrize_fc=symmetrize_fc,
                export_fc2_text=export_fc2_text,
                fc2_text_name=fc2_text_name,
                shengbte_fc2_name=shengbte_fc2_name,
                compute_kappa=compute_kappa,
                fc3_method=fc3_method,
                kappa_method=_resolve_kappa_method_option(kappa_method, kappa_solver_flag),
                wigner=wigner,
                temperatures=_optional_float_list(temperatures),
                kappa_mesh=None if kappa_mesh == (-1, -1, -1) else kappa_mesh,
                fc3_supercell_dim=None if fc3_supercell_dim == (-1, -1, -1) else fc3_supercell_dim,
                fc3_target_supercell_length=fc3_target_supercell_length,
                max_fc3_supercell_atoms=max_fc3_supercell_atoms,
                fc3_displacement=fc3_displacement,
                fc3_cutoff_pair_distance=fc3_cutoff_pair_distance,
                max_fc3_displacements=max_fc3_displacements,
                phono3py_symprec=phono3py_symprec,
                phono3py_cutoff_frequency=phono3py_cutoff_frequency,
                phono3py_plusminus=phono3py_plusminus,
                phono3py_diagonal=phono3py_diagonal,
                phono3py_symmetry=phono3py_symmetry,
                phono3py_mesh_symmetry=phono3py_mesh_symmetry,
                phono3py_isotope=phono3py_isotope,
                boundary_mfp=boundary_mfp,
                cutoff_pair_distance=cutoff_pair_distance,
                phono3py_symmetrize_fc2=phono3py_symmetrize_fc2,
                phono3py_symmetrize_fc3=phono3py_symmetrize_fc3,
                deepmd_reuse_calculator=deepmd_reuse_calculator,
                deepmd_force_backend=deepmd_force_backend,
                deepmd_device=deepmd_device,
                deepmd_model_head=deepmd_model_head,
                deepmd_deterministic=deepmd_deterministic,
                save_force_audit=save_force_audit,
                n_structures=n_structures,
                rattle_std=rattle_std,
                cutoffs=None if cutoffs == (-1.0, -1.0) else cutoffs,
                min_dist=min_dist,
                plot_dpi=plot_dpi,
                plot_format=plot_format,
                relax_cell=relax_cell,
                relax=relax,
                dos=dos,
                imag_threshold=imag_threshold,
                phonopy_symprec=phonopy_symprec,
                angle_tolerance=angle_tolerance,
                dry_run=dry_run or None,
                print_config=print_config or None,
                overwrite=overwrite or None,
                resume=resume or None,
                log_level=log_level,
            ),
        )
        from phonoflow.workflow.pipeline import run_single_workflow

        result = run_single_workflow(workflow_config)
    except Exception as exc:
        _handle_error(exc)

    console.print("[green]Single workflow completed.[/green]" if result["status"] != "dry-run" else "[green]Dry run completed.[/green]")
    console.print(f"Output directory: {result['outdir']}")
    if result.get("dynamically_stable") is not None:
        console.print(f"Dynamically stable: {result['dynamically_stable']}")
        console.print(f"Minimum frequency: {result['minimum_frequency_THz']:.3f} THz")


@app.command()
def run(
    input_path: Path = typer.Option(..., "--input-path", help="Single structure input path."),
    model_path: Path = typer.Option(..., "--model-path", help="NEP/NEP89 model path."),
    outdir: Optional[Path] = typer.Option(None, "--outdir", help="Output directory."),
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        help="Backend: auto, calorine, dummy, gpumd, deepmd, dpa31, dpa32, dpa33, or dpa4neo.",
    ),
    supercell_dim: Tuple[int, int, int] = typer.Option(
        (-1, -1, -1),
        "--supercell-dim",
        help="Phonopy supercell dimensions, e.g. --supercell-dim 2 2 2.",
        show_default=False,
    ),
    relax: Optional[bool] = typer.Option(
        None, "--relax/--no-relax", help="Enable or disable relaxation.", show_default=False
    ),
    relax_cell: Optional[bool] = typer.Option(
        None,
        "--relax-cell/--no-relax-cell",
        help="Relax atomic positions and cell together, or keep the cell fixed.",
        show_default=False,
    ),
    fmax: Optional[float] = typer.Option(None, "--fmax", help="Relaxation force threshold in eV/A."),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", help="Maximum relaxation optimizer steps."),
    optimizer: Optional[str] = typer.Option(None, "--optimizer", help="ASE optimizer: FIRE or LBFGS."),
    relax_backend: Optional[str] = typer.Option(
        None,
        "--relax-backend",
        help="Relax backend: auto, calorine/nep89, or dpa/deepmd/force for explicit DPA relaxation.",
    ),
    relax_model_path: Optional[Path] = typer.Option(
        None,
        "--relax-model-path",
        help="Optional NEP/NEP89 model path used for DPA default relaxation.",
    ),
    allow_dpa_relax: Optional[bool] = typer.Option(
        None,
        "--allow-dpa-relax/--no-allow-dpa-relax",
        help="Explicitly allow DPA/DeepMD to perform structure relaxation.",
        show_default=False,
    ),
    target_supercell_length: Optional[float] = typer.Option(
        None, "--target-supercell-length", help="Target auto supercell length in Angstrom."
    ),
    max_supercell_atoms: Optional[int] = typer.Option(
        None, "--max-supercell-atoms", help="Maximum atoms allowed in an automatically inferred supercell."
    ),
    min_supercell_dim: Optional[int] = typer.Option(
        None, "--min-supercell-dim", help="Minimum multiplier for each automatic supercell direction."
    ),
    max_supercell_dim: Optional[int] = typer.Option(
        None, "--max-supercell-dim", help="Maximum multiplier for each automatic supercell direction."
    ),
    phonopy_symprec: Optional[float] = typer.Option(None, "--phonopy-symprec", "--symprec", help="phonopy SYMMETRY_TOLERANCE / API symprec. Deprecated alias: --symprec."),
    kpath_mode: Optional[str] = typer.Option(
        None,
        "--kpath-mode",
        help="K-path generator: auto, 3d_seekpath, 2d_ase, or custom.",
    ),
    bandpath_symprec: Optional[float] = typer.Option(
        None,
        "--bandpath-symprec",
        help="3D SeekPath symmetry precision; also used as the ASE 2D bandpath epsilon floor when slabs are auto-detected.",
    ),
    bandpath_with_time_reversal: Optional[bool] = typer.Option(
        None,
        "--bandpath-with-time-reversal/--no-bandpath-with-time-reversal",
        help="Use time-reversal reduction for 3D SeekPath band-path generation.",
        show_default=False,
    ),
    angle_tolerance: Optional[float] = typer.Option(
        None, "--angle-tolerance", help="spglib angle tolerance; -1.0 uses the spglib default."
    ),
    export_fc2_text: Optional[bool] = typer.Option(
        None,
        "--export-fc2-text/--no-export-fc2-text",
        help="Export Phonopy FORCE_CONSTANTS and ShengBTE-style FORCE_CONSTANTS_2ND FC2 text files.",
        show_default=False,
    ),
    compute_kappa: Optional[bool] = typer.Option(
        None,
        "--compute-kappa/--no-compute-kappa",
        help="Enable third-order force constants and lattice thermal conductivity.",
        show_default=False,
    ),
    fc3_method: Optional[str] = typer.Option(None, "--fc3-method", help="FC3 method: finite-displacement or hiphive."),
    kappa_method: Optional[str] = typer.Option(
        None, "--method", "--kappa-method", help="BTE solver method: rta or lbte."
    ),
    kappa_solver_flag: Optional[bool] = typer.Option(
        None,
        "--lbte/--rta",
        help="Shortcut for --method lbte or --method rta.",
        show_default=False,
    ),
    wigner: Optional[str] = typer.Option(None, "--wigner", help="Enable Wigner transport if supported: true or false."),
    temperatures: Optional[list[float]] = typer.Option(
        None, "--temperatures", help="Thermal conductivity temperatures in K."
    ),
    kappa_mesh: Tuple[int, int, int] = typer.Option(
        (-1, -1, -1), "--kappa-mesh", help="phono3py kappa mesh or auto.", show_default=False
    ),
    fc3_supercell_dim: Tuple[int, int, int] = typer.Option(
        (-1, -1, -1), "--fc3-supercell-dim", help="FC3 supercell dimensions or auto.", show_default=False
    ),
    fc3_target_supercell_length: Optional[float] = typer.Option(
        None, "--fc3-target-supercell-length", help="Target auto FC3 supercell length in Angstrom."
    ),
    max_fc3_supercell_atoms: Optional[int] = typer.Option(
        None, "--max-fc3-supercell-atoms", help="Maximum atoms allowed in an automatic FC3 supercell."
    ),
    fc3_displacement: Optional[float] = typer.Option(None, "--fc3-displacement", help="FC3 displacement distance."),
    fc3_cutoff_pair_distance: Optional[float] = typer.Option(
        None, "--fc3-cutoff-pair-distance", help="Optional phono3py FC3 pair cutoff distance."
    ),
    max_fc3_displacements: Optional[int] = typer.Option(
        None, "--max-fc3-displacements", help="Smoke-test cap on FC3 displacements; not for production."
    ),
    phono3py_symprec: Optional[float] = typer.Option(None, "--phono3py-symprec", help="phono3py symmetry precision."),
    phono3py_cutoff_frequency: Optional[float] = typer.Option(
        None, "--phono3py-cutoff-frequency", help="phono3py cutoff frequency in THz."
    ),
    phono3py_plusminus: Optional[str] = typer.Option(
        None, "--phono3py-plusminus", help="phono3py plusminus mode: auto, true, or false."
    ),
    phono3py_diagonal: Optional[bool] = typer.Option(
        None, "--phono3py-diagonal/--no-phono3py-diagonal", help="Use diagonal FC3 displacements.", show_default=False
    ),
    phono3py_symmetry: Optional[bool] = typer.Option(
        None, "--phono3py-symmetry/--no-phono3py-symmetry", help="Use phono3py symmetry reduction.", show_default=False
    ),
    phono3py_mesh_symmetry: Optional[bool] = typer.Option(
        None,
        "--phono3py-mesh-symmetry/--no-phono3py-mesh-symmetry",
        help="Use mesh symmetry in phono3py thermal conductivity.",
        show_default=False,
    ),
    phono3py_isotope: Optional[bool] = typer.Option(
        None, "--isotope/--no-isotope", help="Enable or disable isotope scattering.", show_default=False
    ),
    boundary_mfp: Optional[float] = typer.Option(None, "--boundary-mfp", help="Boundary mean free path; 0 disables it."),
    cutoff_pair_distance: Optional[float] = typer.Option(
        None, "--cutoff-pair-distance", help="phono3py FC3 pair cutoff distance; 0 disables it."
    ),
    phono3py_symmetrize_fc2: Optional[bool] = typer.Option(
        None,
        "--phono3py-symmetrize-fc2/--no-phono3py-symmetrize-fc2",
        "--phono3py-fc2-asr/--no-phono3py-fc2-asr",
        help="Apply phono3py official FC2 force-constant symmetrization. Deprecated alias: --phono3py-fc2-asr.",
        show_default=False,
    ),
    phono3py_symmetrize_fc3: Optional[bool] = typer.Option(
        None,
        "--phono3py-symmetrize-fc3/--no-phono3py-symmetrize-fc3",
        help="Apply phono3py official FC3 force-constant symmetrization.",
        show_default=False,
    ),
    deepmd_reuse_calculator: Optional[bool] = typer.Option(
        None,
        "--deepmd-reuse-calculator/--no-deepmd-reuse-calculator",
        help="PhonoFlow performance policy: reuse one ASE/DeepMD DP calculator in force loops; not a scientific model parameter.",
        show_default=False,
    ),
    deepmd_force_backend: Optional[str] = typer.Option(
        None, "--deepmd-force-backend", help="DeepMD force path: ase or deeppot."
    ),
    deepmd_device: Optional[str] = typer.Option(
        None, "--deepmd-device", help="DeepMD runtime device: cpu, cuda, or auto."
    ),
    deepmd_model_head: Optional[str] = typer.Option(
        None, "--deepmd-model-head", help="DeepMD multitask model head, e.g. OMat24."
    ),
    deepmd_deterministic: Optional[bool] = typer.Option(
        None,
        "--deepmd-deterministic/--no-deepmd-deterministic",
        help="PhonoFlow reproducibility policy: best-effort deterministic thread/environment settings; not a DPA scientific parameter.",
        show_default=False,
    ),
    save_force_audit: Optional[bool] = typer.Option(
        None,
        "--save-force-audit/--no-save-force-audit",
        help="PhonoFlow diagnostic artifact: save finite-displacement force hashes, statistics, and raw arrays.",
        show_default=False,
    ),
    n_structures: Optional[int] = typer.Option(None, "--n-structures", help="HiPhive rattle structure count."),
    rattle_std: Optional[float] = typer.Option(None, "--rattle-std", help="HiPhive rattle standard deviation."),
    cutoffs: Tuple[float, float] = typer.Option(
        (-1.0, -1.0), "--cutoffs", help="HiPhive cutoffs, e.g. --cutoffs 5.0 4.0.", show_default=False
    ),
    min_dist: Optional[float] = typer.Option(None, "--min-dist", help="HiPhive minimum interatomic distance."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Resolve settings and write metadata without running phonons."),
    print_config: bool = typer.Option(False, "--print-config", help="Print resolved settings without running phonons."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Allow replacing files in an existing output directory."),
    resume: bool = typer.Option(False, "--resume", help="Skip calculation if a complete successful result exists."),
) -> None:
    """Run a real single workflow with automatic defaults."""

    _warn_deprecated_symprec_alias()

    try:
        workflow_config = merge_overrides(
            WorkflowConfig(),
            input_path=input_path,
            model_path=model_path,
            outdir=outdir,
            backend=backend,
            supercell_dim=_optional_triplet(supercell_dim),
            relax=relax,
            relax_cell=relax_cell,
            fmax=fmax,
            max_steps=max_steps,
            optimizer=optimizer,
            relax_backend=relax_backend,
            relax_model_path=relax_model_path,
            allow_dpa_relax=allow_dpa_relax,
            target_supercell_length=target_supercell_length,
            max_supercell_atoms=max_supercell_atoms,
            min_supercell_dim=min_supercell_dim,
            max_supercell_dim=max_supercell_dim,
            phonopy_symprec=phonopy_symprec,
            kpath_mode=kpath_mode,
            bandpath_symprec=bandpath_symprec,
            bandpath_with_time_reversal=bandpath_with_time_reversal,
            angle_tolerance=angle_tolerance,
            export_fc2_text=export_fc2_text,
            compute_kappa=compute_kappa,
            fc3_method=fc3_method,
            kappa_method=_resolve_kappa_method_option(kappa_method, kappa_solver_flag),
            wigner=_optional_bool_string(wigner),
            temperatures=_optional_float_list(temperatures),
            kappa_mesh=_optional_triplet(kappa_mesh),
            fc3_supercell_dim=_optional_triplet(fc3_supercell_dim),
            fc3_target_supercell_length=fc3_target_supercell_length,
            max_fc3_supercell_atoms=max_fc3_supercell_atoms,
            fc3_displacement=fc3_displacement,
            fc3_cutoff_pair_distance=fc3_cutoff_pair_distance,
            max_fc3_displacements=max_fc3_displacements,
            phono3py_symprec=phono3py_symprec,
            phono3py_cutoff_frequency=phono3py_cutoff_frequency,
            phono3py_plusminus=phono3py_plusminus,
            phono3py_diagonal=phono3py_diagonal,
            phono3py_symmetry=phono3py_symmetry,
            phono3py_mesh_symmetry=phono3py_mesh_symmetry,
            phono3py_isotope=phono3py_isotope,
            boundary_mfp=boundary_mfp,
            cutoff_pair_distance=cutoff_pair_distance,
            phono3py_symmetrize_fc2=phono3py_symmetrize_fc2,
            phono3py_symmetrize_fc3=phono3py_symmetrize_fc3,
            deepmd_reuse_calculator=deepmd_reuse_calculator,
            deepmd_force_backend=deepmd_force_backend,
            deepmd_device=deepmd_device,
            deepmd_model_head=deepmd_model_head,
            deepmd_deterministic=deepmd_deterministic,
            save_force_audit=save_force_audit,
            n_structures=n_structures,
            rattle_std=rattle_std,
            cutoffs=_optional_float_pair(cutoffs),
            min_dist=min_dist,
            dry_run=(dry_run or print_config) or None,
            print_config=print_config or None,
            overwrite=overwrite or None,
            resume=resume or None,
            run_command=" ".join(sys.argv),
            _explicit_options=_explicit_options(
                input_path=input_path,
                model_path=model_path,
                outdir=outdir,
                backend=backend,
                supercell_dim=None if supercell_dim == (-1, -1, -1) else supercell_dim,
                relax=relax,
                relax_cell=relax_cell,
                fmax=fmax,
                max_steps=max_steps,
                optimizer=optimizer,
                relax_backend=relax_backend,
                relax_model_path=relax_model_path,
                allow_dpa_relax=allow_dpa_relax,
                target_supercell_length=target_supercell_length,
                max_supercell_atoms=max_supercell_atoms,
                min_supercell_dim=min_supercell_dim,
                max_supercell_dim=max_supercell_dim,
                phonopy_symprec=phonopy_symprec,
                kpath_mode=kpath_mode,
                bandpath_symprec=bandpath_symprec,
                bandpath_with_time_reversal=bandpath_with_time_reversal,
                angle_tolerance=angle_tolerance,
                export_fc2_text=export_fc2_text,
                compute_kappa=compute_kappa,
                fc3_method=fc3_method,
                kappa_method=_resolve_kappa_method_option(kappa_method, kappa_solver_flag),
                wigner=wigner,
                temperatures=_optional_float_list(temperatures),
                kappa_mesh=None if kappa_mesh == (-1, -1, -1) else kappa_mesh,
                fc3_supercell_dim=None if fc3_supercell_dim == (-1, -1, -1) else fc3_supercell_dim,
                fc3_target_supercell_length=fc3_target_supercell_length,
                max_fc3_supercell_atoms=max_fc3_supercell_atoms,
                fc3_displacement=fc3_displacement,
                fc3_cutoff_pair_distance=fc3_cutoff_pair_distance,
                max_fc3_displacements=max_fc3_displacements,
                phono3py_symprec=phono3py_symprec,
                phono3py_cutoff_frequency=phono3py_cutoff_frequency,
                phono3py_plusminus=phono3py_plusminus,
                phono3py_diagonal=phono3py_diagonal,
                phono3py_symmetry=phono3py_symmetry,
                phono3py_mesh_symmetry=phono3py_mesh_symmetry,
                phono3py_isotope=phono3py_isotope,
                boundary_mfp=boundary_mfp,
                cutoff_pair_distance=cutoff_pair_distance,
                phono3py_symmetrize_fc2=phono3py_symmetrize_fc2,
                phono3py_symmetrize_fc3=phono3py_symmetrize_fc3,
                deepmd_reuse_calculator=deepmd_reuse_calculator,
                deepmd_force_backend=deepmd_force_backend,
                deepmd_device=deepmd_device,
                deepmd_model_head=deepmd_model_head,
                deepmd_deterministic=deepmd_deterministic,
                save_force_audit=save_force_audit,
                n_structures=n_structures,
                rattle_std=rattle_std,
                cutoffs=None if cutoffs == (-1.0, -1.0) else cutoffs,
                min_dist=min_dist,
                dry_run=dry_run or None,
                print_config=print_config or None,
                overwrite=overwrite or None,
                resume=resume or None,
            ),
        )
        from phonoflow.workflow.pipeline import run_single_workflow

        result = run_single_workflow(workflow_config)
    except Exception as exc:
        _handle_error(exc)

    console.print("[green]Run completed.[/green]" if result["status"] != "dry-run" else "[green]Dry run completed.[/green]")
    console.print(f"Output directory: {result['outdir']}")
    if result.get("dynamically_stable") is not None:
        console.print(f"Dynamically stable: {result['dynamically_stable']}")
        console.print(f"Minimum frequency: {result['minimum_frequency_THz']:.3f} THz")


@app.command("compare-models")
def compare_models_command(
    input_path: Path = typer.Option(..., "--input-path", help="Single structure input path."),
    outdir: Path = typer.Option(..., "--outdir", help="Comparison output directory."),
    models: Optional[str] = typer.Option(None, "--models", help="Legacy comma-separated list of 1-3 models."),
    model: Optional[list[str]] = typer.Option(
        None,
        "--model",
        help="One model selection. Repeat for 1-3 models; accepts nep89, bundled DPA filenames, or model paths.",
    ),
    compute_kappa: bool = typer.Option(
        False,
        "--compute-kappa/--no-compute-kappa",
        help="Enable third-order force constants and lattice thermal conductivity for each model.",
        show_default=False,
    ),
    relax: bool = typer.Option(
        False,
        "--relax/--no-relax",
        help="Relax once with NEP89/Calorine before compare children share the relaxed structure.",
        show_default=False,
    ),
    relax_cell: bool = typer.Option(
        True,
        "--relax-cell/--no-relax-cell",
        help="When compare relaxation is enabled, relax both atom positions and cell or positions only.",
        show_default=False,
    ),
    supercell_dim: Tuple[int, int, int] = typer.Option(
        (-1, -1, -1),
        "--supercell-dim",
        help="Common phonopy supercell dimensions, e.g. --supercell-dim 2 2 2 or auto.",
        show_default=False,
    ),
    mesh: Tuple[int, int, int] = typer.Option(
        (-1, -1, -1),
        "--mesh",
        "--q-mesh",
        help="Common phonopy mesh, e.g. --mesh 20 20 20 or auto.",
        show_default=False,
    ),
    kappa_mesh: Tuple[int, int, int] = typer.Option(
        (-1, -1, -1),
        "--kappa-mesh",
        help="Compatibility alias for the shared DOS/kappa q-mesh.",
        show_default=False,
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Resolve settings without running phonons."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Allow replacing files in existing output directories."),
    target_supercell_length: Optional[float] = typer.Option(
        None, "--target-supercell-length", help="Common target auto supercell length in Angstrom."
    ),
    displacement: Optional[float] = typer.Option(
        None, "--displacement", help="Common harmonic finite-displacement amplitude in Angstrom."
    ),
    fc3_target_supercell_length: Optional[float] = typer.Option(
        None, "--fc3-target-supercell-length", help="Common target auto FC3 supercell length in Angstrom."
    ),
    fc3_supercell_dim: Tuple[int, int, int] = typer.Option(
        (-1, -1, -1),
        "--fc3-supercell-dim",
        help="Common FC3 supercell dimensions or auto.",
        show_default=False,
    ),
    fc3_method: Optional[str] = typer.Option(
        None, "--fc3-method", help="Common FC3 method: finite-displacement or hiphive."
    ),
    fc3_displacement: Optional[float] = typer.Option(
        None,
        "--fc3-displacement",
        help="Atomic displacement amplitude in Angstrom for each FC3 finite-difference structure.",
    ),
    fc3_cutoff_pair_distance: Optional[float] = typer.Option(
        None,
        "--fc3-cutoff-pair-distance",
        help="Optional phono3py FC3 pair cutoff distance for every child workflow.",
    ),
    kappa_method: Optional[str] = typer.Option(
        None,
        "--method",
        "--kappa-method",
        help="Common BTE solver method for every child workflow: rta or lbte.",
    ),
    kappa_solver_flag: Optional[bool] = typer.Option(
        None,
        "--lbte/--rta",
        help="Shortcut for --method lbte or --method rta.",
        show_default=False,
    ),
    temperatures: Optional[list[float]] = typer.Option(
        None,
        "--temperatures",
        help="Common thermal conductivity temperatures in K for every child workflow.",
    ),
    wigner: Optional[str] = typer.Option(
        None, "--wigner", help="Enable Wigner transport for every child workflow if supported: true or false."
    ),
    max_fc3_displacements: Optional[int] = typer.Option(
        None,
        "--max-fc3-displacements",
        help="Smoke-test cap on the number of FC3 displaced structures; not a displacement amplitude.",
    ),
    fmax: Optional[float] = typer.Option(None, "--fmax", help="Common relaxation force threshold in eV/A."),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", help="Common maximum relaxation steps."),
    primitive_matrix: Optional[str] = typer.Option(
        None, "--primitive-matrix", help="Common primitive matrix: auto, P, identity, or none."
    ),
    dos: Optional[bool] = typer.Option(
        None, "--dos/--no-dos", help="Enable or disable DOS for every child workflow.", show_default=False
    ),
    export_fc2_text: Optional[bool] = typer.Option(
        None,
        "--export-fc2-text/--no-export-fc2-text",
        help="Enable or disable FC2 text export for every child workflow.",
        show_default=False,
    ),
    kpath_mode: Optional[str] = typer.Option(
        None,
        "--kpath-mode",
        help="Common k-path generator for every child workflow: auto, 3d_seekpath, 2d_ase, or custom.",
    ),
    bandpath_with_time_reversal: Optional[bool] = typer.Option(
        None,
        "--bandpath-with-time-reversal/--no-bandpath-with-time-reversal",
        help="Common 3D SeekPath time-reversal reduction setting for every child workflow.",
        show_default=False,
    ),
    phonopy_symprec: Optional[float] = typer.Option(
        None,
        "--phonopy-symprec",
        "--symprec",
        help="Common phonopy SYMMETRY_TOLERANCE / API symprec for every child workflow. Deprecated alias: --symprec.",
    ),
    phono3py_symmetrize_fc2: Optional[bool] = typer.Option(
        None,
        "--phono3py-symmetrize-fc2/--no-phono3py-symmetrize-fc2",
        "--phono3py-fc2-asr/--no-phono3py-fc2-asr",
        help="Apply phono3py official FC2 force-constant symmetrization to compute-kappa children. Deprecated alias: --phono3py-fc2-asr.",
        show_default=False,
    ),
    phono3py_symmetrize_fc3: Optional[bool] = typer.Option(
        None,
        "--phono3py-symmetrize-fc3/--no-phono3py-symmetrize-fc3",
        help="Apply phono3py official FC3 force-constant symmetrization to compute-kappa children.",
        show_default=False,
    ),
    phono3py_symprec: Optional[float] = typer.Option(
        None, "--phono3py-symprec", help="phono3py symmetry precision for compute-kappa children."
    ),
    phono3py_cutoff_frequency: Optional[float] = typer.Option(
        None, "--phono3py-cutoff-frequency", help="phono3py cutoff frequency for compute-kappa children."
    ),
    n_structures: Optional[int] = typer.Option(None, "--n-structures", help="Common HiPhive rattle structure count."),
    rattle_std: Optional[float] = typer.Option(None, "--rattle-std", help="Common HiPhive rattle standard deviation."),
    cutoffs: Tuple[float, float] = typer.Option(
        (-1.0, -1.0),
        "--cutoffs",
        help="Common HiPhive cutoffs, e.g. --cutoffs 5.0 4.0.",
        show_default=False,
    ),
    min_dist: Optional[float] = typer.Option(None, "--min-dist", help="Common HiPhive minimum interatomic distance."),
    deepmd_device: Optional[str] = typer.Option(
        None, "--deepmd-device", help="DeepMD device for DPA3/DPA4 children."
    ),
    deepmd_deterministic: Optional[bool] = typer.Option(
        None,
        "--deepmd-deterministic/--no-deepmd-deterministic",
        help="PhonoFlow reproducibility policy for DeepMD child processes; not a DPA scientific parameter.",
        show_default=False,
    ),
    deepmd_reuse_calculator: Optional[bool] = typer.Option(
        None,
        "--deepmd-reuse-calculator/--no-deepmd-reuse-calculator",
        help="PhonoFlow performance policy: reuse the ASE/DeepMD calculator in child force loops.",
        show_default=False,
    ),
    save_force_audit: Optional[bool] = typer.Option(
        None,
        "--save-force-audit/--no-save-force-audit",
        help="PhonoFlow diagnostic artifact for child force hashes, statistics, and raw arrays.",
        show_default=False,
    ),
    dpa_safe_mode: bool = typer.Option(
        False,
        "--dpa-safe-mode",
        help="Explicit DPA4 safety preset: 12 A target and 256-atom auto-supercell cap. Not a formal default.",
    ),
) -> None:
    """Compare one to three independent model workflows on one structure."""

    _warn_deprecated_symprec_alias()
    try:
        from phonoflow.compare_models import compare_models

        selected_models = list(model or [])
        if not selected_models and models:
            selected_models = [item.strip() for item in models.split(",") if item.strip()]
        summary = compare_models(
            input_path=input_path,
            outdir=outdir,
            model_names=selected_models,
            compute_kappa=compute_kappa,
            overwrite=overwrite,
            dry_run=dry_run,
            relax=relax,
            relax_cell=relax_cell,
            isolate=True,
            supercell_dim=None if supercell_dim == (-1, -1, -1) else supercell_dim,
            mesh=None if mesh == (-1, -1, -1) else mesh,
            kappa_mesh=None if kappa_mesh == (-1, -1, -1) else kappa_mesh,
            target_supercell_length=target_supercell_length,
            displacement=displacement,
            fc3_target_supercell_length=fc3_target_supercell_length,
            fc3_supercell_dim=None if fc3_supercell_dim == (-1, -1, -1) else fc3_supercell_dim,
            fc3_method=fc3_method,
            fc3_displacement=fc3_displacement,
            fc3_cutoff_pair_distance=fc3_cutoff_pair_distance,
            kappa_method=_resolve_kappa_method_option(kappa_method, kappa_solver_flag),
            wigner=_optional_bool_string(wigner),
            temperatures=_optional_float_list(temperatures),
            max_fc3_displacements=max_fc3_displacements,
            fmax=fmax,
            max_steps=max_steps,
            primitive_matrix=primitive_matrix,
            dos=dos,
            export_fc2_text=export_fc2_text,
            kpath_mode=kpath_mode,
            bandpath_with_time_reversal=bandpath_with_time_reversal,
            phonopy_symprec=phonopy_symprec,
            phono3py_symmetrize_fc2=phono3py_symmetrize_fc2,
            phono3py_symmetrize_fc3=phono3py_symmetrize_fc3,
            phono3py_symprec=phono3py_symprec,
            phono3py_cutoff_frequency=phono3py_cutoff_frequency,
            n_structures=n_structures,
            rattle_std=rattle_std,
            cutoffs=_optional_float_pair(cutoffs),
            min_dist=min_dist,
            deepmd_device=deepmd_device,
            deepmd_deterministic=deepmd_deterministic,
            deepmd_reuse_calculator=deepmd_reuse_calculator,
            save_force_audit=save_force_audit,
            dpa_safe_mode=dpa_safe_mode,
        )
    except Exception as exc:
        _handle_error(exc)

    console.print("[green]Compare-models workflow completed.[/green]")
    console.print(f"Output directory: {summary['outdir']}")
    console.print(f"Models: {len(summary['models'])}")
    failures = sum(1 for item in summary["models"] if item.get("status") == "failed")
    console.print(f"Failures: {failures}")


@app.command("read-result")
def read_result(
    path: Path = typer.Argument(..., help="Path to result.json or an output directory."),
    json_output: bool = typer.Option(False, "--json", help="Print raw result JSON."),
) -> None:
    """Read and summarize a PhonoFlow result.json file."""

    result_path = path / "result.json" if path.is_dir() else path
    if not result_path.exists():
        console.print(f"[red]Error:[/red] result file not found: {result_path}")
        raise typer.Exit(code=1)
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[red]Error:[/red] could not parse JSON '{result_path}': {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        console.print_json(json.dumps(result, ensure_ascii=False))
        return

    table = Table(title=str(result_path))
    table.add_column("Metric")
    table.add_column("Value")
    keys = [
        "project",
        "version",
        "backend_requested",
        "backend_resolved",
        "structure_formula",
        "structure_type",
        "vacuum_like_directions",
        "n_atoms_unitcell",
        "target_supercell_length",
        "supercell_dim_resolved",
        "supercell_lengths_resolved",
        "n_atoms_supercell",
        "primitive_matrix_resolved",
        "mesh_resolved",
        "n_displaced_supercells",
        "relax",
        "relax_cell",
        "relax_mode",
        "constant_cell",
        "fmax",
        "max_steps",
        "relax_converged",
        "final_max_force_eV_per_A",
        "final_stress_GPa",
        "volume_change_percent",
        "initial_spacegroup",
        "final_spacegroup",
        "spacegroup_changed",
        "spacegroup_change_summary",
        "high_symmetry_path",
        "kpath_mode",
        "kpath_mode_resolved",
        "kpath_dimensionality",
        "kpath_source",
        "kpath_bravais",
        "vacuum_axis_name",
        "kpath",
        "symprec",
        "spacegroup_symprec",
        "bandpath_symprec",
        "bandpath_with_time_reversal",
        "bandpath_structure_source",
        "angle_tolerance",
        "minimum_frequency_THz",
        "maximum_frequency_THz",
        "has_imaginary_frequency",
        "export_fc2_text",
        "force_constants_text_exported",
        "phonopy_force_constants_file",
        "shengbte_fc2_file",
        "group_velocity",
        "thermal_conductivity",
        "input_file_hash",
        "model_file_hash",
        "elapsed_time_seconds",
        "warnings",
        "dos",
        "output_directory",
    ]
    for key in keys:
        value = result.get(key, result.get("backend") if key == "backend_resolved" else "")
        if key.endswith("_hash") and value:
            value = str(value)[:12]
        if key in {"initial_spacegroup", "final_spacegroup"} and isinstance(value, dict):
            symbol = value.get("international_symbol")
            number = value.get("spacegroup_number")
            value = f"{symbol} (No. {number})" if symbol and number else "unavailable"
        if key == "group_velocity" and isinstance(value, dict):
            if value.get("available"):
                value = (
                    f"available; data={value.get('data_file')}; plot={value.get('plot_file')}; "
                    f"max={value.get('max_abs_velocity')} {value.get('unit')}; "
                    f"mean={value.get('mean_abs_velocity')} {value.get('unit')}"
                )
            else:
                value = f"not available; {value.get('reason')}"
        if key == "thermal_conductivity" and isinstance(value, dict):
            if value.get("available"):
                files = value.get("files") or {}
                lifetime = value.get("lifetime") or {}
                lifetime_text = (
                    f"; lifetime={lifetime.get('data_file')}/{lifetime.get('plot_file')}"
                    if lifetime.get("available")
                    else f"; lifetime not available ({lifetime.get('reason', 'not available')})"
                )
                value = (
                    f"available; method={value.get('kappa_method')}; "
                    f"fc3={value.get('fc3_method')}; "
                    f"csv={files.get('thermal_conductivity_csv')}; "
                    f"plot={files.get('thermal_conductivity_png')}"
                    f"{lifetime_text}"
                )
            elif value.get("enabled"):
                value = f"not available; {value.get('reason')}"
            else:
                value = "disabled"
        if key == "high_symmetry_path" and isinstance(value, dict):
            value = value.get("display", "not available")
        if key == "kpath" and isinstance(value, dict):
            value = (
                f"mode={value.get('resolved_mode')} ; "
                f"dimensionality={value.get('dimensionality')} ; "
                f"source={value.get('source')} ; "
                f"path={value.get('display_path')}"
            )
        if key == "warnings":
            value = len(value or [])
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def batch(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file."),
    input_dir: Optional[Path] = typer.Option(None, "--input-dir", help="Directory with structure files."),
    outdir: Optional[Path] = typer.Option(None, "--outdir", help="Batch output directory."),
    model_path: Optional[Path] = typer.Option(None, "--model-path", help="NEP or DeepMD model path."),
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        help="Backend: auto, calorine, dummy, gpumd, deepmd, dpa31, dpa32, dpa33, or dpa4neo.",
    ),
    relax: Optional[bool] = typer.Option(None, "--relax/--no-relax", help="Enable or disable relaxation."),
    dos: Optional[bool] = typer.Option(None, "--dos/--no-dos", help="Enable or disable total DOS output."),
    imag_threshold: Optional[float] = typer.Option(None, "--imag-threshold", help="Imaginary-mode threshold in THz."),
    max_workers: Optional[int] = typer.Option(None, "--max-workers", help="Parallel worker count."),
    resume: Optional[bool] = typer.Option(None, "--resume/--no-resume", help="Skip finished structures."),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="Log level."),
) -> None:
    """Run the batch workflow skeleton."""

    try:
        from phonoflow.workflow.batch import run_batch_workflow

        workflow_config = merge_overrides(
            _load_or_default(config),
            input_dir=input_dir,
            outdir=outdir,
            model_path=model_path,
            backend=backend,
            relax=relax,
            dos=dos,
            imag_threshold=imag_threshold,
            max_workers=max_workers,
            resume=resume,
            log_level=log_level,
        )
        results = run_batch_workflow(workflow_config)
    except Exception as exc:
        _handle_error(exc)

    failed = sum(1 for item in results if item.get("status") == "failed")
    console.print("[green]Batch workflow completed.[/green]")
    console.print(f"Structures processed: {len(results)}")
    console.print(f"Failures: {failed}")
    console.print(f"Summary CSV: {Path(workflow_config.outdir or 'results') / 'summary.csv'}")


def main() -> Any:
    """Console script entry point."""

    sys.argv = _preprocess_auto_triplet_options(sys.argv)
    install_optional_deepmd_cuda_probe_warning_filter()
    return app()


if __name__ == "__main__":
    main()
