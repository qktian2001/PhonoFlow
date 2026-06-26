# PhonoFlow

PhonoFlow is a Python command-line workflow for phonon and lattice
thermal-transport calculations with machine-learning interatomic potentials. It
wraps ASE, Phonopy, Phono3py, Calorine CPUNEP, and optional DeepMD-kit backends
so routine structure relaxation, FC2/FC3 generation, phonon post-processing,
and model comparison can be driven from reproducible CLI commands.

This public repository is the CLI/local-engineering release. It does not include
the private Web Studio, model weight files, run results, local archives, or
private runtime configuration. Users provide their own structure files and model
paths.

For real NEP/NEP89 calculations, the recommended backend is Calorine CPUNEP.

## Capabilities

- Harmonic phonon workflow: relaxation, FC2, band structure, DOS, group
  velocity, acoustic-sum-rule handling, and stability summaries.
- Anharmonic thermal workflow: FC3, phonon lifetime, and thermal conductivity
  through Phono3py.
- Thermal-transport methods: RTA and LBTE where supported by the installed
  Phono3py stack, with optional Wigner transport capability detection.
- Backends: `dummy` for tests, `calorine` for NEP/NEP89 potentials through
  Calorine CPUNEP, and DeepMD/DPA aliases where DeepMD-kit is installed.
- Multi-model comparison through the `compare-models` command when available in
  the installed checkout.
- Explicit output metadata: resolved settings, command record, timing,
  structure provenance, and JSON/text summaries.

## Installation

Create a Python 3.10+ environment, then install the project from the repository
root:

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
```

Optional backend extras can be installed as needed:

```bash
python -m pip install -e ".[calorine]"
python -m pip install -e ".[thermal]"
python -m pip install -e ".[hiphive]"
```

DeepMD/DPA calculations require a compatible DeepMD-kit installation and model
file supplied by the user.

## CLI Quick Start

Check the command surface and dependency status:

```bash
phonoflow --help
phonoflow version
phonoflow doctor --verbose
phonoflow init-config --out config.yaml
```

Run a deterministic dummy workflow:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --backend dummy \
  --outdir work/si_dummy \
  --overwrite
```

Inspect automatic settings without running the full calculation:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --backend calorine \
  --model-path /path/to/nep-model.txt \
  --outdir work/si_dry_run \
  --dry-run \
  --overwrite
```

## Single-Model Example

The standard single-structure command uses a structure file, backend, model
path, and output directory:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --backend calorine \
  --model-path /path/to/nep-model.txt \
  --outdir work/si_calorine \
  --supercell-dim auto \
  --mesh auto \
  --relax \
  --relax-cell \
  --overwrite
```

To include third-order force constants and thermal conductivity:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --backend calorine \
  --model-path /path/to/nep-model.txt \
  --outdir work/si_kappa \
  --compute-kappa \
  --fc3-method finite-displacement \
  --fc3-supercell-dim auto \
  --kappa-mesh auto \
  --method rta \
  --temperatures 300 \
  --overwrite
```

Use `--method lbte` or `--lbte` for LBTE when the local Phono3py installation
and calculation size support it. Use `--wigner true` only when the installed
thermal backend reports Wigner transport support.

## Multi-Model Compare

If the `compare-models` command is present in the installed checkout, compare
multiple model/backend pairs with one parent command:

```bash
phonoflow compare-models \
  --input-path examples/Si.vasp \
  --outdir work/compare_si \
  --model-label nep_a \
  --backend calorine \
  --model-path /path/to/nep-a.txt \
  --model-label nep_b \
  --backend calorine \
  --model-path /path/to/nep-b.txt \
  --mesh auto \
  --compute-kappa \
  --method rta \
  --overwrite
```

Each child run receives the shared structure and workflow options, while backend
and model path stay specific to the selected model label.

## Key Parameters

- `--input-path`: structure file accepted by ASE, such as VASP/POSCAR or CIF.
- `--model-path`: user-provided potential or DeepMD model file path.
- `--backend`: `auto`, `dummy`, `calorine`, `deepmd`, `dpa`, `dpa31`, `dpa32`,
  `dpa33`, `dpa4neo`, or compatibility aliases supported by the code.
- `--outdir`: output directory for all generated calculation artifacts.
- `--relax` / `--no-relax`: enable or skip structure relaxation.
- `--relax-cell` / `--no-relax-cell`: relax both atoms and cell, or keep the
  cell fixed.
- `--supercell-dim`: FC2 supercell dimensions, for example `2 2 2`, or `auto`.
- `--target-supercell-length`: target length used when FC2 supercell inference
  is automatic.
- `--mesh` / `--q-mesh`: shared q mesh for DOS and harmonic post-processing.
- `--displacement`: harmonic finite-displacement amplitude.
- `--compute-kappa`: enable FC3 and thermal-conductivity calculation.
- `--fc3-method`: `finite-displacement` or `hiphive`.
- `--fc3-supercell-dim`: FC3 supercell dimensions or `auto`.
- `--fc3-displacement`: third-order displacement amplitude.
- `--kappa-mesh`: q mesh used by Phono3py thermal conductivity.
- `--method` / `--kappa-method`: `rta` or `lbte`.
- `--temperatures`: one or more temperatures in K.
- `--phono3py-symmetrize-fc2` and `--phono3py-symmetrize-fc3`: apply official
  Phono3py force-constant symmetrization in the finite-displacement route.
- `--n-structures`, `--rattle-std`, `--cutoffs`, `--min-dist`: HiPhive fitting
  controls when `--fc3-method hiphive` is selected.
- `--dry-run`: resolve settings and write metadata without running the full
  workflow.
- `--overwrite`: intentionally reuse an existing output directory.

## Outputs

Depending on selected options, PhonoFlow may write:

- `resolved_settings.json` and `resolved_settings.yaml`
- `run_command.txt`
- `structure_provenance.json`
- `spacegroup_report.json` and `spacegroup_report.txt`
- `force_constants.hdf5`
- `FORCE_CONSTANTS` and `FORCE_CONSTANTS_2ND`
- `phonopy.yaml`
- `band.yaml`
- `phonon_band.csv`, `phonon_band.dat`, and band metadata JSON files
- `phonon_dos.dat`
- `phonon_group_velocity.csv`
- `phono3py_params.yaml`
- `fc2.hdf5`, `fc3.hdf5`, and `kappa-*.hdf5`
- `phonon_lifetime.csv`
- `thermal_conductivity.csv`
- `result.json`
- `summary.txt`
- calculation timing and diagnostics files

Run artifacts are intentionally ignored by Git in this public release.

## Tests

Compile Python files:

```bash
python -m compileall src tests scripts
```

Run the test suite:

```bash
PYTHONPATH=src python -m pytest tests -q
```

Some tests exercise optional backends and may skip or fail when the matching
runtime stack is not installed. The dummy backend tests are intended to validate
the baseline workflow without model files.

## Repository Boundary

The public repository is limited to CLI/local engine source, CLI/backend tests,
safe examples, and public documentation. It intentionally excludes:

- Web Studio implementation
- model weights and potential files
- generated results and logs
- local archives and task notes
- private runtime configuration
- database files and account data
- h5/hdf5 thermal artifacts, PNG result plots, command records, summaries, force
  constants, and kappa output files

Use a private workspace for model files, Web Studio deployments, long-running
outputs, and local operational material.
