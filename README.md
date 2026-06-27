# PhonoFlow 1.0

PhonoFlow 1.0 is the initial command-line release of a phonon and lattice
thermal-transport workflow for machine-learning interatomic potentials. It
connects ASE, Phonopy, Phono3py, Calorine CPUNEP, and optional DeepMD-kit
backends behind reproducible CLI commands.

The public repository contains the CLI engine, tests, public documentation, and
one small Si structure for smoke tests. It does not contain model weights, run
results, private application code, local archives, or private runtime
configuration.

## Core Capabilities

- Single-structure harmonic phonon workflow: structure relaxation, FC2 finite
  displacements, force evaluation, Phonopy post-processing, band path, DOS,
  group velocity, stability summary, and reports.
- Optional thermal-conductivity workflow: FC3 finite displacement or HiPhive
  fitting, Phono3py RTA/LBTE execution, lifetimes, kappa tables, and Wigner
  capability handling when supported locally.
- Backend selection: `dummy` for tests, `calorine` for NEP/NEP89 through
  Calorine CPUNEP, optional `deepmd` and DPA aliases when DeepMD-kit is
  installed, plus a GPUMD-oriented backend module.
- Automatic defaults for supercells and q meshes, with explicit CLI/config
  overrides for production calculations.
- Multi-model comparison through `compare-models`, including shared workflow
  settings and per-model backend/model paths.
- Reproducibility artifacts: resolved settings, command record, timing,
  structure provenance, space-group report, force-audit diagnostics, JSON/text
  summaries, and optional FC2 text exports.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `docs/` | PhonoFlow 1.0 CLI documentation, including command usage, configuration, outputs, architecture, and testing notes. |
| `examples/` | Minimal public structure examples for smoke tests and quick-start commands. The current public fixture is `Si.vasp`. |
| `scripts/` | Small maintenance and validation helpers used by the command-line project. |
| `src/phonoflow/` | Core Python package: CLI entry points, workflow orchestration, calculator backends, phonon/thermal logic, reporting, and I/O helpers. |
| `tests/` | Public pytest suite for CLI behavior, configuration, workflow plumbing, backends, reporting, and output validation. |
| `.gitattributes` | Git text/binary handling rules for consistent repository checkout behavior. |
| `.gitignore` | Ignore rules for generated outputs, caches, model files, local archives, and other non-source artifacts. |
| `LICENSE` | MIT license for the public release. |
| `README.md` | Project overview, quick start, command summary, and repository boundary. |
| `pyproject.toml` | Python package metadata, dependencies, optional extras, console script entry point, and test/tool configuration. |

## Install

Use Python 3.10 or newer.

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
```

Optional extras:

```bash
python -m pip install -e ".[calorine]"
python -m pip install -e ".[thermal]"
python -m pip install -e ".[hiphive]"
```

DeepMD/DPA runs require a compatible DeepMD-kit installation and user-provided
model files.

Calorine CPUNEP is used for NEP/NEP89 workflows; DPA/DeepMD workflows are
available through the DeepMD backend and DPA model aliases.

## Quick Start

Check the command surface and dependency status:

```bash
phonoflow --help
phonoflow --help-all
phonoflow version
phonoflow doctor --verbose
```

Generate a complete example config:

```bash
phonoflow init-config --out config.yaml
```

### NEP/NEP89 Workflows

Harmonic phonons only, using `run`:

```bash
phonoflow run \
  --input-path examples/Si.vasp \
  --model-path /path/to/nep-model.txt \
  --backend calorine \
  --outdir work/nep_phonon \
  --supercell-dim auto \
  --mesh auto \
  --relax \
  --overwrite
```

Harmonic phonons only, using `single`:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --model-path /path/to/nep-model.txt \
  --backend calorine \
  --outdir work/nep_phonon_single \
  --supercell-dim auto \
  --mesh auto \
  --relax \
  --overwrite
```

Thermal conductivity, using `run`:

```bash
phonoflow run \
  --input-path examples/Si.vasp \
  --model-path /path/to/nep-model.txt \
  --backend calorine \
  --outdir work/nep_kappa \
  --supercell-dim auto \
  --mesh auto \
  --compute-kappa \
  --fc3-method finite-displacement \
  --fc3-supercell-dim auto \
  --kappa-mesh auto \
  --method rta \
  --temperatures 300 \
  --overwrite
```

Thermal conductivity, using `single`:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --model-path /path/to/nep-model.txt \
  --backend calorine \
  --outdir work/nep_kappa_single \
  --supercell-dim auto \
  --mesh auto \
  --compute-kappa \
  --fc3-method finite-displacement \
  --fc3-supercell-dim auto \
  --kappa-mesh auto \
  --method rta \
  --temperatures 300 \
  --overwrite
```

### DPA/DeepMD Workflows

Harmonic phonons only, using `run`:

```bash
phonoflow run \
  --input-path examples/Si.vasp \
  --model-path /path/to/DPA4-Neo-OMat24-v20260528_rc.pt \
  --backend dpa4neo \
  --outdir work/dpa_phonon \
  --supercell-dim auto \
  --mesh auto \
  --overwrite
```

Harmonic phonons only, using `single`:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --model-path /path/to/DPA4-Neo-OMat24-v20260528_rc.pt \
  --backend dpa4neo \
  --outdir work/dpa_phonon_single \
  --supercell-dim auto \
  --mesh auto \
  --overwrite
```

Thermal conductivity, using `run`:

```bash
phonoflow run \
  --input-path examples/Si.vasp \
  --model-path /path/to/DPA4-Neo-OMat24-v20260528_rc.pt \
  --backend dpa4neo \
  --outdir work/dpa_kappa \
  --supercell-dim auto \
  --mesh auto \
  --compute-kappa \
  --fc3-method finite-displacement \
  --fc3-supercell-dim auto \
  --kappa-mesh auto \
  --method rta \
  --temperatures 300 \
  --overwrite
```

Thermal conductivity, using `single`:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --model-path /path/to/DPA4-Neo-OMat24-v20260528_rc.pt \
  --backend dpa4neo \
  --outdir work/dpa_kappa_single \
  --supercell-dim auto \
  --mesh auto \
  --compute-kappa \
  --fc3-method finite-displacement \
  --fc3-supercell-dim auto \
  --kappa-mesh auto \
  --method rta \
  --temperatures 300 \
  --overwrite
```

### Compare Models

Compare NEP/NEP89 models:

```bash
phonoflow compare-models \
  --input-path examples/Si.vasp \
  --outdir work/compare_nep \
  --model-label nep_a --backend calorine --model-path /path/to/nep-a.txt \
  --model-label nep_b --backend calorine --model-path /path/to/nep-b.txt \
  --mesh auto \
  --overwrite
```

Compare DPA/DeepMD models:

```bash
phonoflow compare-models \
  --input-path examples/Si.vasp \
  --outdir work/compare_dpa \
  --model-label dpa31 --backend dpa31 --model-path /path/to/DPA-3.1-3M.pt \
  --model-label dpa4neo --backend dpa4neo --model-path /path/to/DPA4-Neo-OMat24-v20260528_rc.pt \
  --mesh auto \
  --overwrite
```

## Commands

- `phonoflow version`: print the package version.
- `phonoflow --help-all`: print all workflow parameters, defaults, and
  purposes.
- `phonoflow doctor`: check required and optional runtime dependencies.
- `phonoflow init-config`: write a full YAML configuration template.
- `phonoflow single`: run from a YAML config plus CLI overrides.
- `phonoflow run`: run one structure with required `--input-path` and
  `--model-path`; this is the simplest production entry point.
- `phonoflow compare-models`: run one to three model workflows and compare
  outputs.
- `phonoflow read-result`: summarize an existing `result.json`.
- `phonoflow batch`: batch workflow skeleton for a directory of structures.

## Documentation

- [Docs index](docs/index.md)
- [CLI reference](docs/cli.md)
- [Configuration reference](docs/configuration.md)
- [Output files](docs/outputs.md)
- [Architecture](docs/architecture.md)
- [Testing](docs/testing.md)

## Tests

```bash
python -m compileall src tests scripts
PYTHONPATH=src python -m pytest tests -q
```

Optional backend tests may skip when their runtime stack is not installed. The
dummy backend validates the baseline workflow without private model files.

## Repository Boundary

The public repository is for CLI source, tests, public docs, and the small Si
example. Generated calculations, HDF5 artifacts, PNG plots, model files,
private notes, archives, database files, and private application files are
intentionally kept out of Git.
