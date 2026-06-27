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

PhonoFlow is a Python command-line package. Python 3.10 or newer is required;
Python 3.11 is a good default for a fresh environment. Linux or WSL is
recommended for production runs, especially for DeepMD/DPA, Phono3py, HiPhive,
and WTE workflows.

### 1. Get the Source Code

```bash
git clone https://github.com/qktian2001/PhonoFlow.git
cd PhonoFlow
```

### 2. Create a Clean Python Environment

Conda or mamba is recommended because scientific Python, Phono3py, DeepMD-kit,
and compiled optional dependencies are easier to keep isolated.

```bash
conda create -n phonoflow python=3.11 pip git -c conda-forge
conda activate phonoflow
python -m pip install --upgrade pip setuptools wheel
```

On Ubuntu/WSL, install basic build tools if pip has to build any dependency from
source:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git
```

### 3. Install the Core CLI

This installs the PhonoFlow console command plus the core dependencies declared
in `pyproject.toml`: NumPy, SciPy, pandas, matplotlib, ASE, Phonopy, spglib,
SeekPath, Pydantic, Typer, Rich, and PyYAML.

```bash
python -m pip install -e .
```

Verify the baseline command-line installation:

```bash
phonoflow --help
phonoflow version
phonoflow doctor --verbose
```

The baseline install supports the `dummy` backend and command/config
validation. Real NEP, thermal-conductivity, HiPhive, WTE, and DPA/DeepMD runs
need the optional stacks below.

### 4. Install Developer Tools

Install this when you want to run the public test suite or contribute changes:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

### 5. Install NEP/NEP89 Support with Calorine CPUNEP

Calorine CPUNEP is the production backend used by PhonoFlow for NEP/NEP89 model
files such as `nep89_20250409.txt`.

```bash
python -m pip install -e ".[calorine]"
```

Check that the API required by PhonoFlow is importable:

```bash
python - <<'PY'
from calorine.calculators import CPUNEP
print("Calorine CPUNEP import OK")
PY

phonoflow doctor --verbose
```

Run a NEP/NEP89 calculation by passing your own potential file:

```bash
phonoflow run \
  --input-path examples/Si.vasp \
  --model-path /path/to/nep-model.txt
```

### 6. Install Thermal-Conductivity Support

Thermal conductivity, FC3 finite displacements, Phono3py RTA/LBTE, kappa HDF5
parsing, and lifetime extraction require the `thermal` extra:

```bash
python -m pip install -e ".[thermal]"
```

Verify Phono3py and HDF5 support:

```bash
python - <<'PY'
import h5py
import phono3py
print("phono3py", phono3py.__version__)
print("h5py", h5py.__version__)
PY
```

Example finite-displacement thermal run:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --model-path /path/to/nep-model.txt \
  --backend calorine \
  --outdir work/nep_kappa \
  --compute-kappa \
  --fc3-method finite-displacement \
  --fc3-supercell-dim auto \
  --kappa-mesh auto \
  --method rta \
  --temperatures 300 \
  --overwrite
```

### 7. Install HiPhive FC3 Fitting Support

HiPhive is optional. Use it when you want `--fc3-method hiphive` instead of
direct Phono3py finite-displacement FC3 generation.

```bash
python -m pip install -e ".[hiphive]"
```

Verify the import:

```bash
python - <<'PY'
import hiphive
print("hiphive import OK")
PY
```

Example HiPhive thermal run:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --model-path /path/to/nep-model.txt \
  --backend calorine \
  --outdir work/nep_hiphive_kappa \
  --compute-kappa \
  --fc3-method hiphive \
  --n-structures 200 \
  --rattle-std 0.02 \
  --cutoffs 5.0 4.0 \
  --kappa-mesh auto \
  --method rta \
  --temperatures 300 \
  --overwrite
```

### 8. Install DeepMD-kit for DPA/DeepMD Models

DPA and DeepMD workflows are available through the `deepmd` backend and DPA
aliases (`dpa31`, `dpa32`, `dpa33`, `dpa4neo`) when DeepMD-kit is installed and
you provide compatible model files (`.pt`, `.pth`, or `.pb`). DeepMD-kit is not
declared as a PhonoFlow extra because CPU/GPU, CUDA, MPI, PyTorch, and DPA model
compatibility must match the user's environment.

For a CPU-oriented environment, start with:

```bash
python -m pip install deepmd-kit
```

For GPU/CUDA DPA runs, install the DeepMD-kit build that matches your CUDA,
driver, PyTorch, MPI, and model requirements. If you see messages such as
`Cannot find libcudart.so.12`, the DeepMD-kit build expects a CUDA runtime that
is not visible in the current environment; either install the matching CUDA
runtime or use a CPU-compatible DeepMD-kit build.

Verify the DeepMD ASE calculator used by PhonoFlow:

```bash
python - <<'PY'
from deepmd.calculator import DP
print("DeepMD DP calculator import OK")
PY
```

Example DPA/DeepMD run:

```bash
phonoflow run \
  --input-path examples/Si.vasp \
  --model-path /path/to/DPA4-Neo-OMat24-v20260528_rc.pt
```

For explicit DPA options:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --model-path /path/to/DPA4-Neo-OMat24-v20260528_rc.pt \
  --backend dpa4neo \
  --deepmd-device cpu \
  --outdir work/dpa_phonon \
  --overwrite
```

PhonoFlow defaults DPA/DeepMD structure relaxation to NEP89/Calorine when
relaxation is enabled, unless you explicitly request DPA relaxation with
`--allow-dpa-relax`. For DPA-only environments, use `--no-relax` or provide a
valid NEP/NEP89 relaxation model through the relevant relax options.

### 9. Install WTE / Wigner Transport Support

Wigner transport is optional and is requested with `--wigner true`. PhonoFlow
uses the external `phono3py-wte` plugin and checks that the plugin registers
`wte-rta` and `wte-lbte` with Phono3py before enabling WTE.

Install Phono3py first:

```bash
python -m pip install -e ".[thermal]"
```

Then install the WTE plugin from source in the same Python environment:

```bash
mkdir -p .vendor
git clone https://github.com/MSimoncelli/phono3py-wte.git .vendor/phono3py-wte
python -m pip install -e .vendor/phono3py-wte
```

If the WTE plugin source you use needs compatibility edits for your Phonopy or
Phono3py version, apply them before the editable install. Verify that PhonoFlow
sees WTE as available:

```bash
python - <<'PY'
from phonoflow.thermal.wte_backend import get_wte_backend_capability
capability = get_wte_backend_capability()
print(capability["available"])
print(capability.get("registered_methods"))
PY
```

Example WTE run:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --model-path /path/to/nep-model.txt \
  --backend calorine \
  --outdir work/nep_wte \
  --compute-kappa \
  --wigner true \
  --method rta \
  --temperatures 300 \
  --overwrite
```

### 10. GPUMD Status

PhonoFlow 1.0 includes a GPUMD-oriented backend module and `phonoflow doctor`
checks whether the `gpumd` executable is on `PATH`, but real GPUMD force
evaluation and relaxation are not implemented in this public CLI release.

```bash
which gpumd || echo "gpumd command not on PATH"
phonoflow doctor --verbose
```

### 11. Recommended Complete Install

For NEP/NEP89 phonons, finite-displacement thermal conductivity, HiPhive, tests,
and docs validation in one environment:

```bash
python -m pip install -e ".[dev,calorine,thermal,hiphive]"
phonoflow doctor --verbose
python -m pytest tests -q
```

Add DeepMD-kit and phono3py-wte only when you need DPA/DeepMD or Wigner
transport. Model files are not included in the public repository; pass your own
NEP/NEP89 or DeepMD/DPA model with `--model-path`.

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

Use `run` when you want the shortest direct CLI command. It requires only
`--input-path` and `--model-path`; PhonoFlow infers the backend from the model
file (`.txt` for NEP/NEP89 through Calorine, `.pt/.pth/.pb` for DeepMD/DPA) and
uses automatic defaults for output directory, supercell, mesh, and harmonic
phonon settings. Use `single` when you want the same workflow with a YAML
config and explicit CLI overrides.

### NEP/NEP89 Workflows

Harmonic phonons only, using minimal `run` direct CLI mode:

```bash
phonoflow run \
  --input-path examples/Si.vasp \
  --model-path /path/to/nep-model.txt
```

Harmonic phonons only, using `single` config-compatible mode:

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

Thermal conductivity, using `single` with explicit second- and third-order
settings:

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

Harmonic phonons only, using minimal `run` direct CLI mode:

```bash
phonoflow run \
  --input-path examples/Si.vasp \
  --model-path /path/to/DPA4-Neo-OMat24-v20260528_rc.pt
```

Harmonic phonons only, using `single` config-compatible mode:

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

Thermal conductivity, using `single` with explicit second- and third-order
settings:

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
- `phonoflow single`: run one workflow from an optional YAML config plus CLI
  overrides.
- `phonoflow run`: run one structure directly from the required `--input-path`
  and `--model-path`, with backend and workflow defaults inferred automatically.
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
