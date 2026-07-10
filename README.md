# PhonoFlow 1.0

PhonoFlow 1.0 is the initial command-line release of a phonon and lattice
thermal-transport workflow for machine-learning interatomic potentials. It
connects ASE, Phonopy, Phono3py, Calorine CPUNEP, and optional DeepMD-kit
backends behind reproducible CLI commands.

The public repository contains the CLI engine, tests, public documentation, and
small public structure examples for smoke tests and quick-start commands. It
does not contain model weights, run results, private application code, local
archives, or private runtime configuration.

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
  overrides for production calculations. FC2 and FC3 auto supercells now obey
  both the target length and the default 200-atom cap; cubic cells keep equal
  multipliers when possible.
- Independent harmonic/DOS and thermal meshes: `--mesh auto` resolves the
  Phonopy/DOS mesh, while `--kappa-mesh auto` resolves the Phono3py
  thermal-conductivity mesh.
- Local CPU scheduling for finite-displacement force evaluation:
  `--force-workers`, `--force-parallel-backend`, `--force-chunk-size`, and the
  optional file-locked `--cpu-queue` help control CPU usage without changing
  the physical model.
- Multi-model comparison through `compare-models`, including shared workflow
  settings and per-model backend/model paths.
- Reproducibility artifacts: resolved settings, command record, timing,
  structure provenance, space-group report, force-audit diagnostics, JSON/text
  summaries, and optional FC2 text exports.
- Origin/direct/process force-evaluation modes are available for comparison and
  debugging: `origin` preserves the original Phono3py-style FC3 force loop,
  `direct` runs PhonoFlow's no-process path, and `process` enables
  process-level displacement parallelism.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `docs/` | PhonoFlow 1.0 CLI documentation, including command usage, configuration, outputs, architecture, and testing notes. |
| `examples/` | Minimal public structure examples for smoke tests and quick-start commands, including `Si.vasp`, `SiC.vasp`, `BAs.vasp`, `Diamond.vasp`, and `GaN.vasp`. |
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

The workflow below was validated in a fresh `phonoflow_test` conda environment
on Ubuntu 24.04 / WSL2 with Python 3.11.15. The tested path installed the core
CLI, then the recommended scientific extras, and verified `phonoflow doctor`,
a dry-run CLI job, Calorine CPUNEP availability, and the focused public tests.

### Fast Path: Recommended Local Install

For most Linux users who want NEP/NEP89 phonons, finite-displacement thermal
conductivity, HiPhive fitting, and local tests in one environment, use this
single flow:

```bash
conda create -n phonoflow python=3.11 pip git -c conda-forge -y
conda activate phonoflow

git clone https://github.com/qktian2001/PhonoFlow.git
cd PhonoFlow

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,calorine,thermal,hiphive]"

phonoflow --help
phonoflow version
phonoflow doctor --verbose
python -m pytest tests/test_cli_version.py tests/test_config.py tests/test_auto_supercell.py tests/test_cpu_queue_cli_options.py -q
```

Expected result:

- `phonoflow doctor --verbose` should report Python, NumPy, ASE, Phonopy,
  SeekPath, matplotlib, and Calorine CPUNEP as available.
- `GPUMD executable` may remain optional/missing unless you installed GPUMD
  separately.
- The focused test command should finish with all tests passing.

Use the component-by-component instructions below when you want a smaller core
environment, DeepMD/DPA support, WTE support, or development-only tooling.

### 1. Get the Source Code

```bash
git clone https://github.com/qktian2001/PhonoFlow.git
cd PhonoFlow
```

### 2. Create a Clean Python Environment

Conda or mamba is recommended because scientific Python, Phono3py, DeepMD-kit,
and compiled optional dependencies are easier to keep isolated.

```bash
conda create -n phonoflow python=3.11 pip git -c conda-forge -y
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

For NEP/NEP89 phonons, finite-displacement thermal conductivity, HiPhive, and
tests in one environment, this is the same tested install used by the fast path:

```bash
python -m pip install -e ".[dev,calorine,thermal,hiphive]"
phonoflow doctor --verbose
python -m pytest tests/test_cli_version.py tests/test_config.py tests/test_auto_supercell.py tests/test_cpu_queue_cli_options.py -q
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

Important current defaults:

- `--mesh auto` controls the harmonic Phonopy/DOS mesh and resolves to the core
  3D default `21 21 21` unless slab/vacuum detection selects a 2D-style mesh.
- `--kappa-mesh auto` controls the Phono3py thermal-conductivity mesh and
  resolves to the core default `11 11 11`.
- FC2 `--supercell-dim auto` uses `--target-supercell-length` with the default
  200-atom auto-supercell cap.
- FC3 `--fc3-supercell-dim auto` uses `--fc3-target-supercell-length` with the
  default 200-atom auto-supercell cap.
- For cubic cells, automatic FC2/FC3 supercells keep the three multipliers equal
  when that satisfies the length and atom-count constraints.

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
  --force-workers 24 \
  --force-parallel-backend process \
  --force-chunk-size 1 \
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
  --force-workers 24 \
  --force-parallel-backend process \
  --force-chunk-size 1 \
  --deepmd-device cpu \
  --deepmd-torch-threads 1 \
  --method rta \
  --temperatures 300 \
  --overwrite
```

For a 36-core workstation, a common single-job starting point is:

```bash
--force-workers 36 --force-parallel-backend process --force-chunk-size 1
```

For CPU DeepMD/DPA runs, start with one Torch thread per force worker:

```bash
--deepmd-device cpu --deepmd-torch-threads 1
```

These resource flags change scheduling and throughput only; they are not
scientific model parameters.

To compare scheduling paths on a small system, use:

```bash
# Original Phono3py-style FC3 force loop, useful as a reference baseline.
--force-parallel-backend origin --force-workers 1

# PhonoFlow direct no-process path.
--force-parallel-backend direct --force-workers 1

# PhonoFlow process-level displacement parallelism.
--force-parallel-backend process --force-workers 24 --force-chunk-size 1
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

Compare runs accept the same thermal and force-scheduling controls used by
`single`, and apply them to each child workflow:

```bash
phonoflow compare-models \
  --input-path examples/Si.vasp \
  --outdir work/compare_kappa \
  --model-label nep89 --backend calorine --model-path /path/to/nep-model.txt \
  --model-label dpa4neo --backend dpa4neo --model-path /path/to/DPA4-Neo-OMat24-v20260528_rc.pt \
  --compute-kappa \
  --mesh auto \
  --kappa-mesh auto \
  --method rta \
  --temperatures 300 \
  --force-workers 24 \
  --force-parallel-backend process \
  --force-chunk-size 1 \
  --deepmd-device cpu \
  --deepmd-torch-threads 1 \
  --overwrite
```

When several local CLI jobs need to share one workstation, the optional CPU
queue can reserve file-locked local CPU slots before the workflow starts:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --model-path /path/to/nep-model.txt \
  --compute-kappa \
  --cpu-queue \
  --cpu-queue-total-slots 36 \
  --cpu-queue-max-running-jobs 2 \
  --cpu-queue-job-slots 18 \
  --force-workers 18 \
  --force-parallel-backend process \
  --force-chunk-size 1 \
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

Local resource controls available on `single`, `run`, and `compare-models`
include:

- `--force-workers`: number of displaced structures evaluated concurrently.
- `--force-parallel-backend`: `origin`, `direct`, `serial`, `process`, or
  `worker_queue`.
- `--force-chunk-size`: scheduler chunk size; for current NEP/Calorine and
  CPU DeepMD/DPA FC3 force loops, `1` is the recommended starting point.
- `--force-max-pending-tasks`: bound queued scheduler work when enabled.
- `--deepmd-device`, `--deepmd-torch-threads`, and
  `--deepmd-deterministic`: DeepMD/DPA runtime controls.
- `--cpu-queue`, `--cpu-queue-total-slots`,
  `--cpu-queue-max-running-jobs`, `--cpu-queue-job-slots`,
  `--cpu-queue-state-dir`, and `--cpu-queue-timeout`: optional local CPU slot
  reservation before a CLI workflow starts.

## Full Local Configuration Parameter Table

The table below mirrors the current `WorkflowConfig` defaults. For the expanded
explanations and examples, see [Configuration reference](docs/configuration.md).
Generate the live YAML template with `phonoflow init-config --out config.yaml`.

### Paths And Backends

| Field | Default | CLI option / scope | Purpose |
| --- | --- | --- | --- |
| `input_path` | `null` | `--input-path` | Single input structure path. |
| `input_dir` | `null` | config/batch | Batch input directory. |
| `outdir` | `null` | `--outdir` | Output directory; resolved automatically when omitted. |
| `model_path` | `null` | `--model-path` | NEP, DeepMD, DPA, or compatible model path. |
| `backend` | `auto` | `--backend` | `auto`, `dummy`, `calorine`, `gpumd`, `deepmd`, `dpa`, `dpa3`, `dpa4`, `dpa31`, `dpa32`, `dpa33`, or `dpa4neo`. |
| `backend_alias` | `null` | metadata | Resolved user-facing backend alias. |
| `dpa_model_name` | `null` | metadata | Resolved DPA model name. |

### FC2, Harmonic Phonons, Band, DOS

| Field | Default | CLI option / scope | Purpose |
| --- | --- | --- | --- |
| `supercell_dim` | `auto` | `--supercell-dim` | FC2 supercell triplet or `auto`. |
| `target_supercell_length` | `15.0` | `--target-supercell-length` | FC2 auto target length in Angstrom. |
| `min_supercell_dim` | `1` | `--min-supercell-dim` | Minimum auto multiplier. |
| `max_supercell_dim` | `6` | `--max-supercell-dim` | Maximum auto multiplier. |
| `max_supercell_atoms` | `200` | `--max-supercell-atoms` | Atom cap for auto FC2 supercells. |
| `displacement` | `0.01` | `--displacement` | FC2 finite-displacement amplitude in Angstrom. |
| `fc_method` | `finite-displacement` | `--fc-method` | Harmonic force-constant method. |
| `mesh` | `auto` | `--mesh` | Harmonic/DOS mesh; independent from `kappa_mesh`. |
| `primitive_matrix` | `P` | `--primitive-matrix` | Phonopy primitive matrix setting. |
| `dos` | `true` | `--dos` / `--no-dos` | Compute phonon DOS outputs. |
| `asr` | `true` | `--asr` / `--no-asr` | Apply acoustic sum rule where possible. |
| `symmetrize_fc` | `true` | `--symmetrize-fc` / `--no-symmetrize-fc` | Symmetrize FC2 where possible. |
| `export_fc2_text` | `true` | `--export-fc2-text` / `--no-export-fc2-text` | Export Phonopy and ShengBTE-style FC2 text files. |
| `fc2_text_name` | `FORCE_CONSTANTS` | `--fc2-text-name` | Phonopy FC2 text filename. |
| `shengbte_fc2_name` | `FORCE_CONSTANTS_2ND` | `--shengbte-fc2-name` | ShengBTE-style FC2 filename. |
| `band` | `auto` | `--band` | Legacy band selector; usually leave `auto`. |
| `kpath_mode` | `auto` | `--kpath-mode` | `auto`, `3d_seekpath`, `2d_ase`, or `custom`. |
| `band_npoints` | `101` | config | Points per band segment. |
| `bandpath_symprec` | `1e-5` | `--bandpath-symprec` | SeekPath and 2D ASE precision. |
| `bandpath_with_time_reversal` | `false` | `--bandpath-with-time-reversal` | Use time-reversal reduction for 3D paths. |
| `phonopy_symprec` | `1e-5` | `--phonopy-symprec` | Phonopy symmetry precision. |
| `angle_tolerance` | `-1.0` | `--angle-tolerance` | spglib angle tolerance; `-1.0` means default. |
| `imag_threshold` | `-0.1` | `--imag-threshold` | Imaginary-mode stability threshold in THz. |

### Relaxation

| Field | Default | CLI option / scope | Purpose |
| --- | --- | --- | --- |
| `relax` | `true` | `--relax` / `--no-relax` | Enable or skip relaxation. |
| `relax_cell` | `true` | `--relax-cell` / `--no-relax-cell` | Relax cell and positions, or positions only. |
| `fmax` | `1e-5` | `--fmax` | Relaxation force threshold in eV/A. |
| `max_steps` | `2000` | `--max-steps` | Maximum optimizer steps. |
| `optimizer` | `FIRE` | `--optimizer` | ASE optimizer name, commonly `FIRE` or `LBFGS`. |
| `relax_backend` | `auto` | `--relax-backend` | Relaxation backend policy. |
| `relax_model_path` | `null` | `--relax-model-path` | Optional relaxation-specific model path. |
| `allow_dpa_relax` | `false` | `--allow-dpa-relax` | Explicitly permit DPA/DeepMD relaxation. |

### FC3 And Thermal Conductivity

| Field | Default | CLI option / scope | Purpose |
| --- | --- | --- | --- |
| `compute_kappa` | `false` | `--compute-kappa` | Enable FC3 and thermal conductivity. |
| `fc3_method` | `finite-displacement` | `--fc3-method` | `finite-displacement` or `hiphive`. |
| `kappa_method` | `rta` | `--method`, `--lbte`, `--rta` | Phono3py solver method. |
| `wigner` | `false` | `--wigner true/false` | Request Wigner transport when available. |
| `temperatures` | `[300.0]` | `--temperatures` | One or more temperatures in K. |
| `kappa_mesh` | `auto` | `--kappa-mesh` | Thermal-conductivity mesh; `auto` resolves to `11 11 11`. |
| `fc3_supercell_dim` | `auto` | `--fc3-supercell-dim` | FC3 supercell triplet or `auto`. |
| `fc3_target_supercell_length` | `10.0` | `--fc3-target-supercell-length` | FC3 auto target length in Angstrom. |
| `max_fc3_supercell_atoms` | `200` | `--max-fc3-supercell-atoms` | Atom cap for auto FC3 supercells. |
| `fc3_displacement` | `0.03` | `--fc3-displacement` | FC3 displacement amplitude in Angstrom. |
| `fc3_cutoff_pair_distance` | `null` | `--fc3-cutoff-pair-distance` | Optional phono3py FC3 pair cutoff. |
| `max_fc3_displacements` | `null` | `--max-fc3-displacements` | Smoke-test cap; not for production convergence. |
| `phono3py_symprec` | `1e-5` | `--phono3py-symprec` | phono3py symmetry precision. |
| `phono3py_cutoff_frequency` | `1e-4` | `--phono3py-cutoff-frequency` | phono3py cutoff frequency in THz. |
| `phono3py_plusminus` | `auto` | `--phono3py-plusminus` | `auto`, `true`, or `false`. |
| `phono3py_diagonal` | `false` | `--phono3py-diagonal` | Use diagonal FC3 displacements. |
| `phono3py_symmetry` | `true` | `--phono3py-symmetry` | Use phono3py symmetry reduction. |
| `phono3py_mesh_symmetry` | `true` | `--phono3py-mesh-symmetry` | Use mesh symmetry in kappa. |
| `phono3py_isotope` | `false` | `--isotope` / `--no-isotope` | Enable isotope scattering. |
| `boundary_mfp` | `0.0` | `--boundary-mfp` | Boundary mean free path; zero disables it. |
| `cutoff_pair_distance` | `0.0` | `--cutoff-pair-distance` | Additional pair cutoff; zero disables it. |
| `phono3py_symmetrize_fc2` | `true` | `--phono3py-symmetrize-fc2` | Apply official phono3py FC2 symmetrization. |
| `phono3py_symmetrize_fc3` | `true` | `--phono3py-symmetrize-fc3` | Apply official phono3py FC3 symmetrization. |

### DeepMD And DPA

| Field | Default | CLI option / scope | Purpose |
| --- | --- | --- | --- |
| `deepmd_reuse_calculator` | `true` | `--deepmd-reuse-calculator` | Reuse one ASE/DeepMD calculator inside force loops. |
| `deepmd_force_backend` | `ase` | `--deepmd-force-backend` | `ase` or `deeppot`. |
| `deepmd_device` | `cpu` | `--deepmd-device` | `auto`, `cpu`, or `cuda`. |
| `deepmd_model_head` | `null` | `--deepmd-model-head` | Optional multitask model head. |
| `deepmd_torch_threads` | `null` | `--deepmd-torch-threads` | Torch threads per DeepMD force worker. |
| `deepmd_deterministic` | `false` | `--deepmd-deterministic` | Best-effort deterministic DeepMD subprocess policy. |

### Local CPU Scheduling And Queue

| Field | Default | CLI option / scope | Purpose |
| --- | --- | --- | --- |
| `max_concurrent_jobs` | `1` | config/resource metadata | Maximum concurrent jobs for CPU budget estimation. |
| `batch_workers` | `1` | config/resource metadata | Batch worker count for CPU budget estimation. |
| `force_workers` | `1` | `--force-workers` | Displaced structures evaluated concurrently inside one workflow. |
| `force_parallel_backend` | `serial` | `--force-parallel-backend` | `origin`, `direct`, `serial`, `process`, or `worker_queue`. |
| `force_chunk_size` | `null` | `--force-chunk-size` | Scheduler chunk size; `1` is the current recommended CPU FC3 starting point. |
| `force_max_pending_tasks` | `null` | `--force-max-pending-tasks` | Bound pending scheduler chunks/tasks. |
| `cpu_queue_enabled` | `false` | `--cpu-queue` | Enable optional file-locked local CPU-slot queue. |
| `cpu_queue_total_slots` | `null` | `--cpu-queue-total-slots` | Total local CPU slots managed by the queue. |
| `cpu_queue_max_running_jobs` | `1` | `--cpu-queue-max-running-jobs` | Maximum jobs allowed to hold queue leases. |
| `cpu_queue_job_slots` | `1` | `--cpu-queue-job-slots` | CPU slots requested by this CLI job. |
| `cpu_queue_state_dir` | `null` | `--cpu-queue-state-dir` | Queue state directory. |
| `cpu_queue_timeout` | `null` | `--cpu-queue-timeout` | Maximum seconds to wait for a queue lease. |
| `auto_cpu_budget` | `true` | `--auto-cpu-budget` | Record CPU budget and oversubscription warnings. |
| `save_force_audit` | `false` | `--save-force-audit` | Save force hashes, statistics, and raw diagnostics. |

### HiPhive, Output, Runtime

| Field | Default | CLI option / scope | Purpose |
| --- | --- | --- | --- |
| `n_structures` | `200` | `--n-structures` | HiPhive rattle structure count. |
| `rattle_std` | `0.02` | `--rattle-std` | HiPhive rattle standard deviation. |
| `cutoffs` | `[5.0, 4.0]` | `--cutoffs` | HiPhive cutoff radii. |
| `min_dist` | `1.8` | `--min-dist` | HiPhive minimum interatomic distance. |
| `plot_dpi` | `300` | config | Plot resolution. |
| `plot_format` | `png` | config | Plot format. |
| `max_workers` | `1` | legacy/config metadata | Reserved compatibility worker-count field. |
| `dry_run` | `false` | `--dry-run` | Resolve settings without heavy calculation. |
| `print_config` | `false` | `--print-config` | Print resolved settings. |
| `overwrite` | `false` | `--overwrite` | Allow replacing files in an existing output directory. |
| `resume` | `false` | `--resume` | Reuse complete successful outputs where supported. |
| `log_level` | `INFO` | `--log-level` | Logging verbosity. |

Legacy config aliases are limited to compatibility helpers: `dos_mesh -> mesh`,
`symprec -> phonopy_symprec`, and
`phono3py_fc2_asr -> phono3py_symmetrize_fc2`. `q_mesh` is no longer part of the
configuration chain; use `mesh` for harmonic/DOS and `kappa_mesh` for thermal
conductivity.

## Documentation

- [Docs index](docs/index.md)
- [CLI reference](docs/cli.md)
- [Configuration reference](docs/configuration.md)
- [Output files](docs/outputs.md)
- [CPU queue scheduler](docs/cpu_queue_scheduler.md)
- [FC3 displacement acceleration](docs/fc3_displacement_acceleration.md)
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

The public repository is for CLI source, tests, public docs, and small public
structure examples. Generated calculations, HDF5 artifacts, PNG plots, model
files, private notes, archives, database files, and private application files
are intentionally kept out of Git.
