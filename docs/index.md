# PhonoFlow 1.0 Documentation

PhonoFlow 1.0 is a command-line workflow for phonons and lattice thermal
transport with machine-learning interatomic potentials.

The important entry points are:

- `phonoflow run` for a direct single-structure calculation.
- `phonoflow single` for config-driven runs.
- `phonoflow compare-models` for one-structure, multi-model comparison.
- `phonoflow doctor` for runtime checks.
- `phonoflow init-config` for a complete editable YAML template.
- `phonoflow read-result` for summarizing an existing `result.json`.

Core implementation areas:

- `src/phonoflow/cli.py`: Typer command surface and CLI override plumbing.
- `src/phonoflow/config.py`: validated `WorkflowConfig` defaults and aliases.
- `src/phonoflow/workflow/`: relaxation, displacement, force evaluation, FC2,
  provenance, reports, output policy, and pipeline orchestration.
- `src/phonoflow/thermal/`: FC3, Phono3py, lifetime, and kappa helpers.
- `src/phonoflow/calculators/`: dummy, Calorine, DeepMD, and GPUMD backend
  integration.
- `src/phonoflow/kpath/`: 3D SeekPath and 2D ASE band-path selection.
- `src/phonoflow/reporting/`: JSON, text, CSV, timing, and summary outputs.

Read next:

- [CLI reference](cli.md)
- [Configuration reference](configuration.md)
- [Output files](outputs.md)
- [Architecture](architecture.md)
- [Testing](testing.md)
