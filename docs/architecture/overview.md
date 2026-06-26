# PhonoFlow Architecture Overview

PhonoFlow is organized as a Python package under `src/phonoflow`. The public
CLI layer parses user options, builds a `WorkflowConfig`, resolves defaults, and
dispatches to workflow modules that handle structure IO, relaxation, force
evaluation, harmonic phonons, and optional thermal transport.

## Main Layers

- `phonoflow.cli`: Typer command definitions for version, doctor, config
  generation, single runs, and model comparison.
- `phonoflow.config`: Pydantic configuration model, aliases, validators, and
  default values.
- `phonoflow.workflow`: orchestration for relaxation, displacement generation,
  phonon calculations, output policy, and metadata.
- `phonoflow.backends`: calculator integration points for dummy, Calorine
  CPUNEP, DeepMD/DPA, and placeholder backends.
- `phonoflow.thermal`: FC3 and Phono3py thermal-transport helpers.
- `phonoflow.reporting`: run summaries, timing reports, and result packaging.
- `phonoflow.analysis`: post-processing helpers for stability, band labels,
  space-group reporting, and structure provenance.

## Data Flow

1. CLI options and optional YAML config are merged into `WorkflowConfig`.
2. Structure input is read through ASE.
3. Optional relaxation is run with the selected calculator policy.
4. FC2 displacements are generated and evaluated.
5. Phonopy builds harmonic force constants and phonon outputs.
6. Optional FC3 and kappa workflows call Phono3py.
7. Reports and machine-readable artifacts are written to `--outdir`.

Generated outputs are not part of the public repository and should stay in a
local working directory.
