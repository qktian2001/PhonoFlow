# Architecture

PhonoFlow 1.0 is organized as a small CLI application around a validated
workflow configuration and explicit backend adapters.

## Flow

1. `phonoflow.cli` parses the command and CLI overrides.
2. `phonoflow.config.WorkflowConfig` validates values, applies aliases, and
   resolves defaults.
3. `phonoflow.workflow.pipeline.run_single_workflow` chooses output policy,
   records provenance, and orchestrates the run.
4. `phonoflow.workflow.relax` optionally relaxes the structure.
5. `phonoflow.workflow.displace` creates Phonopy displacement structures.
6. `phonoflow.calculators` evaluates forces through the selected backend.
7. `phonoflow_scheduler` optionally parallelizes finite-displacement force
   tasks while preserving displacement order.
8. `phonoflow.workflow.phonon` builds FC2, runs harmonic post-processing, and
   writes band/DOS/group-velocity/stability outputs.
9. `phonoflow.thermal` optionally builds FC3 and runs Phono3py kappa.
10. `phonoflow.reporting` writes JSON, text, CSV, and timing summaries.

## Backends

- `dummy`: deterministic test backend with no private model files.
- `calorine`: NEP/NEP89 via Calorine CPUNEP.
- `deepmd` and DPA aliases: optional DeepMD-kit integration.
- `gpumd`: GPUMD-oriented backend module.

Backend selection is controlled by `backend`, `model_path`, and optional
DeepMD/DPA fields. Real production accuracy depends on the supplied model and
backend installation.

## K-Path Selection

`kpath_mode=auto` selects an appropriate path:

- 3D structures use SeekPath.
- 2D slab-like structures can use the ASE 2D path helper.
- Custom path plumbing is preserved behind the `custom` mode.

The relevant controls are `bandpath_symprec`,
`bandpath_with_time_reversal`, `phonopy_symprec`, and `angle_tolerance`.

## Thermal Path

Thermal conductivity is disabled by default. When `compute_kappa=true`,
PhonoFlow resolves FC3 settings, creates third-order displacements, evaluates
forces, writes Phono3py parameters, and extracts kappa/lifetime outputs.

`max_fc3_displacements` exists for smoke tests and debugging. Production runs
should converge supercell size, displacement amplitude, q mesh, and the chosen
RTA/LBTE settings without that cap.

## Local CPU Scheduling

Finite-displacement force evaluation is routed through
`src/phonoflow/workflow/force_eval.py`, which adapts workflow structures into
order-preserving scheduler tasks. The independent scheduler package provides:

- `serial` and `direct` single-process paths,
- `process` multiprocessing with calculator reuse and bounded pending work,
- `worker_queue` prototype worker-pull scheduling,
- CPU resource-budget warnings,
- shared thread environment variables for BLAS/OpenMP/DeepMD,
- an optional file-locked CPU queue for coordinating multiple local CLI jobs.

The scheduler changes how independent force tasks are executed. It does not
change displacement generation, force-constant assembly, Phono3py equations, or
model parameters.

## Reproducibility

The pipeline records:

- resolved settings and their sources,
- command line,
- software/backend metadata,
- structure hashes,
- space-group report,
- timing,
- optional force-audit diagnostics.

These artifacts are intended to make a completed run inspectable without
depending on old chat notes or local manual records.
