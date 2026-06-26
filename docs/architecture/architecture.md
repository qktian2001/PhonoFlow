# Architecture

PhonoFlow is organized as a small, testable Python package. The public
`phonoflow` package and CLI are available for new users, while the historical
`phonoflow` package path remains as the implementation namespace for
backward compatibility during the rename.

## Main Modules

- `config`: Pydantic configuration model and YAML helpers.
- `io`: structure and result input/output helpers built around `pathlib`.
- `calculators`: backend abstraction plus dummy, Calorine, and GPUMD backends.
- `workflow`: single and batch workflow orchestration.
- `analysis`: stability and future phonon-analysis helpers.
- `reporting`: JSON, text, and CSV report writers.
- `plotting`: reserved plotting interfaces for band and DOS outputs.

## Backend Design

Backends implement `CalculatorBackend`:

- `check_available()`
- `calculate_energy_forces(atoms)`
- `relax_structure(atoms, outdir, config)`

The dummy backend is fully runnable and returns zero energy and zero forces for
workflow tests. Calorine CPUNEP is the only supported real NEP/NEP89 backend in
the current version. GPUMD remains a placeholder. Optional backend imports or
executable checks are isolated so a user without those tools can still run the
package, CLI, doctor command, and tests with the dummy backend.

## Extension Plan

The workflow modules separate relaxation, displacement, force evaluation, and
phonon post-processing. Calorine plugs into the real single workflow through an
ASE-compatible calculator, so future real backends should not require
duplicating the Phonopy pipeline.
