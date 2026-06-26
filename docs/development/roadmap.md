# Roadmap

## v0.1 Complete

- Standard Python package layout.
- Typer CLI.
- Pydantic configuration.
- ASE structure IO.
- Calculator backend abstraction.
- Dummy backend.
- GPUMD placeholder backend.
- Single and batch workflow skeletons.
- Stability analysis.
- JSON, text, and CSV reports.
- README, docs, examples, and pytest tests.

## v0.2-p3 Complete

- Calorine CPUNEP backend.
- Calorine Si validation workflow.
- `doctor --verbose` diagnostics for Calorine CPUNEP.

## v0.2-p4 Complete

- Auto defaults.
- Improved band plot labels and separators.
- Primitive-matrix control.
- ASR and force-constant symmetrization.
- Phonon DOS.
- `read-result` command.
- `summary.txt` output.
- More robust result validation.

## v0.2-p5 Complete

- Removed legacy backend code from the current real workflow.
- Calorine CPUNEP is the only supported real NEP/NEP89 backend.
- `backend=auto` resolves only to Calorine and never to dummy.
- Stabilized Calorine error messages and result metadata.
- Improved band plotting from Phonopy distances and path segments.
- Cleaned README, CLI help, doctor, and tests around the Calorine mainline.

## v0.2-p6 Complete

- Refactored the band plot chain into data, label, I/O, export, and plotting modules.
- Unified Gamma and boundary tick label handling across band data and band path JSON.
- Added `FORCE_CONSTANTS` and `FORCE_CONSTANTS_2ND` text FC2 export.
- Clarified `fc_method=finite-displacement` as the current force-constant route.
- Reserved `fc_method=hiphive` for future HiPhive fitting; not implemented.

## v0.2-p7 Complete

- Print and save resolved settings with `user/default/auto` source tracking.
- Add `resolved_settings.json`, `resolved_settings.yaml`, and `run_command.txt`.
- Add `--dry-run`, `--print-config`, `--overwrite`, and simple `--resume` behavior.
- Record input/model SHA256 hashes and software versions in `result.json`.
- Improve `summary.txt`, `read-result`, and validation around reproducibility metadata.

## v0.2-p8 Complete

- Make the default relaxation mode bulk-oriented atom + cell relaxation.
- Tighten default relaxation to `fmax=1e-5` and `max_steps=2000`.
- Keep `--no-relax-cell` for fixed-cell atom-position relaxation and `--no-relax` for skipping relaxation.
- Add heuristic structure classification with vacuum-direction warnings for 2D, slab, surface, and interface-like inputs.
- Record relaxation mode, initial/final cell, volume change, final stress, and warnings in resolved settings, `result.json`, `summary.txt`, and `read-result`.

## v0.2-p9 Complete

- Use separate automatic targets: 15 Å for FC2/harmonic calculations and 10 Å
  for FC3/thermal calculations.
- Record auto supercell dimensions, lengths, atom counts, bounds, and warnings.
- Keep detected vacuum-like directions at multiplier 1 during auto supercell inference.
- Add spglib initial/final space-group diagnostics.
- Write `spacegroup_report.json` and `spacegroup_report.txt`.
- Show space-group comparison in `result.json`, `summary.txt`, `read-result`, and validation.

## v0.3

- Real batch calculation workflow.
- Resume behavior for completed real calculations.
- Batch-level aggregation of real phonon results.

## v0.4

- GPUMD backend.
- Command-line GPUMD force evaluation and relaxation integration.

## v0.5+

- phono3py integration.
- Lattice thermal conductivity workflows.
- Web/API interface.
