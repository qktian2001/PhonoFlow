# Output Files

PhonoFlow writes run artifacts into `outdir`. Exact files depend on selected
options, backend availability, and whether `--dry-run` is used.

## Always Important

- `result.json`: machine-readable run summary.
- `summary.txt`: human-readable summary.
- `resolved_settings.json` and `resolved_settings.yaml`: final settings after
  defaults, config, aliases, and CLI overrides.
- `run_command.txt`: command used for the run.
- `timing_breakdown.json` and timing summaries when timing data is available.

## Structure and Symmetry

- `structure_provenance.json`: hashes, formula, atom count, and structure
  identity metadata.
- `spacegroup_report.json` and `spacegroup_report.txt`: spglib/space-group
  report.
- Relaxed structure files when relaxation is enabled.

## Harmonic Phonons

- `phonopy.yaml`: Phonopy state.
- `force_constants.hdf5`: FC2 data.
- `FORCE_CONSTANTS`: Phonopy text FC2 export when enabled.
- `FORCE_CONSTANTS_2ND`: ShengBTE-style FC2 text export when enabled.
- `band.yaml`: band structure data.
- `phonon_band.csv` and `phonon_band.dat`: tabular band output.
- Band metadata JSON files and generated band plots.
- `phonon_dos.dat`: DOS output when DOS is enabled.
- `phonon_group_velocity.csv`: group velocity output when available.

## Thermal Conductivity

When `compute_kappa` is enabled:

- `phono3py_params.yaml`: Phono3py state.
- `fc2.hdf5` and `fc3.hdf5`: second- and third-order force constants for
  Phono3py.
- `kappa-*.hdf5`: Phono3py kappa output.
- `thermal_conductivity.csv`: extracted kappa table.
- `phonon_lifetime.csv`: lifetime table when available.
- Thermal plots when plotting is enabled.

## Diagnostics

When `save_force_audit` is enabled, PhonoFlow writes finite-displacement force
diagnostics including force hashes, statistics, and raw arrays. These files are
for reproducibility and debugging, not for Git tracking.

## Git Policy

Run outputs, model files, HDF5 artifacts, plots, command records, summaries,
and local archives are generated artifacts. They should stay outside the public
repository history unless a specific small fixture is intentionally added.
