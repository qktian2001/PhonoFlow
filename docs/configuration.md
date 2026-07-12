# Configuration Reference

This page documents the local command-line configuration surface implemented by
`WorkflowConfig` in `src/phonoflow/config.py`. Generate the live template with:

```bash
phonoflow init-config --out config.yaml
```

CLI options override YAML values when both are provided. The runtime source of
truth is:

```bash
phonoflow --help-all
phonoflow single --help
phonoflow compare-models --help
```

## Important Defaults

- Harmonic/DOS mesh: `mesh: auto` resolves to `21 21 21` for bulk structures.
  For a detected single-vacuum-axis 2D slab it resolves to a denser in-plane
  mesh such as `51 51 1`.
- Thermal-conductivity mesh: `kappa_mesh: auto` resolves independently to
  `11 11 11`.
- FC2 auto supercell: target length `15.0 A`, per-axis bounds `1..6`, and
  default atom cap `200`.
- FC3 auto supercell: target length `10.0 A`, per-axis bounds inherited from
  the FC2 auto-supercell logic where applicable, and default atom cap `200`.
- If the active lattice lengths are equal, auto supercell inference keeps equal
  multipliers when possible, for example cubic cells prefer `4 4 4` rather than
  asymmetric alternatives with the same atom cap.
- DPA/DeepMD model aliases resolve through the local model registry. Generic
  `backend: dpa` requires an explicit `model_path`.
- The optional local CPU queue is disabled by default and only affects CLI runs
  when `cpu_queue_enabled: true` or `--cpu-queue` is used.

## Minimal Examples

Dummy smoke test:

```yaml
input_path: examples/Si.vasp
outdir: work/si_dummy
backend: dummy
overwrite: true
```

Real NEP/NEP89 harmonic run:

```yaml
input_path: examples/Si.vasp
model_path: /path/to/nep-model.txt
backend: calorine
outdir: work/si_nep_phonon
supercell_dim: auto
mesh: auto
relax: true
relax_cell: true
overwrite: true
```

Real thermal-conductivity run:

```yaml
input_path: examples/Si.vasp
model_path: /path/to/nep-model.txt
backend: calorine
outdir: work/si_nep_kappa
compute_kappa: true
fc3_method: finite-displacement
fc3_supercell_dim: auto
kappa_mesh: auto
kappa_method: rta
temperatures: [300.0]
force_workers: 24
force_parallel_backend: process
force_chunk_size: 1
overwrite: true
```

## Parameter Inventory

The table below is organized by local CLI behavior. Defaults match the current
`WorkflowConfig()` values in the project code.

### Paths And Backend Selection

| Field | Default | CLI option | Purpose |
| --- | --- | --- | --- |
| `input_path` | `null` | `--input-path` | Single input structure path. |
| `input_dir` | `null` | batch/config only | Batch input directory. |
| `outdir` | `null` | `--outdir` | Output directory. If omitted, command code resolves a backend-specific default under `results/`. |
| `model_path` | `null` | `--model-path` | User-provided NEP, DeepMD, DPA, or compatible model path. Required for real non-dummy backends unless a bundled DPA alias resolves. |
| `backend` | `auto` | `--backend` | Calculator backend. Accepted values include `auto`, `dummy`, `calorine`, `gpumd`, `deepmd`, `dpa`, `dpa3`, `dpa4`, `dpa31`, `dpa32`, `dpa33`, `dpa4neo`. |
| `backend_alias` | `null` | internal metadata | Records the user-facing backend alias after DPA/default resolution. |
| `dpa_model_name` | `null` | internal metadata | Records the resolved DPA model name. |

### FC2, Harmonic Phonons, Band, DOS

| Field | Default | CLI option | Purpose |
| --- | --- | --- | --- |
| `supercell_dim` | `auto` | `--supercell-dim` | FC2 supercell dimensions as three integers, or `auto`. |
| `target_supercell_length` | `15.0` | `--target-supercell-length` | Target length in Angstrom for automatic FC2 supercell inference. |
| `min_supercell_dim` | `1` | `--min-supercell-dim` | Minimum automatic FC2 multiplier. |
| `max_supercell_dim` | `6` | `--max-supercell-dim` | Maximum automatic FC2 multiplier. |
| `max_supercell_atoms` | `300` | `--max-supercell-atoms` | Maximum atoms allowed in an automatically inferred FC2 supercell. |
| `displacement` | `0.01` | `--displacement` | Harmonic finite-displacement amplitude in Angstrom. |
| `fc_method` | `finite-displacement` | `--fc-method` | Harmonic force-constant method. `finite-displacement` is the production path; `hiphive` is reserved where supported. |
| `mesh` | `auto` | `--mesh` | Harmonic/DOS Phonopy mesh. `auto` is independent from `kappa_mesh`. |
| `primitive_matrix` | `P` | `--primitive-matrix` | Phonopy primitive matrix: `P`, `identity`, `none`, or `auto`. |
| `dos` | `true` | `--dos` / `--no-dos` | Compute phonon DOS outputs. |
| `asr` | `true` | `--asr` / `--no-asr` | Apply acoustic sum rule where possible. |
| `symmetrize_fc` | `true` | `--symmetrize-fc` / `--no-symmetrize-fc` | Symmetrize FC2 where possible. |
| `export_fc2_text` | `true` | `--export-fc2-text` / `--no-export-fc2-text` | Export Phonopy and ShengBTE-style FC2 text files. |
| `fc2_text_name` | `FORCE_CONSTANTS` | `--fc2-text-name` | Phonopy FC2 text filename. |
| `shengbte_fc2_name` | `FORCE_CONSTANTS_2ND` | `--shengbte-fc2-name` | ShengBTE-style FC2 filename. |
| `band` | `auto` | `--band` | Legacy band selector; normally leave `auto`. |
| `kpath_mode` | `auto` | `--kpath-mode` | K-path generator: `auto`, `3d_seekpath`, `2d_ase`, or `custom`. |
| `band_npoints` | `101` | config only | Points per band segment. |
| `bandpath_symprec` | `1e-5` | `--bandpath-symprec` | SeekPath precision and ASE 2D bandpath epsilon floor. |
| `bandpath_with_time_reversal` | `false` | `--bandpath-with-time-reversal` / `--no-bandpath-with-time-reversal` | Use time-reversal reduction for 3D SeekPath band paths. |
| `phonopy_symprec` | `1e-5` | `--phonopy-symprec` | Phonopy symmetry precision. Deprecated alias: `symprec`. |
| `angle_tolerance` | `-1.0` | `--angle-tolerance` | spglib angle tolerance; `-1.0` uses the library default. |
| `imag_threshold` | `-0.1` | `--imag-threshold` | Imaginary-mode stability threshold in THz. |

### Relaxation

| Field | Default | CLI option | Purpose |
| --- | --- | --- | --- |
| `relax` | `true` | `--relax` / `--no-relax` | Enable or skip structure relaxation. |
| `relax_cell` | `true` | `--relax-cell` / `--no-relax-cell` | Relax cell and positions together, or positions only. |
| `fmax` | `1e-5` | `--fmax` | Relaxation force threshold in eV/A. |
| `max_steps` | `2000` | `--max-steps` | Maximum optimizer steps. |
| `optimizer` | `FIRE` | `--optimizer` | ASE optimizer name, commonly `FIRE` or `LBFGS`. |
| `relax_backend` | `auto` | `--relax-backend` | Relaxation backend. `auto` prefers the workflow backend policy; DPA relaxation requires explicit opt-in. |
| `relax_model_path` | `null` | `--relax-model-path` | Optional relaxation-specific model path. |
| `allow_dpa_relax` | `false` | `--allow-dpa-relax` / `--no-allow-dpa-relax` | Explicitly permit DPA/DeepMD to perform relaxation. |

### FC3 And Thermal Conductivity

| Field | Default | CLI option | Purpose |
| --- | --- | --- | --- |
| `compute_kappa` | `false` | `--compute-kappa` / `--no-compute-kappa` | Enable FC3 and lattice thermal conductivity. |
| `fc3_method` | `finite-displacement` | `--fc3-method` | FC3 method: `finite-displacement` or `hiphive`. |
| `kappa_method` | `rta` | `--method`, `--kappa-method`, `--lbte`, `--rta` | Phono3py BTE solver method: RTA or LBTE. |
| `wigner` | `false` | `--wigner true/false` | Request Wigner transport when supported by the local Phono3py stack. |
| `temperatures` | `[300.0]` | `--temperatures` | One or more temperatures in K. |
| `kappa_mesh` | `auto` | `--kappa-mesh` | Phono3py thermal-conductivity mesh. Independent from `mesh`. |
| `fc3_supercell_dim` | `auto` | `--fc3-supercell-dim` | FC3 supercell dimensions as three integers, or `auto`. |
| `fc3_target_supercell_length` | `10.0` | `--fc3-target-supercell-length` | Target length in Angstrom for automatic FC3 supercell inference. |
| `max_fc3_supercell_atoms` | `300` | `--max-fc3-supercell-atoms` | Maximum atoms allowed in an automatically inferred FC3 supercell. |
| `fc3_displacement` | `0.03` | `--fc3-displacement` | FC3 finite-displacement amplitude in Angstrom. |
| `fc3_cutoff_pair_distance` | `null` | `--fc3-cutoff-pair-distance` | Optional phono3py FC3 pair cutoff distance. |
| `max_fc3_displacements` | `null` | `--max-fc3-displacements` | Smoke-test cap on FC3 displaced structures. Do not use for production convergence. |
| `phono3py_symprec` | `1e-5` | `--phono3py-symprec` | Phono3py symmetry precision. |
| `phono3py_cutoff_frequency` | `1e-4` | `--phono3py-cutoff-frequency` | Phono3py cutoff frequency in THz. |
| `phono3py_plusminus` | `auto` | `--phono3py-plusminus` | phono3py plus/minus displacement mode: `auto`, `true`, or `false`. |
| `phono3py_diagonal` | `false` | `--phono3py-diagonal` / `--no-phono3py-diagonal` | Use diagonal FC3 displacements. |
| `phono3py_symmetry` | `true` | `--phono3py-symmetry` / `--no-phono3py-symmetry` | Use phono3py symmetry reduction for FC3 displacements. |
| `phono3py_mesh_symmetry` | `true` | `--phono3py-mesh-symmetry` / `--no-phono3py-mesh-symmetry` | Use mesh symmetry in thermal-conductivity calculations. |
| `phono3py_isotope` | `false` | `--isotope` / `--no-isotope` | Enable isotope scattering. |
| `boundary_mfp` | `0.0` | `--boundary-mfp` | Boundary mean free path; `0` disables it. |
| `cutoff_pair_distance` | `0.0` | `--cutoff-pair-distance` | Additional phono3py pair cutoff; `0` disables it. |
| `phono3py_symmetrize_fc2` | `true` | `--phono3py-symmetrize-fc2` / `--no-phono3py-symmetrize-fc2` | Apply official phono3py FC2 symmetrization for kappa workflows. |
| `phono3py_symmetrize_fc3` | `true` | `--phono3py-symmetrize-fc3` / `--no-phono3py-symmetrize-fc3` | Apply official phono3py FC3 symmetrization. |

### DeepMD And DPA

| Field | Default | CLI option | Purpose |
| --- | --- | --- | --- |
| `deepmd_reuse_calculator` | `true` | `--deepmd-reuse-calculator` / `--no-deepmd-reuse-calculator` | Reuse one ASE/DeepMD calculator inside force loops where possible. |
| `deepmd_force_backend` | `ase` | `--deepmd-force-backend` | DeepMD force path: `ase` or `deeppot`. |
| `deepmd_device` | `cpu` | `--deepmd-device` | DeepMD runtime device: `auto`, `cpu`, or `cuda`. |
| `deepmd_model_head` | `null` | `--deepmd-model-head` | Optional multitask DeepMD/DPA model head. |
| `deepmd_torch_threads` | `null` | `--deepmd-torch-threads` | Torch intra-op threads per DeepMD force worker. Use `1` with many `force_workers` on CPU. |
| `deepmd_deterministic` | `false` | `--deepmd-deterministic` / `--no-deepmd-deterministic` | Best-effort reproducibility policy for DeepMD subprocesses. |

### Local CPU Scheduling And Queueing

| Field | Default | CLI option | Purpose |
| --- | --- | --- | --- |
| `max_concurrent_jobs` | `1` | config/resource metadata | Maximum concurrently running jobs used for resource-budget estimation. |
| `batch_workers` | `1` | config/resource metadata | CLI batch worker count used for resource-budget estimation. |
| `force_workers` | `1` | `--force-workers` | Maximum displaced structures evaluated concurrently inside one workflow. |
| `force_parallel_backend` | `serial` | `--force-parallel-backend` | Force-loop backend: `origin`, `direct`, `serial`, `process`, or `worker_queue`. |
| `force_chunk_size` | `null` | `--force-chunk-size` | Internal scheduler chunk size. `null` lets the backend-aware policy choose; current recommended finite-displacement CPU setting is `1`. |
| `force_max_pending_tasks` | `null` | `--force-max-pending-tasks` | Bound on pending scheduler chunks/tasks. `null` uses scheduler defaults. |
| `cpu_queue_enabled` | `false` | `--cpu-queue` | Enable optional file-locked local CPU-slot queue. |
| `cpu_queue_total_slots` | `null` | `--cpu-queue-total-slots` | Total local CPU slots managed by the queue. |
| `cpu_queue_max_running_jobs` | `1` | `--cpu-queue-max-running-jobs` | Maximum jobs allowed to hold queue leases simultaneously. |
| `cpu_queue_job_slots` | `1` | `--cpu-queue-job-slots` | CPU slots requested by this CLI job. |
| `cpu_queue_state_dir` | `null` | `--cpu-queue-state-dir` | Directory for file-locked CPU queue state. |
| `cpu_queue_timeout` | `null` | `--cpu-queue-timeout` | Maximum seconds to wait for a CPU queue lease. `null` waits according to scheduler defaults. |
| `auto_cpu_budget` | `true` | `--auto-cpu-budget` / `--no-auto-cpu-budget` | Record estimated CPU parallelism and oversubscription warnings. |
| `save_force_audit` | `false` | `--save-force-audit` / `--no-save-force-audit` | Save force hashes, statistics, and raw-array diagnostics. |

### HiPhive FC3 Fitting

| Field | Default | CLI option | Purpose |
| --- | --- | --- | --- |
| `n_structures` | `200` | `--n-structures` | HiPhive rattle structure count. |
| `rattle_std` | `0.02` | `--rattle-std` | HiPhive rattle standard deviation. |
| `cutoffs` | `[5.0, 4.0]` | `--cutoffs` | HiPhive cutoff radii, for example `--cutoffs 5.0 4.0`. |
| `min_dist` | `1.8` | `--min-dist` | HiPhive minimum interatomic distance. |

### Output, Runtime, And Control

| Field | Default | CLI option | Purpose |
| --- | --- | --- | --- |
| `plot_dpi` | `300` | config only | Plot resolution. |
| `plot_format` | `png` | config only | Plot format. Current workflows write PNG plots. |
| `max_workers` | `1` | legacy/config metadata | Reserved worker-count field kept for compatibility. |
| `dry_run` | `false` | `--dry-run` | Resolve settings and write metadata without heavy calculation. |
| `print_config` | `false` | `--print-config` | Print resolved settings. |
| `overwrite` | `false` | `--overwrite` | Allow replacing files in an existing output directory. |
| `resume` | `false` | `--resume` | Reuse complete successful outputs where supported. |
| `log_level` | `INFO` | `--log-level` | Logging verbosity. |

## Accepted Legacy Aliases

- `dos_mesh` sets `mesh` when no explicit `mesh` is provided.
- `symprec` is accepted as a deprecated alias for `phonopy_symprec`.
- `phono3py_fc2_asr` is accepted as a deprecated alias for
  `phono3py_symmetrize_fc2`.
- `pynep` is intentionally rejected. Use `backend: calorine` for real NEP/NEP89
  runs or `backend: dummy` for smoke tests.

`q_mesh` is no longer part of the configuration chain. Use `mesh` for
harmonic/DOS calculations and `kappa_mesh` for thermal conductivity.
