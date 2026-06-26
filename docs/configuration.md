# Configuration Reference

Configuration is defined by `WorkflowConfig` in `src/phonoflow/config.py`.
`phonoflow init-config --out config.yaml` writes a complete YAML template.
CLI options override config-file values when both are provided.

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
input_path: structure.vasp
model_path: /path/to/nep-model.txt
backend: calorine
outdir: work/phonon
supercell_dim: auto
mesh: auto
relax: true
relax_cell: true
overwrite: true
```

Thermal-conductivity run:

```yaml
input_path: structure.vasp
model_path: /path/to/nep-model.txt
backend: calorine
outdir: work/kappa
compute_kappa: true
fc3_method: finite-displacement
fc3_supercell_dim: auto
kappa_mesh: auto
kappa_method: rta
temperatures: [300.0]
overwrite: true
```

## Parameter Inventory

The table lists the PhonoFlow 1.0 config fields and defaults from the current
code. Use the live `init-config` output as the final source when editing a run.

| Field | Default | Purpose |
| --- | --- | --- |
| `input_path` | `null` | Single input structure path. |
| `input_dir` | `null` | Batch input directory. |
| `outdir` | `null` | Output directory; command code resolves a default when omitted. |
| `model_path` | `null` | User-provided NEP, DeepMD, or compatible model path. |
| `backend` | `auto` | Calculator backend selection. |
| `backend_alias` | `null` | Optional backend alias metadata. |
| `dpa_model_name` | `null` | Optional named DPA model selector. |
| `supercell_dim` | `auto` | FC2 supercell dimensions or automatic inference. |
| `mesh` | `auto` | Harmonic/DOS q mesh; `q_mesh` is accepted as an alias. |
| `target_supercell_length` | `15.0` | Target length for automatic FC2 supercell inference. |
| `min_supercell_dim` | `1` | Minimum automatic FC2 multiplier. |
| `max_supercell_dim` | `6` | Maximum automatic FC2 multiplier. |
| `max_supercell_atoms` | `1000` | Maximum atoms in automatic FC2 supercell. |
| `relax` | `true` | Enable structure relaxation. |
| `relax_cell` | `true` | Relax cell and positions together. |
| `displacement` | `0.01` | Harmonic finite-displacement amplitude. |
| `fmax` | `1e-5` | Relaxation force threshold in eV/A. |
| `max_steps` | `2000` | Maximum relaxation optimizer steps. |
| `optimizer` | `FIRE` | ASE optimizer name. |
| `relax_backend` | `auto` | Backend used for relaxation. |
| `relax_model_path` | `null` | Optional relaxation-specific model path. |
| `allow_dpa_relax` | `false` | Explicitly permit DPA/DeepMD relaxation. |
| `band` | `auto` | Legacy band selector. |
| `kpath_mode` | `auto` | `auto`, `3d_seekpath`, `2d_ase`, or `custom`. |
| `band_npoints` | `101` | Points per band segment. |
| `bandpath_symprec` | `1e-5` | SeekPath/2D ASE band-path precision. |
| `bandpath_with_time_reversal` | `false` | Use time-reversal reduction for 3D SeekPath. |
| `fc_method` | `finite-displacement` | Harmonic force-constant method. |
| `compute_kappa` | `false` | Enable FC3 and thermal conductivity. |
| `fc3_method` | `finite-displacement` | FC3 method: finite displacement or HiPhive. |
| `kappa_method` | `rta` | Thermal solver method: RTA or LBTE. |
| `wigner` | `false` | Request Wigner transport when available. |
| `temperatures` | `[300.0]` | Thermal conductivity temperatures in K. |
| `kappa_mesh` | `auto` | Phono3py kappa mesh. |
| `fc3_supercell_dim` | `auto` | FC3 supercell dimensions or automatic inference. |
| `fc3_target_supercell_length` | `10.0` | Target length for automatic FC3 supercell inference. |
| `max_fc3_supercell_atoms` | `256` | Maximum atoms in automatic FC3 supercell. |
| `fc3_displacement` | `0.03` | FC3 displacement amplitude. |
| `fc3_cutoff_pair_distance` | `null` | Optional FC3 pair cutoff. |
| `max_fc3_displacements` | `null` | Optional smoke-test cap on FC3 displacements. |
| `phono3py_symprec` | `1e-5` | Phono3py symmetry precision. |
| `phono3py_cutoff_frequency` | `1e-4` | Phono3py cutoff frequency in THz. |
| `phono3py_plusminus` | `auto` | Phono3py plus/minus displacement mode. |
| `phono3py_diagonal` | `false` | Use diagonal FC3 displacements. |
| `phono3py_symmetry` | `true` | Use Phono3py symmetry reduction. |
| `phono3py_mesh_symmetry` | `true` | Use mesh symmetry for kappa. |
| `phono3py_isotope` | `false` | Enable isotope scattering. |
| `boundary_mfp` | `0.0` | Boundary mean free path; zero disables it. |
| `cutoff_pair_distance` | `0.0` | Phono3py pair cutoff; zero disables it. |
| `phono3py_symmetrize_fc2` | `true` | Apply official Phono3py FC2 symmetrization. |
| `phono3py_symmetrize_fc3` | `true` | Apply official Phono3py FC3 symmetrization. |
| `deepmd_reuse_calculator` | `true` | Reuse one DeepMD calculator in force loops. |
| `deepmd_force_backend` | `ase` | DeepMD force path: ASE or DeePMD direct. |
| `deepmd_device` | `cpu` | DeepMD runtime device. |
| `deepmd_model_head` | `null` | Optional multitask DeepMD model head. |
| `deepmd_deterministic` | `false` | Best-effort deterministic DeepMD environment. |
| `save_force_audit` | `false` | Save finite-displacement force diagnostics. |
| `n_structures` | `200` | HiPhive rattle structure count. |
| `rattle_std` | `0.02` | HiPhive rattle standard deviation. |
| `cutoffs` | `[5.0, 4.0]` | HiPhive cutoff radii. |
| `min_dist` | `1.8` | HiPhive minimum interatomic distance. |
| `primitive_matrix` | `P` | Phonopy primitive matrix setting. |
| `dos` | `true` | Compute DOS outputs. |
| `asr` | `true` | Apply acoustic sum rule where possible. |
| `symmetrize_fc` | `true` | Symmetrize FC2 where possible. |
| `export_fc2_text` | `true` | Export Phonopy and ShengBTE-style FC2 text files. |
| `fc2_text_name` | `FORCE_CONSTANTS` | Phonopy FC2 text filename. |
| `shengbte_fc2_name` | `FORCE_CONSTANTS_2ND` | ShengBTE-style FC2 text filename. |
| `plot_dpi` | `300` | Plot resolution. |
| `plot_format` | `png` | Plot format; current release writes PNG. |
| `imag_threshold` | `-0.1` | Imaginary-mode stability threshold in THz. |
| `phonopy_symprec` | `1e-5` | Phonopy symmetry precision. |
| `angle_tolerance` | `-1.0` | spglib angle tolerance; `-1.0` means default. |
| `max_workers` | `1` | Reserved worker count field. |
| `dry_run` | `false` | Resolve settings without heavy calculation. |
| `print_config` | `false` | Print resolved settings. |
| `overwrite` | `false` | Allow replacing existing output directories. |
| `resume` | `false` | Reuse complete successful outputs. |
| `log_level` | `INFO` | Logging verbosity. |

## Accepted Aliases

- `q_mesh` sets both `mesh` and `kappa_mesh`.
- `dos_mesh` sets `mesh` when no explicit `mesh` is provided.
- `symprec` is accepted as a deprecated alias for `phonopy_symprec`.
- `phono3py_fc2_asr` is accepted as a deprecated alias for
  `phono3py_symmetrize_fc2`.
- `pynep` is intentionally rejected; use `calorine` for real NEP/NEP89 runs.
