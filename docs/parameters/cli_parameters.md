# CLI Parameter Guide

This guide summarizes the public command-line parameters most users adjust.
Run `phonoflow single --help` for the exact option list in a local checkout.

## Inputs And Backend

- `--input-path`: structure file path, such as VASP/POSCAR or CIF.
- `--model-path`: user-provided model or potential file.
- `--backend`: calculator backend. Common values are `dummy`, `calorine`,
  `deepmd`, `dpa`, `dpa31`, `dpa32`, `dpa33`, and `dpa4neo`.
- `--outdir`: output directory for generated calculation files.
- `--config`: YAML file loaded before command-line overrides.

## Relaxation

- `--relax` / `--no-relax`: enable or skip geometry relaxation.
- `--relax-cell` / `--no-relax-cell`: relax cell vectors together with atoms,
  or keep the cell fixed.
- `--fmax`: force threshold for relaxation.
- `--max-steps`: maximum optimizer steps.
- `--optimizer`: ASE optimizer name, commonly `FIRE` or `LBFGS`.

## Harmonic Phonons

- `--supercell-dim`: FC2 supercell dimensions or `auto`.
- `--target-supercell-length`: target length used by automatic FC2 supercell
  selection.
- `--max-supercell-atoms`: atom-count cap for automatic FC2 supercells.
- `--displacement`: harmonic finite-displacement amplitude.
- `--mesh` / `--q-mesh`: q mesh used for DOS and harmonic post-processing.
- `--kpath-mode`: band-path mode, such as `auto`, `3d_seekpath`, `2d_ase`, or
  `custom`.
- `--phonopy-symprec`: phonopy symmetry precision.

## Thermal Transport

- `--compute-kappa`: enable FC3 and thermal conductivity.
- `--fc3-method`: `finite-displacement` or `hiphive`.
- `--fc3-supercell-dim`: FC3 supercell dimensions or `auto`.
- `--fc3-target-supercell-length`: target length for automatic FC3 supercell
  selection.
- `--fc3-displacement`: third-order displacement amplitude.
- `--kappa-mesh`: Phono3py thermal-conductivity mesh.
- `--method` / `--kappa-method`: `rta` or `lbte`.
- `--temperatures`: thermal-conductivity temperatures in K.
- `--wigner`: enable Wigner transport when the installed backend supports it.
- `--phono3py-symmetrize-fc2`: apply official Phono3py FC2 symmetrization in
  the finite-displacement route.
- `--phono3py-symmetrize-fc3`: apply official Phono3py FC3 symmetrization in
  the finite-displacement route.

## HiPhive Controls

These parameters apply when `--fc3-method hiphive` is selected:

- `--n-structures`: number of rattled structures.
- `--rattle-std`: displacement standard deviation for rattling.
- `--cutoffs`: cluster cutoffs.
- `--min-dist`: minimum allowed interatomic distance.

## Execution Controls

- `--dry-run`: resolve settings and write metadata without running the full
  workflow.
- `--print-config`: print resolved settings.
- `--overwrite`: allow reuse of an existing output directory.
- `--resume`: skip a completed run when reusable output is already present.
- `--log-level`: set runtime logging verbosity.
