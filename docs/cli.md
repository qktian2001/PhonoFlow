# CLI Reference

Use `phonoflow --help` for the live command list. The commands below describe
the PhonoFlow 1.0 command-line surface implemented in `src/phonoflow/cli.py`.

## Runtime Checks

```bash
phonoflow version
phonoflow doctor --verbose
```

`doctor` checks core packages and optional backend modules. Use it before a
production run when Calorine, Phono3py, HiPhive, or DeepMD behavior matters.

## Config Template

```bash
phonoflow init-config --out config.yaml
```

The generated file contains every `WorkflowConfig` field. It is suitable for
editing and then running through `phonoflow single --config config.yaml`.

## Single-Structure Runs

`run` is the most direct production entry point:

```bash
phonoflow run \
  --input-path structure.vasp \
  --model-path nep-model.txt \
  --backend calorine \
  --outdir work/run01 \
  --supercell-dim auto \
  --mesh auto \
  --overwrite
```

`single` supports a config file and the same important overrides:

```bash
phonoflow single \
  --config config.yaml \
  --input-path structure.vasp \
  --model-path nep-model.txt \
  --backend calorine \
  --outdir work/run01 \
  --overwrite
```

Useful control flags:

- `--dry-run`: resolve settings and write metadata without force/phonon work.
- `--print-config`: print resolved settings.
- `--overwrite`: intentionally reuse an existing output directory.
- `--resume`: skip recalculation when a complete successful result exists.

## Harmonic Phonons

Main harmonic controls:

- `--supercell-dim`: FC2 supercell as three integers, or `auto`.
- `--target-supercell-length`: automatic FC2 supercell target length.
- `--max-supercell-atoms`: upper atom-count limit for automatic FC2 cells.
- `--displacement`: finite-displacement amplitude.
- `--mesh` / `--q-mesh`: DOS and harmonic q mesh, or `auto`.
- `--primitive-matrix`: `P`, `identity`, `none`, or `auto`.
- `--asr` and `--symmetrize-fc`: harmonic force-constant cleanup controls.
- `--export-fc2-text`: write Phonopy and ShengBTE-style FC2 text files.

## Relaxation

Relaxation is enabled by default.

- `--relax` / `--no-relax`: enable or skip relaxation.
- `--relax-cell` / `--no-relax-cell`: relax the cell together with positions,
  or keep the cell fixed.
- `--fmax`: force threshold in eV/A.
- `--max-steps`: maximum optimizer steps.
- `--optimizer`: `FIRE` or `LBFGS`.
- `--relax-backend`: `auto`, `calorine`, `nep89`, `deepmd`, `dpa`, or `force`
  style choices supported by the backend code.
- `--allow-dpa-relax`: explicitly permit DPA/DeepMD relaxation.

## Band Path, DOS, and Stability

- `--kpath-mode`: `auto`, `3d_seekpath`, `2d_ase`, or `custom`.
- `--bandpath-symprec`: SeekPath precision and 2D ASE epsilon floor.
- `--bandpath-with-time-reversal`: enable 3D SeekPath time-reversal reduction.
- `--phonopy-symprec`: Phonopy symmetry precision.
- `--angle-tolerance`: spglib angle tolerance; `-1.0` uses spglib default.
- `--dos` / `--no-dos` is available through config fields.
- `--imag-threshold` controls the stability threshold in THz.

## Thermal Conductivity

Enable FC3 and kappa:

```bash
phonoflow run \
  --input-path structure.vasp \
  --model-path nep-model.txt \
  --backend calorine \
  --outdir work/kappa \
  --compute-kappa \
  --fc3-method finite-displacement \
  --fc3-supercell-dim auto \
  --kappa-mesh auto \
  --method rta \
  --temperatures 300
```

Important thermal controls:

- `--compute-kappa`: enable third-order force constants and kappa.
- `--fc3-method`: `finite-displacement` or `hiphive`.
- `--fc3-supercell-dim`: FC3 supercell as three integers, or `auto`.
- `--fc3-target-supercell-length`: automatic FC3 supercell target length.
- `--max-fc3-supercell-atoms`: atom limit for automatic FC3 cells.
- `--fc3-displacement`: FC3 displacement amplitude.
- `--max-fc3-displacements`: smoke-test cap; not for production convergence.
- `--kappa-mesh`: Phono3py kappa mesh, or `auto`.
- `--method` / `--kappa-method`: `rta` or `lbte`.
- `--lbte` / `--rta`: shortcut for the method.
- `--temperatures`: one or more temperatures in K.
- `--wigner true`: request Wigner transport when supported locally.
- `--isotope`: enable isotope scattering.
- `--boundary-mfp`: boundary mean free path; `0` disables it.
- `--cutoff-pair-distance`: FC3 pair cutoff; `0` disables it.
- `--phono3py-symmetrize-fc2` and `--phono3py-symmetrize-fc3`: official
  Phono3py force-constant symmetrization.

## DeepMD and DPA Options

Use these only when DeepMD-kit is installed and the model supports the selected
runtime behavior:

- `--backend deepmd`, `--backend dpa31`, `--backend dpa32`, `--backend dpa33`,
  or `--backend dpa4neo`.
- `--deepmd-force-backend`: `ase` or `deeppot`.
- `--deepmd-device`: `cpu`, `cuda`, or `auto`.
- `--deepmd-model-head`: multitask model head, for example `OMat24`.
- `--deepmd-reuse-calculator`: reuse one calculator in force loops.
- `--deepmd-deterministic`: best-effort deterministic environment settings.

## Compare Models

```bash
phonoflow compare-models \
  --input-path structure.vasp \
  --outdir work/compare \
  --model-label model_a --backend calorine --model-path model-a.txt \
  --model-label model_b --backend calorine --model-path model-b.txt \
  --mesh auto \
  --overwrite
```

The command shares workflow options across child runs and keeps model-specific
labels, backends, and model paths separate.

## Read Existing Results

```bash
phonoflow read-result --path work/run01/result.json
```

This prints a concise summary of a previous PhonoFlow result file.
