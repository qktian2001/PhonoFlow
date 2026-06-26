# Installing Calorine

Calorine CPUNEP is the currently recommended real NEP/NEP89 backend for
PhonoFlow.

## Recommended Environment

Use WSL/Linux and run from the Linux filesystem working copy:

```bash
cd ~/PhonoFlow
```

Avoid running real Calorine/Phonopy calculations from `/mnt/c/...`, because
editable installs, compiled extensions, and heavy file IO can be slower or less
reliable there.

## Install

```bash
python -m pip install -e ".[dev,calorine]"
```

or:

```bash
python -m pip install -e ".[dev]"
python -m pip install calorine
```

## Check CPUNEP

```bash
python -c "from calorine.calculators import CPUNEP; print('Calorine CPUNEP OK')"
python -m phonoflow doctor --verbose
```

## Si Validation

```bash
python -m phonoflow single \
  --input-path examples/Si.vasp \
  --model-path nep89_potential/nep89_20250409.txt \
  --outdir results/Si_calorine_validation \
  --backend calorine \
  --relax \
  --supercell-dim 2 2 2 \
  --displacement 0.01 \
  --fmax 1e-5 \
  --max-steps 500 \
  --band auto

python scripts/validate_output.py results/Si_calorine_validation
```

## Common Issues

- `calorine` is not installed: run `python -m pip install calorine`.
- CPUNEP API unavailable: upgrade Calorine or confirm that
  `from calorine.calculators import CPUNEP` works.
- NEP potential cannot be read: the NEP89 file may not be compatible with the
  installed Calorine version. Try a standard `nep.txt`, upgrade Calorine, or
  verify that the file is a valid NEP/NEP89 potential.
- `force_constants.hdf5` is missing: inspect `run.log` for the displacement or
  force-evaluation step that failed.
- `phonon_band.png` is missing or empty: check that `band.yaml` was generated
  and contains frequencies.
- Small imaginary frequencies near the configured threshold can be numerical
  noise. Large imaginary frequencies may indicate a real instability or an
  input/potential mismatch.
