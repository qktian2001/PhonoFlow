# CLI Quickstart

Install PhonoFlow in editable mode:

```bash
python -m pip install -e .
```

Check the command surface:

```bash
phonoflow --help
phonoflow doctor --verbose
```

Run a dummy calculation that does not need a model file:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --backend dummy \
  --outdir work/si_dummy \
  --overwrite
```

Run a Calorine-backed phonon calculation with a user-supplied potential:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --backend calorine \
  --model-path /path/to/nep-model.txt \
  --outdir work/si_calorine \
  --supercell-dim auto \
  --mesh auto \
  --overwrite
```

Add thermal conductivity:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --backend calorine \
  --model-path /path/to/nep-model.txt \
  --outdir work/si_kappa \
  --compute-kappa \
  --fc3-method finite-displacement \
  --kappa-mesh auto \
  --method rta \
  --temperatures 300 \
  --overwrite
```

Generated outputs should stay outside Git-tracked files.
