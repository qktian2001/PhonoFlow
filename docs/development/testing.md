# Development And Testing

Use an isolated Python environment with the development dependencies installed:

```bash
python -m pip install -e ".[dev]"
```

Compile the Python package, tests, and scripts:

```bash
python -m compileall src tests scripts
```

Run the test suite:

```bash
PYTHONPATH=src python -m pytest tests -q
```

Optional backend tests may skip or fail when their external runtime stack is not
installed. Do not change scientific workflow logic merely to satisfy a missing
optional dependency.

Before publishing a public export, verify:

```bash
git diff --check
```

Also inspect the export directory for generated artifacts, model files, private
runtime configuration, and local-only archives before uploading.
