# wq_framework

Open-source, extensible framework for predicting river water quality
parameters using machine learning, satellite-derived climate covariates, and
a pluggable Strategy+Registry architecture.

**Status: early development (v1, in progress).** This repository currently
implements Stage 1 of the pipeline: input loading and schema validation.
Other stages (preprocessing, satellite integration, feature selection,
modeling, metrics, outputs) are planned but not yet implemented.

## Implemented so far

- `wq_framework.io` — loading `stations.csv` and the measurements file
  (CSV or Excel), validating file structure against `config/schema.yaml`,
  and the cell-level sanity-bound validation gate (invalid cells are
  nulled individually with user confirmation; nothing is silently dropped).

## Quick start

```bash
pip install -e ".[dev]"
python examples/run_example.py
pytest tests/ -v
```

## Input format

See [`docs/input_schema.md`](docs/input_schema.md) for the full format
specification, and `examples/` for sample files and `examples/run_example.py`
for usage.

## Design principles

- **Config-driven**: every stage's method and parameters are set via YAML,
  not hardcoded.
- **Extensible**: every stage is a Strategy+Registry plugin point — new
  methods are added by registering a new class, without touching core code.
- **Data-preserving**: defaults favor minimizing data loss (e.g. cell-level,
  not row-level, invalid-value handling) given the small datasets typical
  of this domain.

## License

TBD — repository is currently private during initial development.
