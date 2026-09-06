# wq_framework

**An open-source and extensible platform for benchmarking machine learning models for water quality prediction.**

`wq_framework` `wq_framework` is a config-driven, plugin-based framework that enables researchers and practitioners in the water resources sector to compare machine learning models across different regions and datasets. The current release represents an initial reference implementation: the framework predicts nine hydrochemical parameters (TDS, pH, HCO₃, Cl, SO₄, Ca, Mg, Na, K) at monitoring stations using tabular measurements combined with ERA5-Land climate covariates retrieved from Google Earth Engine. The long-term goal is to establish a community-driven benchmark that can be extended to related domains, such as drought and flood forecasting.

## Vision: why a benchmark?

Water resource managers and decision-makers are often hesitant to rely on AI-based predictions. A model that performs well in one region may fail in another, and published results are rarely reproducible across different datasets. This project bridges this gap by providing a shared, transparent platform where any algorithm can be trained, evaluated, and compared under identical conditions.

The long-term vision is a benchmark that:

- **Compares models fairly** — All models run the same process of preprocessing , feature selection, evaluation, and reporting pipeline. Therefore, differences in performance reflect the characteristics of the models themselves, rather than their surrounding code.
- **Supports generalization studies** — the same framework can be applied to different regions, parameters, and data sources, making it possible to determine which algorithms are appropriate for which regions.
- **Builds trust** — every stage reports what it did and why, giving practitioners a transparent, auditable record rather than a black box.
- **Extends beyond water quality** — At its core, the architecture of this system is domain-agnostic; therefore, the platform can be adapted for other forecasting tasks, such as drought or flood prediction.

## Features

- **Pluggable architecture**: Each stage of the pipeline—including preprocessing, satellite data integration, feature selection, modeling, evaluation metrics, and output generation—is a plugin point based on the Strategy+Registry pattern. New methods are added to the system not by modifying the main codebase, but through class registration.
- **Config-driven**: All methods and parameters are configured in a YAML configuration file, rather than being hardcoded.
- **Flexible data sources**: Climate variables can be obtained automatically from Google Earth Engine or from an offline file—both methods use a common interface.
- **Data-preserving defaults**: Outlier detection and validation at the cell level (rather than row-level removal), designed for the small datasets common in this field.
- **Transparent reporting**: Each stage generates a report on what has been removed, modified, or selected, along with the reason for doing so.

## Architecture

The pipeline is:

```
Input (stations + measurements)
  → Schema validation
  → Preprocessing (outlier detection)
  → Satellite data integration (climate covariates)
  → Feature selection
  → Modeling + evaluation
  → Output generation
```

Each stage consists of an **abstract base class** with a fixed interface. Concrete implementations register themselves under a specific name (e.g. `@register_model("catboost")`) and are selected via a YAML configuration file. The core **pipeline** code never changes when a new method is added.

**Current reference implementation** (v1, in progress):

- **Models**: TabPFN, TabM, xRFM, CatBoost — chosen for strong performance on small tabular datasets.
- **Satellite covariates**: precipitation, temperature, potential evapotranspiration, solar radiation, and soil moisture (ERA5-Land, configurable buffer radii).
- **Evaluation**: R², MSE, RMSE, MAE, NSE, plus a composite LP ranking score.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick start

```bash
python examples/run_example.py
pytest tests/ -v
```

See [`docs/input_schema.md`](docs/input_schema.md) for the input format specification and [`config/default_config.yaml`](config/default_config.yaml) for the reference configuration with all options documented.

## Roadmap

- **v1 (current)**: complete the pipeline — satellite caching, lag features, orchestration, feature selection, modeling, metrics, and output generation.
- **v1.1/v2**: lightweight web interface (Streamlit) for non-programmer practitioners.
- **Future**: Model persistence — preserving the best-performing model and providing a dedicated inference module for operational use; extensible to other forecasting tasks.

## License

MIT — see [LICENSE](LICENSE).
