# Effects of Feature Scaling and PCA on Classical Machine Learning Models

This repository contains the source code, experiment configuration, per-block results, and analysis scripts for the **AIO Conquer 2026** project.

The project studies the effects of two preprocessing techniques on Fashion-MNIST:

- feature scaling with `StandardScaler`;
- dimensionality reduction with Principal Component Analysis (PCA).

Three classical machine learning models are evaluated:

- K-Nearest Neighbors;
- Logistic Regression;
- Random Forest.

The repository provides a configurable experimental pipeline, stores results for each data block, and supports statistical analysis. The complete methodology, results, and discussion are presented in the project report.

## Experimental Design

The official Fashion-MNIST train and test sets are kept separate and then independently divided into 10 stratified, non-overlapping train-test block pairs:

The main pipeline groups are:

| Experiment | Pipelines |
|---|---|
| Scaling | `raw`, `scaled` |
| PCA | `no_pca`, `pca_50`, `pca_75`, `pca_90` |

All experiment settings are managed in [`configs/experiment.yaml`](configs/experiment.yaml).

## Repository Structure

```text
.
├── configs/
│   └── experiment.yaml       # Central experiment configuration
├── data/
│   ├── blocks/               # Train-test block indices
│   └── processed/            # Downloaded and cached data
├── notebooks/                # Supporting notebooks for inspection or analysis
├── results/
│   ├── per_block/            # Per-block results and statistical reports
│   └── figures/              # Generated figures
├── src/
│   ├── analysis/             # Statistical tests and visualization
│   ├── data/                 # Dataset loading and block management
│   ├── experiments/          # Experiment execution
│   ├── models/               # Model definitions and parameter grids
│   ├── pipelines/            # Scaling and PCA pipelines
│   └── config.py             # Configuration and path handling
├── tests/                    # Data and pipeline tests
├── environment.yml           # Conda environment
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/AIVIETNAM-AIO-QuocAnh/aio-conquer-module-2.git
cd aio-conquer-module-2
```

Create the Conda environment:

```bash
conda env create -f environment.yml
```

Then activate the environment name defined in the `name:` field of `environment.yml`:

```bash
conda activate <environment-name>
```

## Running the Project

Run the following commands from the repository root.

### 1. Run tests

```bash
pytest -q
```

The tests check block integrity, dataset coverage, block sizes, and basic pipeline functionality.

### 2. Run experiments

```bash
python -m src.experiments.experiment
```

This entry point executes the experiment cases enabled in [`src/experiments/experiment.py`](src/experiments/experiment.py). Results are saved to:

```text
results/per_block/
```

The two main result files are:

```text
results/per_block/Scaling.csv
results/per_block/PCA.csv
```

### 3. Run statistical analysis

```bash
python -m src.analysis.stats_tests
```

This script reads `Scaling.csv` and `PCA.csv`, performs paired comparisons, and writes statistical reports to `results/per_block/`, including:

```text
scaling_stats_report.csv
pca_stats_report.csv
no_pca_vs_scaled_diff.csv
no_pca_vs_scaled_diff_summary.csv
```

### 4. Generate figures

```bash
python -m src.analysis.draw_figure
```

Generated figures are saved to:

```text
results/figures/
```

## Output Data

Each row in the result files corresponds to one combination of:

```text
(block_id, pipeline, model)
```

The main fields are:

```text
block_id
pipeline
model
n_features_before
n_features_after
best_params
accuracy
macro_f1
fit_runtime_sec
inference_runtime_sec
```

Results are stored at the individual block level to support aggregation, standard deviation analysis, paired plots, and paired statistical testing.

## Resources

- [Project presentation video](https://youtu.be/DGEcxcbAyvw)
- [Result figures](results/figures/)