# Project Instructions — Predicting Yields at Scale Using Remote Sensing

## Authors

| Name           | Role                                          |
|----------------|-----------------------------------------------|
| James Sayre    | Data assembly, accuracy evaluation, paper     |
| Joel Ferguson  | Satellite extraction, ML models               |
| Kangogo Sogomo | Coauthor                                      |

## Repository Layout

This is a **code-only** repo at `~/Dropbox/Github/predicting_yields_rs/`.
All data, outputs, plots, and intermediates live in a separate project directory
at `~/Dropbox/Projects/Maize_prediction/`.

### Code repo (`~/Dropbox/Github/predicting_yields_rs/`)

```
CLAUDE.md
PROJECT.md
improvement_plan.md
WITHIN_MUN_R2_PLAN.md
RF_validation_results.md
Code/
├── pipeline.mk              # Master orchestration (stages 3–5)
├── build/                    # Data cleaning & boundary assembly
│   ├── 01_scrape_siap/
│   ├── 02_clean_siap_monthly/
│   ├── 03_assemble_boundaries/
│   ├── 04_build_correspondence/
│   ├── 05_phenology/
│   └── 06_clean_validation/
├── train/                    # Model training & prediction
│   ├── 01_extract_embeddings/       # AEF embeddings (Earth Engine)
│   ├── 02_extract_ndvi_histograms/  # Landsat NDVI histograms (Earth Engine)
│   ├── 03_train_rf_gb/              # AEF → RF/GB models
│   ├── 04_train_cnn/                # NDVI histogram → CNN models
│   └── 05_train_agg_nn/             # AEF → aggregation-constrained NN
└── analysis/                 # Corrections, evaluation, paper outputs
    ├── 01_corrections/
    ├── 02_accuracy_maize/
    ├── 03_accuracy_other_crops/
    └── 04_copy_to_overleaf/
```

### Data / output directory (`~/Dropbox/Projects/Maize_prediction/`)

```
Data/                        # Cleaned data (was data/)
Data/alpha_earth/            # AEF embeddings (was Joel_AEF_Code/alpha_earth/)
Data/predictions/            # Model predictions (was Joel_AEF_Code/predictions/ + Joel_yield_prediction/)
plots/                       # Generated figures
output/                      # Analysis outputs
tables/                      # LaTeX tables
Intermediates/               # Intermediate data products
```

## Makefile Discipline

- **Two-level structure:** `Code/pipeline.mk` includes task-level `.mk` files from each numbered folder.
- Each task folder has exactly one `.mk` file named after the folder (e.g., `03_train_rf_gb/03_train_rf_gb.mk`).
- `pipeline.mk` lives in the code repo but all data paths reference `~/Dropbox/Projects/Maize_prediction/`.
- Notebooks run via `jupyter nbconvert --execute --inplace`.
- Python scripts run via `python3 <script.py>`.
- Always activate the correct conda environment before running.

## Conda Environments (Server: `/home/jsayre/`)

| Environment   | Use Case                        | Key Packages                                    |
|---------------|--------------------------------|------------------------------------------------|
| `mpc_env`     | Geospatial, data cleaning       | geopandas, rasterio, shapely, pandas, fuzzywuzzy |
| `ML_env`      | ML models, Earth Engine         | earthengine-api, torch, sklearn, tensorflow     |
| `linear_est`  | Econometric estimation          | pyfixest, statsmodels                           |

**Rules:**
- Joel's scripts (`train/` folder) use `ML_env`
- James's notebooks (`build/`, `analysis/`) use `mpc_env`
- Do NOT use `sudo` or `apt install`
- Do NOT use bare `python3` (no pandas/pyarrow available outside conda)

Detect environment by home directory: `/home/j/` → laptop, `/home/jsayre/` → server.

## Always Read Before Suggesting

Before making ANY suggestion or diagnosis about code errors:
1. Read the current state of the relevant file(s) using the Read tool
2. Do NOT assume code is the same as what you last saw
3. Do NOT suggest solutions that are already in the code

## Python Code Style

- Write Python code in Jupyter notebooks (`.ipynb`) for James's code
- Use `####` or smaller for all markdown headings (no `#`, `##`, or `###`)
- Align assignment operators with two spaces after `=`:
  ```python
  this_dir    =  os.path.join(top_dir, "This_directory")
  another_dir =  os.path.join(top_dir, "Another_directory")
  ```
- Add brief inline comments for data inputs
- Keep printed data-check lines to a minimum

### Notebook Header

Start each notebook with a markdown cell:

```markdown
#### [Filename]

**Author:** James Sayre
**Email:** jsayre@ucdavis.edu
**Date Modified:** [today's date, YYYY-MM-DD]

**Description:** [Brief description]

**Inputs:** [List of input files/data with paths]
**Outputs:** [List of output files/data with paths]
```

### First Code Cell

```python
import os
import pandas as pd

# ── Directories ──────────────────────────────────────────
home_dir    =  os.path.expanduser("~")
code_dir    =  os.path.join(home_dir, "Dropbox", "Github", "predicting_yields_rs")
proj_dir    =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
data_dir    =  os.path.join(proj_dir, "Data")
out_dir     =  os.path.join(proj_dir, "output")

# ── Inputs ───────────────────────────────────────────────
input_file  =  os.path.join(data_dir, "input.csv")     # description

# ── Outputs ──────────────────────────────────────────────
output_file =  os.path.join(out_dir, "output.csv")      # description
```

## R Code Style

- Use `data.table` syntax
- Use `fixest` for regressions
- Use `ggplot2` with minimal themes for plots

## Editing Existing Code

- Do NOT overwrite existing work unless explicitly asked
- Add new code as separate cells, preserving original code
- When asked to add a regression, keep existing regressions intact

## Git

- Keep commit messages brief; exclude authorship information
- If there are conflicts, check with the user first
