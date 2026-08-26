# Predicting Maize Yields at Large Scale Using Remote Sensing

Code for *Predicting Maize Yields at Large Scale Using Remote Sensing*
(Ferguson, Sogomo, and Sayre). The pipeline downscales municipality-level
(admin-2) SIAP maize yields in Mexico to ~95,000 near-farm census units
(*áreas de control*, ADCs) using Google AlphaEarth Foundations (AEF)
embeddings, and validates against restricted 2022 agricultural census
microdata.

See `PROJECT.md` for a detailed narrative of the repository layout and
methodology; this file is the replication quick-start.

## Layout

This is a **code-only** repository. All data, intermediates, model
predictions, tables, and figures live in a separate (non-git) project
directory:

| Location | Contents |
|---|---|
| `~/Dropbox/Github/predicting_yields_rs/` | this repo (code + makefiles) |
| `~/Dropbox/Projects/Maize_prediction/` | `Data/`, `Intermediates/`, `Data/predictions/`, `plots/`, `tables/`, `output/` |
| `~/Dropbox/Overleaf/Predicting Yields at Scale using RS/` | the paper (`FSS_MaizeYields_WP.tex` → `new_body.tex`) |

The restricted INEGI census microdata cannot be redistributed; scripts that
evaluate against it require access to the corresponding files under
`Data/`.

## Environments

Two conda environments are used, auto-detected by `Code/pipeline.mk`:

| Machine | Geospatial/cleaning env | ML/Earth Engine env | Conda root |
|---|---|---|---|
| Server (Deloach) | `mpc_env` | `ML_env` | `/usr/local/anaconda3` |
| Laptop | `geo_env` | `ml_cuda` | `~/miniforge3` |

Override with `make ... GEO_ENV_NAME=<env> ML_ENV_NAME=<env>
CONDA_SH=<path/to/conda.sh>`.

Key packages: geopandas, rasterio, shapely, pandas, pyarrow, fuzzywuzzy,
openpyxl (cleaning env); scikit-learn, torch, earthengine-api
(ML env). `Code/requirements.txt` is a legacy pip freeze kept for
reference only.

## Running the pipeline

From the repo root:

```bash
make -f Code/pipeline.mk build      # stage 1–2: cleaning & boundary assembly
make -f Code/pipeline.mk train      # stage 3–4: model training & prediction
make -f Code/pipeline.mk analysis   # stage 5: evaluation, tables, figures, copy to Overleaf
make -f Code/pipeline.mk all        # train + analysis
```

Each numbered task folder has exactly one `.mk` file with its targets; the
master `Code/pipeline.mk` chains them.

### Manual (non-automatable) steps

Some stages require interactive authentication and are documented as
`MANUAL` targets that print instructions instead of running:

1. **SIAP scraping** (`build/01_scrape_siap/`) — Selenium + Chrome.
2. **AEF embedding extraction** (`train/01_extract_embeddings/`) — Google
   Earth Engine batch exports (EE auth required).
3. **Landsat NDVI feature extraction** (`train/02_extract_ndvi_histograms/`)
   — Earth Engine. The paper's NDVI (masked) baseline uses the
   cropland-masked `aefn2` features; the full extraction/download recipe is
   in `Code/train/02_extract_ndvi_histograms/AEFN2_PULL_HANDOFF.md`
   (`make extract_cropland_features` prints the summary).

Everything downstream of the extracted features is fully make-driven.

### Ordering note for the NDVI (masked) baseline

`analysis/02_accuracy_maize/masked_muni_cv.py` writes the muni-level
feature cache consumed by
`train/05_train_agg_nn/train_holdout_validation_models.py masked`; the
makefiles encode this dependency, but if running scripts by hand, run the
former first.

## Paper outputs

Every table `\input{}` and figure `\includegraphics{}` in `new_body.tex`
is produced by a make target and copied to Overleaf by
`make -f Code/pipeline.mk analysis` (final step:
`analysis/04_copy_to_overleaf/04_copy_to_overleaf.mk`). Main generators:

| Paper object | Generator |
|---|---|
| Main accuracy tables + common-sample tables + scatter figures | `analysis/02_accuracy_maize/accuracy_main_2022.py` |
| Municipality validation table | `train/05_train_agg_nn/validation_mun_level.py` |
| Census thought experiment | `analysis/02_accuracy_maize/census_thought_all_models.py` |
| Survey-improvement table | `analysis/02_accuracy_maize/mun_survey_improvement.py` |
| ADC-characteristics profile | `analysis/02_accuracy_maize/accuracy_profile_by_adc_chars.py` |
| CIMMYT external validation | `analysis/02_accuracy_maize/accuracy_cimmyt_profile.py` |
| Other-crops table | `train/03_train_rf_gb/gb_aef_hist_ensemble_other_crops.py` |
| Ex-ante targeting figures | `analysis/02_accuracy_maize/exante_trust_composite.py` + `fig_*.py` |
| Census yield maps (CA07, Fig. 1) | `analysis/01_corrections/1_plot_yields_ADC_mun.ipynb` |
| Prediction/error maps (CA22) + municipal SIAP/pred pair | `analysis/01_corrections/plot_yields_ca22_maps.py` |
| SIAP-vs-census hexbin (Fig. 1e) | `analysis/01_corrections/3_plot_siap_vs_adc_census_scatter.ipynb` |
| Monthly harvest figure | `build/02_clean_siap_monthly/plot_maize_monthly_harvesting.py` |
