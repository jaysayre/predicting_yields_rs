# Predicting Maize Yields at Large Scale Using Remote Sensing

Code for *Predicting Maize Yields at Large Scale Using Remote Sensing*
(Ferguson, Sogomo, and Sayre). The pipeline downscales municipality-level
(admin-2) DGSIAP maize yields in Mexico to ~95,000 near-farm census units
(*áreas de control*, ADCs) using Google AlphaEarth Foundations (AEF)
embeddings, and validates against restricted 2022 agricultural census
microdata and CIMMYT farmer-plot yields.

`PROJECT.md` is the narrative of the repository layout and methodology; this
file is the replication quick-start.

## Layout

This is a **code-only** repository. All data, intermediates, model
predictions, tables, and figures live in a separate (non-git) project
directory:

| Location | Contents |
|---|---|
| `~/Dropbox/Github/predicting_yields_rs/` | this repo (code + makefiles) |
| `~/Dropbox/Projects/Maize_prediction/` | `Data/`, `Intermediates/`, `Data/predictions/`, `plots/`, `tables/`, `output/` |
| `~/Dropbox/Overleaf/Predicting Yields at Scale using RS/` | the paper (`FSS_MaizeYields_WP.tex` → `new_body.tex`, `appendix.tex`) |

```
Code/
├── pipeline.mk                  master makefile: build / train / analysis / all
├── envs/                        conda environment specs
├── siap_yields.py               shared DGSIAP yield-panel loader
├── build/                       stage 1: data acquisition, cleaning, boundaries
│   ├── 01_scrape_siap/            monthly production scrape (manual, Selenium)
│   ├── 02_clean_siap_monthly/     clean the monthly panel; harvest-calendar figure
│   ├── 03_assemble_boundaries/    ADC shapefiles, 2022 frame, agland masks
│   ├── 04_build_correspondence/   2007 <-> 2016 ADC correspondence
│   ├── 05_phenology/              NDVI phenology vs DGSIAP planting months
│   └── 06_clean_validation/       CIMMYT farmer plots -> yields + EE geometries
├── train/                       stage 2: feature extraction and models
│   ├── 01_extract_embeddings/     AEF means / percentiles / bin shares (Earth Engine) + consolidation
│   ├── 02_extract_ndvi_histograms/ cropland-masked harmonic-NDVI features (Earth Engine)
│   ├── 03_train_rf_gb/            AEF mean RF; AEF Hist + AEF Hist Ensemble GB; muni CV; other crops
│   └── 04_train_agg_nn/           Agg-NN; holdout retrains; municipality validation table
└── analysis/                    stage 3: evaluation and paper outputs
    ├── 01_corrections/            Figure 1 maps + hexbin; CA22 prediction maps
    ├── 02_accuracy_maize/         every maize table and figure; prose statistics
    ├── 03_accuracy_other_crops/   other-crop tables
    └── 04_copy_to_overleaf/       the single road into the paper
```

Every task folder has exactly one `<folder>.mk` whose header lists its
scripts in run order; scripts are numbered `1_`, `2_`, ... in that order.
Unnumbered scripts are robustness variants or prose-only utilities named in
the same header. `dep/` subfolders (gitignored) hold superseded code that no
paper object depends on.

The restricted INEGI census microdata cannot be redistributed; scripts that
evaluate against it require access to the corresponding files under `Data/`
(the Stata extraction requests run inside the INEGI microdata lab are kept
locally in `Code/MD_lab_code/`, also outside git).

## Environments

Two conda environments, auto-detected by `Code/pipeline.mk`:

| Machine | Geospatial/cleaning env | ML/Earth Engine env | Conda root |
|---|---|---|---|
| Server (Deloach) | `mpc_env` | `ML_env` | `/usr/local/anaconda3` |
| Laptop | `geo_env` | `ml_cuda` | `~/miniforge3` |

Override with `make ... GEO_ENV_NAME=<env> ML_ENV_NAME=<env>
CONDA_SH=<path/to/conda.sh>`. Specs are in `Code/envs/`: `geo_env.yml` and
`ml_env.yml` are the top-level package lists for a fresh solve;
`*_pinned.yml` are the exact versions the paper's outputs were produced with
(`conda env create -f Code/envs/geo_env_pinned.yml`).

## Running the pipeline

From the repo root:

```bash
make -f Code/pipeline.mk build      # stage 1: cleaning & boundary assembly
make -f Code/pipeline.mk train      # stage 2: model training & prediction
make -f Code/pipeline.mk analysis   # stage 3: evaluation, tables, figures, copy to Overleaf
make -f Code/pipeline.mk all        # train + analysis
make -f Code/pipeline.mk -n all     # dry run: list what would be rebuilt
```

Individual stages are `build_01` ... `build_06`, `train_01` ... `train_04`,
`analysis_01` ... `analysis_04`; each task makefile can also be run on its own
(`make -f Code/train/03_train_rf_gb/03_train_rf_gb.mk <target> PROJ_DIR=... DATA_DIR=...`).
Targets are real files, so make only reruns steps whose script or inputs are
newer than their outputs. Notebooks run via `jupyter nbconvert --execute --inplace`.

### Manual (credentialed) steps

Three steps need interactive authentication and are `MANUAL` targets that
print their instructions:

1. **DGSIAP scraping** (`build/01_scrape_siap/`): Selenium + Chrome.
2. **AEF feature extraction** (`train/01_extract_embeddings/`, steps 1-6 and
   11-13): Google Earth Engine batch exports to Drive. Pull each Drive folder
   into the matching `Data/alpha_earth/*_csvs/` directory; `make train_01`
   then consolidates the CSVs into the parquet files the models read.
3. **Landsat NDVI cropland features** (`train/02_extract_ndvi_histograms/`,
   step 2): Earth Engine; `3_pull_cropland_csvs.sh` pulls the exports. The
   full extraction log is in `AEFN2_PULL_HANDOFF.md`.

Everything downstream of the pulled exports is make-driven.

## Paper outputs

Every table `\input{}` and figure `\includegraphics{}` in the paper is
produced by a make target and copied to Overleaf by
`make -f Code/pipeline.mk analysis` (final step:
`analysis/04_copy_to_overleaf/04_copy_to_overleaf.mk`). The one exception is
`datasets_summary.tex`, the hand-written data-summary table.

| Paper object | Generator |
|---|---|
| Main accuracy tables (combined, by season) + common-sample tables + scatter figures | `analysis/02_accuracy_maize/4_accuracy_main_2022.py` |
| Municipality held-out validation table | `train/04_train_agg_nn/4_validation_mun_level.py` |
| Census thought-experiment table | `analysis/02_accuracy_maize/6_census_thought_all_models.py` (+ `train/04_train_agg_nn/5_census_thought_experiment.py`) |
| Municipality-aggregation (survey improvement) table | `analysis/02_accuracy_maize/5_mun_survey_improvement.py` |
| Accuracy by ADC characteristics | `analysis/02_accuracy_maize/7_accuracy_profile_by_adc_chars.py` |
| CIMMYT external validation | `analysis/02_accuracy_maize/8_accuracy_cimmyt_profile.py` |
| Other-crop tables (ADC; municipal) | `analysis/03_accuracy_other_crops/1_other_crops_adc_table.py`, `2_other_crops_mun_agg.py` |
| Featurization diagram (`fig:methodology`) | `analysis/02_accuracy_maize/15_fig_methodology_diagram.py` |
| Pipeline overview (`fig:pipeline`) | `analysis/02_accuracy_maize/16_fig_pipeline_diagram.py` |
| Ex-ante targeting figures | `analysis/02_accuracy_maize/9_` ... `13_` (feature csv → skill index → figures) |
| Census yield maps (`fig:adc_mun_yield_comp`, panels a-d) | `analysis/01_corrections/1_plot_yields_ADC_mun.ipynb` |
| DGSIAP-vs-census hexbin (panel e) | `analysis/01_corrections/2_plot_siap_vs_adc_census_scatter.ipynb` |
| ADC-minus-municipality map (panel f) | `analysis/01_corrections/4_fig1_oaxaca_diff_map.py` |
| Prediction / error maps (CA22) + municipal DGSIAP/prediction pair | `analysis/01_corrections/3_plot_yields_ca22_maps.py` |
| Monthly harvest figure | `build/02_clean_siap_monthly/2_plot_maize_monthly_harvesting.py` |
| Numbers cited only in prose (lambda, w, CIs, sample accounting) | `analysis/02_accuracy_maize/17_` ... `20_` (`make prose_stats`) |

Model prediction files behind the table rows (all produced in-repo):

| Row label | File | Producer |
|---|---|---|
| AEF Hist | `adc_aef_hist_bins_gb_preds.parquet` | `train/03_train_rf_gb/2_gb_aef_hist_ensemble.py` |
| AEF Hist Ens. | `adc_aef_hist_ens_eval.parquet` | same run |
| AEF mean | `adc_alpha_earth_preds_maize.parquet` | `train/03_train_rf_gb/1_rf_yield_prediction.py` |
| Agg-NN | `adc_mlp_yield_preds.csv` | `train/04_train_agg_nn/2_agg_nn_sweep.py` |
| NDVI (masked) | `adc_aefn2_masked_preds.parquet` | `analysis/02_accuracy_maize/2_masked_adc_eval.py` |
| Oracle (ADC-trained) | `oracle_ceiling_2022.csv` | `analysis/02_accuracy_maize/3_oracle_adc_ceiling.py` |

The shrinkage factor lambda = 0.72 and ensemble weight w = 0.5 are selected
from public data by steps 17 and 18 of `analysis/02_accuracy_maize/` and
entered as constants in the generators.
