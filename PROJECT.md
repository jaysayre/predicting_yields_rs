# Predicting Maize Yields at Large Scale Using Remote Sensing

**Paper:** `~/Dropbox/Overleaf/Predicting Yields at Scale using RS/new_body.tex` (+ `appendix.tex`)
**Authors:** Joel Ferguson, Kangogo Sogomo, James Sayre

This file is the narrative companion to `README.md` (the replication
quick-start). It describes what each stage of the pipeline does, which script
produces which intermediate file, and how the paper objects are assembled.
Section and table numbers are deliberately not used here; paper objects are
named by their file or LaTeX label.

## Overview

We downscale annual municipality-level maize yields published by Mexico's
DGSIAP (Dirección General del Servicio de Información Agroalimentaria y
Pesquera) to ~95,000 agricultural census units (*áreas de control*, ADCs)
using Google AlphaEarth Foundations (AEF) satellite embeddings. Models are
trained only on public municipality-year data (2017-2024) and evaluated
against restricted INEGI 2022 Agricultural Census microdata at the ADC level
and against CIMMYT farmer-plot yields.

Models compared in the paper (row labels of the accuracy tables):

| Row label | Features | Learner | Producer |
|---|---|---|---|
| AEF Hist | 64 dims × 8 fixed-width bin shares on [-0.8, 0.8], trained on K = 5 subsampled draws of N = 2 pixels per dimension (512 features) | HistGradientBoosting | `train/03_train_rf_gb/2_gb_aef_hist_ensemble.py` |
| AEF Hist Ens. | 0.5 × AEF Hist + 0.5 × percentile model (5 percentiles + SD per dim + 64 means, 448 features) | same run | same |
| AEF mean | 64 per-dimension means | Random forest | `train/03_train_rf_gb/1_rf_yield_prediction.py` |
| Agg-NN | 64 means, aggregation-constrained neural net | PyTorch MLP | `train/04_train_agg_nn/2_agg_nn_sweep.py` |
| NDVI (masked) | the AEF featurization applied to harmonically smoothed Landsat NDVI on cropland pixels | HistGradientBoosting | `analysis/02_accuracy_maize/2_masked_adc_eval.py` |
| Oracle | ADC-trained ceiling (not deployable) | HistGradientBoosting | `analysis/02_accuracy_maize/3_oracle_adc_ceiling.py` |

Post-prediction corrections evaluated for every model: an ex-post
aggregate-consistency correction (ADC predictions rescaled so their
ex-ante agland-weighted mean matches the DGSIAP municipal yield) and
within-municipality shrinkage `z_shrink = mean + lambda (z - mean)` with
lambda = 0.72. Both lambda and the ensemble weight w = 0.5 are chosen from
public data alone (steps 17-18 of `analysis/02_accuracy_maize/`).

## Repository layout

The code lives in this git repository; all data and outputs live in a
separate, non-versioned project directory.

```
~/Dropbox/Github/predicting_yields_rs/           this repo
├── README.md  PROJECT.md
└── Code/
    ├── pipeline.mk            master makefile (build / train / analysis / all)
    ├── envs/                  conda environment specs (geo_env, ml_env; *_pinned = exact versions)
    ├── siap_yields.py         shared DGSIAP yield-panel loader
    ├── build/01 .. 06         stage 1
    ├── train/01 .. 04         stage 2
    └── analysis/01 .. 04      stage 3

~/Dropbox/Projects/Maize_prediction/             data project (not in git)
├── Data/
│   ├── SIAP/Cleaned/          siap_ag_prod_estimation_by_season.dta (municipality × crop × cycle × year yields)
│   ├── SIAP_monthly/          scraped monthly production + cleaned panel
│   ├── SIAP_agland/Output/    2007_adcs_agland_area.csv (ADC agland + irrigated share, the ex-ante weights)
│   ├── INEGI/                 census boundaries; CA22 microdata-lab outputs (restricted)
│   ├── Shapefiles/            adc_shapefile.shp (canonical ADC polygons)
│   ├── muncodes/shp/          MUNICIPIOS.shp (municipality polygons)
│   ├── alpha_earth/           AEF parquets + the per-state GEE CSV exports (*_csvs/)
│   ├── cropland_features/     Landsat cropland-masked NDVI features (csvs_crop_aefn2/, muni cache)
│   ├── CIMMYT/                farmer-plot logbooks; EE geometries
│   ├── corr_tables/           2007 <-> 2016 ADC correspondence
│   └── predictions/           every model's ADC / municipality predictions
├── Intermediates/CIMMYT/      cleaned CIMMYT plots + yields
├── plots/                     generated figures and the main accuracy .tex tables
├── tables/                    generated .tex tables
└── output/                    prose statistics (json)
```

Conventions:

- Each task folder has exactly one `<folder>.mk`; its header lists the
  folder's scripts in run order and every script is numbered `1_`, `2_`, ...
  in that order. Unnumbered scripts are robustness variants or prose-only
  utilities named in the same header.
- Makefile targets are the output files, so `make` only reruns a step whose
  script or inputs are newer than its outputs. `make -n` shows what would run.
- `dep/` subfolders (gitignored) hold superseded code that no paper object
  depends on: the NDVI-histogram CNN and its extraction chain, the
  448-feature percentile-only "AEF Hist" of earlier drafts, the East Africa
  extension, and one-off diagnostics.
- Nothing writes into the Overleaf directory except
  `analysis/04_copy_to_overleaf/04_copy_to_overleaf.mk`.
- The INEGI microdata-lab Stata requests (`Code/MD_lab_code/`, gitignored)
  produced the census ground truth (`adc_land_use_ca22_adc07.dta`,
  `adc_land_szn_ca22_adc07.dta`, `rendimiento_agr_adc.dta`); they cannot be
  run outside the lab.

## Stage 1: build (`make -f Code/pipeline.mk build`)

| Folder | Steps | Key outputs |
|---|---|---|
| `build/01_scrape_siap` | `1_ScrapeMonthlyProductionSIAP.ipynb` (manual; Selenium) | `Data/SIAP_monthly/Input/Y*S*M*.csv` |
| `build/02_clean_siap_monthly` | `1_CleanSIAPMonthlydata.ipynb`; `2_plot_maize_monthly_harvesting.py` | `Data/SIAP_monthly/Output/mnthly_siap.dta`; `plots/maize_monthly_harvesting.png` (`fig:monthly_maize`) |
| `build/03_assemble_boundaries` | `1_assemble_2016_AMCA_shapefile`, `2_assemble_marco_geo_2022`, `3_assemble_SIAP_agland`, `4_segment_inegi_agland` (notebooks) | `INEGI/Areas_Censal_Agropecuario_2016/census_areas.shp`, `Marco_Geo_2022/Outputs/ageb_22.shp`, `Mexico_agland/Outputs/inegi_agland_16_adc.shp`, `SIAP_agland/Output/2007_adcs_agland_area.csv` |
| `build/04_build_correspondence` | `1_build_correspondence_ADCs`, `2_associate_boxes_ADCs` (notebooks) | `Data/corr_tables/ca07_ca16_corr.dta` |
| `build/05_phenology` | `1_Determine_Monthly_peaks.ipynb` | `Data/planting_months_harmonic_regression.csv`, `muni_ndvi_{peak,min}_dates.csv`, `SIAP_monthly/Output/comp_sat_plant.csv` |
| `build/06_clean_validation` | `1_clean_CIMMYT_farmer_plots.ipynb`; `2_prepare_cimmyt_geometries.ipynb` | `Intermediates/CIMMYT/cimmyt_maize_yields.csv`, `cimmyt_plot_locations.shp`; `Data/CIMMYT/cimmyt_plot_geometries_for_ee.csv` |

The municipality yield panel `Data/SIAP/Cleaned/siap_ag_prod_estimation_by_season.dta`
(DGSIAP *Anuario Estadístico de la Producción Agrícola*, 2003-2025, by crop
cycle) and the INEGI locality catalogue are vendored into the data project.

## Stage 2: train (`make -f Code/pipeline.mk train`)

### `train/01_extract_embeddings` — AEF features (Earth Engine, manual; consolidation make-driven)

| Step | Script | Output |
|---|---|---|
| 1 | `1_ee_alpha_earth_download.py` (MODE = `mexico_mun`, `mexico_adc`) | Drive: per-dimension means on ESA WorldCover cropland pixels |
| 2-5 | `2_`..`5_ee_alpha_earth_histogram*.py` | Drive: percentiles p10/25/50/75/90 + SD per dimension, municipality and ADC, 2017-2024 |
| 6 | `6_ee_alpha_earth_binned_hist.py` | Drive: 64 × 8 fixed-width bin shares, municipality and ADC |
| 7 | `7_consolidate_mean_parquet.py` | `alpha_earth_mex_muns.parquet`, `alpha_earth_mex_adcs.parquet` |
| 8-9 | `8_consolidate_hist_parquet.py`, `9_consolidate_hist_parquet_mun.py` | `alpha_earth_mex_adcs_hist.parquet`, `alpha_earth_mex_mun_hist.parquet` |
| 10 | `10_consolidate_binned_parquet.py` | `alpha_earth_mex_mun_binned_hist.parquet`, `alpha_earth_mex_adcs_binned_hist.parquet` |
| 11-13 | `11_`..`13_ee_alpha_earth_cimmyt*.py` | Drive: the same three feature sets for buffered CIMMYT plots |
| 14 | `14_consolidate_aef_cimmyt_all.py` | `alpha_earth_cimmyt_plot{,_hist,_binned_hist}.parquet` |
| robustness | `ee_aef_quantile_edges.py`, `ee_alpha_earth_quantile_hist.py`, `consolidate_qbin_parquet.py` | equal-mass quantile-bin variant (`*_qbin_hist.parquet`) |

### `train/02_extract_ndvi_histograms` — NDVI (masked) baseline features

| Step | Script | Output |
|---|---|---|
| 1 | `1_prepare_adc_geometries.py` | `Data/adc_geometries_for_ee.csv`, `muni_geometries_for_ee.csv` (+ planting month) |
| 2 | `2_ls_cropland_features.py --level adc --years 2017-2024` (manual, EE) | Drive: 150 marginal features per ADC-year from a per-pixel 3-harmonic NDVI fit, cropland-masked |
| 3 | `3_pull_cropland_csvs.sh` | `Data/cropland_features/csvs_crop_aefn2/` |

`AEFN2_PULL_HANDOFF.md` is the extraction log and resubmission recipe.

### `train/03_train_rf_gb` — AEF models

| Step | Script | Output |
|---|---|---|
| 1 | `1_rf_yield_prediction.py` | `predictions/adc_alpha_earth_preds_maize.parquet` (AEF mean) |
| 2 | `2_gb_aef_hist_ensemble.py` | `adc_aef_hist_bins_gb_preds.parquet` (AEF Hist), `adc_aef_hist_ens_preds.parquet`, `adc_aef_hist_ens_eval.parquet` (ensemble + census yields + season-matched corrections; backbone of the analysis stage), `adc_aef_hist_ens_components.parquet` |
| 3 | `3_mun_cv_aef_hist_bins.py` | `mun_aef_hist_bins_gb_{loyo,kfold}_preds.parquet` (municipality-level CV) |
| 4 | `4_gb_aef_hist_ensemble_other_crops.py` | `adc_aef_hist_ens_preds_{sorghum,sugar,wheat,avocados}.parquet` |
| robustness | `gb_aef_hist_ensemble_qbin.py`, `_dm.py`, `_oi_trained.py` | quantile-bin, demeaned, fall-winter-trained variants |

### `train/04_train_agg_nn` — Agg-NN and holdout validation

| Step | Script | Output |
|---|---|---|
| 1 | `1_agg_constrained_nn.py` | `adc_agg_nn_preds_maize_phase2_{final,inner_es}.parquet` |
| 2 | `2_agg_nn_sweep.py` | `adc_mlp_yield_preds.csv` (the Agg-NN rows of every table) |
| 3 | `3_train_holdout_validation_models.py [masked]` | `*_holdout_preds.parquet`: every model retrained on 80% of municipalities |
| 4 | `4_validation_mun_level.py` | `tables/validation_mun_level_2022.tex` (held-out municipality-years) |
| 5 | `5_census_thought_experiment.py` | `adc_aggnn_census_trained_preds.parquet` (census-trained Agg-NN) |

## Stage 3: analysis (`make -f Code/pipeline.mk analysis`)

### `analysis/01_corrections` — maps

| Step | Script | Paper object |
|---|---|---|
| 1 | `1_plot_yields_ADC_mun.ipynb` | `fig:adc_mun_yield_comp` panels a-d (CA2007 census maps) |
| 2 | `2_plot_siap_vs_adc_census_scatter.ipynb` | panel e (`siap_mun_vs_adc_census_yield.pdf`) |
| 3 | `3_plot_yields_ca22_maps.py` | `fig:yield_preds_mun`, `fig:adc_yield_pred_comp`, `fig:prediction_error_ADC` |
| 4 | `4_fig1_oaxaca_diff_map.py` | panel f (`maizeyield_adc_mun_diff.png`) |

### `analysis/02_accuracy_maize` — tables, figures, prose statistics

| Step | Script | Paper object |
|---|---|---|
| 1 | `1_masked_muni_cv.py` | NDVI (masked) municipality CV + feature cache |
| 2 | `2_masked_adc_eval.py` | `adc_aefn2_masked_preds.parquet` (NDVI rows) |
| 3 | `3_oracle_adc_ceiling.py` | `oracle_ceiling_2022.csv` (oracle rows) |
| 4 | `4_accuracy_main_2022.py` | `accuracy_{spring_summer,combined,fall_winter}_2022.tex`, `common_sample_*.tex`, `accuracy_scatter_{combined,seasonal}_2022.pdf` |
| 5 | `5_mun_survey_improvement.py` | `accuracy_mun_level_2022.tex` |
| 6 | `6_census_thought_all_models.py` | `census_thought_experiment_2022.tex` |
| 7 | `7_accuracy_profile_by_adc_chars.py` | `accuracy_profile_by_adc_chars.tex` |
| 8 | `8_accuracy_cimmyt_profile.py` | `accuracy_cimmyt_profile.tex` |
| 9-11 | `9_exante_trust_features.py`, `10_exante_trust_composite.py`, `11_exante_trust_across_models.py` | ex-ante downscaling skill index (`exante_trust_index.csv`, `exante_trust_across_models.csv`) |
| 12-13 | `12_fig_representativeness_targeting.py`, `13_fig_ranking_inversion.py` | `fig:exante_targeting`, `fig:inversion` |
| 14-16 | `14_fetch_mun_chips_fig.py` (manual GEE), `15_fig_methodology_diagram.py`, `16_fig_pipeline_diagram.py` | `fig:methodology`, `fig:pipeline` |
| 17-20 | `17_rho_irrigation_bound.py`, `18_w_public_selection.py`, `19_robustness_lambda_ci_qbin.py`, `20_sample_accounting_and_lambda.py` | lambda, w, bootstrap CIs, sample accounting (prose only; `make prose_stats`) |

### `analysis/03_accuracy_other_crops`

`1_other_crops_adc_table.py` → `accuracy_other_crops_adc_2022.tex`;
`2_other_crops_mun_agg.py` → `accuracy_other_crops_mun_2022.tex`.

### `analysis/04_copy_to_overleaf`

Copies every generated table and figure into the Overleaf directory. The
only hand-maintained paper table is `datasets_summary.tex`.

## Evaluation conventions

- **Sample:** ADC-years with 2022 census maize yields matched to 2007 ADC
  polygons; the ensemble-eligible sample is the evaluation sample of the
  main tables. Municipality holdout: 20% of municipalities (all of their
  2017-2024 municipality-years) held out with `GroupShuffleSplit`.
- **Metrics:** overall, between-municipality and within-municipality R²;
  the within-R² is the paper's downscaling criterion.
- **Weights:** ex-ante agland area (`2007_adcs_agland_area.csv`) for every
  municipality aggregation, never census planted area.

## Environments

See `README.md`; specs in `Code/envs/`. `train/` scripts use the ML
environment (torch, scikit-learn, earthengine-api); `build/` and most of
`analysis/` use the geospatial environment (geopandas, rasterio, pandas,
matplotlib with the pgf/Times backend for figures).
