# Predicting Yields at Scale Using Remote Sensing

## Project Overview

This project develops methods to **downscale crop yields** from aggregated administrative data to close-to-farm spatial units in Mexico, using satellite-derived features from Google's AlphaEarth Foundation (AEF) model. The primary application is **maize**, with extensions to sorghum, sugarcane, wheat, and avocados.

**Paper:** `~/Dropbox/Overleaf/Predicting Yields at Scale using RS/new_body.tex`
**Authors:** James Sayre, Joel Ferguson, & Kangogo Sogomo

### Key Idea

Train machine learning models on municipality-level yields (publicly available from SIAP) paired with 64-dimensional AEF satellite embeddings, then predict at the finer ADC level (~95,000 agricultural census areas). Validate against restricted INEGI census microdata.

---

## Repository Layout

The project is split across two locations: a **code repository** (version-controlled) and a **data project** (large files, not in git).

### Code Repository: `~/Dropbox/Github/predicting_yields_rs/`

```
predicting_yields_rs/
├── Code/                          # All project code (James + Joel)
│   ├── build/                     # Data acquisition, cleaning, boundary assembly
│   │   ├── 01_scrape_siap/        # SIAP monthly production web scrapers
│   │   ├── 02_clean_siap_monthly/ # Clean scraped SIAP data
│   │   ├── 03_assemble_boundaries/# ADC shapefiles, agland masks
│   │   ├── 04_build_correspondence/# 2007↔2016 ADC correspondence
│   │   ├── 05_phenology/          # Crop seasonality / NDVI peaks
│   │   └── 06_clean_validation/   # CIMMYT farmer trial data
│   ├── train/                     # Satellite extraction & ML model training
│   │   ├── 01_extract_embeddings/ # AEF embedding extraction from Earth Engine
│   │   ├── 02_extract_ndvi_histograms/ # Landsat NDVI histogram extraction
│   │   ├── 03_train_rf_gb/        # Random Forest / Gradient Boosting models
│   │   ├── 04_train_cnn/          # CNN on NDVI histograms (baseline)
│   │   └── 05_train_agg_nn/       # Aggregation-constrained NN
│   ├── analysis/                  # Post-prediction corrections & accuracy evaluation
│   │   ├── 01_corrections/        # Additive, GP, and irrigation corrections
│   │   ├── 02_accuracy_maize/     # Maize accuracy tables (Tables 1–3)
│   │   └── 03_accuracy_other_crops/ # Other crops accuracy (Tables 4–5)
│   ├── archive/                   # Legacy/exploratory notebooks
│   └── pipeline.mk                # Makefile orchestrating the pipeline
├── PROJECT.md                     # This file
└── README.md                      # Project overview
```

### Data Project: `~/Dropbox/Projects/Maize_prediction/`

```
Maize_prediction/
├── Data/                          # Raw and processed data
│   ├── INEGI/                     # Census boundaries, microdata lab outputs
│   ├── SIAP_monthly/              # Monthly production data (scraped)
│   ├── Mexico_agland/             # Agricultural land masks
│   ├── corr_tables/               # ADC correspondence tables (2007↔2016)
│   ├── CIMMYT/                    # Farmer trial validation data
│   ├── muncodes/                  # Municipality code reference
│   ├── alpha_earth/               # AEF embedding parquet files (~1.4 GB)
│   ├── predictions/               # Model output parquet/CSV files
│   ├── muni_ndvi_*.csv            # Municipality NDVI peak/min dates
│   └── planting_months_harmonic_regression.csv
├── Intermediates/                 # Processed intermediate files (e.g., CIMMYT shapefiles)
├── plots/                         # Generated figures and tables
├── output/                        # Miscellaneous outputs
└── tables/                        # Generated tables
```

**Paper directory:**
```
~/Dropbox/Overleaf/Predicting Yields at Scale using RS/
├── new_body.tex                   # Main paper body
├── FSS_MaizeYields_WP.tex         # Document wrapper
├── figures/                       # All paper figures (PNG/PDF, copied from Maize_prediction/plots/)
├── accuracy_*.tex                 # LaTeX table files (copied from Maize_prediction/plots/)
└── sample-base.bib                # Bibliography
```

---

## Pipeline: Code → Paper

The pipeline has five stages. Each stage lists the code that runs and the outputs it produces.

### Stage 1: Data Acquisition

#### 1a. Scrape SIAP Monthly Production Data

| Script | Location |
|--------|----------|
| `Code/build/01_scrape_siap/1_ScrapeMonthlyProductionSIAP.ipynb` | Latest version (Chrome/Selenium) |
| `Code/archive/2024-10-10_ScrapeMonthlyProductionSIAPvserver.ipynb` | Server version (archived) |

- **Source:** SIAP website (https://nube.siap.gob.mx/avance_agricola/)
- **Output:** `Data/SIAP_monthly/Input/Y[YEAR]S[STATE]M[MONTH].csv` — one CSV per year×state×month
- **Method:** Selenium browser automation iterating over years, states (32), months (12), irrigation types, and crop cycles

#### 1b. Extract AlphaEarth Embeddings from Google Earth Engine

| Script | Location |
|--------|----------|
| `Code/train/01_extract_embeddings/ee_alpha_earth_download.py` | Main extraction script (1,701 lines) |
| `Code/train/01_extract_embeddings/ee_alpha_earth.py` | Simplified municipality-only version |

- **Source:** Google Earth Engine — `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (2017–2024)
- **Method:** Mean-pools 64-dim AEF embeddings over agricultural pixels (ESA WorldCover mask) within each geographic unit
- **Modes:** `mexico_mun` (municipalities), `mexico_adc` (ADCs), `california` (counties/fields), `mexico_huerto` (avocado orchards)
- **Outputs:**
  - `Data/alpha_earth/alpha_earth_mex_muns.parquet` — municipality-level embeddings
  - `Data/alpha_earth/alpha_earth_mex_adcs.parquet` — ADC-level embeddings (514 MB)
  - `Data/alpha_earth/alpha_earth_ca_county.parquet` — California county embeddings

#### 1c. Extract Landsat NDVI Histograms (Baseline RS Method)

| Script | Location |
|--------|----------|
| `Code/train/02_extract_ndvi_histograms/ls_ndvi_hists.py` | Landsat histogram extraction |

- **Source:** Landsat 5/7/8 via Earth Engine
- **Method:** Computes NDVI/GCVI/NDTI histograms (32×32 bins) per municipality per growing season
- **Output:** Municipality histogram CSVs exported to Google Drive, then stored in `Data/muni_hists_state_*.csv`

---

### Stage 2: Data Cleaning & Assembly

#### 2a. Clean SIAP Monthly Data

| Notebook | Purpose |
|----------|---------|
| `Code/build/02_clean_siap_monthly/1_CleanSIAPMonthlydata.ipynb` | Master cleaning pipeline |

- **Inputs:** Raw SIAP CSVs from `Data/SIAP_monthly/Input/`, municipality code reference (`Data/muncodes/muncodes_clean.xlsx`)
- **Key steps:** Fuzzy-matches municipality names to INEGI codes, fills missing month×municipality combinations, computes monthly harvest increments and yields
- **Outputs:**
  - `Data/SIAP_monthly/Output/mnthly_siap.dta` — cleaned monthly data
  - `Data/SIAP_monthly/Output/max_harv_mnth.dta` — peak harvest month by crop/cycle/irrigation
  - `Data/SIAP_monthly/Output/max_harv_mnth_by_year.dta` — year-specific harvest peaks

#### 2b. Assemble Geographic Boundaries

| Notebook | Purpose | Key Output |
|----------|---------|------------|
| `Code/build/03_assemble_boundaries/1_assemble_2016_AMCA_shapefile.ipynb` | Combine 32 state-level AMCA files into national ADC shapefile | `Data/INEGI/Areas_Censal_Agropecuario_2016/census_areas.shp` (298,570 ADCs) |
| `Code/build/03_assemble_boundaries/2_assemble_marco_geo_2022.ipynb` | Prepare 2022 census AGEB boundaries | `Data/INEGI/Marco_Geo_2022/Outputs/ageb_22.shp`, `muns_ca22.shp` |
| `Code/build/03_assemble_boundaries/3_assemble_SIAP_agland.ipynb` | Intersect ADCs with SIAP agricultural land mask | `Data/Mexico_agland/Outputs/inegi_agland_{07,16}_adc.shp` |
| `Code/build/03_assemble_boundaries/4_segment_inegi_agland.ipynb` | Intersect ADCs with INEGI agricultural land mask | Same outputs as above (alternative mask) |

#### 2c. Build ADC Correspondence Tables

| Notebook | Purpose |
|----------|---------|
| `Code/build/04_build_correspondence/1_build_correspondence_ADCs.ipynb` | Spatial join of 2007↔2016 ADCs; aggregate 2007 census data to 2016 boundaries |

- **Outputs:**
  - `Data/corr_tables/ca07_ca16_corr.dta` — 2007→2016 correspondence with area shares
  - `Data/INEGI/MD_lab_outputs/ca2007_maize_amca_adcs.dta` — 2007 census maize aggregated to 2016 ADC level

#### 2d. Clean Validation Data

| Notebook | Purpose | Output |
|----------|---------|--------|
| `Code/build/06_clean_validation/1_clean_CIMMYT_farmer_plots.ipynb` | Clean CIMMYT farmer trial data (2012–2022) | `Data/CIMMYT/cimmyt_plot_locations.shp`, `cimmyt_maize_yields.csv` |

#### 2e. Determine Crop Seasonality

| Notebook | Purpose |
|----------|---------|
| `Code/build/05_phenology/1_Determine_Monthly_peaks.ipynb` | Match satellite NDVI phenology to SIAP planting/harvest months |

- **Inputs:** `Data/SIAP_monthly/Output/mnthly_siap.dta`, `Data/muni_ndvi_peak_dates.csv`, `Data/muni_ndvi_min_dates.csv`
- **Output:** `Data/SIAP_monthly/Output/comp_sat_plant.csv`

---

### Stage 3: Model Training & Prediction

#### 3a. AEF Random Forest / Gradient Boosting (Primary Method)

| Script | Location |
|--------|----------|
| `Code/train/03_train_rf_gb/rf_yield_prediction.py` | RF/GB yield models (427 lines) |

- **Training data:** Municipality-level AEF embeddings (`alpha_earth_mex_muns.parquet`) merged with SIAP yields from `~/Dropbox/Projects/The Promise of Crop Substitution/data/SIAP/Cleaned/siap_ag_prod_estimation_ca2007.dta`
- **Method:** Grid search over max_depth (RF) or learning_rate/n_estimators/max_depth (GB); 80/20 train/val split; retrain best on full data
- **Modes:** `mexico_maize` (IMPROVED=True → GradientBoosting), `california`, `mexico_multi_crop`
- **Outputs (ADC-level predictions):**
  - `Data/predictions/adc_alpha_earth_preds_maize.parquet`
  - `Data/predictions/adc_alpha_earth_preds_{wheat,sorghum,sugar,avocados}.parquet`
  - CSV copies also exported to `Data/predictions/adc_alpha_earth_preds.csv`

#### 3b. NDVI Histogram CNN (Baseline RS Method)

| Script | Location |
|--------|----------|
| `Code/train/02_extract_ndvi_histograms/clean_histograms_redux.py` | Preprocess histograms → pickles |
| `Code/train/04_train_cnn/train_neural_net.py` | 6-layer CNN (Keras/TF) |
| `Code/train/04_train_cnn/ndvi_nn_redux.py` | Improved CNN (PyTorch) with ADC-level subsampling |

- **Input:** 32×32×8 histogram arrays (Landsat bands + LST)
- **Output:** `Data/predictions/adcs_yield_preds.csv`, `Data/predictions/adc_yield_preds_corrected_2022.csv`

#### 3c. MLP on AEF Embeddings

| Script | Location |
|--------|----------|
| `Code/train/03_train_rf_gb/rf_nn_huerto_yield.py` | ResidualMLP (PyTorch) for avocado huertos |
| `Code/archive/NN.ipynb` | NN exploration notebook |

- **Output:** `Data/predictions/adc_mlp_yield_preds.csv` (1M rows: adcid, year, pred_quantity, pred_area, pred_yield)

#### 3d. Aggregation-Constrained NN (In Progress)

| Script | Location |
|--------|----------|
| `Code/train/05_train_agg_nn/agg_constrained_nn.py` | Phase 2 & 3: ADC-level NN with municipality-level loss |

- **Status:** Phase 2 written, training interrupted before completion. Phase 3 not started.
- **Plan doc:** `Code/train/05_train_agg_nn/WITHIN_MUN_R2_PLAN.md`
- **Motivation:** Baseline RF/GB achieves strong between-municipality R² (0.641) but near-zero within-municipality R² (-0.041). This model processes individual ADCs through shared weights, then area-weighted aggregates to municipality for loss computation.
- **Architecture:** ResidualMLP (hidden_dim=256, 3 residual blocks), input = 67 features (64 AEF + irrig_share + log_adc_area + agland_share). Loss = MSE + variance encouragement penalty.
- **Phase 3 additions:** Embedding deviation (ADC − mun mean, 64 dims) + temporal delta (year-over-year change, 64 dims) → 195 total features.
- **To run:**
  ```bash
  cd Code/train/05_train_agg_nn
  source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate ML_env
  python3 agg_constrained_nn.py --epochs 200 --hidden_dim 256 --batch_size 64 --var_lambda 0.01
  # Phase 3: add --phase3 flag
  ```
- **Expected outputs:**
  - `Data/predictions/adc_agg_nn_preds_maize_phase2_final.parquet`
  - `Data/predictions/adc_agg_nn_preds_maize_phase3_final.parquet`

---

### Stage 4: Post-Prediction Corrections

#### 4a. Ex-Post Additive Correction

| Notebook | Location |
|----------|----------|
| `Code/analysis/01_corrections/1_plot_yields_ADC_mun.ipynb` | Applies additive correction and generates maps |

- **Method:** For each municipality, shift all ADC predictions so their area-weighted mean matches the SIAP municipal yield
- **Outputs:**
  - `Data/predictions/adc_yield_preds_corrected_2007.csv` — corrected predictions (2007 census comparison)
  - `Data/predictions/adc_yield_preds_corrected_2022.csv` — corrected predictions (2022 census comparison)

#### 4b. Gaussian Process Correction

| Notebook | Location |
|----------|----------|
| `Code/analysis/01_corrections/2_gp_correction_2022.ipynb` | Semi-parametric GP correction |

- **Method:** Fits a GP on municipality-level residuals (predicted − SIAP) using spatial kernel over municipality centroids. Leave-one-state-out CV for hyperparameter tuning (256 combinations). Predicts corrected yields at ADC level.
- **Inputs:** RS/AEF/MLP predictions, SIAP yields, municipality shapefile centroids
- **Output:** `Data/predictions/adc_gp_yield_preds_2022.csv` (columns: adc, yield_pred_rs_gp, yield_pred_aef_gp, yield_pred_mlp_gp)

#### 4c. Irrigation Adjustment (Phase 1 of Within-R² Improvement)

| Script | Location |
|--------|----------|
| `Code/analysis/01_corrections/irrigation_adjustment.py` | Within-municipality irrigation gradient correction |

- **Method:** Estimates irrigation-yield gradient (β̂ = 1.56 t/ha) from CA2007 census via within-municipality FE regression, applies to ADC predictions: `ADC_pred_adj = mun_mean_pred + β̂ × (irrig_share_adc − irrig_share_mun_mean)`
- **Result:** Within-R² improved from -0.041 → +0.110; overall R² from 0.484 → 0.518
- **Output:** `Data/predictions/adc_alpha_earth_preds_maize_irrig_adj.parquet`

---

### Stage 5: Accuracy Evaluation → Paper Tables & Figures

#### 5a. Maize Accuracy Tables (Paper Tables 1–3)

| Notebook | Location |
|----------|----------|
| `Code/analysis/02_accuracy_maize/1_accuracy_metrics_2022.ipynb` | Core accuracy evaluation |

- **Inputs:**
  - `Data/predictions/adc_yield_preds_corrected_2022.csv` (RS predictions)
  - `Data/predictions/adc_alpha_earth_preds.csv` (AEF predictions)
  - `Data/predictions/adc_mlp_yield_preds.csv` (MLP predictions)
  - `Data/predictions/adc_gp_yield_preds_2022.csv` (GP-corrected predictions)
  - `Data/INEGI/MD_lab_outputs/LM2304-CA22-2025-09-29-superficie_ENTREGA/adc_land_use_ca22_adc07.dta` (INEGI 2022 census, combined season)
  - `Data/INEGI/MD_lab_outputs/LM2304-CA22-2025-09-29-superficie_ENTREGA/adc_land_szn_ca22_adc07.dta` (INEGI 2022 census, by season)
  - `~/Dropbox/Projects/The Promise of Crop Substitution/data/SIAP/Cleaned/siap_ag_prod_estimation_ca2007.dta` (SIAP municipal yields)
- **Metrics:** R², between-R², within-R², RMSE at ADC level
- **Models evaluated:** RS (raw/corrected/GP), AEF (raw/corrected/GP), MLP (raw/corrected/GP), SIAP baseline
- **Outputs → Paper:**
  - `Maize_prediction/plots/accuracy_combined_2022.tex` → **Table 1** (combined-season ADC accuracy)
  - `Maize_prediction/plots/accuracy_fall_winter_2022.tex` → **Table 2** (fall-winter season)
  - `Maize_prediction/plots/accuracy_spring_summer_2022.tex` → **Table 3** (spring-summer season)
  - `Maize_prediction/plots/accuracy_scatter_combined_2022.pdf` → **Figure 4** (hexbin scatter plots)
  - `Maize_prediction/plots/accuracy_scatter_seasonal_2022.pdf` → **Figure 7** (seasonal scatter plots)

#### 5b. Other Crops Accuracy Tables (Paper Tables 4–5)

| Notebook | Location |
|----------|----------|
| `Code/analysis/03_accuracy_other_crops/1_accuracy_other_crops_2022.ipynb` | Non-maize crop evaluation |

- **Inputs:** `Data/predictions/adc_alpha_earth_preds_{sorghum,sugar,wheat,avocados}.csv`, INEGI 2022 census, SIAP yields
- **Outputs → Paper:**
  - `Maize_prediction/plots/accuracy_other_crops_adc_2022.tex` → **Table 4** (ADC-level, 4 crops)
  - `Maize_prediction/plots/accuracy_other_crops_mun_2022.tex` → **Table 5** (municipality-level, 2 panels)

#### 5c. Yield Maps (Paper Figures 1, 3, 5, 6)

| Notebook | Location |
|----------|----------|
| `Code/analysis/01_corrections/1_plot_yields_ADC_mun.ipynb` | Generates all map figures |

- **Inputs:** INEGI 2007/2022 census data, SIAP yields, MLP predictions, ADC/municipality/state shapefiles
- **Outputs → Paper:**
  - `Maize_prediction/plots/maizeyield_mun_allmx.png` → **Figure 1a** (municipality yields, all Mexico)
  - `Maize_prediction/plots/maizeyield_allmx_adc_with_legend.png` → **Figure 1b** (ADC yields, all Mexico)
  - `Maize_prediction/plots/maizeyield_mun.png` → **Figure 1c** (municipality yields, Oaxaca)
  - `Maize_prediction/plots/maizeyield_adc.png` → **Figure 1d** (ADC yields, Oaxaca)
  - `Maize_prediction/plots/maizeyield_mun_siap_allmx_nolegend_2018.png` → **Figure 3a** (SIAP municipality predictions)
  - `Maize_prediction/plots/maizeyield_mun_pred_allmx_nolegend_2018.png` → **Figure 3b** (model municipality predictions)
  - `Maize_prediction/plots/maizeyield_allmx_adc.png` → **Figure 5a** (ground truth ADC yields)
  - `Maize_prediction/plots/maizeyield_adc_preds.png` → **Figure 5b** (predicted ADC yields)
  - `Maize_prediction/plots/maizeyield_adc_pred_error_prederror_ls_noleg.png` → **Figure 6a** (uncorrected error map)
  - `Maize_prediction/plots/maizeyield_adc_pred_error_prederror_rcpred_noleg.png` → **Figure 6b** (corrected error map)
  - `Maize_prediction/plots/maizeyield_adc_pred_error_prederror_munyield.png` → **Figure 6c** (SIAP municipal yield)

#### 5d. Monthly Harvesting Figure (Paper Figure 2)

| Notebook | Location |
|----------|---------|
| `Code/build/02_clean_siap_monthly/1_CleanSIAPMonthlydata.ipynb` | Generates monthly bar plot |

- **Output → Paper:** `Maize_prediction/plots/maize_monthly_harvesting.png` → **Figure 2**

#### 5e. Multi-Year Validation (Supporting)

| Notebook | Location |
|----------|----------|
| `Code/analysis/02_accuracy_maize/accuracy_metrics.ipynb` | Multi-crop, multi-year (2017–2024) accuracy vs SIAP |

- **Outputs:** `Maize_prediction/plots/accuracy_aef_siap.tex`, scatter plots (supports paper's temporal stability claims)

---

## Copying Outputs to Paper Directory

Figures and tables must be **manually copied** from `~/Dropbox/Projects/Maize_prediction/plots/` to the Overleaf directory:

```bash
MP=~/Dropbox/Projects/Maize_prediction
OL="~/Dropbox/Overleaf/Predicting Yields at Scale using RS"

# Tables
cp $MP/plots/accuracy_combined_2022.tex       "$OL/"
cp $MP/plots/accuracy_fall_winter_2022.tex     "$OL/"
cp $MP/plots/accuracy_spring_summer_2022.tex   "$OL/"
cp $MP/plots/accuracy_other_crops_adc_2022.tex "$OL/"
cp $MP/plots/accuracy_other_crops_mun_2022.tex "$OL/"

# Figures
cp $MP/plots/accuracy_scatter_combined_2022.pdf           "$OL/figures/"
cp $MP/plots/accuracy_scatter_seasonal_2022.pdf            "$OL/figures/"
cp $MP/plots/maizeyield_mun_allmx.png                      "$OL/figures/"
cp $MP/plots/maizeyield_allmx_adc_with_legend.png          "$OL/figures/"
cp $MP/plots/maizeyield_mun.png                            "$OL/figures/"
cp $MP/plots/maizeyield_adc.png                            "$OL/figures/"
cp $MP/plots/maize_monthly_harvesting.png                  "$OL/figures/"
cp $MP/plots/maizeyield_mun_siap_allmx_nolegend_2018.png   "$OL/figures/"
cp $MP/plots/maizeyield_mun_pred_allmx_nolegend_2018.png   "$OL/figures/"
cp $MP/plots/maizeyield_allmx_adc.png                      "$OL/figures/"
cp $MP/plots/maizeyield_adc_preds.png                      "$OL/figures/"
cp $MP/plots/maizeyield_adc_pred_error_prederror_ls_noleg.png       "$OL/figures/"
cp $MP/plots/maizeyield_adc_pred_error_prederror_rcpred_noleg.png   "$OL/figures/"
cp $MP/plots/maizeyield_adc_pred_error_prederror_munyield.png       "$OL/figures/"
```

---

## External Data Dependencies

These files live outside this project but are consumed by the pipeline:

| File | Project | Used By |
|------|---------|---------|
| `~/Dropbox/Projects/The Promise of Crop Substitution/data/SIAP/Cleaned/siap_ag_prod_estimation_ca2007.dta` | Crop Substitution | RF training, accuracy evaluation, GP correction |
| `~/Dropbox/Projects/The Promise of Crop Substitution/data/SIAP/Cleaned/siap_ag_prod_estimation_by_season.dta` | Crop Substitution | Seasonal yield data, hi/lo municipality selection |
| `~/Dropbox/Projects/Avocado_Deforestation/Data/raw/spatial/Municipality_shp/MUNICIPIOS.shp` | Avocado Deforestation | Municipality centroids for GP |
| `~/Dropbox/Projects/Avocado_Deforestation/Data/raw/spatial/Municipality_shp/STATES.shp` | Avocado Deforestation | State boundaries for maps |
| `~/Dropbox/Projects/Crop_misallocation/data/SCIAGA/CA2007_adcloc_poly.shp` | Crop Misallocation | 2007 ADC polygons |
| `~/Dropbox/Projects/Crop_misallocation/data/SCIAGA/CA2007_ageb_poly.shp` | Crop Misallocation | 2007 AGEB polygons |

---

## Key Results

| Model | ADC R² (Combined 2022) | Notes |
|-------|------------------------|-------|
| AEF Corrected | 0.519 | Best overall |
| AEF GP | 0.495 | Best for spring-summer |
| AEF Raw | 0.484 | No correction |
| MLP Corrected | ~0.45 | Aggregation-constrained NN |
| RS GP | 0.333 | Traditional Landsat baseline |
| SIAP (baseline) | 0.426 | Municipal average assigned to all ADCs |

Fall-winter season achieves R² = 0.847 (AEF Corrected) due to geographic concentration in irrigated northern Mexico.

---

## Environment & Dependencies

### Conda Environments (Server: `/home/jsayre/`)

| Environment | Use Case | Key Packages |
|-------------|----------|--------------|
| `mpc_env` | Geospatial work, data cleaning | geopandas, rasterio, shapely, pandas, fuzzywuzzy |
| `ML_env` | ML models, Earth Engine | earthengine-api, torch, sklearn, tensorflow |
| `linear_est` | Econometric estimation | pyfixest, statsmodels |

### Joel's Scripts
- `Code/train/*.py` and `Code/analysis/01_corrections/irrigation_adjustment.py` scripts require `ML_env` (earthengine-api, sklearn, torch)
- Earth Engine authentication required for `ee_alpha_earth_download.py`

---

## Complete File Inventory

### Code Files (James)

| File | Stage | Purpose |
|------|-------|---------|
| `Code/build/01_scrape_siap/1_ScrapeMonthlyProductionSIAP.ipynb` | 1a | Scrape SIAP monthly data |
| `Code/build/02_clean_siap_monthly/1_CleanSIAPMonthlydata.ipynb` | 2a | Clean monthly SIAP data |
| `Code/build/03_assemble_boundaries/1_assemble_2016_AMCA_shapefile.ipynb` | 2b | Assemble national ADC shapefile |
| `Code/build/03_assemble_boundaries/2_assemble_marco_geo_2022.ipynb` | 2b | Prepare 2022 AGEB boundaries |
| `Code/build/03_assemble_boundaries/3_assemble_SIAP_agland.ipynb` | 2b | Intersect ADCs with agland mask |
| `Code/build/03_assemble_boundaries/4_segment_inegi_agland.ipynb` | 2b | Intersect ADCs with INEGI agland |
| `Code/build/04_build_correspondence/1_build_correspondence_ADCs.ipynb` | 2c | Build 2007↔2016 ADC correspondence |
| `Code/build/04_build_correspondence/2_associate_boxes_ADCs.ipynb` | 2c | Associate satellite boxes with ADCs |
| `Code/build/06_clean_validation/1_clean_CIMMYT_farmer_plots.ipynb` | 2d | Clean CIMMYT validation data |
| `Code/build/05_phenology/1_Determine_Monthly_peaks.ipynb` | 2e | Match NDVI phenology to SIAP months |
| `Code/analysis/01_corrections/1_plot_yields_ADC_mun.ipynb` | 4a, 5c | Additive correction + all map figures |
| `Code/analysis/01_corrections/2_gp_correction_2022.ipynb` | 4b | Gaussian Process correction |
| `Code/analysis/02_accuracy_maize/1_accuracy_metrics_2022.ipynb` | 5a | Maize accuracy tables (Tables 1–3) |
| `Code/analysis/03_accuracy_other_crops/1_accuracy_other_crops_2022.ipynb` | 5b | Other crops accuracy (Tables 4–5) |

| File | Stage | Purpose |
|------|-------|---------|
| `Code/archive/2022-10-01_ADCs_with_maize.ipynb` | 2b | Identify maize-producing ADCs |
| `Code/archive/2024-01-24_select_hi_lo_yield_muns.ipynb` | 2e | Select hi/lo yield municipalities |
| `Code/archive/2024-01-27_Figure_out_distribution_seasonality.ipynb` | 2e | Seasonality analysis (deprecated) |
| `Code/archive/2023-03-06_new_histogram_method.ipynb` | — | NDVI histogram method development |

### Code Files (Joel)

| File | Stage | Purpose |
|------|-------|---------|
| `Code/train/01_extract_embeddings/ee_alpha_earth_download.py` | 1b | Extract AEF embeddings from Earth Engine |
| `Code/train/03_train_rf_gb/rf_yield_prediction.py` | 3a | RF/GB yield models (primary method) |
| `Code/analysis/01_corrections/irrigation_adjustment.py` | 4c | Irrigation-based correction (Phase 1) |
| `Code/train/05_train_agg_nn/agg_constrained_nn.py` | 3d | Aggregation-constrained NN (Phase 2/3, in progress) |
| `Code/train/03_train_rf_gb/rf_nn_huerto_yield.py` | 3c | Avocado huerto-level predictions |
| `Code/analysis/02_accuracy_maize/accuracy_metrics.ipynb` | 5e | Multi-year validation |
| `Code/train/02_extract_ndvi_histograms/ls_ndvi_hists.py` | 1c | Landsat NDVI histogram extraction |
| `Code/train/04_train_cnn/ndvi_nn_redux.py` | 3b | CNN on NDVI histograms |
| `Code/train/01_extract_embeddings/ee_alpha_earth.py` | 1b | Simplified AEF extraction |
| `Code/train/02_extract_ndvi_histograms/clean_histograms_redux.py` | 3b | Histogram preprocessing |
| `Code/train/04_train_cnn/train_neural_net.py` | 3b | CNN training (Keras) |

### Paper Output Files

| Output File (in `Maize_prediction/plots/`) | Paper Element | Generated By |
|----------------------------|---------------|--------------|
| `accuracy_combined_2022.tex` | Table 1 | `1_accuracy_metrics_2022.ipynb` |
| `accuracy_fall_winter_2022.tex` | Table 2 | `1_accuracy_metrics_2022.ipynb` |
| `accuracy_spring_summer_2022.tex` | Table 3 | `1_accuracy_metrics_2022.ipynb` |
| `accuracy_other_crops_adc_2022.tex` | Table 4 | `1_accuracy_other_crops_2022.ipynb` |
| `accuracy_other_crops_mun_2022.tex` | Table 5 | `1_accuracy_other_crops_2022.ipynb` |
| `maizeyield_mun_allmx.png` | Figure 1a | `1_plot_yields_ADC_mun.ipynb` |
| `maizeyield_allmx_adc_with_legend.png` | Figure 1b | `1_plot_yields_ADC_mun.ipynb` |
| `maizeyield_mun.png` | Figure 1c | `1_plot_yields_ADC_mun.ipynb` |
| `maizeyield_adc.png` | Figure 1d | `1_plot_yields_ADC_mun.ipynb` |
| `maize_monthly_harvesting.png` | Figure 2 | `1_CleanSIAPMonthlydata.ipynb` |
| `maizeyield_mun_siap_allmx_nolegend_2018.png` | Figure 3a | `1_plot_yields_ADC_mun.ipynb` |
| `maizeyield_mun_pred_allmx_nolegend_2018.png` | Figure 3b | `1_plot_yields_ADC_mun.ipynb` |
| `accuracy_scatter_combined_2022.pdf` | Figure 4 | `1_accuracy_metrics_2022.ipynb` |
| `maizeyield_allmx_adc.png` | Figure 5a | `1_plot_yields_ADC_mun.ipynb` |
| `maizeyield_adc_preds.png` | Figure 5b | `1_plot_yields_ADC_mun.ipynb` |
| `maizeyield_adc_pred_error_prederror_ls_noleg.png` | Figure 6a | `1_plot_yields_ADC_mun.ipynb` |
| `maizeyield_adc_pred_error_prederror_rcpred_noleg.png` | Figure 6b | `1_plot_yields_ADC_mun.ipynb` |
| `maizeyield_adc_pred_error_prederror_munyield.png` | Figure 6c | `1_plot_yields_ADC_mun.ipynb` |
| `accuracy_scatter_seasonal_2022.pdf` | Figure 7 | `1_accuracy_metrics_2022.ipynb` |

---

## Notebooks Not Used in Current Paper

These exist in `Code/archive/` or `Code/` and serve the original poppy detection project or exploratory work:

| Notebook | Purpose |
|----------|---------|
| `2020-02-25_cleanEMIF.ipynb` | Clean EMIF migration survey data |
| `2020-02-25_cleanENADID.ipynb` | Clean ENADID survey data |
| `2020-05-27_plot_maps_MX.ipynb` | General Mexico map plotting |
| `2021-04-22_plot_exporter_locations.ipynb` | Map drug exporter locations |
| `2021-04-11_create_suitability_index.ipynb` | Poppy suitability index |
| `2021-03-03_create_suitability_index_Myanmar.ipynb` | Myanmar comparison |
| `2021-04-10_AWS_glacier_storage.ipynb` | AWS cloud storage management |
| `2023-04-16_plot_Kenya_data.ipynb` | Kenya satellite data exploration |
| `investigate_degroot_data.ipynb` | DeGroot dataset exploration |
| `2020-03-12_draw_municipality_maps.R` | R-based municipality maps |
