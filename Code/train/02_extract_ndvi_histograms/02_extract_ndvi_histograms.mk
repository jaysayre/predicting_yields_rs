### 02_extract_ndvi_histograms.mk
# Extract Landsat NDVI histograms from Google Earth Engine
# MANUAL: Requires Earth Engine authentication

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/train/02_extract_ndvi_histograms

.PHONY: extract_histograms extract_monthly_histograms \
       prepare_adc_geometries \
       extract_band_histograms_muni extract_band_histograms_adc \
       extract_3period_histograms_muni extract_3period_histograms_adc \
       clean_3period_histograms_muni clean_3period_histograms_adc

# ── Original 2-period histograms (32x32x3) ───────────────
extract_histograms:
	@echo "MANUAL STEP: Run ls_ndvi_hists.py interactively"
	@echo "  Requires: Earth Engine auth + ML_env"
	@echo "  Args:     0.2 1.0 0.0 12.0 0.0 0.6 32 max"
	@echo "  Output:   GDrive → muni_vi_hists_0.2_1.0_0.0_12.0_0.0_0.6_32_max/"
	@echo ""
	@echo "Then run clean_histograms_redux.py to preprocess → pickles"

# ── New monthly histograms (32 bins × 8 months × 6 indices) ──
extract_monthly_histograms:
	@echo "MANUAL STEP: Run ls_monthly_hists.py interactively"
	@echo "  Requires: Earth Engine auth + ML_env"
	@echo "  Args:     32 8"
	@echo "  Output:   GDrive → muni_monthly_hists_32bins_8mo/"
	@echo ""
	@echo "After GEE exports complete, download from Drive and run:"
	@echo "  python clean_monthly_histograms.py Data/muni_monthly_hists_32bins_8mo Data/monthly_hists_clean 32 8"

# ── NIR/RED band 2D histograms (Stage 3: preserve absolute reflectance) ──
# 5 paired 2D histograms (NDVI, NIR, RED trajectories + early/late spectral
# space). Aim: lift RAW (uncorrected) histogram performance vs the NDVI-only
# 3-period model. Pipeline mirrors the 3-period one.
extract_band_histograms_muni:
	@echo "MANUAL STEP: Run ls_band_hists.py interactively"
	@echo "  Requires: Earth Engine auth + ML_env"
	@echo "  Args:     muni 32"
	@echo "  Output:   GDrive → muni_band_hists_32bins/"
	@echo "  Then: clean (adapt clean_3period_histograms.py for 5 hist cols) +"
	@echo "        train (adapt gb_3period_prediction.py -> gb_band_hist_prediction.py)"

extract_band_histograms_adc:
	@echo "MANUAL STEP: Run ls_band_hists.py interactively  (Args: adc 32)"
	@echo "  Output:   GDrive → adc_band_hists_32bins/"

# ── 3-period 2D histograms ───────────────────────────────

prepare_adc_geometries:
	cd $(DATA_DIR) && $(MPC_ENV) python3 $(TASK_DIR)/prepare_adc_geometries.py

extract_3period_histograms_muni:
	@echo "MANUAL STEP: Run ls_3period_hists.py interactively"
	@echo "  Requires: Earth Engine auth + ML_env"
	@echo "  Args:     muni 16"
	@echo "  Output:   GDrive → muni_3period_hists_16bins/"

extract_3period_histograms_adc:
	@echo "MANUAL STEP: Run ls_3period_hists.py interactively"
	@echo "  Requires: Earth Engine auth + ML_env"
	@echo "  Args:     adc 16"
	@echo "  Output:   GDrive → adc_3period_hists_16bins/"

clean_3period_histograms_muni:
	cd $(DATA_DIR) && $(ML_ENV) python3 $(TASK_DIR)/clean_3period_histograms.py \
		Data/muni_3period_hists_16bins Data/muni_3period_hists_clean 16 muni

clean_3period_histograms_adc:
	cd $(DATA_DIR) && $(ML_ENV) python3 $(TASK_DIR)/clean_3period_histograms.py \
		Data/adc_3period_hists_16bins Data/adc_3period_hists_clean 16 adc

# ── Cropland-masked aefn2 features (the paper's NDVI (masked) baseline) ──
# AEF-mirroring marginal features (150/unit-year) on ESA WorldCover cropland
# pixels. Full battle log + resubmission recipe: AEFN2_PULL_HANDOFF.md.
.PHONY: extract_cropland_features
extract_cropland_features:
	@echo "MANUAL STEP (Earth Engine): the paper's NDVI (masked) feature extraction"
	@echo "  1. $(MPC_ENV) python3 $(TASK_DIR)/prepare_adc_geometries.py    # if geometries stale"
	@echo "  2. Run ls_cropland_features.py (ML env, EE auth) — submits GEE batch exports"
	@echo "  3. TAG=crop_aefn2 $(TASK_DIR)/pull_cropland_csvs.sh            # pull by file-ID from Drive"
	@echo "  4. Consolidate → Data/cropland_features/ (see AEFN2_PULL_HANDOFF.md)"
	@echo "Downstream: analysis/02_accuracy_maize/masked_muni_cv.py + partial_masked_mun_train_adc_eval.py"
