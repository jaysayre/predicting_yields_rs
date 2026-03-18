### 02_extract_ndvi_histograms.mk
# Extract Landsat NDVI histograms from Google Earth Engine
# MANUAL: Requires Earth Engine authentication

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/train/02_extract_ndvi_histograms

.PHONY: extract_histograms extract_monthly_histograms \
       prepare_adc_geometries \
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
