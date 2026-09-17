### 02_extract_ndvi_histograms.mk
# Landsat NDVI features for the paper's "NDVI (masked)" baseline: the AEF
# featurization (means / percentiles / 8 equal-mass bins per dimension) applied
# to harmonically smoothed NDVI on ESA WorldCover cropland pixels. Steps:
#   1_prepare_adc_geometries.py   ADC + municipality polygons -> EE-ready geometry CSVs (+ planting month)
#   2_ls_cropland_features.py     MANUAL (Earth Engine): --level adc --years 2017-2024, tag aefn2
#   3_pull_cropland_csvs.sh       pull the finished exports by file-ID from Drive
#                                 -> Data/cropland_features/csvs_crop_aefn2/
# Downstream: analysis/02_accuracy_maize/1_masked_muni_cv.py (municipality
# features + CV) and 2_masked_adc_eval.py (ADC predictions). Full extraction
# log and resubmission recipe: AEFN2_PULL_HANDOFF.md.
SHELL := /bin/bash   # 'source' for conda activation needs bash, not dash

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/train/02_extract_ndvi_histograms
GEOM_CSV := $(DATA_DIR)/Data/adc_geometries_for_ee.csv
CROP_CSVS := $(DATA_DIR)/Data/cropland_features/csvs_crop_aefn2

.PHONY: extract_ndvi_features prepare_adc_geometries pull_cropland_csvs
extract_ndvi_features: $(GEOM_CSV)
	@echo "MANUAL STEP (Earth Engine auth + ML env):"
	@echo "  python3 $(TASK_DIR)/2_ls_cropland_features.py --level adc --years 2017-2024 --project <ee-project>"
	@echo "then, once the batch exports have finished:"
	@echo "  make -f $(TASK_DIR)/02_extract_ndvi_histograms.mk pull_cropland_csvs   (-> $(CROP_CSVS))"

# Step 1 — geometry CSVs (also writes muni_geometries_for_ee.csv)
prepare_adc_geometries: $(GEOM_CSV)
$(GEOM_CSV): $(TASK_DIR)/1_prepare_adc_geometries.py $(DATA_DIR)/Data/planting_months_harmonic_regression.csv
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# Step 3 — pull the exports (network; rclone 'gdrive' remote + EE task list)
pull_cropland_csvs: $(TASK_DIR)/3_pull_cropland_csvs.sh
	cd $(DATA_DIR) && TAG=crop_aefn2 DST=$(CROP_CSVS) bash $<
