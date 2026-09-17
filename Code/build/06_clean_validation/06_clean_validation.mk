### 06_clean_validation.mk
# CIMMYT farmer-trial validation data. Steps:
#   1_clean_CIMMYT_farmer_plots.ipynb   logbook + yields xlsx -> plot shapefile + maize yields csv
#   2_prepare_cimmyt_geometries.ipynb   buffer plots to polygons, join ADCs/planting month
#                                       -> Data/CIMMYT/cimmyt_plot_geometries_for_ee.csv (train/01 steps 11-13)
SHELL := /bin/bash   # 'source' for conda activation needs bash, not dash

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/build/06_clean_validation

CIMMYT_DIR := $(DATA_DIR)/Intermediates/CIMMYT

.PHONY: clean_validation
clean_validation: $(DATA_DIR)/Data/CIMMYT/cimmyt_plot_geometries_for_ee.csv

# Step 1
$(CIMMYT_DIR)/cimmyt_maize_yields.csv $(CIMMYT_DIR)/cimmyt_plot_locations.shp &: $(TASK_DIR)/1_clean_CIMMYT_farmer_plots.ipynb
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $<

# Step 2
$(DATA_DIR)/Data/CIMMYT/cimmyt_plot_geometries_for_ee.csv: $(TASK_DIR)/2_prepare_cimmyt_geometries.ipynb $(CIMMYT_DIR)/cimmyt_maize_yields.csv $(DATA_DIR)/Data/planting_months_harmonic_regression.csv
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $<
