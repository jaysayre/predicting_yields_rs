### 05_phenology.mk
# Determine crop seasonality (match NDVI phenology to SIAP months)
SHELL := /bin/bash   # 'source' for conda activation needs bash, not dash

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/build/05_phenology

SIAP_OUT := $(DATA_DIR)/Data/SIAP_monthly/Output

# One notebook writes the SIAP-vs-NDVI comparison plus the per-municipality
# planting-month file consumed by train/02 step 1 and build/06 step 2.
.PHONY: phenology
phenology: $(SIAP_OUT)/comp_sat_plant.csv

$(SIAP_OUT)/comp_sat_plant.csv \
$(DATA_DIR)/Data/planting_months_harmonic_regression.csv \
$(DATA_DIR)/Data/muni_ndvi_peak_dates.csv \
$(DATA_DIR)/Data/muni_ndvi_min_dates.csv &: $(TASK_DIR)/1_Determine_Monthly_peaks.ipynb $(SIAP_OUT)/mnthly_siap.dta
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $<
