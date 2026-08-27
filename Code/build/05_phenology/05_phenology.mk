### 05_phenology.mk
# Determine crop seasonality (match NDVI phenology to SIAP months)
SHELL := /bin/bash   # 'source' for conda activation needs bash, not dash

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/build/05_phenology

SIAP_OUT := $(DATA_DIR)/Data/SIAP_monthly/Output

.PHONY: phenology
phenology: $(SIAP_OUT)/comp_sat_plant.csv

$(SIAP_OUT)/comp_sat_plant.csv: $(TASK_DIR)/1_Determine_Monthly_peaks.ipynb
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $<
