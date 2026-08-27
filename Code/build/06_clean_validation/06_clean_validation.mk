### 06_clean_validation.mk
# Clean CIMMYT farmer trial validation data
SHELL := /bin/bash   # 'source' for conda activation needs bash, not dash

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/build/06_clean_validation

CIMMYT_DIR := $(DATA_DIR)/Intermediates/CIMMYT

.PHONY: clean_validation
clean_validation: $(CIMMYT_DIR)/cimmyt_maize_yields.csv

$(CIMMYT_DIR)/cimmyt_maize_yields.csv: $(TASK_DIR)/1_clean_CIMMYT_farmer_plots.ipynb
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $<
