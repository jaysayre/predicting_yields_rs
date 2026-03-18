### 04_build_correspondence.mk
# Build ADC correspondence tables (2007↔2016)

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/build/04_build_correspondence

CORR_DIR := $(DATA_DIR)/Data/corr_tables

.PHONY: build_correspondence
build_correspondence: $(CORR_DIR)/ca07_ca16_corr.dta

$(CORR_DIR)/ca07_ca16_corr.dta: $(TASK_DIR)/1_build_correspondence_ADCs.ipynb $(TASK_DIR)/2_associate_boxes_ADCs.ipynb
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $(TASK_DIR)/1_build_correspondence_ADCs.ipynb
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $(TASK_DIR)/2_associate_boxes_ADCs.ipynb
