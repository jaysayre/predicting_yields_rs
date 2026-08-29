### 03_accuracy_other_crops.mk
# Non-maize crop accuracy (sorghum, sugarcane, wheat, avocados).
#   1_other_crops_adc_table.py   Table A3 from the per-crop prediction parquets
#   2_other_crops_mun_agg.py     Table A4 (mun-aggregated, ex-ante weights)
# The per-crop parquets are trained by train/03_train_rf_gb/
# gb_aef_hist_ensemble_other_crops.py (make train_other_crops).
# dep/1_accuracy_other_crops_2022.ipynb is the superseded legacy notebook; its
# plots/ outputs share filenames with the live tables/ ones — do not re-run it.
SHELL := /bin/bash   # 'source' for conda activation needs bash, not dash

CODE_DIR  := $(PROJ_DIR)/Code
TASK_DIR  := $(CODE_DIR)/analysis/03_accuracy_other_crops
TRAIN_DIR := $(CODE_DIR)/train/03_train_rf_gb

TABLES_DIR := $(DATA_DIR)/tables
PREDS_DIR  := $(DATA_DIR)/Data/predictions

.PHONY: accuracy_other_crops
accuracy_other_crops: \
	$(TABLES_DIR)/accuracy_other_crops_adc_2022.tex \
	$(TABLES_DIR)/accuracy_other_crops_mun_2022.tex

# Per-crop ensemble predictions (training; ~30 min). Grouped target on the
# sorghum parquet as representative — one run writes all four crops.
$(PREDS_DIR)/adc_aef_hist_ens_preds_sorghum.parquet: $(TRAIN_DIR)/gb_aef_hist_ensemble_other_crops.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# Step 1 — Table A3: assemble ensemble rows into the ADC accuracy table
$(TABLES_DIR)/accuracy_other_crops_adc_2022.tex: $(TASK_DIR)/1_other_crops_adc_table.py $(PREDS_DIR)/adc_aef_hist_ens_preds_sorghum.parquet
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# Step 2 — Table A4: municipality-aggregated accuracy (ex-ante agland weights)
$(TABLES_DIR)/accuracy_other_crops_mun_2022.tex: $(TASK_DIR)/2_other_crops_mun_agg.py $(PREDS_DIR)/adc_aef_hist_ens_preds_sorghum.parquet
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<
