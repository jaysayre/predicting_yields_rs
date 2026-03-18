### 01_corrections.mk
# Post-prediction corrections (additive, GP, irrigation)

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/analysis/01_corrections

PREDS_DIR := $(DATA_DIR)/Data/predictions

.PHONY: corrections
corrections: \
	$(PREDS_DIR)/adc_yield_preds_corrected_2022.csv \
	$(PREDS_DIR)/adc_gp_yield_preds_2022.csv \
	$(PREDS_DIR)/adc_alpha_earth_preds_maize_irrig_adj.parquet

# Additive correction + yield maps (Figures 1, 3, 5, 6)
$(PREDS_DIR)/adc_yield_preds_corrected_2022.csv: $(TASK_DIR)/1_plot_yields_ADC_mun.ipynb
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $<

# Gaussian Process correction
$(PREDS_DIR)/adc_gp_yield_preds_2022.csv: $(TASK_DIR)/2_gp_correction_2022.ipynb $(PREDS_DIR)/adc_yield_preds_corrected_2022.csv
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $<

# Irrigation adjustment
$(PREDS_DIR)/adc_alpha_earth_preds_maize_irrig_adj.parquet: $(TASK_DIR)/irrigation_adjustment.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<
