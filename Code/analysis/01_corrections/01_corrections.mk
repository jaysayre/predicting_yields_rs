### 01_corrections.mk
# Post-prediction corrections (additive, GP, irrigation)

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/analysis/01_corrections

PREDS_DIR := $(DATA_DIR)/Data/predictions
PLOTS_DIR := $(DATA_DIR)/plots

.PHONY: corrections
corrections: \
	$(PREDS_DIR)/adc_yield_preds_corrected_2022.csv \
	$(PREDS_DIR)/adc_gp_yield_preds_2022.csv \
	$(PREDS_DIR)/adc_alpha_earth_preds_maize_irrig_adj.parquet \
	$(PLOTS_DIR)/maizeyield_mun_pred_allmx_nolegend_2022.png \
	$(PLOTS_DIR)/siap_mun_vs_adc_census_yield.pdf

# Additive correction + yield maps (Figures 1, 3, 5, 6)
$(PREDS_DIR)/adc_yield_preds_corrected_2022.csv: $(TASK_DIR)/1_plot_yields_ADC_mun.ipynb
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $<

# Gaussian Process correction
$(PREDS_DIR)/adc_gp_yield_preds_2022.csv: $(TASK_DIR)/2_gp_correction_2022.ipynb $(PREDS_DIR)/adc_yield_preds_corrected_2022.csv
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $<

# Irrigation adjustment
$(PREDS_DIR)/adc_alpha_earth_preds_maize_irrig_adj.parquet: $(TASK_DIR)/irrigation_adjustment.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# ── CA22 prediction & error maps (Figures 5–7 + municipal SIAP/pred pair) ──
# AEF Hist Ens. Shrink predictions + error maps on Census-Ag-2022 yields, and
# the _2022 municipal SIAP-vs-predicted pair. The CA07 census-yield maps
# (Figure 1) stay with 1_plot_yields_ADC_mun.ipynb; pass --census-maps to the
# script only if you deliberately want CA22 versions of those.
$(PLOTS_DIR)/maizeyield_mun_pred_allmx_nolegend_2022.png: $(TASK_DIR)/plot_yields_ca22_maps.py
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# Figure 1 panel (e): SIAP municipal yield vs ADC census yield hexbin.
# Writes the PDF to plots/ and directly to Overleaf figures/.
$(PLOTS_DIR)/siap_mun_vs_adc_census_yield.pdf: $(TASK_DIR)/3_plot_siap_vs_adc_census_scatter.ipynb $(PREDS_DIR)/adc_aef_hist_ens_eval.parquet
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $<
