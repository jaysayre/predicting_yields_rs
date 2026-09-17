### 01_corrections.mk
# Figure 1 panels and the CA22 prediction/error maps. Scripts run in step order:
#   1_plot_yields_ADC_mun.ipynb        CA07 census maps (fig:adc_mun_yield_comp a-d) + corrected csv
#   2_plot_siap_vs_adc_census_scatter  DGSIAP-vs-census hexbin (panel e)
#   3_plot_yields_ca22_maps.py         CA22 prediction/error maps + municipal DGSIAP/prediction pair
#   4_fig1_oaxaca_diff_map.py          ADC-minus-municipality map (panel f)
# dep/ holds the retired GP-correction chain and the unconsumed irrigation
# adjustment (neither appears in the paper).
SHELL := /bin/bash   # 'source' for conda activation needs bash, not dash

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/analysis/01_corrections

PREDS_DIR := $(DATA_DIR)/Data/predictions
PLOTS_DIR := $(DATA_DIR)/plots

.PHONY: corrections
corrections: \
	$(PREDS_DIR)/adc_yield_preds_corrected_2022.csv \
	$(PLOTS_DIR)/siap_mun_vs_adc_census_yield.pdf \
	$(PLOTS_DIR)/maizeyield_mun_pred_allmx_nolegend_2022.png \
	$(PLOTS_DIR)/maizeyield_adc_mun_diff.png

# Step 1 — additive correction + CA07 census yield maps (panels a-d)
$(PREDS_DIR)/adc_yield_preds_corrected_2022.csv: $(TASK_DIR)/1_plot_yields_ADC_mun.ipynb
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $<

# Step 2 — panel (e): DGSIAP municipal yield vs ADC census yield hexbin
$(PLOTS_DIR)/siap_mun_vs_adc_census_yield.pdf: $(TASK_DIR)/2_plot_siap_vs_adc_census_scatter.ipynb $(PREDS_DIR)/adc_aef_hist_ens_eval.parquet
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $<

# Step 3 — CA22 prediction & error maps + municipal DGSIAP/prediction pair.
# The CA07 census-yield maps stay with step 1; pass --census-maps to
# this script only if you deliberately want CA22 versions of those.
$(PLOTS_DIR)/maizeyield_mun_pred_allmx_nolegend_2022.png: $(TASK_DIR)/3_plot_yields_ca22_maps.py
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# Step 4 — panel (f): ADC census yield minus municipality census
# average, Oaxaca inset (CA2007 yields on the CA2007 ADC polygons)
$(PLOTS_DIR)/maizeyield_adc_mun_diff.png: $(TASK_DIR)/4_fig1_oaxaca_diff_map.py
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<
