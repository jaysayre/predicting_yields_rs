### 02_clean_siap_monthly.mk
# Clean SIAP monthly production data

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/build/02_clean_siap_monthly

SIAP_OUT  := $(DATA_DIR)/Data/SIAP_monthly/Output
PLOTS_DIR := $(DATA_DIR)/plots

.PHONY: clean_siap
clean_siap: $(SIAP_OUT)/mnthly_siap.dta $(PLOTS_DIR)/maize_monthly_harvesting.png

$(SIAP_OUT)/mnthly_siap.dta: $(TASK_DIR)/1_CleanSIAPMonthlydata.ipynb
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $<

# Monthly harvest-share figure (paper appendix; also used by the PAP/older
# body files). Rebuilt from the cleaned monthly panel.
$(PLOTS_DIR)/maize_monthly_harvesting.png: $(TASK_DIR)/plot_maize_monthly_harvesting.py $(SIAP_OUT)/mnthly_siap.dta
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<
