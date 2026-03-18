### 02_clean_siap_monthly.mk
# Clean SIAP monthly production data

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/build/02_clean_siap_monthly

SIAP_OUT := $(DATA_DIR)/Data/SIAP_monthly/Output

.PHONY: clean_siap
clean_siap: $(SIAP_OUT)/mnthly_siap.dta

$(SIAP_OUT)/mnthly_siap.dta: $(TASK_DIR)/1_CleanSIAPMonthlydata.ipynb
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $<
