### 02_accuracy_maize.mk
# Maize accuracy evaluation → Tables 1–3, Figures 4 & 7

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/analysis/02_accuracy_maize

PLOTS_DIR := $(DATA_DIR)/plots

.PHONY: accuracy_maize
accuracy_maize: \
	$(PLOTS_DIR)/accuracy_combined_2022.tex \
	$(PLOTS_DIR)/accuracy_scatter_combined_2022.pdf

$(PLOTS_DIR)/accuracy_combined_2022.tex $(PLOTS_DIR)/accuracy_scatter_combined_2022.pdf &: $(TASK_DIR)/1_accuracy_metrics_2022.ipynb
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $<
