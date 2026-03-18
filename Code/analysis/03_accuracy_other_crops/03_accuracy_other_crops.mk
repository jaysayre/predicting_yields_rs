### 03_accuracy_other_crops.mk
# Other crops accuracy evaluation → Tables 4–5

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/analysis/03_accuracy_other_crops

PLOTS_DIR := $(DATA_DIR)/plots

.PHONY: accuracy_other_crops
accuracy_other_crops: \
	$(PLOTS_DIR)/accuracy_other_crops_adc_2022.tex \
	$(PLOTS_DIR)/accuracy_other_crops_mun_2022.tex

$(PLOTS_DIR)/accuracy_other_crops_adc_2022.tex $(PLOTS_DIR)/accuracy_other_crops_mun_2022.tex &: $(TASK_DIR)/1_accuracy_other_crops_2022.ipynb
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $<
