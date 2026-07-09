### 05_train_agg_nn.mk
# Train aggregation-constrained NN on AEF embeddings

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/train/05_train_agg_nn

PREDS_DIR := $(DATA_DIR)/Data/predictions
PLOTS_DIR := $(DATA_DIR)/plots

TABLES_DIR := $(DATA_DIR)/tables

.PHONY: train_agg_nn
train_agg_nn: \
	$(PREDS_DIR)/adc_agg_nn_preds_maize_phase2_final.parquet \
	$(PLOTS_DIR)/census_thought_experiment_2022.tex \
	$(TABLES_DIR)/validation_mun_level_2022.tex

# Municipality-level held-out validation (Table \ref{tab:validation_mun})
$(TABLES_DIR)/validation_mun_level_2022.tex: $(TASK_DIR)/validation_mun_level.py $(PREDS_DIR)/adc_agg_nn_preds_maize_phase2_final.parquet
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

$(PREDS_DIR)/adc_agg_nn_preds_maize_phase2_final.parquet: $(TASK_DIR)/agg_constrained_nn.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $< --epochs 200 --hidden_dim 256 --batch_size 64 --var_lambda 0.01

# Census training-quality thought experiment (Table \ref{tab:census_thought})
# Writes census_thought_experiment_2022.tex to plots/ and directly to Overleaf.
$(PLOTS_DIR)/census_thought_experiment_2022.tex: $(TASK_DIR)/census_thought_experiment.py $(PREDS_DIR)/adc_agg_nn_preds_maize_phase2_final.parquet
	cd $(DATA_DIR) && $(ML_ENV) python3 $<
