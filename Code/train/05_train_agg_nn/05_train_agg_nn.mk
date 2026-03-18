### 05_train_agg_nn.mk
# Train aggregation-constrained NN on AEF embeddings

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/train/05_train_agg_nn

PREDS_DIR := $(DATA_DIR)/Data/predictions

.PHONY: train_agg_nn
train_agg_nn: $(PREDS_DIR)/adc_agg_nn_preds_maize_phase2_final.parquet

$(PREDS_DIR)/adc_agg_nn_preds_maize_phase2_final.parquet: $(TASK_DIR)/agg_constrained_nn.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $< --epochs 200 --hidden_dim 256 --batch_size 64 --var_lambda 0.01
