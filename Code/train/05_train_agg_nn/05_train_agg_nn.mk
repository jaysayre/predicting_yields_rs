### 05_train_agg_nn.mk
# Train aggregation-constrained NN on AEF embeddings; also emits Table 1
# (validation_mun_level_2022.tex) because it owns the holdout retrains it
# scores. 04_copy_to_overleaf copies the table into the paper.
# agg_nn_sweep.py (Phase C) is the producer of adc_mlp_yield_preds.csv — the
# "Agg-NN" input of the analysis tables; run via `make agg_nn_sweep` (slow,
# overwrites that csv with a timestamped backup).
SHELL := /bin/bash   # 'source' for conda activation needs bash, not dash

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/train/05_train_agg_nn

PREDS_DIR := $(DATA_DIR)/Data/predictions
PLOTS_DIR := $(DATA_DIR)/plots

TABLES_DIR := $(DATA_DIR)/tables

.PHONY: train_agg_nn
train_agg_nn: \
	$(PREDS_DIR)/adc_agg_nn_preds_maize_phase2_final.parquet \
	$(PREDS_DIR)/adc_aggnn_census_trained_preds.parquet \
	$(TABLES_DIR)/validation_mun_level_2022.tex

# Municipality-level held-out validation (Table \ref{tab:validation_mun})
$(TABLES_DIR)/validation_mun_level_2022.tex: $(TASK_DIR)/validation_mun_level.py $(PREDS_DIR)/adc_agg_nn_preds_maize_phase2_final.parquet $(PREDS_DIR)/adc_aef_hist_ens_holdout_preds.parquet $(PREDS_DIR)/mun_aefn2_masked_gb_holdout_preds.parquet
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# NDVI (masked) holdout row. The muni feature cache is written by
# analysis/02_accuracy_maize/1_masked_muni_cv.py; the cross-stage prerequisite
# is now ENCODED (2026-08-28) so a from-scratch `make all` builds it first.
CROPFEAT_DIR := $(DATA_DIR)/Data/cropland_features
$(CROPFEAT_DIR)/muni_aefn2_masked.parquet: $(CODE_DIR)/analysis/02_accuracy_maize/1_masked_muni_cv.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

$(PREDS_DIR)/mun_aefn2_masked_gb_holdout_preds.parquet: $(TASK_DIR)/train_holdout_validation_models.py $(CROPFEAT_DIR)/muni_aefn2_masked.parquet
	cd $(DATA_DIR) && $(ML_ENV) python3 $< masked

# Holdout-clean model retraining for Table 1 (writes the five *_holdout_*
# prediction files; the ensemble file is last, so it stands in for all five)
$(PREDS_DIR)/adc_aef_hist_ens_holdout_preds.parquet: $(TASK_DIR)/train_holdout_validation_models.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

$(PREDS_DIR)/adc_agg_nn_preds_maize_phase2_final.parquet: $(TASK_DIR)/agg_constrained_nn.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $< --epochs 200 --hidden_dim 256 --batch_size 64 --var_lambda 0.01

# Census training-quality thought experiment: census-trained Agg-NN retrain.
# Saves ensemble-avg ADC 2022 predictions; the table itself is built by
# analysis/02_accuracy_maize/6_census_thought_all_models.py from this file.
$(PREDS_DIR)/adc_aggnn_census_trained_preds.parquet: $(TASK_DIR)/census_thought_experiment.py $(PREDS_DIR)/adc_agg_nn_preds_maize_phase2_final.parquet
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# Phase C hyperparameter sweep + 5-seed ensemble; PRODUCES the Agg-NN input
# adc_mlp_yield_preds.csv consumed by analysis/02 (timestamped backup first).
.PHONY: agg_nn_sweep
agg_nn_sweep: $(TASK_DIR)/agg_nn_sweep.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<
