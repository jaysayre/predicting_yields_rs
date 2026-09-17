### 04_train_agg_nn.mk
# Aggregation-constrained NN ("Agg-NN" benchmark) and the holdout-clean
# retrains behind the municipality validation table. Scripts run in step order:
#   1_agg_constrained_nn.py            Phase-2 Agg-NN -> adc_agg_nn_preds_maize_phase2_{final,inner_es}.parquet
#   2_agg_nn_sweep.py                  Phase-C hyperparameter sweep + 5-seed ensemble
#                                      -> adc_mlp_yield_preds.csv (the "Agg-NN" rows of every table; slow)
#   3_train_holdout_validation_models.py  retrain every model on 80% of municipalities
#                                      -> *_holdout_preds.parquet  (`masked` arg: NDVI (masked) row)
#   4_validation_mun_level.py          Table validation_mun_level_2022.tex (held-out municipality-years)
#   5_census_thought_experiment.py     census-trained Agg-NN retrain -> adc_aggnn_census_trained_preds.parquet
#                                      (census-trained panel of the thought-experiment table, analysis/02 step 6)
SHELL := /bin/bash   # 'source' for conda activation needs bash, not dash

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/train/04_train_agg_nn

AEF_DIR    := $(DATA_DIR)/Data/alpha_earth
PREDS_DIR  := $(DATA_DIR)/Data/predictions
TABLES_DIR := $(DATA_DIR)/tables
CROPFEAT_DIR := $(DATA_DIR)/Data/cropland_features

.PHONY: train_agg_nn
train_agg_nn: \
	$(PREDS_DIR)/adc_agg_nn_preds_maize_phase2_final.parquet \
	$(PREDS_DIR)/adc_mlp_yield_preds.csv \
	$(PREDS_DIR)/adc_aggnn_census_trained_preds.parquet \
	$(TABLES_DIR)/validation_mun_level_2022.tex

# Step 1 — Phase-2 Agg-NN
$(PREDS_DIR)/adc_agg_nn_preds_maize_phase2_final.parquet \
$(PREDS_DIR)/adc_agg_nn_preds_maize_phase2_inner_es.parquet &: $(TASK_DIR)/1_agg_constrained_nn.py $(AEF_DIR)/alpha_earth_mex_adcs.parquet
	cd $(DATA_DIR) && $(ML_ENV) python3 $< --epochs 200 --hidden_dim 256 --batch_size 64 --var_lambda 0.01

# Step 2 — Phase-C sweep: PRODUCES the Agg-NN input of analysis/02 (keeps a
# timestamped backup of the previous csv first)
$(PREDS_DIR)/adc_mlp_yield_preds.csv: $(TASK_DIR)/2_agg_nn_sweep.py $(AEF_DIR)/alpha_earth_mex_adcs.parquet
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# Step 3 — holdout retrains (one run writes the AEF mean, AEF Hist and AEF
# Hist Ens. holdout files; the NDVI (masked) file needs the `masked` argument
# and the municipality feature cache from analysis/02 step 1)
$(PREDS_DIR)/adc_aef_hist_ens_holdout_preds.parquet \
$(PREDS_DIR)/adc_aef_hist_bins_gb_holdout_preds.parquet \
$(PREDS_DIR)/adc_alpha_earth_holdout_preds_maize.parquet &: $(TASK_DIR)/3_train_holdout_validation_models.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

$(CROPFEAT_DIR)/muni_aefn2_masked.parquet \
$(PREDS_DIR)/mun_aefn2_masked_gb_kfold_preds.parquet &: $(CODE_DIR)/analysis/02_accuracy_maize/1_masked_muni_cv.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

$(PREDS_DIR)/mun_aefn2_masked_gb_holdout_preds.parquet: $(TASK_DIR)/3_train_holdout_validation_models.py $(CROPFEAT_DIR)/muni_aefn2_masked.parquet
	cd $(DATA_DIR) && $(ML_ENV) python3 $< masked

# Step 4 — municipality-level held-out validation table
$(TABLES_DIR)/validation_mun_level_2022.tex: $(TASK_DIR)/4_validation_mun_level.py \
		$(PREDS_DIR)/adc_agg_nn_preds_maize_phase2_inner_es.parquet \
		$(PREDS_DIR)/adc_aef_hist_ens_holdout_preds.parquet \
		$(PREDS_DIR)/adc_aef_hist_bins_gb_holdout_preds.parquet \
		$(PREDS_DIR)/adc_alpha_earth_holdout_preds_maize.parquet \
		$(PREDS_DIR)/mun_aefn2_masked_gb_holdout_preds.parquet
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# Step 5 — census-trained Agg-NN (training-quality thought experiment)
$(PREDS_DIR)/adc_aggnn_census_trained_preds.parquet: $(TASK_DIR)/5_census_thought_experiment.py $(PREDS_DIR)/adc_agg_nn_preds_maize_phase2_final.parquet
	cd $(DATA_DIR) && $(ML_ENV) python3 $<
