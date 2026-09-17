### 02_accuracy_maize.mk
# Maize accuracy evaluation — every maize paper table/figure is built here except
# the municipality validation table (train/04, needs the holdout retrains) and
# the other-crop tables (analysis/03). Scripts run in step order:
#   1_masked_muni_cv              NDVI (masked) muni CV + muni feature cache
#   2_masked_adc_eval             NDVI (masked) ADC preds  -> NDVI rows everywhere
#   3_oracle_adc_ceiling          oracle (ADC-trained) benchmark csv
#   4_accuracy_main_2022          main accuracy tables (spring-summer, combined, fall-winter),
#                                 common-sample tables, scatter figures
#   5_mun_survey_improvement      municipality-aggregation table (accuracy_mun_level)
#   6_census_thought_all_models   census thought-experiment table
#   7_accuracy_profile_by_adc_chars  accuracy by ADC characteristics
#   8_accuracy_cimmyt_profile     CIMMYT external-validation table
#   9_exante_trust_features       ex-ante per-municipality feature csv
#   10_exante_trust_composite     unsupervised ex-ante skill index (+ validation prints)
#   11_exante_trust_across_models cross-model RF validation csv
#   12_fig_representativeness_targeting / 13_fig_ranking_inversion   ex-ante targeting figures
#   14_fetch_mun_chips_fig        AEF chips for the featurization diagram (manual GEE fetch, run once)
#   15_fig_methodology_diagram    featurization diagram (fig:methodology)
#   16_fig_pipeline_diagram       pipeline overview (fig:pipeline)
#   17_rho_irrigation_bound       lambda from the irrigation projection (public data)
#   18_w_public_selection         ensemble weight w from public data
#   19_robustness_lambda_ci_qbin  bootstrap CIs, lambda curve, quantile-bin robustness
#   20_sample_accounting_and_lambda  sample-size accounting + dispersion gauge
# Steps 17-20 print the numbers cited in the prose (no .tex output); the
# lambda = 0.72 and w = 0.5 they select are entered as constants in steps 4-16
# (LAM / W in each script). Run them with `make robustness_stats` etc.
# dep/ holds superseded generators and one-off diagnostics.
#
# Model prediction inputs (all produced in-repo):
#   "AEF Hist"      adc_aef_hist_bins_gb_preds.parquet   train/03 step 2
#   "AEF Hist Ens." adc_aef_hist_ens_eval.parquet        train/03 step 2
#   "AEF mean"      adc_alpha_earth_preds_maize.parquet  train/03 step 1
#   "Agg-NN"        adc_mlp_yield_preds.csv              train/04 step 2
#   "NDVI (masked)" adc_aefn2_masked_preds.parquet       step 2 below
SHELL := /bin/bash   # 'source' for conda activation needs bash, not dash

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/analysis/02_accuracy_maize

PLOTS_DIR  := $(DATA_DIR)/plots
TABLES_DIR := $(DATA_DIR)/tables
PREDS_DIR  := $(DATA_DIR)/Data/predictions
EXTRAS_DIR := $(DATA_DIR)/plots/coauthor_extras_paper

.PHONY: accuracy_maize
accuracy_maize: \
	$(PLOTS_DIR)/accuracy_combined_2022.tex \
	$(PLOTS_DIR)/accuracy_spring_summer_2022.tex \
	$(PLOTS_DIR)/accuracy_fall_winter_2022.tex \
	$(PLOTS_DIR)/common_sample_combined_2022.tex \
	$(PLOTS_DIR)/common_sample_spring_summer_2022.tex \
	$(PLOTS_DIR)/accuracy_scatter_combined_2022.pdf \
	$(TABLES_DIR)/accuracy_mun_level_2022.tex \
	$(TABLES_DIR)/census_thought_experiment_2022.tex \
	$(TABLES_DIR)/accuracy_profile_by_adc_chars.tex \
	$(TABLES_DIR)/accuracy_cimmyt_profile.tex \
	$(EXTRAS_DIR)/fig_representativeness_targeting.pdf \
	$(EXTRAS_DIR)/fig_ranking_inversion.pdf \
	$(PLOTS_DIR)/fig_methodology_diagram.pdf \
	$(PLOTS_DIR)/fig_pipeline_diagram.pdf

MODEL_PREDS := $(PREDS_DIR)/adc_aef_hist_bins_gb_preds.parquet $(PREDS_DIR)/adc_aef_hist_ens_eval.parquet \
               $(PREDS_DIR)/adc_alpha_earth_preds_maize.parquet $(PREDS_DIR)/adc_mlp_yield_preds.csv \
               $(PREDS_DIR)/adc_aefn2_masked_preds.parquet

# ── Step 1: NDVI (masked) muni-level CV + feature cache ──────────────────
# Also writes Data/cropland_features/muni_aefn2_masked.parquet, which
# train/04_train_agg_nn (holdout retrains) consumes.
$(PREDS_DIR)/mun_aefn2_masked_gb_kfold_preds.parquet \
$(DATA_DIR)/Data/cropland_features/muni_aefn2_masked.parquet &: $(TASK_DIR)/1_masked_muni_cv.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# ── Step 2: muni-trained masked model scored at the ADC level ────────────
# -> adc_aefn2_masked_preds.parquet, the NDVI rows of every accuracy table.
$(PREDS_DIR)/adc_aefn2_masked_preds.parquet: $(TASK_DIR)/2_masked_adc_eval.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# ── Step 3: oracle ADC-trained ceiling (non-deployable benchmark row) ────
$(PREDS_DIR)/oracle_ceiling_2022.csv: $(TASK_DIR)/3_oracle_adc_ceiling.py
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# ── Step 4: main accuracy tables + common-sample tables + scatter figures ─
# One grouped rule for everything 4_accuracy_main_2022.py writes.
$(PLOTS_DIR)/accuracy_combined_2022.tex $(PLOTS_DIR)/accuracy_spring_summer_2022.tex $(PLOTS_DIR)/accuracy_fall_winter_2022.tex $(PLOTS_DIR)/common_sample_combined_2022.tex $(PLOTS_DIR)/common_sample_spring_summer_2022.tex $(PLOTS_DIR)/accuracy_scatter_combined_2022.pdf $(PLOTS_DIR)/accuracy_scatter_seasonal_2022.pdf &: $(TASK_DIR)/4_accuracy_main_2022.py $(PREDS_DIR)/oracle_ceiling_2022.csv $(MODEL_PREDS)
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# ── Step 5: municipality-aggregation table (ex-ante ag-land weights) ──────
$(TABLES_DIR)/accuracy_mun_level_2022.tex: $(TASK_DIR)/5_mun_survey_improvement.py \
		$(PREDS_DIR)/mun_aefn2_masked_gb_kfold_preds.parquet $(MODEL_PREDS)
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# ── Step 6: census thought-experiment table (all models) ─────────────────
# The Agg-NN (census-trained) panel consumes the retrain saved by
# train/04_train_agg_nn/5_census_thought_experiment.py.
$(TABLES_DIR)/census_thought_experiment_2022.tex: $(TASK_DIR)/6_census_thought_all_models.py $(PREDS_DIR)/adc_aggnn_census_trained_preds.parquet $(MODEL_PREDS)
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# ── Step 7: accuracy by ADC characteristics ──────────────────────────────
$(TABLES_DIR)/accuracy_profile_by_adc_chars.tex: $(TASK_DIR)/7_accuracy_profile_by_adc_chars.py
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# ── Step 8: CIMMYT farmer-trial external validation ──────────────────────
$(TABLES_DIR)/accuracy_cimmyt_profile.tex: $(TASK_DIR)/8_accuracy_cimmyt_profile.py
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# ── Steps 9-11: ex-ante trust chain ──────────────────────────────────────
# 9 builds the per-mun feature csv (extracted from the retired exante
# notebook); 10 adds the unsupervised trust_index + refreshes within_r2_mun
# from the deployed Shrink predictions; 11 owns the cross-model RF csv.
$(EXTRAS_DIR)/exante_trust_index.csv: $(TASK_DIR)/9_exante_trust_features.py $(TASK_DIR)/10_exante_trust_composite.py $(PREDS_DIR)/adc_aef_hist_ens_eval.parquet
	cd $(DATA_DIR) && $(ML_ENV) python3 $(TASK_DIR)/9_exante_trust_features.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $(TASK_DIR)/10_exante_trust_composite.py

$(EXTRAS_DIR)/exante_trust_across_models.csv: $(TASK_DIR)/11_exante_trust_across_models.py $(EXTRAS_DIR)/exante_trust_index.csv $(MODEL_PREDS)
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# ── Steps 12-13: ex-ante targeting figures ───────────────────────────────
$(EXTRAS_DIR)/fig_representativeness_targeting.pdf: $(TASK_DIR)/12_fig_representativeness_targeting.py $(EXTRAS_DIR)/exante_trust_index.csv
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

$(EXTRAS_DIR)/fig_ranking_inversion.pdf: $(TASK_DIR)/13_fig_ranking_inversion.py $(EXTRAS_DIR)/exante_trust_across_models.csv
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# ── Steps 14-16: featurization diagram + pipeline overview ───────────────
# 14 fetches GEE chips (network; run once), 15 draws the featurization
# diagram, 16 the pipeline overview.
.PHONY: fetch_fig2_chips
fetch_fig2_chips: $(TASK_DIR)/14_fetch_mun_chips_fig.py
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

$(PLOTS_DIR)/fig_methodology_diagram.pdf: $(TASK_DIR)/15_fig_methodology_diagram.py $(PREDS_DIR)/mun_aef_hist_bins_gb_loyo_preds.parquet $(DATA_DIR)/Data/alpha_earth/fig2_mun_chips_meta.json
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

$(PLOTS_DIR)/fig_pipeline_diagram.pdf: $(TASK_DIR)/16_fig_pipeline_diagram.py $(PREDS_DIR)/adc_aef_hist_ens_eval.parquet
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# ── Steps 17-20: prose-stat utilities (numbers cited in text, no .tex output)
.PHONY: prose_stats rho_irrigation_bound w_public_selection robustness_stats sample_accounting
prose_stats: rho_irrigation_bound w_public_selection robustness_stats sample_accounting

# 17 — irrigation-projection bound + midpoint estimate for rho / lambda
rho_irrigation_bound: $(TASK_DIR)/17_rho_irrigation_bound.py $(PREDS_DIR)/adc_aef_hist_ens_eval.parquet
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# 18 — public-data selection of the ensemble weight w
w_public_selection: $(TASK_DIR)/18_w_public_selection.py $(PREDS_DIR)/adc_aef_hist_ens_components.parquet
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# 19 — bootstrap CIs, lambda curve, quantile-bin robustness (needs make train_aef_hist_ens_qbin)
robustness_stats: $(TASK_DIR)/19_robustness_lambda_ci_qbin.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# 20 — sample-size accounting + dispersion gauge
sample_accounting: $(TASK_DIR)/20_sample_accounting_and_lambda.py $(PREDS_DIR)/mun_aef_hist_bins_gb_kfold_preds.parquet
	cd $(DATA_DIR) && $(ML_ENV) python3 $<
