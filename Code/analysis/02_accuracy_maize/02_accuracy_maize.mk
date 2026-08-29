### 02_accuracy_maize.mk
# Maize accuracy evaluation — every maize paper table/figure built here except
# Table 1 (train/05, needs the holdout retrains) and Table A3 (analysis/03).
# Scripts are numbered by step:
#   1_masked_muni_cv            NDVI(masked) muni CV + feature cache
#   2_masked_adc_eval           NDVI(masked) ADC preds  -> NDVI rows everywhere
#   3_oracle_adc_ceiling        oracle benchmark csv
#   4_accuracy_main_2022        Tables 2/A1/A2 + A5/A6 + scatter figures
#   5_mun_survey_improvement    Table 5
#   6_census_thought_all_models Table 3
#   7_accuracy_profile_by_adc_chars  Table 4
#   8_accuracy_cimmyt_profile   Table 6
#   9_exante_trust_features     ex-ante per-mun feature csv
#   10_exante_trust_composite   unsupervised trust index (+ validation prints)
#   11_exante_trust_across_models  cross-model RF validation csv
#   12/13_fig_*                 Figures 6 and 7 (pdf)
#   14/15                       Figure 2 chips fetch (manual GEE) + diagram
# Prose-stat utilities (unnumbered, run on demand): robustness_lambda_ci_qbin,
# sample_accounting_and_lambda, rho_irrigation_bound.
# dep/ holds superseded generators and one-off diagnostics.
#
# FROZEN COAUTHOR INPUTS (no in-repo producer; documented 2026-08-28):
#   Data/predictions/adc_alpha_earth_preds.csv     "AEF mean" rows
#   Data/predictions/adc_aef_hist_gb_preds.parquet "AEF Hist" rows
# The "Agg-NN" input adc_mlp_yield_preds.csv is produced by
# train/05_train_agg_nn/agg_nn_sweep.py (Phase C run).
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
	$(EXTRAS_DIR)/fig_ranking_inversion.pdf

# ── Step 1: NDVI (masked) muni-level CV + feature cache ──────────────────
# Also writes Data/cropland_features/muni_aefn2_masked.parquet, which
# train/05_train_agg_nn (holdout retrains) consumes.
$(PREDS_DIR)/mun_aefn2_masked_gb_kfold_preds.parquet: $(TASK_DIR)/1_masked_muni_cv.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# ── Step 2: muni-trained masked model scored at the ADC level ────────────
# -> adc_aefn2_masked_preds.parquet, the NDVI rows of every accuracy table.
$(PREDS_DIR)/adc_aefn2_masked_preds.parquet: $(TASK_DIR)/2_masked_adc_eval.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# ── Step 3: oracle ADC-trained ceiling (non-deployable benchmark row) ────
$(PREDS_DIR)/oracle_ceiling_2022.csv: $(TASK_DIR)/3_oracle_adc_ceiling.py
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# ── Step 4: main season tables + common-sample tables + scatter figures ──
# One grouped rule for everything 4_accuracy_main_2022.py writes.
$(PLOTS_DIR)/accuracy_combined_2022.tex $(PLOTS_DIR)/accuracy_spring_summer_2022.tex $(PLOTS_DIR)/accuracy_fall_winter_2022.tex $(PLOTS_DIR)/common_sample_combined_2022.tex $(PLOTS_DIR)/common_sample_spring_summer_2022.tex $(PLOTS_DIR)/accuracy_scatter_combined_2022.pdf $(PLOTS_DIR)/accuracy_scatter_seasonal_2022.pdf &: $(TASK_DIR)/4_accuracy_main_2022.py $(PREDS_DIR)/oracle_ceiling_2022.csv $(PREDS_DIR)/adc_aefn2_masked_preds.parquet
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# ── Step 5: Table 5 (mun-level survey improvement, ex-ante ag-land weights) ─
$(TABLES_DIR)/accuracy_mun_level_2022.tex: $(TASK_DIR)/5_mun_survey_improvement.py \
		$(PREDS_DIR)/mun_aefn2_masked_gb_kfold_preds.parquet \
		$(PREDS_DIR)/mun_aef_hist_gb_kfold_preds.parquet
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# ── Step 6: Table 3 (census thought experiment, all models) ──────────────
# The Agg-NN (census-trained) panel consumes the retrain saved by
# train/05_train_agg_nn/census_thought_experiment.py.
$(TABLES_DIR)/census_thought_experiment_2022.tex: $(TASK_DIR)/6_census_thought_all_models.py $(PREDS_DIR)/adc_aggnn_census_trained_preds.parquet
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# ── Step 7: Table 4 (accuracy by ADC characteristics) ────────────────────
$(TABLES_DIR)/accuracy_profile_by_adc_chars.tex: $(TASK_DIR)/7_accuracy_profile_by_adc_chars.py
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# ── Step 8: Table 6 (CIMMYT farmer-trial external validation) ────────────
$(TABLES_DIR)/accuracy_cimmyt_profile.tex: $(TASK_DIR)/8_accuracy_cimmyt_profile.py
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# ── Steps 9-11: ex-ante trust chain ──────────────────────────────────────
# 9 builds the per-mun feature csv (extracted from the retired exante
# notebook); 10 adds the unsupervised trust_index + refreshes within_r2_mun
# from the deployed Shrink predictions; 11 owns the cross-model RF csv.
$(EXTRAS_DIR)/exante_trust_index.csv: $(TASK_DIR)/9_exante_trust_features.py $(TASK_DIR)/10_exante_trust_composite.py $(PREDS_DIR)/adc_aef_hist_ens_eval.parquet
	cd $(DATA_DIR) && $(ML_ENV) python3 $(TASK_DIR)/9_exante_trust_features.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $(TASK_DIR)/10_exante_trust_composite.py

$(EXTRAS_DIR)/exante_trust_across_models.csv: $(TASK_DIR)/11_exante_trust_across_models.py $(EXTRAS_DIR)/exante_trust_index.csv
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# ── Steps 12-13: Figures 6 and 7 ─────────────────────────────────────────
$(EXTRAS_DIR)/fig_representativeness_targeting.pdf: $(TASK_DIR)/12_fig_representativeness_targeting.py $(EXTRAS_DIR)/exante_trust_index.csv
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

$(EXTRAS_DIR)/fig_ranking_inversion.pdf: $(TASK_DIR)/13_fig_ranking_inversion.py $(EXTRAS_DIR)/exante_trust_across_models.csv
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# ── Steps 14-15: Figure 2 (methodology diagram) ──────────────────────────
# 14 fetches GEE chips (network; run once), 15 draws the diagram.
.PHONY: fetch_fig2_chips
fetch_fig2_chips: $(TASK_DIR)/14_fetch_mun_chips_fig.py
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

$(PLOTS_DIR)/fig_methodology_diagram.pdf: $(TASK_DIR)/15_fig_methodology_diagram.py
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# ── Prose-stat utilities (numbers cited in text, no .tex output) ─────────
.PHONY: robustness_stats
robustness_stats: $(TASK_DIR)/robustness_lambda_ci_qbin.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

.PHONY: sample_accounting
sample_accounting: $(TASK_DIR)/sample_accounting_and_lambda.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# Irrigation-projection bound + point estimate for rho/lambda (Sec 3.6)
.PHONY: rho_irrigation_bound
rho_irrigation_bound: $(TASK_DIR)/rho_irrigation_bound.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<
