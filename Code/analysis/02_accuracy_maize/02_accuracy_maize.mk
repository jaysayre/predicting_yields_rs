### 02_accuracy_maize.mk
# Maize accuracy evaluation → Tables 1–3, Figures 4 & 7
#
# Tables produced here and \input{} by the paper:
#   1_accuracy_metrics_2022.ipynb  -> accuracy_{combined,fall_winter,spring_summer,mun_level}_2022.tex
#   accuracy_profile_by_adc_chars.py -> accuracy_profile_by_adc_chars.tex   (writes to tables/ + Overleaf direct)
#   accuracy_cimmyt_profile.py       -> accuracy_cimmyt_profile.tex          (writes to tables/ + Overleaf direct)

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/analysis/02_accuracy_maize

PLOTS_DIR  := $(DATA_DIR)/plots
TABLES_DIR := $(DATA_DIR)/tables
PREDS_DIR  := $(DATA_DIR)/Data/predictions

EXTRAS_DIR := $(DATA_DIR)/plots/coauthor_extras_paper

.PHONY: accuracy_maize
accuracy_maize: \
	$(PLOTS_DIR)/accuracy_combined_2022.tex \
	$(PLOTS_DIR)/common_sample_combined_2022.tex \
	$(PLOTS_DIR)/accuracy_scatter_combined_2022.pdf \
	$(TABLES_DIR)/accuracy_profile_by_adc_chars.tex \
	$(TABLES_DIR)/accuracy_cimmyt_profile.tex \
	$(TABLES_DIR)/accuracy_shrink_all_models_2022.tex \
	$(TABLES_DIR)/census_thought_experiment_2022.tex \
	$(TABLES_DIR)/accuracy_mun_level_2022.tex \
	$(EXTRAS_DIR)/fig_representativeness_targeting.png \
	$(EXTRAS_DIR)/fig_ranking_inversion.png

# ── NDVI (masked) baseline: cropland-masked aefn2 features ──────────────
# Muni-level feature cache + random muni-year 5-fold CV predictions. Also
# writes muni_aefn2_masked.parquet, which train/05_train_agg_nn/
# train_holdout_validation_models.py ('masked' arg) consumes.
$(PREDS_DIR)/mun_aefn2_masked_gb_kfold_preds.parquet: $(TASK_DIR)/masked_muni_cv.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# Muni-trained model scored at the ADC level -> adc_aefn2_masked_preds.parquet,
# the NDVI (masked) rows of the main accuracy tables.
$(PREDS_DIR)/adc_aefn2_masked_preds.parquet: $(TASK_DIR)/partial_masked_mun_train_adc_eval.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# Survey-improvement table with EX-ANTE agricultural-land aggregation weights
# (Table \ref{tab:mun_agg_results}); supersedes the notebook's census-weighted version.
$(TABLES_DIR)/accuracy_mun_level_2022.tex: $(TASK_DIR)/mun_survey_improvement.py \
		$(PREDS_DIR)/mun_aefn2_masked_gb_kfold_preds.parquet \
		$(PREDS_DIR)/mun_aef_hist_gb_kfold_preds.parquet
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# Oracle ADC-trained ceiling row (non-deployable); feeds the benchmark sections
# of the combined/spring-summer accuracy tables built by accuracy_main_2022.py.
$(PREDS_DIR)/oracle_ceiling_2022.csv: $(TASK_DIR)/oracle_adc_ceiling.py
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# Census thought-experiment table for ALL models (Table \ref{tab:census_thought}).
# Correction panels are fully reproducible here; the Agg-NN (Census-trained) panel
# value comes from the GPU retrain in train/05_train_agg_nn/census_thought_experiment.py.
$(TABLES_DIR)/census_thought_experiment_2022.tex: $(TASK_DIR)/census_thought_all_models.py
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# Within-mun shrinkage applied to ALL models (Shrink rows in the main tables)
$(TABLES_DIR)/accuracy_shrink_all_models_2022.tex: $(TASK_DIR)/within_mun_shrink_all_models.py
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# Unsupervised ex-ante trust composite (NO ground truth): writes the trust_index
# column into exante_trust_index.csv + the cross-model validation csv. Both ex-ante
# figures depend on it.
$(EXTRAS_DIR)/exante_trust_across_models.csv: $(TASK_DIR)/exante_trust_composite.py
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# Ex-ante targeting figure (representativeness gradient)
$(EXTRAS_DIR)/fig_representativeness_targeting.png: $(TASK_DIR)/fig_representativeness_targeting.py $(EXTRAS_DIR)/exante_trust_across_models.csv
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# Model-ranking inversion figure (municipality-level vs farm-level rank)
$(EXTRAS_DIR)/fig_ranking_inversion.png: $(TASK_DIR)/fig_ranking_inversion.py $(EXTRAS_DIR)/exante_trust_across_models.csv
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# Main season accuracy tables (combined/spring/fall) -- one reproducible generator
# with ex-ante ag-land corrections + per-model shrink rows.
$(PLOTS_DIR)/accuracy_combined_2022.tex $(PLOTS_DIR)/accuracy_spring_summer_2022.tex $(PLOTS_DIR)/accuracy_fall_winter_2022.tex &: $(TASK_DIR)/accuracy_main_2022.py $(PREDS_DIR)/oracle_ceiling_2022.csv
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# Scatter figures still come from the metrics notebook.
$(PLOTS_DIR)/accuracy_scatter_combined_2022.pdf: $(TASK_DIR)/1_accuracy_metrics_2022.ipynb
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $<

# ADC-characteristics accuracy profile (Table \ref{tab:accuracy_profile})
$(TABLES_DIR)/accuracy_profile_by_adc_chars.tex: $(TASK_DIR)/accuracy_profile_by_adc_chars.py
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# CIMMYT farmer-trial external validation (Table \ref{tab:cimmyt_profile})
$(TABLES_DIR)/accuracy_cimmyt_profile.tex: $(TASK_DIR)/accuracy_cimmyt_profile.py
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<
