### 03_train_rf_gb.mk
# Train RF/GB yield models on AEF embeddings (primary method)
SHELL := /bin/bash   # 'source' for conda activation needs bash, not dash

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/train/03_train_rf_gb

PREDS_DIR := $(DATA_DIR)/Data/predictions
TABLES_DIR := $(DATA_DIR)/tables

.PHONY: train_rf_gb
train_rf_gb: \
	$(PREDS_DIR)/adc_alpha_earth_preds_maize.parquet \
	$(PREDS_DIR)/adc_aef_hist_ens_eval.parquet \
	$(PREDS_DIR)/mun_aef_hist_gb_kfold_preds.parquet

$(PREDS_DIR)/adc_alpha_earth_preds_maize.parquet: $(TASK_DIR)/rf_yield_prediction.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# ── AEF Hist Ensemble (paper's primary model) ────────────
# Writes adc_aef_hist_ens_preds.parquet + adc_aef_hist_ens_eval.parquet; the
# eval file carries yields + season-matched corrections and is the backbone of
# 4_accuracy_main_2022.py (Tables 2/A1/A2).
$(PREDS_DIR)/adc_aef_hist_ens_eval.parquet: $(TASK_DIR)/gb_aef_hist_ensemble.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# Robustness variants: demeaned features (dm) and quantile-bin-only (qbin)
.PHONY: train_aef_hist_ens_dm train_aef_hist_ens_qbin
train_aef_hist_ens_dm: $(TASK_DIR)/gb_aef_hist_ensemble_dm.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<
train_aef_hist_ens_qbin: $(TASK_DIR)/gb_aef_hist_ensemble_qbin.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# Muni-level random muni-year 5-fold CV for AEF Hist (feeds the survey-
# improvement table in analysis/02_accuracy_maize/5_mun_survey_improvement.py)
$(PREDS_DIR)/mun_aef_hist_gb_kfold_preds.parquet: $(TASK_DIR)/mun_cv_aef_hist.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# ── East Africa extension — NOT in the current paper (cut 2026-06-10) ──
# Trains the AEF Hist Ensemble on HarvestStat Africa admin-2 maize yields
# (LOYO CV inside the script) and evaluates on GROW-Africa plot-level yields.
.PHONY: accuracy_africa
accuracy_africa: $(TABLES_DIR)/accuracy_africa_profile.tex

$(TABLES_DIR)/accuracy_africa_profile.tex: $(TASK_DIR)/accuracy_africa_profile.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# ── Histogram RF/GB ──────────────────────────────────────
HIST_PRED_SCRIPT := $(TASK_DIR)/rf_histogram_prediction.py

.PHONY: train_hist_rf
train_hist_rf: $(HIST_PRED_SCRIPT)
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# ── 3-period histogram GB ────────────────────────────────
.PHONY: train_3period_gb
train_3period_gb: $(TASK_DIR)/gb_3period_prediction.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# Fall-winter-trained ensemble robustness (Sec 5.5 numbers)
.PHONY: train_oi_ensemble
train_oi_ensemble: $(TASK_DIR)/gb_aef_hist_ensemble_oi_trained.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<
