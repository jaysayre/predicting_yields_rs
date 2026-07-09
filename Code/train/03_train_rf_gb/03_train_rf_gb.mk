### 03_train_rf_gb.mk
# Train RF/GB yield models on AEF embeddings (primary method)

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/train/03_train_rf_gb

PREDS_DIR := $(DATA_DIR)/Data/predictions
TABLES_DIR := $(DATA_DIR)/tables

.PHONY: train_rf_gb
train_rf_gb: $(PREDS_DIR)/adc_alpha_earth_preds_maize.parquet

$(PREDS_DIR)/adc_alpha_earth_preds_maize.parquet: $(TASK_DIR)/rf_yield_prediction.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# ── East Africa extension (Table \ref{tab:africa_profile}) ──
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
