### 04_train_cnn.mk
# Train CNN on NDVI histograms (baseline + improved + temporal)

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/train/04_train_cnn
HIST_DIR := $(CODE_DIR)/train/02_extract_ndvi_histograms

PREDS_DIR := $(DATA_DIR)/Data/predictions

# Year range for leave-one-year-out CV
YEARS := $(shell seq 2003 2022)

# ── Baseline CNN (original Keras, 2-period histograms) ────

.PHONY: train_cnn
train_cnn: $(PREDS_DIR)/adcs_yield_preds.csv

# Step 1: Preprocess histograms → pickles
$(PREDS_DIR)/.histograms_cleaned: $(HIST_DIR)/clean_histograms_redux.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<
	touch $@

# Step 2: Train CNN (original TF/Keras)
$(PREDS_DIR)/adcs_yield_preds.csv: $(PREDS_DIR)/.histograms_cleaned $(TASK_DIR)/ndvi_nn_redux.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $(TASK_DIR)/ndvi_nn_redux.py

# ── Improved CNN v2 (PyTorch, 2-period histograms) ────────

.PHONY: train_cnn_v2
train_cnn_v2: $(PREDS_DIR)/.cnn_v2_done

$(PREDS_DIR)/.cnn_v2_done: $(PREDS_DIR)/.histograms_cleaned $(TASK_DIR)/ndvi_nn_v2.py
	@for yr in $(YEARS); do \
		echo "Training CNN v2 holdout year $$yr"; \
		cd $(DATA_DIR) && $(ML_ENV) python3 $(TASK_DIR)/ndvi_nn_v2.py $$yr; \
	done
	touch $@

# ── Monthly histogram preprocessing ──────────────────────

.PHONY: clean_monthly_hists
clean_monthly_hists: $(DATA_DIR)/Data/monthly_hists_clean/monthly_hists_all.pkl

$(DATA_DIR)/Data/monthly_hists_clean/monthly_hists_all.pkl: $(HIST_DIR)/clean_monthly_histograms.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $< \
		Data/muni_monthly_hists_32bins_8mo \
		Data/monthly_hists_clean \
		32 8

# ── Temporal CNN (PyTorch, monthly histograms) ────────────

.PHONY: train_temporal_cnn
train_temporal_cnn: $(PREDS_DIR)/.temporal_cnn_done

$(PREDS_DIR)/.temporal_cnn_done: $(DATA_DIR)/Data/monthly_hists_clean/monthly_hists_all.pkl $(TASK_DIR)/ndvi_nn_temporal.py
	@for yr in $(YEARS); do \
		echo "Training temporal CNN holdout year $$yr"; \
		cd $(DATA_DIR) && $(ML_ENV) python3 $(TASK_DIR)/ndvi_nn_temporal.py $$yr; \
	done
	touch $@
