### pipeline.mk
# Master Makefile — Predicting Yields at Scale Using Remote Sensing
# Authors: James Sayre, Joel Ferguson, Kangogo Sogomo
#
# Usage:
#   make -f Code/pipeline.mk build        # Data cleaning & assembly (some steps manual)
#   make -f Code/pipeline.mk train        # Model training & prediction
#   make -f Code/pipeline.mk analysis     # Corrections → evaluation → copy to Overleaf
#   make -f Code/pipeline.mk all          # Full pipeline (stages 3–5)
#
# Run from code repo root: ~/Dropbox/Github/predicting_yields_rs/

SHELL         := /bin/bash
PROJ_DIR      := $(shell cd $(dir $(lastword $(MAKEFILE_LIST)))/.. && pwd)
CODE_DIR      := $(PROJ_DIR)/Code
DATA_DIR      := $(HOME)/Dropbox/Projects/Maize_prediction

# ── Conda activation prefix ─────────────────────────────
CONDA_ACTIVATE  = source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate
MPC_ENV         = $(CONDA_ACTIVATE) mpc_env &&
ML_ENV          = $(CONDA_ACTIVATE) ML_env &&

# ── Notebook execution ───────────────────────────────────
NB_EXEC       = jupyter nbconvert --execute --inplace --ExecutePreprocessor.timeout=3600

# ── Overleaf directory ───────────────────────────────────
OVERLEAF_DIR  := $(HOME)/Dropbox/Overleaf/Predicting Yields at Scale using RS

# ═══════════════════════════════════════════════════════════
# Stage: Build (data cleaning & assembly)
# Most steps require manual execution (Selenium, Earth Engine)
# ═══════════════════════════════════════════════════════════

.PHONY: build build_01 build_02 build_03 build_04 build_05 build_06

build_01:
	@echo "MANUAL: Run Code/build/01_scrape_siap/ (requires Selenium + Chrome)"

build_02: build_01
	$(MAKE) -f $(CODE_DIR)/build/02_clean_siap_monthly/02_clean_siap_monthly.mk \
		PROJ_DIR=$(PROJ_DIR) DATA_DIR="$(DATA_DIR)" MPC_ENV="$(MPC_ENV)" NB_EXEC="$(NB_EXEC)"

build_03:
	$(MAKE) -f $(CODE_DIR)/build/03_assemble_boundaries/03_assemble_boundaries.mk \
		PROJ_DIR=$(PROJ_DIR) DATA_DIR="$(DATA_DIR)" MPC_ENV="$(MPC_ENV)" NB_EXEC="$(NB_EXEC)"

build_04: build_03
	$(MAKE) -f $(CODE_DIR)/build/04_build_correspondence/04_build_correspondence.mk \
		PROJ_DIR=$(PROJ_DIR) DATA_DIR="$(DATA_DIR)" MPC_ENV="$(MPC_ENV)" NB_EXEC="$(NB_EXEC)"

build_05: build_02
	$(MAKE) -f $(CODE_DIR)/build/05_phenology/05_phenology.mk \
		PROJ_DIR=$(PROJ_DIR) DATA_DIR="$(DATA_DIR)" MPC_ENV="$(MPC_ENV)" NB_EXEC="$(NB_EXEC)"

build_06:
	$(MAKE) -f $(CODE_DIR)/build/06_clean_validation/06_clean_validation.mk \
		PROJ_DIR=$(PROJ_DIR) DATA_DIR="$(DATA_DIR)" MPC_ENV="$(MPC_ENV)" NB_EXEC="$(NB_EXEC)"

build: build_02 build_03 build_04 build_05 build_06

# ═══════════════════════════════════════════════════════════
# Stage: Train (model training & prediction)
# Extraction steps (01, 02) require Earth Engine → manual
# ═══════════════════════════════════════════════════════════

.PHONY: train train_01 train_02 train_03 train_04 train_04_v2 train_04_temporal train_05

train_01:
	@echo "MANUAL: Run Code/train/01_extract_embeddings/ (requires Earth Engine auth)"

train_02:
	@echo "MANUAL: Run Code/train/02_extract_ndvi_histograms/ (requires Earth Engine auth)"
	@echo "  Original:  ls_ndvi_hists.py 0.2 1.0 0.0 12.0 0.0 0.6 32 max"
	@echo "  Monthly:   ls_monthly_hists.py 32 8"

train_03:
	$(MAKE) -f $(CODE_DIR)/train/03_train_rf_gb/03_train_rf_gb.mk \
		PROJ_DIR=$(PROJ_DIR) DATA_DIR="$(DATA_DIR)" ML_ENV="$(ML_ENV)"

train_04:
	$(MAKE) -f $(CODE_DIR)/train/04_train_cnn/04_train_cnn.mk train_cnn \
		PROJ_DIR=$(PROJ_DIR) DATA_DIR="$(DATA_DIR)" ML_ENV="$(ML_ENV)"

train_04_v2:
	$(MAKE) -f $(CODE_DIR)/train/04_train_cnn/04_train_cnn.mk train_cnn_v2 \
		PROJ_DIR=$(PROJ_DIR) DATA_DIR="$(DATA_DIR)" ML_ENV="$(ML_ENV)"

train_04_temporal:
	$(MAKE) -f $(CODE_DIR)/train/04_train_cnn/04_train_cnn.mk train_temporal_cnn \
		PROJ_DIR=$(PROJ_DIR) DATA_DIR="$(DATA_DIR)" ML_ENV="$(ML_ENV)"

train_05:
	$(MAKE) -f $(CODE_DIR)/train/05_train_agg_nn/05_train_agg_nn.mk \
		PROJ_DIR=$(PROJ_DIR) DATA_DIR="$(DATA_DIR)" ML_ENV="$(ML_ENV)"

train: train_03 train_05

# ═══════════════════════════════════════════════════════════
# Stage: Analysis (corrections → evaluation → paper outputs)
# ═══════════════════════════════════════════════════════════

.PHONY: analysis analysis_01 analysis_02 analysis_03 analysis_04

analysis_01: train_03
	$(MAKE) -f $(CODE_DIR)/analysis/01_corrections/01_corrections.mk \
		PROJ_DIR=$(PROJ_DIR) DATA_DIR="$(DATA_DIR)" MPC_ENV="$(MPC_ENV)" ML_ENV="$(ML_ENV)" NB_EXEC="$(NB_EXEC)"

analysis_02: analysis_01
	$(MAKE) -f $(CODE_DIR)/analysis/02_accuracy_maize/02_accuracy_maize.mk \
		PROJ_DIR=$(PROJ_DIR) DATA_DIR="$(DATA_DIR)" MPC_ENV="$(MPC_ENV)" NB_EXEC="$(NB_EXEC)"

analysis_03: analysis_01
	$(MAKE) -f $(CODE_DIR)/analysis/03_accuracy_other_crops/03_accuracy_other_crops.mk \
		PROJ_DIR=$(PROJ_DIR) DATA_DIR="$(DATA_DIR)" MPC_ENV="$(MPC_ENV)" NB_EXEC="$(NB_EXEC)"

analysis_04: analysis_02 analysis_03
	$(MAKE) -f $(CODE_DIR)/analysis/04_copy_to_overleaf/04_copy_to_overleaf.mk \
		PROJ_DIR=$(PROJ_DIR) DATA_DIR="$(DATA_DIR)" OVERLEAF_DIR="$(OVERLEAF_DIR)"

analysis: analysis_04

# ═══════════════════════════════════════════════════════════
# Full pipeline (stages 3–5)
# ═══════════════════════════════════════════════════════════

.PHONY: all
all: train analysis
