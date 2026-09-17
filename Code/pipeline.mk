### pipeline.mk
# Master Makefile — Predicting Maize Yields at Large Scale Using Remote Sensing
# Authors: James Sayre, Joel Ferguson, Kangogo Sogomo
#
# Usage (from the code repo root):
#   make -f Code/pipeline.mk build        # stage 1: data cleaning & boundary assembly
#   make -f Code/pipeline.mk train        # stage 2: model training & prediction
#   make -f Code/pipeline.mk analysis     # stage 3: evaluation -> tables/figures -> copy to Overleaf
#   make -f Code/pipeline.mk all          # train + analysis
#   make -f Code/pipeline.mk -n all       # dry run: list what would be (re)built
#
# Each numbered task folder has exactly one <folder>.mk with its targets and a
# step-numbered script list; this file only chains them. Stages that need
# interactive credentials (Selenium scrape, Earth Engine exports) print their
# instructions instead of running; everything downstream of the pulled exports
# is make-driven.

SHELL         := /bin/bash
# GNU make cannot handle spaces in prerequisite paths, so prefer the space-free
# ~/Dropbox symlink over the physical path (laptop physical path contains
# "CalAg Dropbox/Jay Sayre"). Falls back to the resolved path elsewhere.
ifneq ($(wildcard $(HOME)/Dropbox/Github/predicting_yields_rs/Code/pipeline.mk),)
  PROJ_DIR    := $(HOME)/Dropbox/Github/predicting_yields_rs
else
  PROJ_DIR    := $(shell cd $(dir $(lastword $(MAKEFILE_LIST)))/.. && pwd)
endif
CODE_DIR      := $(PROJ_DIR)/Code
DATA_DIR      := $(HOME)/Dropbox/Projects/Maize_prediction

# ── Conda activation prefix ─────────────────────────────
# Auto-detect machine: server (Deloach, /usr/local/anaconda3, envs mpc_env/ML_env)
# vs laptop (~/miniforge3, envs geo_env/ml_cuda). Override with e.g.
#   make -f Code/pipeline.mk analysis GEO_ENV_NAME=myenv ML_ENV_NAME=myenv
# Environment specs: Code/envs/*.yml
ifneq ($(wildcard /usr/local/anaconda3/etc/profile.d/conda.sh),)
  CONDA_SH     ?= /usr/local/anaconda3/etc/profile.d/conda.sh
  GEO_ENV_NAME ?= mpc_env
  ML_ENV_NAME  ?= ML_env
else
  CONDA_SH     ?= $(HOME)/miniforge3/etc/profile.d/conda.sh
  GEO_ENV_NAME ?= geo_env
  ML_ENV_NAME  ?= ml_cuda
endif
CONDA_ACTIVATE  = source $(CONDA_SH) && conda activate
MPC_ENV         = $(CONDA_ACTIVATE) $(GEO_ENV_NAME) &&
ML_ENV          = $(CONDA_ACTIVATE) $(ML_ENV_NAME) &&

# ── Notebook execution ───────────────────────────────────
NB_EXEC       = jupyter nbconvert --execute --inplace --ExecutePreprocessor.timeout=3600

# ── Overleaf directory ───────────────────────────────────
OVERLEAF_DIR  := $(HOME)/Dropbox/Overleaf/Predicting Yields at Scale using RS

# Common variables handed to every task makefile
SUBMAKE_VARS  = PROJ_DIR=$(PROJ_DIR) DATA_DIR="$(DATA_DIR)" MPC_ENV="$(MPC_ENV)" ML_ENV="$(ML_ENV)" NB_EXEC="$(NB_EXEC)"

# ═══════════════════════════════════════════════════════════
# Stage 1: Build (data cleaning & assembly)
# ═══════════════════════════════════════════════════════════

.PHONY: build build_01 build_02 build_03 build_04 build_05 build_06

build_01:   # SIAP monthly production scrape (Selenium + Chrome)
	$(MAKE) -f $(CODE_DIR)/build/01_scrape_siap/01_scrape_siap.mk $(SUBMAKE_VARS)

build_02:   # clean the scraped SIAP monthly panel + harvest-calendar figure
	$(MAKE) -f $(CODE_DIR)/build/02_clean_siap_monthly/02_clean_siap_monthly.mk $(SUBMAKE_VARS)

build_03:   # ADC shapefiles, 2022 geostatistical frame, agland masks
	$(MAKE) -f $(CODE_DIR)/build/03_assemble_boundaries/03_assemble_boundaries.mk $(SUBMAKE_VARS)

build_04: build_03   # 2007 <-> 2016 ADC correspondence
	$(MAKE) -f $(CODE_DIR)/build/04_build_correspondence/04_build_correspondence.mk $(SUBMAKE_VARS)

build_05: build_02   # crop phenology: NDVI peaks vs SIAP planting months
	$(MAKE) -f $(CODE_DIR)/build/05_phenology/05_phenology.mk $(SUBMAKE_VARS)

build_06: build_05   # CIMMYT farmer-plot validation data + EE geometries
	$(MAKE) -f $(CODE_DIR)/build/06_clean_validation/06_clean_validation.mk $(SUBMAKE_VARS)

build: build_02 build_04 build_05 build_06

# ═══════════════════════════════════════════════════════════
# Stage 2: Train (feature extraction, model training & prediction)
# Steps 01-02 submit Earth Engine exports (manual); their consolidation
# targets run once the exports have been pulled from Drive.
# ═══════════════════════════════════════════════════════════

.PHONY: train train_01 train_02 train_03 train_04

train_01:   # AEF embeddings: EE exports (manual) + CSV -> parquet consolidation
	$(MAKE) -f $(CODE_DIR)/train/01_extract_embeddings/01_extract_embeddings.mk $(SUBMAKE_VARS)

train_02:   # Landsat NDVI cropland features for the NDVI (masked) baseline
	$(MAKE) -f $(CODE_DIR)/train/02_extract_ndvi_histograms/02_extract_ndvi_histograms.mk $(SUBMAKE_VARS)

train_03:   # AEF mean RF, AEF Hist + AEF Hist Ensemble GB, muni CV, other crops
	$(MAKE) -f $(CODE_DIR)/train/03_train_rf_gb/03_train_rf_gb.mk $(SUBMAKE_VARS)

train_04: train_03   # Agg-NN, holdout retrains, validation table, census-trained retrain
	$(MAKE) -f $(CODE_DIR)/train/04_train_agg_nn/04_train_agg_nn.mk $(SUBMAKE_VARS)

train: train_03 train_04

# ═══════════════════════════════════════════════════════════
# Stage 3: Analysis (corrections -> evaluation -> paper outputs)
# ═══════════════════════════════════════════════════════════

.PHONY: analysis analysis_01 analysis_02 analysis_03 analysis_04

analysis_01: train_03   # Figure 1 maps/hexbin, CA22 prediction maps
	$(MAKE) -f $(CODE_DIR)/analysis/01_corrections/01_corrections.mk $(SUBMAKE_VARS)

analysis_02: analysis_01   # every maize table and figure
	$(MAKE) -f $(CODE_DIR)/analysis/02_accuracy_maize/02_accuracy_maize.mk $(SUBMAKE_VARS)

analysis_03: analysis_01   # other-crop tables
	$(MAKE) -f $(CODE_DIR)/analysis/03_accuracy_other_crops/03_accuracy_other_crops.mk $(SUBMAKE_VARS)

analysis_04: analysis_02 analysis_03   # copy every table and figure into Overleaf
	$(MAKE) -f $(CODE_DIR)/analysis/04_copy_to_overleaf/04_copy_to_overleaf.mk \
		PROJ_DIR=$(PROJ_DIR) DATA_DIR="$(DATA_DIR)" OVERLEAF_DIR="$(OVERLEAF_DIR)"

analysis: analysis_04

# ═══════════════════════════════════════════════════════════
# Full pipeline (stages 2-3; stage 1 and the EE exports are run once by hand)
# ═══════════════════════════════════════════════════════════

.PHONY: all
all: train analysis
