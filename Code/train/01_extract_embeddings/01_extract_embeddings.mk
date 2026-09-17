### 01_extract_embeddings.mk
# AlphaEarth Foundations (AEF) embedding features from Google Earth Engine.
# The EE exports are MANUAL (interactive EE auth; hours of batch tasks); the
# CSV -> parquet consolidation is make-driven. Scripts run in step order:
#   1_ee_alpha_earth_download.py         MANUAL: per-dimension MEANS (MODE=mexico_mun, then mexico_adc)
#                                        -> Drive alpha_earth_agmask/, alpha_earth_adcs/
#   2_ee_alpha_earth_histogram.py        MANUAL: ADC percentiles+SD, 2022 proof of concept (Oaxaca)
#   3_ee_alpha_earth_histogram_mun.py    MANUAL: municipality percentiles+SD -> Drive alpha_earth_mun_hist/
#   4_ee_alpha_earth_histogram_allyears.py MANUAL: ADC + municipality percentiles, 2017-21 & 2023-24
#   5_ee_alpha_earth_histogram_2022.py   MANUAL: ADC percentiles, 2022, remaining states
#   6_ee_alpha_earth_binned_hist.py      MANUAL: 64 x 8 fixed-width bin shares on [-0.8, 0.8], mun + ADC
#   7_consolidate_mean_parquet.py        means CSVs   -> alpha_earth_mex_muns.parquet, alpha_earth_mex_adcs.parquet
#   8_consolidate_hist_parquet.py        ADC pct CSVs -> alpha_earth_mex_adcs_hist.parquet
#   9_consolidate_hist_parquet_mun.py    mun pct CSVs -> alpha_earth_mex_mun_hist.parquet
#   10_consolidate_binned_parquet.py     bin CSVs     -> alpha_earth_mex_{mun,adcs}_binned_hist.parquet  (the "AEF Hist" inputs)
#   11_ee_alpha_earth_cimmyt.py          MANUAL: CIMMYT plot means        (needs build/06 step 2 geometries)
#   12_ee_alpha_earth_cimmyt_hist.py     MANUAL: CIMMYT plot percentiles
#   13_ee_alpha_earth_cimmyt_binned_hist.py MANUAL: CIMMYT plot bin shares
#   14_consolidate_aef_cimmyt_all.py     CIMMYT CSVs  -> alpha_earth_cimmyt_plot{,_hist,_binned_hist}.parquet
# Robustness (equal-mass quantile bins; prose numbers only): ee_aef_quantile_edges.py,
#   ee_alpha_earth_quantile_hist.py (MANUAL), consolidate_qbin_parquet.py (make consolidate_qbin).
SHELL := /bin/bash   # 'source' for conda activation needs bash, not dash

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/train/01_extract_embeddings
AEF_DIR  := $(DATA_DIR)/Data/alpha_earth

# Where the Drive exports were pulled to (override on the command line if elsewhere)
MUN_MEAN_CSVS ?= $(AEF_DIR)/mun_mean_csvs
ADC_MEAN_CSVS ?= $(AEF_DIR)/adcs_mean_csvs
MUN_HIST_CSVS ?= $(AEF_DIR)/mun_hist_csvs
ADC_HIST_CSVS ?= $(AEF_DIR)/adcs_hist_csvs
MUN_BIN_CSVS  ?= $(AEF_DIR)/mun_binned_hist_csvs
ADC_BIN_CSVS  ?= $(AEF_DIR)/adcs_binned_hist_csvs
CIMMYT_CSVS   ?= $(AEF_DIR)/cimmyt_csvs

AEF_PARQUETS := \
	$(AEF_DIR)/alpha_earth_mex_muns.parquet \
	$(AEF_DIR)/alpha_earth_mex_adcs.parquet \
	$(AEF_DIR)/alpha_earth_mex_mun_hist.parquet \
	$(AEF_DIR)/alpha_earth_mex_adcs_hist.parquet \
	$(AEF_DIR)/alpha_earth_mex_mun_binned_hist.parquet \
	$(AEF_DIR)/alpha_earth_mex_adcs_binned_hist.parquet \
	$(AEF_DIR)/alpha_earth_cimmyt_plot.parquet \
	$(AEF_DIR)/alpha_earth_cimmyt_plot_hist.parquet \
	$(AEF_DIR)/alpha_earth_cimmyt_plot_binned_hist.parquet

.PHONY: extract_embeddings consolidate_embeddings consolidate_qbin
extract_embeddings:
	@echo "MANUAL STEPS (Earth Engine auth + ML env), run in order from $(TASK_DIR):"
	@echo "  1_ee_alpha_earth_download.py (MODE=mexico_mun, then mexico_adc)   -> Drive alpha_earth_agmask/, alpha_earth_adcs/"
	@echo "  2_ .. 5_ee_alpha_earth_histogram*.py                               -> Drive alpha_earth_mun_hist/, alpha_earth_adcs_hist/"
	@echo "  6_ee_alpha_earth_binned_hist.py                                    -> Drive alpha_earth_mun_binned_hist/, alpha_earth_adcs_binned_hist/"
	@echo "  11_ .. 13_ee_alpha_earth_cimmyt*.py                                -> Drive ae_cimmyt*/"
	@echo "Pull each Drive folder into the matching *_csvs directory under $(AEF_DIR), then:"
	@echo "  make -f $(CODE_DIR)/pipeline.mk train_01   (re-run; consolidates CSVs -> parquet)"
	@$(MAKE) -f $(TASK_DIR)/01_extract_embeddings.mk consolidate_embeddings \
		PROJ_DIR=$(PROJ_DIR) DATA_DIR="$(DATA_DIR)" MPC_ENV="$(MPC_ENV)" || true

consolidate_embeddings: $(AEF_PARQUETS)

# Step 7 — means
$(AEF_DIR)/alpha_earth_mex_muns.parquet: $(TASK_DIR)/7_consolidate_mean_parquet.py | $(MUN_MEAN_CSVS)
	cd $(DATA_DIR) && $(MPC_ENV) python3 $< --level mun --csv_dir $(MUN_MEAN_CSVS)
$(AEF_DIR)/alpha_earth_mex_adcs.parquet: $(TASK_DIR)/7_consolidate_mean_parquet.py | $(ADC_MEAN_CSVS)
	cd $(DATA_DIR) && $(MPC_ENV) python3 $< --level adc --csv_dir $(ADC_MEAN_CSVS)

# Steps 8-9 — percentiles + SD
$(AEF_DIR)/alpha_earth_mex_adcs_hist.parquet: $(TASK_DIR)/8_consolidate_hist_parquet.py | $(ADC_HIST_CSVS)
	cd $(DATA_DIR) && $(MPC_ENV) python3 $< --csv_dir $(ADC_HIST_CSVS)
$(AEF_DIR)/alpha_earth_mex_mun_hist.parquet: $(TASK_DIR)/9_consolidate_hist_parquet_mun.py | $(MUN_HIST_CSVS)
	cd $(DATA_DIR) && $(MPC_ENV) python3 $< --csv_dir $(MUN_HIST_CSVS)

# Step 10 — fixed-width bin shares (AEF Hist)
$(AEF_DIR)/alpha_earth_mex_mun_binned_hist.parquet: $(TASK_DIR)/10_consolidate_binned_parquet.py | $(MUN_BIN_CSVS)
	cd $(DATA_DIR) && $(MPC_ENV) python3 $< --level mun --csv_dir $(MUN_BIN_CSVS)
$(AEF_DIR)/alpha_earth_mex_adcs_binned_hist.parquet: $(TASK_DIR)/10_consolidate_binned_parquet.py | $(ADC_BIN_CSVS)
	cd $(DATA_DIR) && $(MPC_ENV) python3 $< --level adc --csv_dir $(ADC_BIN_CSVS)

# Step 14 — CIMMYT plots (one CSV folder per feature type under $(CIMMYT_CSVS))
$(AEF_DIR)/alpha_earth_cimmyt_plot.parquet: $(TASK_DIR)/14_consolidate_aef_cimmyt_all.py | $(CIMMYT_CSVS)
	cd $(DATA_DIR) && $(MPC_ENV) python3 $< --type mean --csv_dir $(CIMMYT_CSVS)/mean
$(AEF_DIR)/alpha_earth_cimmyt_plot_hist.parquet: $(TASK_DIR)/14_consolidate_aef_cimmyt_all.py | $(CIMMYT_CSVS)
	cd $(DATA_DIR) && $(MPC_ENV) python3 $< --type hist --csv_dir $(CIMMYT_CSVS)/hist
$(AEF_DIR)/alpha_earth_cimmyt_plot_binned_hist.parquet: $(TASK_DIR)/14_consolidate_aef_cimmyt_all.py | $(CIMMYT_CSVS)
	cd $(DATA_DIR) && $(MPC_ENV) python3 $< --type binned_hist --csv_dir $(CIMMYT_CSVS)/binned_hist

# Robustness: quantile-bin variant of step 10
consolidate_qbin: $(TASK_DIR)/consolidate_qbin_parquet.py
	cd $(DATA_DIR) && $(MPC_ENV) python3 $<

# A missing CSV directory means the Drive exports have not been pulled yet
$(MUN_MEAN_CSVS) $(ADC_MEAN_CSVS) $(MUN_HIST_CSVS) $(ADC_HIST_CSVS) $(MUN_BIN_CSVS) $(ADC_BIN_CSVS) $(CIMMYT_CSVS):
	@echo "MISSING $@ : pull the corresponding Drive export folder here first (make extract_embeddings)"; exit 1
