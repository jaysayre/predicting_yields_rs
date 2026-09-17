### 03_train_rf_gb.mk
# Train the AEF yield models (paper Sec. 3). Scripts run in step order:
#   1_rf_yield_prediction.py            "AEF mean" benchmark: RF on the 64 per-dimension means
#                                       -> adc_alpha_earth_preds_maize.parquet
#   2_gb_aef_hist_ensemble.py           AEF Hist (64x8 subsampled bin shares, 512 feats) + percentile
#                                       component + their w=0.5 ensemble (AEF Hist Ens.)
#                                       -> adc_aef_hist_bins_gb_preds.parquet, adc_aef_hist_ens_preds.parquet,
#                                          adc_aef_hist_ens_eval.parquet (yields + season-matched corrections;
#                                          backbone of analysis/02), adc_aef_hist_ens_components.parquet (w sweep)
#   3_mun_cv_aef_hist_bins.py           municipality-level LOYO + random 5-fold CV of AEF Hist
#                                       (featurization-diagram panel 4; prose stats)
#   4_gb_aef_hist_ensemble_other_crops.py  per-crop ensembles (sorghum, sugarcane, wheat, avocado)
#                                       -> adc_aef_hist_ens_preds_<crop>.parquet (other-crop tables)
# Robustness variants (prose numbers; not in the default chain):
#   gb_aef_hist_ensemble_qbin.py        equal-mass quantile bins       make train_aef_hist_ens_qbin
#   gb_aef_hist_ensemble_dm.py          demeaned features              make train_aef_hist_ens_dm
#   gb_aef_hist_ensemble_oi_trained.py  fall-winter-trained ensemble   make train_oi_ensemble
# dep/ holds the superseded NDVI-histogram / harmonic GB models, the 448-feature
# percentile-only muni CV, and the East Africa extension (cut 2026-06-10).
SHELL := /bin/bash   # 'source' for conda activation needs bash, not dash

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/train/03_train_rf_gb

AEF_DIR   := $(DATA_DIR)/Data/alpha_earth
PREDS_DIR := $(DATA_DIR)/Data/predictions

AEF_MEANS := $(AEF_DIR)/alpha_earth_mex_muns.parquet $(AEF_DIR)/alpha_earth_mex_adcs.parquet
AEF_HISTS := $(AEF_DIR)/alpha_earth_mex_mun_hist.parquet $(AEF_DIR)/alpha_earth_mex_adcs_hist.parquet \
             $(AEF_DIR)/alpha_earth_mex_mun_binned_hist.parquet $(AEF_DIR)/alpha_earth_mex_adcs_binned_hist.parquet

.PHONY: train_rf_gb train_other_crops
train_rf_gb: \
	$(PREDS_DIR)/adc_alpha_earth_preds_maize.parquet \
	$(PREDS_DIR)/adc_aef_hist_ens_eval.parquet \
	$(PREDS_DIR)/mun_aef_hist_bins_gb_loyo_preds.parquet \
	$(PREDS_DIR)/adc_aef_hist_ens_preds_sorghum.parquet

# Step 1 — AEF mean benchmark (IMPROVED=False: RF on the 64 means alone)
$(PREDS_DIR)/adc_alpha_earth_preds_maize.parquet: $(TASK_DIR)/1_rf_yield_prediction.py $(AEF_MEANS)
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# Step 2 — AEF Hist + AEF Hist Ensemble (the paper's primary models)
$(PREDS_DIR)/adc_aef_hist_ens_eval.parquet \
$(PREDS_DIR)/adc_aef_hist_ens_preds.parquet \
$(PREDS_DIR)/adc_aef_hist_bins_gb_preds.parquet \
$(PREDS_DIR)/adc_aef_hist_ens_components.parquet &: $(TASK_DIR)/2_gb_aef_hist_ensemble.py $(AEF_MEANS) $(AEF_HISTS)
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# Step 3 — municipality-level CV of AEF Hist (LOYO + random muni-year 5-fold)
$(PREDS_DIR)/mun_aef_hist_bins_gb_loyo_preds.parquet \
$(PREDS_DIR)/mun_aef_hist_bins_gb_kfold_preds.parquet &: $(TASK_DIR)/3_mun_cv_aef_hist_bins.py $(AEF_DIR)/alpha_earth_mex_mun_binned_hist.parquet
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# Step 4 — other crops (~30 min; one run writes all four crop parquets, the
# sorghum file stands in for the group). analysis/03 carries the same rule so
# it can be rebuilt from the analysis stage alone.
train_other_crops: $(PREDS_DIR)/adc_aef_hist_ens_preds_sorghum.parquet
$(PREDS_DIR)/adc_aef_hist_ens_preds_sorghum.parquet: $(TASK_DIR)/4_gb_aef_hist_ensemble_other_crops.py $(AEF_MEANS) $(AEF_HISTS)
	cd $(DATA_DIR) && $(ML_ENV) python3 $<

# ── Robustness variants ───────────────────────────────────
.PHONY: train_aef_hist_ens_dm train_aef_hist_ens_qbin train_oi_ensemble
train_aef_hist_ens_dm: $(TASK_DIR)/gb_aef_hist_ensemble_dm.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<
train_aef_hist_ens_qbin: $(TASK_DIR)/gb_aef_hist_ensemble_qbin.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<
train_oi_ensemble: $(TASK_DIR)/gb_aef_hist_ensemble_oi_trained.py
	cd $(DATA_DIR) && $(ML_ENV) python3 $<
