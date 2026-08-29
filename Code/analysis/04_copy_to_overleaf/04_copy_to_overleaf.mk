### 04_copy_to_overleaf.mk
# Copy all tables and figures to Overleaf directory
SHELL := /bin/bash   # 'source' for conda activation needs bash, not dash

CODE_DIR   := $(PROJ_DIR)/Code
PLOTS_DIR  := $(DATA_DIR)/plots
TABLES_DIR := $(DATA_DIR)/tables
EXTRAS_DIR := $(DATA_DIR)/plots/coauthor_extras_paper
FIG_DIR    := $(OVERLEAF_DIR)/figures

.PHONY: copy_to_overleaf
copy_to_overleaf:
	@echo "Copying tables to Overleaf..."
	cp "$(PLOTS_DIR)/accuracy_combined_2022.tex"        "$(OVERLEAF_DIR)/"
	cp "$(PLOTS_DIR)/accuracy_fall_winter_2022.tex"     "$(OVERLEAF_DIR)/"
	cp "$(PLOTS_DIR)/accuracy_spring_summer_2022.tex"   "$(OVERLEAF_DIR)/"
	# 2026-08-16: sourced from TABLES_DIR, not PLOTS_DIR. gb_aef_hist_ensemble_other_crops.py
	# writes to tables/; the plots/ copy is a stale Feb file with no AEF Hist Ens. rows,
	# and copying it silently reverted the Overleaf table.
	cp "$(TABLES_DIR)/accuracy_other_crops_adc_2022.tex" "$(OVERLEAF_DIR)/"
	cp "$(PLOTS_DIR)/common_sample_combined_2022.tex"        "$(OVERLEAF_DIR)/"
	cp "$(PLOTS_DIR)/common_sample_spring_summer_2022.tex"   "$(OVERLEAF_DIR)/"
	# accuracy_other_crops_mun_2022.tex (Table 8) cut from the paper (2026-06-10)
	cp "$(TABLES_DIR)/accuracy_mun_level_2022.tex"           "$(OVERLEAF_DIR)/"
	cp "$(TABLES_DIR)/validation_mun_level_2022.tex"          "$(OVERLEAF_DIR)/"
	cp "$(TABLES_DIR)/census_thought_experiment_2022.tex"    "$(OVERLEAF_DIR)/"
	cp "$(TABLES_DIR)/accuracy_profile_by_adc_chars.tex"     "$(OVERLEAF_DIR)/"
	cp "$(TABLES_DIR)/accuracy_cimmyt_profile.tex"           "$(OVERLEAF_DIR)/"
	@echo "Copying figures to Overleaf..."
	cp "$(EXTRAS_DIR)/fig_representativeness_targeting.pdf" "$(FIG_DIR)/"
	cp "$(EXTRAS_DIR)/fig_ranking_inversion.pdf"           "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/fig_methodology_diagram.pdf"         "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/accuracy_scatter_combined_2022.pdf"  "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/accuracy_scatter_seasonal_2022.pdf"  "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/siap_mun_vs_adc_census_yield.pdf"    "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_mun_allmx.png"            "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_allmx_adc_with_legend.png" "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_mun.png"                  "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_adc.png"                  "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_adc_mun_diff.png"         "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maize_monthly_harvesting.png"        "$(FIG_DIR)/"
	# 2026-08: municipal SIAP/pred pair regenerated on CA22 -> _2022 suffix
	cp "$(PLOTS_DIR)/maizeyield_mun_siap_allmx_2022.png"  "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_mun_pred_allmx_2022.png"  "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_adc_truth_2022_matched.png"  "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_adc_preds_2022_matched.png"  "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_allmx_adc.png"            "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_adc_preds.png"            "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_adc_pred_error_prederror_ls_noleg.png"     "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_adc_pred_error_prederror_rcpred_noleg.png" "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_adc_pred_error_prederror_munyield.png"     "$(FIG_DIR)/"
	@echo "Done — all tables and figures copied to Overleaf."
