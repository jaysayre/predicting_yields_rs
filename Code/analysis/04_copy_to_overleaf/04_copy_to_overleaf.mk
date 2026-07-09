### 04_copy_to_overleaf.mk
# Copy all tables and figures to Overleaf directory

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
	cp "$(PLOTS_DIR)/accuracy_other_crops_adc_2022.tex" "$(OVERLEAF_DIR)/"
	# accuracy_other_crops_mun_2022.tex (Table 8) cut from the paper (2026-06-10)
	cp "$(TABLES_DIR)/accuracy_mun_level_2022.tex"           "$(OVERLEAF_DIR)/"
	cp "$(TABLES_DIR)/validation_mun_level_2022.tex"          "$(OVERLEAF_DIR)/"
	cp "$(TABLES_DIR)/census_thought_experiment_2022.tex"    "$(OVERLEAF_DIR)/"
	@echo "Copying figures to Overleaf..."
	cp "$(EXTRAS_DIR)/fig_representativeness_targeting.png" "$(FIG_DIR)/"
	cp "$(EXTRAS_DIR)/fig_ranking_inversion.png"           "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/accuracy_scatter_combined_2022.pdf"  "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/accuracy_scatter_seasonal_2022.pdf"  "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_mun_allmx.png"            "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_allmx_adc_with_legend.png" "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_mun.png"                  "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_adc.png"                  "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maize_monthly_harvesting.png"        "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_mun_siap_allmx_nolegend_2018.png"  "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_mun_pred_allmx_nolegend_2018.png"  "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_allmx_adc.png"            "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_adc_preds.png"            "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_adc_pred_error_prederror_ls_noleg.png"     "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_adc_pred_error_prederror_rcpred_noleg.png" "$(FIG_DIR)/"
	cp "$(PLOTS_DIR)/maizeyield_adc_pred_error_prederror_munyield.png"     "$(FIG_DIR)/"
	@echo "Done — all tables and figures copied to Overleaf."
