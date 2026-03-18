### 01_scrape_siap.mk
# Scrape SIAP monthly production data via Selenium
# MANUAL: Requires Chrome/Selenium — not automatable in headless pipeline

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/build/01_scrape_siap

.PHONY: scrape_siap
scrape_siap:
	@echo "MANUAL STEP: Run 1_ScrapeMonthlyProductionSIAP.ipynb interactively"
	@echo "  Requires: Chrome browser + Selenium"
	@echo "  Output:   data/SIAP_monthly/Input/Y*S*M*.csv"
