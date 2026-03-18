### 01_extract_embeddings.mk
# Extract AlphaEarth Foundation embeddings from Google Earth Engine
# MANUAL: Requires Earth Engine authentication

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/train/01_extract_embeddings

.PHONY: extract_embeddings
extract_embeddings:
	@echo "MANUAL STEP: Run ee_alpha_earth_download.py interactively"
	@echo "  Requires: Earth Engine auth + ML_env"
	@echo "  Output:   Maize_prediction/Data/alpha_earth/*.parquet"
