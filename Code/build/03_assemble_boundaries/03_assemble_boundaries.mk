### 03_assemble_boundaries.mk
# Assemble geographic boundaries (ADC shapefiles, agland masks)

CODE_DIR := $(PROJ_DIR)/Code
TASK_DIR := $(CODE_DIR)/build/03_assemble_boundaries

INEGI_DIR := $(DATA_DIR)/Data/INEGI
AGLAND_DIR := $(DATA_DIR)/Data/Mexico_agland/Outputs

.PHONY: assemble_boundaries
assemble_boundaries: \
	$(INEGI_DIR)/Areas_Censal_Agropecuario_2016/census_areas.shp \
	$(INEGI_DIR)/Marco_Geo_2022/Outputs/ageb_22.shp \
	$(AGLAND_DIR)/inegi_agland_16_adc.shp

$(INEGI_DIR)/Areas_Censal_Agropecuario_2016/census_areas.shp: $(TASK_DIR)/1_assemble_2016_AMCA_shapefile.ipynb
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $<

$(INEGI_DIR)/Marco_Geo_2022/Outputs/ageb_22.shp: $(TASK_DIR)/2_assemble_marco_geo_2022.ipynb
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $<

$(AGLAND_DIR)/inegi_agland_16_adc.shp: $(TASK_DIR)/3_assemble_SIAP_agland.ipynb $(TASK_DIR)/4_segment_inegi_agland.ipynb
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $(TASK_DIR)/3_assemble_SIAP_agland.ipynb
	cd $(DATA_DIR) && $(MPC_ENV) $(NB_EXEC) $(TASK_DIR)/4_segment_inegi_agland.ipynb
