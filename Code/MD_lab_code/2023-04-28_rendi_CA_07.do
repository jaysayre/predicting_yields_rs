/* rendi_CA_07.do -- jsayre@berkeley.edu

11-23-2022 Note: need to investigate the ADC-renglon match more
*/

// set locale_ui en
set more off

local date = "2023-04-28"

******* Directorios
local wrk_dir = "Z:\Procesamiento\Trabajo\"
local ag_census_clean_07_dir = "`wrk_dir'Censo Agrícola\Cleaned\"

local intermed_dir = "`wrk_dir'Intermediates\"
local results_dir = "Z:\Resultados\LM2304-CA-`date'\"
capture mkdir "`results_dir'"

******* Insumos
local cup_to_mun = "`ag_census_clean_07_dir'geo_identifier_muncode_only_2007.dta"
local muncodes = "`wrk_dir'Codigos municipios\muncodes_clean.dta"
local labor = "`ag_census_clean_07_dir'labor_2007.dta"
local land = "`ag_census_clean_07_dir'land_2007.dta"
local irrig = "`ag_census_clean_07_dir'land_by_use_2007.dta"
local non_labor1 = "`ag_census_clean_07_dir'non_labor_inputs_1"
local non_labor2 = "`ag_census_clean_07_dir'non_labor_inputs_2"
local non_labor3 = "`ag_census_clean_07_dir'non_labor_inputs_3"
local non_labor4 = "`ag_census_clean_07_dir'non_labor_inputs_4"
local finance = "`ag_census_clean_07_dir'finance"
local market = "`ag_census_clean_07_dir'markets"
local contracts = "`ag_census_clean_07_dir'ag_contracting_07.dta"
local contracts_by_crop = "`ag_census_clean_07_dir'ag_contracts_namedcrop_07.dta"
local spring_sum_ag = "`ag_census_clean_07_dir'ag_prod_spring_summer.dta"
local fall_winter_ag = "`ag_census_clean_07_dir'ag_prod_fall_winter.dta"
local perennial_ag = "`ag_census_clean_07_dir'ag_prod_perennials.dta"
local crop_a_name  =  "`wrk_dir'cultivo_a_nombre.dta"

******* Intermediarios
local assembled_data1 = "`ag_census_clean_07_dir'ca2007_prepped_farmdata"
local assembled_data2 = "`ag_census_clean_07_dir'ca2007_prepped_farmcropadc_data"
local assembled_data_full = "`ag_census_clean_07_dir'ca2007_prepped_data_FULL"
local geo_farm = "`ag_census_clean_07_dir'geo_identifier_2007.dta"
local geo_farm_nor = "`ag_census_clean_07_dir'geo_identifier_notby_renglon_2007.dta"
local geo_farm_ageb = "`ag_census_clean_07_dir'geo_identifier_ageb_only_2007.dta"
local geo_farm_reorder = "`ag_census_clean_07_dir'geo_identifier_ordered_renglon_2007.dta"

local modifier = ""

******* Rendimiento
local rendi_adc = "`results_dir'rendimiento_agr_adc.dta"
local rendi_ageb = "`results_dir'rendimiento_agr_ageb.dta"

****************************************************************************************************************************************************************
****************************************************************************************************************************************************************
****************************************************************************************************************************************************************
****************************************************************************************************************************************************************
****************************************************************************************************************************************************************

/*
****************
***(1) location*
****************

*** J: My version of geolocators
*** the id_cup x renglon part really matters! different renglons can be in different agebs/muns/adcs

use "`geo_farm'", clear

preserve
collapse (sum) renglon, by(id_cup muncode ageb adc)
egen countcups = count(renglon), by(id_cup)
keep if countcups == 1
drop countcups renglon
rename muncode muncode_on
*rename locality locality_on
rename ageb ageb_on
rename adc adc_on
save "`geo_farm_nor'", replace
restore

collapse (sum) renglon, by(id_cup muncode ageb)
egen countcups = count(renglon), by(id_cup)
keep if countcups == 1
drop countcups renglon
rename muncode muncode_from_ageb_match
rename ageb ageb_from_ageb_match
save "`geo_farm_ageb'", replace
*/

/*
***************************
******* (3) TOTAL LAND*****
***************************

*** VARIABLES 
*** land "Total land in farm unit"
use "`land'", clear
	
****************************
*******(4) LAND BY TYPE*****
****************************

*** VARIABLES 
*** land_rain "Total rain fed land in farm unit"
*** land_irr "Total irrigated land in farm unit"

merge 1:1 id_cup using "`irrig'", nogen

************************************************************
*******(7) CAPITAL AND INTERMEDIATE INPUTS******************
************************************************************

***VARIABLES*

***impr_seed "Area with improved seeds"
***chem_fert "Area with chemical fertilizers"
***nat_fert "Area with natural fertilizers"
***herb_insect "Application of insecticides"
***graft_trees  "Performed tree grafts"
***tech_ass "Received technical assistance"
merge 1:1 id_cup using "`non_labor1'", nogen

***tractor
***quant_trucks
***quant_tractor
*merge 1:1 id_cup using "`non_labor2'", nogen

***packer
***selector
***dehydrator
***processor
***shredder
*merge 1:1 id_cup using "`non_labor3'", nogen

***animals "Used animals"
merge 1:1 id_cup using "`non_labor4'", nogen

*************
*(8) FINANCE*
*************
***credit 
***insurance
*merge 1:1 id_cup using "`finance'", nogen

***************
***(9) MARKETS*
***************
merge 1:1 id_cup using  "`market'", nogen

************************
*** (10) CONTRACTING ***
************************
merge 1:1 id_cup using "`contracts'", nogen

*************************
**** (11) MEMBERSHIP ****
*************************

merge 1:1 id_cup using "`ag_census_clean_07_dir'membership", nogen
save "`assembled_data1'", replace
*/

/*
*********************************************
******  Merge id_cup+renglon with ADC  ******
*********************************************
use "`spring_sum_ag'", clear
gen type = "p-v"
append using "`fall_winter_ag'"
replace type = "o-i" if missing(type)
append using "`perennial_ag'"
replace type = "peren" if missing(type)
*** From here, agricultural production data is in the id_cup-renglon level
*** Keep this at this level (for now) to merge in and appropriately weight geo characteristics

*** CAUTION: THIS PART IS SPECULATIVE AND ONLY POSSIBLY CORRECT
*** In the terrenos file, renglons appear to be ordered 1-N, but in ag. data, no
*** Maybe ordering corresponds to the correct renglon?
*** Therefore, stack all id_cup/renglons to do merge

preserve
append using "`ag_census_clean_07_dir'ag_prod_greenhouse.dta"
append using "`ag_census_clean_07_dir'cultivos_intercalados.dta"
collapse (sum) land_input, by(id_cup renglon)
drop land_input
egen grp_cup_renglon = rank(renglon), by(id_cup) unique
rename renglon renglon_orig
rename grp_cup_renglon renglon
merge m:1 id_cup renglon using "`geo_farm'", nogen keep(3)

drop renglon
rename renglon_orig renglon
rename muncode muncode_renglon_reorder
rename ageb ageb_renglon_reorder
rename adc adc_renglon_reorder
save "`geo_farm_reorder'", replace
restore

*** CAUTION: THIS PART IS SPECULATIVE AND ONLY POSSIBLY CORRECT
merge m:1 id_cup renglon using "`geo_farm'", nogen keep(1 3)
*** If merge didn't work, match with renglon == 1, if there is only one adc listed for the farm unit in "TERRENOS.dta"
merge m:1 id_cup using "`geo_farm_nor'", nogen keep(1 3)
replace muncode = muncode_on if missing(muncode)
replace ageb = ageb_on if missing(ageb)
replace adc = adc_on if missing(adc)
drop muncode_on ageb_on adc_on

merge m:1 id_cup using "`geo_farm_ageb'", nogen keep(1 3)
replace muncode = muncode_from_ageb_match if missing(muncode)
replace ageb = ageb_from_ageb_match if missing(ageb)
drop muncode_from_ageb_match ageb_from_ageb_match

merge m:1 id_cup renglon using "`geo_farm_reorder'", nogen keep(1 3)
gen ageb_match = 1 if ageb == ageb_renglon_reorder
replace ageb_match = 1 if missing(ageb)
replace ageb_match = 0 if missing(ageb_match)

*** CAUTION: THIS PART IS SPECULATIVE AND ONLY POSSIBLY CORRECT
gen adc_experimental = adc
replace adc_experimental = adc_renglon_reorder if missing(adc_experimental) & ageb_match == 1
gen ageb_experimental = ageb 
replace ageb_experimental = ageb_renglon_reorder if missing(ageb_experimental)
gen muncode_experimental = muncode 
replace muncode_experimental = muncode_renglon_reorder if missing(muncode_experimental)
drop ageb_renglon_reorder adc_renglon_reorder muncode_renglon_reorder ageb_match

rename muncode muncode_adc
merge m:1 id_cup using "`cup_to_mun'", nogen keep(1 3)
replace muncode_adc = muncode if missing(muncode_adc)

*** Why doesn't this match? (i.e. terrenos != ubicacion). In these cases, may wish to drop these adc codes
gen flag_geo_merge = 1 if muncode_adc != muncode
replace flag_geo_merge = 0 if missing(flag_geo_merge)

*** DONT NECESSARILY WANT TO DO THIS
***replace ageb = "" if flag_geo_merge == 1
***replace adc = "" if flag_geo_merge == 1
***replace muncode_adc = muncode if flag_geo_merge == 1
drop muncode 
rename muncode_adc muncode
replace muncode_experimental = muncode if missing(muncode_experimental)

*** If individual crop is contracted
merge m:1 id_cup nombre_cul using "`contracts_by_crop'", nogen keep(1 3)
replace is_listed_contract_crp = 0 if missing(is_listed_contract_crp)

save "`assembled_data2'", replace
*/

use "`assembled_data2'", clear

*** Convert to name abbreviations
rename nombre_cul cultivo
replace cultivo = "CASTANA STERCULIA" if cultivo == "CASTAA STERCULIA"
*** 2022-11-22: All matches _merge == 3
merge m:1 cultivo using "`crop_a_name'", keep(3) nogen

*** Just use experimental ADCs+AGEBs
replace ageb = ageb_experimental if missing(ageb)
replace adc = adc_experimental if missing(adc)

*** Sum input/output up to farm by adc level
gen num_terrenos = 1
collapse (sum) land_*t vol_output num_terrenos, by(id_cup name ageb adc type)

**** If only matching with fall-winter, spring-summer, and perennial then about 1,204,146 farms lost
*** rest are found in greenhouse
merge m:1 id_cup using "`assembled_data1'", nogen keep(3)

*** Variables vary at farm, but not crop, level
rename land_irr ha_irrig
rename land_rain ha_rainfed
gen share_irrig = ha_irrig/(ha_irrig+ha_rainfed)

*** Different ways to divide output for multi crop farms
*egen ha_planted_inputs = total(land_input), by(id_cup)
*gen crop_share_farm = land_input/ha_planted_inputs

*** Produce summary statistics by crop
*gen uses_fert = 1 if area_chem_fert+area_nat_fert > 0
*replace uses_fert = 0 if missing(uses_fert)

rename land_input sup_sem
rename land_output sup_cos
rename vol_output Q

*** Compute indicators for if rainfed or not, even if not super precise since at farm level, not crop
gen sup_irrig  = sup_sem*share_irrig
gen sup_rf     = sup_sem*(1-share_irrig)

*** Gen mun. + entidad ind.
gen muncode  = substr(ageb,1,5)
gen CVE_ENT = substr(ageb,1,2)
destring muncode, replace

save "`assembled_data_full'", replace

use "`assembled_data_full'", clear 

*** Now, produce yield estimates at ADC level
preserve
drop if missing(adc)
collapse (sum) sup_sem sup_cos Q sup_irrig sup_rf num_terrenos, by(name CVE_ENT muncode ageb adc type)

*** Generate yield
gen yield = Q/sup_sem
gen yield_irrig = Q/sup_irrig
gen yield_rf = Q/sup_rf

save "`rendi_adc'", replace
restore 

*** Now, produce yield estimates at AGEB level
drop if missing(ageb)
collapse (sum) sup_sem sup_cos Q sup_irrig sup_rf num_terrenos, by(name CVE_ENT muncode ageb type)

*** Generate yield
gen yield = Q/sup_sem
gen yield_irrig = Q/sup_irrig
gen yield_rf = Q/sup_rf

save "`rendi_ageb'", replace

