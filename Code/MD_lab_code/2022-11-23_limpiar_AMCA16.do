/* limpiar_AMCRA16.do -- Jay Sayre, jsayre@berkeley.edu

Objetivo: limpiar y obtener estadisticas descriptivas de la Actualización del Marco Censal Agropecuario 2016

Insumos: En Procesamiento/Trabajo/
Rendimientos: TBD
*/

*clear all
set more off

*** Selecc. fecha para exportar rendimientos
local date = "2022-11-23"

*** Directorios
local wrk_dir = "Z:\Procesamiento\Trabajo\"
local results_dir = "Z:\Resultados\LM2304-AMCA-`date'\"
local ca_results_dir = "Z:\Resultados\LM2304-CA-`date'\"


local ag_census_dir = "`wrk_dir'Censo Agrícola\"
local amca_dir = "`wrk_dir'AMCA\"
local intermed_dir = "`wrk_dir'Intermediates\"

*** Crear carpetas
capture mkdir "`results_dir'"
capture mkdir "`intermed_dir'"

*** Insumos
local amca1         =  "`amca_dir'AMCA_BASE_INTEGRADA_1.dta"
local amca2         =  "`amca_dir'AMCA_BASE_INTEGRADA_2.dta"
local crop_a_name   =  "`wrk_dir'cultivo_a_nombre.dta"
local siap_data     =  "`wrk_dir'siap_ag_prod_estimation_ca2007.dta"
local rendi_adc     =  "`ca_results_dir'rendimiento_agr_adc.dta"
*** 2022-11-23: You have to update ADC16 list -- Nuevo Leon (19) is missing, rest is fine
local ca07_a_ca16   =  "`wrk_dir'ca07_ca16_corr.dta"

*** Intermediarios
local crops_1       =  "`intermed_dir'crops_1.dta"
local crops_2       =  "`intermed_dir'crops_2.dta"

local amca_1_clean  =  "`intermed_dir'AMCA_1.dta"
local amca_2_clean  =  "`intermed_dir'AMCA_2.dta"
local siap2007      =  "`intermed_dir'SIAP2007.dta"
local siap2016      =  "`intermed_dir'SIAP2016.dta"
local amca_agebint  =  "`intermed_dir'AMCA_sup_ageb_temp.dta"
local amca_adc_int  =  "`intermed_dir'AMCA_sup_adc_temp.dta"
local ca16_tbl_by_n =  "`intermed_dir'ca07_ca16_corr_by_crop.dta"
local ca16_tbl_sub  =  "`intermed_dir'ca07_ca16_corr_by_crop_subset.dta"
local ca07_y_wght   =  "`intermed_dir'ca07_yield_wght_adc.dta"
local ca07_y_wageb  =  "`intermed_dir'ca07_yield_wght_ageb.dta"

*** Rendimiento
local amca_ageb     =  "`results_dir'AMCA_sup_ageb.dta"
local amca_adc      =  "`results_dir'AMCA_sup_adc.dta"

*** Codigo
/*
*** Read in and clean SIAP data
*** 2022-11-23: Need to double check this year matches with AMCA 2016
use year muncode name q ha_plante* if year == 2007 using "`siap_data'", clear
destring muncode, replace
rename q q_2007_siap
rename ha_plante* ha_plante*_07siap
drop year
save "`siap2007'", replace
*** 2022-11-23: Need to double check this year matches with AMCA 2016
use year muncode name q if year == 2016 using "`siap_data'", clear
destring muncode, replace
drop year
save "`siap2016'", replace
*/
/*
*** Read in AMCA data
*** Has states 1-16
use "`amca1'", clear
gen muncode = CVE_ENT+CVE_MUN
destring muncode, replace
label variable muncode "5 digit loc. code"
gen ageb = CVE_ENT+CVE_MUN+CVE_AGEB
gen adc = CVE_ENT+CVE_MUN+CVE_AGEB+CVE_MZA
drop CVE_MUN CVE_AGEB CVE_MZA CVE_LOC

keep if DES_GR == "AGRICULTURA"
drop DES_GR CVE_PPALGR

*** Drop codes, just keep data
drop CVE_DERECH CVE_TENENC CVE_PPALES

*** Fix gran productor var
destring gp, replace
replace gp = 0 if gp != 1

*** Gen better irrig var
gen irrig = 0 
replace irrig = 1 if CVE_TIPO == "R"
drop CVE_TIPO

gen str30 crop = substr(DES_CUGAFO,1,30)
gen cultivo = ustrto(ustrnormalize(crop, "nfd"), "ascii", 2)
drop DES_CUGAFO crop CVE_CUGAFO

*** Gen land type
gen ownership = ustrto(ustrnormalize(DES_DERECH, "nfd"), "ascii", 2)
gen community = ustrto(ustrnormalize(DES_TENENC, "nfd"), "ascii", 2)
gen greenhouse = 0 
replace greenhouse = 1 if DES_ES == "PROTEGIDA"
drop DES_DERECH DES_TENENC DES_ES

replace cultivo = "AMAPOLA AMARILLA" if cultivo == "AMAPOLA DE JARDIN"
replace cultivo = "ALFALFA VERDE" if cultivo == "ALFALFA"
replace cultivo = "ALFALFA VERDE" if cultivo == "ALFALFILLA"
replace cultivo = "ALGODON HUESO" if cultivo == "ALGODON"
replace cultivo = "ALGODON HUESO" if cultivo == "ALGODONCILLO"
replace cultivo = "ARROZ PALAY" if cultivo == "ARROZ"
replace cultivo = "ARRUGULA" if cultivo == "ARUGULA"
replace cultivo = "AVELLANO" if cultivo == "AVELLANA"
replace cultivo = "AVENA GRANO" if cultivo == "AVENA"
replace cultivo = "AZUCENA" if cultivo == "AZUCENA AMARILLA"
replace cultivo = "CACAO" if cultivo == "CACAO VOLADOR"
replace cultivo = "CAFE CEREZA" if cultivo == "CAFE"
replace cultivo = "CAFE CEREZA" if cultivo == "CAFE CIMARRON"
replace cultivo = "CANELO" if cultivo == "CANELA"
replace cultivo = "CAPULIN" if cultivo == "CAPULIN AGARROSO"
replace cultivo = "CASTANO" if cultivo == "CASTAA"
replace cultivo = "CANA DE AZUCAR" if cultivo == "CAA DE AZUCAR"
replace cultivo = "CANA DE AZUCAR" if cultivo == "CAA SILVESTRE"
replace cultivo = "CEBADA GRANO" if cultivo == "CEBADA"
replace cultivo = "CEDRO" if cultivo == "CEDRO ROJO"
replace cultivo = "CHILE VERDE" if cultivo == "CHILE"
replace cultivo = "COL (REPOLLO)" if cultivo == "COL"
replace cultivo = "COCO" if cultivo == "COCO, PALMA DE"
replace cultivo = "EBO (JANAMARGO O VEZA)" if cultivo == "EBO"
replace cultivo = "GARBANZO GRANO" if cultivo == "GARBANZO"
replace cultivo = "GRANADA" if cultivo == "GRANADA CHINA"
replace cultivo = "HABA VERDE" if cultivo == "HABA"
replace cultivo = "JACA (JACKFRUIT)" if cultivo == "JACA"
replace cultivo = "JATROPHA" if cultivo == "JATROFA"
replace cultivo = "TOMATE ROJO (JITOMATE)" if cultivo == "JITOMATE"
replace cultivo = "AGAVE" if cultivo == "MAGUEY"
replace cultivo = "AGAVE" if cultivo == "MAGUEY MORADO"
replace cultivo = "MAIZ GRANO" if cultivo == "MAIZ"
replace cultivo = "MALVA ROSA" if cultivo == "MALVA"
replace cultivo = "MANGLE ROJO" if cultivo == "MANGLE NEGRO"
replace cultivo = "MOSTAZA NEGRA" if cultivo == "MOSTAZA"
replace cultivo = "NOPALITOS" if cultivo == "NOPAL"
replace cultivo = "PASTOS" if cultivo == "PASTO ESPAOL"
replace cultivo = "PASTOS" if cultivo == "PASTO LISTON"
replace cultivo = "PASTOS" if cultivo == "PASTO MONDO"
replace cultivo = "PASTOS" if cultivo == "PASTO CULTIVADOS"
replace cultivo = "PEPINO" if cultivo == "PEPINO AMARGO"
replace cultivo = "PINO" if cultivo == "PINO CIPRES ITALIANO"
replace cultivo = "PINO" if cultivo == "PINO PIONERO"
replace cultivo = "PITAYA" if cultivo == "PITAHAYA MORADA"
replace cultivo = "PINA" if cultivo == "PIA"
replace cultivo = "REMOLACHA FORRAJERA" if cultivo == "REMOLACHA AZUCARERA"
replace cultivo = "COL (REPOLLO)" if cultivo == "REPOLLO CHINO"
replace cultivo = "ROMERITO" if cultivo == "ROMERITOS"
replace cultivo = "SORGO GRANO" if cultivo == "SORGO"
replace cultivo = "TOMATE ROJO (JITOMATE)" if cultivo == "TOMATE CIMARRON"
replace cultivo = "TOMATE ROJO (JITOMATE)" if cultivo == "TOMATE DE CASCARA"
replace cultivo = "TOMATE ROJO (JITOMATE)" if cultivo == "TOMATE ROJO"
replace cultivo = "TOMATE VERDE" if cultivo == "TOMATILLO"
replace cultivo = "TORONJA (POMELO)" if cultivo == "TORONJA"
replace cultivo = "TRIGO GRANO" if cultivo == "TRIGO"
replace cultivo = "TRITICALE GRANO" if cultivo == "TRITICALE"
replace cultivo = "YUCA ALIMENTICIA" if cultivo == "YUCA"
replace cultivo = "RYE GRASS EN VERDE" if cultivo == "ZACATE"

merge m:1 cultivo using "`crop_a_name'", keep(1 3) nogen

rename ID_TERR id_terr
rename CVE_ENT cve_ent
rename SUPER_CART super_cart
rename SUP_SEM sup_sem

*** Save to intermediate file
save "`amca_1_clean'", replace

*preserve
*keep cultivo
*duplicates drop cultivo, force
*save "`crops_1'", replace
*restore

*** States 17-32
use "`amca2'", clear
gen muncode = cve_ent+cve_mun
destring muncode, replace
label variable muncode "5 digit loc. code"
gen ageb = cve_ent+cve_mun+cve_ageb
gen adc = cve_ent+cve_mun+cve_ageb+cve_mza
drop cve_mun cve_ageb cve_mza cve_loc D_R

keep if des_gr == "AGRICULTURA"
drop des_gr cve_ppalgr

*** Drop codes, just keep data
drop cve_derech cve_tenenc cve_ppales

*** Fix gran productor var
destring gp, replace
replace gp = 0 if gp != 1

*** Gen better irrig var
gen irrig = 0 
replace irrig = 1 if cve_tipo == "R"
drop cve_tipo

*** Gen land type
gen ownership = ustrto(ustrnormalize(des_derech, "nfd"), "ascii", 2)
gen community = ustrto(ustrnormalize(des_tenenc, "nfd"), "ascii", 2)
gen greenhouse = 0
replace greenhouse = 1 if des_es == "PROTEGIDA"  
drop des_derech des_tenenc des_es

gen str30 crop = substr(des_cugafo,1,30)
gen cultivo = ustrto(ustrnormalize(crop, "nfd"), "ascii", 2)
drop des_cugafo crop cve_cugafo

replace cultivo = "AMAPOLA AMARILLA" if cultivo == "AMAPOLA DE JARDIN"
replace cultivo = "ALFALFA VERDE" if cultivo == "ALFALFA"
replace cultivo = "ALFALFA VERDE" if cultivo == "ALFALFILLA"
replace cultivo = "ALGODON HUESO" if cultivo == "ALGODON"
replace cultivo = "ALGODON HUESO" if cultivo == "ALGODONCILLO"
replace cultivo = "ARROZ PALAY" if cultivo == "ARROZ"
replace cultivo = "ARRUGULA" if cultivo == "ARUGULA"
replace cultivo = "AVELLANO" if cultivo == "AVELLANA"
replace cultivo = "AVENA GRANO" if cultivo == "AVENA"
replace cultivo = "AZUCENA" if cultivo == "AZUCENA AMARILLA"
replace cultivo = "CACAO" if cultivo == "CACAO VOLADOR"
replace cultivo = "CAFE CEREZA" if cultivo == "CAFE"
replace cultivo = "CAFE CEREZA" if cultivo == "CAFE CIMARRON"
replace cultivo = "CANELO" if cultivo == "CANELA"
replace cultivo = "CAPULIN" if cultivo == "CAPULIN AGARROSO"
replace cultivo = "CASTANO" if cultivo == "CASTAA"
replace cultivo = "CANA DE AZUCAR" if cultivo == "CAA DE AZUCAR"
replace cultivo = "CANA DE AZUCAR" if cultivo == "CAA SILVESTRE"
replace cultivo = "CEBADA GRANO" if cultivo == "CEBADA"
replace cultivo = "CEDRO" if cultivo == "CEDRO ROJO"
replace cultivo = "CHILE VERDE" if cultivo == "CHILE"
replace cultivo = "COL (REPOLLO)" if cultivo == "COL"
replace cultivo = "COCO" if cultivo == "COCO, PALMA DE"
replace cultivo = "EBO (JANAMARGO O VEZA)" if cultivo == "EBO"
replace cultivo = "GARBANZO GRANO" if cultivo == "GARBANZO"
replace cultivo = "GRANADA" if cultivo == "GRANADA CHINA"
replace cultivo = "HABA VERDE" if cultivo == "HABA"
replace cultivo = "JACA (JACKFRUIT)" if cultivo == "JACA"
replace cultivo = "JATROPHA" if cultivo == "JATROFA"
replace cultivo = "TOMATE ROJO (JITOMATE)" if cultivo == "JITOMATE"
replace cultivo = "AGAVE" if cultivo == "MAGUEY"
replace cultivo = "AGAVE" if cultivo == "MAGUEY MORADO"
replace cultivo = "MAIZ GRANO" if cultivo == "MAIZ"
replace cultivo = "MALVA ROSA" if cultivo == "MALVA"
replace cultivo = "MANGLE ROJO" if cultivo == "MANGLE NEGRO"
replace cultivo = "MOSTAZA NEGRA" if cultivo == "MOSTAZA"
replace cultivo = "NOPALITOS" if cultivo == "NOPAL"
replace cultivo = "PASTOS" if cultivo == "PASTO ESPAOL"
replace cultivo = "PASTOS" if cultivo == "PASTO LISTON"
replace cultivo = "PASTOS" if cultivo == "PASTO MONDO"
replace cultivo = "PASTOS" if cultivo == "PASTO CULTIVADOS"
replace cultivo = "PEPINO" if cultivo == "PEPINO AMARGO"
replace cultivo = "PINO" if cultivo == "PINO CIPRES ITALIANO"
replace cultivo = "PINO" if cultivo == "PINO PIONERO"
replace cultivo = "PITAYA" if cultivo == "PITAHAYA MORADA"
replace cultivo = "PINA" if cultivo == "PIA"
replace cultivo = "REMOLACHA FORRAJERA" if cultivo == "REMOLACHA AZUCARERA"
replace cultivo = "COL (REPOLLO)" if cultivo == "REPOLLO CHINO"
replace cultivo = "ROMERITO" if cultivo == "ROMERITOS"
replace cultivo = "SORGO GRANO" if cultivo == "SORGO"
replace cultivo = "TOMATE ROJO (JITOMATE)" if cultivo == "TOMATE CIMARRON"
replace cultivo = "TOMATE ROJO (JITOMATE)" if cultivo == "TOMATE DE CASCARA"
replace cultivo = "TOMATE ROJO (JITOMATE)" if cultivo == "TOMATE ROJO"
replace cultivo = "TOMATE VERDE" if cultivo == "TOMATILLO"
replace cultivo = "TORONJA (POMELO)" if cultivo == "TORONJA"
replace cultivo = "TRIGO GRANO" if cultivo == "TRIGO"
replace cultivo = "TRITICALE GRANO" if cultivo == "TRITICALE"
replace cultivo = "YUCA ALIMENTICIA" if cultivo == "YUCA"
replace cultivo = "RYE GRASS EN VERDE" if cultivo == "ZACATE"

merge m:1 cultivo using "`crop_a_name'", keep(1 3) nogen

*** Save to intermediate file
save "`amca_2_clean'", replace

*preserve
*keep cultivo
*duplicates drop cultivo, force
*save "`crops_2'", replace
*restore
*/

/*
use "`amca_1_clean'", clear
append using "`amca_2_clean'"

*** Compute indicators for if rainfed or not
gen sup_irrig  = sup_sem*irrig
gen sup_rf     = sup_sem*(1-irrig)
gen sup_gp     = sup_sem*gp
gen sup_not_gp = sup_sem*(1-gp)


*** Gen ind for num of terrenos
gen num_terrenos = 1

*** Replace name for non matched crops
replace name = "Not specified" if missing(name)

*** Collapse down to ADC level
collapse (sum) super_cart sup_sem sup_irrig sup_rf sup_gp sup_not_gp num_terrenos, by(cve_ent muncode ageb adc name greenhouse)

merge m:1 muncode name using "`siap2016'", keep(1 3) nogen

*** Weight yields by land harvested alone
egen mun_total_sup_crop   = total(sup_sem), by(muncode name)
gen sup_share             = sup_sem/mun_total_sup_crop
gen q_share_equiv         = sup_share*q
gen yield_equiv           = q_share_equiv/sup_sem
drop mun_total_sup_crop sup_share q_share_equiv

*** Weight yields by land harvested, counting the land of grand producers as 50% more important
gen sup_gpw_sem           = 1.5*sup_gp + sup_not_gp
egen mun_tot_sup_crop_gpw = total(sup_gpw_sem), by(muncode name)
gen sup_gpw_share         = sup_gpw_sem/mun_tot_sup_crop_gpw
gen q_share_gpw           = sup_gpw_share*q
gen yield_gpw             = q_share_gpw/sup_sem
drop sup_gpw_sem mun_tot_sup_crop_gpw sup_gpw_share q_share_gpw


*** Keep only if not greenhouse
keep if greenhouse == 0 
drop greenhouse

save "`amca_adc_int'", replace
**** Checkpoint 1: Will return here
*/
/*
*** This step is to weight Q by Q by ADC in CA07
*** 2022-11-23: You have to update ADC16 list -- Nuevo Leon is missing
*** Produce conversion table from CA16 to CA07 by crop
use "`amca_adc_int'", clear

preserve
keep name
duplicates drop name, force 
sort name
quietly levelsof name, local(cltvos)
restore

clear all
set obs 0
gen name = ""
save "`ca16_tbl_by_n'", replace

foreach cul in `cltvos' {
	use "`ca07_a_ca16'", clear
	display "CROP: `cul'"
	gen name = "`cul'"
	quietly append using "`ca16_tbl_by_n'"
	sleep 1
	quietly save "`ca16_tbl_by_n'", replace
}

rename adc adc07
rename adc16 adc
*** Subset down to only crop-ADC pairs existant in data
*** Merge works for all states except for all observations in Nuevo Leon, need to check ADC table
merge m:1 adc name using "`amca_adc_int'", keep(2 3) keepusing(adc name) nogen
rename adc adc16
rename adc07 adc

*** 2022-11-23: TEMPORARY, REPLACE ONCE NUEVO LEON ISSUE IS FIXED
replace adc = adc16 if missing(adc)
replace adc07share = 1 if missing(adc07share)
*** 2022-11-23: TEMPORARY, REPLACE ONCE NUEVO LEON ISSUE IS FIXED
save "`ca16_tbl_sub'", replace

use "`ca16_tbl_sub'", clear
*** Merge with production/yield data from CA07
*** MERGE == 1, 348,656
*** MERGE == 2, 509,860
*** MERGE == 3, 234,623

merge m:1 adc name using "`rendi_adc'", keep(1 3)
gen has_data = 0
replace has_data = 1 if _merge == 3 
drop _merge

*** Since so many matches missing, reweight each by only remaining matches
gen rwht_share = adc07share * has_data
egen data_total_share = total(rwht_share), by(adc16 name) 
replace adc07share = adc07share/data_total_share
drop if has_data == 0
gen round_share = round(adc07share,0.0000001)
drop has_data rwht_share data_total_share adc07share
rename round_share adc07share

foreach var in sup_sem sup_cos Q sup_irrig sup_rf yield yield_rf yield_irrig {
	replace `var' = `var'*adc07share
}

collapse (sum) sup_sem sup_cos Q sup_irrig sup_rf, by(adc16 name)
rename sup* sup*_07
rename Q Q_07
gen yield_07       = Q_07/sup_sem
gen yield_irrig_07 = Q_07/sup_irrig_07
gen yield_rf_07    = Q_07/sup_rf_07
rename adc16 adc 
save "`ca07_y_wght'", replace

gen ageb  = substr(adc,1,10)
collapse (sum) sup* Q_07, by(ageb name)
gen yield_07       = Q_07/sup_sem
gen yield_irrig_07 = Q_07/sup_irrig_07
gen yield_rf_07    = Q_07/sup_rf_07
save "`ca07_y_wageb'", replace
*** This step is to weight Q by Q by ADC in CA07
*/

/*
**** Checkpoint 1: Now go back to checkpoint 1
use "`amca_adc_int'", clear
merge 1:1 adc  name using "`ca07_y_wght'", keep(1 3)
gen has_noca07data = 0
replace has_noca07data = 1 if _merge != 3
drop _merge

*** Many missing observations from CA2007 in terms of ADCs, so fill in remainder
*** (i.e. whatever the mun total is from CA07-SIAP total) with SIAP muncode level
*** total, weighted either equally or by GP status. Then use output quantities to
*** weight total production in 2016 from SIAP again 
merge m:1 muncode name using "`siap2007'", keep(1 3) nogen
egen Q_est_adc_tot_07 = total(Q_07), by(muncode name)
egen sup_est_adc_tot_07 = total(sup_sem_07), by(muncode name)

*** determine CA07-SIAP total remainder
gen Q_remaining_07 = q_2007_siap-Q_est_adc_tot_07
replace Q_remaining_07 = 0 if Q_remaining_07 < 0
gen ha_remaining_07 = ha_planted_07siap-sup_est_adc_tot_07
replace ha_remaining_07 = 0 if ha_remaining_07 < 0

**** Allocate remainder based on ha planted in 2016 AMCA
gen sup_sem_no_ca07 = has_noca07data*sup_sem
gen sup_gp_no_ca07 = has_noca07data*sup_gp
gen sup_nogp_no_ca07 = has_noca07data*sup_not_gp

*** Weight yields by land harvested alone
egen mun_total_sup_crop_noca07   = total(sup_sem_no_ca07), by(muncode name)
gen sup_share_noca07      = sup_sem_no_ca07/mun_total_sup_crop_noca07
gen q_equiv_sh_noca07     = sup_share_noca07*Q_remaining_07
gen ha_equiv_sh_noca07    = sup_share_noca07*ha_remaining_07
drop mun_total_sup_crop_noca07 sup_share_noca07 sup_sem_no_ca07

*** Weight yields by land harvested, counting the land of grand producers as 50% more important
gen sup_gpw_sem_noca07           = 1.5*sup_gp_no_ca07 + sup_nogp_no_ca07
egen mun_tot_sup_crop_gpw_noca07 = total(sup_gpw_sem_noca07), by(muncode name)
gen sup_gpw_share_noca07         = sup_gpw_sem_noca07/mun_tot_sup_crop_gpw_noca07
gen q_gpw_share_noca07           = sup_gpw_share_noca07*Q_remaining_07
gen ha_gpw_share_noca07          = sup_gpw_share_noca07*ha_remaining_07

drop sup_gpw_sem_noca07 mun_tot_sup_crop_gpw_noca07 sup_gpw_share_noca07 sup_gp sup_gp_no_ca07 sup_not_gp sup_nogp_no_ca07 ha_remaining_07
drop sup_cos_07 sup_irrig_07 sup_rf_07 yield_07 yield_irrig_07 yield_rf_07 has_noca07data ha_planted_irrig_07siap ha_planted_rf_07siap Q_remaining_07

*** Now have Q produced at ADC by crop, where I fill in missing Q with imputation
gen Q_07_equiv = Q_07
replace Q_07_equiv = q_equiv_sh_noca07 if missing(Q_07)
gen Q_07_gpw = Q_07
replace Q_07_gpw = q_gpw_share_noca07 if missing(Q_07)
gen ha_07_equiv = sup_sem_07
replace ha_07_equiv = ha_equiv_sh_noca07 if missing(sup_sem_07)
gen ha_07_gpw = sup_sem_07
replace ha_07_gpw = ha_gpw_share_noca07 if missing(sup_sem_07)


**** Compute weight to avg. yield based on comp. to med in 2007
gen yield_07_equiv = Q_07_equiv/ha_07_equiv
gen yield_07_gpw = Q_07_gpw/ha_07_gpw

egen yield_07_equiv_med = median(yield_07_equiv), by(muncode name)
egen yield_07_gpw_med = median(yield_07_gpw), by(muncode name)
gen yield_07_equiv_scale = yield_07_equiv/yield_07_equiv_med
gen yield_07_gpw_scale = yield_07_gpw/yield_07_gpw_med
replace yield_07_equiv_scale = 1 if missing(yield_07_equiv_scale)
replace yield_07_gpw_scale = 1 if missing(yield_07_gpw_scale)

drop Q_07 q_equiv_sh_noca07 q_gpw_share_noca07 sup_sem_07 ha_equiv_sh_noca07 ha_gpw_share_noca07
drop ha_planted_07siap sup_est_adc_tot_07 ha_07_equiv ha_07_gpw
drop yield_07_equiv yield_07_gpw yield_07_equiv_med yield_07_gpw_med


*** Compute mun weight by crop in 2007
egen Q_07_tot_mun = rowmax(q_2007_siap Q_est_adc_tot_07)
*** 2022-11-23: These below are equivalent to above
*egen Q_07_equiv_tot = total(Q_07_equiv), by(muncode name)
*egen Q_07_gpw_tot = total(Q_07_gpw), by(muncode name)
gen Q_07_equiv_share = Q_07_equiv/Q_07_tot_mun
gen Q_07_gpw_share = Q_07_gpw/Q_07_tot_mun
gen q_16_equiv = Q_07_equiv_share*q
gen q_16_gpw = Q_07_gpw_share*q
gen yield_07w_equiv = q_16_equiv/sup_sem
gen yield_07w_gpw = q_16_gpw/sup_sem
gen yield_07yw_equiv = yield_equiv*yield_07_equiv_scale
gen yield_07yw_gpw = yield_gpw*yield_07_gpw_scale

drop yield_07_equiv_scale yield_07_gpw_scale
drop q q_2007_siap Q_est_adc_tot_07 Q_07_equiv Q_07_gpw Q_07_tot_mun Q_07_equiv_share Q_07_gpw_share q_16_equiv q_16_gpw

save "`amca_adc'", replace
*/
*** DO exact same for AGEB

use "`amca_1_clean'", clear
append using "`amca_2_clean'"

*** Compute indicators for if rainfed or not
gen sup_irrig  = sup_sem*irrig
gen sup_rf     = sup_sem*(1-irrig)
gen sup_gp     = sup_sem*gp
gen sup_not_gp = sup_sem*(1-gp)


*** Gen ind for num of terrenos
gen num_terrenos = 1

*** Replace name for non matched crops
replace name = "Not specified" if missing(name)

*** Collapse down to ADC level
collapse (sum) super_cart sup_sem sup_irrig sup_rf sup_gp sup_not_gp num_terrenos, by(cve_ent muncode ageb name greenhouse)

merge m:1 muncode name using "`siap2016'", keep(1 3) nogen

*** Weight yields by land harvested alone
egen mun_total_sup_crop   = total(sup_sem), by(muncode name)
gen sup_share             = sup_sem/mun_total_sup_crop
gen q_share_equiv         = sup_share*q
gen yield_equiv           = q_share_equiv/sup_sem
drop mun_total_sup_crop sup_share q_share_equiv

*** Weight yields by land harvested, counting the land of grand producers as 50% more important
gen sup_gpw_sem           = 1.5*sup_gp + sup_not_gp
egen mun_tot_sup_crop_gpw = total(sup_gpw_sem), by(muncode name)
gen sup_gpw_share         = sup_gpw_sem/mun_tot_sup_crop_gpw
gen q_share_gpw           = sup_gpw_share*q
gen yield_gpw             = q_share_gpw/sup_sem
drop sup_gpw_sem mun_tot_sup_crop_gpw sup_gpw_share q_share_gpw


*** Keep only if not greenhouse
keep if greenhouse == 0 
drop greenhouse

save "`amca_agebint'", replace
use "`amca_agebint'", clear
merge 1:1 ageb  name using "`ca07_y_wageb'", keep(1 3)
gen has_noca07data = 0
replace has_noca07data = 1 if _merge != 3
drop _merge

*** Many missing observations from CA2007 in terms of AGEBs, so fill in remainder
*** (i.e. whatever the mun total is from CA07-SIAP total) with SIAP muncode level
*** total, weighted either equally or by GP status. Then use output quantities to
*** weight total production in 2016 from SIAP again 
merge m:1 muncode name using "`siap2007'", keep(1 3) nogen
egen Q_est_AGEB_tot_07 = total(Q_07), by(muncode name)
egen sup_est_AGEB_tot_07 = total(sup_sem_07), by(muncode name)

*** determine CA07-SIAP total remainder
gen Q_remaining_07 = q_2007_siap-Q_est_AGEB_tot_07
replace Q_remaining_07 = 0 if Q_remaining_07 < 0
gen ha_remaining_07 = ha_planted_07siap-sup_est_AGEB_tot_07
replace ha_remaining_07 = 0 if ha_remaining_07 < 0

**** Allocate remainder based on ha planted in 2016 AMCA
gen sup_sem_no_ca07 = has_noca07data*sup_sem
gen sup_gp_no_ca07 = has_noca07data*sup_gp
gen sup_nogp_no_ca07 = has_noca07data*sup_not_gp

*** Weight yields by land harvested alone
egen mun_total_sup_crop_noca07   = total(sup_sem_no_ca07), by(muncode name)
gen sup_share_noca07      = sup_sem_no_ca07/mun_total_sup_crop_noca07
gen q_equiv_sh_noca07     = sup_share_noca07*Q_remaining_07
gen ha_equiv_sh_noca07    = sup_share_noca07*ha_remaining_07
drop mun_total_sup_crop_noca07 sup_share_noca07 sup_sem_no_ca07

*** Weight yields by land harvested, counting the land of grand producers as 50% more important
gen sup_gpw_sem_noca07           = 1.5*sup_gp_no_ca07 + sup_nogp_no_ca07
egen mun_tot_sup_crop_gpw_noca07 = total(sup_gpw_sem_noca07), by(muncode name)
gen sup_gpw_share_noca07         = sup_gpw_sem_noca07/mun_tot_sup_crop_gpw_noca07
gen q_gpw_share_noca07           = sup_gpw_share_noca07*Q_remaining_07
gen ha_gpw_share_noca07          = sup_gpw_share_noca07*ha_remaining_07

drop sup_gpw_sem_noca07 mun_tot_sup_crop_gpw_noca07 sup_gpw_share_noca07 sup_gp sup_gp_no_ca07 sup_not_gp sup_nogp_no_ca07 ha_remaining_07
drop sup_cos_07 sup_irrig_07 sup_rf_07 yield_07 yield_irrig_07 yield_rf_07 has_noca07data ha_planted_irrig_07siap ha_planted_rf_07siap Q_remaining_07

*** Now have Q produced at AGEB by crop, where I fill in missing Q with imputation
gen Q_07_equiv = Q_07
replace Q_07_equiv = q_equiv_sh_noca07 if missing(Q_07)
gen Q_07_gpw = Q_07
replace Q_07_gpw = q_gpw_share_noca07 if missing(Q_07)
gen ha_07_equiv = sup_sem_07
replace ha_07_equiv = ha_equiv_sh_noca07 if missing(sup_sem_07)
gen ha_07_gpw = sup_sem_07
replace ha_07_gpw = ha_gpw_share_noca07 if missing(sup_sem_07)


**** Compute weight to avg. yield based on comp. to med in 2007
gen yield_07_equiv = Q_07_equiv/ha_07_equiv
gen yield_07_gpw = Q_07_gpw/ha_07_gpw

egen yield_07_equiv_med = median(yield_07_equiv), by(muncode name)
egen yield_07_gpw_med = median(yield_07_gpw), by(muncode name)
gen yield_07_equiv_scale = yield_07_equiv/yield_07_equiv_med
gen yield_07_gpw_scale = yield_07_gpw/yield_07_gpw_med
replace yield_07_equiv_scale = 1 if missing(yield_07_equiv_scale)
replace yield_07_gpw_scale = 1 if missing(yield_07_gpw_scale)

drop Q_07 q_equiv_sh_noca07 q_gpw_share_noca07 sup_sem_07 ha_equiv_sh_noca07 ha_gpw_share_noca07
drop ha_planted_07siap sup_est_AGEB_tot_07 ha_07_equiv ha_07_gpw
drop yield_07_equiv yield_07_gpw yield_07_equiv_med yield_07_gpw_med


*** Compute mun weight by crop in 2007
egen Q_07_tot_mun = rowmax(q_2007_siap Q_est_AGEB_tot_07)
*** 2022-11-23: These below are equivalent to above
*egen Q_07_equiv_tot = total(Q_07_equiv), by(muncode name)
*egen Q_07_gpw_tot = total(Q_07_gpw), by(muncode name)
gen Q_07_equiv_share = Q_07_equiv/Q_07_tot_mun
gen Q_07_gpw_share = Q_07_gpw/Q_07_tot_mun
gen q_16_equiv = Q_07_equiv_share*q
gen q_16_gpw = Q_07_gpw_share*q
gen yield_07w_equiv = q_16_equiv/sup_sem
gen yield_07w_gpw = q_16_gpw/sup_sem
gen yield_07yw_equiv = yield_equiv*yield_07_equiv_scale
gen yield_07yw_gpw = yield_gpw*yield_07_gpw_scale

drop yield_07_equiv_scale yield_07_gpw_scale
drop q q_2007_siap Q_est_AGEB_tot_07 Q_07_equiv Q_07_gpw Q_07_tot_mun Q_07_equiv_share Q_07_gpw_share q_16_equiv q_16_gpw

save "`amca_ageb'", replace
