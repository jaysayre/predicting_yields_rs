/* ca2022_land_use.do -- jsayre@ucdavis.edu

Written to clean the various TRD files of the Censo Agropecuaria 2022, and
assemble geographic data about land use and agricultural yields.

*/

clear all

set locale_ui en
set more off

local date         =  "2025-09-29"

*** Directorios
local wrk_dir      =  "Z:\Procesamiento\Trabajo\"
local ca22_dir     =  "`wrk_dir'Censo Agropecuario\Original\"
local ca22_cln_dir =  "`wrk_dir'Censo Agropecuario\Cleaned\"

local intermed_dir =  "`wrk_dir'Intermediates\"
local tbl_matcheo  =  "`wrk_dir'Matcheo_UP_ADC_CA\"
local results_dir  =  "Z:\Resultados\LM2304-CA22-`date'-superficie\"
capture mkdir "`results_dir'"

*** Insumos
local ag_22        =  "`ca22_dir'TRD_AGRICULTURA_CIELO_ABIERTO.dta"
local greenhouse22 =  "`ca22_dir'TRD_AGRICULTURA_PROTEGIDA.dta"
local instal22     =  "`ca22_dir'TRD_INST_AGRI_PROT.dta"
local labor_22     =  "`ca22_dir'TRD_MANO_OBRA.dta"
local problems     =  "`ca22_dir'TRD_PROBLEMATICA.dta"
local land_size22  =  "`ca22_dir'TRD_SUPERFICIES_UP.dta"
local tech         =  "`ca22_dir'TRD_TEC_AGRI_CA.dta"
local tech_grnhous =  "`ca22_dir'TRD_TEC_AGRI_PROT.dta"
local cultivos     =  "`wrk_dir'cultivo_a_nombre.dta"

*** 2022- ADC AGEB MUN REFERENCIAS
local loc_clean    =  "`tbl_matcheo'location.dta"
*** 2007 - ADC AGEB MUN REFERENCIAS
local loc_cln_ca07 =  "`tbl_matcheo'location_22_mtch_to_07.dta"

*** Intermediarios
local ag_clean     =  "`ca22_cln_dir'ag.dta"
local ag_id_rain   =  "`ca22_cln_dir'ag_w_id_precip.dta"
local ag_grnhs     =  "`ca22_cln_dir'ag_greenhouse.dta"
local instal_clean =  "`ca22_cln_dir'installations.dta"
local labor_clean  =  "`ca22_cln_dir'labor.dta"
local prob_clean   =  "`ca22_cln_dir'problems.dta"
local land_clean   =  "`ca22_cln_dir'land_size.dta"
local tech_clean   =  "`ca22_cln_dir'tech.dta"
local tech_gh_cln  =  "`ca22_cln_dir'tech_greenhouse.dta"

*** Rendimiento
*** 2022 ADC
local mun_land_use =  "`results_dir'mun_land_use_ca22.dta"
local ageb_lnd_use =  "`results_dir'ageb_land_use_ca22.dta"
local adc_land_use =  "`results_dir'adc_land_use_ca22.dta"
local ageb_lnd_one =  "`results_dir'ageb_land_use_1adc_ca22.dta"
local adc_lnd_one  =  "`results_dir'adc_land_use_1adc_ca22.dta"
local mun_land_szn =  "`results_dir'mun_land_szn_ca22.dta"
local ageb_lnd_szn =  "`results_dir'ageb_land_szn_ca22.dta"
local adc_land_szn =  "`results_dir'adc_land_szn_ca22.dta"
local ageb_szn_one =  "`results_dir'ageb_land_szn_1adc_ca22.dta"
local adc_szn_one  =  "`results_dir'adc_land_szn_1adc_ca22.dta"
*** 2007 ADC
local mun_l07_use  =  "`results_dir'mun_land_use_ca22_adc07.dta"
local ageb_l07_use =  "`results_dir'ageb_land_use_ca22_adc07.dta"
local adc_l07_use  =  "`results_dir'adc_land_use_ca22_adc07.dta"
local mun_l07_szn  =  "`results_dir'mun_land_szn_ca22_adc07.dta"
local ageb_l07_szn =  "`results_dir'ageb_land_szn_ca22_adc07.dta"
local adc_l07_szn  =  "`results_dir'adc_land_szn_ca22_adc07.dta"


/*
************************************************************************************************************************************************************

****** LOCATION OF FARM UNIT (CLEANED IN LM1325)
******
*2022*
******

************************************************************************************************************************************************************

************************************************************************************************************************************************************

**** AG CIELO ABIERTO
use "`ag_22'", clear

drop D_R

gen cultivo = upper(ustrto(ustrnormalize(aa111_02_c, "nfd"), "ascii", 2))
drop aa111_02_c

replace cultivo = "CANA DE AZUCAR"         if cultivo == "CAA DE AZUCAR"
replace cultivo = "PINA"                   if cultivo == "PIA"
replace cultivo = "ARROZ PALAY"            if cultivo == "ARROZ"
replace cultivo = "CHILE VERDE"            if cultivo == "CHILE"
replace cultivo = "MAIZ GRANO"             if cultivo == "MAIZ GRANO AMARILLO"
replace cultivo = "MAIZ GRANO"             if cultivo == "MAIZ GRANO BLANCO"
replace cultivo = "OTRAS ESPECIES"         if cultivo == "OTROS CULTIVOS"	
replace cultivo = "PASTOS"                 if cultivo == "PASTO CULTIVADO"
replace cultivo = "SORGO FORRAJERO VERDE"  if cultivo == "SORGO FORRAJERO"
replace cultivo = "TOMATE ROJO (JITOMATE)" if cultivo == "JITOMATE (TOMATE ROJO)"
replace cultivo = "TRIGO GRANO"            if cultivo == "TRIGO FORRAJERO"
replace cultivo = "LIMON"                  if cultivo == "LIMN"
replace cultivo = "TOMATE VERDE"           if cultivo == "TOMATE DE CASCARA (TOMATILLO)"
replace cultivo = "CAFE CEREZA"            if cultivo == "CAFU"
replace cultivo = "CALABAZA"               if cultivo == "CALABAZA/CALABACITA"
replace cultivo = "ALFALFA VERDE"          if cultivo == "ALFALFA"
replace cultivo = "ALGODON HUESO"          if cultivo == "ALGODON"
replace cultivo = "COCO DE ACEITE"         if cultivo == "COCO"
replace cultivo = "NOPALITOS"              if cultivo == "NOPAL VERDURA"
replace cultivo = "SANDIA"                 if cultivo == "SANDYA"
replace cultivo = "PLATANO"                if cultivo == "PLTANO"
replace cultivo = "ESPARRAGO"              if cultivo == "ESPRRAGO"

*** Checked and all merge okay
merge m:1 cultivo using "`cultivos'", keep(1 3) nogen
rename cultivo nombre_cul  

rename aa111_05 irrig
replace irrig = 0 if irrig == 2
rename aa111_04_m plant_month
destring plant_month, replace
rename aa111_04_a plant_year
destring plant_year, replace
rename aa111_03_n land_input
rename aa111_13_n land_output 
rename aa111_17_n vol_output
rename da111_n q_seed
rename da112_n q_self_cons
rename da113_n q_animal
rename da115_n q_sold
rename da114_n q_lost
rename da114_01	number_units_prod_losses

rename da114_02_c cause_losses
gen losses_almacenamiento = 1     if cause_losses == "Da±os durante el almacenamiento"
gen losses_packing = 1            if cause_losses == "Da±os durante el empaque"
gen losses_transport = 1          if cause_losses == "Da±os durante el transporte"
gen losses_other = 1              if cause_losses == "Da±os por otras causas"
gen losses_pests = 1              if cause_losses == "Da±os por plagas o enfermedades"
gen losses_prod_selec = 1         if cause_losses == "Da±os por seleccion del producto"
replace losses_almacenamiento = 0 if missing(losses_almacenamiento)
replace losses_packing = 0        if missing(losses_packing)
replace losses_transport = 0      if missing(losses_transport)
replace losses_other = 0          if missing(losses_other)
replace losses_pests = 0          if missing(losses_pests)
replace losses_prod_selec = 0     if missing(losses_prod_selec)
drop cause_losses
gen greenhouse = 0
save "`ag_clean'", replace


************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************

************************************************************************************************************************************************************
****** GREENHOUSE
******
*2022*
******

use "`greenhouse22'", clear
drop D_R

gen cultivo = upper(ab112_02_c)
drop ab112_02_c
replace cultivo = "CHILE VERDE"            if cultivo == "CHILE"
replace cultivo = "TOMATE ROJO (JITOMATE)" if cultivo == "JITOMATE (TOMATE ROJO)"
replace cultivo = "OTRAS ESPECIES"         if cultivo == "OTROS CULTIVOS"	
*** Checked and all merge okay
merge m:1 cultivo using "`cultivos'", keep(1 3) nogen
drop cultivo
label variable ac111_02_c "Nombre del tipo de estructura o construcción donde se produjo el cultivo"
*** Takes values "CASA SOMBRA" "INVERNADERO" "MACROTUNEL" "MALLA SOMBRA" "MICROTUNEL" 
*** "OTRA ESTRUCTURA O CONSTRUCCION" "PABELLON" "TECHO SOMBRA" "VIVERO"
label variable ab111_02_c "Nombre del tipo de producto obtenido"
*** Takes values "ESPECIES FORESTALES" "ESPECIES O PLANTAS AROMATICAS" "ESQUEJES O PLANTULAS"
***              "HORTALIZAS O FRUTAS" "OTRO TIPO DE PRODUCTO"         "PLANTAS ORNAMENTALES"
***              "ESPECIES O PLANTAS AROMATICAS" is mostly other species
*** Just want to count ag production here
keep if ab111_02_c == "HORTALIZAS O FRUTAS"
rename ab113_01_n land_input
gen land_output = land_input
rename ab114_01_n vol_output

drop ac111_02_c ab111_02_c
gen greenhouse = 1 
gen irrig = 1

save "`ag_grnhs'", replace
************************************************************************************************************************************************************

************************************************************************************************************************************************************

****** INSTALLATIONS IN FARM UNIT

use "`instal22'", clear

drop D_R
*** Clean large (grand) producer variable
gen gp_2 = 1 if tipo_up == "1"
replace gp_2 = 0 if tipo_up == "2"
drop tipo_up

*** Here, I'm guessing 2 means no, 0 no info
rename ac112_07 distributor
label var distributor "Has bodega, almacén, centro de acopio"
replace distributor = 0 if distributor == 2

rename ac112_08 storage
label var storage "Has storage facilities"
replace storage = 0 if storage == 2

rename ac112_02 shredder
label var shredder "Has a shredder"
replace shredder = 0 if shredder == 2

rename ac112_03 dehydrator
label var dehydrator "Has a dehydrator"
replace dehydrator = 0 if dehydrator == 2

rename ac112_04 packer
label var packer "Is a packer"
replace packer = 0 if packer == 2

rename ac112_05 selector
label var selector "Is a selector"
replace selector = 0 if selector == 2

rename ac112_06 cold_storage
label var cold_storage "Has cold storage"
replace cold_storage = 0 if cold_storage == 2

rename ac112_99 other_installation
label var other_installation "Has other installation"
replace other_installation = 0 if other_installation == 2

save "`instal_clean'", replace
************************************************************************************************************************************************************

************************************************************************************************************************************************************
************************************************************************************************************************************************************
******** LABOR ************
use "`labor_22'", clear
drop D_R

*** Clean large (grand) producer variable
gen gp_6 = 1 if tipo_up == "1"
replace gp_6 = 0 if tipo_up == "2"
drop tipo_up


label variable mo115_02	"Cantidad de jornaleros contratados"
rename mo115_02 L_jornaleros
label variable mo115_03	"Cantidad de mujeres jornaleras contratadas"
rename mo115_03 L_jornaleros_female
*** I've checked that L_jornaleros-L_jornaleros_female seems to be L_male
gen L_jornaleros_male = L_jornaleros-L_jornaleros_female
label variable mo115_2_21 "Cantidad de jornaleros contratados menores de 16 años"
label variable mo115_2_31 "Cantidad de mujeres jornaleras contratadas menores de 16 años"
label variable mo115_2_22 "Cantidad de jornaleros contratados de 16 a 40 años"
label variable mo115_2_32 "Cantidad de mujeres jornaleras contratadas de 16 a 40 años"
label variable mo115_2_23 "Cantidad de jornaleros contratados de 41 años o más"
label variable mo115_2_33 "Cantidad de mujeres jornaleras contratadas de 41 años o más"
*** Place of origin only blank for observations without jornaleros
label variable mo115_2_1c "Nombre del lugar de procedencia de los jornaleros"
gen jornaleros_from_nearby = 1 if mo115_2_1c == "LOS ALREDEDORES O ZONAS CERCANAS"
replace jornaleros_from_nearby = 0 if missing(jornaleros_from_nearby)
gen jornaleros_from_further = 1 if mo115_2_1c == "OTRA PARTE DEL MISMO ESTADO"
replace jornaleros_from_further = 0 if missing(jornaleros_from_further)
gen jornaleros_from_other_state = 1 if mo115_2_1c == "OTRO ESTADO O ENTIDAD FEDERATIVA"
replace jornaleros_from_other_state = 0 if missing(jornaleros_from_other_state)
gen jornaleros_extranjeros = 1 if mo115_2_1c == "OTRO PAIS"
replace jornaleros_extranjeros = 0 if missing(jornaleros_extranjeros)
label variable mo115_214c "Nombre del país de procedencia de los jornaleros"
*** Country of origin only displays if mo115_2_1c == "OTRO PAIS"
*** Basically all except for 59 observations are from "GUATEMALA"


*** CAUTION: If jornaleros not hired, then often times variables about hours worked, etc. are 0
*** Not blank, as they should be 
label variable mo115_04	"Promedio de horas trabajadas por los jornaleros hombres al día"
rename mo115_04 avg_hours_jornaleros_male
replace avg_hours_jornaleros_male = . if L_jornaleros_male == 0
label variable mo115_06	"Promedio de horas trabajadas por los jornaleros mujeres al día"
rename mo115_06 avg_hours_jornaleros_female
replace avg_hours_jornaleros_female = . if L_jornaleros_female == 0
label variable mo115_05	"Promedio de días que contrata a los jornaleros hombres"
rename mo115_05 avg_days_jornaleros_male
replace avg_days_jornaleros_male = . if L_jornaleros_male == 0
label variable mo115_07	"Promedio de días que contrata a los jornaleros mujeres"
rename mo115_07 avg_days_jornaleros_female
replace avg_days_jornaleros_female = . if L_jornaleros_female == 0
*** Payments dont correlate too much by avg_days or avg_hours so Im assuming this is per hour or day (mean 227 pesos)
label variable mo119	"Cantidad pagada por jornal a los hombres"
rename mo119 payment_jornales_male
replace payment_jornales_male = . if L_jornaleros_male == 0
label variable mo119_01	"Cantidad pagada por jornal a las mujeres"
rename mo119_01 payment_jornales_female
replace payment_jornales_female = . if L_jornaleros_female == 0
label variable mo111	"Cantidad de familiares del productor que participaron"
rename mo111 L_family
label variable mo112_01	"Cantidad de mujeres familiares del productor"
rename mo112_01 L_family_female
*** I've checked that L_family-L_family_female=L_family_male
gen L_family_male = L_family-L_family_female
label variable mo122	"Cantidad de familiares del productor que recibieron sueldo o salario"
rename mo122 L_family_wage
gen L_family_nowage = L_family-L_family_wage
label variable mo122_01	"Cantidad de familiares mujeres que recibieron sueldo o salario"
rename mo122 L_family_wage_female
gen L_family_wage_male = L_family_wage-L_family_wage_female
*** Wage earning family members not counted in L_perm
label variable mo114	"Cantidad de personas que se contrataron por 6 meses o más"
rename mo114 L_perm
label variable mo114_01	"Cantidad de mujeres que se contrataron por 6 meses o más"
rename mo114_01 L_perm_female
label variable mo115	"Cantidad de personas que se contrataron por menos de 6 meses, sin contar a los jornaleros ni familiares"
rename mo115 L_temp
label variable mo115_01	"Cantidad de mujeres que se contrataron por menos de 6 meses, sin contar a los jornaleros"
rename mo115_01  L_temp_female
label variable mo118	"Unidad de producción donde el productor participa en las labores agropecuarias"
rename mo118 L_owner
*** Not sure I fully understand this variable, either takes values of 1 or 2
replace L_owner = 0 if L_owner == 2

drop mo1*
save "`labor_clean'", replace

************************************************************************************************************************************************************

****** PROBLEMS
******

use "`problems'", clear
drop D_R

*** Clean large (grand) producer variable
gen gp_3 = 1 if tipo_up == "1"
replace gp_3 = 0 if tipo_up == "2"
drop tipo_up

*** For some reason only the last variables have entries for 2
foreach varname in "se116" "pp111_01" "pp111_04" "pp111_06" "pp111_08" "pp111_11" "pp111_12" "pp111_13" "pp111_18" "pp111_19" "pp111_20" "pp111_25" "pp111_26" "pp111_99" {
	replace `varname' = 0 if `varname' == 2
}

save "`prob_clean'", replace

************************************************************************************************************************************************************

************************************************************************************************************************************************************

****** LAND SIZE
******
******

use "`land_size22'", clear
drop D_R

*** Clean large (grand) producer variable
gen gp = 1 if tipo_up == "1"
replace gp = 0 if tipo_up == "2"
drop tipo_up

gen num_plots_owned = up110
label var num_plots_owned "Total plots owned by producer"
gen land = up120_norm
label var land "Total land in farm unit"

keep id_ca22_up num_plots_owned land gp
 
save "`land_clean'", replace

************************************************************************************************************************************************************

************************************************************************************************************************************************************

****** TECHNOLOGY
******
*2022*
******

use "`tech'", clear
drop D_R

*** Clean large (grand) producer variable
gen gp_4 = 1 if tipo_up == "1"
replace gp_4 = 0 if tipo_up == "2"
drop tipo_up

*** 2 == no, 0 == unsure or NA
foreach varname in "at111_14_1" "at111_14_2" "at111_14_3" "at111_14_4" "at111_19" "at111_23" "at112_04" "at112_06" "at111_27_1" "at111_27_2" "at111_28_1" "at111_28_2" "at111_30_1" "at111_30_2" "at112_08" "at112_02" "at112_03" "at111_32" "at112_09" "at112_19" "at112_05" "at112_10" "at112_99" {
	replace `varname' = 0 if `varname' == 2
}

rename at111_14_1 native_seed
label var native_seed "Production unit using native criolla seed"
rename at111_14_2 impr_seed
label var impr_seed "Production unit using improved seed"
rename at111_14_3 cert_seed
label var cert_seed "Production unit using certified seed"
rename at111_19 chem_fert
label var chem_fert "Production unit using chemical fertilizers"
rename at111_23 nat_fert
label var nat_fert "Production unit using natural fertilizers"
rename at111_14_4 gen_seed
label var gen_seed "Used genetically modified or transgenic seed"

gen uses_fert = 0
replace uses_fert = 1 if chem_fert == 1
replace uses_fert = 1 if nat_fert == 1

rename at111_27_1 chem_herb
rename at111_27_2 org_herb
rename at111_28_1 chem_insect
rename at111_28_2 org_insect
rename at111_30_1 chem_fung
rename at111_30_2 org_fung

gen herb_insect = 0
foreach y in "chem_herb" "org_herb" "chem_insect" "org_insect" {
	replace herb_insect = 1 if `y' == 1
}

rename at112_02 planter
label var planter "Production unit using planting machinery"
rename at112_03 thresher_harvester
label var thresher_harvester "Production unit using thresher or harvester"
rename at111_32 stripper
label var stripper "Production unit using stripper/desgranadoras" 
rename at112_04 animals
label var animals "Production unit using draft animals or yoke for agricultural work"
rename at112_09 bio_pest_control
label var bio_pest_control "Performed biological control of pests"
rename at112_11 graft_trees
label var graft_trees "Production unit using grafted trees"
rename at112_06 crop_rotation
label var crop_rotation "Performed crop rotation to improve soil"
rename at112_08 pruning
label var pruning "Performed pruning"
rename at112_05 soil_cons
label var soil_cons "Used labranza for soil conservation"
rename at112_10 tech_ass
label var tech_ass "Received technical assistance"
rename at112_99 other_tech
label var other_tech "Surface enabled with other technology"
rename at112_19 acolchado
label var acolchado "Production unit using acolchado"


save "`tech_clean'", replace

************************************************************************************************************************************************************

************************************************************************************************************************************************************

****** TECHNOLOGY -- PROTECTED AG
******
*2022*
******

use "`tech_grnhous'", clear
drop D_R

*** Clean large (grand) producer variable
gen gp_5 = 1 if tipo_up == "1"
replace gp_5 = 0 if tipo_up == "2"
drop tipo_up

*** 2 == no, 0 == unsure or NA
foreach varname in "at113_06" "at113_08" "at113_09" "at113_11" "at113_12" "at113_13" "at113_14" "at113_99" {
	replace `varname' = 0 if `varname' == 2
}

rename at113_06 ventilation
rename at113_08 air_extraction
rename at113_09 thermal_screen
rename at113_11 humid_wall
rename at113_12 heating
rename at113_13 controlled_polinization
rename at113_14 tech_ass_greenhouse
rename at113_99 other_tech_greenhouse

keep id_ca22_up tech_ass_greenhouse other_tech_greenhouse gp_5

save "`tech_gh_cln'", replace

************************************************************************************************************************************************************
*/


************************************************************************************************************************************************************
****************************************************** Produce land use for all crops/growing seasons ******************************************************
************************************************************************************************************************************************************


use "`ag_clean'", clear
append using "`ag_grnhs'"

merge m:1 id_ca22_up using "`land_clean'", keep(3) nogen

*** CA07 ADC
* Keep only open-field
keep if greenhouse==0
drop greenhouse

merge m:1 id_ca22_up using "`loc_cln_ca07'", keep(3) nogen keepusing(id_ca22_up cve_geo* share_adc07_match*)

keep id_ca22_up cve_geo* share_adc07_match* land* vol* gp irrig name plant_month

* Reshape wide -> long over the 20 potential matches
gen long obsid = _n
reshape long cve_geo share_adc07_match, i(obsid id_ca22_up land_input land_output vol_output gp irrig name plant_month) j(match_rank)

* Keep only valid matches
drop if missing(cve_geo) | missing(share_adc07_match) | share_adc07_match<=0
rename cve_geo adc

* Derive 2007 geography (state and municipality) from the 2007 ADC code
gen str2 cve_ent = substr(adc,1,2)
gen str5 muncode = substr(adc,1,5)
gen str9 cve_ageb = substr(adc,1,9)

* Weight the core measures by the match share
gen land_input_w  = land_input  * share_adc07_match
gen land_output_w = land_output * share_adc07_match
gen vol_output_w  = vol_output  * share_adc07_match

* Re-create splits OFF THE WEIGHTED VALUES
gen land_irr_input   = land_input_w   if irrig==1
gen land_rf_input    = land_input_w   if irrig==0
gen land_irr_output  = land_output_w  if irrig==1
gen land_rf_output   = land_output_w  if irrig==0
gen vol_irr_output   = vol_output_w   if irrig==1
gen vol_rf_output    = vol_output_w   if irrig==0

gen land_ngp_input   = land_input_w   if gp==0
gen land_gp_input    = land_input_w   if gp==1
gen land_ngp_output  = land_output_w  if gp==0
gen land_gp_output   = land_output_w  if gp==1
gen vol_ngp_output   = vol_output_w   if gp==0
gen vol_gp_output    = vol_output_w   if gp==1

* Use the weighted totals under the usual names for downstream code
drop land_input land_output vol_output
rename land_input_w  land_input
rename land_output_w land_output
rename vol_output_w vol_output

* Count unique UPs per ADC07 (and per season later) to enforce >=3 rule correctly
egen tag_up = tag(adc id_ca22_up name)
rename tag_up num_up

* ---------- ALL SEASONS: collapse to ADC07 ----------
preserve
    collapse (sum) land_*t vol_*t num_up, by(cve_ent muncode adc name)
    

    * Shares / yields
    gen share_irrig = land_irr_input/(land_irr_input + land_rf_input)
    keep if num_up >= 3

    gen yield       = vol_output/land_input
    gen yield_irrig = vol_irr_output/land_irr_input
    gen yield_rf    = vol_rf_output/land_rf_input
    gen yield_gp    = vol_gp_output/land_gp_input
    gen yield_no_gp = vol_ngp_output/land_ngp_input

    order cve_ent muncode adc name num_up land_* vol_* share_irrig yield*
    save "`adc_l07_use'", replace
restore

preserve
	*** If perennial, ignore planting month entirely for season exercise ****
	gen     type = "peren" if name == "Agave"
	replace type = "peren" if name == "Alfalfa"
	replace type = "peren" if name == "Apple"
	replace type = "peren" if name == "Avocados"
	replace type = "peren" if name == "Bananas"
	replace type = "peren" if name == "Blackberry"
	replace type = "peren" if name == "Blueberries"
	replace type = "peren" if name == "Cactus"
	replace type = "peren" if name == "Cocoa"
	replace type = "peren" if name == "Coconuts"
	replace type = "peren" if name == "Coffee"
	replace type = "peren" if name == "Grapes"
	replace type = "peren" if name == "Guava"
	replace type = "peren" if name == "Lemons"
	replace type = "peren" if name == "Mango"
	replace type = "peren" if name == "Oranges"
	replace type = "peren" if name == "Other species"
	replace type = "peren" if name == "Papayas"
	replace type = "peren" if name == "Pastures"
	replace type = "peren" if name == "Pineapples"
	replace type = "peren" if name == "Sugar"
	replace type = "peren" if name == "Walnut"
	*** INEGI defines primavera-verano as anything planted between March 1 and Sep 30
	replace type = "p-v" if missing(type) & plant_month >= 3 & plant_month <= 9
	replace type = "o-i" if missing(type) & (plant_month < 3 | plant_month > 9)

    collapse (sum) land_*t vol_*t num_up, by(cve_ent muncode adc type name)

    gen share_irrig = land_irr_input/(land_irr_input + land_rf_input)
    keep if num_up >= 3

    gen yield       = vol_output/land_input
    gen yield_irrig = vol_irr_output/land_irr_input
    gen yield_rf    = vol_rf_output/land_rf_input
    gen yield_gp    = vol_gp_output/land_gp_input
    gen yield_no_gp = vol_ngp_output/land_ngp_input

    order cve_ent muncode adc type name num_up land_* vol_* share_irrig yield*
    save "`adc_l07_szn'", replace
restore

/*
*** CA22 ADC
gen land_irr_input   = land_input  if irrig == 1
gen land_rf_input    = land_input  if irrig == 0
gen land_irr_output  = land_output if irrig == 1
gen land_rf_output   = land_output if irrig == 0
gen vol_irr_output   = vol_output  if irrig == 1
gen vol_rf_output    = vol_output  if irrig == 0

gen land_ngp_input   = land_input  if gp == 0
gen land_gp_input    = land_input  if gp == 1
gen land_ngp_output  = land_output if gp == 0
gen land_gp_output   = land_output if gp == 1
gen vol_ngp_output   = vol_output  if gp == 0
gen vol_gp_output    = vol_output  if gp == 1

merge m:1 id_ca22_up using "`loc_clean'", keep(3) nogen keepusing(id_ca22_up cve_ent muncode adc ageb cve_geo1 cve_ageb1)

rename adc       adc_on
rename ageb      ageb_on
rename cve_geo1  adc_first
rename cve_ageb1 ageb_first

gen num_up = 1

preserve
*** Collapse down and save at ADC level
collapse (sum) land_*t vol_*t q* num_up, by(cve_ent muncode ageb_first adc_first name greenhouse)

*** Keep only if not greenhouse
keep if greenhouse == 0 
drop greenhouse

gen share_irrig = land_irr_input/(land_irr_input+land_rf_input)

keep if num_up >= 3

*** Generate yield
gen yield       = vol_output/land_input
gen yield_irrig = vol_irr_output/land_irr_input
gen yield_rf    = vol_rf_output/land_rf_input
gen yield_gp    = vol_gp_output/land_gp_input
gen yield_no_gp = vol_ngp_output/land_ngp_input
save "`adc_land_use'", replace
restore

preserve
*** Collapse down and save at AGEB level
collapse (sum) land_*t vol_*t q* num_up, by(cve_ent muncode ageb_first name greenhouse)


*** Keep only if not greenhouse
keep if greenhouse == 0 
drop greenhouse

gen share_irrig = land_irr_input/(land_irr_input+land_rf_input)

keep if num_up >= 3

*** Generate yield
gen yield       = vol_output/land_input
gen yield_irrig = vol_irr_output/land_irr_input
gen yield_rf    = vol_rf_output/land_rf_input
gen yield_gp    = vol_gp_output/land_gp_input
gen yield_no_gp = vol_ngp_output/land_ngp_input
save "`ageb_lnd_use'", replace
restore

preserve
*** Collapse down and save at ADC level
collapse (sum) land_*t vol_*t q* num_up, by(cve_ent muncode ageb_on adc_on name greenhouse)


*** Keep only if not greenhouse
keep if greenhouse == 0 
drop greenhouse

gen share_irrig = land_irr_input/(land_irr_input+land_rf_input)

keep if num_up >= 3

*** Generate yield
gen yield       = vol_output/land_input
gen yield_irrig = vol_irr_output/land_irr_input
gen yield_rf    = vol_rf_output/land_rf_input
gen yield_gp    = vol_gp_output/land_gp_input
gen yield_no_gp = vol_ngp_output/land_ngp_input
save "`adc_lnd_one'", replace
restore

preserve
*** Collapse down and save at AGEB level
collapse (sum) land_*t vol_*t q* num_up, by(cve_ent muncode ageb_on name greenhouse)


*** Keep only if not greenhouse
keep if greenhouse == 0 
drop greenhouse

gen share_irrig = land_irr_input/(land_irr_input+land_rf_input)

keep if num_up >= 3

*** Generate yield
gen yield       = vol_output/land_input
gen yield_irrig = vol_irr_output/land_irr_input
gen yield_rf    = vol_rf_output/land_rf_input
gen yield_gp    = vol_gp_output/land_gp_input
gen yield_no_gp = vol_ngp_output/land_ngp_input
save "`ageb_lnd_one'", replace
restore



preserve
*** Collapse down and save at MUN level
collapse (sum) land_*t vol_*t q* num_up, by(cve_ent muncode name greenhouse)

*** Keep only if not greenhouse
keep if greenhouse == 0 
drop greenhouse

keep if num_up >= 3
gen share_irrig = land_irr_input/(land_irr_input+land_rf_input)

*** Generate yield
gen yield       = vol_output/land_input
gen yield_irrig = vol_irr_output/land_irr_input
gen yield_rf    = vol_rf_output/land_rf_input
gen yield_gp    = vol_gp_output/land_gp_input
gen yield_no_gp = vol_ngp_output/land_ngp_input
save "`mun_land_use'", replace
restore

************************************************************************************************************************************************************
************************************************************ Produce land use by growing seasons ***********************************************************
************************************************************************************************************************************************************


*** If perennial, ignore planting month entirely for season exercise ****
gen     type = "peren" if name == "Agave"
replace type = "peren" if name == "Alfalfa"
replace type = "peren" if name == "Apple"
replace type = "peren" if name == "Avocados"
replace type = "peren" if name == "Bananas"
replace type = "peren" if name == "Blackberry"
replace type = "peren" if name == "Blueberries"
replace type = "peren" if name == "Cactus"
replace type = "peren" if name == "Cocoa"
replace type = "peren" if name == "Coconuts"
replace type = "peren" if name == "Coffee"
replace type = "peren" if name == "Grapes"
replace type = "peren" if name == "Guava"
replace type = "peren" if name == "Lemons"
replace type = "peren" if name == "Mango"
replace type = "peren" if name == "Oranges"
replace type = "peren" if name == "Other species"
replace type = "peren" if name == "Papayas"
replace type = "peren" if name == "Pastures"
replace type = "peren" if name == "Pineapples"
replace type = "peren" if name == "Sugar"
replace type = "peren" if name == "Walnut"
*** INEGI defines primavera-verano as anything planted between March 1 and Sep 30
replace type = "p-v" if missing(type) & plant_month >= 3 & plant_month <= 9
replace type = "o-i" if missing(type) & (plant_month < 3 | plant_month > 9)


preserve
*** Collapse down and save at ADC level
collapse (sum) land_*t vol_*t q* num_up, by(cve_ent muncode ageb_first adc_first type name greenhouse)

*** Keep only if not greenhouse
keep if greenhouse == 0 
drop greenhouse

gen share_irrig = land_irr_input/(land_irr_input+land_rf_input)


keep if num_up >= 3

*** Generate yield
gen yield       = vol_output/land_input
gen yield_irrig = vol_irr_output/land_irr_input
gen yield_rf    = vol_rf_output/land_rf_input
gen yield_gp    = vol_gp_output/land_gp_input
gen yield_no_gp = vol_ngp_output/land_ngp_input
save "`adc_land_szn'", replace
restore

preserve
*** Collapse down and save at AGEB level
collapse (sum) land_*t vol_*t q* num_up, by(cve_ent muncode ageb_first type name greenhouse)


*** Keep only if not greenhouse
keep if greenhouse == 0 
drop greenhouse

gen share_irrig = land_irr_input/(land_irr_input+land_rf_input)

keep if num_up >= 3

*** Generate yield
gen yield       = vol_output/land_input
gen yield_irrig = vol_irr_output/land_irr_input
gen yield_rf    = vol_rf_output/land_rf_input
gen yield_gp    = vol_gp_output/land_gp_input
gen yield_no_gp = vol_ngp_output/land_ngp_input
save "`ageb_lnd_szn'", replace
restore

preserve
*** Collapse down and save at ADC level
collapse (sum) land_*t vol_*t q* num_up, by(cve_ent muncode ageb_on adc_on type name greenhouse)

*** Keep only if not greenhouse
keep if greenhouse == 0 
drop greenhouse

gen share_irrig = land_irr_input/(land_irr_input+land_rf_input)

keep if num_up >= 3

*** Generate yield
gen yield       = vol_output/land_input
gen yield_irrig = vol_irr_output/land_irr_input
gen yield_rf    = vol_rf_output/land_rf_input
gen yield_gp    = vol_gp_output/land_gp_input
gen yield_no_gp = vol_ngp_output/land_ngp_input
save "`adc_szn_one'", replace
restore

preserve
*** Collapse down and save at AGEB level
collapse (sum) land_*t vol_*t q* num_up, by(cve_ent muncode ageb_on type name greenhouse)


*** Keep only if not greenhouse
keep if greenhouse == 0 
drop greenhouse

gen share_irrig = land_irr_input/(land_irr_input+land_rf_input)

keep if num_up >= 3

*** Generate yield
gen yield       = vol_output/land_input
gen yield_irrig = vol_irr_output/land_irr_input
gen yield_rf    = vol_rf_output/land_rf_input
gen yield_gp    = vol_gp_output/land_gp_input
gen yield_no_gp = vol_ngp_output/land_ngp_input
save "`ageb_szn_one'", replace
restore


preserve
*** Collapse down and save at MUN level
collapse (sum) land_*t vol_*t q* num_up, by(cve_ent muncode type name greenhouse)

*** Keep only if not greenhouse
keep if greenhouse == 0 
drop greenhouse
keep if num_up >= 3

gen share_irrig = land_irr_input/(land_irr_input+land_rf_input)

*** Generate yield
gen yield       = vol_output/land_input
gen yield_irrig = vol_irr_output/land_irr_input
gen yield_rf    = vol_rf_output/land_rf_input
gen yield_gp    = vol_gp_output/land_gp_input
gen yield_no_gp = vol_ngp_output/land_ngp_input
save "`mun_land_szn'", replace
restore

************************************************************************************************************************************************************
*/