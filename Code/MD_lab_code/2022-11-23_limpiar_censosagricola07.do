/* censosagricolas_limpiar_07.do -- jsayre@berkeley.edu

Based on code written by M. A. Bolhuis and M. Gonzalez-Navarro, modified by James Sayre.
*/

*DONE:
*2007
*Labor inputs
*Land inputs
*Farm level output
*Quantities (volume by crop)
*Estimate Farm Level TFP and its distribution at locality level (mean, variance, etc)

clear all

set locale_ui en
set more off

local date = "2022-11-23"

*** Directorios
local wrk_dir = "Z:\Procesamiento\Trabajo\"
local ag_census_07_dir = "`wrk_dir'Censo Agrícola\"
local ag_census_clean_07_dir = "`wrk_dir'Censo Agrícola\Cleaned\"


local intermed_dir = "`wrk_dir'Intermediates\"
local results_dir = "Z:\Resultados\LM2304-CA-`date'\"

capture mkdir "`results_dir'"

*** Insumos
local ag_contract07 = "`ag_census_07_dir'TRD_AGRICULTURA_BAJO_CONTRATO.dta"
local ag_oi_07 = "`ag_census_07_dir'TRD_AGRICULTURA_OI.dta"
local perennials07 = "`ag_census_07_dir'TRD_AGRICULTURA_PERENNES.dta"
local ag_pv_07 = "`ag_census_07_dir'TRD_AGRICULTURA_PV.dta"
local land_size07 = "`ag_census_07_dir'TRD_CARACT_UP.dta"
local destination07 = "`ag_census_07_dir'TRD_DESTINO_PROD_AGRICOLA.dta"
local exports07 = "`ag_census_07_dir'TRD_EXPORTACION_AGRICOLA.dta"
local intercalado_oi_07 = "`ag_census_07_dir'TRD_INTERCALADOS_OI.DTA"
local intercalado_peren_07 = "`ag_census_07_dir'TRD_INTERCALADOS_PERENNES.DTA"
local intercalado_pv_07 = "`ag_census_07_dir'TRD_INTERCALADOS_PV.DTA"
local organization_07 = "`ag_census_07_dir'TRD_ORGANIZACION_ENTRE_PRODUCTORES.dta"
local greenhouse07 = "`ag_census_07_dir'TRD_PLANTAS_INVERNADERO.dta"
local plantas_vivero_07 = "`ag_census_07_dir'TRD_PLANTAS_VIVERO.DTA"
local irrig_07 = "`ag_census_07_dir'TRD_RIEGO_SUPERF_TIPO_FUENTE.dta"
local instal07 = "`ag_census_07_dir'TRD_TECNOLOGIA_AGRICOLA.dta"
local location2_07 = "`ag_census_07_dir'TRD_TERRENOS_UP.dta"
local location_07 = "`ag_census_07_dir'TRD_UBICACION_UP.dta"

*** No los solicite
*local labor_07 = "`ag_census_07_dir'TRD_ORGANIZACION_MANEJO_UP.dta"
*local soil_use_07 = "`ag_census_07_dir'TRD_Uso_de_Suelo.dta"
*local tractors_07 = "`ag_census_07_dir'TRD_TRACTORES_VEHICULOS_MAQUINARIA.dta"
*local installations_07 = "`ag_census_07_dir'TRD_CONSTRUCCIONES_INSTALACIONES.dta"
*local credito_07 = "`ag_census_07_dir'TRD_CREDITO_SEGURO_APOYOS.dta"

*** Nuevas
local training = "`ag_census_07_dir'TRD_CAPACITACION_ASISTENCIA_TECNICA.dta"
local rights = "`ag_census_07_dir'TRD_DERECHOS.dta"
local losses = "`ag_census_07_dir'TRD_NO_SEMB_SUPERF_CAUSA.dta"
local trees = "`ag_census_07_dir'TRD_PRODUCCION_MADERABLE.dta"
local jungle = "`ag_census_07_dir'TRD_SUPERFICIES_BOSQUE_SELVA.dta"
local ownership = "`ag_census_07_dir'TRD_TENENCIA.dta"

***Rendimiento

************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************

*CONTENT DO-FILE*
*(0) CONTRACTING BEHAVIOR
*(1) LOCALIZATION OF FARM UNIT	* DONE
*(2) LABOR INPUTS	* DONE
*(3) TOTAL LAND 	* DONE
*(4) LAND BY TYPE	* DONE
*(5) OUTPUT AND LAND INPUT BY CROP	* DONE
*(6) GREENHOUSE PRODUCTION	* DONE
*(7) CAPITAL AND INTERMEDIATE INPUTS	* DONE
*(8) FINANCE	* DONE
*(9) MARKETS	* DONE

************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************

***************************
* (0) CONTRACTING BEHAVIOR*
***************************
/*
****** 2007 ******
use "`ag_contract07'", clear
drop D_R
rename p0070101 land_used_to_grow
rename p0070105 ha_land_used_to_grow
rename p0150001 has_contract_with_comp
rename p0150102 cve_cultivo1
rename p0150202 cve_cultivo2
rename p0150302 cve_cultivo3
rename p0150401 contract_with_packer
rename p0150501 contract_with_agroindustry
rename p0150601 contract_with_commercializer
rename p0150701 specify_contract_other_industry
rename p0150702 contract_with_other_industry
rename cultivo_1 nme_cultivo1
rename cultivo_2 nme_cultivo2
rename cultivo_3 nme_cultivo3

replace nme_cultivo1 = " " if nme_cultivo1 == "NO ESPECIFICADO"
replace nme_cultivo2 = " " if nme_cultivo2 == "NO ESPECIFICADO"
replace nme_cultivo3 = " " if nme_cultivo3 == "NO ESPECIFICADO"

preserve
drop nme_cultivo* cve_cultivo*
save "`ag_census_clean_07_dir'ag_contracting_07.dta", replace
restore


drop if nme_cultivo1 == " "
drop *contract* specify* *land*

reshape long cve_cultivo nme_cultivo, i(id_cup) j(renglon_esque)
drop if nme_cultivo == " "
drop renglon_esque

gen name_crop = ustrto(ustrnormalize(nme_cultivo, "nfd"), "ascii", 2)
drop nme_cultivo
rename name_crop nombre_cul
replace nombre_cul = "CASTANO" if nombre_cul == "CASTAO"
replace nombre_cul = "CANA DE AZUCAR" if nombre_cul == "CAA DE AZUCAR"
replace nombre_cul = "PINA"  if nombre_cul == "PIA"
replace nombre_cul = "PINANONA"  if nombre_cul == "PIANONA"
replace nombre_cul = "PINON" if nombre_cul == "PION"

gen is_listed_contract_crp = 1
duplicates drop id_cup nombre_cul, force
drop cve_cultivo

save "`ag_census_clean_07_dir'ag_contracts_namedcrop_07.dta", replace
*/

************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************

***************************
* (1) LOCALIZATION OF FARM UNIT*
***************************

*VARIABLES*

*loc
*mun
*state
*state_discordant
*mun_discordant


************************************************************************************************************************************************************
*********************************************************************************************************************************************************

******
*2007*
******	
/*
use "`location_07'",clear
*p0000101	Codigo de estatus que guarda la unidad de produccin	Status code that saves the unit of production
*p0000102	Folio construido para los terrenos que conforman la unidad de produccion	Folio built for the lands that make up the production unit
*p0000201	Clave de la entidad donde se ubican los terrenos	Key of the entity where the land is located
rename p0000201 CVE_EDO
*p0000301	Clave del municipio donde se ubican los terrenos	Key of the municipality where the land is located
rename p0000301 CVE_MPIO
gen muncode = CVE_EDO+CVE_MPIO
keep id_cup muncode
destring muncode, replace
label variable muncode "5 digit loc. code"
drop if missing(muncode)
save "`ag_census_clean_07_dir'geo_identifier_muncode_only_2007.dta", replace


*** J: My version of geolocators, which provide ADC level info on plot location
*** the id_cup x renglon part really matters! different renglons can be in different agebs/muns/adcs
use "`location2_07'", clear
label variable id_cup "Unit of production ID"
label variable renglon "ID of field"
gen muncode = p0030102+p0030104
destring muncode, replace
label variable muncode "5 digit loc. code"
gen locality = p0030102+p0030104+p0030106
gen ageb = p0030102+p0030104+p0030107
gen adc = p0030102+p0030104+p0030107+p0030108
label variable locality "Locality"
label variable ageb "AGEB"
label variable adc "Número de área de control o manzana"
drop p*

*** There are two farms in which the renglons span across adcs/agebs
drop if id_cup == "000104558" & renglon == 1 & adc != "25001155-7002"
drop if id_cup == "000104558" & renglon == 2 & adc != "25001155-7034"
drop if id_cup == "003671115" & renglon == 1 & adc != "05009045-6015"
duplicates drop id_cup renglon if id_cup == "000104558" | id_cup == "003671115", force

save "`ag_census_clean_07_dir'geo_identifier_2007.dta", replace
*/

************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************	
	
**************
*(2) LABOR INPUTS*
**************

*Note: Available in terms of persons working on the farm, broken down by paid workers and permanent workers.

*VARIABLES*

*family_farm
*labor
*labor_paid
*labor_paid_6mo
*labor_female 
*labor_paid_female 
*labor_paid_6mo_female

************************************************************************************************************************************************************
/*
**** 2022-11-18: Missing labor 
******
*2007*
******

*TRD_ORGANIZACION_MANEJO_Up
	use "`labor_07'",clear
	*p1210101	Organizacion individual para realizar las labores agropecuarias o forestales	Individual organization to carry out agricultural or forestry work
	*p1210201	Se organiza con su esposa o hijos para realizar las labores agropecuarias o forestales	Organized with his wife or children to carry out agricultural or forestry work
	gen family_farm=0
	replace family_farm=1 if p1210101==1 | p1210201==1  	
	label var family_farm "farm is managed individually or with family and kids"
	*p1210301	Se organiza como un grupo o cooperativa para realizar las labores agropecuarias o forestales	Organized as a group or cooperative to carry out agricultural and forestry work
	*p1210401	Total de integrantes del grupo o cooperativa	Total members of the group or cooperative
	*p1210501	Se organiza como una empresa para realizar las labores agropecuarias o forestales	Organized as a company to carry out agricultural and forestry work
	*p1210701	Total de integrantes de la otra forma en que se organiza para realizar las labores agropecuarias	Total members of the other way in which it is organized to carry out agricultural work
	*p1220001	participacion de familiares del productor en las labores agropecuarias o forestales	participation of family members of the producer in agricultural or forestry work
	*p1220002	Total de participantes familiares del productor en las labores agropecuarias o forestales	Total family members of the producer in agricultural or forestry work
	gen labor=p1220002+p1230002  
	label var labor "labor is contracted labor plus family labor"
	gen labor_unpaid=p1220002
	*p1220101	Cuantas eran personas menores a 12 anos familiares	How many were people under 12 years of age
	*p1220102	Cuantas eran mujeres menores a 12 anos familiares	How many were women under 12 years old
	*p1220201	Cuantas eran personas de 12 a 18 anos familiares	How many were people aged 12 to 18 years old
	*p1220202	Cuantas eran mujeres de 12 a 18 anos familiares	How many were women aged 12 to 18 years old
	*p1220301	Cuantas eran personas de 18 a 60 anos familiares	How many were people aged 18-60 years
	*p1220302	Cuantas eran mujeres de 18 a 60 anos familiares	How many were women aged 18-60 years
	*p1220401	Cuantas eran personas de mas de 60 anos familiares	How many were people older than 60 years old
	*p1220402	Cuantas eran mujeres de mas de 60 anos familiares	How many were women over 60 years of age
	*p1230001	personas contratadas	Employees
	*p1230002	personas totales contratadas	Total employees
	gen labor_unpaid_female=p1220102 + p1220202 + p1220302 + p1220402
	gen labor_paid=p1230002
	*p1230101	personas contratadas por 6 meses o mas	people hired for 6 months or more
	gen labor_paid_6mo=p1230101
	*p1230102	Mujeres contratadas por 6 meses o mas	Women hired for 6 months or more
	gen labor_paid_6mo_female = p1230102
	*p1230201	personas contratadas por 6 meses o mas provenientes de los alrededores o zonas cercanas	people hired for 6 months or more from the surrounding area or nearby areas
	*p1230301	personas contratadas por 6 meses o mas  provenientes de otra parte del mismo estado	persons hired for 6 months or more from another part of the same state
	*p1230401	personas contratadas por 6 meses o mas provenientes de otro estado	persons hired for 6 months or more from another state
	*p1230501	personas contratadas por 6 meses o mas provenientes de otro pais	persons hired for 6 months or more from another country
	*p1230601	personas contratadas por menos de 6 meses	people hired for less than 6 months
	*p1230602	Mujeres contratadas por menos de 6 meses	Women hired for less than 6 months
	gen labor_paid_female = p1230102 + p1230602
	gen labor_female = labor_paid_female + p1220102 + p1220202 + p1220302 + p1220402
	*p1230701	personas contratadas por menos de 6 meses provenientes de los alrededores o zonas cercanas	persons hired for less than 6 months from the surrounding area or nearby areas
	*p1230801	personas contratadas por menos de 6 meses provenientes de otra parte del mismo estado	people hired for less than 6 months from another part of the same state
	*p1230901	personas contratadas por menos de 6 meses provenientes de otro estado	people hired for less than 6 months from another state
	*p1231001	personas contratadas por menos de 6 meses provenientes de otro pais	persons hired for less than 6 months from another country
	*p1240001	productor contratado para otra empresa	producer hired for another company
	*p1240101	Responsable contratado para otra empresa	Responsible hired for another company
	*p1250001	productor contratado para tareas o labores agropecuarias o forestales	producer hired for agricultural or forestry tasks or tasks
	*p1250101	productor contratado para tareas o labores agropecuarias o forestales por un periodo menor a 6 me	producer hired for agricultural or forestry tasks or tasks for a period of less than 6 m
	*p1250201	productor contratado para tareas o labores agropecuarias o forestales por un periodo de 6 meses o mas	producer hired for agricultural or forestry tasks or work for a period of 6 months or more
	*p1250301	Labores realizadas en los alrededores o zonas cercanas	Work performed in the surrounding area or nearby
	*p1250401	Labores realizadas en otra parte del mismo estado	Work performed elsewhere in the same state
	*p1250501	Labores realizadas en otro estado	Work performed in another state
	*p1250601	Labores realizadas en otro pais	Work performed in another country
	
	keep id_cup family_farm labor* 
	save "`ag_census_clean_07_dir'labor_2007.dta", replace
*/
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************

*************************
***** (3) TOTAL LAND*****
*************************

**********************************************************************************************************************************************************
******
*2007*
******
/*
*TRD_CARACT_UP	
	use "`land_size07'", clear

	*p0020101	Total de terrenos que manejo en el Municipio	Total land managed by the Municipality	
	*p0040104	Superficie total de los terrenos de la unidad de produccion	Total land area of ??the production unit
	gen land=p0040104
	label var land "Total land in farm unit"
	
	keep land id_cup
	save "`ag_census_clean_07_dir'land_2007.dta", replace

************************************************************************************************************************************************************

**************************
*****(4) LAND BY TYPE*****
**************************

*2007 asked questions by plot, about type of land (disregarding use).

*VARIABLES

*land_irr
*land_rain

******
*2007*
******
	use "`irrig_07'", clear
	*p0170104	Superficie de temporal
	rename p0170104 land_rain
	label var land_rain "all rainfed agricultural land"
	*p0170204	Superficie de riego
	rename p0170204 land_irr
	label var land_irr "irrigated agriculture land"
	
	keep id_cup land_rain land_irr
	save "`ag_census_clean_07_dir'land_by_use_2007.dta", replace

	
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************


*******************************************
*****(5) OUTPUT AND LAND INPUT BY CROP*****
*******************************************

******
*2007*
******

*VARIABLES*

*For i E {pv,oi,per}, j E C, where C is set of crops* :

*prod_unit_i
*land_input_i_j
*vol_output_i_j
*land_output_i_j
*land_use_i_total
*land_input_i_total

*****************************************************************************************************************************

	
	*There are lots of crops (338 in total). Some are very low prevalence.
	*These are the 10 most prevalent in terms of number of plots (note: in area planted this may be different):
	
	*	                 nombre_cul |      Freq.     Percent        Cum.
	*----------------------------+-----------------------------------
	*                 nombre_cul |      Freq.     Percent        Cum.
	*----------------------------+-----------------------------------
	*            AVENA FORRAJERA |     87,834        2.43        2.43	Oat forage
	*                   CALABAZA |     44,416        1.23        3.66	Pumpkin
	*               CEBADA GRANO |     31,666        0.88        4.54	Barley
	*                CHILE VERDE |     34,593        0.96        5.50	Green chile
	*                     FRIJOL |    527,857       14.62       20.12	Bean
	*             MAIZ FORRAJERO |     33,062        0.92       21.04	Forage maize
	*                 MAIZ GRANO |  2,658,863       73.65       94.68	Corn grain
	*      SORGO FORRAJERO VERDE |     53,734        1.49       96.17	Sorghum (forage green)
	*                SORGO GRANO |    109,418        3.03       99.20	Sorghum (grain)
	*                TRIGO GRANO |     28,808        0.80      100.00	Wheat
	*----------------------------+-----------------------------------
	*                      Total |  3,610,251      100.00

	*These top 10 crops account for 92.8% of all plots in agriculture spring-summer cycle (3,887,232):


	
*TRD_AGRICULTURA_PV
use "`ag_pv_07'", clear	
*RENGLON	Identificador de cultivo en P-V, en cuestionario	Culture identifier in P-V, questionnaire	
*p008n02	Clave del cultivo en P-V	P-V crop key
rename p008n02 cve_cultivo
*p008n06	Superficie sembrada del cultivo en P-V	Sown crop area in P-V
rename p008n06 land_input
*p008n10	Superficie cosechada del cultivo en P-V	Harvested crop area in P-V
rename p008n10 land_output
*p008n14	Volumen cosechado del cultivo en P-V	Harvested volume of the crop in P-V
rename p008n14 vol_output
*p008n19	Cuanto vendio o espera vender del cultivo en P-V	How much you sold or expect to sell from the crop in P-V
*p008n31	Superficie de cultivo organico del cultivo en P-V	Organic cultivation area of crop in P-V
*Nombre_cul	Nombre del cultivo	Crop name
	
gen name_crop = ustrto(ustrnormalize(nombre_cul, "nfd"), "ascii", 2)
drop nombre_cul
rename name_crop nombre_cul
replace nombre_cul = "CASTANO" if nombre_cul == "CASTAO"
replace nombre_cul = "CANA DE AZUCAR" if nombre_cul == "CAA DE AZUCAR"
replace nombre_cul = "PINA"  if nombre_cul == "PIA"
replace nombre_cul = "PINANONA"  if nombre_cul == "PIANONA"
replace nombre_cul = "PINON" if nombre_cul == "PION"
	
drop if nombre_cul == " "

*CROPS PV
	
preserve
*Create alphabetic list of crop names*
*collapse to 1 observation per crop name:
*Note that the number of farm units is small for many of these crops:
collapse (count) vol_output, by(nombre_cul)	
*sort alphabetically:
sort nombre_cul
*only keep crop names:
keep nombre_cul
quietly levelsof nombre_cul, local(cltvos)
save "`ag_census_clean_07_dir'cropnames_ag_spring_summer.dta", replace
restore 

*** 11/26 -- Don't want to get rid of renglon!!!!
***collapse (sum) land_input land_output vol_output, by(id_cup nombre_cul cve_cultivo)
keep id_cup renglon nombre_cul cve_cultivo land_input land_output vol_output

save "`ag_census_clean_07_dir'ag_prod_spring_summer.dta", replace


*****************************************************************************************************************************
	
	*In the fall winter cycle there are only 434,508 cultivated plots.
	*There are lots of crops (257 in total). Some are very low prevalence.
	*gen one=1
	*egen x=total(one),by(nombre_cul)
	*These are the most prevalent in terms of number of plots (using same plot cutoff from primavera-verano, 23,000 plots)
	*tab nombre_cul if x>23000
	
	*	nombre_cul |      Freq.     Percent        Cum.
	*----------------------------+-----------------------------------
	*            AVENA FORRAJERA |     45,171       13.58       13.58	Oat forage
	*                     FRIJOL |     95,375       28.66       42.24	Bean
	*                 MAIZ GRANO |    167,162       50.24       92.48	Corn grain
	*                TRIGO GRANO |     25,019        7.52      100.00	Wheat
	*----------------------------+-----------------------------------
	*                      Total |    332,727      100.00


	*These top 4 crops account for 76.6% of all plots in agriculture fall-winter cycle (434,508):

	*TRD_AGRICULTURA_OI
use "`ag_oi_07'", clear
*RENGLON	Identificador de cultivo en O-I	Culture identifier in O-I
*p010n02	Clave del cultivo en O-I	Crop key in O-I
rename p010n02 cve_cultivo
*p010n06	Superficie sembrada del cultivo en O-I	Sown crop area in O-I
rename p010n06 land_input
*p010n10	Superficie cosechada del cultivo en O-I	Cultivated crop area in O-I
rename p010n10 land_output
*p010n14	Volumen cosechado del cultivo en O-I	Harvested volume of the crop in O-I
rename p010n14 vol_output
*p010n19 	Vendio o espera vender del cultivo en O-I	Sold or expected to sell from the crop in O-I
*p010n31	Superficie de cultivo organico del cultivo en O-I	Organic crop area of ??O-I culture
*Nombre_cul	Nombre del cultivo	Culture name
		
gen name_crop = ustrto(ustrnormalize(nombre_cul, "nfd"), "ascii", 2)
drop nombre_cul
rename name_crop nombre_cul
replace nombre_cul = "CASTANO" if nombre_cul == "CASTAO"
replace nombre_cul = "CANA DE AZUCAR" if nombre_cul == "CAA DE AZUCAR"
replace nombre_cul = "PINA"  if nombre_cul == "PIA"
replace nombre_cul = "PINANONA"  if nombre_cul == "PIANONA"
replace nombre_cul = "PINON" if nombre_cul == "PION"
drop if nombre_cul == " "
	
preserve	
*Create alphabetic list of crop names*
*collapse to 1 observation per crop name:
*Note that the number of farm units is small for many of these crops:
collapse (count) vol_output, by(nombre_cul)	
*sort alphabetically:
sort nombre_cul 
*only keep crop names:
keep nombre_cul 
quietly levelsof nombre_cul, local(cltvos)
save "`ag_census_clean_07_dir'cropnames_ag_fall_winter.dta", replace
restore


**collapse (sum) land_input land_output vol_output, by(id_cup nombre_cul cve_cultivo)
keep id_cup renglon nombre_cul cve_cultivo land_input land_output vol_output

save "`ag_census_clean_07_dir'ag_prod_fall_winter.dta", replace
	

*****************************************************************************************************************************

	*There are 1,483,960 cultivated plots with perennials.
	*There are lots of crops (368 in total). Some are very low prevalence.
	*gen one=1
	*egen x=total(one),by(nombre_cul)
	*These are the most prevalent in terms of number of plots (using same plot cutoff from primavera-verano, 23,000 plots)
	*tab nombre_cul if x>23000
	
	*                 nombre_cul |      Freq.     Percent        Cum.
	*----------------------------+-----------------------------------
	*                   AGUACATE |     58,071        4.67        4.67	Avocado
	*              ALFALFA VERDE |     51,079        4.11        8.77	Alfalfa
	*                      CACAO |     41,140        3.31       12.08	Cacao
	*                CAFE CEREZA |    338,590       27.22       39.30	Coffee
	*             CAÑA DE AZUCAR |    141,813       11.40       50.70	Sugar cane
	*                      COPRA |     29,440        2.37       53.07	Coconut
	*                    DURAZNO |     32,228        2.59       55.66	Peach
	*                      LIMON |     48,989        3.94       59.59	Lemon	
	*                      MANGO |     60,984        4.90       64.50	Mango
	*                    NARANJA |    113,207        9.10       73.60	Orange
	*                     PASTOS |     23,211        1.87       75.46	Pastures
	*                    PLATANO |     36,458        2.93       78.39	Banana
	*         RYE GRASS EN VERDE |    268,790       21.61      100.00	Grassland
	*----------------------------+-----------------------------------
	*                      Total |  1,244,000      100.00
	*These top 13 crops account for 83.8% of all plots in perennials (1,483,592):
	*display 1244000/148396


*TRD_AGRICULTURA_PERENNES
	use "`perennials07'", clear
	*RENGLON	Identificador del cultivo perenne	Perennial crop identifier
	*p012n02	Clave del cultivo perenne	Key of the perennial crop
	rename p012n02 cve_cultivo
	*p012n06	Superficie plantada de cultivo perenne	Perennial planted area
	rename p012n06 land_input
	*p012n10	Superficie en produccion del cultivo perenne	Perennial crop production area
	rename p012n10 land_output
	*p012n14	Volumen cosechado del cultivo perenne	Harvested volume of perennial crop
	rename p012n14 vol_output
	*p012n19	Vendio o espera vender del cultivo perenne	Sold or expected to sell perennial crop
	*p012n31	Superficie de cultivo organico del cultivo perenne	Organic perennial crop area
	*Nombre_cul	Nombre del cultivo	Culture name
	
	gen name_crop = ustrto(ustrnormalize(nombre_cul, "nfd"), "ascii", 2)
	drop nombre_cul
	rename name_crop nombre_cul
	replace nombre_cul = "CASTANO" if nombre_cul == "CASTAO"
	replace nombre_cul = "CANA DE AZUCAR" if nombre_cul == "CAA DE AZUCAR"
	replace nombre_cul = "PINA"  if nombre_cul == "PIA"
	replace nombre_cul = "PINANONA"  if nombre_cul == "PIANONA"
	replace nombre_cul = "PINON" if nombre_cul == "PION"
	drop if nombre_cul == " "
		
	preserve
	*Create alphabetic list of crop names*
	*collapse to 1 observation per crop name:
	*Note that the number of farm units is small for many of these crops:
	collapse (count) vol_output, by(nombre_cul)	
	*sort alphabetically:
	sort nombre_cul 
	*only keep crop names:
	keep nombre_cul
	quietly levelsof nombre_cul, local(cltvos)
	save "`ag_census_clean_07_dir'cropnames_ag_perennials.dta", replace
	restore

	***collapse (sum) land_input land_output vol_output, by(id_cup nombre_cul cve_cultivo)	
	keep id_cup renglon nombre_cul cve_cultivo land_input land_output vol_output

	save "`ag_census_clean_07_dir'ag_prod_perennials.dta", replace

************************************************************************************************************************************************************


*2007 Greenhouse production
	*Only 15,497 greenhouse units.  However, here is where lots of the greenhouse tomatoes for export come in.  XXX Ignore for now
	*tab nombre_cul if x>1000
	*                nombre_cul |      Freq.     Percent        Cum.
	*---------------------------+-----------------------------------
	*                CRISANTEMO |      1,229       25.91       25.91
	*    TOMATE ROJO (JITOMATE) |      3,514       74.09      100.00
	*---------------------------+-----------------------------------
	*                     Total |      4,743      100.00
	
	*TRD_PLANTAS_INVERNADERO
	use "`greenhouse07'", clear
	*RENGLON	Identificador de la planta de invernadero, en cuestionario	Identifier of the greenhouse plant, questionnaire
	*p030n01	Clave Planta del invernadero	Key Greenhouse Plant
	rename p030n01 cve_cultivo
	*p030n05	Cantidad de planta de invernadero vendida	Amount of greenhouse plant sold
	rename p030n05 quant_main_gh
	*Nombre_cul	Nombre del cultivo	Crop name
	
	gen name_crop = ustrto(ustrnormalize(nombre_cul, "nfd"), "ascii", 2)
	drop nombre_cul
	rename name_crop nombre_cul
	replace nombre_cul = "CACTUS DEDITO DE NINO" if nombre_cul == "CACTUS DEDITO DE NIO"
	replace nombre_cul = "UNA DE GATO" if nombre_cul == "UA DE GATO"
	replace nombre_cul = "PINA"  if nombre_cul == "PIA"
	replace nombre_cul = "PINANONA"  if nombre_cul == "PIANONA"
	replace nombre_cul = "PINON" if nombre_cul == "PION"
	
	drop if nombre_cul==""
	drop if quant_main_gh==.

	preserve
	*Create alphabetic list of crop names*
	*collapse to 1 observation per crop name:
	*Note that the number of farm units is small for many of these crops:
	collapse (count) quant_main_gh, by(nombre_cul)	
	*sort alphabetically:
	sort nombre_cul 
	*only keep crop names:
	keep nombre_cul 
	quietly levelsof nombre_cul, local(cltvos)
	save "`ag_census_clean_07_dir'cropnames_greenhouse.dta", replace
	restore
	
	*collapse (sum) quant_main_gh, by(id_cup nombre_cul cve_cultivo)
	keep id_cup renglon nombre_cul cve_cultivo quant_main_gh

	save "`ag_census_clean_07_dir'ag_prod_greenhouse.dta", replace


************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************

**********************************************************
*****(7) CAPITAL AND INTERMEDIATE INPUTS******************
**********************************************************

*VARIABLES*

*impr_seed
*chem_fert
*nat_fert
*herb_insect
*graft_trees
*tech_ass
*animals
*tractor
*quant_trucks
*quant_tractor
*packer
*selector
*dehydrator
*processor
*shredder

************************************************************************************************************************************************************	
******
*2007*
******

*TRD_TECNOLOGIA_AGRICOLA	AGRICULTURAL TECHNOLOGY
	use "`instal07'", clear
	*p0230105	Superficie habilitada con fertilizantes quimicos	Surface enabled with chemical fertilizers
	rename p0230105 area_chem_fert
	label var area_chem_fert "Area with chemical fertilizers"
	
	gen chem_fert = 0 
	replace chem_fert = 1 if area_chem_fert>0 & area_chem_fert!=.
	
	*p0230205	Superficie habilitada con semilla mejorada	"improved seeds"
	rename p0230205 area_impr_seed
	label var area_impr_seed "Area with improved seeds"
	
	gen impr_seed = 0
	replace impr_seed = 1 if area_impr_seed>0 & area_impr_seed!=.
	
	*p0230301	Usa semilla geneticamente modificada o transgenica	Used genetically modified or transgenic seed
	rename p0230301 gen_seed
	label var gen_seed "Used genetically modified or transgenic seed"
	*p0230405	Superficie habilitada con abonos naturales	Area enabled with natural fertilizers
	rename p0230405 area_nat_fert
	label var area_nat_fert "Area with natural fertilizers"
	
	gen nat_fert = 0
	replace nat_fert = 1 if area_nat_fert>0 & area_nat_fert!=.
	
	*p0230505	Superficie habilitada con herbicidas quimicos	"chemical herbicides"
	rename p0230505 area_chem_herb
	label var area_chem_herb "Area with chemical herbicides"
	*p0230605	Superficie habilitada con herbicidas organicos	"organic herbicides"
	rename p0230605 area_org_herb
	label var area_org_herb "Area with organic herbicides"
	*p0230705	Superficie habilitada con insecticidas quimicos	"chemical insecticides"
	rename p0230705 area_chem_insect
	label var area_chem_insect "Area enabled with chemical insecticides"
	*p0230805	Superficie habilitada con insecticidas organicos	"organic insecticides"
	rename p0230805 area_org_insect
	label var area_org_insect "Area enabled with organic insecticides"
	
	gen herb_insect = 0
	foreach y in "chem_herb" "org_herb" "chem_insect" "org_insect" {
	replace herb_insect = 1 if area_`y'>0 & area_`y'!=.
	}
	
	*p0230901	Realiza control biologico de plagas	Performed biological control of pests
	rename p0230901 bio_pest_control
	label var bio_pest_control "Performed biological control of pests"
	*p0231001	Realizo injertos de arboles	"tree grafts"
	rename p0231001 graft_trees
	label var graft_trees "Performed tree grafts"
	*p0231101	Realizo rotacion de cultivos	"crop rotation"
	rename p0231101 crop_rotation
	label var crop_rotation "Performed crop rotation"
	*p0231201	Realizo podas	"pruning"
	rename p0231201 pruning
	label var pruning "Performed pruning"
	*p0231305	Superficie con practica de quemas controladas	Surface with controlled burning practice
	rename p0231305 area_controlled_burning
	label var area_controlled_burning "Surface with controlled burning practice"
	*p0231401	Recibio asistencia tecnica	Received technical assistance
	rename p0231401 tech_ass
	label var tech_ass "Received technical assistance"
	*p0231505	Superficie habilitada con otra tecnologia	Surface enabled with other technology
	rename p0231505 area_other_tech
	label var area_other_tech "Surface enabled with other technology"
	
	keep id_cup impr_seed chem_fert nat_fert herb_insect graft_trees tech_ass area*
	
	save "`ag_census_clean_07_dir'non_labor_inputs_1", replace
*/
/*
*** 2022-11-18: Missing tractors module
*TRD_TRACTORES_VEHICULOS_MAQUINARIA	TRACTORS, VEHICLES, MACHINERY
	use "`tractors_07'", clear
	*p1110001	Utilizo tractor para las actividades agropecuarias y forestales	I use tractor for agricultural and forestry activities
	rename p1110001 tractor
	label var tractor "Uses tractor for agricultural and forestry activities"
	*p1110101	Tractor rentado	Rented Tractor
	rename p1110101 rent_tractor
	label var rent_tractor "Rented tractor"
	*p1110201	Tractor prestado	Tractor loaned
	rename p1110201 borr_tractor
	label var borr_tractor "Borrowed tractor"
	*p1110301	Tractor de un grupo del que forma parte el productor	Tractor of a group of which the producer is part
	rename p1110301 coll_tractor
	label var coll_tractor "Collective tractor"
	*p1110401	Tractor propio	Own tractor
	rename p1110401 own_tractor
	label var own_tractor "Own tractor"
	*p1110501	Cuantos tractores tenia	How many tractors did you have
	rename p1110501 quant_tractor
	label var quant_tractor "How many tractors do you have"
	*p1110601	Potencia del tractor 1
	rename p1110601 pot_tractor1
	label var pot_tractor1 "Power of tractor 1"
	*p1110606	Anos de uso del tractor 1
	rename p1110606 years_tractor1
	label var years_tractor1 "Age of tractor 1"
	*p1110607	No sabe cuantos anos de uso tenia el tractor 1
	*p1110608	El tractor estaba en condiciones de funcionamiento 1
	rename p1110608 work_tractor1
	label var work_tractor1 "Tractor 1 in working condition"
	*p1110701	Potencia del tractor 2
	rename p1110701 pot_tractor2
	label var pot_tractor2 "Power of tractor 2"
	*p1110706	Anos de uso del tractor 2
	rename p1110706 years_tractor2
	label var years_tractor2 "Age of tractor 2"
	*p1110707	No sabe cuantos anos de uso tenia el tractor 2
	*p1110708	El tractor estaba en condiciones de funcionamiento 2
	rename p1110708 work_tractor2
	label var work_tractor2 "Tractor 2 in working condition"
	*p1110801	Potencia del tractor 3
	rename p1110801 pot_tractor3
	label var pot_tractor3 "Power of tractor 3"
	*p1110806	Anos de uso del tractor 3
	rename p1110806 years_tractor3
	label var years_tractor3 "Years of use tractor 3"
	*p1110807	No sabe cuantos anos de uso tenia el tractor 3
	*p1110808	El tractor estaba en condiciones de funcionamiento 3
	rename p1110808 work_tractor3
	label var work_tractor3 "Tractor 3 in working condition"
	*p1110901	Potencia del tractor 4
	rename p1110901 pot_tractor4
	label var pot_tractor4 "Power of tractor 4"
	*p1110906	Anos de uso del tractor 4
	rename p1110906 years_tractor4
	label var years_tractor4 "Years of use tractor 4"
	*p1110907	No sabe cuantos anos de uso tenia el tractor 4
	*p1110908	El tractor estaba en condiciones de funcionamiento 4
	rename p1110908 work_tractor4
	label var work_tractor4 "Tractor 4 in working condition"
	*p1111001	Potencia del tractor 5
	rename p1111001 pot_tractor5
	label var pot_tractor5 "Power of tractor 5"
	*p1111006	Anos de uso del tractor 5
	rename p1111006 years_tractor5
	label var years_tractor5 "Years of use tractor 5"
	*p1111007	No sabe cuantos anos de uso tenia el tractor 5
	*p1111008	El tractor estaba en condiciones de funcionamiento 5
	rename p1111008 work_tractor5
	label var work_tractor5 "Tractor 5 in working condition"
	*p1120001	Camiones o camionetas en propiedad
	rename p1120001 prop_trucks
	label var prop_trucks "Trucks or vans on property"
	*p1120002	Cuantos camiones o camionetas tenia	How many trucks or vans did you have
	rename p1120002 quant_trucks
	label var quant_trucks "How many trucks or vans did you have"
	*p1120105	Capacidad del vehiculo 1
	rename p1120105 cap_truck1
	label var cap_truck1 "Capacity truck 1"
	*p1120106	Estaba en condiciones de funcionamiento 1
	rename p1120106 work_truck1
	label var work_truck1 "Truck 1 in working condition"
	*p1120205	Capacidad del vehiculo 2
	rename p1120205 cap_truck2
	label var cap_truck2 "Capacity truck 2"
	*p1120206	Estaba en condiciones de funcionamiento 2
	rename p1120206 work_truck2
	label var work_truck2 "Truck 2 in working condition"
	*p1120305	Capacidad del vehiculo 3
	rename p1120305 cap_truck3
	label var cap_truck3 "Capacity truck 3"
	*p1120306	Estaba en condiciones de funcionamiento 3
	rename p1120306 work_truck3
	label var work_truck3 "Truck 3 in working condition"
	*p1120405	Capacidad del vehiculo 4
	rename p1120405 cap_truck4
	label var cap_truck4 "Capacity truck 4"
	*p1120406	Estaba en condiciones de funcionamiento 4
	rename p1120406 work_truck4
	label var work_truck4 "Truck 4 in working condition"
	*p1120505	Capacidad del vehiculo 5
	rename p1120505 cap_truck5
	label var cap_truck5 "Capacity truck 5"
	*p1120506	Estaba en condiciones de funcionamiento 5
	rename p1120506 work_truck5
	label var work_truck5 "Truck 5 in working condition"
	*p1130001	Tenia maquinaria agropecuaria o forestal en propiedad
	rename p1130001 prop_machinery
	label var prop_machinery "Machinery on property"
	*p1130002	Cantidad de maquinaria agropecuaria o forestal
	rename p1130002 quant_machinery
	label var quant_machinery "Amount of agricultural or forestry machinery"
	*p1130102	Tipo de maquinaria 1
	rename p1130102 machinery1
	label var machinery1 "Machinery type 1"
	*p1130103	Anos de uso de la maquinaria 1
	rename p1130103 years_mach1
	label var years_mach1 "Years of use of the machinery 1"
	*p1130108	Estaba en condiciones de funcionamiento 1
	rename p1130108 work_mach1
	label var work_mach1 "Machine 1 in working order"
	*p1130202	Tipo de maquinaria 2
	rename p1130202 machinery2
	label var machinery2 "Machinery type 2"
	*p1130203	Anos de uso de la maquinaria 2
	rename p1130203 years_mach2
	label var years_mach2 "Years of use of the machinery 2"
	*p1130208	Estaba en condiciones de funcionamiento 2
	rename p1130208 work_mach2
	label var work_mach2 "Machine 2 in working order"
	*p1130302	Tipo de maquinaria 3
	rename p1130302 machinery3
	label var machinery3 "Machinery type 3"
	*p1130303	Anos de uso de la maquinaria 3
	rename p1130303 years_mach3
	label var years_mach3 "Years of use of the machinery 3"
	*p1130308	Estaba en condiciones de funcionamiento 3
	rename p1130308 work_mach3
	label var work_mach3 "Machine 3 in working order"
	*p1130402	Tipo de maquinaria 4
	rename p1130402 machinery4
	label var machinery4 "Machinery type 4"
	*p1130403	Anos de uso de la maquinaria 4
	rename p1130403 years_mach4
	label var years_mach4 "Years of use of the machinery 4"
	*p1130408	Estaba en condiciones de funcionamiento 4
	rename p1130408 work_mach4
	label var work_mach4 "Machine 4 in working order"
	*p1130502	Tipo de maquinaria 5
	rename p1130502 machinery5
	label var machinery5 "Machinery type 5"
	*p1130503	Anos de uso de la maquinaria 5
	rename p1130503 years_mach5
	label var years_mach5 "Years of use of the machinery 5"
	*p1130508	Estaba en condiciones de funcionamiento 5
	rename p1130508 work_mach5
	label var work_mach5 "Machine 5 in working order"
	*Maquina_1	Nombre del maquina 1
	rename maquina_1 n_machinery1
	label var n_machinery1 "Name machinery 1"
	*Maquina_2	Nombre del maquina 2
	rename maquina_2 n_machinery2
	label var n_machinery2 "Name machinery 2"
	*Maquina_3	Nombre del maquina 3
	rename maquina_3 n_machinery3
	label var n_machinery3 "Name machinery 3"
	*Maquina_4	Nombre del maquina 4
	rename maquina_4 n_machinery4
	label var n_machinery4 "Name machinery 4"
	*Maquina_5	Nombre del maquina 5
	rename maquina_5 n_machinery5
	label var n_machinery5 "Name machinery 5"

	keep id_cup tractor quant_trucks quant_tractor 
	save "`ag_census_clean_07_dir'non_labor_inputs_2", replace
*/
/*
*** 2022-11-18: Missing installations module
*TRD_CONSTRUCCIONES_INSTALACIONES	BUILDING INSTALLATIONS
	use "`installations_07'", clear	
	*p0240101	Contaba con beneficiadora de cafe o cacao	I counted on coffee or cocoa processing plant
	rename p0240101 processor
	label var processor "Counted coffee or cocoa processing plant"
	*p0240102	Cuantos anos tiene con la instalacion beneficiadora de cafe o cacao	How old are you with the coffee or cocoa processing plant
	rename p0240102 years_processor
	label var years_processor "Years coffee or cocoa processing plant"
	*p0240201	Contaba con desfibradora	I had a shredder
	rename p0240201 shredder
	label var shredder "Counted a shredder"
	*p0240202	Cuantos anos tiene con la instalacion desfibradora	How old are you with the shredding plant
	rename p0240202 years_shredder
	label var years_shredder "Years shredder"
	*p0240301	Contaba con deshidratadora	I had dehydrator
	rename p0240301 dehydrator
	label var dehydrator "Counted a dehydrator"
	*p0240302	Cuantos anos tiene con la instalacion deshidratadora	How old are you with the dehydrating plant
	rename p0240302 years_dehydrator
	label var years_dehydrator "Years dehydrator"
	*p0240401	Contaba con empacadora de frutas o verduras	I had a fruit or vegetable packing machine
	rename p0240401 packer
	label var packer "Counted a packer"
	*p0240402	Cuantos anos tiene con la instalacion empacadora de frutas o verduras	How old are you with the fruit and vegetable packing plant
	rename p0240402 years_packer
	label var years_packer "Years packer"
	*p0240501	Contaba con seleccionadora	I had a selector
	rename p0240501 selector
	label var selector "Counted a selector"
	*p0240502	Cuantos anos tiene con la instalacion seleccionadora	How old are you with the sorting plant
	rename p0240502 years_selector
	label var years_selector "Years selector"
	*p0240602	Contaba con otra instalacion	Had another installation
	rename p0240602 other_instal
	label var other_instal "Counted another installation"
	*p0240603	Cuantos anos tiene con la otra instalacion	How old are you with the other installation
	rename p0240603 years_other_instal
	label var years_other_instal "Years other installation"
	*p0250101	Tiene vivero en el terreno	It has nursery on the ground
	rename p0250101 nursery
	label var nursery "Has a nusery on the ground"
	*p0260104	Superficie que ocupa el vivero	Area occupied by the nursery
	rename p0260104 area_nursery
	label var area_nursery "Area occupied by the nursery"
	*p0280101	Tiene invernadero en el terreno	Has greenhouse on the ground
	rename p0280101 greenhouse
	label var greenhouse "Has a greenhouse on the ground"
	*p0290004	Superficie que ocupa el invernadero	Surface occupied by the greenhouse
	rename p0290004 area_greenhouse
	label var area_greenhouse "Area occupied by the greenhouse"
	*p0290101	En que ano se instalo el invernadero	In what year was the greenhouse installed

	keep id_cup packer selector dehydrator processor shredder greenhouse
	save "`ag_census_clean_07_dir'non_labor_inputs_3", replace
*/
/*
*TRD_RIEGO_SUPERF_TIPO_FUENTE	
	use "`irrig_07'", clear	

	*p0170104	Superficie de temporal
	*p0170204	Superficie de riego
	*p0180101	Utiliza canales recubiertos para el riego
	*p0180201	Utiliza canales de tierra para el riego
	*p0180301	Utiliza sistema de aspersion para el riego
	*p0180401	Utiliza sistema de microaspersion para el riego
	*p0180501	Utiliza sistema de goteo para el riego
	*p0180602	Utiliza algun otro sistema para el riego
	*p0190101	El agua proviene de bordo u hoya o jaguey
	*p0190201	El agua proviene de pozo profundo
	*p0190304	A que profundidad extrae el agua
	*p0190401	El agua proviene de pozo a cielo abierto
	*p0190501	El agua proviene de un rio
	*p0190601	El agua proviene de un manantial
	*p0190701	El agua proviene de una presa
	*p0190801	Especifica la otra fuente
	*p0190802	El agua proviene de otra fuente
	*p0200101	Utiliza agua blanca para el riego
	*p0200201	Utiliza agua negra para el riego
	*p0200301	Utiliza agua Tratada para el riego
	*p0200401	No sabe
	*p0210101	Utilizo animales de tiro, tronco o yunta	Used shot, trunk or yoke animals
	rename p0210101 animals
	*p02710201	Los animales de tiro, son propios
	*p027710301	Utilizo tractor
	*p0220101	Sembro con coa, azadon u otra herramienta
	rename p0220101 hoe
	label var hoe "Sowed with hoe or other tool"
	
	keep id_cup animals
	save "`ag_census_clean_07_dir'non_labor_inputs_4", replace
*/

************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************

*************
*(8) FINANCE*
*************

*VARIABLES*

*credit
*insurance

************************************************************************************************************************************************************
/*
*** 2022-11-18: Missing credit module
*2007*
*TRD_CREDITO_SEGURO_APOYOS	CREDIT / INSURANCE SUPPORTS
	use "`credito_07'", clear	
	*p1140101	Obtuvo algun credito o prestamo	
	rename p1140101 credit
	*p1150101	Lo proporciono la banca comercial
	*p1150201	Lo proporciono una SOFOL
	*p1150301	Lo proporciono una union de credito
	*p1150401	Lo proporciono financiera rural
	*p1150500	Nombre de otra fuente
	*p1150502	Lo proporciono otra fuente
	*p1150601	Los fondos provienen de FIRA
	*p1160101	Credito de avio
	*p1160105	Monto del credito de avio
	*p1160201	Credito refaccionario
	*p1160205	Monto del credito refaccionario
	*p1160300	Nombre de credito de otro tipo
	*p1160302	Otro tipo de credito
	*p1160306	Monto del credito de otro tipo
	*p1170100	Contaba con algun seguro
	rename p1170100 insurance
	*p1170101	Seguro contratado con AGROASEMEX
	*p1170200	Nombre de otra institucion aseguradora
	*p1170202	Seguro contratado con otra institucion
	*p1180001	Obtuvo apoyo de programas gubernamentales
	*p1180101	Procampo
	*p1180201	Programa ganadero
	*p1180301	Programa diesel agropecuario
	*p1180401	Programa de apoyos directos al ingreso objetivo
	*p1180501	Programa de inversion rural
	*p1180601	Programa para el desarrollo de capacidades
	*p1180701	Programa para el desarrollo rural
	*p1180801	Programa de fomento agricola
	*p1180901	Programa de la mujer en el sector agrario
	*p1181001	Programa de desarrollo forestal prodefor
	*p1181101	Comision nacional forestal
	*p1181201	Programa de acuacultura y pesca
	*p1181301	Programa de atencion a jornaleros agricolas
	*p1181401	Programa de empleo temporal
	*p1181501	Programa de vivienda rural
	*p1181601	Seguro popular
	*p1181701	Programa oportunidades
	*p1181801	Programa de opciones productivas
	*p1181901	Fondo de apoyo para proyectos productivos
	*p1182001	Programa de infraestructura hidroagricola
	*p1182101	Programa de fondos regionales indigenas
	*p1182201	Programa para el desarrollo regional sustentable
	*p1182301	Programa de equidad de genero y pueblos indigenas
	*p1182402	Recibio apoyo de otro programa
	*p1190101	Ahorro parte de los ingresos
	*p1200101	Lo maneja con la banca comercial
	*p1200201	Lo maneja con la banca publica
	*p1200301	Lo maneja con una union de credito
	*p1200401	Lo maneja con una caja de ahorro
	*p1200502	Lo maneja con otra institucion
	
	keep id_cup credit insurance
	save "`ag_census_clean_07_dir'finance", replace
*/

************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
/*
*************
*(9) MARKETS*
*************

*VARIABLES*

*seller
*exporter

*2007*

*TRD_DESTINO_PROD_AGRICOLA*
	use "`destination07'", clear	
	*p0320101	Selecciono o seleccionara semilla para la siembra
	rename p0320101 selecc_seed
	label variable selecc_seed "Selecciono o seleccionará semilla para la siembra"
	
	*p0330101	Parte de la produccion destinada para el consumo de su familia
	rename p0330101 self_cons
	*p0340101	Parte de la produccion destinada para el consumo de sus animales
	rename p0340101 cons_animales
	*p0350001	Vendio o espera vender parte de la produccion agricola	Sold or expected to sell part of the agricultural production
	rename p0350001 market
	label var market "Production unit selling all or part of the agricultural production"
	*p0350101	Vendio a un intermediario
	rename p0350101 sells_intermediate
	*p0350201	Vendio a un mayorista
	rename p0350201 sells_mayorista
	*p0350301	Vendio a una cadena comercial
	rename p0350301 sells_commercial_chain
	*p0350401	Vendio una empacadora o agroindustria
	rename p0350401 sells_empacadora
	*p0350502	Vendio o espera vender parte de la produccion agricola a otro tipo de compradores
	rename p0350502 sells_other_type_buyer
	*p0360001	Vendio o espera vender al extranjero	Sold or expected to sell abroad
	rename p0360001 exporter
	label var exporter "Production unit selling all or part of the agricultural production to other country"
	*p0370001	Se procesa o transforma parte de la produccion agricola
	rename p0370001 processes_output
	*p0370101	Vende los productos obtenidos
	rename p0370101 sells_processed_output
	
	save "`ag_census_clean_07_dir'markets", replace
	
	
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************

***********************
*(10) GROUP MEMBERSHIP*
***********************

*VARIABLES*
*TBD*

*2007*

*TRD_ORGANIZACION_ENTRE_PRODUCTORES*


************************************************************************************************************************************************************


use "`organization_07'", clear
drop D_R

label variable p1270101 "Organización con otros productores para obtener apoyos o servicios"
label variable p1280101 "El productor se organiza o pertenece a un grupo para la obtención de crédito o comercializar los"
label variable p1280102 "Años de constitución"
*label variable p1280103 "Cantidad de socios"
*label variable p1280104 "Socios mujeres"
label variable p1280201 "El productor se organiza o pertenece a una sociedad de producción rural"
label variable p1280202 "Años de constitución"
*label variable p1280203 "Cantidad de socios"
*label variable p1280204 "Socios mujeres"
label variable p1280301 "El productor se organiza o pertenece a una sociedad cooperativa"
label variable p1280302 "Años de constitución"
*label variable p1280303 "Cantidad de socios"
*label variable p1280304 "Socios mujeres"
label variable p1280401 "El productor se organiza o pertenece a una sociedad civil"
label variable p1280402 "Años de constitución"
*label variable p1280403 "Cantidad de socios"
*label variable p1280404 "Socios mujeres"
label variable p1280501 "El productor se organiza o pertenece a una sociedad de solidaridad social"
label variable p1280502 "Años de constitución"
*label variable p1280503 "Cantidad de socios"
*label variable p1280504 "Socios mujeres"
gen farm_part_society = 1 if p1280201 == 1 | p1280301 == 1 |  p1280401 == 1 | p1280501 == 1
replace farm_part_society = 0 if missing(farm_part_society)
label variable p1280601 "El productor se organiza o pertenece a una unión de crédito"
label variable p1280602 "Años de constitución"
*label variable p1280603 "Cantidad de socios"
*label variable p1280604 "Socios mujeres"
label variable p1280701 "El productor se organiza o pertenece a una cooperativa de ahorro y crédito"
label variable p1280702 "Años de constitución"
*label variable p1280703 "Cantidad de socios"
*label variable p1280704 "Socios mujeres"
gen farm_part_credit_un = 1 if p1280601 == 1 | p1280701 == 1
replace farm_part_credit_un = 0 if missing(farm_part_credit_un)
label variable p1280801 "El productor se organiza o pertenece a una sociedad anónima"
rename p1280801 farm_part_llc
replace farm_part_llc = 0 if farm_part_llc == 2
label variable p1280802 "Años de constitución"
rename p1280802 years_since_llc_founded
*label variable p1280803 "Cantidad de socios"
*label variable p1280804 "Socios mujeres"
label variable p1280901 "El productor se organiza o pertenece a una asociación ganadera local"
label variable p1280902 "Años de constitución"
*label variable p1280903 "Cantidad de socios"
*label variable p1280904 "Socios mujeres"
label variable p1281001 "El productor se organiza o pertenece a una asociación agrícola local"
label variable p1281002 "Años de constitución"
*label variable p1281003 "Cantidad de socios"
*label variable p1281004 "Socios mujeres"
label variable p1281101 "El productor se organiza o pertenece a una unión agrícola regional"
label variable p1281102 "Años de constitución"
label variable p1281201 "El productor se organiza o pertenece a una unión ganadera regional"
label variable p1281202 "Años de constitución"
gen farm_part_association = 1 if p1281001 == 1 | p1281101 == 1
replace farm_part_association = 0 if missing(farm_part_association)
*label variable p1281301 "El productor se organiza o pertenece a una asociación de silvicultores"
*label variable p1281302 "Años de constitución"
label variable p1281401 "El productor se organiza o pertenece a una unión de sociedades de producción rural"
label variable p1281402 "Años de constitución"
label variable p1281502 "El productor se organiza o pertenece a otra forma de organización"
label variable p1281503 "Años de constitución"
*label variable p1281504 "Cantidad de socios"
*label variable p1281505 "Socios mujeres"
label variable p1290101 "Apoyo para la compra de insumos"
rename p1290101 assistance_insumos 
label variable p1290201 "Apoyo para asistencia técnica"
rename p1290201 assistance_tech
label variable p1290301 "Apoyo para la producción por contrato"
rename p1290301 assistance_contract
label variable p1290401 "Apoyo para el procesamiento y transformación de la producción"
rename p1290401 assistance_processing
label variable p1290501 "Apoyo para la comercialización"
rename p1290501 assistance_comercial
label variable p1290601 "Apoyo para seguro agropecuario"
rename p1290601 assistance_insur
label variable p1290701 "Apoyo para cobertura de precios"
rename p1290701 assistance_prices
label variable p1290801 "Apoyo para financiamiento"
rename p1290801 assistance_financing
label variable p1290902 "Algún otro tipo de apoyo"
label variable p1300101 "Participa en un comité de sistema producto"
rename p1300101 part_product_committee
label variable p1300201 "Participa en un consejo municipal de desarrollo rural sustentable"
label variable p1300301 "Participa en una asociación agrícola"
rename p1300301 part_ag_association
label variable p1300401 "Participa en una asociación ganadera"
label variable p1300501 "Participa en organizaciones campesinas"
label variable p1300601 "Participa en organizaciones empresariales"
label variable p1300702 "Participa en alguna otra asociación u organización"

keep id_cup years* farm* assistance* part*
save "`ag_census_clean_07_dir'membership", replace

************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************

***********************
*(11) MIXED CROPS
***********************

************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************
************************************************************************************************************************************************************



*2007 Land area by use crop in agriculture, spring-summer cycle (side by side plantings)
*There are 102,734 plots that use side by side agriculture of multiple crops.
use "`intercalado_pv_07'", clear
gen type = "p-v"
drop if cultivo_1 == " "
drop D_R
rename p009n02 cve_cultivo1
rename p009n04 cve_cultivo2
rename p009n06 cve_cultivo3
rename p009n10 ha_intercalado

*2007 Land area by use crop in agriculture, fall-winter cycle (side by side plantings)
*There are 5,727 plots that use side by side agriculture of multiple crops.
append using "`intercalado_oi_07'"
drop if cultivo_1 == " "
rename p011n02 cve_cultivo1_oi
rename p011n04 cve_cultivo2_oi
rename p011n06 cve_cultivo3_oi
rename p011n10 ha_intercalado_oi
replace cve_cultivo1 = cve_cultivo1_oi if missing(cve_cultivo1)
replace cve_cultivo2 = cve_cultivo2_oi if missing(cve_cultivo2)
replace cve_cultivo3 = cve_cultivo3_oi if missing(cve_cultivo3)
replace ha_intercalado = ha_intercalado_oi if missing(ha_intercalado)
drop D_R cve_cultivo*_oi ha_intercalado_oi
replace type = "o-i" if missing(type)

*2007 Land area by use crop in agriculture, perennials
*There are 47,835 plots that use perennials side by side cultivation of multiple crops.
append using "`intercalado_peren_07'"
drop if cultivo_1 == " "
rename p013n02 cve_cultivo1_peren
rename p013n04 cve_cultivo2_peren
rename p013n06 cve_cultivo3_peren
rename p013n10 ha_intercalado_peren
replace cve_cultivo1 = cve_cultivo1_peren if missing(cve_cultivo1)
replace cve_cultivo2 = cve_cultivo2_peren if missing(cve_cultivo2)
replace cve_cultivo3 = cve_cultivo3_peren if missing(cve_cultivo3)
replace ha_intercalado = ha_intercalado_peren if missing(ha_intercalado)
drop D_R cve_cultivo*_peren ha_intercalado_peren
replace type = "peren" if missing(type)

*2007 Output of potted seedling crops (vivero)
*Only 12,844 vivieros (small plants production)
append using "`plantas_vivero_07'"
drop if nombre_cul==" "
replace cve_cultivo1 = p027n01 if missing(cve_cultivo1)
replace cultivo_1 = nombre_cul if missing(cultivo_1)
rename p027n05 quantity_vivero 
drop D_R p027n01 nombre_cul
replace  type = "vivero" if missing(type) 
save "`ag_census_clean_07_dir'cultivos_intercalados.dta", replace
*/

*** More data
*** "`training'"
*** "`rights'"
*** "`losses'"
*** "`trees'"
*** "`jungle'"
*** "`ownership'"
