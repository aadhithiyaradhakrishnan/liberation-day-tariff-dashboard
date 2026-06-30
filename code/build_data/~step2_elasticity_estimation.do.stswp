cls
clear all
*
import delimited "data/ITPDS/ITPD_S_R1.1_2019.csv", clear

egen sector = group(broad_sector) 

keep exporter_iso3 importer_iso3 industry_id trade sector
rename sector broad_sectors

* ------------------------------------------------------------
* Assign sector codes based on industry_id
* ------------------------------------------------------------

* 1. Create a new variable "Sector" with missing values
generate byte sector = .

* 2. Assign sectors according to industry_id ranges/specifications

* Agriculture (sector = 1): industries 1–26
replace sector = 1 if inrange(industry_id, 1, 26)

* Forestry (sector = 2): industry 27
replace sector = 2 if industry_id == 27

* Fishing (sector = 5): industry 28
replace sector = 5 if industry_id == 28

* Mining of coal (sector = 10): industries 29–30
replace sector = 10 if inrange(industry_id, 29, 30)

* Extraction of crude petroleum and natural gas (sector = 11): industry 31
replace sector = 11 if industry_id == 31

* Mining of iron ores (sector = 13): industry 32
replace sector = 13 if industry_id == 32

* Other mining and quarrying (sector = 14): industry 33
replace sector = 14 if industry_id == 33

* Electricity & Gas (sector = 40): industries 34–35
replace sector = 40 if inrange(industry_id, 34, 35)

* Food products & beverages (sector = 15): industries 36–52
replace sector = 15 if inrange(industry_id, 36, 52)

* Tobacco products (sector = 16): industry 53
replace sector = 16 if industry_id == 53

* Textiles (sector = 17): industries 54–59
replace sector = 17 if inrange(industry_id, 54, 59)

* Wearing apparel; dressing & dyeing of fur (sector = 18): industries 60–61
replace sector = 18 if inrange(industry_id, 60, 61)

* Leather, luggage & footwear (sector = 19): industries 62–64
replace sector = 19 if inrange(industry_id, 62, 64)

* Wood & cork (ex-furniture); straw products (sector = 20): industries 65–69
replace sector = 20 if inrange(industry_id, 65, 69)

* Paper & paper products (sector = 21): industries 70–72
replace sector = 21 if inrange(industry_id, 70, 72)

* Publishing, printing & recorded media (sector = 22): industries 73–78
replace sector = 22 if inrange(industry_id, 73, 78)

* Coke, refined petroleum & nuclear fuel (sector = 23): industries 79–80
replace sector = 23 if inrange(industry_id, 79, 80)

* Chemicals & chemical products (sector = 24): industries 81–90
replace sector = 24 if inrange(industry_id, 81, 90)

* Rubber & plastics products (sector = 25): industries 91–93
replace sector = 25 if inrange(industry_id, 91, 93)

* Non-metallic mineral products (sector = 26): industries 94–101
replace sector = 26 if inrange(industry_id, 94, 101)

* Basic metals (sector = 27): industries 102–103
replace sector = 27 if inrange(industry_id, 102, 103)

* Fabricated metal products (sector = 28): industries 104–108
replace sector = 28 if inrange(industry_id, 104, 108)

* Machinery & equipment n.e.c. (sector = 29): industries 109–123
replace sector = 29 if inrange(industry_id, 109, 123)

* Office, accounting & computing machinery (sector = 30): industry 124
replace sector = 30 if industry_id == 124

* Electrical machinery & apparatus (sector = 31): industries 125–130
replace sector = 31 if inrange(industry_id, 125, 130)

* Radio, TV & communication equipment (sector = 32): industries 131–133
replace sector = 32 if inrange(industry_id, 131, 133)

* Medical, precision & optical instruments (sector = 33): industries 134–137
replace sector = 33 if inrange(industry_id, 134, 137)

* Motor vehicles, trailers & semi-trailers (sector = 34): industries 138–140
replace sector = 34 if inrange(industry_id, 138, 140)

* Other transport equipment (sector = 35): industries 141–147
replace sector = 35 if inrange(industry_id, 141, 147)

* Furniture; manufacturing n.e.c.; recycling (sector = 36): industries 148–153
replace sector = 36 if inrange(industry_id, 148, 153)

tempfile temp_tariff temp_trade temp_gravity


collapse (sum) trade (mean) broad_sectors, ///
           by(exporter_iso3 importer_iso3 sector)	
		   
save `temp_trade.dta'	

import delimited using "data/Dynamic_Gravity_Database/release_2.1_2015_2019.csv", delimiter(comma) varnames(1) clear
     keep year iso3_o iso3_d colony_ever member_gatt_joint member_wto_joint member_eu_joint ///
		  agree_fta agree_eia agree_cu agree_psa agree_pta common_language contiguity distance
	keep if year == 2019	  
	rename iso3_o exporter_iso3
	rename iso3_d importer_iso3
	// WTO and RTA indicators
    gen wto = (member_wto_joint==1) | (member_gatt_joint==1)
	gen rta = (agree_pta==1) | (agree_cu==1) | (agree_fta==1) | (agree_eia==1) | (agree_psa==1)
	replace wto = 0 if exporter_iso3 == importer_iso3
	replace rta = 0 if exporter_iso3 == importer_iso3
	collapse (mean) common_language contiguity distance colony_ever wto rta, by(importer_iso3 exporter_iso3)

save `temp_gravity'	


use "data/sectoral_tariffs/tariff_isic33_88_21_vbeta1-2024-12.dta"
	keep if year == 2019
	keep iso1 iso2 year tariff tariff_w sector
	rename iso2 exporter_iso3
	rename iso1 importer_iso3
	save `temp_tariff'

	
use `temp_trade', clear

    merge 1:1 exporter_iso3 importer_iso3 sector using `temp_tariff'
	
	
	replace _m = 3 if exporter_iso3==importer_iso3
	keep if _m==3
	drop _m	
	replace tariff = 0 if exporter_iso3==importer_iso3
	
	merge m:1 exporter_iso3 importer_iso3 using `temp_gravity'
	keep if _m==3
	
    save data/ITPDS/estimation_sample, replace

		

local sectors  1 2 3 
local titles  `" "Agriculture" "Manufacturing" "Mining"  "'	

foreach sec in `sectors' {	
	
	use data/ITPDS/estimation_sample, replace
	
	local sector_name : word `sec' of `titles'
	
	keep if broad_sectors == `sec'
	rename (exporter_iso3 importer_iso3) (exporter importer)
		
	bys exporter sector: egen flag=total(trade) /* in order to exclude exporters that never export a given sector */
	drop if flag==0
	drop flag
		
	gen ln_tariff = log(1 + tariff/100)
	gen ln_trade  = log(trade)
	gen ln_dist   = log(distance)

gen border = (importer != exporter)
	
egen long fe_imp=group(importer sector)
egen long fe_exp=group(exporter sector)
egen long fe_imp_exp=group(importer exporter)


local fe fe_imp fe_exp

eststo result_`sec': reghdfe ln_trade ln_tariff ln_dist common_language contiguity colony_ever rta border, a(`fe') vce(robust)
	*estadd local Sample_name "`sector_name'"
	estadd local Exporter  "Yes"
	estadd local Importer  "Yes"
	estadd local Border    "Yes"
	estadd local Language  "Yes"
	estadd local Agreement "Yes"
	estadd local Colony   "Yes"
	count
	estadd scalar N_all = `r(N)'
	unique exporter
	estadd scalar N_E = `r(unique)'
	unique importer
	estadd scalar N_I = `r(unique)'
	unique sector
	estadd scalar N_S = `r(unique)'
	


}

esttab result_1 result_2 result_3 ///
            using "output/Table_7.tex", booktabs se label ///
			star(* 0.10 ** 0.05 *** 0.01) drop(_cons common_language contiguity colony_ever rta border) ///
			replace substitute(\_ _) nonotes noobs b(%9.3f) se(%9.3f) ///
			mtitle("Agriculture" "Manufacturing" "Mining") prefoot([2em]) ///
			scalars("N_all  Observations" "N_E Exporters" "N_I Importers" "Exporter Exporter Product FE" "Importer Importer Product FE" "Border Control for Common Border" "Language Control for Common Language" "Agreement Control for Trade Agreement" "Colony Control for Colonial Links") ///
			sfmt(%12.0gc %9.0gc %9.0gc %9.0gc) ///
		    coef(ln_tariff "\$ \ln (1+ t_{ij}) \$" ln_dist "\$ \ln (1+ \text{Dist}_{ij}) \$")
	