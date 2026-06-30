cls
clear all

********************************************************************************
********************************************************************************
* GDP Data
import excel "data/base_data/raw_data/GDP_wdi_raw.xls", sheet("Data_stata") firstrow clear
reshape long gdp, i(CountryCode) j(year) string
destring year, replace
rename CountryCode iso3
drop if gdp==.
bysort iso3 : egen max_year=max(year)
* Take latest GDP available
keep if max_year==year
* Aggregate countries
replace iso3="FRA" if iso3=="MCO"
replace CountryName="France" if CountryName=="Monaco" 
replace iso3="CHE" if iso3=="LIE"
replace CountryName="Switzerland" if CountryName=="Liechtenstein" 
replace iso3="USA" if iso3=="VIR"
replace CountryName="United States" if CountryName=="Virgin Islands (U.S.)" 
replace iso3="USA" if iso3=="PRI"
replace CountryName="United States" if CountryName=="Puerto Rico" 
collapse (sum) gdp (max) max_year, by(iso3 CountryName)

* Correct names to merge with tariff calculator
replace CountryName = "Syria" if CountryName == "Syrian Arab Republic"
replace CountryName = "Iran" if CountryName == "Iran, Islamic Rep."
replace CountryName = "Republic of Yemen" if CountryName == "Yemen, Rep."
replace CountryName = "Sint Maarten" if CountryName == "Sint Maarten (Dutch part)"
replace CountryName = "St Vincent and the Grenadines" if CountryName == "St. Vincent and the Grenadines"
replace CountryName = "Venezuela" if CountryName == "Venezuela, RB"
replace CountryName = "St Lucia" if CountryName == "St. Lucia"
replace CountryName = "Macau" if CountryName == "Macao SAR, China"
replace CountryName = "South Korea" if CountryName == "Korea"
replace CountryName = "Laos" if CountryName == "Lao PDR"
replace CountryName = "Micronesia" if CountryName == "Micronesia, Fed. Sts."
replace CountryName = "Hong Kong" if CountryName == "Hong Kong SAR, China"
replace CountryName = "Congo (Brazzaville)" if CountryName == "Congo, Rep."
replace CountryName = "Egypt" if CountryName == "Egypt, Arab Rep."
replace CountryName = "Gambia" if CountryName == "Gambia, The"
replace CountryName = "Burma" if CountryName == "Myanmar"
replace CountryName = "St Kitts and Nevis" if CountryName == "St. Kitts and Nevis"
replace CountryName = "Kyrgyzstan" if CountryName == "Kyrgyz Republic"
replace CountryName = "Congo (Kinshasa)" if CountryName == "Congo, Dem. Rep."
replace CountryName = "East Timor" if CountryName == "Timor-Leste"
replace CountryName = "Turkey" if CountryName == "Turkiye"
replace CountryName = "Vietnam" if CountryName == "Viet Nam"
replace CountryName = "Bahamas" if CountryName == "Bahamas, The"
replace CountryName = "Brunei" if CountryName == "Brunei Darussalam"
replace CountryName = "South Korea" if CountryName == "Korea, Rep."

save "data/base_data/raw_data/GDP_all.dta", replace
* 258 countries and aggregation of countries

********************************************************************************
********************************************************************************
import excel "data/base_data/raw_data/calculator_countries.xlsx", sheet("Foglio1") firstrow clear
merge 1:1 CountryName using "data/base_data/raw_data/GDP_all.dta"
drop if _merge==1
*Cook Islands: not in WDI
*St Pierre and Miquelon: not in WDI
*Taiwan: not in WDI
*Wallis and Futuna: not in WDI
drop _merge
* EU countries
merge 1:1 iso3 using "data/base_data/raw_data/EU_countries.dta"
replace Tariff_formula=0.39 if _merge==3
replace applied_tariff=0.2 if _merge==3
drop _merge
* Set zero tariffs for US and Russia, to have them in the dataset
replace Tariff_formula=0 if iso3=="USA"
replace applied_tariff=0 if iso3=="USA"
replace Tariff_formula=0 if iso3=="RUS"
replace applied_tariff=0 if iso3=="RUS"

drop if applied_tariff==.

drop if CountryName=="European Union"
* Left with 194 countries

drop max_year country_names
order iso3 CountryName
save "data/base_data/raw_data/country_data.dta", replace

********************************************************************************
********************************************************************************
* Trade Data
* country codes
import delimited "data/base_data/raw_data/country_codes_V202501.csv", clear 
save "data/base_data/raw_data/cepii23_countrycodes.dta" , replace 
* trade data 
import delimited "data/base_data/raw_data/BACI_HS22_Y2023_V202501.csv", clear 
* Sum across all products
collapse (sum) export_ij=v, by(t i j)
drop t
rename i country_code
merge m:1 country_code using "data/base_data/raw_data/cepii23_countrycodes.dta" 
keep if _merge==3
drop _merge
rename country_iso3 iso3_o
drop country_code country_name country_iso2  

rename j country_code
merge m:1 country_code using "data/base_data/raw_data/cepii23_countrycodes.dta" 
keep if _merge==3
drop _merge
rename country_iso3 iso3_d
drop country_code country_name country_iso2  
 
collapse (sum) export_ij, by(iso3_o iso3_d)
drop if iso3_o==iso3_d
save "data/base_data/raw_data/cepii_trade_23_all.dta", replace
********************************************************************************
********************************************************************************
* Country list
use "data/base_data/raw_data/country_data.dta", clear
keep iso3
sort iso3
gen iso_code=_n 
save "data/base_data/raw_data/ctry_list_calc.dta", replace
* US ID = 185
* CAN = 31
* MEX = 115
* RUS = 150
********************************************************************************
********************************************************************************
* Compile GDP and Trade data including only those 205 countries
* GDP
use "data/base_data/raw_data/ctry_list_calc.dta", clear
merge 1:1 iso3 using "data/base_data/raw_data/country_data.dta"
keep if _merge==3
drop _merge
sort iso_code
keep gdp
export delimited using "data/base_data/gdp.csv", replace
********************************************************************************
* Trade flows
* First, we create a dataset with all possible country pairs
use "data/base_data/raw_data/ctry_list_calc.dta", clear
keep iso3 iso_code
drop iso_code
gen dummy = 1
tempfile orig
save `orig', replace
rename iso3 iso3_o
joinby dummy using `orig'
rename iso3 iso3_d
* Drop ii pairs
drop if iso3_o == iso3_d
drop dummy
* merge with CEPII data
merge 1:1 iso3_o iso3_d using "data/base_data/raw_data/cepii_trade_23_all.dta"
drop if _merge==2
keep iso3_o iso3_d export_ij

* Restrict sample of origins
rename iso3_o iso3
* keep only countries in the list
merge m:1 iso3 using "data/base_data/raw_data/ctry_list_calc.dta"
drop if _merge==2
drop _merge
rename iso3 iso3_o
rename iso_code iso3_id_o
 

* Restrict sample of destinations
rename iso3_d iso3
* keep only countries in the list
merge m:1 iso3 using "data/base_data/raw_data/ctry_list_calc.dta"
drop if _merge==2
drop _merge
rename iso3 iso3_d
rename iso_code iso3_id_d
 

sort iso3_id_o

save "data/base_data/raw_data/bilateral_cepii_calc.dta", replace

* File for matlab
drop iso3_o iso3_d
reshape wide export_ij , i(iso3_id_o) j(iso3_id_d)
* Many columns are empty
save "data/base_data/raw_data/pairs_cepii_calc.dta", replace
drop iso3_id_o 
export delimited using "data/base_data/trade_cepii.csv", replace
********************************************************************************
********************************************************************************
* EU country codes (select the country numerical ids for calculation of tariffs)
use "data/base_data/raw_data/ctry_list_calc.dta", clear
merge 1:1 iso3 using "data/base_data/raw_data/EU_countries.dta"
keep if _merge ==3
drop _merge
save "data/base_data/raw_data/eu_list_calc.dta", replace
********************************************************************************
********************************************************************************
* Tariff data (uniform the tariff data with the numberical country id)
use "data/base_data/raw_data/ctry_list_calc.dta", clear
merge 1:1 iso3 using "data/base_data/raw_data/country_data.dta"
keep if _merge==3
drop _merge
sort iso_code
keep applied_tariff
export delimited using "data/base_data/tariffs.csv", replace
********************************************************************************
********************************************************************************
* Country lables
use "data/base_data/raw_data/ctry_list_calc.dta", clear
merge 1:1 iso3 using "data/base_data/raw_data/country_data.dta"
keep if _merge==3
drop _merge
keep iso3 iso_code CountryName
export delimited using "data/base_data/country_labels.csv", replace

********************************************************************************
********************************************************************************
* Share of CEPII flows
use "data/base_data/raw_data/cepii_trade_23_all.dta", clear
merge 1:1 iso3_o iso3_d using "data/base_data/raw_data/bilateral_cepii_calc.dta" 
drop if _merge==2
replace export_ij=0 if export_ij==.

collapse (sum) export_ij, by(_merge)
egen sumtot=sum(export_ij)
gen share=export_ij/sumtot
gen variable_descr="Trade Included"
replace variable_descr="Trade Excluded" if _merge==1
list variable_descr share

* Clean up
erase "data/base_data/raw_data/GDP_all.dta"
erase "data/base_data/raw_data/cepii23_countrycodes.dta"
erase "data/base_data/raw_data/cepii_trade_23_all.dta"
erase "data/base_data/raw_data/country_data.dta"
erase "data/base_data/raw_data/pairs_cepii_calc.dta"
erase "data/base_data/raw_data/bilateral_cepii_calc.dta"
