
import delimited "data/base_data/country_labels.csv", clear
rename iso3 importer_iso3
save temp_imp_lab, replace

rename importer_iso3 exporter_iso3
save temp_exp_lab, replace


import delimited "data/ITPDS/ITPD_S_R1.1_2019.csv", clear


collapse (sum) trade, by (exporter_iso3 importer_iso3 broad_sector)

merge m:1 importer_iso3 using temp_imp_lab
keep if _m==3
keep exporter_iso3 importer_iso3 broad_sector trade

merge m:1 exporter_iso3 using temp_exp_lab
keep if _m==3
keep exporter_iso3 importer_iso3 broad_sector trade

sort broad_sector importer_iso3 exporter_iso3 
export delimited using "data/ITPDS/trade_ITPD.csv", novarnames replace


erase temp_exp_lab.dta
erase temp_imp_lab.dta
