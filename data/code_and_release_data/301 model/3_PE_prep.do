****************************************
*** Set directory and open data file ***
****************************************
cd "${mainpath}"

*************************************
*** Save annual data for PE model ***
*************************************

use D_all_data, clear
replace duty301_scaled = 0 if duty301_scaled==.
gen val_duties = val*duty301_scaled
collapse (sum) val_duties val, by(year ctryname naics4)
gen duty = val_duties/val
rename ctryname source
save PE_duties, replace

use D_all_source_district, clear
replace cv = cv/(10^9)

preserve
collapse (sum) cv, by(source naics4)
replace cv=0 if source=="China"
egen source_rank = rank(-cv), by(naics4)
gen source_top = source if source_rank<=3
replace source_top = "China" if source=="China"
replace source_top = "ROW" if source_top==""
replace source_rank = 0 if source_top=="China"
replace source_rank = 4 if source_top=="ROW"
keep naics4 source source_top source_rank
tempfile topsectors
save `topsectors'
restore

merge m:1 naics4 source using `topsectors', keep(1 3) nogen
collapse (sum) cv, by(year source_top naics4)
rename source_top source
merge 1:1 year source naics4 using PE_duties, keep(1 3) nogen
keep year source naics4 cv duty
replace duty = 0 if duty==.

save PE_trade_data, replace

********************************************************************************
*** Estimate elasticities ******************************************************
********************************************************************************
use D_all_source_district, clear

***********************************
*** Define additional variables ***
***********************************
gen lcr = ldpv/cv
label variable lcr "LDPV/CV ratio"

gen log_ldpv = log(ldpv)
gen log_lcr = log(lcr)

******************************
*** Generate fixed effects ***
******************************
egen sit = group(naics4 source year)
egen sdt = group(naics4 district year)

******************************
*** Regressions by NAICS-4 ***
******************************

gen sigma = .
gen sigma_se = .
levelsof naics4, local(sectors) 
foreach n4 of local sectors {
	di "Sector: `n4'"
	capture: reghdfe log_ldpv log_lcr if naics4 == "`n4'", a(sit sdt) vce(robust)
	if _rc==0 {
		replace sigma = 1-_b[log_lcr] if naics4 == "`n4'"
		replace sigma_se = _se[log_lcr] if naics4 == "`n4'"
	}
}
preserve

* Generate a pooled average.
reghdfe log_ldpv log_lcr, a(sit sdt) vce(robust)
gen sigma_pool = 1-_b[log_lcr]
gen sigma_se_pool = _se[log_lcr]

collapse (mean) sigma sigma_se sigma_pool sigma_se_pool, by(naics4)
save PE_sigma, replace
restore

********************************************************************************
*** Domestic data **************************************************************
********************************************************************************
import excel "D_GO_by_NAICS.xlsx", sheet("GO_by_NAICS") firstrow case(lower) clear
rename (d e f g h i) (go2016 go2017 go2018 go2019 go2020 go2021)
rename naicscode naics6
drop line name
reshape long go, i(naics6) j(year)
gen naics4 = substr(naics6,1,4)
collapse (sum) go, by(naics4 year)
replace go = go/(10^3)
gen source="United States"
gen duty=0
rename go cv
save PE_domestic_data, replace






