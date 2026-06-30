cd "${mainpath}"
use D_all_data, clear

****************************
*** Define fixed effects ***
****************************

egen ht = group(hts10 modate) // product-time
egen ct = group(ctryname modate) // country-time
egen cs = group(ctryname naics4)

*************************************************
*** Create dummies for passthrough regression ***
*************************************************
gen et_trim = et

local etlo = -6
local ethi = 32
local etmax = `ethi' - `etlo' + 1
replace et_trim = . if et<`etlo'
replace et_trim = `ethi' if et>`ethi' & et~=.
/* Uncomment this block to include pre-event effects
* Measure pre-event effects
forval t=`etlo'/-1 {
	local t_tmp = `t'-`etlo'+1
	* Amiti Redding Weinstein variables
	gen arw_`t_tmp' = 0
	replace arw_`t_tmp' = arw_`t_tmp' + 0.25 if et_trim==`t' & action232=="steel" & target_ever232==1
	replace arw_`t_tmp' = arw_`t_tmp' + 0.10 if et_trim==`t' & target_ever==1 & action232=="aluminum" & target_ever232==1
	replace arw_`t_tmp' = arw_`t_tmp' + 0.25 if et_trim==`t' & target_ever==1 & inlist(action301,"99038801","99038802")
	replace arw_`t_tmp' = arw_`t_tmp' + 0.10 if et_trim==`t' & target_ever==1 & inlist(action301,"99038803","99038804")
	replace arw_`t_tmp' = arw_`t_tmp' + 0.15 if et_trim==`t' & target_ever==1 & action301=="99038815"	
}
*/

* Measure post-event effects
forval t=0/`ethi' {
	local t_tmp = `t'-`etlo'+1
	gen arw_`t_tmp' = 0
	replace arw_`t_tmp' = ltf_scaled if et_trim==`t' & target_ever==1
}



*******************
*** Event Study ***
*******************
gen all = 1
egen related = max(target_ever), by(naics4)
gen steel_tmp = 1 if action232=="steel"
replace steel_tmp = steel_tmp*input
egen steel_input = max(steel_tmp), by(naics4)
drop steel_tmp
egen steelalum = max(target_ever232), by(naics4)

*choose subsets to analyze
* use "all" to recreate fig from ch 6, "steel" to recreate fig in appendix G,
* or "all steel" to run both consecutively.
local subsets = "all"
local lhs = "p pduty val q1"

*This step takes a long time to run. Consider looking at fewer lefthand side
* variables. For example, replace the definition of lhs with "p pduty" to only
* look at the effect on prices, not import values or quantities.
foreach subset in `subsets' {
	preserve
	foreach y in `lhs' {
		reghdfe l`y' arw_* if `subset'==1, a(id ct ht) cluster(hts8 ctryname) resid(r`y'_arw)	
		gen b_arw_`y' = .
		gen se_arw_`y' = .
		local j = 1-`etlo' // if not using pre-event effects
		* local j = 1 // if using pre-event effect
		forval i=`j'/`etmax' {
			replace b_arw_`y' 	=   _b[arw_`i'] if et_trim==`i'+`etlo'-1
			replace se_arw_`y'	=   _se[arw_`i'] if et_trim==`i'+`etlo'-1
		}
	}
	collapse (mean) b_* se_*, by(et_trim)
	export excel using "Results\arw_collected.xlsx", sheet("`subset'") firstrow(variables) sheetmodify
	restore
}
