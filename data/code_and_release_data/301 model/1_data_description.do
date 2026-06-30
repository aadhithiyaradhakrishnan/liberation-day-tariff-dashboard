****************************************
*** Set directory and open data file ***
****************************************
cd "${mainpath}"
use D_all_data, clear

*********************
*** Basic cleanup ***
*********************
* Convert import values to billions of dollars
replace val = val/(10^9)

sort hts10 modate

*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~* General information *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

***********************************************
*** Tariff coverage (2016 and 2017 vs 2021) ***
***********************************************
preserve
gen pre = 1 if year<2018
replace pre=0 if pre!=1
keep if (pre==1 | year==2021) & target_ever301==1
egen count_year = tag(year pre action301)
egen count_affected = tag(hts10 pre)
gen val_duty = val*duty301_scaled
collapse (sum) val val_duty count_affected count_year, by(pre action301)
gen duty = 100*val_duty/val
replace val = val/count_year
drop val_duty count_year
reshape wide val count_affected duty, i(action301) j(pre)
keep action301 val1 count_affected1 val0 duty0
order action301 count_affected1 duty0 val1 val0
export excel using Results\data_description_301.xlsx, sheet("tariff_coverage") sheetmodify firstrow(variables)
restore


*************************************
*** Imports by source, 301 status ***
*************************************
preserve
gen hts10_affected = 1 if action301!="" & action301!="99038816"
keep if hts10_affected==1
gen china = 0
replace china = 1 if ctryname=="China"
collapse (sum) val, by(year china)
reshape wide val, i(china) j(year)
export excel using Results\data_description_301.xlsx, sheet("annual_imports") sheetmodify firstrow(variables)
restore

*********************************************
*** Figure: imports by source, 301 status ***
*********************************************
preserve
gen mtype = 0
replace mtype=1 if ctryname=="China"
replace mtype=2 if ctryname=="China" & target_current301==1
collapse (sum) val, by(modate mtype)
reshape wide val, i(modate) j(mtype)
label var val0 "Rest of world"
label var val1 "China nonsubject"
label var val2 "China subject"
export excel using Results\data_description_301.xlsx, sheet("fig_imports") sheetmodify firstrow(varlabels)
restore

****************************************************
*** Figure: Changes in AUV for affected products ***
****************************************************
preserve
gen china = 0
replace china=1 if ctryname=="China"
gen hts10_affected = 1 if action301!="" & action301!="99038816"
keep if hts10_affected==1 & !missing(p)
egen first_tag = tag(hts10 ctryname)
gen normp_set1 = p*first_tag
egen normp_set2 = max(normp_set1), by(hts10 ctryname)
gen normp = 100*p/normp_set2
gen normpduty = normp*(1+duty301_scaled)

by modate, sort: egen plo = pctile(normp), p(5)
by modate, sort: egen phi = pctile(normp), p(95)
keep if inrange(normp, plo, phi)

replace normp = normp*q1
replace normpduty = normpduty*q1
collapse (sum) q1 normp normpduty, by(modate china)
replace normp = normp/q1
replace normpduty = normpduty/q1
drop q1

reshape wide normp normpduty, i(modate) j(china)
drop normpduty0
label var normp0 "Rest of world"
label var normp1 "China not including tariff"
label var normpduty1 "China including tariff"
export excel using Results\data_description_301.xlsx, sheet("fig_auv") sheetmodify firstrow(varlabels)
restore

*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~* Sector-specific *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
gen topsector = 0
foreach n in "3152" "3344" "3341" "3371" "3363" "3359" "3399" "3343" "3339" "3261" {
	replace topsector=1 if naics4=="`n'"
}
keep if topsector==1

***************************
*** Most-affected NAICS ***
***************************
preserve
keep if ctryname=="China"
merge m:1 naics4 using D_naics4_descriptions, nogen keep(3)
replace naics4_description = proper(naics4_description)
gen val_affected = val*target_ever301
gen val_duties = val*duty301_scaled
collapse (sum) val_affected val val_duties (first) naics4_description, by(year naics4)
gen duties = val_duties/val
drop val_duties

gen tar_set = duties if year==2020
egen tar = max(tar_set), by(naics4)
replace tar = 100*tar
drop tar_set duties
keep if year==2017 | year==2016
collapse (sum) val_affected val (mean) tar (first) naics4_description, by(naics4)
replace val_affected = val_affected/2
replace val = val/2

order naics4 naics4_description val_affected val tar
gsort -val_affected
label var naics4 "NAICS"
label var naics4_description "Description"
label var val_affected "Value of affected imports (\$B)"
label var val "Value of all imports (\$B)"
label var tar "Average tariff in 2020"
export excel using Results\data_description_301.xlsx, sheet("top_sectors") sheetmodify firstrow(varlabels)
restore

*************************************
*** Figure: Import share by NAICS ***
*************************************
levelsof naics4, local(sectors)
foreach n of local sectors	{
	preserve
	keep if naics4=="`n'"
	gen mtype = 0
	replace mtype=1 if ctryname=="China"
	replace mtype=2 if ctryname=="China" & target_current301==1
	collapse (sum) val, by(modate mtype)
	reshape wide val, i(modate) j(mtype)
	label var val0 "Other"
	label var val1 "China nonsubject"
	label var val2 "China subject"
	export excel using Results\fig_imports_301.xlsx, sheet("imports_`n'") sheetmodify firstrow(varlabels)
	restore
}

***********************************
*** Table: Top sources by NAICS ***
***********************************
preserve
collapse (sum) val, by(ctryname naics4)
egen source_rank=rank(-val) if ctryname!="China", by(naics4)
gen source_top=ctryname if source_rank<=3 | ctryname=="China"
replace source_top = "ROW" if source_top==""
replace source_rank = 0 if source_top=="China"
replace source_rank = 4 if source_top=="ROW"
collapse (sum) val, by(naics4 source_top source_rank)
sort naics4 source_rank
reshape wide source_top val, i(naics4) j(source_rank)
export excel using Results\data_description_301.xlsx, sheet("tab_top_sources") sheetmodify firstrow(varlabels)
restore

********************************************************************************
*** Sector-specific subsections ************************************************
********************************************************************************

* Build domestic data file at NAICS-4 level and combine with trade data
preserve
import excel "D_GO_by_NAICS.xlsx", sheet("GO_by_NAICS") firstrow case(lower) clear
rename (d e f g h i) (val2016 val2017 val2018 val2019 val2020 val2021)
rename naicscode naics4
replace naics4 = substr(naics4,1,4)
collapse (sum) val2016 val2017 val2018 val2019 val2020 val2021, by(naics4)
reshape long val, i(naics4) j(year)
replace val = val/(10^3)
replace naics4="3152" if naics4=="3150"
replace naics4="3162" if naics4=="3160"
tempfile dom_data
save `dom_data', replace

import excel "D_price_indices.xlsx", sheet("BLS PPI") firstrow case(lower) clear
rename annual* plevel*
drop plevel2022
reshape long plevel, i(naics4) j(year)
tostring naics4, replace
merge 1:1 naics4 year using `dom_data', keep(3) nogen
gen source="United States"
reshape wide val plevel, i(naics4 source) j(year)
save `dom_data', replace
restore

* Collapse import data to val and price level 
gen source=ctryname
drop if year>=2022
replace source="Rest of world" if source!="United States" & source!="China"
collapse (sum) val q1, by(hts10 naics4 year source)
gen p = val/q1
egen first_tag = tag(hts10 source)
gen normp_set1 = p*first_tag
egen normp_set2 = max(normp_set1), by(hts10 source)
gen normp = 100*p/normp_set2
gen normval = val*(q1!=.)

replace normp = normp*normval
collapse (sum) val normval normp, by(naics4 year source)
replace normp = normp/normval
drop normval
rename normp plevel

reshape wide val plevel, i(naics4 source) j(year)
append using `dom_data'

forvalues yy=21(-1)16 {
	* This makes 2016 the "base year" with a value of 100
	replace plevel20`yy' = 100*plevel20`yy'/plevel2016
}

*********************
*** Import values ***
*********************

levelsof naics4, local(sectors)
foreach n of local sectors	{
	preserve
	keep if naics4=="`n'"
	drop plevel*
	gen mtype = 0 if source=="China"
	replace mtype = 1 if source=="United States"
	replace mtype = 2 if source=="Rest of world"
	gsort mtype
	drop mtype
	export excel using Results\sector_subsections_imports.xlsx, sheet("imports_`n'") sheetmodify firstrow(varlabels) keepcellfmt
	restore
}

**************************
*** Average unit value ***
**************************

levelsof naics4, local(sectors)
foreach n of local sectors	{
	preserve
	keep if naics4=="`n'"
	drop val*
	gen mtype = 0 if source=="China"
	replace mtype = 1 if source=="United States"
	replace mtype = 2 if source=="Rest of world"
	gsort mtype
	drop mtype
	export excel using Results\sector_subsections_auv.xlsx, sheet("auv_`n'") sheetmodify firstrow(varlabels) keepcellfmt
	restore
}