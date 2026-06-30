
* This script reads and cleans data for analysis in Matlab and produces Table 7 in 
* "Making America Great Again? The Economic Impacts of Liberation Day Tariffs" 
* by Anna Ignatenko, Luca Macedoni, Ahmad Lashkaripour, Ina Simonovska
* for publication in the Journal of International Economics (Last update: July 2025)
* email for inquiries: ahmadlp@gmail.com

clear all
cls
version 18
di "$S_DATE $S_TIME"

ssc install estout
ssc install reghdfe

* ------ Set Path -------
global current_dir "`c(pwd)'"  
cd "$current_dir"

* ------ Read and prepare aggregate Data -------
do code/build_data/step0_build_agg_data

* ------ Read and prepare sectoral Data -------
do code/build_data/step1_build_sectoral_data

* ------ Run the Trade Elasticity Estimation -------
do code/build_data/step2_elasticity_estimation

* ------ Erase Intermediate Data -------
erase "data/ITPDS/estimation_sample.dta"
