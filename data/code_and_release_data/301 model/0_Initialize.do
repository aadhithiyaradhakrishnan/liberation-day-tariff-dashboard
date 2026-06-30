*** Set the actual filepath here. Be sure that there is a Results subfolder. ***
* Run only lines 3 and 4 if you want to run the other files manually.
global mainpath "C:\users\Name\Documents\Code Release"
cd "${mainpath}"

* Calculate and export various descriptive statistics from the trade and tariff data.
do 1_data_description

* Run ARW-style regressions to find the elasticity of trade outcomes to the tariffs.
do 2_eventstudy

* Produce the files necessary to run the partial equilibrium model in Python.
do 3_PE_prep