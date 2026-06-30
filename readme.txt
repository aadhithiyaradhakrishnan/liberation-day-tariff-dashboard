# Description

This package contains the code and data necessary to replicate the tables and figure in
"Making America Great Again: The Economic Impacts of Liberation Day Tariffs"
for publication in the Journal of International Economics (latest update: July 2025)
Authors: Anna Ignatenko, Luca Macedoni, Ahmad Lashkaripour, Ina Simonovska


## Directory Structure

```
replication_package/
├── code/              # Source code for compiling data and running analysis
│   ├── analysis/      # executes the counterfactual policy simulations in Matlab and produces Tables 1-4 and 8-11
│   ├── build_data/    # builds data for the Matlab simulations + generates Table 7
│   ├── global_map/    # creates figure 1 using Python
│ 
│   
├── data/            	           # Data files
│   ├── Dynamic_Gravity_Database/  # gravity data for trade elasticity estimation
│   ├── ITPDS/          	   # sectoral Trade Data
│   ├── base_data/      	   # aggregate trade and tariffs data for baseline analysis
│   └── sectoral_tariffs/  	   # sector-specific tariff data
│ 
│ 
├── output/        # Analysis outputs
├── readme.txt     # This file
│ 
│ 
├── run_all_matlab.m  # MATLAB master script
└── run_all_stata.do  # Stata master script
```

## Prerequisites

To use this package, you will need:
- MATLAB (tested on Matlab 2024b)
- Stata (tested on Stata 18)

## Getting Started

1. **Data Sources**
   - The package includes raw data files in the `data/` directory
   - Key data sources:
     - BACI HS22 Y2023 V202501: Trade flow data
     - World Development Indicators GDP data
     - USTR tariff calculations (https://tariffs.inasimonovska.com)
     – International Trade and Production Database for Simulation (ITPD-S)  
     – Dynamic Gravity Dataset (DGD)
     – Feodora Teti's Global Tariff Database (v_beta1-2024-12) from Teti (2024)

2. **Running the Analysis**

   **IMPORTANT**: The scripts must be run in this specific order:
   
   a. First run the Stata analysis:
   ```
   stata run_all_stata.do
   ```
   - This script processes raw data and generates the baseline simulation data
   - Output will be saved in the `base_data/` directory

   b. Then run the MATLAB analysis:
   ```
   matlab run_all_matlab.m
   ```
   - This script uses the processed data from Stata
   - Results will be saved in the `output/` directory


## Contact

For questions with the replication package, please email Ahmad Lashkaripour at ahmadlp@gmail.com
