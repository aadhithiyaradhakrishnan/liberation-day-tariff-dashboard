# Data Files Inventory

This document lists the main data files and folders in `data/` and their purpose. Filenames were updated for clarity.

## Extension / project data (renamed)

| Current name | Purpose |
|--------------|---------|
| **pharma_bilateral_trade.xlsx** | Bilateral trade flows (pharma-related); product-level trade for sector exposure and trade-weighted tariffs. |
| **us_tariff_schedule_2025_hts8.csv** | U.S. tariff schedule at HTS8 level (2025); MFN and partner-specific rates; used for sector-level tariff shocks. |
| **retail_prices_illustrative.csv** | Illustrative product-level “before/after tariff” prices for retail; used for simple examples only. |
| **daily_price_indices_cavallo_etal.csv** | Daily price indices (Cavallo et al.–style) for US, Canada, Mexico, China; retail/consumer price evidence. |
| **ICIO_small_ReadMe.xlsx** | OECD ICIO “small” documentation (RowItems, ColItems, sector/country codes). |
| **OECD_ICIO_SML_2016_2022/** | OECD ICIO small symmetrical tables by year (e.g., 2018_SML.csv); global IO matrices for supply-chain calibration. |
| **WIOD_WIOT_Excel/** | WIOD world input–output tables in Excel (2000–2014); optional robustness / historical context. |
| **code_and_release_data/** | USITC 232 and 301 model code and data (steel/aluminum, manufacturing GO, price indices). |

## Replication package (do not rename)

Files and folders from the official Mendeley replication package for “Making America Great Again? The Economic Impacts of Liberation Day Tariffs” (e.g., `base_data/`, ITPDS, sectoral_tariffs) keep their original names. See README and DATA_README.md for paths and download instructions.

## Old → new name reference

| Old name | New name |
|----------|----------|
| TradeData.xlsx | pharma_bilateral_trade.xlsx |
| tariff_database_2025.csv | us_tariff_schedule_2025_hts8.csv |
| Tariff.csv | retail_prices_illustrative.csv |
| Cavallo_Llamas_Vazquez_countries.csv | daily_price_indices_cavallo_etal.csv |
| ReadMe_ICIO_small.xlsx | ICIO_small_ReadMe.xlsx |
| 2016-2022_SML/ | OECD_ICIO_SML_2016_2022/ |
| WIOTS_in_EXCEL/ | WIOD_WIOT_Excel/ |

**Note:** Any code or scripts that load these files by name must use the new names (or be updated).
