# Large Data Files

Due to GitHub file size limitations, the following large data files (6GB+) are **not included** in this repository. You will need to download them separately to run the full replication.

## Excluded Files

The following files are listed in `.gitignore` and must be obtained separately:

### Sectoral Tariffs (5.2 GB total)
- `data/sectoral_tariffs/tariff_isic33_88_21_vbeta1-2024-12.csv` (2.7 GB)
- `data/sectoral_tariffs/tariff_isic33_88_21_vbeta1-2024-12.dta` (2.5 GB)
- `data/sectoral_tariffs/ref_isic33_vbeta1-2024-12.csv`
- `data/sectoral_tariffs/ref_isic33_vbeta1-2024-12.dta`

### Trade Data (1.4 GB total)
- `data/ITPDS/ITPD_S_R1.1_2019.csv` (966 MB)
- `data/ITPDS/estimation_sample.dta` (67 MB)
- `data/base_data/raw_data/BACI_HS22_Y2023_V202501.csv` (348 MB)
- `data/Dynamic_Gravity_Database/release_2.1_2015_2019.csv` (125 MB)

## How to Obtain the Data

### Option 1: Original Replication Package
If you have access to the original MATLAB replication package, simply copy these files from the original `data/` folder to the corresponding locations in this repository.

### Option 2: Download from Sources
- **ITPD (International Trade and Production Database)**: https://www.usitc.gov/data/gravity/itpd.htm
- **BACI (Base pour l'Analyse du Commerce International)**: http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37
- **Dynamic Gravity Database**: https://www.usitc.gov/data/gravity/dgd.htm
- **Sectoral Tariffs**: Contact the paper authors or check the original replication package

## Directory Structure

After obtaining the files, your directory structure should look like:

```
data/
├── Dynamic_Gravity_Database/
│   └── release_2.1_2015_2019.csv
├── ITPDS/
│   ├── ITPD_S_R1.1_2019.csv
│   ├── estimation_sample.dta
│   └── trade_ITPD.csv (included in repo)
├── base_data/
│   ├── raw_data/
│   │   ├── BACI_HS22_Y2023_V202501.csv
│   │   └── ... (other files included in repo)
│   ├── country_labels.csv (included)
│   ├── gdp.csv (included)
│   └── tariffs.csv (included)
└── sectoral_tariffs/
    ├── tariff_isic33_88_21_vbeta1-2024-12.csv
    ├── tariff_isic33_88_21_vbeta1-2024-12.dta
    ├── ref_isic33_vbeta1-2024-12.csv
    └── ref_isic33_vbeta1-2024-12.dta
```

## Verifying Your Setup

After placing the files, you can verify your setup by running:

```bash
python3 run_all_python.py
```

The script will fail with clear error messages if any required data files are missing.

## Questions?

If you have trouble obtaining the data files, please:
1. Contact the paper authors
2. Open an issue on this repository
3. Email: ahmadlp@gmail.com (from the original readme)
