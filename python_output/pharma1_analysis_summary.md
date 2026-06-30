# Pharma1 Tariff Cost And Drug Price Analysis

Objective: analyze how tariffs on imported pharmaceutical inputs increase production costs and drug prices when short-run substitution is limited.

## Data

- Source workbook: `data/Pharma1.xlsx`, sheet `Query Results`.
- Import measure: Customs Value, Imports For Consumption.
- Main baseline year: 2024, the latest complete full year in the workbook.
- Rows used: 191 country-HTS-year rows.
- Total 2024 pharma imports: $210.8B.
- Tariff matching coverage: 99.82% of import value.
- Unmatched countries excluded from tariff weighting: Taiwan ($0.372B), Myanmar (Burma) ($0.000B).

## Key Results

- Trade-weighted Liberation Day tariff: 21.28%.
- Production/input-cost route: +2.47% drug price pressure, using ICIO imported intermediate input share 12.36%, pass-through 88%, and IO multiplier 1.067x.
- Broader import-dependence route: +10.34% drug price pressure, using Pharma1 imports scaled to the project import-dependency benchmark (51.74%).
- Implied import volume response: -40.4%, using elasticity 2.3.
- Supplier HHI: 1047 pre-tariff to 1065 post-tariff (18 change).
- Low-tariff supplier share (tariff <= 10%): 14.5% of 2024 import value.

## Top Suppliers, 2024

| Rank | Country | Imports | Share | Tariff | Post-Tariff Share | Shift |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Ireland | $54.7B | 25.9% | 20% | 26.2% | 0.30pp |
| 2 | Switzerland | $19.2B | 9.1% | 31% | 7.5% | -1.58pp |
| 3 | Singapore | $16.6B | 7.9% | 10% | 9.7% | 1.85pp |
| 4 | Germany | $16.2B | 7.7% | 20% | 7.8% | 0.09pp |
| 5 | Belgium | $12.3B | 5.9% | 20% | 5.9% | 0.07pp |
| 6 | India | $12.3B | 5.8% | 27% | 5.2% | -0.66pp |
| 7 | Italy | $11.2B | 5.3% | 20% | 5.4% | 0.06pp |
| 8 | Japan | $7.3B | 3.5% | 24% | 3.3% | -0.22pp |
| 9 | China | $6.9B | 3.3% | 54% | 1.9% | -1.42pp |
| 10 | United Kingdom | $6.9B | 3.3% | 10% | 4.1% | 0.77pp |
| 11 | Netherlands | $6.7B | 3.2% | 20% | 3.2% | 0.04pp |
| 12 | Slovenia | $5.2B | 2.4% | 20% | 2.5% | 0.03pp |
| 13 | Canada | $5.1B | 2.4% | 10% | 3.0% | 0.57pp |
| 14 | Hungary | $4.3B | 2.0% | 20% | 2.1% | 0.02pp |
| 15 | France | $3.9B | 1.9% | 20% | 1.9% | 0.02pp |

## HTS Composition, 2024

| HTS | Imports | Share |
|---|---:|---:|
| 3002 | $108.0B | 51.2% |
| 3004 | $100.4B | 47.6% |
| 3003 | $2.4B | 1.1% |

## Year Totals In Pharma1

| Year | Imports |
|---:|---:|
| 2018 | $113.6B |
| 2019 | $126.2B |
| 2020 | $140.2B |
| 2021 | $149.0B |
| 2022 | $164.4B |
| 2023 | $177.9B |
| 2024 | $210.8B |
| 2025 | $208.9B |

## Interpretation

The Pharma1 data show a broad, high-value import base with no large zero-tariff escape valve in the matched tariff schedule. The model allows some sourcing reallocation toward lower-tariff suppliers, but the HHI barely moves and the import-volume response is strongly negative. In the short run, that means tariff pressure mainly transmits through imported input and finished-drug costs rather than being neutralized by rapid supplier switching.
