# Pharma1 Supply Chain Risk Analysis

Objective: evaluate how tariffs affect pharmaceutical supply-chain risk by analyzing dependence on key importing countries, changes in sourcing patterns, and supplier concentration.

## Supply Dependence

- 2024 Pharma1 import base: $210.8B across 97 supplier countries.
- Top 5 suppliers account for 56.5% of imports.
- Top 10 suppliers account for 77.7% of imports.
- Ireland alone supplies 25.9% of Pharma1 imports, creating a single-country dependence point even though the broader supplier base is diversified.
- 84.2% of imports come from countries facing tariffs of at least 20%.
- 12.4% of imports come from countries facing tariffs of at least 30%.

## Sourcing Pattern Changes

Post-tariff sourcing is modeled as share_post[j] proportional to share_pre[j] x (1 + tariff[j])^(-2.3).

| Direction | Country | Tariff | Pre Share | Post Share | Shift |
|---|---|---:|---:|---:|---:|
| Gain | Singapore | 10% | 7.9% | 9.7% | +1.85pp |
| Gain | United Kingdom | 10% | 3.3% | 4.1% | +0.77pp |
| Gain | Canada | 10% | 2.4% | 3.0% | +0.57pp |
| Gain | Ireland | 20% | 25.9% | 26.2% | +0.30pp |
| Gain | Australia | 10% | 0.6% | 0.8% | +0.15pp |
| Loss | Switzerland | 31% | 9.1% | 7.5% | -1.58pp |
| Loss | China | 54% | 3.3% | 1.9% | -1.42pp |
| Loss | India | 27% | 5.8% | 5.2% | -0.66pp |
| Loss | Japan | 24% | 3.5% | 3.3% | -0.22pp |
| Loss | South Korea | 26% | 1.4% | 1.3% | -0.14pp |

## Supplier Concentration

- HHI before tariffs: 1047.
- HHI after tariff-induced reallocation: 1065.
- HHI change: 18 points.

The HHI remains in the unconcentrated range, so the main risk is not monopoly-style concentration. The risk is tariff-correlated dependence: several large suppliers face 20% or higher tariffs, while low-tariff alternatives account for only a limited share of the baseline supply base.

## Risk Takeaway

Tariffs raise pharmaceutical supply-chain risk by making the existing import network more expensive and pushing sourcing toward a small set of lower-tariff alternatives. Because those alternatives start from a modest import share, substitution is partial rather than complete. The result is a supply chain that remains diversified on paper but becomes more fragile in practice: high-tariff key suppliers lose share, lower-tariff suppliers gain share, and the overall concentration metric moves only slightly.
