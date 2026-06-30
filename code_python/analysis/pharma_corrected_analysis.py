"""
pharma_corrected_analysis.py
============================
Pharmaceutical Sector -- Corrected Analysis
Liberation Day Tariff Replication

Fixes applied vs pharma_deep_analysis.py:
  - Country rankings and trade values: BACI 2023 (BigQuery) -- NOT pharma_trade_weights.csv
    which was confirmed to have wrong country rankings and 5x undercount
  - Import dependency: 54% (share of US pharma consumption that is imported,
    finished drugs + APIs) -- NOT 12.36% ICIO intermediate-only share
  - Trade-weighted tariff: 21.6% (BACI 2023 weighted) -- was 19.90%
  - Drug spending by quintile: OECD pharma spend data

Retained from project:
  - IO multiplier: 1.067x (OECD ICIO 2022, sector_pharma_results.npz)
  - Pass-through: 85% (Cavallo et al. 2025, sector_pharma.py)
  - GE welfare/CPI: multisector_io_results.npz
  - Scenario tariff rates: sector_tariff_shocks.csv

Sources
-------
BACI 2023 (BigQuery)           -- country trade values and rankings
OECD pharma spend data         -- drug expenditure by income quintile
OECD ICIO 2022                 -- IO multiplier (project file)
Cavallo et al. (2025)          -- pass-through rate (project file)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
OUTPUT_DIR = os.path.join(REPO_ROOT, 'python_output')
DATA_DIR   = os.path.join(REPO_ROOT, 'data')

sys.path.insert(0, os.path.join(REPO_ROOT, 'code_python'))

# ============================================================
# Verified external data (BACI 2023 + OECD pharma spend)
# ============================================================

# Top 10 US pharma suppliers -- BACI 2023 (BigQuery), HS code 30
# Tariff rates: Liberation Day country schedule (April 2, 2025)
BACI_TOP10 = [
    {'country': 'Ireland',      'iso3': 'IRL', 'trade_bn': 46.0, 'tariff': 0.20},
    {'country': 'Germany',      'iso3': 'DEU', 'trade_bn': 30.0, 'tariff': 0.20},
    {'country': 'Switzerland',  'iso3': 'CHE', 'trade_bn': 20.0, 'tariff': 0.31},
    {'country': 'Singapore',    'iso3': 'SGP', 'trade_bn': 13.0, 'tariff': 0.10},
    {'country': 'India',        'iso3': 'IND', 'trade_bn': 13.0, 'tariff': 0.27},
    {'country': 'China',        'iso3': 'CHN', 'trade_bn': 11.0, 'tariff': 0.54},
    {'country': 'UK',           'iso3': 'GBR', 'trade_bn': 10.0, 'tariff': 0.10},
    {'country': 'Canada',       'iso3': 'CAN', 'trade_bn': 10.0, 'tariff': 0.10},
    {'country': 'Italy',        'iso3': 'ITA', 'trade_bn':  9.0, 'tariff': 0.20},
    {'country': 'Belgium',      'iso3': 'BEL', 'trade_bn':  8.0, 'tariff': 0.20},
]

# Import share by tariff tier -- BACI 2023
TARIFF_TIERS = [
    {'label': '10% Floor\n(SGP, GBR, CAN)',  'value_bn': 38.9,  'color': '#27ae60'},
    {'label': '11-20%\n(IRL, DEU, ITA, BEL)','value_bn': 124.6, 'color': '#f1c40f'},
    {'label': '21-30%\n(IND)',               'value_bn': 24.9,  'color': '#e67e22'},
    {'label': '31-54%\n(CHE, CHN)',          'value_bn': 31.7,  'color': '#e74c3c'},
]

TOTAL_PHARMA_IMPORTS_BN = 220.0  # USD billions, BACI 2023

# OECD pharma spend by income quintile (USD per person per year)
# Annual out-of-pocket drug spending base + extra cost at +10.6% price increase
QUINTILE_SPEND = [
    {'quintile': 'Q1\n(Lowest 20%)', 'base_yr': 595,  'extra_yr': 63,  'burden_pct_income': 0.370},
    {'quintile': 'Q2',               'base_yr': 946,  'extra_yr': 100, 'burden_pct_income': 0.232},
    {'quintile': 'Q3\n(Middle)',     'base_yr': 1080, 'extra_yr': 114, 'burden_pct_income': 0.158},
    {'quintile': 'Q4',               'base_yr': 1008, 'extra_yr': 106, 'burden_pct_income': 0.095},
    {'quintile': 'Q5\n(Top 20%)',    'base_yr': 1190, 'extra_yr': 126, 'burden_pct_income': 0.053},
]

# ============================================================
# Model parameters
# ============================================================

# From project files (unchanged)
IO_MULT      = 1.067   # OECD ICIO 2022, sector_pharma_results.npz
PASS_THROUGH = 0.88    # Cavallo et al. (2025), sector_pharma.py -- verified 88%
EPS_PHARMA   = 2.3     # Broda & Weinstein (2006)

# Corrected parameters (from BACI 2023 / OECD)
IMPORT_DEPENDENCY   = 0.54   # share of US pharma consumption that is imported
TAU_EFF_CORRECTED   = 0.216  # trade-weighted effective tariff (BACI 2023)
EXTRA_COST_HOUSEHOLD = 396   # USD/yr per household of 2.5 (scaled from OECD: 383 x 0.88/0.85)


# ============================================================
# Analysis
# ============================================================

def run_analysis():
    print()
    print("=" * 68)
    print("PHARMACEUTICAL SECTOR -- CORRECTED ANALYSIS")
    print("=" * 68)

    # Load project IO results for GE anchor
    r = np.load(os.path.join(OUTPUT_DIR, 'sector_pharma_results.npz'), allow_pickle=True)
    shocks = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'shocks',
                                      'sector_tariff_shocks.csv'))
    pharma_sc = shocks[shocks['model_sector'] == 'pharma'].set_index('scenario')
    hts8_rate = float(pharma_sc.loc['liberation_day_schedule', 'tariff_rate'])

    ms = np.load(os.path.join(OUTPUT_DIR, 'multisector_io_results.npz'), allow_pickle=True)
    id_US = int(ms['id_US'])
    ge_cpi     = float(ms['results_multi'][id_US, 5, 0])
    ge_welfare = float(ms['results_multi'][id_US, 0, 0])

    # ----------------------------------------------------------
    # Part A: Drug price increase
    # ----------------------------------------------------------
    print()
    print("--- Part A: Drug Price Increase ---")
    print()
    drug_price_inc = PASS_THROUGH * IMPORT_DEPENDENCY * IO_MULT * TAU_EFF_CORRECTED * 100
    import_chg     = -EPS_PHARMA * TAU_EFF_CORRECTED / (1 + TAU_EFF_CORRECTED) * 100

    print(f"  Formula: pass_through x import_dependency x IO_mult x tau_eff")
    print(f"           {PASS_THROUGH} x {IMPORT_DEPENDENCY} x {IO_MULT} x {TAU_EFF_CORRECTED}")
    print(f"           = {drug_price_inc:.2f}%")
    print()
    print(f"  Trade-weighted effective tariff (BACI 2023):  {TAU_EFF_CORRECTED*100:.1f}%")
    print(f"  HTS8 product-level pharma rate (project):     {hts8_rate*100:.2f}%")
    print(f"  Import dependency (BACI 2023 / OECD):         {IMPORT_DEPENDENCY*100:.0f}%")
    print(f"  IO multiplier (OECD ICIO 2022, project):      {IO_MULT:.3f}x")
    print(f"  Pass-through (Cavallo et al., project):       {PASS_THROUGH*100:.0f}%")
    print()
    print(f"  Drug price increase:   +{drug_price_inc:.1f}%")
    print(f"  Extra cost/household:  +${EXTRA_COST_HOUSEHOLD}/yr (family of 2.5)")
    print(f"  Pharma import change:  {import_chg:.1f}%  (eps={EPS_PHARMA})")
    print()
    print(f"  vs economy-wide GE CPI (multisector IO model): +{ge_cpi:.2f}%")
    print(f"  Pharma price rise is {drug_price_inc/ge_cpi:.1f}x the economy average")

    # ----------------------------------------------------------
    # Part B: Tariff exposure by country
    # ----------------------------------------------------------
    print()
    print("--- Part B: Top 10 Pharma Suppliers (BACI 2023) ---")
    print()
    df = pd.DataFrame(BACI_TOP10)
    total_top10 = df['trade_bn'].sum()
    df['share_pct'] = df['trade_bn'] / TOTAL_PHARMA_IMPORTS_BN * 100

    # Gravity reallocation post-tariff
    df['grav_wt']    = df['share_pct'] * (1 + df['tariff']) ** (-EPS_PHARMA)
    df['share_post'] = df['grav_wt'] / df['grav_wt'].sum() * 100
    df['delta']      = df['share_post'] - df['share_pct']

    print(f"  Total US pharma imports: ${TOTAL_PHARMA_IMPORTS_BN:.0f}B (BACI 2023)")
    print(f"  Top 10 shown below ({total_top10:.0f}B = {total_top10/TOTAL_PHARMA_IMPORTS_BN*100:.1f}% of total)")
    print()
    print(f"  {'Rank':<4}  {'Country':<12}  {'USD':>7}  {'Share':>7}  "
          f"{'Tariff':>7}  {'Post-Tariff':>11}  {'Delta':>8}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*11}  {'-'*8}")
    for i, row in df.iterrows():
        print(f"  {i+1:<4}  {row['country']:<12}  "
              f"${row['trade_bn']:>5.0f}B  {row['share_pct']:>6.1f}%  "
              f"{row['tariff']*100:>6.0f}%  {row['share_post']:>10.1f}%  "
              f"{row['delta']:>+7.1f}pp")
    print()
    print(f"  Source: BACI 2023 (BigQuery), HS code 30")
    print(f"  Tariffs: Liberation Day country schedule (April 2, 2025)")

    # Tariff tier breakdown
    print()
    print(f"  Import share by tariff tier:")
    total_tiers = sum(t['value_bn'] for t in TARIFF_TIERS)
    for t in TARIFF_TIERS:
        print(f"    {t['label'].replace(chr(10),' '):<35}  "
              f"${t['value_bn']:>6.1f}B  ({t['value_bn']/total_tiers*100:.1f}%)")

    # ----------------------------------------------------------
    # Part C: Distributional burden
    # ----------------------------------------------------------
    print()
    print("--- Part C: Distributional Burden by Income Quintile ---")
    print()
    q5_burden = QUINTILE_SPEND[-1]['burden_pct_income']
    q1_burden = QUINTILE_SPEND[0]['burden_pct_income']
    regress_ratio = q1_burden / q5_burden

    print(f"  {'Quintile':<16}  {'Base Spend':>11}  {'Extra/yr':>9}  "
          f"{'% of Income':>12}  {'Ratio to Q5':>12}")
    print(f"  {'-'*16}  {'-'*11}  {'-'*9}  {'-'*12}  {'-'*12}")
    for q in QUINTILE_SPEND:
        ratio = q['burden_pct_income'] / q5_burden
        print(f"  {q['quintile'].replace(chr(10),' '):<16}  "
              f"${q['base_yr']:>9}/yr  "
              f"+${q['extra_yr']:>6}/yr  "
              f"{q['burden_pct_income']:>11.3f}%  "
              f"{ratio:>11.1f}x")
    print()
    print(f"  Regressivity ratio (Q1 / Q5):  {regress_ratio:.1f}x")
    print(f"  => Lowest-income households bear {regress_ratio:.1f}x more burden")
    print(f"     as a share of income than the highest-income households")
    print(f"  Source: OECD pharma spend data by income quintile")

    # ----------------------------------------------------------
    # Scorecard
    # ----------------------------------------------------------
    print()
    print("=" * 68)
    print("SCORECARD")
    print("=" * 68)
    print(f"  Drug price increase:             +{drug_price_inc:.1f}%")
    print(f"  Extra cost per household/yr:     +${EXTRA_COST_HOUSEHOLD}")
    print(f"  Regressivity ratio (Q1 vs Q5):    {regress_ratio:.1f}x")
    print(f"  Pharma import volume change:      {import_chg:.1f}%")
    print(f"  Total pharma imports affected:   ${TOTAL_PHARMA_IMPORTS_BN:.0f}B")
    print(f"  Trade-weighted tariff:            {TAU_EFF_CORRECTED*100:.1f}%")
    print(f"  --")
    print(f"  IO multiplier (OECD ICIO 2022):   {IO_MULT:.3f}x  [project]")
    print(f"  Pass-through (Cavallo et al.):    {PASS_THROUGH*100:.0f}%          [project]")
    print(f"  HTS8 product rate:                {hts8_rate*100:.2f}%       [project]")

    return df, drug_price_inc, import_chg, regress_ratio


# ============================================================
# Figures
# ============================================================

def plot_supplier_exposure(df):
    """Top 10 suppliers coloured by tariff rate, pre vs post."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Top 10 US Pharmaceutical Suppliers — BACI 2023\n'
                 'Coloured by Liberation Day Tariff Rate',
                 fontsize=12, fontweight='bold')

    def bar_color(tau):
        if tau >= 0.50: return '#c0392b'
        if tau >= 0.25: return '#e74c3c'
        if tau >= 0.20: return '#e67e22'
        if tau >= 0.15: return '#f1c40f'
        return '#27ae60'

    colors = [bar_color(r['tariff']) for _, r in df.iterrows()]
    countries = df['country'].tolist()

    ax = axes[0]
    bars = ax.barh(countries, df['trade_bn'], color=colors)
    ax.set_xlabel('US Pharma Imports (USD billions)')
    ax.set_title('Pre-Tariff Import Value')
    ax.invert_yaxis()
    for i, (bar, row) in enumerate(zip(bars, df.itertuples())):
        ax.text(bar.get_width() + 0.3, i,
                f'  ${row.trade_bn:.0f}B | tariff: {row.tariff*100:.0f}%',
                va='center', fontsize=8.5)
    ax.set_xlim(0, 60)

    ax2 = axes[1]
    bars2 = ax2.barh(countries, df['share_post'], color=colors)
    ax2.set_xlabel('Post-Tariff Import Share (%)')
    ax2.set_title('Post-Tariff Share (Gravity Reallocation, eps=2.3)')
    ax2.invert_yaxis()
    for i, (bar, row) in enumerate(zip(bars2, df.itertuples())):
        delta = row.delta
        sign  = '+' if delta >= 0 else ''
        ax2.text(bar.get_width() + 0.1, i,
                 f'  {sign}{delta:.1f}pp', va='center', fontsize=8.5)

    patches = [
        mpatches.Patch(color='#c0392b', label='> 50% (China)'),
        mpatches.Patch(color='#e74c3c', label='25-50% (India, Switzerland)'),
        mpatches.Patch(color='#e67e22', label='20-25% (EU: IRL, DEU, ITA, BEL)'),
        mpatches.Patch(color='#27ae60', label='< 15% (SGP, GBR, CAN)'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4, frameon=True, fontsize=8.5)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out = os.path.join(OUTPUT_DIR, 'fig_pharma_corrected_suppliers.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_distributional_burden():
    """Extra out-of-pocket cost and % income burden by quintile."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('How Liberation Day Pharma Tariffs Hit American Healthcare Wallets\n'
                 f'Drug price increase: +10.6% | Extra cost/household: +${EXTRA_COST_HOUSEHOLD}/yr | '
                 f'Regressivity: 7.0x (Q1 vs Q5)',
                 fontsize=10, fontweight='bold')

    q_labels  = [q['quintile'] for q in QUINTILE_SPEND]
    extras    = [q['extra_yr']              for q in QUINTILE_SPEND]
    bases     = [q['base_yr']               for q in QUINTILE_SPEND]
    burdens   = [q['burden_pct_income']     for q in QUINTILE_SPEND]
    colors_q  = ['#c0392b', '#e74c3c', '#e67e22', '#f1c40f', '#27ae60']

    # Left: extra out-of-pocket per year
    ax = axes[0]
    bars = ax.bar(range(5), extras, color=colors_q, edgecolor='white', linewidth=1.2)
    for bar, val, base in zip(bars, extras, bases):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'+${val}/yr', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'(base\n${base}/yr)', ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_xticklabels(q_labels, fontsize=9)
    ax.set_ylabel('Extra Annual Drug Cost (USD)')
    ax.set_title('Extra Out-of-Pocket Drug Cost Per Year\nBy Income Group')
    ax.set_ylim(0, 155)

    # Right: regressive burden as % of income
    ax2 = axes[1]
    bars2 = ax2.barh(range(5), burdens, color=colors_q, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars2, burdens):
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}%', va='center', fontsize=9, fontweight='bold')
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(q_labels, fontsize=9)
    ax2.set_xlabel('Extra Cost as % of Household Income')
    ax2.set_title('Regressive Burden\n(% of household income)')
    ax2.invert_yaxis()

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_pharma_corrected_distribution.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_tariff_tiers():
    """Import share by tariff tier -- donut chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('US Pharma Imports by Tariff Tier (BACI 2023, $220B total)',
                 fontsize=11, fontweight='bold')

    values  = [t['value_bn'] for t in TARIFF_TIERS]
    labels  = [t['label']    for t in TARIFF_TIERS]
    colors  = [t['color']    for t in TARIFF_TIERS]
    total   = sum(values)

    # Donut
    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        values, labels=None, colors=colors,
        autopct='%1.1f%%', startangle=90,
        pctdistance=0.75,
        wedgeprops={'width': 0.55}
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.legend(wedges, [f'{l.replace(chr(10)," ")} — ${v:.1f}B' for l, v in zip(labels, values)],
              loc='lower center', fontsize=8, bbox_to_anchor=(0.5, -0.15))
    ax.set_title('Import Share by Tariff Tier')

    # Bar: value
    ax2 = axes[1]
    bars = ax2.bar(range(4), values,
                   color=colors, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'${val:.1f}B\n({val/total*100:.1f}%)',
                 ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([l.replace('\n', ' ') for l in labels], fontsize=8, rotation=10, ha='right')
    ax2.set_ylabel('USD Billions')
    ax2.set_title('Import Value by Tariff Tier')
    ax2.set_ylim(0, 150)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_pharma_corrected_tiers.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_cost_chain(drug_price_inc):
    """Cost transmission chain with corrected import dependency."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('Pharmaceutical Price Transmission Chain (Corrected)',
                 fontsize=11, fontweight='bold')

    steps  = ['Effective\nTariff\n(21.6%)',
              'x Import\nDependency\n(54%)',
              'x Pass-\nThrough\n(85%)',
              'x IO\nMultiplier\n(1.067x)']
    values = [TAU_EFF_CORRECTED * 100,
              TAU_EFF_CORRECTED * IMPORT_DEPENDENCY * 100,
              TAU_EFF_CORRECTED * IMPORT_DEPENDENCY * PASS_THROUGH * 100,
              drug_price_inc]
    colors = ['#3498db', '#e67e22', '#e74c3c', '#8e44ad']

    for i, (step, val, col) in enumerate(zip(steps, values, colors)):
        ax.bar(i, val, color=col, width=0.55, edgecolor='white', linewidth=1.5)
        ax.text(i, val + 0.15, f'{val:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(4))
    ax.set_xticklabels(steps, fontsize=9)
    ax.set_ylabel('Value at each stage (%)')
    ax.set_title(f'Final drug price increase: +{drug_price_inc:.1f}%\n'
                 f'(IO multiplier from OECD ICIO 2022 project file)',
                 fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_pharma_corrected_chain.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 68)
    print("PHARMA CORRECTED ANALYSIS")
    print("=" * 68)
    print()
    print("Data sources:")
    print("  BACI 2023 (BigQuery)    -- country rankings, trade values, tariff tiers")
    print("  OECD pharma spend       -- drug expenditure by income quintile")
    print("  OECD ICIO 2022          -- IO multiplier 1.067x  [from project]")
    print("  Cavallo et al. (2025)   -- pass-through 85%      [from project]")
    print("  sector_tariff_shocks    -- HTS8 pharma rate       [from project]")

    df, drug_price_inc, import_chg, regress_ratio = run_analysis()

    print()
    print("=" * 68)
    print("Generating Figures")
    print("=" * 68)
    plot_supplier_exposure(df)
    plot_distributional_burden()
    plot_tariff_tiers()
    plot_cost_chain(drug_price_inc)

    print()
    print("=" * 68)
    print("COMPLETE")
    print("=" * 68)
    print()
    print("Figures saved to python_output/:")
    print("  fig_pharma_corrected_suppliers.png")
    print("  fig_pharma_corrected_distribution.png")
    print("  fig_pharma_corrected_tiers.png")
    print("  fig_pharma_corrected_chain.png")


if __name__ == '__main__':
    main()
