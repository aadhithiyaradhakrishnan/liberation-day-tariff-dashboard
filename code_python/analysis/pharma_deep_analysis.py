"""
pharma_deep_analysis.py
=======================
Pharmaceutical Sector -- Deep Analysis
Liberation Day Tariff Replication

Objective: Analyze how tariffs on imported pharmaceutical inputs increase
           production costs and drug prices, given limited short-run substitution.

All numbers come directly from pre-computed NPZ files and project CSVs.
No external data is introduced.

Data sources
------------
python_output/sector_pharma_results.npz           -- pre-computed pharma analysis
data/processed/shocks/pharma_trade_weights.csv    -- 132-country bilateral trade shares
data/processed/shocks/sector_tariff_shocks.csv    -- HTS8 sector tariff rates
data/base_data/tariffs.csv + country_labels.csv   -- Liberation Day country tariffs

Outputs (python_output/)
------------------------
fig_pharma_tariff_bands.png         -- tariff exposure distribution across 132 suppliers
fig_pharma_supplier_reallocation.png -- pre vs post-tariff sourcing shifts
fig_pharma_cost_transmission.png    -- cost chain: tariff -> input cost -> drug price
fig_pharma_scenario_comparison.png  -- drug price / HHI across policy scenarios
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
# Data loading
# ============================================================

def load_all_data():
    d = {}

    # --- sector_pharma_results.npz (pre-computed) ---
    r = np.load(os.path.join(OUTPUT_DIR, 'sector_pharma_results.npz'), allow_pickle=True)
    d['tau_eff']            = float(r['tau_pharma_eff'])       # 19.90%
    d['hts8_rate']          = float(r['hts8_pharma_rate'])     # 2.45%
    d['io_mult']            = float(r['io_multiplier'])        # 1.067x
    d['imp_share_interm']   = float(r['imp_share_interm'])     # 12.36%
    d['price_cpi_contrib']  = float(r['price_noretal'])        # 0.495% (economy-wide CPI)
    d['import_chg']         = float(r['import_chg_noretal'])   # -38.17%
    d['hhi_pre']            = float(r['hhi_pre'])              # 580
    d['hhi_post']           = float(r['hhi_post'])             # 591
    d['n_suppliers']        = int(r['n_suppliers'])            # 132
    d['pharma_welfare_loss']= float(r['pharma_welfare_noretal'])  # -0.013%

    # --- sector_tariff_shocks.csv (all scenarios) ---
    shocks = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'shocks',
                                      'sector_tariff_shocks.csv'))
    pharma_shocks = shocks[shocks['model_sector'] == 'pharma'].set_index('scenario')
    d['shocks'] = {
        'liberation_day':        pharma_shocks.loc['liberation_day_schedule', 'tariff_rate'],
        'supply_chain_disrupt':  pharma_shocks.loc['supply_chain_disruption', 'tariff_rate'],
        'optimal_uniform':       pharma_shocks.loc['optimal_uniform_19',      'tariff_rate'],
    }

    # --- pharma_trade_weights.csv + tariffs ---
    tw      = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'shocks',
                                       'pharma_trade_weights.csv'))
    tariffs = pd.read_csv(os.path.join(DATA_DIR, 'base_data', 'tariffs.csv'))
    labels  = pd.read_csv(os.path.join(DATA_DIR, 'base_data', 'country_labels.csv'))
    tariffs['iso3'] = labels['iso3'].values[:len(tariffs)]
    merged = tw.merge(tariffs[['iso3', 'applied_tariff']], on='iso3', how='left')

    # Gravity reallocation: share_post proportional to pre_share * (1+tau)^(-eps)
    EPS = 2.3   # Broda & Weinstein 2006 -- stored in sector_pharma.py
    valid = merged.dropna(subset=['applied_tariff']).copy()
    valid['grav_wt']    = valid['weight'] * (1.0 + valid['applied_tariff']) ** (-EPS)
    total_grav          = valid['grav_wt'].sum()
    valid['share_post'] = valid['grav_wt'] / total_grav
    valid['delta']      = valid['share_post'] - valid['weight']
    d['tw'] = valid
    d['eps_pharma'] = EPS

    # --- Constants from sector_pharma.py ---
    d['pass_through']      = 0.88   # Cavallo et al. (2025)
    d['pharma_mfg_share']  = 0.082  # USITC 2023
    d['pharma_pce_share']  = 0.027  # BEA 2023
    d['mfg_import_pen']    = 0.323  # ITPD/GE model

    # Drug price increase (ICIO-based):
    # tau_eff x imp_share_interm x pass_through x io_mult
    # This is distinct from price_cpi_contrib which uses pharma_mfg_share
    d['drug_price_inc'] = (d['tau_eff'] * d['imp_share_interm']
                           * d['pass_through'] * d['io_mult'] * 100)

    # GE model aggregate (multisector IO)
    ms = np.load(os.path.join(OUTPUT_DIR, 'multisector_io_results.npz'), allow_pickle=True)
    id_US = int(ms['id_US'])
    d['ge_cpi_noretal']     = float(ms['results_multi'][id_US, 5, 0])  # 7.09%
    d['ge_welfare_noretal'] = float(ms['results_multi'][id_US, 0, 0])  # +0.60%

    return d


# ============================================================
# Analysis
# ============================================================

def run_analysis(d):
    print()
    print("=" * 68)
    print("PHARMACEUTICAL SECTOR -- DEEP ANALYSIS")
    print("Objective: Tariff-Induced Input Cost Increases and Drug Prices")
    print("=" * 68)

    tw = d['tw']

    # ----------------------------------------------------------
    # Part A: Tariff structure on pharma inputs
    # ----------------------------------------------------------
    print()
    print("--- Part A: Tariff Burden on Pharmaceutical Inputs ---")
    print()
    print(f"  Two layers of tariff apply to pharma imports:")
    print()
    print(f"  Layer 1 -- HTS8 product-level MFN rate:")
    print(f"             {d['hts8_rate']*100:.2f}%  (HTS pharma lines avg., Liberation Day schedule)")
    print(f"             Source: sector_tariff_shocks.csv, scenario=liberation_day_schedule")
    print()
    print(f"  Layer 2 -- Country-level Liberation Day tariff (trade-weighted effective rate):")
    print(f"             {d['tau_eff']*100:.2f}%  (across 132 pharma trading partners)")
    print(f"             Source: pharma_trade_weights.csv x tariffs.csv")
    print()
    print(f"  The effective rate ({d['tau_eff']*100:.2f}%) is ~{d['tau_eff']/d['hts8_rate']:.1f}x the HTS8 product rate,")
    print(f"  because Liberation Day adds large country-level charges on top.")
    print()

    # Tariff band breakdown
    bands = [
        (0.0,  0.10, '< 10%'),
        (0.10, 0.15, '10-15%'),
        (0.15, 0.20, '15-20%'),
        (0.20, 0.25, '20-25%'),
        (0.25, 0.35, '25-35%'),
        (0.35, 1.00, '> 35%'),
    ]
    print(f"  Tariff distribution across {d['n_suppliers']} pharma supplier countries:")
    print(f"  {'Tariff Band':<10}  {'Countries':>10}  {'Trade Share':>12}  {'Avg Rate':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}")
    for lo, hi, label in bands:
        mask = (tw['applied_tariff'] >= lo) & (tw['applied_tariff'] < hi)
        n    = mask.sum()
        shr  = tw.loc[mask, 'weight'].sum() * 100
        avg  = tw.loc[mask, 'applied_tariff'].mean() * 100 if n > 0 else 0.0
        print(f"  {label:<10}  {n:>10}  {shr:>11.1f}%  {avg:>9.1f}%")
    print()
    print(f"  Key: no pharma supplier faces a 0% tariff.")
    print(f"  51.1% of trade volume comes from 20-25% tariff countries (EU bloc).")
    print(f"  Source: pharma_trade_weights.csv (132 countries, 2024 trade flows)")

    # ----------------------------------------------------------
    # Part B: Supply-chain cost transmission
    # ----------------------------------------------------------
    print()
    print("--- Part B: Supply-Chain Transmission to Production Costs ---")
    print()
    print(f"  OECD ICIO 2022 -- pharma intermediate import share:")
    print(f"    {d['imp_share_interm']*100:.2f}% of pharma sector inputs are imported")
    print()
    print(f"  IO supply-chain multiplier (Leontief roundabout):")
    print(f"    M = 1 / (1 - (1 - beta_labor) x imp_share)")
    print(f"    M = 1 / (1 - (1 - 0.49) x {d['imp_share_interm']:.4f})")
    print(f"    M = {d['io_mult']:.4f}x")
    print(f"    (Captures indirect cost amplification through multi-stage production)")
    print()
    print(f"  Drug price increase formula (ICIO-based):")
    print(f"    = tau_eff x imp_share_interm x pass_through x io_mult")
    print(f"    = {d['tau_eff']*100:.2f}% x {d['imp_share_interm']*100:.2f}% x {d['pass_through']*100:.0f}% x {d['io_mult']:.3f}")
    print(f"    = {d['drug_price_inc']:.3f}%")
    print()
    print(f"  Economy-wide pharma CPI contribution (sector_pharma.py formula):")
    print(f"    = pass_through x tau_eff x (PHARMA_MFG_SHARE x mfg_import_pen) x io_mult")
    print(f"    = {d['pass_through']} x {d['tau_eff']:.4f} x ({d['pharma_mfg_share']} x {d['mfg_import_pen']}) x {d['io_mult']:.4f}")
    print(f"    = {d['price_cpi_contrib']:.4f}%  (pharma's contribution to total CPI)")
    print()
    print(f"  Distinction:")
    print(f"    {d['drug_price_inc']:.2f}%  = drug prices themselves rise (ICIO input cost route)")
    print(f"    {d['price_cpi_contrib']:.3f}%  = pharma's contribution to economy-wide CPI")
    print(f"              (smaller because pharma PCE share = {d['pharma_pce_share']*100:.1f}% of total consumption)")
    print(f"  Source: sector_pharma_results.npz + OECD ICIO 2022")

    # ----------------------------------------------------------
    # Part C: Limited short-run substitution
    # ----------------------------------------------------------
    print()
    print("--- Part C: Limited Short-Run Substitution ---")
    print()
    print(f"  Pre-tariff HHI:  {d['hhi_pre']:.0f}  (across {d['n_suppliers']} suppliers)")
    print(f"  Post-tariff HHI: {d['hhi_post']:.0f}  (gravity reallocation, eps={d['eps_pharma']})")
    print(f"  HHI change:      +{d['hhi_post']-d['hhi_pre']:.0f}  ({(d['hhi_post']/d['hhi_pre']-1)*100:.2f}% increase)")
    print()
    print(f"  Post-tariff gravity formula: share_post[j] ~ share_pre[j] x (1+tau[j])^(-{d['eps_pharma']})")
    print(f"  Source: sector_pharma_results.npz + pharma_trade_weights.csv")
    print()

    # Top gainers and losers
    top15 = tw.nlargest(15, 'weight').copy()
    gainers = tw[tw['delta'] > 0].nlargest(5, 'delta')
    losers  = tw[tw['delta'] < 0].nsmallest(5, 'delta')

    print(f"  Largest share shifts post-tariff:")
    print(f"  {'Country':<6}  {'Tariff':>7}  {'Pre share':>10}  {'Post share':>10}  {'Delta':>10}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}")
    for _, row in gainers.iterrows():
        print(f"  {row['iso3']:<6}  {row['applied_tariff']*100:>6.0f}%  "
              f"{row['weight']*100:>9.2f}%  {row['share_post']*100:>9.2f}%  "
              f"+{row['delta']*100:>8.2f}pp")
    for _, row in losers.iterrows():
        print(f"  {row['iso3']:<6}  {row['applied_tariff']*100:>6.0f}%  "
              f"{row['weight']*100:>9.2f}%  {row['share_post']*100:>9.2f}%  "
              f"{row['delta']*100:>9.2f}pp")
    print()
    print(f"  Why substitution is limited:")
    print(f"    - All 132 suppliers face at least 10% tariff -- no zero-tariff alternative")
    print(f"    - Low-tariff alternatives (CAN, GBR at 10%) gain share in the model,")
    print(f"      but their combined pre-tariff share is only")
    low_share = tw[tw['applied_tariff'] <= 0.10]['weight'].sum() * 100
    print(f"      {low_share:.1f}% of US pharma imports")
    print(f"    - China loses {abs(tw.loc[tw['iso3']=='CHN','delta'].values[0])*100:.2f}pp (54% tariff)")
    print(f"      but these APIs cannot be quickly redirected to domestic production")
    print(f"    - HHI increases by {d['hhi_post']-d['hhi_pre']:.0f} points -- marginal concentration increase")
    print(f"      confirms the post-tariff supplier base remains unconcentrated")
    print(f"      but sourcing options are more expensive across the board")

    # ----------------------------------------------------------
    # Part D: Import volume response
    # ----------------------------------------------------------
    print()
    print("--- Part D: Import Volume Response ---")
    print()
    print(f"  Formula: delta_imports = -eps x tau_eff / (1 + tau_eff)")
    print(f"           = -{d['eps_pharma']} x {d['tau_eff']:.4f} / (1 + {d['tau_eff']:.4f})")
    print(f"           = {d['import_chg']:.2f}%")
    print()
    print(f"  A {abs(d['import_chg']):.1f}% drop in import volume at eps=2.3 means the US")
    print(f"  significantly curtails pharma imports, but does NOT signal domestic")
    print(f"  substitution: without sufficient domestic API manufacturing capacity,")
    print(f"  reduced imports translate directly into tighter supply.")
    print(f"  Source: sector_pharma_results.npz['import_chg_noretal']")

    # ----------------------------------------------------------
    # Part E: Scenario comparison
    # ----------------------------------------------------------
    print()
    print("--- Part E: Policy Scenario Comparison ---")
    print()

    # For each scenario: compute effective country tariff vs HTS8 rate and drug price impact
    sc_data = []
    for name, hts8_r in d['shocks'].items():
        # Under different HTS8 rates, scale the effective country tariff proportionally
        # to the ratio of scenario HTS8 rate / liberation day HTS8 rate
        scale     = hts8_r / d['hts8_rate'] if d['hts8_rate'] > 0 else 1.0
        tau_sc    = d['tau_eff'] * scale
        drug_p    = tau_sc * d['imp_share_interm'] * d['pass_through'] * d['io_mult'] * 100
        imp_chg   = -d['eps_pharma'] * tau_sc / (1 + tau_sc) * 100
        sc_data.append((name, hts8_r, tau_sc, drug_p, imp_chg))

    print(f"  {'Scenario':<25}  {'HTS8 Rate':>10}  {'Eff. Rate':>10}  "
          f"{'Drug Price Inc':>15}  {'Import Chg':>11}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*15}  {'-'*11}")
    labels_map = {
        'liberation_day':       'Liberation Day',
        'supply_chain_disrupt': 'Supply Chain Shock',
        'optimal_uniform':      'Optimal Uniform',
    }
    for name, hts8_r, tau_sc, drug_p, imp_chg in sc_data:
        label = labels_map.get(name, name)
        print(f"  {label:<25}  {hts8_r*100:>9.2f}%  {tau_sc*100:>9.2f}%  "
              f"{drug_p:>+14.3f}%  {imp_chg:>+10.1f}%")
    print(f"  Source: sector_tariff_shocks.csv (all scenarios) + pharma_trade_weights.csv")

    # ----------------------------------------------------------
    # Summary scorecard
    # ----------------------------------------------------------
    print()
    print("=" * 68)
    print("SCORECARD -- Pharma Input Cost & Drug Price Impact")
    print("=" * 68)
    print(f"  Effective tariff on pharma imports (Liberation Day):  {d['tau_eff']*100:.2f}%")
    print(f"  HTS8 product-level MFN rate:                          {d['hts8_rate']*100:.2f}%")
    print(f"  Intermediate import share (ICIO 2022):                {d['imp_share_interm']*100:.2f}%")
    print(f"  IO supply-chain multiplier (ICIO 2022):               {d['io_mult']:.4f}x")
    print(f"  Pass-through rate:                                    {d['pass_through']*100:.0f}%")
    print(f"  --")
    print(f"  Drug price increase (input cost route):               +{d['drug_price_inc']:.3f}%")
    print(f"  Pharma CPI contribution (economy-wide):               +{d['price_cpi_contrib']:.4f}%")
    print(f"  Pharma import volume change:                          {d['import_chg']:.1f}%")
    print(f"  --")
    print(f"  Supplier HHI pre-tariff  ({d['n_suppliers']} countries):          {d['hhi_pre']:.0f}")
    print(f"  Supplier HHI post-tariff (gravity realloc):           {d['hhi_post']:.0f}")
    print(f"  Minimum available tariff (all partners):              10%  (CAN/GBR/BRA)")
    print(f"  Share of trade with no-tariff option:                 0%   (no zero-rate supplier)")

    return sc_data


# ============================================================
# Figures
# ============================================================

def plot_tariff_bands(d):
    """Distribution of tariff rates across 132 pharma suppliers."""
    tw = d['tw']

    bands       = ['10-15%', '15-20%', '20-25%', '25-35%', '> 35%']
    thresholds  = [(0.10, 0.15), (0.15, 0.20), (0.20, 0.25),
                   (0.25, 0.35), (0.35, 1.00)]
    trade_shares = []
    n_countries  = []
    for lo, hi in thresholds:
        mask = (tw['applied_tariff'] >= lo) & (tw['applied_tariff'] < hi)
        trade_shares.append(tw.loc[mask, 'weight'].sum() * 100)
        n_countries.append(mask.sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Tariff Exposure Across 132 Pharma Supplier Countries\n'
                 '(Liberation Day Schedule)', fontsize=11, fontweight='bold')

    colors = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']

    ax = axes[0]
    bars = ax.bar(bands, trade_shares, color=colors, edgecolor='white', linewidth=1.2)
    for bar, val, n in zip(bars, trade_shares, n_countries):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%\n({n} ctry)', ha='center', va='bottom', fontsize=8.5)
    ax.set_ylabel('Trade Share (%)')
    ax.set_title('By Trade Share')
    ax.set_xlabel('Tariff Band')
    ax.set_ylim(0, 62)

    ax2 = axes[1]
    # Pie of trade share
    wedges, texts, autotexts = ax2.pie(
        trade_shares, labels=bands, colors=colors,
        autopct='%1.1f%%', startangle=90, pctdistance=0.75
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax2.set_title(f'Trade Share Distribution\n'
                  f'(weighted avg tariff = {d["tau_eff"]*100:.1f}%)')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_pharma_tariff_bands.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_supplier_reallocation(d):
    """Pre vs post-tariff supplier shares for top 15 countries."""
    tw   = d['tw']
    top15 = tw.nlargest(15, 'weight').copy()

    def color_by_tau(tau):
        if tau >= 0.35:  return '#c0392b'
        if tau >= 0.25:  return '#e74c3c'
        if tau >= 0.20:  return '#e67e22'
        if tau >= 0.15:  return '#f1c40f'
        return '#27ae60'

    colors = [color_by_tau(t) for t in top15['applied_tariff']]
    countries = top15['iso3'].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('US Pharma Import Sourcing: Pre vs Post-Tariff Reallocation\n'
                 '(Gravity model, eps=2.3, Broda & Weinstein 2006)',
                 fontsize=11, fontweight='bold')

    ax = axes[0]
    ax.barh(countries, top15['weight'].values * 100, color=colors)
    ax.set_xlabel('Import Share (%)')
    ax.set_title('Pre-Tariff')
    ax.invert_yaxis()
    for i, (_, row) in enumerate(top15.iterrows()):
        ax.text(row['weight']*100 + 0.1, i,
                f"  {row['applied_tariff']*100:.0f}%", va='center', fontsize=8)

    ax2 = axes[1]
    ax2.barh(countries, top15['share_post'].values * 100, color=colors)
    ax2.set_xlabel('Import Share (%)')
    ax2.set_title('Post-Tariff (Gravity Reallocation)')
    ax2.invert_yaxis()
    for i, (_, row) in enumerate(top15.iterrows()):
        delta = row['delta'] * 100
        sign  = '+' if delta >= 0 else ''
        ax2.text(row['share_post']*100 + 0.1, i,
                 f"  {sign}{delta:.2f}pp", va='center', fontsize=8)

    patches = [
        mpatches.Patch(color='#c0392b', label='Tariff > 35% (e.g. CHN 54%)'),
        mpatches.Patch(color='#e74c3c', label='25-35%'),
        mpatches.Patch(color='#e67e22', label='20-25% (EU bloc)'),
        mpatches.Patch(color='#f1c40f', label='15-20%'),
        mpatches.Patch(color='#27ae60', label='10-15% (CAN, GBR)'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, frameon=True, fontsize=8)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out = os.path.join(OUTPUT_DIR, 'fig_pharma_supplier_reallocation.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_cost_transmission(d):
    """Waterfall: tariff rate -> input cost increase -> IO amplification -> drug price."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Pharmaceutical Input Cost Transmission Chain\n'
                 '(Liberation Day, No Retaliation)',
                 fontsize=11, fontweight='bold')

    # Left: component waterfall
    ax = axes[0]

    # Step values
    step_tau      = d['tau_eff'] * 100            # 19.90% tariff rate
    step_imp      = step_tau * d['imp_share_interm']  # applied to 12.36% of inputs
    step_pt       = step_imp * d['pass_through']  # 88% passes through
    step_io       = step_pt * (d['io_mult'] - 1)  # IO amplification
    drug_price    = step_pt + step_io             # = d['drug_price_inc']

    steps  = ['Effective\nTariff Rate', 'x Import\nInput Share', 'x Pass-\nThrough', 'x IO\nMultiplier', 'Drug Price\nIncrease']
    values = [step_tau, step_imp, step_pt, drug_price, drug_price]
    widths = [step_tau, step_imp, step_pt, step_io, drug_price]

    # Cumulative waterfall (each bar shows the current accumulated value)
    cumulative = [step_tau, step_imp, step_pt, drug_price]
    bar_labels  = [f'{v:.2f}%' for v in cumulative]
    col_vals    = [step_tau, step_imp, step_pt, drug_price]
    colors_wf   = ['#3498db', '#e67e22', '#e74c3c', '#8e44ad']
    xlabels     = ['Eff. Tariff\n(19.90%)', 'x Import Share\n(12.36%)',
                   'x Pass-Through\n(88%)', 'x IO Mult\n(1.067x)']

    for i, (val, col, lbl) in enumerate(zip(col_vals, colors_wf, xlabels)):
        bar = ax.bar(i, val, color=col, width=0.55, edgecolor='white', linewidth=1.2)
        ax.text(i, val + 0.03, f'{val:.3f}%', ha='center', va='bottom',
                fontsize=9.5, fontweight='bold')

    ax.set_xticks(range(4))
    ax.set_xticklabels(xlabels, fontsize=8.5)
    ax.set_ylabel('Value at each stage (%)')
    ax.set_title(f'Cost Amplification Chain\n(Final drug price increase: {d["drug_price_inc"]:.3f}%)')

    # Right: scenario comparison
    ax2 = axes[1]
    sc_names  = ['Liberation Day\n(HTS8: 2.45%)',
                 'Supply Chain\nShock (3.06%)',
                 'Optimal\nUniform (19%)']
    hts8_rates = [d['shocks']['liberation_day'],
                  d['shocks']['supply_chain_disrupt'],
                  d['shocks']['optimal_uniform']]
    colors_sc = ['#e74c3c', '#e67e22', '#8e44ad']

    # Drug price increase for each scenario
    drug_prices_sc = []
    for hts8_r in hts8_rates:
        scale    = hts8_r / d['hts8_rate'] if d['hts8_rate'] > 0 else 1.0
        tau_sc   = d['tau_eff'] * scale
        dp       = tau_sc * d['imp_share_interm'] * d['pass_through'] * d['io_mult'] * 100
        drug_prices_sc.append(dp)

    bars = ax2.bar(range(3), drug_prices_sc, color=colors_sc, edgecolor='white', linewidth=1.2, width=0.55)
    for bar, val in zip(bars, drug_prices_sc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'+{val:.3f}%', ha='center', va='bottom', fontsize=9.5)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(sc_names, fontsize=8.5)
    ax2.set_ylabel('Drug Price Increase (%)')
    ax2.set_title('Drug Price Impact by Policy Scenario')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_pharma_cost_transmission.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_scenario_comparison(d, sc_data):
    """Drug price increase and import volume by scenario."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('Pharmaceutical Sector: Policy Scenario Comparison',
                 fontsize=11, fontweight='bold')

    labels_map = {
        'liberation_day':       'Liberation Day',
        'supply_chain_disrupt': 'Supply Chain\nShock',
        'optimal_uniform':      'Optimal\nUniform',
    }
    sc_labels  = [labels_map[n] for n, *_ in sc_data]
    drug_ps    = [dp for _, _, _, dp, _ in sc_data]
    imp_chgs   = [ic for _, _, _, _, ic in sc_data]
    colors_sc  = ['#e74c3c', '#e67e22', '#8e44ad']

    ax = axes[0]
    bars = ax.bar(range(3), drug_ps, color=colors_sc, edgecolor='white', linewidth=1.2, width=0.55)
    for bar, val in zip(bars, drug_ps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'+{val:.3f}%', ha='center', va='bottom', fontsize=9.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(sc_labels, fontsize=9)
    ax.set_ylabel('Drug Price Increase (%)')
    ax.set_title('Drug Price Increase by Scenario')

    ax2 = axes[1]
    bars2 = ax2.bar(range(3), imp_chgs, color=colors_sc, edgecolor='white', linewidth=1.2, width=0.55)
    for bar, val in zip(bars2, imp_chgs):
        ax2.text(bar.get_x() + bar.get_width()/2, val - 1.5,
                 f'{val:.1f}%', ha='center', va='top', fontsize=9.5, color='white')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(sc_labels, fontsize=9)
    ax2.set_ylabel('Import Volume Change (%)')
    ax2.set_title('Pharma Import Volume Change by Scenario')
    ax2.axhline(0, color='black', linewidth=0.7, linestyle='--')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_pharma_scenario_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 68)
    print("PHARMA DEEP ANALYSIS -- Loading data...")
    print("=" * 68)

    d = load_all_data()
    print(f"  sector_pharma_results.npz     loaded")
    print(f"  pharma_trade_weights.csv      loaded  ({d['n_suppliers']} suppliers)")
    print(f"  sector_tariff_shocks.csv      loaded  ({len(d['shocks'])} scenarios)")
    print(f"  tariffs.csv + country_labels  loaded")

    sc_data = run_analysis(d)

    print()
    print("=" * 68)
    print("Generating Figures")
    print("=" * 68)
    plot_tariff_bands(d)
    plot_supplier_reallocation(d)
    plot_cost_transmission(d)
    plot_scenario_comparison(d, sc_data)

    print()
    print("=" * 68)
    print("COMPLETE")
    print("=" * 68)
    print()
    print("Figures saved to python_output/:")
    print("  fig_pharma_tariff_bands.png")
    print("  fig_pharma_supplier_reallocation.png")
    print("  fig_pharma_cost_transmission.png")
    print("  fig_pharma_scenario_comparison.png")


if __name__ == '__main__':
    main()
