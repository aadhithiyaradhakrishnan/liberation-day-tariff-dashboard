"""
manufacturing_sourcing_analysis.py
====================================
Manufacturing Sector -- Deep Analysis
Liberation Day Tariff Replication

Objective: To analyze how Liberation Day tariffs affect manufacturing
           production, import dependence, and prices by examining changes
           in trade flows and input costs.

Data sources
------------
python_output/sector_manufacturing_results.npz   -- pre-computed sector results
python_output/sector_manufacturing_naics.npz     -- NAICS gross output data
python_output/baseline_results.npz               -- 194-country GE model
python_output/multisector_io_results.npz         -- multisector IO model
data/ITPDS/trade_ITPD.csv                        -- bilateral manufacturing trade
data/base_data/tariffs.csv + country_labels.csv  -- Liberation Day tariff rates
data/processed/shocks/sector_tariff_shocks.csv   -- HTS8 scenario rates
OECD ICIO 2022                                   -- IO multipliers (via data_utils)

Outputs (python_output/)
------------------------
fig_mfg_top10_suppliers.png
fig_mfg_trade_flow_scenarios.png
fig_mfg_input_cost_chain.png
fig_mfg_cross_sector_comparison.png
fig_mfg_tariff_band_exposure.png
"""

import os, sys
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
from utils.data_utils import load_icio_sector_multipliers


# ============================================================
# Load data
# ============================================================

def load_data():
    d = {}

    # sector_manufacturing_results.npz
    m = np.load(os.path.join(OUTPUT_DIR, 'sector_manufacturing_results.npz'), allow_pickle=True)
    d['tau_mfg']          = float(m['tau_mfg_avg'])            # 27.00%
    d['import_pen']       = float(m['import_penetration_mfg']) # 32.32%
    d['cpi_mfg_contrib']  = float(m['cpi_mfg_contribution'])   # 6.79%
    d['mfg_import_chg']   = float(m['mfg_import_change'])      # -80.80%
    d['io_mult_mfg']      = float(m['io_mult_mfg'])            # 1.094x
    d['io_mult_steel']    = float(m['io_mult_steel'])           # 1.090x
    d['imp_share_mfg']    = float(m['imp_share_mfg'])          # 16.82%
    d['imp_share_steel']  = float(m['imp_share_steel'])         # 16.14%
    d['hts8_mfg']         = float(m['hts8_mfg_rate'])          # 3.61%
    d['hts8_steel']       = float(m['hts8_steel_rate'])         # 1.09%

    # sector_manufacturing_naics.npz
    n = np.load(os.path.join(OUTPUT_DIR, 'sector_manufacturing_naics.npz'), allow_pickle=True)
    d['total_mfg_go']     = float(n['total_mfg_go'])           # $6.32T
    d['steel_go_share']   = float(n['steel_go_share'])          # 1.75%
    d['n_naics']          = int(n['n_naics_sectors'])           # 235
    d['latest_year']      = int(n['latest_year'])               # 2021

    # baseline_results.npz -- US scenario results
    b = np.load(os.path.join(OUTPUT_DIR, 'baseline_results.npz'), allow_pickle=True)
    results  = b['results']
    id_US    = int(b['id_US'])
    # cols: [welfare, deficit, exports, imports, employment, CPI, tariff_rev]
    # scenarios: 0=USTR_no_retal, 3=Opt_no_retal, 4=Opt_retal, 5=Recip_retal, 7=Lump_sum
    d['baseline_sc'] = {
        'USTR + No Retaliation':      {c: results[id_US, i, 0] for i, c in
                                        enumerate(['welfare','deficit','exports','imports','employment','cpi','tariff_rev'])},
        'USTR + Reciprocal Retal.':   {c: results[id_US, i, 5] for i, c in
                                        enumerate(['welfare','deficit','exports','imports','employment','cpi','tariff_rev'])},
        'Optimal + Optimal Retal.':   {c: results[id_US, i, 4] for i, c in
                                        enumerate(['welfare','deficit','exports','imports','employment','cpi','tariff_rev'])},
        'USTR + Lump-Sum Rebate':     {c: results[id_US, i, 7] for i, c in
                                        enumerate(['welfare','deficit','exports','imports','employment','cpi','tariff_rev'])},
    }

    # multisector IO results
    ms = np.load(os.path.join(OUTPUT_DIR, 'multisector_io_results.npz'), allow_pickle=True)
    id_US_ms = int(ms['id_US'])
    res_ms   = ms['results_multi']
    d['ms_sc'] = {
        'IO: No Retaliation':   {c: res_ms[id_US_ms, i, 0] for i, c in
                                  enumerate(['welfare','deficit','exports','imports','employment','cpi','tariff_rev'])},
        'IO: Full Retaliation': {c: res_ms[id_US_ms, i, 1] for i, c in
                                  enumerate(['welfare','deficit','exports','imports','employment','cpi','tariff_rev'])},
    }

    # ITPD trade -- manufacturing top suppliers
    itpd = pd.read_csv(os.path.join(DATA_DIR, 'ITPDS', 'trade_ITPD.csv'),
                       header=None, names=['exporter','importer','sector','value'])
    tariffs = pd.read_csv(os.path.join(DATA_DIR, 'base_data', 'tariffs.csv'))
    labels  = pd.read_csv(os.path.join(DATA_DIR, 'base_data', 'country_labels.csv'))
    tariffs['iso3'] = labels['iso3'].values[:len(tariffs)]

    us_mfg = itpd[(itpd['importer']=='USA') & (itpd['sector']=='Manufacturing')
                  & (itpd['exporter']!='USA')].copy()
    us_mfg = us_mfg.merge(tariffs[['iso3','applied_tariff']],
                           left_on='exporter', right_on='iso3', how='left')
    total  = us_mfg['value'].sum()

    top15 = (us_mfg.groupby(['exporter','applied_tariff'])['value']
             .sum().sort_values(ascending=False).head(15).reset_index())
    top15['share_pct']  = top15['value'] / total * 100

    # Post-tariff gravity reallocation (eps = 3.8)
    EPS = 3.8
    valid = us_mfg.dropna(subset=['applied_tariff']).copy()
    valid['grav_wt']    = valid['value'] * (1 + valid['applied_tariff'])**(-EPS)
    total_grav          = valid['grav_wt'].sum()
    valid['share_post'] = valid['grav_wt'] / total_grav * 100
    valid['share_pre']  = valid['value'] / total * 100

    top15_post = valid.groupby(['exporter','applied_tariff'])\
                      .agg({'share_pre':'sum','share_post':'sum'}).reset_index()
    top15_post['delta'] = top15_post['share_post'] - top15_post['share_pre']
    top15_post = top15_post.nlargest(15, 'share_pre')

    d['us_mfg']        = us_mfg
    d['top15']         = top15
    d['top15_post']    = top15_post
    d['total_mfg_imp'] = total
    d['eps_mfg']       = EPS

    # HTS8 scenario tariffs
    shocks = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'shocks',
                                      'sector_tariff_shocks.csv'))
    d['shocks'] = shocks[shocks['model_sector']=='manufacturing_other']\
                        .set_index('scenario')['tariff_rate'].to_dict()

    # ICIO multipliers
    d['icio'] = load_icio_sector_multipliers()

    return d


# ============================================================
# Analysis
# ============================================================

def run_analysis(d):
    print()
    print("=" * 68)
    print("MANUFACTURING SECTOR -- DEEP ANALYSIS")
    print("Objective: Tariff effects on production, import dependence, prices")
    print("=" * 68)

    # ----------------------------------------------------------
    # Part A: Tariff structure
    # ----------------------------------------------------------
    print()
    print("--- Part A: Tariff Structure on Manufacturing Imports ---")
    print()
    print(f"  Two tariff layers:")
    print(f"    HTS8 product-level rate (manufacturing_other): {d['hts8_mfg']*100:.2f}%")
    print(f"    HTS8 product-level rate (steel_aluminum):      {d['hts8_steel']*100:.2f}%")
    print(f"    Country-level effective rate (trade-weighted):  {d['tau_mfg']*100:.2f}%")
    print(f"    Multiplier: {d['tau_mfg']/d['hts8_mfg']:.1f}x -- Liberation Day country charges")
    print(f"    dominate the product-level rate")
    print()
    print(f"  HTS8 rates by scenario (manufacturing_other):")
    for sc, rate in d['shocks'].items():
        if sc != 'baseline_no_tariffs':
            print(f"    {sc:<25}: {rate*100:.2f}%")
    print()
    print(f"  Top 10 manufacturing import sources:")
    print(f"  {'Rank':<5}  {'Country':<8}  {'Share':>8}  {'Tariff':>8}  "
          f"{'Post-Share':>11}  {'Change':>8}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*11}  {'-'*8}")
    for i, row in d['top15_post'].head(10).iterrows():
        rank = d['top15_post'].head(10).index.get_loc(i) + 1
        print(f"  {rank:<5}  {row['exporter']:<8}  {row['share_pre']:>7.2f}%  "
              f"{row['applied_tariff']*100:>7.0f}%  "
              f"{row['share_post']:>10.2f}%  {row['delta']:>+7.2f}pp")
    print(f"  Source: data/ITPDS/trade_ITPD.csv + tariffs.csv")

    # ----------------------------------------------------------
    # Part B: Import dependence and trade flow changes
    # ----------------------------------------------------------
    print()
    print("--- Part B: Import Dependence and Trade Flow Changes ---")
    print()
    print(f"  Import penetration (mfg, ITPD): {d['import_pen']*100:.2f}%")
    print(f"    => Nearly 1 in 3 dollars of manufacturing demand is met by imports")
    print()
    print(f"  Sector-specific import volume change (formula-based):")
    print(f"    delta_M = -eps x tau/(1+tau)")
    print(f"            = -{d['eps_mfg']} x {d['tau_mfg']:.4f}/(1+{d['tau_mfg']:.4f})")
    print(f"            = {d['mfg_import_chg']:.2f}%")
    print(f"    Source: sector_manufacturing_results.npz['mfg_import_change']")
    print()
    print(f"  GE model trade flow changes (economy-wide, baseline_results.npz):")
    print(f"  {'Scenario':<30}  {'Imports':>9}  {'Exports':>9}  {'Employment':>11}")
    print(f"  {'-'*30}  {'-'*9}  {'-'*9}  {'-'*11}")
    for name, vals in d['baseline_sc'].items():
        print(f"  {name:<30}  {vals['imports']:>+8.2f}%  "
              f"{vals['exports']:>+8.2f}%  {vals['employment']:>+10.2f}%")
    print()
    print(f"  Multisector IO model (multisector_io_results.npz):")
    for name, vals in d['ms_sc'].items():
        print(f"    {name}: imports={vals['imports']:+.2f}%  exports={vals['exports']:+.2f}%")

    # ----------------------------------------------------------
    # Part C: Input cost and price transmission
    # ----------------------------------------------------------
    print()
    print("--- Part C: Input Cost Transmission to Manufacturing Prices ---")
    print()
    print(f"  OECD ICIO 2022 -- manufacturing intermediate import share:")
    print(f"    manufacturing_other : {d['imp_share_mfg']*100:.2f}%")
    print(f"    steel_aluminum      : {d['imp_share_steel']*100:.2f}%")
    print()
    print(f"  IO supply-chain multipliers (Leontief roundabout):")
    print(f"    manufacturing_other : {d['io_mult_mfg']:.4f}x")
    print(f"    steel_aluminum      : {d['io_mult_steel']:.4f}x")
    print()

    # Input cost increase formula
    PASS_THROUGH = 0.85
    input_cost_mfg   = d['tau_mfg'] * d['imp_share_mfg'] * d['io_mult_mfg'] * 100
    input_cost_steel = d['tau_mfg'] * d['imp_share_steel'] * d['io_mult_steel'] * 100
    price_mfg        = input_cost_mfg * PASS_THROUGH
    price_steel      = input_cost_steel * PASS_THROUGH

    print(f"  Manufacturing price increase (input cost route):")
    print(f"    = tau_eff x imp_share x IO_mult")
    print(f"    = {d['tau_mfg']*100:.2f}% x {d['imp_share_mfg']*100:.2f}% x {d['io_mult_mfg']:.4f}")
    print(f"    = {input_cost_mfg:.3f}% (input cost increase)")
    print(f"    x pass-through (85%) = {price_mfg:.3f}% price increase to downstream")
    print()
    print(f"  Steel/aluminum price increase:")
    print(f"    = {d['tau_mfg']*100:.2f}% x {d['imp_share_steel']*100:.2f}% x {d['io_mult_steel']:.4f}")
    print(f"    = {input_cost_steel:.3f}% (input cost) -> {price_steel:.3f}% price increase")
    print()
    print(f"  Manufacturing CPI contribution (NPZ, all channels):")
    print(f"    = {d['cpi_mfg_contrib']:.4f}%")
    print(f"    Source: sector_manufacturing_results.npz['cpi_mfg_contribution']")
    print()
    print(f"  GE economy-wide CPI by scenario:")
    print(f"  {'Scenario':<30}  {'CPI':>8}  {'Welfare':>9}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*9}")
    for name, vals in d['baseline_sc'].items():
        print(f"  {name:<30}  {vals['cpi']:>+7.2f}%  {vals['welfare']:>+8.2f}%")

    # ----------------------------------------------------------
    # Part D: Cross-sector comparison
    # ----------------------------------------------------------
    print()
    print("--- Part D: Cross-Sector Comparison (Manufacturing vs Others) ---")
    print()
    print(f"  {'Sector':<25}  {'Imp Share':>10}  {'IO Mult':>9}  "
          f"{'HTS8 Rate':>10}  {'Input Cost':>11}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*9}  {'-'*10}  {'-'*11}")
    icio = d['icio']
    shocks_all = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'shocks',
                                          'sector_tariff_shocks.csv'))
    ld_shocks = shocks_all[shocks_all['scenario']=='liberation_day_schedule']\
                          .set_index('model_sector')['tariff_rate'].to_dict()
    for sec in ['manufacturing_other','steel_aluminum','pharma','retail_consumer','energy_primary']:
        if sec not in icio: continue
        imp_s  = icio[sec]['import_share_interm']
        io_m   = icio[sec]['io_multiplier']
        hts8_r = ld_shocks.get(sec, 0.0)
        # Use mfg effective tariff for manufacturing; note others differ
        eff_tau = d['tau_mfg'] if 'manufactur' in sec or 'steel' in sec else hts8_r
        ic = eff_tau * imp_s * io_m * 100
        print(f"  {sec:<25}  {imp_s*100:>9.2f}%  {io_m:>8.4f}x  "
              f"{hts8_r*100:>9.2f}%  {ic:>10.3f}%")
    print(f"  Source: OECD ICIO 2022 + sector_tariff_shocks.csv")

    # ----------------------------------------------------------
    # Part E: Tariff band exposure
    # ----------------------------------------------------------
    print()
    print("--- Part E: Import Exposure by Tariff Band ---")
    us_mfg = d['us_mfg']
    total  = d['total_mfg_imp']
    bands  = [
        (0,  15, '<= 15%  (CAN, MEX)'),
        (15, 22, '15-22%  (DEU, IRL, ITA)'),
        (22, 30, '22-30%  (IND, KOR)'),
        (30, 40, '30-40%  (VNM, CHE)'),
        (40, 100,'> 40%   (CHN)'),
    ]
    print()
    print(f"  {'Tariff Band':<30}  {'Countries':>10}  {'Import Value':>13}  {'Share':>7}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*13}  {'-'*7}")
    for lo, hi, label in bands:
        mask = (us_mfg['applied_tariff']*100 >= lo) & (us_mfg['applied_tariff']*100 < hi)
        n    = us_mfg.loc[mask, 'exporter'].nunique()
        val  = us_mfg.loc[mask, 'value'].sum()
        shr  = val / total * 100
        print(f"  {label:<30}  {n:>10}  ${val/1e3:>11.0f}M  {shr:>6.1f}%")
    print(f"  Source: data/ITPDS/trade_ITPD.csv")

    # Scorecard
    print()
    print("=" * 68)
    print("SCORECARD")
    print("=" * 68)
    print(f"  Effective tariff (trade-weighted):           {d['tau_mfg']*100:.2f}%")
    print(f"  HTS8 product rate (manufacturing):            {d['hts8_mfg']*100:.2f}%")
    print(f"  Import penetration:                           {d['import_pen']*100:.2f}%")
    print(f"  IO multiplier (manufacturing):                {d['io_mult_mfg']:.4f}x")
    print(f"  Manufacturing input cost increase:           +{input_cost_mfg:.3f}%")
    print(f"  Sector import volume change:                  {d['mfg_import_chg']:.2f}%")
    print(f"  Manufacturing CPI contribution:              +{d['cpi_mfg_contrib']:.4f}%")
    print(f"  Economy CPI (USTR no-retal, baseline):       +{d['baseline_sc']['USTR + No Retaliation']['cpi']:.2f}%")
    print(f"  Economy CPI (full retaliation):              +{d['baseline_sc']['USTR + Reciprocal Retal.']['cpi']:.2f}%")
    print(f"  Largest import source:  CHN 21.1% share, 54% tariff")
    print(f"  Total mfg gross output (BEA 2021):           ${d['total_mfg_go']/1e6:.2f}T")

    return input_cost_mfg, price_mfg


# ============================================================
# Figures
# ============================================================

def plot_top10_suppliers(d):
    top = d['top15_post'].head(10).copy().reset_index(drop=True)

    def col(tau):
        if tau >= 0.40: return '#c0392b'
        if tau >= 0.30: return '#e74c3c'
        if tau >= 0.22: return '#e67e22'
        if tau >= 0.15: return '#f1c40f'
        return '#27ae60'

    colors   = [col(t) for t in top['applied_tariff']]
    countries = top['exporter'].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('US Manufacturing Import Sources: Pre vs Post-Tariff Shares\n'
                 '(ITPD trade data, Liberation Day tariffs, gravity reallocation eps=3.8)',
                 fontsize=11, fontweight='bold')

    ax = axes[0]
    bars = ax.barh(countries, top['share_pre'], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel('Import Share (%)')
    ax.set_title('Pre-Tariff Share')
    for i, (bar, row) in enumerate(zip(bars, top.itertuples())):
        ax.text(bar.get_width() + 0.1, i,
                f'  {row.applied_tariff*100:.0f}% tariff', va='center', fontsize=8)
    ax.set_xlim(0, 28)

    ax2 = axes[1]
    bars2 = ax2.barh(countries, top['share_post'], color=colors)
    ax2.invert_yaxis()
    ax2.set_xlabel('Post-Tariff Share (%)')
    ax2.set_title('Post-Tariff Share (Projected)')
    for i, (bar, row) in enumerate(zip(bars2, top.itertuples())):
        sign = '+' if row.delta >= 0 else ''
        ax2.text(bar.get_width() + 0.1, i,
                 f'  {sign}{row.delta:.2f}pp', va='center', fontsize=8)
    ax2.set_xlim(0, 28)

    patches = [
        mpatches.Patch(color='#27ae60', label='<= 15% (CAN)'),
        mpatches.Patch(color='#f1c40f', label='15-22% (MEX, DEU, IRL)'),
        mpatches.Patch(color='#e67e22', label='22-30% (JPN, IND, KOR)'),
        mpatches.Patch(color='#e74c3c', label='30-40% (VNM, CHE)'),
        mpatches.Patch(color='#c0392b', label='> 40% (CHN)'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, frameon=True, fontsize=8.5)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out = os.path.join(OUTPUT_DIR, 'fig_mfg_top10_suppliers.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_trade_flow_scenarios(d):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Manufacturing Trade Flow Changes by Policy Scenario\n'
                 '(Source: baseline_results.npz, id_US=184)',
                 fontsize=11, fontweight='bold')

    names   = list(d['baseline_sc'].keys())
    imports = [d['baseline_sc'][n]['imports']    for n in names]
    exports = [d['baseline_sc'][n]['exports']    for n in names]
    cpi     = [d['baseline_sc'][n]['cpi']        for n in names]
    emp     = [d['baseline_sc'][n]['employment'] for n in names]
    colors  = ['#e74c3c','#e67e22','#c0392b','#f1c40f']
    x       = np.arange(len(names))
    short   = ['USTR\nNo Retal','USTR\nRecip Retal','Opt\nOpt Retal','USTR\nLump-Sum']

    ax = axes[0]
    b1 = ax.bar(x - 0.2, imports, 0.38, label='Imports', color=colors, alpha=0.9)
    b2 = ax.bar(x + 0.2, exports, 0.38, label='Exports',
                color=colors, alpha=0.55, edgecolor='black', linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel('%  change'); ax.set_title('Import & Export Volume Changes')
    for b in list(b1)+list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()-3,
                f'{b.get_height():.0f}%', ha='center', va='top', fontsize=7, color='white')
    ax.legend(fontsize=8)

    ax2 = axes[1]
    bars = ax2.bar(x, cpi, color=colors, edgecolor='white', linewidth=1.1, width=0.55)
    for bar, val in zip(bars, cpi):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                 f'+{val:.2f}%', ha='center', va='bottom', fontsize=8.5)
    ax2.set_xticks(x); ax2.set_xticklabels(short, fontsize=8)
    ax2.set_ylabel('CPI change (%)'); ax2.set_title('Economy-Wide CPI Impact')

    ax3 = axes[2]
    emp_colors = ['#27ae60' if v >= 0 else '#e74c3c' for v in emp]
    bars3 = ax3.bar(x, emp, color=emp_colors, edgecolor='white', linewidth=1.1, width=0.55)
    for bar, val in zip(bars3, emp):
        offset = 0.01 if val >= 0 else -0.04
        ax3.text(bar.get_x()+bar.get_width()/2, val+offset,
                 f'{val:+.2f}%', ha='center', va='bottom', fontsize=8.5)
    ax3.axhline(0, color='black', linewidth=0.7, linestyle='--')
    ax3.set_xticks(x); ax3.set_xticklabels(short, fontsize=8)
    ax3.set_ylabel('Employment change (%)'); ax3.set_title('US Employment Change')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_mfg_trade_flow_scenarios.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_input_cost_chain(d, input_cost_mfg, price_mfg):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Manufacturing Input Cost Transmission Chain\n'
                 '(USTR Liberation Day, No Retaliation)',
                 fontsize=11, fontweight='bold')

    steps  = ['Effective\nTariff\n(27.0%)',
              'x Import\nInput Share\n(16.82%)',
              'x IO\nMultiplier\n(1.094x)',
              'Downstream\nPrice Rise\n(x85% PT)']
    values = [d['tau_mfg']*100,
              d['tau_mfg']*d['imp_share_mfg']*100,
              input_cost_mfg,
              price_mfg]
    colors = ['#3498db','#e67e22','#e74c3c','#8e44ad']

    ax = axes[0]
    for i, (step, val, col) in enumerate(zip(steps, values, colors)):
        ax.bar(i, val, color=col, width=0.55, edgecolor='white', linewidth=1.5)
        ax.text(i, val + 0.06, f'{val:.3f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(steps, fontsize=8.5)
    ax.set_ylabel('Value at each stage (%)')
    ax.set_title(f'Input cost increase: +{input_cost_mfg:.3f}%\n'
                 f'Downstream price rise: +{price_mfg:.3f}%')

    # Right: manufacturing vs steel comparison
    ax2 = axes[1]
    cats    = ['Mfg Input\nCost', 'Steel Input\nCost', 'Mfg CPI\nContrib.', 'Economy\nCPI (no retal)']
    ic_steel = d['tau_mfg'] * d['imp_share_steel'] * d['io_mult_steel'] * 100
    vals2   = [input_cost_mfg, ic_steel,
               d['cpi_mfg_contrib'],
               d['baseline_sc']['USTR + No Retaliation']['cpi']]
    cols2   = ['#e74c3c','#e67e22','#8e44ad','#3498db']
    bars = ax2.bar(range(4), vals2, color=cols2, edgecolor='white', linewidth=1.2, width=0.55)
    for bar, val in zip(bars, vals2):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                 f'{val:.3f}%', ha='center', va='bottom', fontsize=9.5)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(cats, fontsize=8.5)
    ax2.set_ylabel('%')
    ax2.set_title('Key Price Metrics Comparison')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_mfg_input_cost_chain.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_cross_sector_comparison(d):
    icio     = d['icio']
    shocks   = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'shocks',
                                        'sector_tariff_shocks.csv'))
    ld       = shocks[shocks['scenario']=='liberation_day_schedule']\
                     .set_index('model_sector')['tariff_rate'].to_dict()

    sectors  = ['manufacturing_other','steel_aluminum','pharma',
                'retail_consumer','energy_primary']
    labels   = ['Manufacturing','Steel/Al','Pharma','Retail','Energy']
    colors   = ['#e74c3c','#e67e22','#3498db','#27ae60','#95a5a6']

    imp_shares = [icio[s]['import_share_interm']*100 for s in sectors]
    io_mults   = [icio[s]['io_multiplier']           for s in sectors]
    hts8_rates = [ld.get(s, 0)*100                   for s in sectors]

    x = np.arange(len(sectors))
    w = 0.28
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Cross-Sector Comparison: Manufacturing vs Other Sectors\n'
                 '(OECD ICIO 2022 + sector_tariff_shocks.csv)',
                 fontsize=11, fontweight='bold')

    for ax, vals, title, ylabel in [
        (axes[0], imp_shares, 'Intermediate Import Share (%)\n(OECD ICIO 2022)', '%'),
        (axes[1], io_mults,   'IO Supply-Chain Multiplier\n(Leontief roundabout)', 'x'),
        (axes[2], hts8_rates, 'HTS8 Product-Level Rate (%)\n(Liberation Day)', '%'),
    ]:
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=1.1, width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002 if ylabel=='x' else bar.get_height()+0.05,
                    f'{val:.2f}{ylabel}', ha='center', va='bottom', fontsize=8.5)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5, rotation=10)
        ax.set_ylabel(ylabel); ax.set_title(title)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_mfg_cross_sector_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_tariff_band_exposure(d):
    us_mfg = d['us_mfg']
    total  = d['total_mfg_imp']
    bands  = [('<= 15%\n(CAN,MEX)',0,15),('15-22%\n(DEU,IRL)',15,22),
              ('22-30%\n(JPN,IND,KOR)',22,30),('30-40%\n(VNM)',30,40),('> 40%\n(CHN)',40,100)]
    colors = ['#27ae60','#f1c40f','#e67e22','#e74c3c','#c0392b']

    pre_shares, post_shares = [], []
    valid = d['top15_post']
    all_valid = d['us_mfg'].dropna(subset=['applied_tariff']).copy()
    all_valid['grav_wt']    = all_valid['value'] * (1+all_valid['applied_tariff'])**(-d['eps_mfg'])
    all_valid['share_pre']  = all_valid['value'] / total * 100
    all_valid['share_post'] = all_valid['grav_wt'] / all_valid['grav_wt'].sum() * 100

    for label, lo, hi in bands:
        mask = (all_valid['applied_tariff']*100 >= lo) & (all_valid['applied_tariff']*100 < hi)
        pre_shares.append(all_valid.loc[mask,'share_pre'].sum())
        post_shares.append(all_valid.loc[mask,'share_post'].sum())

    xlabels = [b[0] for b in bands]
    x = np.arange(len(bands))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x-w/2, pre_shares,  w, label='Pre-Tariff',  color=colors, alpha=0.9)
    b2 = ax.bar(x+w/2, post_shares, w, label='Post-Tariff',
                color=colors, alpha=0.55, edgecolor='black', linewidth=0.8)
    for bar, val in list(zip(b1,pre_shares))+list(zip(b2,post_shares)):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel('Import Share (%)'); ax.set_title(
        'Manufacturing Import Share Reallocation by Tariff Band\n'
        '(Pre vs Post-Tariff, Liberation Day, gravity eps=3.8)', fontsize=10)
    ax.legend(fontsize=9)
    ax.axhline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_mfg_tariff_band_exposure.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 68)
    print("MANUFACTURING ANALYSIS -- Loading data...")
    print("=" * 68)

    d = load_data()
    print(f"  sector_manufacturing_results.npz  loaded")
    print(f"  sector_manufacturing_naics.npz    loaded  ({d['n_naics']} NAICS sectors)")
    print(f"  baseline_results.npz              loaded  (4 scenarios)")
    print(f"  multisector_io_results.npz        loaded")
    print(f"  ITPD trade data                   loaded  ({d['us_mfg']['exporter'].nunique()} mfg import partners)")
    print(f"  OECD ICIO 2022                    loaded  ({len(d['icio'])} model sectors)")

    input_cost_mfg, price_mfg = run_analysis(d)

    print()
    print("=" * 68)
    print("Generating Figures")
    print("=" * 68)
    plot_top10_suppliers(d)
    plot_trade_flow_scenarios(d)
    plot_input_cost_chain(d, input_cost_mfg, price_mfg)
    plot_cross_sector_comparison(d)
    plot_tariff_band_exposure(d)

    print()
    print("=" * 68)
    print("COMPLETE -- Figures saved to python_output/:")
    print("  fig_mfg_top10_suppliers.png")
    print("  fig_mfg_trade_flow_scenarios.png")
    print("  fig_mfg_input_cost_chain.png")
    print("  fig_mfg_cross_sector_comparison.png")
    print("  fig_mfg_tariff_band_exposure.png")
    print("=" * 68)


if __name__ == '__main__':
    main()
