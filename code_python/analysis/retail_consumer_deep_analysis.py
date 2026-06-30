"""
retail_consumer_deep_analysis.py
=================================
Retail & Consumer Sector -- Deep Analysis
Liberation Day Tariff Replication

Objective 1: How tariff-induced cost increases are transmitted to consumers
             (retail prices, CPI changes under different policy scenarios)
Objective 2: Distributional impact on households -- price burden by income
             quintile and overall consumer welfare

All numbers come directly from pre-computed NPZ files and project CSVs.
No external assumptions are introduced.

Data sources
------------
python_output/sector_retail_results.npz      -- pre-computed retail analysis
python_output/baseline_results.npz           -- 194-country GE model
data/daily_price_indices_cavallo_etal.csv    -- Cavallo et al. observed prices
data/retail_prices_illustrative.csv          -- product-level prices

Outputs (python_output/)
------------------------
fig_rc_obj1_transmission_chain.png
fig_rc_obj1_scenario_comparison.png
fig_rc_obj1_cavallo_empirical.png
fig_rc_obj2_quintile_incidence.png
fig_rc_obj2_welfare_vs_priceburden.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

    # --- sector_retail_results.npz (pre-computed) ---
    r = np.load(os.path.join(OUTPUT_DIR, 'sector_retail_results.npz'), allow_pickle=True)
    d['ge_cpi_noretal']            = float(r['ge_cpi_noretal'])        # 7.09%
    d['ge_cpi_retal']              = float(r['ge_cpi_retal'])          # 4.38%
    d['ge_welfare_noretal']        = float(r['ge_welfare_noretal'])    # +0.60%
    d['ge_welfare_retal']          = float(r['ge_welfare_retal'])      # -1.02%
    d['first_order_cpi']           = float(r['first_order_cpi'])       # 13.55%
    d['ge_amplification']          = float(r['ge_amplification'])      # 0.523
    d['regress_ratio']             = float(r['regress_ratio'])         # 1.414
    d['quintile_incidence_noretal']= r['quintile_incidence_noretal']   # (5,)
    d['quintile_incidence_retal']  = r['quintile_incidence_retal']     # (5,)
    d['cavallo_usa_30d']           = float(r['cavallo_usa_30d'])       # 0.071%
    d['cavallo_usa_90d']           = float(r['cavallo_usa_90d'])       # 0.103%
    d['cavallo_china_90d']         = float(r['cavallo_china_90d'])     # 1.41%
    d['retail_product_passthrough']= float(r['retail_product_passthrough'])  # 29.7%

    # --- baseline_results.npz (194-country GE, 9 scenarios) ---
    b = np.load(os.path.join(OUTPUT_DIR, 'baseline_results.npz'), allow_pickle=True)
    results_base = b['results']
    id_US = int(b['id_US'])   # 184
    # Columns: [welfare, deficit, exports, imports, employment, CPI, tariff_rev/E]
    # Scenarios: 0=USTR_no_retal, 1=?, 3=Optimal_no_retal, 4=Optimal_retal,
    #            5=Recip_retal, 7=Lump_sum, 8=High_eps
    d['baseline_scenarios'] = {
        'USTR + Income Tax Relief':     {'welfare': results_base[id_US, 0, 0],
                                          'cpi':     results_base[id_US, 5, 0]},
        'USTR + Lump-Sum Rebate':       {'welfare': results_base[id_US, 0, 7],
                                          'cpi':     results_base[id_US, 5, 7]},
        'Optimal + No Retaliation':     {'welfare': results_base[id_US, 0, 3],
                                          'cpi':     results_base[id_US, 5, 3]},
        'USTR + Reciprocal Retaliation':{'welfare': results_base[id_US, 0, 5],
                                          'cpi':     results_base[id_US, 5, 5]},
        'Optimal + Optimal Retaliation':{'welfare': results_base[id_US, 0, 4],
                                          'cpi':     results_base[id_US, 5, 4]},
    }

    # --- Cavallo et al. daily price indices ---
    cav = pd.read_csv(os.path.join(DATA_DIR, 'daily_price_indices_cavallo_etal.csv'))
    cav['date'] = pd.to_datetime(cav['date'], format='%d%b%Y', errors='coerce')
    cav = cav.sort_values('date').reset_index(drop=True)
    d['cavallo_df'] = cav

    # --- retail_prices_illustrative.csv ---
    rp = pd.read_csv(os.path.join(DATA_DIR, 'retail_prices_illustrative.csv'))
    rp.columns = [c.strip() for c in rp.columns]
    rp['price_chg_pct'] = (
        (rp['Price After Tariff'] - rp['Price Before Tariff'])
        / rp['Price Before Tariff'] * 100
    )
    d['retail_prices'] = rp

    # BLS CEX 2023 Table 1101 -- goods budget shares by income quintile
    # (stored in sector_retail.py as GOODS_BUDGET_SHARE; reproduced here verbatim)
    d['goods_budget_share'] = {
        'Q1 (Lowest)':  0.41,
        'Q2':           0.37,
        'Q3 (Middle)':  0.35,
        'Q4':           0.31,
        'Q5 (Highest)': 0.29,
    }

    return d


# ============================================================
# Objective 1 -- Price Transmission
# ============================================================

def objective1_price_transmission(d):
    print()
    print("=" * 68)
    print("OBJECTIVE 1: Price Transmission to Consumers")
    print("=" * 68)

    # ----------------------------------------------------------
    # 1A. Transmission chain
    # ----------------------------------------------------------
    print()
    print("--- 1A. Tariff-to-Consumer Price Transmission Chain ---")
    print()
    print(f"  Step 1  First-order CPI estimate (sector tariffs x import")
    print(f"          shares x pass-through rates x IO multipliers):")
    print(f"          {d['first_order_cpi']:.2f}%")
    print()
    print(f"  Step 2  General equilibrium (GE) adjustment factor:")
    print(f"          {d['ge_amplification']:.3f}x")
    print(f"          (GE wages/demand adjust, partially offsetting direct")
    print(f"          price effect -- GE < first-order by design)")
    print()
    print(f"  Step 3  GE model CPI outcome (multisector IO model):")
    print(f"          No retaliation : {d['ge_cpi_noretal']:.2f}%")
    print(f"          Full retaliation: {d['ge_cpi_retal']:.2f}%")
    print()
    print(f"  Step 4  Retail-level pass-through (from retail_prices_illustrative.csv):")
    print(f"          {d['retail_product_passthrough']*100:.1f}% average across 1,000 products")
    print()
    print(f"  Source check:")
    print(f"    first_order_cpi     = sector_retail_results.npz['first_order_cpi']")
    print(f"    ge_amplification    = sector_retail_results.npz['ge_amplification']")
    print(f"    ge_cpi_noretal      = sector_retail_results.npz['ge_cpi_noretal']")
    print(f"    retail_passthrough  = sector_retail_results.npz['retail_product_passthrough']")

    # ----------------------------------------------------------
    # 1B. Multi-scenario CPI comparison (baseline_results.npz)
    # ----------------------------------------------------------
    print()
    print("--- 1B. Consumer Price Impact by Policy Scenario ---")
    print(f"  {'Scenario':<35}  {'CPI Change':>10}  {'Welfare':>10}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}")
    for name, vals in d['baseline_scenarios'].items():
        cpi_s     = vals['cpi']
        welfare_s = vals['welfare']
        flag = " *" if welfare_s > 0 else ""
        print(f"  {name:<35}  {cpi_s:>+9.2f}%  {welfare_s:>+9.2f}%{flag}")
    print(f"  * Positive welfare despite price rise = income tax relief offsets")
    print(f"  Source: baseline_results.npz results[id_US=184, col=[0,5], scenario]")

    # ----------------------------------------------------------
    # 1C. Retail product pass-through by category
    # ----------------------------------------------------------
    rp = d['retail_prices']
    cat_pt = rp.groupby('Product Type')['price_chg_pct'].agg(['mean', 'min', 'max', 'count'])
    cat_pt.columns = ['Avg %chg', 'Min', 'Max', 'N']
    print()
    print("--- 1C. Retail Price Changes by Product Category (retail_prices_illustrative.csv) ---")
    print(f"  {'Category':<20}  {'Avg Change':>10}  {'Min':>7}  {'Max':>7}  {'N':>5}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*5}")
    for cat, row in cat_pt.iterrows():
        print(f"  {cat:<20}  {row['Avg %chg']:>+9.1f}%  {row['Min']:>+6.1f}%  {row['Max']:>+6.1f}%  {int(row['N']):>5}")
    overall_avg = rp['price_chg_pct'].mean()
    print(f"  {'Overall average':<20}  {overall_avg:>+9.1f}%")
    print(f"  Source: data/retail_prices_illustrative.csv (price_after - price_before)/price_before")

    # ----------------------------------------------------------
    # 1D. Cavallo empirical vs model
    # ----------------------------------------------------------
    cav = d['cavallo_df']
    liberation_day = pd.Timestamp('2025-04-02')
    pre_day  = cav[cav['date'] < liberation_day].iloc[-1]
    post_30  = cav[(cav['date'] >= liberation_day) &
                   (cav['date'] <= liberation_day + pd.Timedelta(days=30))]
    post_90  = cav[(cav['date'] >= liberation_day) &
                   (cav['date'] <= liberation_day + pd.Timedelta(days=90))]

    print()
    print("--- 1D. Observed vs Model-Predicted CPI (Cavallo et al. daily indices) ---")
    print(f"  Liberation Day: April 2, 2025")
    print(f"  Reference date: {pre_day['date'].date()}  (last observation before event)")
    print()
    print(f"  Country    30-day observed   90-day observed")
    print(f"  ---------  ---------------  ---------------")
    for col, label in [('index_usa','USA'), ('index_china','China'),
                       ('index_canada','Canada'), ('index_mexico','Mexico')]:
        c30 = (post_30[col].mean() / pre_day[col] - 1) * 100 if len(post_30) > 0 else float('nan')
        c90 = (post_90[col].mean() / pre_day[col] - 1) * 100 if len(post_90) > 0 else float('nan')
        print(f"  {label:<9}  {c30:>+13.3f}%  {c90:>+13.3f}%")
    print()
    print(f"  Model-predicted US CPI (GE, no retaliation): +{d['ge_cpi_noretal']:.2f}%")
    print(f"  Observed US price index change (90-day):     +{d['cavallo_usa_90d']*100:.3f}%")
    print(f"  Gap: model predicts ~{d['ge_cpi_noretal']:.0f}x more than observed short-run")
    print(f"  Interpretation: Liberation Day tariffs are announced structural shocks;")
    print(f"  Cavallo indices reflect daily retail scanner prices which adjust gradually.")
    print(f"  Source: data/daily_price_indices_cavallo_etal.csv +")
    print(f"          sector_retail_results.npz['cavallo_usa_90d']")

    return cat_pt, post_30, post_90, pre_day


# ============================================================
# Objective 2 -- Distributional Impact
# ============================================================

def objective2_distributional_impact(d):
    print()
    print("=" * 68)
    print("OBJECTIVE 2: Distributional Impact by Income Quintile")
    print("=" * 68)

    quintile_n = d['quintile_incidence_noretal']
    quintile_r = d['quintile_incidence_retal']
    goods_share = d['goods_budget_share']
    q_labels = list(goods_share.keys())

    # ----------------------------------------------------------
    # 2A. Quintile incidence table
    # ----------------------------------------------------------
    print()
    print("--- 2A. Price Burden by Income Quintile ---")
    print(f"  Formula: burden[Q] = goods_budget_share[Q] x goods_price_increase")
    print(f"           where goods_price_increase = GE_CPI / avg_goods_share")
    avg_share = np.mean(list(goods_share.values()))
    gpi_n = d['ge_cpi_noretal'] / avg_share
    gpi_r = d['ge_cpi_retal']   / avg_share
    print(f"           No-retal: {d['ge_cpi_noretal']:.2f}% / {avg_share:.3f} = {gpi_n:.2f}%")
    print(f"           Retal:    {d['ge_cpi_retal']:.2f}% / {avg_share:.3f} = {gpi_r:.2f}%")
    print()
    print(f"  {'Quintile':<16}  {'Goods Share':>11}  {'No Retal':>10}  {'Retal':>10}  {'Ratio to Q5':>12}")
    print(f"  {'-'*16}  {'-'*11}  {'-'*10}  {'-'*10}  {'-'*12}")
    q5_n = quintile_n[-1]
    for i, (q, gs) in enumerate(goods_share.items()):
        ratio = quintile_n[i] / q5_n if q5_n != 0 else 1.0
        print(f"  {q:<16}  {gs*100:>10.0f}%  {quintile_n[i]:>+9.2f}%  {quintile_r[i]:>+9.2f}%  {ratio:>11.2f}x")
    print()
    print(f"  Regressivity ratio (Q1 / Q5, no retaliation): {d['regress_ratio']:.3f}x")
    print(f"  => Lowest-income households bear {(d['regress_ratio']-1)*100:.1f}% more")
    print(f"     price burden than the highest-income households")
    print(f"  Source: sector_retail_results.npz['quintile_incidence_noretal/retal']")
    print(f"          BLS CEX 2023 Table 1101 goods budget shares (embedded in sector_retail.py)")

    # ----------------------------------------------------------
    # 2B. Welfare vs price burden
    # ----------------------------------------------------------
    print()
    print("--- 2B. Aggregate Consumer Welfare vs Price Burden ---")
    print(f"  Scenario                        GE CPI    Welfare    Verdict")
    print(f"  {'-'*30}  {'-'*8}  {'-'*9}  {'-'*30}")

    # Multisector IO model results (from sector_retail_results.npz)
    rows_io = [
        ('IO model: No retaliation',   d['ge_cpi_noretal'], d['ge_welfare_noretal']),
        ('IO model: Full retaliation', d['ge_cpi_retal'],   d['ge_welfare_retal']),
    ]
    # Baseline model results (from baseline_results.npz)
    rows_base = [
        (name, vals['cpi'], vals['welfare'])
        for name, vals in d['baseline_scenarios'].items()
    ]
    all_rows = rows_io + rows_base
    for name, cpi_v, welf_v in all_rows:
        verdict = "Welfare gain, price burden" if welf_v > 0 else "Welfare loss + price burden"
        print(f"  {name:<30}  {cpi_v:>+7.2f}%  {welf_v:>+8.2f}%  {verdict}")

    print()
    print(f"  Key insight: Income tax relief (scenario 0) converts a CPI increase")
    print(f"  of +{d['baseline_scenarios']['USTR + Income Tax Relief']['cpi']:.2f}% into a net welfare GAIN of")
    print(f"  +{d['baseline_scenarios']['USTR + Income Tax Relief']['welfare']:.2f}% by recycling tariff revenue as income tax cuts.")
    print(f"  Without that relief (lump-sum), welfare is near zero")
    print(f"  (+{d['baseline_scenarios']['USTR + Lump-Sum Rebate']['welfare']:.2f}%) with a higher CPI of")
    print(f"  +{d['baseline_scenarios']['USTR + Lump-Sum Rebate']['cpi']:.2f}%.")

    # ----------------------------------------------------------
    # 2C. Retaliation effect on distributional burden
    # ----------------------------------------------------------
    print()
    print("--- 2C. Effect of Retaliation on Quintile Burden ---")
    print(f"  Retaliation reduces the overall CPI from {d['ge_cpi_noretal']:.2f}% to")
    print(f"  {d['ge_cpi_retal']:.2f}% (a {d['ge_cpi_noretal']-d['ge_cpi_retal']:.2f}pp reduction).")
    print(f"  This proportionally reduces burden across ALL quintiles:")
    print()
    print(f"  {'Quintile':<16}  {'No Retal':>10}  {'Retal':>10}  {'Relief':>10}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*10}  {'-'*10}")
    for i, q in enumerate(q_labels):
        relief = quintile_n[i] - quintile_r[i]
        print(f"  {q:<16}  {quintile_n[i]:>+9.2f}%  {quintile_r[i]:>+9.2f}%  {relief:>+9.2f}pp")
    print()
    print(f"  However, retaliation also turns aggregate welfare negative")
    print(f"  ({d['ge_welfare_retal']:+.2f}%) -- a trade-off between distributional")
    print(f"  relief and aggregate national income.")


# ============================================================
# Figures -- Objective 1
# ============================================================

def plot_obj1_transmission_chain(d):
    """Waterfall chart: first-order -> GE -> GE CPI by scenario."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Objective 1: Tariff-to-Consumer Price Transmission',
                 fontsize=12, fontweight='bold')

    # Left: waterfall decomposition
    ax = axes[0]
    steps  = ['First-Order\nDirect CPI', 'GE Adjustment\n(0.52x factor)', 'GE CPI\n(No Retal)']
    values = [d['first_order_cpi'],
              d['ge_cpi_noretal'] - d['first_order_cpi'],
              0]  # reference bar
    cumulative = [0, d['first_order_cpi'], d['ge_cpi_noretal']]
    colors_wf  = ['#e74c3c', '#3498db', '#27ae60']

    bottoms = [0, d['first_order_cpi'], 0]
    heights = [d['first_order_cpi'],
               d['ge_cpi_noretal'] - d['first_order_cpi'],
               d['ge_cpi_noretal']]
    c_list = ['#e74c3c', '#3498db', '#27ae60']

    for i, (step, bot, ht, col) in enumerate(zip(steps, bottoms, heights, c_list)):
        ax.bar(i, ht, bottom=bot, color=col, width=0.55, edgecolor='white', linewidth=1.2)
        label_y = bot + ht/2
        ax.text(i, label_y, f'{bot+ht:.2f}%', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, fontsize=9)
    ax.set_ylabel('CPI Change (%)')
    ax.set_title('Price Transmission Chain\n(USTR Tariffs, No Retaliation)')
    ax.axhline(0, color='black', linewidth=0.7)

    # Annotate GE factor
    ax.annotate('GE adjustment\nreduces first-order\nestimate by 47.7%',
                xy=(1, (d['first_order_cpi'] + d['ge_cpi_noretal'])/2),
                xytext=(1.55, 10.5),
                fontsize=8, color='#2c3e50',
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.0))

    # Right: retail product pass-through by category
    ax2 = axes[1]
    rp  = d['retail_prices']
    cat_pt = rp.groupby('Product Type')['price_chg_pct'].mean().sort_values(ascending=False)
    bars = ax2.bar(range(len(cat_pt)), cat_pt.values,
                   color=['#e74c3c','#e67e22','#f1c40f','#27ae60','#3498db'],
                   edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, cat_pt.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(range(len(cat_pt)))
    ax2.set_xticklabels(cat_pt.index, fontsize=9, rotation=15, ha='right')
    ax2.set_ylabel('Average Price Change (%)')
    ax2.set_title(f'Retail Product Pass-Through by Category\n'
                  f'(overall avg = {rp["price_chg_pct"].mean():.1f}%; N=1,000 products)')
    ax2.axhline(d['retail_product_passthrough']*100, color='#7f8c8d',
                linestyle='--', linewidth=1.2, label=f'Overall avg {d["retail_product_passthrough"]*100:.1f}%')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_rc_obj1_transmission_chain.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_obj1_scenario_comparison(d):
    """Grouped bar chart: CPI vs welfare across all policy scenarios."""
    fig, ax = plt.subplots(figsize=(12, 5))

    names = list(d['baseline_scenarios'].keys())
    cpi_vals     = [d['baseline_scenarios'][n]['cpi']     for n in names]
    welfare_vals = [d['baseline_scenarios'][n]['welfare'] for n in names]

    x = np.arange(len(names))
    w = 0.38
    bars_cpi  = ax.bar(x - w/2, cpi_vals,     w, label='CPI Change',     color='#e74c3c', alpha=0.88)
    bars_welf = ax.bar(x + w/2, welfare_vals, w, label='Welfare Change', color='#27ae60', alpha=0.88)

    for bar in list(bars_cpi) + list(bars_welf):
        h = bar.get_height()
        y_pos = h + 0.15 if h >= 0 else h - 0.5
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{h:+.2f}%', ha='center', va='bottom', fontsize=7.5)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5, rotation=12, ha='right')
    ax.set_ylabel('Percentage Change (%)')
    ax.set_title('Consumer Price Burden vs National Welfare by Policy Scenario\n'
                 '(Source: baseline_results.npz, id_US=184)', fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_rc_obj1_scenario_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_obj1_cavallo_empirical(d):
    """Time series of Cavallo daily price indices with Liberation Day annotated."""
    cav = d['cavallo_df']
    liberation_day = pd.Timestamp('2025-04-02')

    fig, ax = plt.subplots(figsize=(13, 5))

    col_meta = {
        'index_usa':    ('#e74c3c', 'USA'),
        'index_china':  ('#f39c12', 'China'),
        'index_canada': ('#3498db', 'Canada'),
        'index_mexico': ('#27ae60', 'Mexico'),
    }
    for col, (color, label) in col_meta.items():
        ax.plot(cav['date'], cav[col], label=label, color=color, linewidth=1.5)

    ax.axvline(liberation_day, color='black', linestyle='--', linewidth=1.8,
               label='Liberation Day (Apr 2, 2025)')
    ax.axvspan(liberation_day, liberation_day + pd.Timedelta(days=90),
               alpha=0.07, color='red', label='90-day post-event window')

    # Annotate observed vs model for USA
    usa_90d_pct = d['cavallo_usa_90d'] * 100
    ax.annotate(f'USA observed +{usa_90d_pct:.2f}% (90d)\nvs GE model +{d["ge_cpi_noretal"]:.1f}%',
                xy=(liberation_day + pd.Timedelta(days=45),
                    cav.loc[cav['date'] >= liberation_day, 'index_usa'].mean()),
                xytext=(liberation_day + pd.Timedelta(days=80), 1.012),
                fontsize=8.5, color='#e74c3c',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.0))

    ax.set_xlabel('Date')
    ax.set_ylabel('Price Index (Oct 1, 2024 = 1.0)')
    ax.set_title('Cavallo et al. Daily Retail Price Indices: US vs Comparator Countries\n'
                 '(Base: Oct 1, 2024)', fontsize=11)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.3f}'))

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_rc_obj1_cavallo_empirical.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Figures -- Objective 2
# ============================================================

def plot_obj2_quintile_incidence(d):
    """Grouped bar chart: price burden by income quintile, two scenarios."""
    quintile_n = d['quintile_incidence_noretal']
    quintile_r = d['quintile_incidence_retal']
    goods_share = d['goods_budget_share']
    q_labels = list(goods_share.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Objective 2: Distributional Incidence of Liberation Day Tariffs',
                 fontsize=12, fontweight='bold')

    # Left: grouped bar -- no retal vs retal
    ax = axes[0]
    x = np.arange(len(q_labels))
    w = 0.38
    bars_n = ax.bar(x - w/2, quintile_n, w, label='No Retaliation', color='#e74c3c', alpha=0.88)
    bars_r = ax.bar(x + w/2, quintile_r, w, label='Full Retaliation', color='#c0392b', alpha=0.60)

    for bar, val in list(zip(bars_n, quintile_n)) + list(zip(bars_r, quintile_r)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(q_labels, fontsize=9)
    ax.set_ylabel('Price Burden (% of total budget)')
    ax.set_title(f'Price Burden by Quintile\n(Regressivity ratio = {d["regress_ratio"]:.2f}x)')
    ax.legend(fontsize=9)

    # Right: goods budget share driving regressivity
    ax2 = axes[1]
    share_vals = [v * 100 for v in goods_share.values()]
    colors_q = ['#c0392b', '#e74c3c', '#e67e22', '#f39c12', '#f1c40f']
    bars_s = ax2.bar(x, share_vals, color=colors_q, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars_s, share_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.0f}%', ha='center', va='bottom', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(q_labels, fontsize=9)
    ax2.set_ylabel('Goods Budget Share (%)')
    ax2.set_title('Mechanism: Lower-Income Households\nSpend More of Budget on Goods\n'
                  '(BLS CEX 2023 Table 1101)')
    ax2.set_ylim(0, 52)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_rc_obj2_quintile_incidence.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_obj2_welfare_vs_priceburden(d):
    """Scatter-style comparison: CPI vs welfare across scenarios."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # IO model points
    io_pts = [
        ('IO: No Retaliation',   d['ge_cpi_noretal'], d['ge_welfare_noretal'],  '#27ae60', 'o'),
        ('IO: Full Retaliation', d['ge_cpi_retal'],   d['ge_welfare_retal'],    '#e74c3c', 's'),
    ]
    # Baseline points
    base_pts = [
        (name, vals['cpi'], vals['welfare'], '#3498db', '^')
        for name, vals in d['baseline_scenarios'].items()
    ]

    all_pts = io_pts + base_pts
    for label, cpi_v, welf_v, color, marker in all_pts:
        ax.scatter(cpi_v, welf_v, c=color, marker=marker, s=110, zorder=5)
        offset_x = 0.1
        offset_y = 0.04
        ax.annotate(label, (cpi_v, welf_v),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=7.5, color=color)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    # Shade quadrants
    ax.fill_betweenx([-2, 0], 0, 20, alpha=0.04, color='red',
                     label='Price burden + welfare loss')
    ax.fill_betweenx([0, 2.5], 0, 20, alpha=0.04, color='green',
                     label='Price burden + welfare gain')

    ax.set_xlabel('Consumer Price Burden (CPI % change)', fontsize=10)
    ax.set_ylabel('National Welfare Change (%)', fontsize=10)
    ax.set_title('Consumer Price Burden vs National Welfare\nby Policy Scenario',
                 fontsize=11)
    ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_rc_obj2_welfare_vs_priceburden.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 68)
    print("RETAIL & CONSUMER SECTOR -- DEEP ANALYSIS")
    print("Liberation Day Tariff Replication")
    print("=" * 68)

    print("\nLoading data...")
    d = load_all_data()
    print("  sector_retail_results.npz  loaded")
    print("  baseline_results.npz       loaded  (194 countries, 9 scenarios)")
    print("  daily_price_indices_cavallo_etal.csv  loaded"
          f"  ({len(d['cavallo_df'])} days)")
    print("  retail_prices_illustrative.csv  loaded"
          f"  ({len(d['retail_prices'])} products)")

    # Objective 1
    objective1_price_transmission(d)

    # Objective 2
    objective2_distributional_impact(d)

    # Figures
    print()
    print("=" * 68)
    print("Generating Figures")
    print("=" * 68)
    plot_obj1_transmission_chain(d)
    plot_obj1_scenario_comparison(d)
    plot_obj1_cavallo_empirical(d)
    plot_obj2_quintile_incidence(d)
    plot_obj2_welfare_vs_priceburden(d)

    print()
    print("=" * 68)
    print("COMPLETE")
    print("=" * 68)
    print()
    print("Figures saved to python_output/:")
    print("  fig_rc_obj1_transmission_chain.png")
    print("  fig_rc_obj1_scenario_comparison.png")
    print("  fig_rc_obj1_cavallo_empirical.png")
    print("  fig_rc_obj2_quintile_incidence.png")
    print("  fig_rc_obj2_welfare_vs_priceburden.png")


if __name__ == '__main__':
    main()
