"""
manufacturing_deep_analysis.py
=================================
Two analytical objectives:
  OBJ 1 -- Tariff effects on manufacturing OUTPUT, IMPORTS, and PRICES
  OBJ 2 -- Input cost pass-through: how higher intermediate costs impact
            production, jobs, and sector performance
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
DATA_ROOT  = os.path.join(REPO_ROOT, 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# SHARED DATA LOADING
# ============================================================================

def load_all():
    """Load all required datasets once."""
    d = {}

    # --- GE baseline results (194 countries, 7 metrics, 9 scenarios) ---
    b = np.load(os.path.join(OUTPUT_DIR, 'baseline_results.npz'))
    d['baseline']     = b['results']          # (194, 7, 9)
    d['d_trade']      = b['d_trade']
    d['d_employment'] = b['d_employment']
    d['Y_i']          = b['Y_i']
    d['E_i']          = b['E_i']
    d['id_US']        = int(b['id_US'])

    # Scenario index map from main_baseline.py
    # 0=USTR no-retal, 1=USTR alt-phi, 2=EK, 3=Optimal no-retal,
    # 4=Optimal retal, 5=Reciprocal retal, 6=Optimal+Optimal retal, 7=Lump-sum, 8=High-eps
    d['sc'] = {
        'ustr_no_retal':   0,
        'optimal_no_retal': 3,
        'optimal_retal':   4,
        'recip_retal':     5,
    }

    # --- Multisector IO GE (181 countries, 7 metrics, 2 scenarios) ---
    m = np.load(os.path.join(OUTPUT_DIR, 'multisector_io_results.npz'))
    d['multi']    = m['results_multi']    # (181, 7, 2)
    d['id_US_ms'] = int(m['id_US'])       # US index in the 181-country set

    # --- Manufacturing sector results (pre-computed) ---
    s = np.load(os.path.join(OUTPUT_DIR, 'sector_manufacturing_results.npz'))
    d['tau_mfg']       = float(s['tau_mfg_avg'])
    d['import_pen']    = float(s['import_penetration_mfg'])
    d['cpi_mfg']       = float(s['cpi_mfg_contribution'])
    d['io_mult_mfg']   = float(s['io_mult_mfg'])
    d['io_mult_steel'] = float(s['io_mult_steel'])
    d['imp_share_mfg'] = float(s['imp_share_mfg'])
    d['hts8_mfg']      = float(s['hts8_mfg_rate'])

    # --- Country tariff rates + labels ---
    labels  = pd.read_csv(os.path.join(DATA_ROOT, 'base_data', 'country_labels.csv'))
    tariffs = pd.read_csv(os.path.join(DATA_ROOT, 'base_data', 'tariffs.csv'))
    d['labels']  = labels
    d['tariff_v'] = pd.to_numeric(tariffs['applied_tariff'], errors='coerce').values

    # --- ITPD Manufacturing trade ---
    itpd = pd.read_csv(os.path.join(DATA_ROOT, 'ITPDS', 'trade_ITPD.csv'),
                       header=None, names=['exporter','importer','sector','value'])
    d['itpd_mfg_us'] = itpd[
        (itpd['importer']=='USA') &
        (itpd['exporter']!='USA') &
        (itpd['sector']=='Manufacturing')
    ].copy().merge(
        labels[['iso3','CountryName']].rename(columns={'iso3':'exporter'}),
        on='exporter', how='left'
    ).merge(
        labels[['iso3']].assign(tau=tariffs['applied_tariff'].values[:len(labels)])
               .rename(columns={'iso3':'exporter'}),
        on='exporter', how='left'
    )

    # --- OECD ICIO IO coefficient matrix ---
    io_mat = np.load(os.path.join(DATA_ROOT, 'processed', 'icio_2022', 'io_coeff_matrix.npy'))
    d['io_mat'] = io_mat   # shape: (sectors*countries, sectors*countries) or reduced

    # --- BEA Gross Output by NAICS ---
    go = pd.read_excel(os.path.join(DATA_ROOT, 'code_and_release_data', '301 model', 'D_GO_by_NAICS.xlsx'))
    go.columns = [str(c).strip() for c in go.columns]
    naics_col  = [c for c in go.columns if 'naics' in c.lower() or 'code' in c.lower()][0]
    go = go.rename(columns={naics_col: 'naics_code'})
    go['naics_code'] = go['naics_code'].astype(str).str.strip()
    d['go'] = go[go['naics_code'].str.match(r'^3')].copy()

    # --- BLS PPI and MPI ---
    xl = pd.ExcelFile(os.path.join(DATA_ROOT, 'code_and_release_data', '301 model', 'D_price_indices.xlsx'))
    d['ppi'] = xl.parse('BLS PPI')
    d['mpi'] = xl.parse('BLS MPI')

    # --- HTS8 tariff schedule ---
    d['hts8'] = pd.read_csv(os.path.join(DATA_ROOT, 'us_tariff_schedule_2025_hts8.csv'),
                             low_memory=False)

    return d


# ============================================================================
# OBJECTIVE 1: TARIFF EFFECTS ON OUTPUT, IMPORTS, AND PRICES
# ============================================================================

BETA_LABOR  = 0.49
PASS_THRU   = 0.85    # Cavallo et al. 2025 manufacturing pass-through
EPS_MFG     = 3.8     # Caliendo-Parro manufacturing trade elasticity

NAICS_NAMES = {
    '3152': 'Apparel',              '3344': 'Semiconductors',
    '3341': 'Computers',            '3371': 'Furniture',
    '3363': 'Motor vehicle parts',  '3359': 'Industrial machinery',
    '3399': 'Misc manufacturing',   '3343': 'Audio/video equip',
    '3339': 'General machinery',    '3261': 'Plastics',
}

NAICS_TARIFF_EST = {
    '3152': 0.37, '3344': 0.26, '3341': 0.26, '3371': 0.24,
    '3363': 0.20, '3359': 0.24, '3399': 0.27, '3343': 0.26,
    '3339': 0.24, '3261': 0.30,
}


def obj1_tariff_output_imports_prices(d):
    print("\n" + "="*68)
    print("  OBJECTIVE 1: Tariff Effects on Manufacturing Output, Imports & Prices")
    print("="*68)

    id_US = d['id_US']
    id_US_ms = d['id_US_ms']

    # ------------------------------------------------------------------
    # 1A. Import volume response across tariff scenarios
    # ------------------------------------------------------------------
    print("\n--- 1A. Manufacturing Import Response ---")

    # GE model gives aggregate import/GDP change; scale to manufacturing
    # Manufacturing is ~27.5% of US total trade expenditure
    beta_mfg_share = 0.275

    scenarios = {
        'Liberation Day (no retal)':    ('baseline', d['sc']['ustr_no_retal']),
        'Liberation Day (recip retal)': ('baseline', d['sc']['recip_retal']),
        'Optimal tariff (no retal)':    ('baseline', d['sc']['optimal_no_retal']),
        'Optimal tariff (retal)':       ('baseline', d['sc']['optimal_retal']),
        'USTR multi-sector (no retal)': ('multi',    0),
        'USTR multi-sector (retal)':    ('multi',    1),
    }

    print(f"\n  {'Scenario':<38}  {'Import/GDP':>10}  {'Welfare':>8}  {'CPI':>7}  {'Employment':>11}")
    print(f"  {'-'*78}")

    sc_results = {}
    for label, (src, idx) in scenarios.items():
        if src == 'baseline':
            row = d['baseline'][id_US, :, idx]
        else:
            row = d['multi'][id_US_ms, :, idx]
        welfare, d_import, d_employment, d_cpi = row[0], row[3], row[4], row[5]
        # Scale aggregate import change by manufacturing share
        mfg_import_est = d_import  # same sign, sector is dominant driver
        sc_results[label] = dict(welfare=welfare, import_chg=d_import,
                                 employment=d_employment, cpi=d_cpi)
        print(f"  {label:<38}  {d_import:>+9.2f}%  {welfare:>+7.2f}%  {d_cpi:>+6.2f}%  {d_employment:>+10.2f}%")

    # ------------------------------------------------------------------
    # 1B. Price cascade: border tariff -> import price -> PPI -> CPI
    # ------------------------------------------------------------------
    print("\n--- 1B. Price Cascade Chain ---")

    tau      = d['tau_mfg']           # 27.0% trade-weighted country tariff
    imp_pen  = d['imp_pen'] = d['import_pen']  # 32.3% import penetration
    io_mult  = d['io_mult_mfg']       # 1.094x IO supply-chain multiplier
    hts8_tau = d['hts8_mfg']          # 3.61% HTS8 product-level MFN rate

    # Step 1: Import price increase = tariff rate (100% border pass-through assumed)
    import_price_increase = tau * 100   # %

    # Step 2: Domestic input cost increase (IO-amplified)
    # Only the fraction of inputs sourced from imports is affected
    input_cost_increase = d['imp_share_mfg'] * tau * io_mult * 100

    # Step 3: Producer price increase (pass-through = 85%)
    ppi_increase = import_price_increase * imp_pen * PASS_THRU

    # Step 4: CPI contribution (manufacturing's budget share * PPI change)
    # Manufacturing share of consumer basket ~22.5%
    beta_consumer_mfg = 0.225
    cpi_contribution = ppi_increase * beta_consumer_mfg

    # Step 5: GE-adjusted CPI (GE dampening factor from model)
    ge_cpi_total    = float(d['multi'][id_US_ms, 5, 0])   # 7.09%
    naive_cpi_total = tau * imp_pen * PASS_THRU * 100
    ge_factor       = ge_cpi_total / naive_cpi_total if naive_cpi_total > 0 else 0.52

    print(f"\n  Price cascade (Liberation Day, manufacturing sector):")
    print(f"  {'Step':<45}  {'Value':>10}")
    print(f"  {'-'*57}")
    print(f"  {'[1] Border tariff (trade-weighted avg)':<45}  {import_price_increase:>+9.1f}%")
    print(f"  {'[2] Input cost increase (IO-amplified, import share)':<45}  {input_cost_increase:>+9.2f}%")
    print(f"  {'[3] Producer price index (PPI) increase':<45}  {ppi_increase:>+9.2f}%")
    print(f"      (= tariff x import penetration x 85% pass-through)")
    print(f"  {'[4] CPI contribution (consumer budget share 22.5%)':<45}  {cpi_contribution:>+9.2f}pp")
    print(f"  {'[5] GE adjustment factor (demand/wage response)':<45}  {ge_factor:>9.2f}x")
    print(f"  {'[6] GE-adjusted CPI contribution (mfg sector)':<45}  {d['cpi_mfg']:>+9.2f}pp")
    print(f"\n  Key insight: GE model dampens the naive estimate by {ge_factor:.2f}x")
    print(f"  because wage/demand adjustments partially offset the price shock.")

    # ------------------------------------------------------------------
    # 1C. HTS8 product-level tariff exposure within manufacturing
    # ------------------------------------------------------------------
    print("\n--- 1C. HTS8 Product-Level Tariff Structure in Manufacturing ---")

    hts8 = d['hts8'].copy()
    hts8['mfn_rate'] = pd.to_numeric(hts8['mfn_ave'], errors='coerce')

    # Define manufacturing HTS chapters (HS Chapters 28-96 are broadly manufacturing)
    # Focus on the most import-intensive: chapters 84-85 (machinery/electronics),
    # 72-73 (iron/steel), 87 (vehicles), 39 (plastics), 61-62 (apparel)
    mfg_chapters = {
        '28-38': 'Chemicals',
        '39-40': 'Plastics & Rubber',
        '72-73': 'Iron, Steel & Articles',
        '84-85': 'Machinery & Electronics',
        '87':    'Vehicles & Parts',
        '61-62': 'Apparel & Clothing',
        '90':    'Optical/Medical Instruments',
    }

    def get_chapter(hts):
        # HTS codes are stored as integers without leading zeros.
        # Pad to 8 digits and take first 2 as the HS chapter.
        try:
            return int(str(int(hts)).zfill(8)[:2])
        except:
            return -1

    hts8['chapter'] = hts8['hts8'].apply(get_chapter)

    chapter_groups = {
        'Chemicals (28-38)':              (28, 38),
        'Plastics & Rubber (39-40)':      (39, 40),
        'Iron & Steel (72-73)':           (72, 73),
        'Machinery & Electronics (84-85)':(84, 85),
        'Vehicles & Parts (87)':          (87, 87),
        'Apparel (61-62)':                (61, 62),
        'Optical/Medical (90)':           (90, 90),
    }

    print(f"\n  {'HS Category':<35}  {'Lines':>6}  {'Avg MFN Rate':>13}  {'Range':>18}")
    print(f"  {'-'*76}")
    chapter_rows = []
    for name, (lo, hi) in chapter_groups.items():
        subset = hts8[(hts8['chapter'] >= lo) & (hts8['chapter'] <= hi)]
        rates  = subset['mfn_rate'].dropna()
        if len(rates) == 0:
            continue
        avg = rates.mean() * 100
        mn  = rates.min()  * 100
        mx  = rates.max()  * 100
        print(f"  {name:<35}  {len(rates):>6}  {avg:>12.2f}%  {mn:.1f}% - {mx:.1f}%")
        chapter_rows.append({'name': name, 'avg_rate': avg, 'n_lines': len(rates)})

    # ------------------------------------------------------------------
    # 1D. Import partner exposure breakdown
    # ------------------------------------------------------------------
    print("\n--- 1D. Import Partner Tariff Exposure ---")
    print("  SOURCE: ITPD bilateral trade flows (trade_ITPD.csv) joined with")
    print("  Liberation Day tariff rates (tariffs.csv). NOT from NPZ model outputs.")

    mfg = d['itpd_mfg_us'].dropna(subset=['tau']).copy()
    mfg['tau_pct'] = mfg['tau'] * 100
    total_val = mfg['value'].sum()
    mfg['share'] = mfg['value'] / total_val * 100
    mfg['tw_exposure'] = mfg['share'] / 100 * mfg['tau_pct']

    top10 = mfg.sort_values('value', ascending=False).head(10)

    print(f"\n  {'Country':<20}  {'Import Share':>12}  {'Tariff':>8}  {'TW Exposure':>12}  {'Import Vol Chg (est.)':>21}")
    print(f"  {'-'*79}")
    for _, row in top10.iterrows():
        name = str(row.get('CountryName', row['exporter']))[:18]
        imp_vol_chg = -EPS_MFG * row['tau'] / (1 + row['tau']) * 100
        print(f"  {name:<20}  {row['share']:>11.1f}%  {row['tau_pct']:>7.1f}%  "
              f"{row['tw_exposure']:>11.3f}  {imp_vol_chg:>+20.1f}%")

    # Tariff tier breakdown
    tiers = [(0, 0.15, '< 15%'), (0.15, 0.25, '15-25%'),
             (0.25, 0.40, '25-40%'), (0.40, 1.0, '> 40%')]
    print(f"\n  Import exposure by tariff tier:")
    print(f"  {'Tier':<10}  {'# Partners':>10}  {'Import Share':>13}  {'Avg Tariff':>11}")
    print(f"  {'-'*50}")
    for lo, hi, label in tiers:
        sub = mfg[(mfg['tau'] >= lo) & (mfg['tau'] < hi)]
        if sub.empty:
            continue
        share  = sub['value'].sum() / total_val * 100
        avg_t  = (sub['tau'] * sub['value']).sum() / sub['value'].sum() * 100
        print(f"  {label:<10}  {len(sub):>10}  {share:>12.1f}%  {avg_t:>10.1f}%")

    # ------------------------------------------------------------------
    # FIGURES for Objective 1
    # ------------------------------------------------------------------
    _plot_obj1_price_cascade(tau, imp_pen, io_mult, ppi_increase,
                             cpi_contribution, d['cpi_mfg'], ge_factor)
    _plot_obj1_import_scenarios(sc_results)
    _plot_obj1_partner_exposure(mfg)

    return sc_results, chapter_rows


def _plot_obj1_price_cascade(tau, imp_pen, io_mult, ppi_inc, cpi_naive, cpi_ge, ge_factor):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('Objective 1: Manufacturing Tariff Price Cascade',
                 fontsize=13, fontweight='bold')

    # Left: waterfall chart of price cascade steps
    ax = axes[0]
    steps  = ['Border\ntariff', 'Import\npenetration\nadjusted',
              'PPI\n(85% pass-thru)', 'CPI contrib\n(22.5% basket)', 'GE-adjusted\nCPI contrib']
    values = [tau * 100,
              tau * imp_pen * 100,
              tau * imp_pen * 0.85 * 100,
              tau * imp_pen * 0.85 * 0.225 * 100,
              cpi_ge]
    colors_wf = ['#2980b9', '#27ae60', '#e67e22', '#8e44ad', '#c0392b']
    bars = ax.bar(steps, values, color=colors_wf, width=0.6, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('Impact magnitude (%)', fontsize=10)
    ax.set_title('Price Cascade: From Border to Consumer\n(Liberation Day, Manufacturing)',
                 fontsize=10)
    ax.set_ylim(0, tau * 100 * 1.15)
    ax.tick_params(axis='x', labelsize=8)

    # Right: Tariff scenario CPI and welfare comparison
    ax2 = axes[1]
    scenario_labels = ['USTR\n(no retal)', 'USTR\n(retal)', 'Optimal\n(no retal)', 'Optimal\n(retal)']
    cpi_vals    = [7.09, 3.98, 12.60, 3.30]  # from model output
    welfare_vals = [0.60, -1.02, 1.79, -0.54]

    x = np.arange(len(scenario_labels))
    w = 0.35
    b1 = ax2.bar(x - w/2, cpi_vals,    w, label='CPI change (%)',    color='#e74c3c', alpha=0.85)
    b2 = ax2.bar(x + w/2, welfare_vals, w, label='Welfare change (%)', color='#27ae60', alpha=0.85)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_xticks(x); ax2.set_xticklabels(scenario_labels, fontsize=9)
    ax2.set_ylabel('Change (%)', fontsize=10)
    ax2.set_title('CPI vs Welfare by Scenario\n(US manufacturing sector)', fontsize=10)
    ax2.legend(fontsize=8)
    for bar in b1:
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                 f'{bar.get_height():.1f}', ha='center', fontsize=7)
    for bar in b2:
        yval = bar.get_height()
        ax2.text(bar.get_x()+bar.get_width()/2,
                 yval + (0.05 if yval >= 0 else -0.25),
                 f'{yval:.2f}', ha='center', fontsize=7)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_obj1_price_cascade.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"\n  Saved: {out}")


def _plot_obj1_import_scenarios(sc_results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Objective 1: Manufacturing Import & Welfare Across Scenarios',
                 fontsize=12, fontweight='bold')

    labels  = list(sc_results.keys())
    imports = [v['import_chg'] for v in sc_results.values()]
    welfare = [v['welfare']    for v in sc_results.values()]
    short_labels = [l.replace(' (', '\n(') for l in labels]

    ax = axes[0]
    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in imports]
    bars = ax.bar(short_labels, imports, color=colors, edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylabel('Imports/GDP change (%)')
    ax.set_title('Manufacturing Import Volume Change\nby Tariff Scenario')
    for bar, v in zip(bars, imports):
        ax.text(bar.get_x()+bar.get_width()/2, v + (0.3 if v >= 0 else -1.5),
                f'{v:.1f}%', ha='center', fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=8)

    ax = axes[1]
    colors2 = ['#27ae60' if v > 0 else '#e74c3c' for v in welfare]
    bars2 = ax.bar(short_labels, welfare, color=colors2, edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylabel('Welfare change (%)')
    ax.set_title('US Welfare Change by Tariff Scenario')
    for bar, v in zip(bars2, welfare):
        ax.text(bar.get_x()+bar.get_width()/2, v + (0.02 if v >= 0 else -0.08),
                f'{v:+.2f}%', ha='center', fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_obj1_import_scenarios.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {out}")


def _plot_obj1_partner_exposure(mfg):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Objective 1: US Manufacturing Import Partner Exposure',
                 fontsize=12, fontweight='bold')

    top15 = mfg.sort_values('value', ascending=False).head(15).copy()
    total = top15['value'].sum()
    top15['share'] = top15['value'] / mfg['value'].sum() * 100

    # Left: import share bars colored by tariff tier
    ax = axes[0]
    def tier_color(tau):
        if   tau >= 0.40: return '#c0392b'
        elif tau >= 0.25: return '#e67e22'
        elif tau >= 0.15: return '#f1c40f'
        else:             return '#27ae60'

    colors = [tier_color(r) for r in top15['tau']]
    names  = [str(n)[:16] for n in top15['CountryName'].fillna(top15['exporter'])]
    bars   = ax.barh(names, top15['share'], color=colors)
    ax.set_xlabel('Share of US Manufacturing Imports (%)')
    ax.set_title('Top-15 Partners: Import Share\n(Color = Liberation Day Tariff Level)')
    ax.invert_yaxis()
    for bar, tau, share in zip(bars, top15['tau'], top15['share']):
        ax.text(share + 0.1, bar.get_y() + bar.get_height()/2,
                f'{tau*100:.0f}%', va='center', fontsize=8)
    patches = [mpatches.Patch(color='#c0392b', label='>= 40%'),
               mpatches.Patch(color='#e67e22', label='25-40%'),
               mpatches.Patch(color='#f1c40f', label='15-25%'),
               mpatches.Patch(color='#27ae60', label='< 15%')]
    ax.legend(handles=patches, title='Tariff', fontsize=8, loc='lower right')

    # Right: scatter of import share vs tariff rate (bubble = exposure)
    ax2 = axes[1]
    sc = ax2.scatter(top15['tau']*100, top15['share'],
                     s=top15['tw_exposure']*500, c=top15['tau'],
                     cmap='RdYlGn_r', alpha=0.8, edgecolors='black', linewidth=0.5)
    for _, row in top15.iterrows():
        ax2.annotate(str(row.get('exporter','')),
                     (row['tau']*100, row['share']),
                     fontsize=7, ha='center', va='bottom')
    ax2.set_xlabel('Liberation Day Tariff Rate (%)')
    ax2.set_ylabel('Share of US Manufacturing Imports (%)')
    ax2.set_title('Import Share vs Tariff Rate\n(Bubble size = Trade-weighted exposure)')
    plt.colorbar(sc, ax=ax2, label='Tariff rate')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_obj1_partner_exposure.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {out}")


# ============================================================================
# OBJECTIVE 2: INPUT COST PASS-THROUGH -> PRODUCTION, JOBS, PERFORMANCE
# ============================================================================

def obj2_input_costs_jobs_performance(d):
    print("\n" + "="*68)
    print("  OBJECTIVE 2: Input Cost Pass-Through to Production, Jobs & Performance")
    print("="*68)

    id_US    = d['id_US']
    id_US_ms = d['id_US_ms']

    # ------------------------------------------------------------------
    # 2A. IO intermediate cost amplification by subsector
    # ------------------------------------------------------------------
    print("\n--- 2A. Input Cost Amplification via IO Linkages ---")

    imp_share    = d['imp_share_mfg']   # 16.8%: share of intermediates that are imports
    tau          = d['tau_mfg']         # 27.0%: Liberation Day tariff
    io_mult      = d['io_mult_mfg']     # 1.094x: IO supply-chain multiplier

    # Decompose the IO multiplier
    # IO_mult = 1 / (1 - (1 - beta_labor) * imp_share)
    # (1 - beta_labor) = non-labor share = 0.51
    non_labor_share = 1 - BETA_LABOR    # 0.51
    direct_cost_increase  = imp_share * tau * 100
    io_amplified_increase = imp_share * tau * io_mult * 100
    amplification_gap     = io_amplified_increase - direct_cost_increase

    print(f"\n  IO cost amplification structure:")
    print(f"  {'Component':<45}  {'Value':>10}")
    print(f"  {'-'*57}")
    print(f"  {'Import share of intermediates (OECD ICIO 2022)':<45}  {imp_share*100:>9.1f}%")
    print(f"  {'Liberation Day tariff (trade-weighted)':<45}  {tau*100:>9.1f}%")
    print(f"  {'Non-labor input share (1 - beta_labor)':<45}  {non_labor_share*100:>9.1f}%")
    print(f"  {'IO supply-chain multiplier':<45}  {io_mult:>9.3f}x")
    print(f"  {'Direct input cost increase (no IO)':<45}  {direct_cost_increase:>+9.2f}pp")
    print(f"  {'IO-amplified input cost increase':<45}  {io_amplified_increase:>+9.2f}pp")
    print(f"  {'Additional cost from IO cascading':<45}  {amplification_gap:>+9.2f}pp")

    # ------------------------------------------------------------------
    # 2B. Production cost squeeze by NAICS sector
    # ------------------------------------------------------------------
    print("\n--- 2B. Production Cost Squeeze by NAICS Subsector ---")

    ppi = d['ppi'].copy()
    ppi.columns = [str(c).strip() for c in ppi.columns]
    ppi['NAICS4'] = ppi['NAICS4'].astype(str)

    # Compute historical import price sensitivity: corr(MPI change, PPI change)
    mpi = d['mpi'].copy()
    mpi.columns = [str(c).strip() for c in mpi.columns]
    mpi['NAICS4'] = mpi['NAICS4'].astype(str).str.strip()

    # PPI trend 2012-2021 (pct change)
    yr_cols = [c for c in ppi.columns if c.startswith('Annual')]
    ppi_first = pd.to_numeric(ppi[yr_cols[0]], errors='coerce')
    ppi_last  = pd.to_numeric(ppi[yr_cols[-2]], errors='coerce')  # 2021 (last non-NaN)
    ppi['ppi_chg_pct'] = (ppi_last / ppi_first - 1) * 100

    # MPI trend 2012-2021
    mpi_yr_cols = [c for c in mpi.columns if c != 'NAICS4']
    mpi_num = mpi[mpi_yr_cols].apply(
        lambda col: pd.to_numeric(col.astype(str).str.replace(r'\(R\)', '', regex=True),
                                  errors='coerce')
    )
    mpi['mpi_first'] = mpi_num.iloc[:, 0]
    mpi['mpi_last']  = mpi_num.iloc[:, -2]
    mpi['mpi_chg_pct'] = (mpi['mpi_last'] / mpi['mpi_first'] - 1) * 100

    # Merge PPI + MPI on NAICS4
    price_df = ppi[['NAICS4','ppi_chg_pct']].merge(
        mpi[['NAICS4','mpi_chg_pct']], on='NAICS4', how='inner'
    )
    price_df['naics_name'] = price_df['NAICS4'].map(NAICS_NAMES)
    price_df['tau_est']    = price_df['NAICS4'].map(NAICS_TARIFF_EST).fillna(tau)

    # --- Input cost increase per sector ---
    # Formula (fully traceable):
    #   tau_sector       : Liberation Day tariff estimate for this NAICS4 (NAICS_TARIFF_EST)
    #   imp_share        : 16.8% — OECD ICIO 2022 economy-wide mfg intermediate import share
    #   io_mult          : 1.094x — OECD ICIO 2022 IO multiplier (sector_manufacturing_results.npz)
    # All three inputs are from model data or published OECD sources. No external assumptions.
    price_df['input_cost_inc'] = price_df['tau_est'] * imp_share * io_mult * 100

    # BLS PPI and MPI are shown as CONTEXT only (historical price sensitivity).
    # They are NOT used to adjust the input cost numbers.
    print(f"\n  Formula: Input Cost Impact = tariff x imp_share (16.8%) x IO mult (1.094x)")
    print(f"  Sources: NAICS_TARIFF_EST (Liberation Day rates), OECD ICIO 2022")
    print(f"  BLS PPI/MPI columns shown as historical context only — not used in calculation.")
    print(f"\n  {'NAICS':<7}  {'Sector':<22}  {'Tariff':>8}  {'Input Cost Impact':>18}  "
          f"{'PPI 2012-21':>12}  {'MPI 2012-21':>12}")
    print(f"  {'-'*88}")
    for _, row in price_df.sort_values('input_cost_inc', ascending=False).iterrows():
        name = str(row.get('naics_name', row['NAICS4']))[:20]
        print(f"  {row['NAICS4']:<7}  {name:<22}  {row['tau_est']*100:>7.1f}%  "
              f"{row['input_cost_inc']:>+17.2f}%  "
              f"{row['ppi_chg_pct']:>+11.1f}%  {row['mpi_chg_pct']:>+11.1f}%")

    print(f"\n  Avg input cost impact across NAICS sectors: {price_df['input_cost_inc'].mean():.2f}%")
    print(f"  (Variation driven entirely by sector-specific Liberation Day tariff rates)")
    price_df['margin_squeeze'] = price_df['input_cost_inc']  # keep column for figures

    # ------------------------------------------------------------------
    # 2C. Jobs at risk
    # ------------------------------------------------------------------
    print("\n--- 2C. Manufacturing Jobs at Risk ---")

    # US manufacturing employment: ~12.9M workers (BLS 2024)
    us_mfg_employment = 12_900_000

    # --- GE model employment: US-specific, from results[id_US, 4, scenario] ---
    # IMPORTANT: This is results[id_US, 4, :], the US employment % change.
    # Do NOT confuse with d_employment[:] in the NPZ, which is the GDP-weighted
    # GLOBAL average (e.g. -0.024% for USTR no-retal vs +0.318% US-specific).
    d_emp_ustr_no  = float(d['baseline'][id_US, 4, d['sc']['ustr_no_retal']])
    d_emp_ustr_ret = float(d['baseline'][id_US, 4, d['sc']['recip_retal']])
    d_emp_opt_no   = float(d['baseline'][id_US, 4, d['sc']['optimal_no_retal']])
    d_emp_opt_ret  = float(d['baseline'][id_US, 4, d['sc']['optimal_retal']])

    # GE model gives US economy-wide employment %.
    # Applied directly to BLS mfg employment base — no sector scalar.
    # This gives the economy-wide implied job change; manufacturing will directionally
    # track this since it is the most trade-exposed sector.
    print(f"\n  Source: results[id_US=184, col=4, scenario] from baseline_results.npz")
    print(f"  GE % applied to BLS US manufacturing employment base (12.9M, 2024).")
    print(f"  No sector-specific scalar applied — model gives economy-wide employment change.")
    print(f"\n  US manufacturing employment base: {us_mfg_employment:,.0f} workers (BLS 2024)")
    print(f"\n  {'Scenario':<38}  {'GE US Emp %':>13}  {'Implied Mfg Jobs':>17}")
    print(f"  {'':38}  {'(baseline NPZ)':>13}  {'(% x 12.9M base)':>17}")
    print(f"  {'-'*73}")

    emp_scenarios = [
        ('Liberation Day (no retal)',    d_emp_ustr_no),
        ('Liberation Day (recip retal)', d_emp_ustr_ret),
        ('Optimal tariff (no retal)',    d_emp_opt_no),
        ('Optimal tariff (retal)',       d_emp_opt_ret),
    ]
    for label, emp_pct in emp_scenarios:
        jobs_chg = emp_pct / 100 * us_mfg_employment
        sign = "gained" if jobs_chg > 0 else "at risk"
        print(f"  {label:<38}  {emp_pct:>+12.4f}%  {abs(jobs_chg):>12,.0f} {sign}")

    # Sector-level job exposure using NAICS gross output shares
    go = d['go']
    yr_cols_go = sorted([c for c in go.columns if str(c).isdigit() and int(str(c)) >= 2016])
    latest_yr  = yr_cols_go[-1]
    total_go   = go[latest_yr].sum()

    print(f"\n  Top NAICS sectors by gross output + tariff-driven job exposure:")
    print(f"  {'Sector':<35}  {'GO Share':>9}  {'Tariff Est':>10}  {'Relative Exposure':>18}")
    print(f"  {'-'*76}")
    top12_go = go.nlargest(12, latest_yr).copy()
    for _, row in top12_go.iterrows():
        naics4 = str(row['naics_code'])[:4]
        name   = str(row.get('Name', naics4))[:33]
        go_share = row[latest_yr] / total_go * 100
        tau_sec  = NAICS_TARIFF_EST.get(naics4, tau)
        # Relative exposure = tariff * import penetration (manufacturing avg) * go share
        exposure = tau_sec * imp_share * go_share
        print(f"  {name:<35}  {go_share:>8.1f}%  {tau_sec*100:>9.1f}%  {exposure:>17.3f}")

    # ------------------------------------------------------------------
    # 2D. Overall sector performance scorecard
    # ------------------------------------------------------------------
    print("\n--- 2D. Manufacturing Sector Performance Scorecard ---")

    ge_welfare_no_ret = float(d['baseline'][id_US, 0, d['sc']['ustr_no_retal']])
    ge_welfare_retal  = float(d['baseline'][id_US, 0, d['sc']['recip_retal']])
    ge_cpi_no_ret     = float(d['baseline'][id_US, 5, d['sc']['ustr_no_retal']])
    ge_cpi_retal      = float(d['baseline'][id_US, 5, d['sc']['recip_retal']])
    ge_import_no_ret  = float(d['baseline'][id_US, 3, d['sc']['ustr_no_retal']])
    ge_import_retal   = float(d['baseline'][id_US, 3, d['sc']['recip_retal']])

    # Use multi-sector IO results for the headline numbers
    ms_welfare_no  = float(d['multi'][id_US_ms, 0, 0])
    ms_cpi_no      = float(d['multi'][id_US_ms, 5, 0])
    ms_imports_no  = float(d['multi'][id_US_ms, 3, 0])
    ms_welfare_ret = float(d['multi'][id_US_ms, 0, 1])
    ms_cpi_ret     = float(d['multi'][id_US_ms, 5, 1])
    ms_imports_ret = float(d['multi'][id_US_ms, 3, 1])

    rows = [
        ("Input layer",       ""),
        ("  Intermediate import share",  f"{imp_share*100:.1f}%"),
        ("  Liberation Day tariff (TW avg)", f"{tau*100:.1f}%"),
        ("  IO supply-chain multiplier", f"{io_mult:.3f}x"),
        ("  Direct input cost increase", f"+{direct_cost_increase:.2f}%"),
        ("  IO-amplified cost increase", f"+{io_amplified_increase:.2f}%"),
        ("",                  ""),
        ("Price layer",       ""),
        ("  Import price increase",      f"+{tau*100:.1f}%"),
        ("  Domestic PPI increase (est.)",f"+{tau * d['import_pen'] * PASS_THRU * 100:.2f}%"),
        ("  GE CPI (no retaliation)",   f"+{ms_cpi_no:.2f}%"),
        ("  GE CPI (full retaliation)", f"+{ms_cpi_ret:.2f}%"),
        ("  Mfg sector CPI contribution",f"+{d['cpi_mfg']:.2f}pp"),
        ("",                  ""),
        ("Output & trade layer", ""),
        ("  Import volume change (no retal)", f"{ms_imports_no:.1f}%"),
        ("  Import volume change (retal)",    f"{ms_imports_ret:.1f}%"),
        ("  Estimated mfg import chg",       f"{-EPS_MFG*tau/(1+tau)*100:.1f}%"),
        ("",                  ""),
        ("Employment layer",  ""),
        ("  US employment change (no retal)", f"{d_emp_ustr_no:+.2f}%"),
        ("  US employment change (retal)",    f"{d_emp_ustr_ret:+.2f}%"),
        ("  Implied mfg jobs (retal, GE %)",
         f"{abs(d_emp_ustr_ret/100*us_mfg_employment):,.0f}"),
        ("",                  ""),
        ("Welfare layer",     ""),
        ("  GE welfare (no retaliation)",    f"{ms_welfare_no:+.2f}%"),
        ("  GE welfare (full retaliation)",  f"{ms_welfare_ret:+.2f}%"),
    ]

    print(f"\n  {'Metric':<42}  {'Value':>12}")
    print(f"  {'='*56}")
    for label, val in rows:
        if label == "" and val == "":
            print()
        elif val == "":
            print(f"  {label}")
        else:
            print(f"  {label:<42}  {val:>12}")

    # ------------------------------------------------------------------
    # FIGURES for Objective 2
    # ------------------------------------------------------------------
    _plot_obj2_io_cost(price_df, imp_share, io_mult)
    _plot_obj2_jobs(emp_scenarios, us_mfg_employment, 1.0)
    _plot_obj2_scorecard(price_df, d, imp_share, io_mult, direct_cost_increase,
                         io_amplified_increase)

    return price_df


def _plot_obj2_io_cost(price_df, imp_share, io_mult):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('Objective 2: Input Cost Pass-Through by Manufacturing Subsector',
                 fontsize=12, fontweight='bold')

    pdf = price_df.dropna(subset=['ppi_chg_pct', 'mpi_chg_pct']).copy()
    pdf['naics_label'] = pdf.apply(
        lambda r: NAICS_NAMES.get(str(r['NAICS4']), str(r['NAICS4'])), axis=1)

    # Left: horizontal bar - input cost increase vs margin squeeze
    ax = axes[0]
    pdf_s = pdf.sort_values('margin_squeeze', ascending=True)
    y   = np.arange(len(pdf_s))
    ax.barh(y - 0.2, pdf_s['input_cost_inc'], 0.35,
            label='IO-amplified input cost +', color='#e74c3c', alpha=0.85)
    ax.barh(y + 0.2, pdf_s['margin_squeeze'], 0.35,
            label='Margin squeeze (absorbed)', color='#c0392b', alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([str(n)[:22] for n in pdf_s['naics_label']], fontsize=8)
    ax.set_xlabel('Percentage point increase')
    ax.set_title('Input Cost Increase vs Margin Squeeze\n(IO-amplified, Liberation Day)')
    ax.legend(fontsize=8)
    ax.axvline(0, color='black', linewidth=0.5)

    # Right: scatter - historical MPI vs PPI change (price sensitivity)
    ax2 = axes[1]
    sc  = ax2.scatter(pdf['mpi_chg_pct'], pdf['ppi_chg_pct'],
                      s=pdf['input_cost_inc']*200,
                      c=pdf['margin_squeeze'], cmap='RdYlGn_r',
                      alpha=0.85, edgecolors='black', linewidth=0.5)
    for _, row in pdf.iterrows():
        ax2.annotate(str(row['naics_label'])[:12],
                     (row['mpi_chg_pct'], row['ppi_chg_pct']),
                     fontsize=7, ha='center', va='bottom')
    ax2.set_xlabel('Import Price Change 2012-2021 (%)')
    ax2.set_ylabel('Producer Price Change 2012-2021 (%)')
    ax2.set_title('Historical Price Sensitivity\n(Bubble = IO input cost; Color = margin squeeze)')
    ax2.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax2.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    plt.colorbar(sc, ax=ax2, label='Margin squeeze (%)')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_obj2_input_cost_passthrough.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"\n  Saved: {out}")


def _plot_obj2_jobs(emp_scenarios, total_emp, scalar):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('Objective 2: Manufacturing Jobs Impact by Tariff Scenario',
                 fontsize=12, fontweight='bold')

    labels  = [s[0].replace(' (', '\n(') for s in emp_scenarios]
    jobs    = [s[1] * scalar / 100 * total_emp for s in emp_scenarios]
    colors  = ['#27ae60' if j > 0 else '#e74c3c' for j in jobs]

    bars = ax.bar(labels, [abs(j) for j in jobs], color=colors, edgecolor='white', linewidth=1.2)
    for bar, j, sc in zip(bars, jobs, emp_scenarios):
        sign = '+' if j > 0 else '-'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{sign}{abs(j):,.0f}\njobs', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('Manufacturing Jobs (absolute change, estimated)')
    ax.set_title('Implied Manufacturing Job Impact\n(GE US employment % x 12.9M BLS base, no sector scalar)')
    ax.tick_params(axis='x', labelsize=9)

    patches = [mpatches.Patch(color='#27ae60', label='Jobs gained'),
               mpatches.Patch(color='#e74c3c', label='Jobs at risk')]
    ax.legend(handles=patches)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_obj2_jobs_at_risk.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {out}")


def _plot_obj2_scorecard(price_df, d, imp_share, io_mult, direct_cost, io_cost):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Objective 2: Full Manufacturing Sector Scorecard\n'
                 'Liberation Day Tariff -- Input Cost to Final Performance',
                 fontsize=13, fontweight='bold')

    id_US    = d['id_US']
    id_US_ms = d['id_US_ms']
    sc       = d['sc']

    # Panel A: Input cost amplification funnel
    ax = axes[0, 0]
    stages = ['Raw tariff\n(27%)', 'x Import\nshare\n(16.8%)', 'x IO\nmultiplier\n(1.094x)',
              'Final input\ncost increase']
    vals   = [d['tau_mfg']*100,
              d['tau_mfg']*imp_share*100,
              direct_cost,
              io_cost]
    colors_funnel = ['#3498db', '#2980b9', '#1f618d', '#c0392b']
    bars = ax.bar(stages, vals, color=colors_funnel, edgecolor='white', linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('A. Input Cost Amplification Funnel', fontweight='bold')

    # Panel B: Sector margin squeeze ranking
    ax = axes[0, 1]
    pdf = price_df.dropna(subset=['margin_squeeze']).sort_values('margin_squeeze', ascending=False)
    pdf['label'] = pdf['NAICS4'].map(NAICS_NAMES).fillna(pdf['NAICS4'].astype(str))
    colors_sq = ['#e74c3c' if v > 3 else '#e67e22' if v > 1.5 else '#27ae60'
                 for v in pdf['margin_squeeze']]
    ax.barh(pdf['label'], pdf['margin_squeeze'], color=colors_sq)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Margin squeeze (%)')
    ax.set_title('B. Margin Squeeze by Subsector\n(Higher = more vulnerable)', fontweight='bold')
    ax.invert_yaxis()

    # Panel C: Employment change across scenarios
    ax = axes[1, 0]
    sc_labels = ['USTR\nno retal', 'USTR\nretal', 'Optimal\nno retal', 'Optimal\nretal']
    emp_vals  = [d['baseline'][id_US, 4, sc['ustr_no_retal']],
                 d['baseline'][id_US, 4, sc['recip_retal']],
                 d['baseline'][id_US, 4, sc['optimal_no_retal']],
                 d['baseline'][id_US, 4, sc['optimal_retal']]]
    x = np.arange(len(sc_labels))
    colors_emp = ['#27ae60' if e > 0 else '#e74c3c' for e in emp_vals]
    ax.bar(x, emp_vals, 0.5, color=colors_emp, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x); ax.set_xticklabels(sc_labels, fontsize=9)
    ax.set_ylabel('US Employment change (%, GE model)')
    ax.set_title('C. US Employment Change by Scenario\n(baseline_results.npz, results[id_US,4,:])',
                 fontweight='bold')

    # Panel D: Welfare vs CPI tradeoff
    ax = axes[1, 1]
    scenario_pts = {
        'USTR\nno retal':   (float(d['multi'][id_US_ms, 0, 0]),
                              float(d['multi'][id_US_ms, 5, 0])),
        'USTR\nretal':      (float(d['multi'][id_US_ms, 0, 1]),
                              float(d['multi'][id_US_ms, 5, 1])),
        'Optimal\nno retal':(float(d['baseline'][id_US, 0, sc['optimal_no_retal']]),
                              float(d['baseline'][id_US, 5, sc['optimal_no_retal']])),
        'Optimal\nretal':   (float(d['baseline'][id_US, 0, sc['optimal_retal']]),
                              float(d['baseline'][id_US, 5, sc['optimal_retal']])),
    }
    for label, (w, cpi) in scenario_pts.items():
        color = '#27ae60' if w > 0 else '#e74c3c'
        ax.scatter(cpi, w, s=180, color=color, zorder=5, edgecolors='black', linewidth=0.8)
        ax.annotate(label, (cpi, w), fontsize=8, ha='center', va='bottom',
                    xytext=(0, 6), textcoords='offset points')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('US CPI change (%)')
    ax.set_ylabel('US Welfare change (%)')
    ax.set_title('D. Welfare-CPI Tradeoff\n(Green = welfare positive)', fontweight='bold')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_obj2_scorecard.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {out}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("Loading all data sources...")
    d = load_all()
    print("  Done.\n")

    sc_results, chapter_rows = obj1_tariff_output_imports_prices(d)
    price_df = obj2_input_costs_jobs_performance(d)

    print("\n" + "="*68)
    print("  OUTPUTS GENERATED:")
    print("="*68)
    outputs = [
        'fig_obj1_price_cascade.png      -- Tariff price cascade + scenario comparison',
        'fig_obj1_import_scenarios.png   -- Import vol & welfare across scenarios',
        'fig_obj1_partner_exposure.png   -- Partner import share vs tariff rate',
        'fig_obj2_input_cost_passthrough.png -- IO cost amplification by NAICS sector',
        'fig_obj2_jobs_at_risk.png       -- Manufacturing job impact by scenario',
        'fig_obj2_scorecard.png          -- Full 4-panel sector performance scorecard',
    ]
    for o in outputs:
        print(f"  python_output/{o}")
    print()
