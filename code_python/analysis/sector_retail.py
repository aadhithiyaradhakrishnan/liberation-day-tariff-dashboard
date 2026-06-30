"""
sector_retail.py
================
Retail & Consumer Sector Analysis -- Liberation Day Tariff Impacts

Data sources used:
  - data/daily_price_indices_cavallo_etal.csv
      Cavallo et al. daily price indices for US, Canada, Mexico, China
      (Oct 2024 - Feb 2026) -- empirical pass-through evidence
  - data/retail_prices_illustrative.csv
      Product-level before/after tariff prices (999 products, 5 categories)
  - data/processed/icio_2022/io_coeff_matrix.npy + sector_map.csv
      OECD ICIO 2022 -- sector-specific IO intermediate import shares
  - data/processed/shocks/sector_tariff_shocks.csv
      HTS8-derived sector tariff shocks
  - data/base_data/tariffs.csv + country_labels.csv + ITPD trade data
      Liberation Day country tariff rates and trade flows
  - python_output/multisector_io_results.npz + baseline_results.npz
      GE model aggregate outcomes from Phase 1

Key outputs (python_output/):
  Table_S5_retail_passthrough.tex  -- sector tariff pass-through decomposition
  Table_S6_retail_incidence.tex    -- income quintile distributional incidence
  fig_retail_cpi_decomposition.png
  fig_retail_welfare_vs_cpi.png
  fig_retail_quintile_incidence.png
  fig_retail_cavallo_prices.png    -- Cavallo empirical price trajectory
  sector_retail_results.npz
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

sys.path.insert(0, os.path.join(REPO_ROOT, 'code_python'))

from utils.data_utils import (
    find_data_root,
    load_icio_sector_multipliers,
    get_hts8_tariff_by_sector,
    load_cavallo_price_indices,
    summarize_cavallo_pass_through,
    load_retail_prices_illustrative,
)

DATA_ROOT = find_data_root()

# ---------------------------------------------------------------------------
# Structural constants
# ---------------------------------------------------------------------------

# BLS CEX 2023 Table 1101 -- goods budget shares by income quintile
GOODS_BUDGET_SHARE = {
    'Q1 (Lowest)':  0.41,
    'Q2':           0.37,
    'Q3 (Middle)':  0.35,
    'Q4':           0.31,
    'Q5 (Highest)': 0.29,
}

# Sector pass-through rates, Cavallo et al. (2025)
PASS_THROUGH = {
    'Agriculture':       0.70,
    'Manufacturing':     0.85,
    'Mining and Energy': 0.60,
    'Services':          0.00,
}

BETA_LABOR = 0.49   # labor share in value added (from main_io.py)


# ---------------------------------------------------------------------------
# ITPD sector tariff exposure
# ---------------------------------------------------------------------------

def _load_itpd_sector_exposure():
    """
    Compute sector-level average tariff and import penetration from ITPD data.
    Falls back to stored constants if the ITPD CSV is not present in the worktree.
    """
    itpd_path = os.path.join(REPO_ROOT, 'data', 'ITPDS', 'trade_ITPD.csv')
    if not os.path.exists(itpd_path):
        itpd_path = os.path.join(DATA_ROOT, 'ITPDS', 'trade_ITPD.csv')

    tariffs_path   = os.path.join(DATA_ROOT, 'base_data', 'tariffs.csv')
    countries_path = os.path.join(DATA_ROOT, 'base_data', 'country_labels.csv')

    if not os.path.exists(itpd_path):
        # Use stored constants from previous run
        return {
            'Agriculture':       {'avg_tariff': 0.1774, 'import_pen': 0.2755},
            'Manufacturing':     {'avg_tariff': 0.2700, 'import_pen': 0.3232},
            'Mining and Energy': {'avg_tariff': 0.1273, 'import_pen': 0.2304},
            'Services':          {'avg_tariff': 0.1814, 'import_pen': 0.0287},
        }

    df      = pd.read_csv(itpd_path, header=None,
                          names=['exporter', 'importer', 'sector', 'value'])
    tariffs = pd.read_csv(tariffs_path, header=0)
    labels  = pd.read_csv(countries_path)

    tariffs_merged = labels[['iso3']].copy()
    tariffs_merged['applied_tariff'] = tariffs['applied_tariff'].values[:len(labels)]

    us_imports = df[(df['importer'] == 'USA') & (df['exporter'] != 'USA')].copy()
    us_imports = us_imports.merge(
        tariffs_merged.rename(columns={'iso3': 'exporter',
                                       'applied_tariff': 'tau_exp'}),
        on='exporter', how='left'
    )

    results = {}
    sectors = df['sector'].unique()
    us_total = df[df['importer'] == 'USA']['value'].sum()

    for sec in sectors:
        sec_df = us_imports[us_imports['sector'] == sec]
        total  = df[(df['importer'] == 'USA') & (df['sector'] == sec)]['value'].sum()
        imp_v  = sec_df['value'].sum()
        if imp_v > 0:
            avg_tau = float((sec_df['tau_exp'] * sec_df['value']).sum() / imp_v)
        else:
            avg_tau = 0.0
        import_pen = float(imp_v / total) if total > 0 else 0.0
        results[sec] = {'avg_tariff': avg_tau, 'import_pen': import_pen}

    return results


# ---------------------------------------------------------------------------
# LaTeX table writers
# ---------------------------------------------------------------------------

def _write_table_s5(sector_data, icio_mults, hts8_shocks,
                    first_order_cpi, ge_cpi_noretal, ge_cpi_retal,
                    ge_amplification):
    path = os.path.join(OUTPUT_DIR, 'Table_S5_retail_passthrough.tex')

    # Map ITPD sector names to ICIO model sector keys
    sector_to_model = {
        'Agriculture':       'energy_primary',    # closest available
        'Manufacturing':     'manufacturing_other',
        'Mining and Energy': 'energy_primary',
        'Services':          'services_other',
    }

    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Tariff Pass-Through Decomposition by Sector (No Retaliation)}',
        r'\label{tab:retail_passthrough}',
        r'\begin{tabular}{lccccc}',
        r'\toprule',
        r'Sector & Avg Tariff & Import Pen. & Pass-Through & IO Mult. & CPI Contrib. \\',
        r'\midrule',
    ]
    for sec, d in sector_data.items():
        pt   = PASS_THROUGH.get(sec, 0.0)
        mkey = sector_to_model.get(sec, 'manufacturing_other')
        mult = icio_mults.get(mkey, {}).get('io_multiplier', 1.0)
        cpi_contrib = pt * d['avg_tariff'] * d['import_pen'] * mult * 100
        lines.append(
            f"{sec} & {d['avg_tariff']*100:.1f}\\% & {d['import_pen']*100:.1f}\\% "
            f"& {pt*100:.0f}\\% & {mult:.3f}x & {cpi_contrib:.2f}pp \\\\"
        )

    lines += [
        r'\midrule',
        f'First-order direct CPI estimate & '
        f'\\multicolumn{{5}}{{r}}{{{first_order_cpi:.2f}\\%}} \\\\',
        f'GE model CPI (no retaliation) & '
        f'\\multicolumn{{5}}{{r}}{{{ge_cpi_noretal:.2f}\\%}} \\\\',
        f'GE model CPI (retaliation) & '
        f'\\multicolumn{{5}}{{r}}{{{ge_cpi_retal:.2f}\\%}} \\\\',
        f'GE adjustment factor (GE / direct estimate) & '
        f'\\multicolumn{{5}}{{r}}{{{ge_amplification:.2f}x}} \\\\',
        r'\midrule',
        r'\multicolumn{6}{l}{\textit{HTS8 product-level structural tariff rates (Liberation Day):}} \\',
    ]
    for sec, rate in sorted(hts8_shocks.items()):
        if sec != 'services_other':
            lines.append(
                f'\\quad {sec.replace("_", " ").title()} & '
                f'\\multicolumn{{5}}{{r}}{{{rate*100:.2f}\\%}} \\\\'
            )
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\small',
        r'\item IO multipliers from OECD ICIO 2022 (sector-specific). '
        r'HTS8 rates reflect product-level MFN tariff structure.',
        r'\item GE amplification = ratio of GE model CPI to first-order estimate.',
        r'\end{tablenotes}',
        r'\end{table}',
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


def _write_table_s6(quintile_noretal, quintile_retal, regress_ratio,
                    retail_product_passthrough):
    path = os.path.join(OUTPUT_DIR, 'Table_S6_retail_incidence.tex')
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Distributional Incidence of Liberation Day Tariffs by Income Quintile}',
        r'\label{tab:retail_incidence}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'Income Quintile & Goods Budget & Price Burden & Price Burden & Ratio to \\',
        r'& Share & (No Retaliation) & (Retaliation) & Q5 \\',
        r'\midrule',
    ]
    q5_noretal = list(quintile_noretal.values())[-1]
    for q, share in GOODS_BUDGET_SHARE.items():
        burden_n = quintile_noretal[q]
        burden_r = quintile_retal[q]
        ratio    = burden_n / q5_noretal if q5_noretal != 0 else 1.0
        lines.append(
            f"{q} & {share*100:.0f}\\% & "
            f"{burden_n:.2f}\\% & {burden_r:.2f}\\% & "
            f"{ratio:.2f}x \\\\"
        )
    lines += [
        r'\midrule',
        f'Regressivity ratio (Q1/Q5) & \\multicolumn{{4}}{{r}}'
        f'{{{regress_ratio:.2f}x}} \\\\',
        r'\midrule',
        r'\multicolumn{5}{l}{\textit{Illustrative product-level pass-through '
        r'(retail\_prices\_illustrative.csv):}} \\',
    ]
    for cat, pt in retail_product_passthrough.items():
        lines.append(
            f'\\quad {cat} & \\multicolumn{{4}}{{r}}{{{pt*100:.1f}\\%}} \\\\'
        )
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\small',
        r'\item Goods budget shares: BLS CEX 2023 Table 1101. '
        r'Product pass-through computed from retail\_prices\_illustrative.csv.',
        r'\end{tablenotes}',
        r'\end{table}',
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_cpi_decomposition(sector_data, ge_cpi_noretal, icio_mults):
    sector_to_model = {
        'Agriculture': 'energy_primary',
        'Manufacturing': 'manufacturing_other',
        'Mining and Energy': 'energy_primary',
        'Services': 'services_other',
    }
    sectors = list(sector_data.keys())
    contribs = []
    for sec in sectors:
        d    = sector_data[sec]
        pt   = PASS_THROUGH.get(sec, 0.0)
        mkey = sector_to_model.get(sec, 'manufacturing_other')
        mult = icio_mults.get(mkey, {}).get('io_multiplier', 1.0)
        contribs.append(pt * d['avg_tariff'] * d['import_pen'] * mult * 100)
    total_direct = sum(contribs)
    residual = ge_cpi_noretal - total_direct

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Consumer Price Impact Decomposition (No Retaliation)',
                 fontsize=12, fontweight='bold')

    colors = ['#27ae60', '#e74c3c', '#f39c12', '#bdc3c7']
    ax = axes[0]
    bars = ax.bar(sectors, contribs, color=colors)
    ax.set_ylabel('CPI Contribution (pp)')
    ax.set_title('By Sector (IO-Adjusted Direct Effect)')
    for bar, val in zip(bars, contribs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}pp', ha='center', va='bottom', fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right', fontsize=9)

    ax = axes[1]
    labels_pie = sectors + ['GE amplification\n(wage/price effects)']
    sizes_pie  = contribs + [max(0, residual)]
    colors_pie = colors + ['#9b59b6']
    wedges, texts, autotexts = ax.pie(
        sizes_pie, labels=labels_pie, colors=colors_pie,
        autopct='%1.1f%%', startangle=90, pctdistance=0.75
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title(f'Share of Total GE CPI ({ge_cpi_noretal:.1f}%)')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_retail_cpi_decomposition.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def _plot_welfare_vs_cpi(scenarios, welfare_vals, cpi_vals):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(scenarios))
    w = 0.35
    bars1 = ax.bar(x - w/2, welfare_vals, w, label='National Welfare', color='#27ae60')
    bars2 = ax.bar(x + w/2, cpi_vals,     w, label='Consumer Price Burden', color='#e74c3c')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.set_ylabel('Percentage Change (%)')
    ax.set_title('National Welfare vs Consumer Price Burden\n(Liberation Day Tariffs)',
                 fontsize=11)
    ax.legend()
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                h + (0.1 if h >= 0 else -0.3),
                f'{h:.2f}%', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_retail_welfare_vs_cpi.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def _plot_quintile_incidence(quintile_noretal, quintile_retal):
    fig, ax = plt.subplots(figsize=(9, 5))
    quintiles = list(quintile_noretal.keys())
    x = np.arange(len(quintiles))
    w = 0.35
    ax.bar(x - w/2, list(quintile_noretal.values()), w,
           label='No Retaliation', color='#e74c3c')
    ax.bar(x + w/2, list(quintile_retal.values()),   w,
           label='Full Retaliation', color='#c0392b', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(quintiles, fontsize=9)
    ax.set_ylabel('Effective Tariff Burden (% of Total Budget)')
    ax.set_title('Distributional Incidence: Tariff Burden by Income Quintile\n'
                 '(Lower-income households spend more of budget on tradeable goods)',
                 fontsize=10)
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_retail_quintile_incidence.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def _plot_cavallo_prices(df):
    """
    Plot Cavallo daily price index trajectory for US vs comparator countries,
    with Liberation Day annotated.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    event = pd.Timestamp('2025-04-02')
    colors = {'index_usa': '#e74c3c', 'index_canada': '#3498db',
              'index_mexico': '#27ae60', 'index_china': '#f39c12'}
    labels = {'index_usa': 'USA', 'index_canada': 'Canada',
              'index_mexico': 'Mexico', 'index_china': 'China'}

    for col, color in colors.items():
        ax.plot(df['date'], df[col], label=labels[col], color=color, linewidth=1.5)

    ax.axvline(event, color='black', linestyle='--', linewidth=1.5,
               label='Liberation Day (Apr 2, 2025)')
    ax.axvspan(event, event + pd.Timedelta(days=90),
               alpha=0.08, color='red', label='90-day post-event window')

    ax.set_xlabel('Date')
    ax.set_ylabel('Price Index (Oct 2024 = 1.0)')
    ax.set_title('Cavallo et al. Daily Price Indices: US vs Comparator Countries\n'
                 '(Base: Oct 1, 2024)', fontsize=11)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.3f}'))
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_retail_cavallo_prices.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_retail():
    print("=" * 72)
    print("Retail & Consumer Sector Analysis")
    print("=" * 72)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. GE model results
    # ------------------------------------------------------------------
    print("\n[1/7] Loading GE model results...")
    npz_ms = np.load(os.path.join(OUTPUT_DIR, 'multisector_io_results.npz'))
    results_multi = npz_ms['results_multi']
    id_US_ms = int(npz_ms['id_US'])

    # Values stored in npz are already in percentage form (e.g. 0.677 = 0.677%)
    ge_welfare_noretal = float(results_multi[id_US_ms, 0, 0])
    ge_welfare_retal   = float(results_multi[id_US_ms, 0, 1])
    ge_cpi_noretal     = float(results_multi[id_US_ms, 5, 0])
    ge_cpi_retal       = float(results_multi[id_US_ms, 5, 1])

    # Lump-sum rebate scenario from baseline (scenario 1 = USTR lump-sum)
    npz_base = np.load(os.path.join(OUTPUT_DIR, 'baseline_results.npz'))
    results_base = npz_base['results']
    id_US_base = int(npz_base['id_US'])
    ge_welfare_rebate = float(results_base[id_US_base, 0, 1])
    ge_cpi_rebate     = float(results_base[id_US_base, 5, 1]) \
                        if results_base.shape[1] > 5 else ge_cpi_noretal * 1.58

    print(f"  No retaliation: welfare={ge_welfare_noretal:.2f}%, CPI={ge_cpi_noretal:.2f}%")
    print(f"  Retaliation:    welfare={ge_welfare_retal:.2f}%, CPI={ge_cpi_retal:.2f}%")

    # ------------------------------------------------------------------
    # 2. OECD ICIO IO multipliers (sector-specific)
    # ------------------------------------------------------------------
    print("\n[2/7] Loading OECD ICIO sector IO multipliers...")
    icio_mults = load_icio_sector_multipliers(beta_labor=BETA_LABOR)
    for sec, v in sorted(icio_mults.items()):
        print(f"  {sec:25s}: import_share={v['import_share_interm']*100:.1f}%  "
              f"IO_mult={v['io_multiplier']:.3f}x")

    # ------------------------------------------------------------------
    # 3. Sector-level tariff exposure (from ITPD)
    # ------------------------------------------------------------------
    print("\n[3/7] Computing sector-level tariff exposure from ITPD...")
    sector_data = _load_itpd_sector_exposure()
    for sec, d in sector_data.items():
        print(f"  {sec:20s}: tariff={d['avg_tariff']*100:.2f}%, "
              f"import_pen={d['import_pen']*100:.2f}%")

    # ------------------------------------------------------------------
    # 4. HTS8 structural tariff rates
    # ------------------------------------------------------------------
    print("\n[4/7] Loading HTS8 sector tariff shocks...")
    hts8_shocks = get_hts8_tariff_by_sector(scenario='liberation_day_schedule')
    for sec, rate in sorted(hts8_shocks.items()):
        print(f"  {sec:25s}: {rate*100:.2f}%")

    # ------------------------------------------------------------------
    # 5. First-order CPI and GE amplification
    # ------------------------------------------------------------------
    print("\n[5/7] Computing tariff pass-through to consumer prices...")

    sector_to_model = {
        'Agriculture': 'energy_primary',
        'Manufacturing': 'manufacturing_other',
        'Mining and Energy': 'energy_primary',
        'Services': 'services_other',
    }

    first_order_cpi = 0.0
    for sec, d in sector_data.items():
        pt   = PASS_THROUGH.get(sec, 0.0)
        mkey = sector_to_model.get(sec, 'manufacturing_other')
        mult = icio_mults.get(mkey, {}).get('io_multiplier', 1.0)
        contrib = pt * d['avg_tariff'] * d['import_pen'] * mult * 100
        first_order_cpi += contrib
        print(f"  {sec:20s}: {contrib:.2f}pp  (IO mult={mult:.3f}x)")

    print(f"\n  Total IO-adjusted CPI estimate:  {first_order_cpi:.2f}%")
    print(f"  GE model CPI (no retaliation):   {ge_cpi_noretal:.2f}%")

    ge_amplification = ge_cpi_noretal / first_order_cpi if first_order_cpi > 0 else 1.0
    print(f"  GE adjustment factor: {ge_amplification:.2f}x")
    if ge_amplification < 1.0:
        print("  (GE < direct estimate: wage/demand adjustment partially offsets prices)")
    else:
        print("  (GE > direct estimate: additional general-equilibrium amplification)")

    # GE-scaled sector CPI contributions
    goods_cpi_noretal = ge_cpi_noretal  # all comes through goods sectors
    beta_goods = sum(GOODS_BUDGET_SHARE.values()) / len(GOODS_BUDGET_SHARE)
    goods_price_increase_noretal = goods_cpi_noretal / beta_goods
    goods_price_increase_retal   = ge_cpi_retal   / beta_goods

    # ------------------------------------------------------------------
    # 6. Retail prices illustrative pass-through
    # ------------------------------------------------------------------
    print("\n[6/7] Loading product-level retail prices (illustrative)...")
    rp = load_retail_prices_illustrative()
    retail_pt_by_cat = rp.groupby('product_type')['implied_pass_through'].mean().to_dict()
    overall_retail_pt = rp['implied_pass_through'].mean()
    print(f"  Products loaded: {len(rp)}")
    for cat, pt in sorted(retail_pt_by_cat.items()):
        print(f"  {cat:20s}: {pt*100:.1f}% pass-through")
    print(f"  Overall average: {overall_retail_pt*100:.1f}%")

    # ------------------------------------------------------------------
    # 6b. Cavallo price indices (empirical evidence)
    # ------------------------------------------------------------------
    print("\n  Loading Cavallo et al. daily price indices...")
    cavallo_df = load_cavallo_price_indices()
    cavallo_30  = summarize_cavallo_pass_through(window_days=30)
    cavallo_90  = summarize_cavallo_pass_through(window_days=90)
    print(f"  Date range: {cavallo_df['date'].min().date()} "
          f"to {cavallo_df['date'].max().date()}")
    print(f"  Post-Liberation Day USA change (30-day): "
          f"{cavallo_30.get('usa_change', 0)*100:+.3f}%")
    print(f"  Post-Liberation Day USA change (90-day): "
          f"{cavallo_90.get('usa_change', 0)*100:+.3f}%")
    print(f"  Post-Liberation Day China change (90-day): "
          f"{cavallo_90.get('china_change', 0)*100:+.3f}%")

    # ------------------------------------------------------------------
    # 7. Distributional incidence by quintile
    # ------------------------------------------------------------------
    print("\n[7/7] Computing distributional incidence by income quintile...")
    quintile_noretal = {}
    quintile_retal   = {}
    for q, share in GOODS_BUDGET_SHARE.items():
        quintile_noretal[q] = share * goods_price_increase_noretal
        quintile_retal[q]   = share * goods_price_increase_retal

    q1 = quintile_noretal['Q1 (Lowest)']
    q5 = quintile_noretal['Q5 (Highest)']
    regress_ratio = q1 / q5 if q5 != 0 else 1.0

    for q, burden in quintile_noretal.items():
        print(f"  {q}: {burden:.2f}% price burden")
    print(f"  Regressivity ratio (Q1/Q5): {regress_ratio:.2f}x")

    # Welfare gap scenarios
    scenarios      = ['No Retaliation', 'Full Retaliation']
    welfare_vals   = [ge_welfare_noretal,  ge_welfare_retal]
    cpi_vals       = [ge_cpi_noretal,      ge_cpi_retal]

    # ------------------------------------------------------------------
    # 8. Generate outputs
    # ------------------------------------------------------------------
    print("\n  Generating tables and figures...")

    _write_table_s5(sector_data, icio_mults, hts8_shocks,
                    first_order_cpi, ge_cpi_noretal, ge_cpi_retal,
                    ge_amplification)

    _write_table_s6(quintile_noretal, quintile_retal, regress_ratio,
                    retail_pt_by_cat)

    _plot_cpi_decomposition(sector_data, ge_cpi_noretal, icio_mults)
    _plot_welfare_vs_cpi(scenarios, welfare_vals, cpi_vals)
    _plot_quintile_incidence(quintile_noretal, quintile_retal)
    _plot_cavallo_prices(cavallo_df)

    np.savez(
        os.path.join(OUTPUT_DIR, 'sector_retail_results.npz'),
        ge_cpi_noretal        = ge_cpi_noretal,
        ge_cpi_retal          = ge_cpi_retal,
        ge_welfare_noretal    = ge_welfare_noretal,
        ge_welfare_retal      = ge_welfare_retal,
        first_order_cpi       = first_order_cpi,
        ge_amplification      = ge_amplification,
        regress_ratio         = regress_ratio,
        quintile_incidence_noretal = np.array(list(quintile_noretal.values())),
        quintile_incidence_retal   = np.array(list(quintile_retal.values())),
        cavallo_usa_30d       = cavallo_30.get('usa_change', np.nan),
        cavallo_usa_90d       = cavallo_90.get('usa_change', np.nan),
        cavallo_china_90d     = cavallo_90.get('china_change', np.nan),
        retail_product_passthrough = overall_retail_pt,
    )
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'sector_retail_results.npz')}")

    print("\n[OK] Retail & consumer analysis complete.")
    return {
        'ge_cpi_noretal':     ge_cpi_noretal,
        'ge_amplification':   ge_amplification,
        'regress_ratio':      regress_ratio,
        'cavallo_usa_90d':    cavallo_90.get('usa_change', np.nan),
    }


if __name__ == '__main__':
    analyze_retail()
