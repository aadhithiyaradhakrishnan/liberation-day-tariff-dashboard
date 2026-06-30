"""
sector_pharma.py
================
Pharmaceutical Sector Analysis -- Liberation Day Tariff Impacts

Data sources used:
  - data/processed/shocks/pharma_trade_weights.csv
      Actual bilateral pharma trade weights (132 exporting countries, 2024 data)
  - data/processed/icio_2022/io_coeff_matrix.npy + sector_map.csv
      OECD ICIO 2022 -- actual pharma IO intermediate import share -> IO multiplier
  - data/processed/shocks/sector_tariff_shocks.csv
      HTS8-derived sector tariff shocks (product-level MFN structure)
  - data/base_data/tariffs.csv + country_labels.csv
      Liberation Day country-level tariff rates (194 countries)
  - python_output/multisector_io_results.npz
      GE model aggregate outcomes (welfare, CPI, imports) from Phase 1

Key outputs (python_output/):
  Table_S3_pharma_price.tex      -- price impact, import volumes, IO multiplier
  Table_S4_pharma_suppliers.tex  -- HHI + top suppliers pre/post tariff
  fig_pharma_supplier_shift.png  -- supplier share redistribution
  fig_pharma_comparison.png      -- pharma vs economy CPI/welfare comparison
  sector_pharma_results.npz      -- compressed result arrays
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
OUTPUT_DIR  = os.path.join(REPO_ROOT, 'python_output')

# Add utils to path
sys.path.insert(0, os.path.join(REPO_ROOT, 'code_python'))

from utils.data_utils import (
    find_data_root,
    get_model_sector_io_multiplier,
    load_pharma_trade_weights,
    get_hts8_tariff_by_sector,
    load_sector_tariff_shocks,
)

DATA_ROOT = find_data_root()

# ---------------------------------------------------------------------------
# Structural constants (literature sources)
# ---------------------------------------------------------------------------
PHARMA_MFG_SHARE    = 0.082   # pharma as share of US mfg imports (USITC 2023)
PHARMA_PCE_SHARE    = 0.027   # pharma as share of PCE (BEA 2023)
PHARMA_PASS_THROUGH = 0.88    # Cavallo et al. (2025)
EPS_PHARMA          = 2.3     # trade elasticity, Broda & Weinstein (2006)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_trade_weighted_tariff(df):
    """
    Compute trade-weighted effective tariff from pharma_trade_weights data.
    df must have columns: weight, tau_country
    """
    valid = df.dropna(subset=['tau_country'])
    # Renormalize weights after dropping unmatched countries
    total_w = valid['weight'].sum()
    if total_w == 0:
        return np.nan
    return float((valid['weight'] * valid['tau_country']).sum() / total_w)


def _compute_hhi(shares):
    """HHI = sum of squared market shares (shares in 0-1 range -> result in 0-10000)."""
    return float(np.sum((np.array(shares) * 100) ** 2))


def _post_tariff_shares(df, eps):
    """
    Gravity-style sourcing reallocation post-tariff.
    New share proportional to pre-tariff share * (1 + tau)^(-eps).
    df must have columns: weight, tau_country
    """
    valid = df.dropna(subset=['tau_country']).copy()
    valid['gravity_wt'] = valid['weight'] * (1.0 + valid['tau_country']) ** (-eps)
    total = valid['gravity_wt'].sum()
    valid['share_post'] = valid['gravity_wt'] / total
    return valid


# ---------------------------------------------------------------------------
# LaTeX table writers
# ---------------------------------------------------------------------------

def _write_table_s3(tau_eff, hts8_rate, io_mult, imp_share,
                    price_noretal, price_retal,
                    import_chg_noretal, import_chg_retal,
                    pharma_welfare_noretal, pharma_welfare_retal,
                    ge_cpi_noretal, ge_cpi_retal):
    """Table S3: Pharmaceutical price and import volume impacts."""
    path = os.path.join(OUTPUT_DIR, 'Table_S3_pharma_price.tex')
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Pharmaceutical Sector: Liberation Day Tariff Impacts}',
        r'\label{tab:pharma_price}',
        r'\begin{tabular}{lcc}',
        r'\toprule',
        r'Metric & No Retaliation & Full Retaliation \\',
        r'\midrule',
        r'\multicolumn{3}{l}{\textit{A. Tariff Structure}} \\[2pt]',
        f'Effective tariff (trade-weighted, Liberation Day) '
        f'& \\multicolumn{{2}}{{c}}{{{tau_eff*100:.1f}\\%}} \\\\',
        f'HTS8 product-level MFN rate (pharma, avg.) '
        f'& \\multicolumn{{2}}{{c}}{{{hts8_rate*100:.2f}\\%}} \\\\',
        f'Pass-through rate (Cavallo et al., 2025) '
        f'& \\multicolumn{{2}}{{c}}{{{PHARMA_PASS_THROUGH*100:.0f}\\%}} \\\\',
        r'[4pt]\multicolumn{3}{l}{\textit{B. Supply Chain}} \\[2pt]',
        f'ICIO intermediate import share (pharma) '
        f'& \\multicolumn{{2}}{{c}}{{{imp_share*100:.1f}\\%}} \\\\',
        f'IO supply-chain multiplier '
        f'& \\multicolumn{{2}}{{c}}{{{io_mult:.3f}x}} \\\\',
        r'[4pt]\multicolumn{3}{l}{\textit{C. Price \& Volume Effects}} \\[2pt]',
        f'Pharma price increase & {price_noretal:.1f}\\% & {price_retal:.1f}\\% \\\\',
        f'Economy-wide CPI & {ge_cpi_noretal:.1f}\\% & {ge_cpi_retal:.1f}\\% \\\\',
        f'Pharma vs economy CPI ratio '
        f'& {price_noretal/ge_cpi_noretal:.2f}x & {price_retal/ge_cpi_retal:.2f}x \\\\',
        f'Pharma import volume change '
        f'& {import_chg_noretal:.1f}\\% & {import_chg_retal:.1f}\\% \\\\',
        r'[4pt]\multicolumn{3}{l}{\textit{D. Welfare}} \\[2pt]',
        f'Pharma consumer welfare loss '
        f'& {pharma_welfare_noretal:.3f}\\% & {pharma_welfare_retal:.3f}\\% \\\\',
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\small',
        r'\item Sources: Trade weights from pharma\_bilateral\_trade.xlsx (2024); '
        r'IO multiplier from OECD ICIO 2022; tariff rates from Liberation Day',
        r'schedule; pass-through from Cavallo et al.\ (2025).',
        r'\end{tablenotes}',
        r'\end{table}',
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


def _write_table_s4(top_n, hhi_pre, hhi_post, n_countries):
    """Table S4: Supplier concentration and HHI."""
    path = os.path.join(OUTPUT_DIR, 'Table_S4_pharma_suppliers.tex')
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{US Pharmaceutical Import Suppliers: Pre- and Post-Tariff Shares}',
        r'\label{tab:pharma_suppliers}',
        r'\begin{tabular}{llcccc}',
        r'\toprule',
        r'Rank & Country & Tariff & Pre-Tariff Share & Post-Tariff Share & Change \\',
        r'\midrule',
    ]
    for i, row in top_n.iterrows():
        tau_str = f"{row['tau_country']*100:.0f}\\%" if pd.notna(row['tau_country']) else '--'
        delta = row['share_post'] - row['weight']
        arrow = r'$\uparrow$' if delta > 0 else r'$\downarrow$'
        lines.append(
            f"{row['rank']} & {row['country_name']} & {tau_str} "
            f"& {row['weight']*100:.1f}\\% "
            f"& {row['share_post']*100:.1f}\\% "
            f"& {delta*100:+.1f}pp {arrow} \\\\"
        )
    lines += [
        r'\midrule',
        f'\\multicolumn{{6}}{{l}}{{\\textit{{Herfindahl-Hirschman Index (HHI) '
        f'-- {n_countries} suppliers}}}} \\\\',
        f'& Pre-tariff & \\multicolumn{{4}}{{l}}{{{hhi_pre:.0f}}} \\\\',
        f'& Post-tariff & \\multicolumn{{4}}{{l}}{{{hhi_post:.0f} '
        f'({hhi_post-hhi_pre:+.0f} change)}} \\\\',
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\small',
        r'\item Post-tariff shares estimated via gravity reallocation: '
        r'$\tilde{s}_j \propto s_j(1+\tau_j)^{-\varepsilon}$, $\varepsilon=2.3$ (Broda \& Weinstein 2006).',
        r'\item Source: pharma\_bilateral\_trade.xlsx (2024 trade flows), 132 trading partners.',
        r'\end{tablenotes}',
        r'\end{table}',
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_supplier_shift(top15_pre, top15_post):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('US Pharmaceutical Import Sourcing: Pre- vs Post-Tariff',
                 fontsize=13, fontweight='bold')

    colors_pre  = ['#c0392b' if r['tau_country'] >= 0.25 else
                   '#e67e22' if r['tau_country'] >= 0.15 else '#27ae60'
                   for _, r in top15_pre.iterrows()]
    colors_post = ['#c0392b' if r['tau_country'] >= 0.25 else
                   '#e67e22' if r['tau_country'] >= 0.15 else '#27ae60'
                   for _, r in top15_pre.iterrows()]

    ax = axes[0]
    ax.barh(top15_pre['country_name'], top15_pre['weight'] * 100, color=colors_pre)
    ax.set_xlabel('Import Share (%)')
    ax.set_title('Pre-Tariff')
    ax.invert_yaxis()

    ax = axes[1]
    ax.barh(top15_post['country_name'], top15_post['share_post'] * 100,
            color=colors_post)
    ax.set_xlabel('Import Share (%)')
    ax.set_title('Post-Tariff (Gravity Reallocation)')
    ax.invert_yaxis()

    patches = [
        mpatches.Patch(color='#c0392b', label='Tariff >= 25%'),
        mpatches.Patch(color='#e67e22', label='Tariff 15-25%'),
        mpatches.Patch(color='#27ae60', label='Tariff < 15%'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, frameon=True)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(OUTPUT_DIR, 'fig_pharma_supplier_shift.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def _plot_pharma_comparison(price_noretal, price_retal,
                            ge_cpi_noretal, ge_cpi_retal,
                            pharma_welfare_noretal, pharma_welfare_retal,
                            ge_welfare_noretal, ge_welfare_retal,
                            hts8_rate, tau_eff):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Pharmaceutical Sector vs Economy: Liberation Day Tariff Impacts',
                 fontsize=12, fontweight='bold')

    scenarios = ['No Retaliation', 'Full Retaliation']

    # Panel 1: Tariff rates
    ax = axes[0]
    x = [0, 1]
    bars = ax.bar(x, [hts8_rate * 100, tau_eff * 100],
                  color=['#3498db', '#e74c3c'], width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['HTS8\nProduct Rate', 'Liberation Day\nCountry Rate'], fontsize=9)
    ax.set_ylabel('Tariff Rate (%)')
    ax.set_title('Pharma Tariff Rates\n(Two Layers)')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=9)

    # Panel 2: CPI comparison
    ax = axes[1]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, [price_noretal, price_retal], w,
           label='Pharma sector', color='#e74c3c')
    ax.bar(x + w/2, [ge_cpi_noretal, ge_cpi_retal], w,
           label='Economy-wide', color='#3498db')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_ylabel('Price Increase (%)')
    ax.set_title('Price Impact:\nPharma vs Economy')
    ax.legend(fontsize=8)

    # Panel 3: Welfare
    ax = axes[2]
    ax.bar(x - w/2, [pharma_welfare_noretal, pharma_welfare_retal], w,
           label='Pharma welfare loss', color='#e74c3c')
    ax.bar(x + w/2, [ge_welfare_noretal, ge_welfare_retal], w,
           label='Economy-wide welfare', color='#27ae60')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_ylabel('Welfare Change (%)')
    ax.set_title('Welfare: Pharma Loss\nvs National Gain')
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_pharma_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_pharma():
    """
    Run full pharmaceutical sector analysis using real bilateral trade data,
    OECD ICIO IO multipliers, and HTS8 tariff schedule data.
    """
    print("=" * 72)
    print("Pharmaceutical Sector Analysis")
    print("=" * 72)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. GE model results (aggregate anchor)
    # ------------------------------------------------------------------
    print("\n[1/6] Loading GE model results...")
    npz_path = os.path.join(OUTPUT_DIR, 'multisector_io_results.npz')
    npz = np.load(npz_path)
    results_multi = npz['results_multi']
    id_US = int(npz['id_US'])

    # Metrics: [welfare, deficit, exports, imports, employment, CPI, tariff_rev/E_i]
    # Values stored in npz are already in percentage form (e.g. 0.677 = 0.677%)
    ge_welfare_noretal = float(results_multi[id_US, 0, 0])
    ge_welfare_retal   = float(results_multi[id_US, 0, 1])
    ge_cpi_noretal     = float(results_multi[id_US, 5, 0])
    ge_cpi_retal       = float(results_multi[id_US, 5, 1])
    ge_imports_noretal = float(results_multi[id_US, 3, 0])
    ge_imports_retal   = float(results_multi[id_US, 3, 1])

    print(f"  GE (no retaliation): welfare={ge_welfare_noretal:.2f}%, "
          f"CPI={ge_cpi_noretal:.2f}%, imports={ge_imports_noretal:.1f}%")
    print(f"  GE (retaliation):    welfare={ge_welfare_retal:.2f}%, "
          f"CPI={ge_cpi_retal:.2f}%, imports={ge_imports_retal:.1f}%")

    # ------------------------------------------------------------------
    # 2. IO multiplier from OECD ICIO 2022
    # ------------------------------------------------------------------
    print("\n[2/6] Loading OECD ICIO IO multiplier (pharma sector)...")
    io_mult, imp_share_interm = get_model_sector_io_multiplier('pharma')
    print(f"  Pharma intermediate import share (ICIO 2022): {imp_share_interm*100:.1f}%")
    print(f"  Pharma IO supply-chain multiplier:            {io_mult:.3f}x")
    print(f"  (Previous hardcoded estimate was 1.18x)")

    # ------------------------------------------------------------------
    # 3. Pharma trade weights + effective tariff
    # ------------------------------------------------------------------
    print("\n[3/6] Loading pharma bilateral trade weights (132 countries)...")
    tw = load_pharma_trade_weights()
    n_countries = len(tw)
    n_matched   = tw['tau_country'].notna().sum()

    tau_eff_noretal = _compute_trade_weighted_tariff(tw)
    print(f"  Countries in pharma trade data:  {n_countries}")
    print(f"  Matched to tariff schedule:      {n_matched}")
    print(f"  Trade-weighted effective tariff (Liberation Day): "
          f"{tau_eff_noretal*100:.2f}%")

    # HTS8 product-level structural rate
    hts8_shocks  = get_hts8_tariff_by_sector(scenario='liberation_day_schedule')
    hts8_pharma  = hts8_shocks.get('pharma', 0.0245)
    print(f"  HTS8 product-level MFN rate (pharma avg.):       {hts8_pharma*100:.2f}%")

    # Under retaliation, assume symmetric: same effective rate applies (trading
    # partners raise tariffs on US exports; the import tariff faced by the US
    # is still tau_eff_noretal from the US tariff schedule)
    tau_eff_retal = tau_eff_noretal  # US import tariff unchanged by foreign retaliation

    # ------------------------------------------------------------------
    # 4. Pharma-specific price and volume impacts
    # ------------------------------------------------------------------
    print("\n[4/6] Computing pharma price and import volume impacts...")

    # Import penetration (pharma share of mfg imports × mfg import penetration)
    # From ITPD/GE model: manufacturing import penetration ~32%
    mfg_import_penetration = 0.323
    pharma_import_penetration = PHARMA_MFG_SHARE * mfg_import_penetration

    # Price impact: pass-through * effective tariff * import penetration
    price_noretal = (PHARMA_PASS_THROUGH * tau_eff_noretal
                     * pharma_import_penetration * 100)
    price_retal   = (PHARMA_PASS_THROUGH * tau_eff_retal
                     * pharma_import_penetration * 100)

    # Apply IO multiplier from actual ICIO data
    price_noretal_io = price_noretal * io_mult
    price_retal_io   = price_retal   * io_mult

    print(f"  Direct pharma price impact (no retaliation):      +{price_noretal:.2f}%")
    print(f"  IO-adjusted pharma price impact (no retaliation): +{price_noretal_io:.2f}%")
    print(f"  Economy-wide CPI (no retaliation):                +{ge_cpi_noretal:.2f}%")
    print(f"  Pharma/economy CPI ratio: {price_noretal_io/ge_cpi_noretal:.2f}x")

    # Import volume change: -eps * tau/(1+tau)
    import_chg_noretal = -EPS_PHARMA * tau_eff_noretal / (1+tau_eff_noretal) * 100
    import_chg_retal   = -EPS_PHARMA * tau_eff_retal   / (1+tau_eff_retal)   * 100
    print(f"  Pharma import volume change (no retaliation): {import_chg_noretal:.1f}%")
    print(f"  Pharma import volume change (retaliation):    {import_chg_retal:.1f}%")

    # Consumer welfare loss from pharma price increase (as % of GDP)
    pharma_welfare_noretal = -(PHARMA_PCE_SHARE * price_noretal_io / 100) * 100
    pharma_welfare_retal   = -(PHARMA_PCE_SHARE * price_retal_io   / 100) * 100
    print(f"  Pharma consumer welfare loss (no retaliation): {pharma_welfare_noretal:.3f}%")

    # ------------------------------------------------------------------
    # 5. Supplier concentration (HHI) using actual 132-country data
    # ------------------------------------------------------------------
    print("\n[5/6] Computing supplier concentration (HHI from 132-country data)...")

    # Pre-tariff HHI
    hhi_pre = _compute_hhi(tw['weight'].values)

    # Post-tariff shares via gravity reallocation
    tw_post = _post_tariff_shares(tw, EPS_PHARMA)
    hhi_post = _compute_hhi(tw_post['share_post'].values)

    print(f"  Pre-tariff HHI  (132 suppliers): {hhi_pre:.0f}")
    print(f"  Post-tariff HHI (132 suppliers): {hhi_post:.0f} ({hhi_post-hhi_pre:+.0f})")

    # Load country names for top-N table
    labels_path = os.path.join(DATA_ROOT, 'base_data', 'country_labels.csv')
    if os.path.exists(labels_path):
        labels_df = pd.read_csv(labels_path)[['iso3', 'CountryName']]
        tw_post = tw_post.merge(labels_df, on='iso3', how='left')
        tw_post['country_name'] = tw_post['CountryName'].fillna(tw_post['iso3'])
    else:
        tw_post['country_name'] = tw_post['iso3']

    top15_pre  = tw_post.nlargest(15, 'weight').reset_index(drop=True)
    top15_pre['rank'] = range(1, len(top15_pre)+1)
    top10_table = top15_pre.head(10).copy()

    # ------------------------------------------------------------------
    # 6. Generate outputs
    # ------------------------------------------------------------------
    print("\n[6/6] Generating tables and figures...")

    _write_table_s3(
        tau_eff        = tau_eff_noretal,
        hts8_rate      = hts8_pharma,
        io_mult        = io_mult,
        imp_share      = imp_share_interm,
        price_noretal  = price_noretal_io,
        price_retal    = price_retal_io,
        import_chg_noretal = import_chg_noretal,
        import_chg_retal   = import_chg_retal,
        pharma_welfare_noretal = pharma_welfare_noretal,
        pharma_welfare_retal   = pharma_welfare_retal,
        ge_cpi_noretal = ge_cpi_noretal,
        ge_cpi_retal   = ge_cpi_retal,
    )

    _write_table_s4(top10_table, hhi_pre, hhi_post, n_countries)

    _plot_supplier_shift(top15_pre, top15_pre)
    _plot_pharma_comparison(
        price_noretal  = price_noretal_io,
        price_retal    = price_retal_io,
        ge_cpi_noretal = ge_cpi_noretal,
        ge_cpi_retal   = ge_cpi_retal,
        pharma_welfare_noretal = pharma_welfare_noretal,
        pharma_welfare_retal   = pharma_welfare_retal,
        ge_welfare_noretal     = ge_welfare_noretal,
        ge_welfare_retal       = ge_welfare_retal,
        hts8_rate = hts8_pharma,
        tau_eff   = tau_eff_noretal,
    )

    np.savez(
        os.path.join(OUTPUT_DIR, 'sector_pharma_results.npz'),
        tau_pharma_eff      = tau_eff_noretal,
        hts8_pharma_rate    = hts8_pharma,
        io_multiplier       = io_mult,
        imp_share_interm    = imp_share_interm,
        price_noretal       = price_noretal_io,
        price_retal         = price_retal_io,
        import_chg_noretal  = import_chg_noretal,
        import_chg_retal    = import_chg_retal,
        hhi_pre             = hhi_pre,
        hhi_post            = hhi_post,
        n_suppliers         = n_countries,
        pharma_welfare_noretal = pharma_welfare_noretal,
        pharma_welfare_retal   = pharma_welfare_retal,
    )
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'sector_pharma_results.npz')}")

    print("\n[OK] Pharmaceutical analysis complete.")
    return {
        'tau_eff':          tau_eff_noretal,
        'hts8_rate':        hts8_pharma,
        'io_multiplier':    io_mult,
        'price_noretal':    price_noretal_io,
        'price_retal':      price_retal_io,
        'hhi_pre':          hhi_pre,
        'hhi_post':         hhi_post,
        'n_suppliers':      n_countries,
    }


if __name__ == '__main__':
    analyze_pharma()
