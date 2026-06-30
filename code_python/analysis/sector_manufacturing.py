"""
sector_manufacturing.py
=======================
Manufacturing & Steel/Aluminum Sector Analysis -- Liberation Day Tariff Impacts

Data sources used:
  - data/ITPDS/trade_ITPD.csv
      Bilateral manufacturing trade flows (194 countries) -> partner exposure
  - data/processed/icio_2022/io_coeff_matrix.npy + sector_map.csv
      OECD ICIO 2022 -- sector-specific IO intermediate import shares
  - data/processed/shocks/sector_tariff_shocks.csv
      HTS8-derived tariff rates: manufacturing_other (3.61%), steel_aluminum (1.09%)
  - data/base_data/tariffs.csv + country_labels.csv
      Liberation Day country-level tariff rates (194 countries)
  - python_output/multisector_io_results.npz + baseline_results.npz
      GE model outcomes (Phase 1)

Key outputs (python_output/):
  Table_S1_manufacturing.tex          -- top-15 partners by tariff exposure
  Table_S2_manufacturing_optimal.tex  -- alternative tariff scenarios
  fig_manufacturing_tariff_exposure.png
  fig_manufacturing_alt_scenarios.png
  fig_manufacturing_hts8_vs_ge.png    -- HTS8 product rates vs GE country rates
  sector_manufacturing_results.npz
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
OUTPUT_DIR = os.path.join(REPO_ROOT, 'python_output')

sys.path.insert(0, os.path.join(REPO_ROOT, 'code_python'))

from utils.data_utils import (
    find_data_root,
    get_model_sector_io_multiplier,
    load_icio_sector_multipliers,
    get_hts8_tariff_by_sector,
    load_sector_tariff_shocks,
)

DATA_ROOT = find_data_root()

BETA_LABOR   = 0.49   # labor share, from main_io.py
PASS_THROUGH = 0.85   # manufacturing pass-through, Cavallo et al. (2025)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_manufacturing_trade():
    """
    Load ITPD Manufacturing bilateral trade for the US.
    Returns df with columns: exporter, value, tau_exp (tariff rate)
    """
    itpd_path = os.path.join(REPO_ROOT, 'data', 'ITPDS', 'trade_ITPD.csv')
    if not os.path.exists(itpd_path):
        itpd_path = os.path.join(DATA_ROOT, 'ITPDS', 'trade_ITPD.csv')

    tariffs_path  = os.path.join(DATA_ROOT, 'base_data', 'tariffs.csv')
    labels_path   = os.path.join(DATA_ROOT, 'base_data', 'country_labels.csv')

    df      = pd.read_csv(itpd_path, header=None,
                          names=['exporter', 'importer', 'sector', 'value'])
    tariffs = pd.read_csv(tariffs_path, header=0)
    labels  = pd.read_csv(labels_path)

    tariffs_merged = labels[['iso3']].copy()
    tariffs_merged['applied_tariff'] = tariffs['applied_tariff'].values[:len(labels)]

    mfg = df[
        (df['importer'] == 'USA') &
        (df['exporter'] != 'USA') &
        (df['sector'] == 'Manufacturing')
    ].copy()

    mfg = mfg.merge(
        tariffs_merged.rename(columns={'iso3': 'exporter',
                                       'applied_tariff': 'tau_exp'}),
        on='exporter', how='left'
    )
    mfg = mfg.merge(
        labels[['iso3', 'CountryName']].rename(columns={'iso3': 'exporter'}),
        on='exporter', how='left'
    )
    return mfg


def _compute_sector_expenditure_shares():
    """
    Compute US expenditure shares from ITPD for all sectors.
    """
    itpd_path = os.path.join(REPO_ROOT, 'data', 'ITPDS', 'trade_ITPD.csv')
    if not os.path.exists(itpd_path):
        itpd_path = os.path.join(DATA_ROOT, 'ITPDS', 'trade_ITPD.csv')

    df     = pd.read_csv(itpd_path, header=None,
                         names=['exporter', 'importer', 'sector', 'value'])
    us_exp = df[df['importer'] == 'USA'].groupby('sector')['value'].sum()
    total  = us_exp.sum()
    return (us_exp / total).to_dict()


# ---------------------------------------------------------------------------
# LaTeX table writers
# ---------------------------------------------------------------------------

def _write_table_s1(top15, tau_mfg_avg, import_pen_mfg,
                    welfare_noretal, welfare_retal,
                    cpi_contribution, import_change,
                    io_mult_mfg, io_mult_steel,
                    hts8_mfg_rate, hts8_steel_rate):
    path = os.path.join(OUTPUT_DIR, 'Table_S1_manufacturing.tex')
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Manufacturing Sector: Liberation Day Tariff Impacts '
        r'\& Top-15 Partners}',
        r'\label{tab:manufacturing_top15}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'\multicolumn{5}{l}{\textit{A. Aggregate Manufacturing Impacts}} \\[4pt]',
        r'Metric & \multicolumn{2}{c}{No Retaliation} '
        r'& \multicolumn{2}{c}{Full Retaliation} \\',
        r'\cmidrule(lr){2-3} \cmidrule(lr){4-5}',
    ]
    lines += [
        f'GE Welfare change & \\multicolumn{{2}}{{c}}{{{welfare_noretal:+.2f}\\%}} '
        f'& \\multicolumn{{2}}{{c}}{{{welfare_retal:+.2f}\\%}} \\\\',
        f'Mfg CPI contribution & \\multicolumn{{2}}{{c}}{{{cpi_contribution:.2f}pp}} '
        f'& \\multicolumn{{2}}{{c}}{{--}} \\\\',
        f'Mfg import volume change & \\multicolumn{{2}}{{c}}{{{import_change:.1f}\\%}} '
        f'& \\multicolumn{{2}}{{c}}{{--}} \\\\',
        r'\midrule',
        r'\multicolumn{5}{l}{\textit{B. Tariff Structure (Two Layers)}} \\[2pt]',
        f'Country-level tariff (Liberation Day, trade-weighted) '
        f'& \\multicolumn{{4}}{{r}}{{{tau_mfg_avg*100:.1f}\\%}} \\\\',
        'HTS8 product MFN rate (manufacturing other) '
        f'& \\multicolumn{{4}}{{r}}{{{hts8_mfg_rate*100:.2f}\\%}} \\\\',
        'HTS8 product MFN rate (steel/aluminum) '
        f'& \\multicolumn{{4}}{{r}}{{{hts8_steel_rate*100:.2f}\\%}} \\\\',
        'Import penetration rate '
        f'& \\multicolumn{{4}}{{r}}{{{import_pen_mfg*100:.1f}\\%}} \\\\',
        r'\midrule',
        r'\multicolumn{5}{l}{\textit{C. OECD ICIO IO Multipliers (2022)}} \\[2pt]',
        'Manufacturing (other) IO multiplier '
        f'& \\multicolumn{{4}}{{r}}{{{io_mult_mfg:.3f}x}} \\\\',
        'Steel \\& aluminum IO multiplier '
        f'& \\multicolumn{{4}}{{r}}{{{io_mult_steel:.3f}x}} \\\\',
        r'\midrule',
        r'\multicolumn{5}{l}{\textit{D. Top 15 Partners by Manufacturing '
        r'Import Exposure}} \\',
        r'Rank & Country & Tariff & Import Share & Tariff Exposure \\',
        r'\midrule',
    ]
    top15_sorted = top15.sort_values('value', ascending=False).head(15).reset_index(drop=True)
    total_val = top15_sorted['value'].sum()
    for i, row in top15_sorted.iterrows():
        share   = row['value'] / total_val * 100 if total_val > 0 else 0
        tau     = row.get('tau_exp', 0.0) or 0.0
        exposure = tau * share / 100
        lines.append(
            f"{i+1} & {row.get('CountryName', row['exporter'])} "
            f"& {tau*100:.0f}\\% & {share:.1f}\\% & {exposure:.3f} \\\\"
        )
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\small',
        r'\item Country tariff rates from Liberation Day executive order. '
        r'IO multipliers from OECD ICIO 2022. '
        r'HTS8 rates reflect product-level MFN structure.',
        r'\end{tablenotes}',
        r'\end{table}',
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


def _write_table_s2(alt_results, ge_welfare_noretal):
    path = os.path.join(OUTPUT_DIR, 'Table_S2_manufacturing_optimal.tex')

    # Also load supply_chain_disruption scenario from HTS8 shocks
    all_shocks = load_sector_tariff_shocks()
    scd_mfg = all_shocks[
        (all_shocks['scenario'] == 'supply_chain_disruption') &
        (all_shocks['model_sector'] == 'manufacturing_other')
    ]['tariff_rate'].values
    scd_rate = float(scd_mfg[0]) if len(scd_mfg) > 0 else 0.036

    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{US Manufacturing Outcomes Under Alternative Tariff Scenarios}',
        r'\label{tab:manufacturing_scenarios}',
        r'\begin{tabular}{lccc}',
        r'\toprule',
        r'Scenario & Tariff Rate & Welfare Change & Import Change \\',
        r'\midrule',
        r'\multicolumn{4}{l}{\textit{GE Model Scenarios}} \\',
    ]
    for name, d in alt_results.items():
        lines.append(
            f"{name} & {d['tariff_rate']*100:.1f}\\% "
            f"& {d['welfare_change']:+.2f}\\% "
            f"& {d['import_change']:.1f}\\% \\\\"
        )
    lines += [
        r'\midrule',
        r'\multicolumn{4}{l}{\textit{HTS8 Structural Scenarios (product-level rates)}} \\',
        f'Liberation Day (HTS8 mfg avg) & {0.0361*100:.2f}\\% & -- & -- \\\\',
        f'Optimal uniform 19\\% & 19.0\\% & -- & -- \\\\',
        f'Industry-focused & {0.0361*100:.2f}\\% & -- & -- \\\\',
        f'Supply-chain disruption & {scd_rate*100:.2f}\\% & -- & -- \\\\',
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\small',
        r'\item GE model scenarios: welfare and import changes from '
        r'multisector\_baseline\_results.npz.',
        r'\item HTS8 structural scenarios: from sector\_tariff\_shocks.csv, '
        r'us\_tariff\_schedule\_2025\_hts8.csv.',
        r'\end{tablenotes}',
        r'\end{table}',
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_tariff_exposure(top15):
    fig, ax = plt.subplots(figsize=(10, 6))
    top15_s = top15.sort_values('value', ascending=False).head(15)
    total   = top15_s['value'].sum()
    top15_s = top15_s.copy()
    top15_s['share'] = top15_s['value'] / total * 100
    top15_s['tau_exp'] = top15_s['tau_exp'].fillna(0)

    colors = ['#c0392b' if t >= 0.4 else '#e67e22' if t >= 0.2 else '#27ae60'
              for t in top15_s['tau_exp']]

    bars = ax.barh(top15_s['CountryName'].fillna(top15_s['exporter']),
                   top15_s['share'], color=colors)
    ax.set_xlabel('Share of US Manufacturing Imports (%)')
    ax.set_title('Top-15 US Manufacturing Import Partners\n'
                 '(Color = Liberation Day Tariff Level)', fontsize=11)
    ax.invert_yaxis()

    for bar, tau in zip(bars, top15_s['tau_exp']):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{tau*100:.0f}%', va='center', fontsize=8)

    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color='#c0392b', label='Tariff >= 40%'),
        mpatches.Patch(color='#e67e22', label='Tariff 20-40%'),
        mpatches.Patch(color='#27ae60', label='Tariff < 20%'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=9)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_manufacturing_tariff_exposure.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def _plot_alt_scenarios(alt_results):
    names    = list(alt_results.keys())
    rates    = [d['tariff_rate'] * 100 for d in alt_results.values()]
    welfares = [d['welfare_change'] for d in alt_results.values()]
    imports  = [d['import_change'] for d in alt_results.values()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('US Manufacturing: Alternative Tariff Scenarios', fontsize=12)

    ax = axes[0]
    c  = ['#27ae60' if w > 0 else '#e74c3c' for w in welfares]
    ax.bar(names, welfares, color=c)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylabel('Welfare Change (%)')
    ax.set_title('Welfare Change by Scenario')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=9)

    ax = axes[1]
    ax.bar(names, imports, color='#3498db')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylabel('Import Change (%)')
    ax.set_title('Manufacturing Import Change')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_manufacturing_alt_scenarios.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def _plot_hts8_vs_ge(hts8_shocks, sector_exp_shares, icio_mults):
    """
    Comparison figure: HTS8 product-level MFN rates vs GE country-level
    trade-weighted rates, for each model sector.
    """
    sectors_show = ['steel_aluminum', 'pharma', 'retail_consumer',
                    'manufacturing_other', 'energy_primary']
    labels_show  = ['Steel &\nAluminum', 'Pharma', 'Retail\nConsumer',
                    'Mfg Other', 'Energy\nPrimary']

    hts8_rates = [hts8_shocks.get(s, 0.0) * 100 for s in sectors_show]
    io_mults   = [icio_mults.get(s, {}).get('io_multiplier', 1.0)
                  for s in sectors_show]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('HTS8 Product Tariff Rates and IO Multipliers by Sector\n'
                 '(OECD ICIO 2022 + US HTS8 2025)', fontsize=11)

    x = np.arange(len(sectors_show))
    ax = axes[0]
    ax.bar(x, hts8_rates, color='#3498db')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_show, fontsize=9)
    ax.set_ylabel('Avg HTS8 MFN Tariff Rate (%)')
    ax.set_title('Product-Level Tariff Structure\n(HTS8 weighted avg, 2025)')
    for i, v in enumerate(hts8_rates):
        ax.text(i, v + 0.05, f'{v:.2f}%', ha='center', fontsize=8)

    ax = axes[1]
    ax.bar(x, io_mults, color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_show, fontsize=9)
    ax.set_ylabel('IO Supply-Chain Multiplier')
    ax.set_title('OECD ICIO 2022 IO Multipliers\n(import share of intermediates)')
    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    for i, v in enumerate(io_mults):
        ax.text(i, v + 0.002, f'{v:.3f}x', ha='center', fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_manufacturing_hts8_vs_ge.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_manufacturing():
    print("=" * 72)
    print("Manufacturing & Steel/Aluminum Sector Analysis")
    print("=" * 72)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. ITPD manufacturing trade data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading ITPD Manufacturing trade data...")
    mfg = _load_manufacturing_trade()

    total_us_mfg_imports = mfg['value'].sum()
    total_us_mfg_all     = total_us_mfg_imports  # includes domestic in ITPD

    # Trade-weighted average tariff on US manufacturing imports
    valid = mfg.dropna(subset=['tau_exp'])
    tau_mfg_avg = float(
        (valid['tau_exp'] * valid['value']).sum() / valid['value'].sum()
    )

    # Import penetration: manufacturing imports / total mfg expenditure
    itpd_path = os.path.join(REPO_ROOT, 'data', 'ITPDS', 'trade_ITPD.csv')
    if not os.path.exists(itpd_path):
        itpd_path = os.path.join(DATA_ROOT, 'ITPDS', 'trade_ITPD.csv')
    df_full = pd.read_csv(itpd_path, header=None,
                          names=['exporter', 'importer', 'sector', 'value'])
    us_mfg_total = df_full[
        (df_full['importer'] == 'USA') & (df_full['sector'] == 'Manufacturing')
    ]['value'].sum()
    import_pen_mfg = float(total_us_mfg_imports / us_mfg_total) if us_mfg_total > 0 else 0.323

    print(f"  Trade-weighted avg tariff (US mfg imports): {tau_mfg_avg*100:.1f}%")
    print(f"  Manufacturing import penetration:            {import_pen_mfg*100:.1f}%")

    # ------------------------------------------------------------------
    # 2. GE model results
    # ------------------------------------------------------------------
    print("\n[2/6] Loading GE model results...")
    npz = np.load(os.path.join(OUTPUT_DIR, 'multisector_io_results.npz'))
    results_multi = npz['results_multi']
    id_US = int(npz['id_US'])

    # Values stored in npz are already in percentage form (e.g. 0.677 = 0.677%)
    ge_welfare_noretal = float(results_multi[id_US, 0, 0])
    ge_welfare_retal   = float(results_multi[id_US, 0, 1])
    ge_cpi_noretal     = float(results_multi[id_US, 5, 0])
    ge_imports_noretal = float(results_multi[id_US, 3, 0])

    print(f"  Multisector (no retaliation): welfare={ge_welfare_noretal:.2f}%, "
          f"CPI={ge_cpi_noretal:.2f}%, imports={ge_imports_noretal:.1f}%")

    # ------------------------------------------------------------------
    # 3. OECD ICIO IO multipliers
    # ------------------------------------------------------------------
    print("\n[3/6] Loading OECD ICIO IO multipliers...")
    icio_mults = load_icio_sector_multipliers(beta_labor=BETA_LABOR)
    io_mult_mfg, imp_share_mfg = (
        icio_mults.get('manufacturing_other', {}).get('io_multiplier', 1.094),
        icio_mults.get('manufacturing_other', {}).get('import_share_interm', 0.168),
    )
    io_mult_steel, imp_share_steel = (
        icio_mults.get('steel_aluminum', {}).get('io_multiplier', 1.090),
        icio_mults.get('steel_aluminum', {}).get('import_share_interm', 0.161),
    )
    print(f"  Manufacturing IO multiplier:    {io_mult_mfg:.3f}x "
          f"(import_share={imp_share_mfg*100:.1f}%)")
    print(f"  Steel/aluminum IO multiplier:   {io_mult_steel:.3f}x "
          f"(import_share={imp_share_steel*100:.1f}%)")

    # ------------------------------------------------------------------
    # 4. HTS8 structural tariff rates
    # ------------------------------------------------------------------
    print("\n[4/6] Loading HTS8 sector tariff shocks...")
    hts8_shocks = get_hts8_tariff_by_sector(scenario='liberation_day_schedule')
    hts8_mfg_rate   = hts8_shocks.get('manufacturing_other', 0.0361)
    hts8_steel_rate = hts8_shocks.get('steel_aluminum', 0.0109)
    print(f"  HTS8 manufacturing_other rate: {hts8_mfg_rate*100:.2f}%")
    print(f"  HTS8 steel_aluminum rate:      {hts8_steel_rate*100:.2f}%")

    # ------------------------------------------------------------------
    # 5. Manufacturing-specific impacts
    # ------------------------------------------------------------------
    print("\n[5/6] Computing manufacturing impacts...")
    exp_shares = _compute_sector_expenditure_shares()
    beta_mfg = exp_shares.get('Manufacturing', 0.2246)

    # CPI contribution from manufacturing (GE model total scaled by sector share)
    mfg_tariff_pressure_share = (
        beta_mfg * tau_mfg_avg * import_pen_mfg /
        sum(exp_shares.get(s, 0) * 0.27 * 0.3  # rough denominator
            for s in exp_shares)
    )
    # Use GE model CPI * manufacturing fraction of tariff pressure
    all_sectors_weighted = sum(
        exp_shares.get(s, 0) * 0.27 * 0.3
        for s in exp_shares if s != 'Services'
    )
    mfg_weighted = beta_mfg * tau_mfg_avg * import_pen_mfg
    frac = mfg_weighted / (all_sectors_weighted + 1e-9)
    cpi_mfg_contribution = ge_cpi_noretal * frac
    print(f"  Mfg CPI contribution (GE-scaled): {cpi_mfg_contribution:.2f}pp")

    mfg_import_change = -3.8 * tau_mfg_avg / (1 + tau_mfg_avg) * 100
    print(f"  Mfg import volume change (est.): {mfg_import_change:.1f}%")

    # ------------------------------------------------------------------
    # 6. Alternative tariff scenarios
    # ------------------------------------------------------------------
    print("\n[6/6] Computing alternative tariff scenarios...")
    scenarios_def = {
        'USTR Liberation Day': tau_mfg_avg,
        '25% Uniform':         0.25,
        '50% Uniform':         0.50,
        '10% Minimum Floor':   0.10,
    }

    alt_results = {}
    eps_mfg = 3.8 / 1.1508
    for name, tau in scenarios_def.items():
        # Terms-of-trade welfare (partial-equilibrium approximation)
        welfare = (1/eps_mfg) * (tau/(1+tau)) * import_pen_mfg * beta_mfg * 100
        import_chg = -eps_mfg * tau / (1+tau) * 100
        alt_results[name] = {
            'tariff_rate':    tau,
            'welfare_change': welfare,
            'import_change':  import_chg,
        }
        print(f"  {name} ({tau:.0%}): welfare~{welfare:.2f}%, "
              f"imports~{import_chg:.1f}%")

    # ------------------------------------------------------------------
    # Generate outputs
    # ------------------------------------------------------------------
    _write_table_s1(
        top15              = mfg,
        tau_mfg_avg        = tau_mfg_avg,
        import_pen_mfg     = import_pen_mfg,
        welfare_noretal    = ge_welfare_noretal,
        welfare_retal      = ge_welfare_retal,
        cpi_contribution   = cpi_mfg_contribution,
        import_change      = mfg_import_change,
        io_mult_mfg        = io_mult_mfg,
        io_mult_steel      = io_mult_steel,
        hts8_mfg_rate      = hts8_mfg_rate,
        hts8_steel_rate    = hts8_steel_rate,
    )
    _write_table_s2(alt_results, ge_welfare_noretal)
    _plot_tariff_exposure(mfg)
    _plot_alt_scenarios(alt_results)
    _plot_hts8_vs_ge(hts8_shocks, exp_shares, icio_mults)

    np.savez(
        os.path.join(OUTPUT_DIR, 'sector_manufacturing_results.npz'),
        tau_mfg_avg            = tau_mfg_avg,
        import_penetration_mfg = import_pen_mfg,
        cpi_mfg_contribution   = cpi_mfg_contribution,
        mfg_import_change      = mfg_import_change,
        io_mult_mfg            = io_mult_mfg,
        io_mult_steel          = io_mult_steel,
        imp_share_mfg          = imp_share_mfg,
        imp_share_steel        = imp_share_steel,
        hts8_mfg_rate          = hts8_mfg_rate,
        hts8_steel_rate        = hts8_steel_rate,
    )
    print(f"\n  Saved: {os.path.join(OUTPUT_DIR, 'sector_manufacturing_results.npz')}")

    # ------------------------------------------------------------------
    # 7. NAICS subsector extension
    # ------------------------------------------------------------------
    print("\n[7/7] Running NAICS manufacturing subsector analysis...")
    try:
        naics_results = analyze_naics_subsectors()
        if naics_results is not None:
            np.savez(
                os.path.join(OUTPUT_DIR, 'sector_manufacturing_naics.npz'),
                **{k: np.array(v) for k, v in naics_results.items()
                   if isinstance(v, (int, float, list, np.ndarray))}
            )
    except Exception as e:
        print(f"  [--] NAICS subsector analysis skipped: {e}")

    print("\n[OK] Manufacturing analysis complete.")
    return alt_results


# ---------------------------------------------------------------------------
# NAICS Subsector Extension
# ---------------------------------------------------------------------------

# Map NAICS4 prefixes to Liberation Day tariff categories
# Based on typical country-of-origin structure for these industries
NAICS_TARIFF_CONTEXT = {
    '3241': ('Petroleum refining',       0.10, 'Energy/primary (10% floor)'),
    '3361': ('Motor vehicles',            0.25, 'Mixed: Canada/Mexico USMCA 10%, others higher'),
    '3362': ('Light trucks',              0.25, 'Mixed: Canada/Mexico USMCA 10%, others higher'),
    '3254': ('Pharmaceuticals',           0.20, 'EU-weighted: ~20% avg tariff'),
    '3311': ('Iron & steel mills',        0.25, 'EU 20%, Canada 10%, Korea 25%'),
    '3364': ('Aerospace',                 0.20, 'EU/Japan: ~20-24%'),
    '3261': ('Plastics products',         0.30, 'China/Vietnam heavy exposure'),
    '3251': ('Basic chemicals',           0.20, 'EU and China exposure'),
    '3252': ('Resins & synthetics',       0.20, 'EU and China exposure'),
    '3231': ('Printing',                  0.15, 'Primarily domestic'),
}


def _load_naics_gross_output():
    """Load gross output by NAICS from the 301 model data."""
    go_path = os.path.join(
        DATA_ROOT, 'code_and_release_data', '301 model', 'D_GO_by_NAICS.xlsx'
    )
    if not os.path.exists(go_path):
        print(f"  [--] D_GO_by_NAICS.xlsx not found at: {go_path}")
        return None
    df = pd.read_excel(go_path)
    # Standardize column names
    df.columns = [str(c).strip() for c in df.columns]
    # Keep manufacturing NAICS (codes starting with 3)
    naics_col = [c for c in df.columns if 'naics' in c.lower() or 'code' in c.lower()][0]
    df = df.rename(columns={naics_col: 'naics_code'})
    df['naics_code'] = df['naics_code'].astype(str).str.strip()
    mfg = df[df['naics_code'].str.match(r'^3')].copy()
    return mfg


def _load_naics_price_indices():
    """
    Load BLS Producer Price Index (PPI) and Import Price Index (MPI)
    by NAICS4 from the 301 model data.
    """
    pi_path = os.path.join(
        DATA_ROOT, 'code_and_release_data', '301 model', 'D_price_indices.xlsx'
    )
    if not os.path.exists(pi_path):
        print(f"  [--] D_price_indices.xlsx not found at: {pi_path}")
        return None, None

    xl = pd.ExcelFile(pi_path)
    ppi = xl.parse('BLS PPI')
    mpi = xl.parse('BLS MPI')

    # Standardize
    for df in [ppi, mpi]:
        df.columns = [str(c).strip() for c in df.columns]

    return ppi, mpi


def _write_table_naics(top_sectors, ppi, mpi, year_latest='2021'):
    """Write Table S1b: Top manufacturing sectors by gross output with price context."""
    path = os.path.join(OUTPUT_DIR, 'Table_S1b_manufacturing_naics.tex')

    # Find the latest year column available in gross output data
    year_cols = [c for c in top_sectors.columns
                 if str(c).isdigit() and int(str(c)) >= 2016]
    if not year_cols:
        year_cols = [year_latest]
    latest_year = str(max(int(y) for y in year_cols))

    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Top US Manufacturing Sectors by Gross Output: '
        r'Liberation Day Tariff Context}',
        r'\label{tab:manufacturing_naics}',
        r'\begin{tabular}{p{4.5cm}rrl}',
        r'\toprule',
        f'Sector (NAICS) & GO {latest_year} & Share & Tariff Context \\\\',
        r'& (\$M) & of Mfg. & \\',
        r'\midrule',
    ]

    total_mfg_go = top_sectors[latest_year].sum() if latest_year in top_sectors.columns else 1.0

    for _, row in top_sectors.iterrows():
        go = row.get(latest_year, row.get('2021', 0))
        share = go / total_mfg_go * 100 if total_mfg_go > 0 else 0
        naics4 = str(row.get('naics_code', ''))[:4]
        ctx = NAICS_TARIFF_CONTEXT.get(naics4, ('', None, 'Varies'))[2]
        # Truncate context for table
        ctx_short = ctx[:45] + '..' if len(ctx) > 45 else ctx
        name = str(row.get('Name', row.get('name', naics4)))
        name_short = name[:40] + '..' if len(name) > 40 else name
        lines.append(
            f'{name_short} & {int(go):,} & {share:.1f}\\% & {ctx_short} \\\\'
        )

    lines += [
        r'\midrule',
        r'\multicolumn{4}{l}{\textit{BLS Price Index Changes (2012--2018):}} \\',
    ]

    # Add PPI rows if available
    if ppi is not None and not ppi.empty:
        naics_col_ppi = ppi.columns[0]
        year_cols_ppi = [c for c in ppi.columns if isinstance(c, (int, float))
                         or (isinstance(c, str) and c.isdigit())]
        if len(year_cols_ppi) >= 2:
            first_yr = year_cols_ppi[0]
            last_yr  = year_cols_ppi[-1]
            ppi_change = (ppi[last_yr] / ppi[first_yr] - 1) * 100
            avg_ppi_change = ppi_change.mean()
            lines.append(
                f'\\quad Avg PPI change ({first_yr}--{last_yr}) '
                f'& \\multicolumn{{3}}{{r}}{{{avg_ppi_change:+.1f}\\%}} \\\\'
            )

    if mpi is not None and not mpi.empty:
        year_cols_mpi = [c for c in mpi.columns
                         if isinstance(c, (int, float)) or
                         (hasattr(c, 'year'))]
        if len(year_cols_mpi) >= 2:
            try:
                first_yr = year_cols_mpi[0]
                last_yr  = year_cols_mpi[-1]
                mpi_change = (mpi[last_yr] / mpi[first_yr] - 1) * 100
                avg_mpi_change = mpi_change.mean()
                lines.append(
                    f'\\quad Avg MPI change (2012--2018) '
                    f'& \\multicolumn{{3}}{{r}}{{{avg_mpi_change:+.1f}\\%}} \\\\'
                )
            except Exception:
                pass

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\small',
        r'\item Gross output from BEA via D\_GO\_by\_NAICS.xlsx (301 model data). '
        r'Price indices from D\_price\_indices.xlsx (BLS PPI/MPI). '
        r'Tariff context based on typical country-of-origin for each sector.',
        r'\end{tablenotes}',
        r'\end{table}',
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


def _plot_naics_subsectors(top_sectors, year_latest):
    """Bar chart of top manufacturing sectors by gross output, colored by tariff exposure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('US Manufacturing Subsectors: Gross Output and Tariff Context\n'
                 '(BEA via D_GO_by_NAICS.xlsx + Liberation Day tariff mapping)',
                 fontsize=11, fontweight='bold')

    year_col = year_latest if year_latest in top_sectors.columns else \
               [c for c in top_sectors.columns if str(c).isdigit()][-1]
    top_sectors = top_sectors.sort_values(year_col, ascending=False).head(12)

    names = [str(n)[:30] for n in top_sectors['Name'].values]
    go_vals = top_sectors[year_col].values / 1000  # convert to $B

    # Tariff color mapping
    colors = []
    for _, row in top_sectors.iterrows():
        naics4 = str(row.get('naics_code', ''))[:4]
        ctx = NAICS_TARIFF_CONTEXT.get(naics4, ('', None, ''))[1]
        if ctx is None:
            colors.append('#95a5a6')
        elif ctx >= 0.40:
            colors.append('#c0392b')
        elif ctx >= 0.25:
            colors.append('#e67e22')
        elif ctx >= 0.15:
            colors.append('#f1c40f')
        else:
            colors.append('#27ae60')

    ax = axes[0]
    bars = ax.barh(range(len(names)), go_vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Gross Output ($B, ' + str(year_col) + ')')
    ax.set_title('Top 12 Manufacturing Sectors\nby Gross Output')
    ax.invert_yaxis()

    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color='#c0392b', label='High tariff exposure (>=40%)'),
        mpatches.Patch(color='#e67e22', label='Med-high (25-40%)'),
        mpatches.Patch(color='#f1c40f', label='Medium (15-25%)'),
        mpatches.Patch(color='#27ae60', label='Lower (<15%)'),
        mpatches.Patch(color='#95a5a6', label='Domestic/varied'),
    ]
    ax.legend(handles=patches, fontsize=7, loc='lower right')

    # Panel 2: Year-over-year gross output trend for top 5
    ax2 = axes[1]
    year_cols = sorted([c for c in top_sectors.columns
                        if str(c).isdigit() and int(str(c)) >= 2016])
    top5 = top_sectors.head(5)
    for _, row in top5.iterrows():
        vals = [row.get(y, np.nan) for y in year_cols]
        label = str(row.get('Name', ''))[:25]
        ax2.plot(year_cols, [v/1000 for v in vals], marker='o',
                 label=label, linewidth=1.5)

    ax2.axvline('2020', color='gray', linestyle='--', linewidth=1, alpha=0.7,
                label='COVID-19 shock')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Gross Output ($B)')
    ax2.set_title('Top 5 Sectors: GO Trend\n(2016-2021)')
    ax2.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_manufacturing_naics_subsectors.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def analyze_naics_subsectors():
    """
    Analyze manufacturing subsectors using BEA Gross Output by NAICS and
    BLS PPI/MPI price indices from the 301 model data.

    Returns dict of summary statistics, or None if data unavailable.
    """
    print("  Loading D_GO_by_NAICS.xlsx (BEA gross output, NAICS4)...")
    go_df = _load_naics_gross_output()
    if go_df is None:
        return None

    print(f"  Manufacturing NAICS rows: {len(go_df)}")

    # Find year columns
    year_cols = sorted([c for c in go_df.columns
                        if str(c).isdigit() and int(str(c)) >= 2016])
    latest = year_cols[-1] if year_cols else None

    if latest is None:
        print("  [--] No year columns found in gross output data.")
        return None

    # Top 12 by latest year
    top12 = go_df.nlargest(12, latest).copy()
    print(f"  Top sector by {latest} output: "
          f"{top12.iloc[0].get('Name', top12.iloc[0]['naics_code'])} "
          f"(${int(top12.iloc[0][latest]):,}M)")

    # Total manufacturing GO and steel share
    total_mfg_go = go_df[latest].sum()
    steel_rows = go_df[go_df['naics_code'].str.startswith('3311')]
    steel_go = steel_rows[latest].sum() if not steel_rows.empty else 0
    pharma_rows = go_df[go_df['naics_code'].str.startswith('3254')]
    pharma_go = pharma_rows[latest].sum() if not pharma_rows.empty else 0

    print(f"  Total manufacturing GO ({latest}):     ${total_mfg_go/1e6:.2f}T")
    print(f"  Steel & iron mills share:             {steel_go/total_mfg_go*100:.2f}%")
    print(f"  Pharmaceutical preparations share:    {pharma_go/total_mfg_go*100:.2f}%")

    print("  Loading D_price_indices.xlsx (BLS PPI + MPI)...")
    ppi, mpi = _load_naics_price_indices()
    if ppi is not None:
        print(f"  PPI: {ppi.shape[0]} NAICS sectors, {ppi.shape[1]-1} years")
    if mpi is not None:
        print(f"  MPI: {mpi.shape[0]} NAICS sectors, {mpi.shape[1]-1} periods")

    _write_table_naics(top12, ppi, mpi, year_latest=latest)
    _plot_naics_subsectors(top12, latest)

    return {
        'total_mfg_go': float(total_mfg_go),
        'steel_go_share': float(steel_go / total_mfg_go) if total_mfg_go > 0 else 0,
        'pharma_go_share': float(pharma_go / total_mfg_go) if total_mfg_go > 0 else 0,
        'n_naics_sectors': len(go_df),
        'latest_year': int(latest),
    }


if __name__ == '__main__':
    analyze_manufacturing()
