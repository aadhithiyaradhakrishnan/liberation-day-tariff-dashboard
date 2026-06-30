"""
pharma1_sourcing_analysis.py
=============================
Pharmaceutical Sourcing Patterns -- Deep Analysis
Liberation Day Tariff Replication

Objective: To evaluate how tariffs reshape U.S. pharmaceutical sourcing
           patterns by analyzing changes in supplier shares after tariff
           exposure.

Data sources (pharma1 files)
-----------------------------
python_output/pharma1_country_exposure_2024.csv     -- 97 countries, pre/post shares
python_output/pharma1_objective2_sourcing_shifts_2025.csv -- 25 countries, 2025 shifts
python_output/pharma1_hts_exposure_2024.csv         -- HTS code breakdown
python_output/pharma1_supply_chain_risk_2024.csv    -- risk tier classification
python_output/pharma1_objective1_dependence_top_suppliers.csv -- rank changes 2024->2025

Outputs (python_output/)
-------------------------
fig_p1_top15_pre_post.png          -- pre vs post-tariff supplier shares
fig_p1_gainers_losers.png          -- who gains and who loses share
fig_p1_tariff_tier_reallocation.png -- reallocation by tariff band
fig_p1_hts_breakdown.png           -- import value by HS code
fig_p1_rank_changes.png            -- 2024 vs 2025 rank comparison
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


# ============================================================
# Load all pharma1 data
# ============================================================

def load_data():
    d = {}

    d['exposure'] = pd.read_csv(
        os.path.join(OUTPUT_DIR, 'pharma1_country_exposure_2024.csv'))

    d['shifts'] = pd.read_csv(
        os.path.join(OUTPUT_DIR, 'pharma1_objective2_sourcing_shifts_2025.csv'))

    d['hts'] = pd.read_csv(
        os.path.join(OUTPUT_DIR, 'pharma1_hts_exposure_2024.csv'))

    d['risk'] = pd.read_csv(
        os.path.join(OUTPUT_DIR, 'pharma1_supply_chain_risk_2024.csv'))

    d['ranks'] = pd.read_csv(
        os.path.join(OUTPUT_DIR, 'pharma1_objective1_dependence_top_suppliers.csv'))

    return d


# ============================================================
# Analysis
# ============================================================

def run_analysis(d):
    exp    = d['exposure']
    shifts = d['shifts']
    hts    = d['hts']
    risk   = d['risk']
    ranks  = d['ranks']

    total_imports = exp['import_value_usd'].sum()

    print()
    print("=" * 68)
    print("PHARMACEUTICAL SOURCING PATTERNS -- ANALYSIS")
    print("Objective: How tariffs reshape US pharma supplier shares")
    print("=" * 68)

    # ----------------------------------------------------------
    # Part A: Baseline sourcing structure (2024)
    # ----------------------------------------------------------
    print()
    print("--- Part A: US Pharma Import Structure, 2024 ---")
    print()
    print(f"  Total US pharma imports: ${total_imports/1e9:.2f}B  ({len(exp)} supplier countries)")
    print()

    # HTS breakdown
    print(f"  By HS product category:")
    hts_map = {3002: 'Biological products (vaccines, blood, immunologicals)',
               3003: 'Medicaments -- bulk (not retail-packaged)',
               3004: 'Medicaments -- retail-packaged'}
    for _, row in hts.iterrows():
        desc = hts_map.get(int(row['hts']), str(row['hts']))
        print(f"    HS {int(row['hts'])}: ${row['import_value_bn']:.1f}B  "
              f"({row['import_share_pct']:.1f}%)  -- {desc}")
    print()

    # Top 10 suppliers
    top10 = exp.head(10)
    cum_share = top10['import_share_pct'].sum()
    print(f"  Top 10 suppliers account for {cum_share:.1f}% of total imports:")
    print()
    print(f"  {'Rank':<5}  {'Country':<15}  {'Value':>9}  "
          f"{'Share':>7}  {'Tariff':>7}  {'Post-Tariff':>12}  {'Change':>8}")
    print(f"  {'-'*5}  {'-'*15}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*12}  {'-'*8}")
    for _, row in top10.iterrows():
        print(f"  {int(row['rank']):<5}  {row['country']:<15}  "
              f"${row['import_value_usd']/1e9:>7.2f}B  "
              f"{row['import_share_pct']:>6.2f}%  "
              f"{row['tariff_pct']:>6.1f}%  "
              f"{row['post_tariff_share_pct']:>11.2f}%  "
              f"{row['delta_share_pp']:>+7.2f}pp")
    print()
    print(f"  Source: pharma1_country_exposure_2024.csv")

    # ----------------------------------------------------------
    # Part B: Tariff exposure by band
    # ----------------------------------------------------------
    print()
    print("--- Part B: Import Exposure by Tariff Band ---")
    print()
    bands = [
        (0,  15, '10-15%  (SGP, GBR, CAN, AUS)'),
        (15, 21, '15-20%  (EU bloc: IRL, DEU, BEL, ITA, NLD, FRA)'),
        (21, 28, '21-27%  (IND, MEX, ISR)'),
        (28, 40, '28-35%  (CHE, KOR)'),
        (40, 100,'> 40%   (CHN)'),
    ]
    print(f"  {'Tariff Band':<40}  {'Countries':>10}  {'Trade Value':>12}  {'Share':>7}")
    print(f"  {'-'*40}  {'-'*10}  {'-'*12}  {'-'*7}")
    for lo, hi, label in bands:
        mask = (exp['tariff_pct'] >= lo) & (exp['tariff_pct'] < hi)
        n    = mask.sum()
        val  = exp.loc[mask, 'import_value_usd'].sum()
        shr  = val / total_imports * 100
        print(f"  {label:<40}  {n:>10}  ${val/1e9:>10.2f}B  {shr:>6.1f}%")
    print()
    print(f"  Source: pharma1_country_exposure_2024.csv")

    # ----------------------------------------------------------
    # Part C: Who gains and who loses (2025 sourcing shifts)
    # ----------------------------------------------------------
    print()
    print("--- Part C: Post-Tariff Sourcing Shifts (2025 Projections) ---")
    print()
    gainers = shifts[shifts['change_pp'] > 0].sort_values('change_pp', ascending=False)
    losers  = shifts[shifts['change_pp'] < 0].sort_values('change_pp')

    print(f"  GAINERS -- countries capturing diverted import share:")
    print(f"  {'Country':<15}  {'Tariff':>7}  {'Pre-Share':>10}  "
          f"{'Post-Share':>11}  {'Gain':>8}")
    print(f"  {'-'*15}  {'-'*7}  {'-'*10}  {'-'*11}  {'-'*8}")
    for _, row in gainers.iterrows():
        print(f"  {row['country']:<15}  {row['tariff_pct']:>6.1f}%  "
              f"{row['pre_tariff_share_pct']:>9.3f}%  "
              f"{row['post_tariff_share_pct']:>10.3f}%  "
              f"{row['change_pp']:>+7.3f}pp")

    print()
    print(f"  LOSERS -- countries losing import share:")
    print(f"  {'Country':<15}  {'Tariff':>7}  {'Pre-Share':>10}  "
          f"{'Post-Share':>11}  {'Loss':>8}")
    print(f"  {'-'*15}  {'-'*7}  {'-'*10}  {'-'*11}  {'-'*8}")
    for _, row in losers.iterrows():
        print(f"  {row['country']:<15}  {row['tariff_pct']:>6.1f}%  "
              f"{row['pre_tariff_share_pct']:>9.3f}%  "
              f"{row['post_tariff_share_pct']:>10.3f}%  "
              f"{row['change_pp']:>+7.3f}pp")
    print()
    print(f"  Source: pharma1_objective2_sourcing_shifts_2025.csv")

    # ----------------------------------------------------------
    # Part D: Rank changes 2024 -> 2025
    # ----------------------------------------------------------
    print()
    print("--- Part D: Supplier Rank Changes 2024 to 2025 ---")
    print()
    print(f"  {'Country':<15}  {'2024 Rank':>10}  {'2024 Share':>11}  "
          f"{'2025 Rank':>10}  {'2025 Share':>11}  {'Rank Change':>12}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*11}  {'-'*12}")
    for _, row in ranks.iterrows():
        chg = int(row['rank_2024']) - int(row['rank_2025'])
        arrow = f"+{chg}" if chg > 0 else str(chg) if chg < 0 else "="
        print(f"  {row['country']:<15}  {int(row['rank_2024']):>10}  "
              f"{row['share_2024_pct']:>10.2f}%  "
              f"{int(row['rank_2025']):>10}  "
              f"{row['share_2025_pct']:>10.2f}%  "
              f"{arrow:>12}")
    print()
    print(f"  Source: pharma1_objective1_dependence_top_suppliers.csv")

    # ----------------------------------------------------------
    # Part E: Key findings
    # ----------------------------------------------------------
    print()
    print("--- Part E: Key Findings ---")
    print()

    top_gainer = gainers.iloc[0]
    top_loser  = losers.iloc[0]
    china_row  = shifts[shifts['country'] == 'China'].iloc[0]
    ireland    = shifts[shifts['country'] == 'Ireland'].iloc[0]

    # Low tariff countries combined gain
    low_tariff_gain = gainers[gainers['tariff_pct'] <= 10]['change_pp'].sum()

    print(f"  1. Ireland remains the dominant supplier (21.8% post-tariff share)")
    print(f"     despite a 20% tariff -- its scale ($45.5B) is irreplaceable short-run.")
    print()
    print(f"  2. Largest share gainer: {top_gainer['country']} (+{top_gainer['change_pp']:.3f}pp)")
    print(f"     at {top_gainer['tariff_pct']:.0f}% tariff -- lowest-tariff major supplier.")
    print()
    print(f"  3. Largest share loser: {top_loser['country']} ({top_loser['change_pp']:.3f}pp)")
    print(f"     at {top_loser['tariff_pct']:.0f}% tariff -- highest tariff among top suppliers.")
    print()
    print(f"  4. China loses {abs(china_row['change_pp']):.3f}pp at 54% tariff.")
    print(f"     Pre-tariff share was only {china_row['pre_tariff_share_pct']:.2f}% -- US pharma")
    print(f"     was already less China-dependent than general manufacturing.")
    print()
    print(f"  5. Low-tariff countries (<=10%: SGP, GBR, CAN, AUS) collectively")
    print(f"     gain +{low_tariff_gain:.3f}pp -- but face hard capacity constraints")
    print(f"     to absorb diverted volumes quickly.")
    print()
    print(f"  6. 56.6% of imports come from EU countries all facing 20% tariff --")
    print(f"     there is no large-scale low-tariff alternative bloc.")

    return gainers, losers


# ============================================================
# Figures
# ============================================================

def plot_top15_pre_post(d):
    exp    = d['exposure'].head(15)
    shifts = d['shifts']

    def bar_color(tau):
        if tau >= 40:  return '#c0392b'
        if tau >= 28:  return '#e74c3c'
        if tau >= 21:  return '#e67e22'
        if tau >= 15:  return '#f1c40f'
        return '#27ae60'

    colors   = [bar_color(t) for t in exp['tariff_pct']]
    countries = exp['country'].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('US Pharmaceutical Sourcing: Pre vs Post-Tariff Supplier Shares\n'
                 '(Top 15 suppliers, 2024 baseline, Liberation Day tariffs)',
                 fontsize=12, fontweight='bold')

    ax = axes[0]
    bars = ax.barh(countries, exp['import_share_pct'], color=colors)
    ax.set_xlabel('Import Share (%)')
    ax.set_title('Pre-Tariff Share (2024)')
    ax.invert_yaxis()
    for i, (bar, row) in enumerate(zip(bars, exp.itertuples())):
        ax.text(bar.get_width() + 0.1, i,
                f'  ${row.import_value_usd/1e9:.1f}B | {row.tariff_pct:.0f}%',
                va='center', fontsize=7.5)
    ax.set_xlim(0, 33)

    ax2 = axes[1]
    bars2 = ax2.barh(countries, exp['post_tariff_share_pct'], color=colors)
    ax2.set_xlabel('Post-Tariff Share (%)')
    ax2.set_title('Post-Tariff Share (Projected 2025)')
    ax2.invert_yaxis()
    for i, (bar, row) in enumerate(zip(bars2, exp.itertuples())):
        delta = row.delta_share_pp
        sign  = '+' if delta >= 0 else ''
        ax2.text(bar.get_width() + 0.1, i,
                 f'  {sign}{delta:.2f}pp', va='center', fontsize=7.5)
    ax2.set_xlim(0, 33)

    patches = [
        mpatches.Patch(color='#27ae60', label='Tariff <= 15% (SGP, GBR, CAN)'),
        mpatches.Patch(color='#f1c40f', label='15-20% (EU: IRL, DEU, BEL, ITA, NLD, FRA)'),
        mpatches.Patch(color='#e67e22', label='21-27% (IND)'),
        mpatches.Patch(color='#e74c3c', label='28-35% (CHE, KOR)'),
        mpatches.Patch(color='#c0392b', label='> 40% (CHN)'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, frameon=True, fontsize=8.5)
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    out = os.path.join(OUTPUT_DIR, 'fig_p1_top15_pre_post.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_gainers_losers(d):
    shifts  = d['shifts']
    gainers = shifts[shifts['change_pp'] > 0].sort_values('change_pp', ascending=False)
    losers  = shifts[shifts['change_pp'] < 0].sort_values('change_pp')

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('Post-Tariff Sourcing Reallocation: Gainers vs Losers\n'
                 '(Liberation Day tariff schedule, gravity reallocation)',
                 fontsize=11, fontweight='bold')

    def col(tau):
        return '#27ae60' if tau <= 15 else '#e74c3c' if tau >= 28 else '#e67e22'

    ax = axes[0]
    bars = ax.barh(gainers['country'], gainers['change_pp'],
                   color=[col(t) for t in gainers['tariff_pct']])
    ax.set_xlabel('Share Change (pp)')
    ax.set_title('Gainers (Diverted Share Captured)')
    ax.invert_yaxis()
    for bar, row in zip(bars, gainers.itertuples()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'  {row.tariff_pct:.0f}% tariff', va='center', fontsize=8)
    ax.set_xlim(0, gainers['change_pp'].max() * 1.5)

    ax2 = axes[1]
    bars2 = ax2.barh(losers['country'], losers['change_pp'],
                     color=[col(t) for t in losers['tariff_pct']])
    ax2.set_xlabel('Share Change (pp)')
    ax2.set_title('Losers (Share Diverted Away)')
    ax2.invert_yaxis()
    for bar, row in zip(bars2, losers.itertuples()):
        ax2.text(bar.get_width() - 0.01, bar.get_y() + bar.get_height()/2,
                 f'  {row.tariff_pct:.0f}% tariff', va='center', fontsize=8, ha='right')
    ax2.set_xlim(losers['change_pp'].min() * 1.4, 0)

    patches = [
        mpatches.Patch(color='#27ae60', label='<= 15% tariff'),
        mpatches.Patch(color='#e67e22', label='16-27% tariff'),
        mpatches.Patch(color='#e74c3c', label='>= 28% tariff'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, frameon=True, fontsize=9)
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    out = os.path.join(OUTPUT_DIR, 'fig_p1_gainers_losers.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_tariff_tier_reallocation(d):
    exp = d['exposure']
    total = exp['import_value_usd'].sum()

    bands = [
        ('<= 15%\n(SGP,GBR,CAN)', 0,  15,  '#27ae60'),
        ('15-20%\n(EU bloc)',      15, 21,  '#f1c40f'),
        ('21-27%\n(IND,MEX)',      21, 28,  '#e67e22'),
        ('28-35%\n(CHE,KOR)',      28, 40,  '#e74c3c'),
        ('> 40%\n(CHN)',           40, 100, '#c0392b'),
    ]

    labels, pre_shares, post_shares, colors = [], [], [], []
    for label, lo, hi, col in bands:
        mask = (exp['tariff_pct'] >= lo) & (exp['tariff_pct'] < hi)
        pre  = exp.loc[mask, 'import_share_pct'].sum()
        post = exp.loc[mask, 'post_tariff_share_pct'].sum()
        labels.append(label)
        pre_shares.append(pre)
        post_shares.append(post)
        colors.append(col)

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - w/2, pre_shares,  w, label='Pre-Tariff',  color=colors, alpha=0.9)
    bars2 = ax.bar(x + w/2, post_shares, w, label='Post-Tariff',
                   color=colors, alpha=0.55, edgecolor='black', linewidth=0.8)

    for bar, val in list(zip(bars1, pre_shares)) + list(zip(bars2, post_shares)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Import Share (%)')
    ax.set_title('Import Share Reallocation by Tariff Tier\n'
                 '(Pre vs Post-Tariff, Liberation Day schedule)',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.axhline(0, color='black', linewidth=0.6)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_p1_tariff_tier_reallocation.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_hts_breakdown(d):
    hts = d['hts']
    hts_map = {3002: 'HS 3002\nBiological products\n(vaccines, blood)',
               3003: 'HS 3003\nMedicaments\n(bulk)',
               3004: 'HS 3004\nMedicaments\n(retail-packaged)'}
    labels = [hts_map.get(int(r['hts']), str(r['hts'])) for _, r in hts.iterrows()]
    values = hts['import_value_bn'].tolist()
    colors = ['#3498db', '#95a5a6', '#e74c3c']

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('US Pharma Imports by HS Product Category (2024)\n'
                 f"Total: ${sum(values):.1f}B",
                 fontsize=11, fontweight='bold')

    ax = axes[0]
    bars = ax.bar(range(3), values, color=colors, edgecolor='white', linewidth=1.2, width=0.55)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'${val:.1f}B', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel('Import Value (USD Billions)')
    ax.set_title('Import Value by HS Code')

    ax2 = axes[1]
    shares = hts['import_share_pct'].tolist()
    wedges, texts, autotexts = ax2.pie(
        shares, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90, pctdistance=0.75
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax2.set_title('Share of Total Pharma Imports')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_p1_hts_breakdown.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_rank_changes(d):
    ranks = d['ranks']

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('Pharma Supplier Rank Changes: 2024 vs 2025\n'
                 '(After Liberation Day tariff exposure)',
                 fontsize=11, fontweight='bold')

    countries = ranks['country'].tolist()
    x = np.arange(len(countries))
    w = 0.35

    bars1 = ax.bar(x - w/2, ranks['share_2024_pct'], w,
                   label='2024 Share', color='#3498db', alpha=0.85)
    bars2 = ax.bar(x + w/2, ranks['share_2025_pct'], w,
                   label='2025 Share (projected)', color='#e74c3c', alpha=0.85)

    for bar, val in list(zip(bars1, ranks['share_2024_pct'])) + \
                    list(zip(bars2, ranks['share_2025_pct'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    # Rank annotations
    for i, row in ranks.iterrows():
        chg = int(row['rank_2024']) - int(row['rank_2025'])
        label = f'#{int(row["rank_2024"])} -> #{int(row["rank_2025"])}'
        color = '#27ae60' if chg > 0 else '#e74c3c' if chg < 0 else '#7f8c8d'
        ax.text(i, max(row['share_2024_pct'], row['share_2025_pct']) + 1.2,
                label, ha='center', fontsize=7.5, color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(countries, fontsize=9)
    ax.set_ylabel('Import Share (%)')
    ax.set_title('Green = moved up, Red = moved down, Grey = unchanged')
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_p1_rank_changes.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 68)
    print("PHARMA1 SOURCING ANALYSIS -- Loading data...")
    print("=" * 68)

    d = load_data()
    print(f"  pharma1_country_exposure_2024.csv         {len(d['exposure'])} countries")
    print(f"  pharma1_objective2_sourcing_shifts_2025   {len(d['shifts'])} countries")
    print(f"  pharma1_hts_exposure_2024.csv             {len(d['hts'])} HS codes")
    print(f"  pharma1_supply_chain_risk_2024.csv        {len(d['risk'])} countries")
    print(f"  pharma1_objective1_dependence_top_suppliers  {len(d['ranks'])} countries")

    gainers, losers = run_analysis(d)

    print()
    print("=" * 68)
    print("Generating Figures")
    print("=" * 68)
    plot_top15_pre_post(d)
    plot_gainers_losers(d)
    plot_tariff_tier_reallocation(d)
    plot_hts_breakdown(d)
    plot_rank_changes(d)

    print()
    print("=" * 68)
    print("COMPLETE -- Figures saved to python_output/:")
    print("  fig_p1_top15_pre_post.png")
    print("  fig_p1_gainers_losers.png")
    print("  fig_p1_tariff_tier_reallocation.png")
    print("  fig_p1_hts_breakdown.png")
    print("  fig_p1_rank_changes.png")
    print("=" * 68)


if __name__ == '__main__':
    main()
