"""
data_utils.py
=============
Shared data-loading utilities for Phase 2 sector analyses.

Handles:
  - Locating the main repository data directory (works from worktree or main repo)
  - Loading OECD ICIO input-output coefficient matrix and computing
    sector-specific IO multipliers for the USA
  - Loading pre-processed sector tariff shocks (from HTS8 schedule)
  - Loading pharma bilateral trade weights
  - Loading Cavallo et al. daily price indices
  - Loading the US 2025 HTS8 tariff schedule
"""

import os
import json
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def find_data_root():
    """
    Walk upward from this file's location searching for the project data
    directory (identified by the presence of DATA_FILES_INVENTORY.md).
    Falls back to an absolute path if the walk fails.
    """
    current = os.path.abspath(os.path.dirname(__file__))
    for _ in range(12):
        candidate = os.path.join(current, 'data')
        if os.path.exists(os.path.join(candidate, 'DATA_FILES_INVENTORY.md')):
            return candidate
        current = os.path.dirname(current)

    # Hard fallback — covers the worktree layout
    fallback = r'C:\Users\joelo\Desktop\Capstone\liberation-day-replication\data'
    if os.path.exists(fallback):
        return fallback

    raise FileNotFoundError(
        "Cannot locate project data directory. "
        "Expected DATA_FILES_INVENTORY.md inside a 'data/' folder."
    )


DATA_ROOT = find_data_root()


# ---------------------------------------------------------------------------
# OECD ICIO input-output data
# ---------------------------------------------------------------------------

def load_icio_sector_multipliers(beta_labor=0.49):
    """
    Compute US sector-specific IO multipliers from the 2022 OECD ICIO data.

    The pre-processed ICIO coefficient matrix (4050x4050) stores, for each
    (source_sector, destination_sector) pair, the share of destination sector
    output whose intermediate inputs come from source_sector.  Column sums = 1.

    For each US model sector we:
      1. Find the subset of columns corresponding to USA x that model sector.
      2. Sum the rows from non-USA sources  ->  intermediate import share.
      3. Compute the Leontief-style roundabout multiplier:
             M = 1 / (1 - (1 - beta_labor) * import_share_interm)

    Parameters
    ----------
    beta_labor : float
        Labor share in value added (default 0.49, from main_io.py).

    Returns
    -------
    dict  model_sector -> dict with keys:
            'import_share_interm'   float  share of intermediates that are imported
            'io_multiplier'         float  roundabout supply-chain multiplier
    """
    icio_dir = os.path.join(DATA_ROOT, 'processed', 'icio_2022')

    # Load index (4050 rows: country x icio_sector pairs)
    idx = pd.read_csv(os.path.join(icio_dir, 'country_sector_index.csv'))

    # Load sector mapping (adds model_sector column)
    smap = pd.read_csv(os.path.join(icio_dir, 'sector_map.csv'))
    idx = idx.merge(
        smap[['country_sector', 'model_sector']],
        on='country_sector', how='left'
    )

    # Load IO coefficient matrix
    A = np.load(os.path.join(icio_dir, 'io_coeff_matrix.npy'))

    usa_mask = (idx['country'] == 'USA').values
    non_usa_mask = ~usa_mask

    results = {}
    for model_sector in idx['model_sector'].dropna().unique():
        usa_sector_cols = np.where(
            usa_mask & (idx['model_sector'] == model_sector)
        )[0]
        if len(usa_sector_cols) == 0:
            continue

        # Average intermediate import share across all ICIO sub-sectors
        shares = []
        for col in usa_sector_cols:
            col_import_share = float(A[non_usa_mask, col].sum())
            shares.append(col_import_share)

        imp_share = float(np.mean(shares))
        multiplier = 1.0 / (1.0 - (1.0 - beta_labor) * imp_share)

        results[model_sector] = {
            'import_share_interm': imp_share,
            'io_multiplier': multiplier,
        }

    return results


def get_model_sector_io_multiplier(model_sector, beta_labor=0.49):
    """
    Convenience wrapper: return the IO multiplier for a single model sector.
    """
    mults = load_icio_sector_multipliers(beta_labor=beta_labor)
    if model_sector in mults:
        return mults[model_sector]['io_multiplier'], mults[model_sector]['import_share_interm']
    return 1.18, 0.30   # fallback if ICIO data unavailable


# ---------------------------------------------------------------------------
# Sector tariff shocks (HTS8-derived)
# ---------------------------------------------------------------------------

def load_sector_tariff_shocks():
    """
    Load pre-processed sector-level tariff shocks derived from the 2025
    US HTS8 tariff schedule.

    Returns a DataFrame with columns:
        scenario, model_sector, tariff_rate

    Scenarios:
        baseline_no_tariffs      - pre-Liberation Day (all zero)
        liberation_day_schedule  - HTS8 product-level rates under Liberation Day
        optimal_uniform_19       - 19% uniform counterfactual
        industry_focused         - sector-differentiated counterfactual
        supply_chain_disruption  - supply-chain shock counterfactual
    """
    path = os.path.join(DATA_ROOT, 'processed', 'shocks', 'sector_tariff_shocks.csv')
    df = pd.read_csv(path)
    return df[['scenario', 'model_sector', 'tariff_rate']]


def get_hts8_tariff_by_sector(scenario='liberation_day_schedule'):
    """
    Return a dict model_sector -> tariff_rate for a given scenario.
    """
    df = load_sector_tariff_shocks()
    subset = df[df['scenario'] == scenario].set_index('model_sector')['tariff_rate']
    return subset.to_dict()


# ---------------------------------------------------------------------------
# Pharma bilateral trade weights
# ---------------------------------------------------------------------------

def load_pharma_trade_weights(tariffs_csv=None, country_labels_csv=None):
    """
    Load actual pharma bilateral trade weights (132 exporting countries)
    and join with Liberation Day tariff rates.

    Parameters
    ----------
    tariffs_csv : str or None
        Path to base_data/tariffs.csv.  Auto-resolved if None.
    country_labels_csv : str or None
        Path to base_data/country_labels.csv.  Auto-resolved if None.

    Returns
    -------
    DataFrame with columns:
        iso3, trade_value_usd, weight, applied_tariff (NaN if not matched)
    """
    base = os.path.join(DATA_ROOT, 'base_data')
    if tariffs_csv is None:
        tariffs_csv = os.path.join(base, 'tariffs.csv')
    if country_labels_csv is None:
        country_labels_csv = os.path.join(base, 'country_labels.csv')

    weights = pd.read_csv(
        os.path.join(DATA_ROOT, 'processed', 'shocks', 'pharma_trade_weights.csv')
    )

    # Load tariff schedule and country codes
    tariffs = pd.read_csv(tariffs_csv)
    labels  = pd.read_csv(country_labels_csv)

    # tariffs.csv has one row per country in the GE model (194 rows, no header key)
    # Align by position with country_labels
    if 'iso3' not in tariffs.columns:
        tariffs = tariffs.copy()
        tariffs['iso3'] = labels['iso3'].values[:len(tariffs)]

    tariffs = tariffs.rename(columns={'applied_tariff': 'tau_country'})

    merged = weights.merge(
        tariffs[['iso3', 'tau_country']],
        on='iso3', how='left'
    )
    return merged


# ---------------------------------------------------------------------------
# Cavallo et al. daily price indices
# ---------------------------------------------------------------------------

def load_cavallo_price_indices():
    """
    Load Cavallo et al. daily price indices for US, Canada, Mexico, China.
    Dates normalized to 1.0 at base date (2024-10-01).

    Returns
    -------
    DataFrame with columns:
        date (datetime), index_canada, index_mexico, index_china, index_usa
    """
    path = os.path.join(DATA_ROOT, 'daily_price_indices_cavallo_etal.csv')
    df = pd.read_csv(path)
    # Handle Stata-style dates like "01oct2024"
    df['date'] = pd.to_datetime(df['date'], format='%d%b%Y', errors='coerce')
    if df['date'].isna().all():
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df.sort_values('date').reset_index(drop=True)


def summarize_cavallo_pass_through(event_date='2025-04-02', window_days=30):
    """
    Compute average daily price index change in the US vs control countries
    in the [event_date, event_date + window_days] window.

    Parameters
    ----------
    event_date : str  Liberation Day announcement date
    window_days : int  Number of days after event to average

    Returns
    -------
    dict with keys: usa_change, canada_change, mexico_change, china_change
                    (all as fractional changes from the event_date level)
    """
    df = load_cavallo_price_indices()
    event = pd.Timestamp(event_date)
    end   = event + pd.Timedelta(days=window_days)

    pre  = df[df['date'] < event].iloc[-1]   # last observation before event
    post = df[(df['date'] >= event) & (df['date'] <= end)]

    if post.empty:
        return {}

    post_avg = post[['index_usa', 'index_canada', 'index_mexico', 'index_china']].mean()

    return {
        'usa_change':    float(post_avg['index_usa']    / pre['index_usa']    - 1),
        'canada_change': float(post_avg['index_canada'] / pre['index_canada'] - 1),
        'mexico_change': float(post_avg['index_mexico'] / pre['index_mexico'] - 1),
        'china_change':  float(post_avg['index_china']  / pre['index_china']  - 1),
    }


# ---------------------------------------------------------------------------
# HTS8 tariff schedule
# ---------------------------------------------------------------------------

def load_hts8_pharma_lines():
    """
    Load HTS8 tariff lines flagged as pharmaceutical products.
    Attempts to parse ad-valorem rates from the MFN rate column.

    Returns
    -------
    DataFrame of pharma HTS8 lines with columns:
        hts8, description, mfn_rate_pct (float, NaN if not ad valorem)
    """
    path = os.path.join(DATA_ROOT, 'us_tariff_schedule_2025_hts8.csv')
    # Read only needed columns to keep memory manageable
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
    except Exception:
        return pd.DataFrame()

    # Filter pharmaceutical products
    if 'pharmaceutical_ind' in df.columns:
        pharma = df[df['pharmaceutical_ind'].str.strip() == 'K'].copy()
    else:
        return pd.DataFrame()

    # Identify MFN rate column
    mfn_col = None
    for col in df.columns:
        if 'mfn' in col.lower() or 'general' in col.lower():
            mfn_col = col
            break

    hts_col = df.columns[0]  # first column is typically the HTS code

    rows = []
    for _, row in pharma.iterrows():
        rate = np.nan
        if mfn_col and pd.notna(row.get(mfn_col, np.nan)):
            raw = str(row[mfn_col]).strip().replace('%', '')
            try:
                rate = float(raw) / 100.0
            except ValueError:
                pass
        rows.append({
            'hts8':         row.get(hts_col, ''),
            'mfn_rate_pct': rate,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Retail prices illustrative
# ---------------------------------------------------------------------------

def load_retail_prices_illustrative():
    """
    Load product-level before/after tariff prices.

    Returns
    -------
    DataFrame with columns:
        product_name, product_type, brand, price_before, price_after,
        implied_pass_through (= (after-before)/before)
    """
    path = os.path.join(DATA_ROOT, 'retail_prices_illustrative.csv')
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    rename = {
        'Product Name':      'product_name',
        'Product Type':      'product_type',
        'Brand Name':        'brand',
        'Price Before Tariff': 'price_before',
        'Price After Tariff':  'price_after',
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if 'price_before' in df.columns and 'price_after' in df.columns:
        df['price_before'] = pd.to_numeric(df['price_before'], errors='coerce')
        df['price_after']  = pd.to_numeric(df['price_after'],  errors='coerce')
        df['implied_pass_through'] = (df['price_after'] - df['price_before']) / df['price_before']

    return df


# ---------------------------------------------------------------------------
# Quick diagnostics
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"Data root: {DATA_ROOT}\n")

    print("[1] ICIO IO multipliers (USA sectors):")
    mults = load_icio_sector_multipliers()
    for sec, v in sorted(mults.items()):
        print(f"  {sec:25s}: import_share={v['import_share_interm']:.3f}  "
              f"multiplier={v['io_multiplier']:.3f}")

    print("\n[2] HTS8 sector tariff shocks (liberation_day_schedule):")
    shocks = get_hts8_tariff_by_sector()
    for sec, rate in sorted(shocks.items()):
        print(f"  {sec:25s}: {rate:.4f}  ({rate*100:.2f}%)")

    print("\n[3] Pharma trade weights (top 10):")
    pw = load_pharma_trade_weights()
    print(pw.nlargest(10, 'weight')[['iso3', 'weight', 'tau_country']].to_string(index=False))

    print("\n[4] Cavallo pass-through summary (30 days post Liberation Day):")
    cpt = summarize_cavallo_pass_through()
    for k, v in cpt.items():
        print(f"  {k}: {v*100:+.3f}%")

    print("\n[5] Retail prices illustrative (by category):")
    rp = load_retail_prices_illustrative()
    summary = rp.groupby('product_type')['implied_pass_through'].mean()
    print(summary.apply(lambda x: f"{x*100:.1f}%").to_string())
