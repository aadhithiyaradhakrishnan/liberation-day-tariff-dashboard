"""
Liberation Day Tariff MCP Server
FastMCP server exposing 6 analysis tools over the GE model and sector datasets.

Start with:
    python -m mcp_server.server
or:
    fastmcp run mcp_server/server.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "python_output")
sys.path.insert(0, os.path.join(ROOT, "code_python"))
sys.path.insert(0, ROOT)

# ── Auto-download large files from Hugging Face if missing ───────────────────
_LARGE_FILES = [
    os.path.join(ROOT, "data", "processed", "icio_2022", "io_coeff_matrix.npy"),
    os.path.join(ROOT, "data", "processed", "icio_2022", "io_intermediate_matrix.npy"),
    os.path.join(ROOT, "data", "code_and_release_data", "301 model", "D_all_data.zip"),
]
if any(not os.path.exists(f) for f in _LARGE_FILES):
    import download_data
    download_data.main()

from utils.solver_utils import solve_nu

# ── Dark Plotly theme (matches dashboard) ─────────────────────────────────────
_THEME = dict(
    paper_bgcolor="#0f1117",
    plot_bgcolor="#0f1117",
    font=dict(family="Inter", color="#cbd5e1"),
    margin=dict(l=20, r=20, t=40, b=20),
    colorway=["#2563eb","#22d3a0","#f87171","#fbbf24","#a78bfa","#fb923c","#38bdf8"],
)

# ── Scenario index map (mirrors dashboard/app.py) ─────────────────────────────
SCENARIO_MAP = {
    "ustr_no_retaliation":         0,
    "ustr_lump_sum":               7,
    "optimal_tariff":              3,
    "ustr_reciprocal_retaliation": 5,
    "ustr_optimal_retaliation":    4,
    "flat_15pct":                  None,   # stored in scenario_15pct.npz
}
# Results column order: [welfare, deficit, exports/GDP, imports/GDP, employment, CPI, rev/GDP]
COL = dict(welfare=0, deficit=1, exports=2, imports=3, employment=4, cpi=5, rev_gdp=6)

# ── Cached data loaders ───────────────────────────────────────────────────────
_cache: dict = {}

def _baseline():
    if "baseline" not in _cache:
        b   = np.load(os.path.join(OUT, "baseline_results.npz"), allow_pickle=True)
        cl  = pd.read_csv(os.path.join(DATA, "base_data", "country_labels.csv"))
        sc15 = np.load(os.path.join(OUT, "scenario_15pct.npz"), allow_pickle=True)
        _cache["baseline"] = {
            "results":      b["results"],       # (194, 7, 9)
            "Y_i":          b["Y_i"],           # (194,)
            "id_US":        int(b["id_US"]),    # 184
            "country_labels": cl,               # iso3, iso_code, CountryName
            "results_15pct":  sc15["results"],  # (194, 7)
        }
    return _cache["baseline"]

def _pharma():
    if "pharma" not in _cache:
        xl  = pd.ExcelFile(os.path.join(DATA, "Pharma1.xlsx"))
        raw = xl.parse("Query Results", header=2)
        monthly = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
        raw["annual_total"] = raw[monthly].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        raw["Year"]       = pd.to_numeric(raw["Year"], errors="coerce")
        raw["HTS Number"] = pd.to_numeric(raw["HTS Number"], errors="coerce")

        dep  = pd.read_csv(os.path.join(OUT, "pharma1_objective1_dependence_2025.csv"))
        src  = pd.read_csv(os.path.join(OUT, "pharma1_objective2_sourcing_shifts_2025.csv"))
        exp  = pd.read_csv(os.path.join(OUT, "pharma1_country_exposure_2024.csv"))
        risk = pd.read_csv(os.path.join(OUT, "pharma1_supply_chain_risk_2024.csv"))
        hts_exp = pd.read_csv(os.path.join(OUT, "pharma1_hts_exposure_2024.csv"))
        _cache["pharma"] = dict(raw=raw, dep=dep, src=src, exp=exp, risk=risk, hts_exp=hts_exp)
    return _cache["pharma"]

def _quintile():
    if "quintile" not in _cache:
        burd = pd.read_csv(os.path.join(OUT, "pharma1_objective3_consumer_burden_2025.csv"))
        retail = pd.read_csv(os.path.join(DATA, "retail_prices_illustrative.csv"))
        retail["pct_increase"] = (
            (retail["Price After Tariff"] - retail["Price Before Tariff"])
            / retail["Price Before Tariff"] * 100
        )
        _cache["quintile"] = dict(burd=burd, retail=retail)
    return _cache["quintile"]

def _manufacturing():
    if "manufacturing" not in _cache:
        naics  = pd.read_excel(os.path.join(DATA, "code_and_release_data", "301 model", "D_GO_by_NAICS.xlsx"))
        shocks = pd.read_csv(os.path.join(DATA, "processed", "shocks", "sector_tariff_shocks.csv"))
        hts    = pd.read_csv(os.path.join(DATA, "us_tariff_schedule_2025_hts8.csv"), low_memory=False)
        hts["mfn_rate"] = pd.to_numeric(hts["mfn_ad_val_rate"], errors="coerce").fillna(0)
        r = np.load(os.path.join(OUT, "sector_manufacturing_results.npz"), allow_pickle=True)
        mfg_stats = {k: float(r[k]) for k in r.keys()}
        _cache["manufacturing"] = dict(naics=naics, shocks=shocks, hts=hts, mfg_stats=mfg_stats)
    return _cache["manufacturing"]

def _tariff_country_df():
    """Merge country_labels + tariffs.csv by row position (row i = iso_code i+1)."""
    if "tariff_df" not in _cache:
        cl = pd.read_csv(os.path.join(DATA, "base_data", "country_labels.csv")).reset_index(drop=True)
        t  = pd.read_csv(os.path.join(DATA, "base_data", "tariffs.csv")).reset_index(drop=True)
        cl["applied_tariff"] = t["applied_tariff"]
        _cache["tariff_df"] = cl
    return _cache["tariff_df"]

def _gdp_df():
    if "gdp_df" not in _cache:
        _cache["gdp_df"] = pd.read_csv(os.path.join(DATA, "base_data", "gdp.csv")).reset_index(drop=True)
    return _cache["gdp_df"]

def _output_map():
    if "output_map" not in _cache:
        _cache["output_map"] = pd.read_csv(os.path.join(OUT, "output_map.csv"))
    return _cache["output_map"]

def _cavallo():
    if "cavallo" not in _cache:
        _cache["cavallo"] = pd.read_csv(os.path.join(DATA, "daily_price_indices_cavallo_etal.csv"))
    return _cache["cavallo"]

def _retail_npz():
    if "retail_npz" not in _cache:
        r = np.load(os.path.join(OUT, "sector_retail_results.npz"), allow_pickle=True)
        _cache["retail_npz"] = {k: r[k] for k in r.keys()}
    return _cache["retail_npz"]

# Region lookup: iso3 → region
_REGION_MAP = {
    "AFG":"South Asia","ALB":"Europe","DZA":"MENA","AGO":"Africa","ATG":"Americas",
    "ARG":"Americas","ARM":"Europe","AUS":"Asia-Pacific","AUT":"Europe","AZE":"Europe",
    "BHS":"Americas","BHR":"MENA","BGD":"South Asia","BRB":"Americas","BLR":"Europe",
    "BEL":"Europe","BLZ":"Americas","BEN":"Africa","BTN":"South Asia","BOL":"Americas",
    "BIH":"Europe","BWA":"Africa","BRA":"Americas","BRN":"Asia-Pacific","BGR":"Europe",
    "BFA":"Africa","BDI":"Africa","CPV":"Africa","KHM":"Asia-Pacific","CMR":"Africa",
    "CAN":"Americas","CAF":"Africa","TCD":"Africa","CHL":"Americas","CHN":"Asia-Pacific",
    "COL":"Americas","COM":"Africa","COD":"Africa","COG":"Africa","CRI":"Americas",
    "CIV":"Africa","HRV":"Europe","CUB":"Americas","CYP":"Europe","CZE":"Europe",
    "DNK":"Europe","DJI":"Africa","DOM":"Americas","ECU":"Americas","EGY":"MENA",
    "SLV":"Americas","GNQ":"Africa","ERI":"Africa","EST":"Europe","SWZ":"Africa",
    "ETH":"Africa","FJI":"Asia-Pacific","FIN":"Europe","FRA":"Europe","GAB":"Africa",
    "GMB":"Africa","GEO":"Europe","DEU":"Europe","GHA":"Africa","GRC":"Europe",
    "GRD":"Americas","GTM":"Americas","GIN":"Africa","GNB":"Africa","GUY":"Americas",
    "HTI":"Americas","HND":"Americas","HUN":"Europe","ISL":"Europe","IND":"South Asia",
    "IDN":"Asia-Pacific","IRN":"MENA","IRQ":"MENA","IRL":"Europe","ISR":"MENA",
    "ITA":"Europe","JAM":"Americas","JPN":"Asia-Pacific","JOR":"MENA","KAZ":"Europe",
    "KEN":"Africa","KIR":"Asia-Pacific","PRK":"Asia-Pacific","KOR":"Asia-Pacific",
    "KWT":"MENA","KGZ":"Europe","LAO":"Asia-Pacific","LVA":"Europe","LBN":"MENA",
    "LSO":"Africa","LBR":"Africa","LBY":"MENA","LIE":"Europe","LTU":"Europe",
    "LUX":"Europe","MDG":"Africa","MWI":"Africa","MYS":"Asia-Pacific","MDV":"South Asia",
    "MLI":"Africa","MLT":"Europe","MHL":"Asia-Pacific","MRT":"Africa","MUS":"Africa",
    "MEX":"Americas","FSM":"Asia-Pacific","MDA":"Europe","MCO":"Europe","MNG":"Asia-Pacific",
    "MNE":"Europe","MAR":"MENA","MOZ":"Africa","MMR":"Asia-Pacific","NAM":"Africa",
    "NRU":"Asia-Pacific","NPL":"South Asia","NLD":"Europe","NZL":"Asia-Pacific","NIC":"Americas",
    "NER":"Africa","NGA":"Africa","MKD":"Europe","NOR":"Europe","OMN":"MENA","PAK":"South Asia",
    "PLW":"Asia-Pacific","PAN":"Americas","PNG":"Asia-Pacific","PRY":"Americas","PER":"Americas",
    "PHL":"Asia-Pacific","POL":"Europe","PRT":"Europe","QAT":"MENA","ROU":"Europe",
    "RUS":"Europe","RWA":"Africa","KNA":"Americas","LCA":"Americas","VCT":"Americas",
    "WSM":"Asia-Pacific","SMR":"Europe","STP":"Africa","SAU":"MENA","SEN":"Africa",
    "SRB":"Europe","SYC":"Africa","SLE":"Africa","SGP":"Asia-Pacific","SVK":"Europe",
    "SVN":"Europe","SLB":"Asia-Pacific","SOM":"Africa","ZAF":"Africa","SSD":"Africa",
    "ESP":"Europe","LKA":"South Asia","SDN":"Africa","SUR":"Americas","SWE":"Europe",
    "CHE":"Europe","SYR":"MENA","TWN":"Asia-Pacific","TJK":"Europe","TZA":"Africa",
    "THA":"Asia-Pacific","TLS":"Asia-Pacific","TGO":"Africa","TON":"Asia-Pacific",
    "TTO":"Americas","TUN":"MENA","TUR":"Europe","TKM":"Europe","TUV":"Asia-Pacific",
    "UGA":"Africa","UKR":"Europe","ARE":"MENA","GBR":"Europe","USA":"Americas",
    "URY":"Americas","UZB":"Europe","VUT":"Asia-Pacific","VEN":"Americas","VNM":"Asia-Pacific",
    "YEM":"MENA","ZMB":"Africa","ZWE":"Africa","ABW":"Americas","HKG":"Asia-Pacific",
    "MAC":"Asia-Pacific","PSE":"MENA","AND":"Europe",
}

def _make_chart(traces: list, layout_overrides: dict) -> dict:
    """Return a plain dict that Plotly's go.Figure() accepts directly."""
    layout = {**_THEME, **layout_overrides}
    return {"data": traces, "layout": layout}

# ── FastMCP server registration (only when run as __main__) ──────────────────
# Tool functions below are plain Python — no decorator needed for dashboard imports.
# FastMCP wiring happens at the bottom inside __main__ guard.


# ─────────────────────────────────────────────────────────────────────────────
def get_welfare_results(country: str = None, scenario: str = "ustr_no_retaliation") -> dict:
    """
    Query GE welfare/CPI/trade results for one or all countries under a given scenario.

    Parameters
    ----------
    country : str or None
        ISO3 country code (e.g. 'USA', 'CHN') or None to return all 194 countries.
    scenario : str
        One of: ustr_no_retaliation, ustr_lump_sum, optimal_tariff,
        ustr_reciprocal_retaliation, ustr_optimal_retaliation, flat_15pct.

    Returns
    -------
    dict with keys: data (list of records), chart_spec (Plotly figure dict)
    """
    try:
        b  = _baseline()
        cl = b["country_labels"]           # cols: iso3, iso_code, CountryName

        sc_idx = SCENARIO_MAP.get(scenario)
        if sc_idx is None and scenario != "flat_15pct":
            return {"data": {"error": f"Unknown scenario '{scenario}'. Valid: {list(SCENARIO_MAP)}"}, "chart_spec": {}}

        # Pull results array — col order: [welfare, deficit, exports, imports, employment, CPI, rev/GDP]
        if scenario == "flat_15pct":
            arr = b["results_15pct"]           # (194, 7)
            get_row = lambda i: arr[i, :]
        else:
            arr = b["results"]                 # (194, 7, 9)
            get_row = lambda i: arr[i, :, sc_idx]

        metric_names = ["welfare_pct", "deficit_pct", "exports_gdp_pct",
                        "imports_gdp_pct", "employment_pct", "cpi_pct", "rev_gdp_pct"]

        records = []
        for idx, row in cl.iterrows():
            rec = {"iso3": row["iso3"], "country": row["CountryName"]}
            vals = get_row(idx)
            for name, val in zip(metric_names, vals):
                rec[name] = round(float(val), 4)
            records.append(rec)

        # Filter by country
        if country:
            iso = country.upper()
            records = [r for r in records if r["iso3"] == iso]
            if not records:
                return {"data": {"error": f"Country '{iso}' not found. Use ISO3 code."}, "chart_spec": {}}

        # Chart: horizontal bar of welfare for all countries (top/bottom 20 if no filter)
        if len(records) > 1:
            df_plot = pd.DataFrame(records).sort_values("welfare_pct")
            show = pd.concat([df_plot.head(15), df_plot.tail(15)]).drop_duplicates("iso3")
            colors = ["#22d3a0" if v >= 0 else "#f87171" for v in show["welfare_pct"]]
            traces = [{"type": "bar", "x": show["welfare_pct"].tolist(),
                       "y": show["country"].tolist(), "orientation": "h",
                       "marker": {"color": colors},
                       "text": [f"{v:+.2f}%" for v in show["welfare_pct"]],
                       "textposition": "outside"}]
            layout = {"title": f"Welfare Change — {scenario}", "xaxis": {"title": "Welfare %"},
                      "yaxis": {"autorange": "reversed"}, "height": 500}
        else:
            r = records[0]
            vals_plot = [r[m] for m in metric_names]
            colors = ["#22d3a0" if v >= 0 else "#f87171" for v in vals_plot]
            traces = [{"type": "bar", "x": metric_names, "y": vals_plot,
                       "marker": {"color": colors},
                       "text": [f"{v:+.2f}%" for v in vals_plot],
                       "textposition": "outside"}]
            layout = {"title": f"{r['country']} — {scenario}", "yaxis": {"title": "% Change"}, "height": 360}

        return {"data": records, "chart_spec": _make_chart(traces, layout)}

    except Exception as e:
        return {"data": {"error": str(e)}, "chart_spec": {}}


# ─────────────────────────────────────────────────────────────────────────────
def get_scenario_comparison(countries: list = None, scenarios: list = None) -> dict:
    """
    Pivot welfare outcomes across multiple countries and scenarios.

    Parameters
    ----------
    countries : list of ISO3 codes, e.g. ["USA", "CHN", "DEU"].
                Defaults to top 10 economies by GDP if None.
    scenarios : list of scenario names. Defaults to all 6 if None.

    Returns
    -------
    dict with keys: data (pivot table as records), chart_spec (grouped bar Plotly dict)
    """
    try:
        b  = _baseline()
        cl = b["country_labels"]

        if scenarios is None:
            scenarios = list(SCENARIO_MAP.keys())
        if countries is None:
            # top 10 by Y_i (GDP proxy)
            Y_i = b["Y_i"]
            top_idx = np.argsort(Y_i)[-10:][::-1]
            countries = cl.iloc[top_idx]["iso3"].tolist()

        countries_upper = [c.upper() for c in countries]
        cl_filtered = cl[cl["iso3"].isin(countries_upper)]

        pivot_rows = []
        for _, row in cl_filtered.iterrows():
            idx  = row.name
            rec  = {"iso3": row["iso3"], "country": row["CountryName"]}
            for sc_name in scenarios:
                sc_idx = SCENARIO_MAP.get(sc_name)
                if sc_name == "flat_15pct":
                    val = float(b["results_15pct"][idx, COL["welfare"]])
                elif sc_idx is not None:
                    val = float(b["results"][idx, COL["welfare"], sc_idx])
                else:
                    val = None
                rec[sc_name] = round(val, 4) if val is not None else None
            pivot_rows.append(rec)

        # Chart: grouped bar, x=country, one bar per scenario
        traces = []
        colors = ["#2563eb","#22d3a0","#fbbf24","#f87171","#a78bfa","#fb923c"]
        country_names = [r["country"] for r in pivot_rows]
        for i, sc_name in enumerate(scenarios):
            vals = [r[sc_name] for r in pivot_rows]
            traces.append({
                "type": "bar", "name": sc_name,
                "x": country_names, "y": vals,
                "marker": {"color": colors[i % len(colors)]},
            })
        layout = {"title": "Welfare % by Country & Scenario", "barmode": "group",
                  "yaxis": {"title": "Welfare %"}, "height": 420,
                  "legend": {"bgcolor": "#1a1d2e", "bordercolor": "#2d3250"},
                  "xaxis": {"tickangle": -25}}

        return {"data": pivot_rows, "chart_spec": _make_chart(traces, layout)}

    except Exception as e:
        return {"data": {"error": str(e)}, "chart_spec": {}}


# ─────────────────────────────────────────────────────────────────────────────
def get_pharma_supplier_risk(country: str = None, hts_code: int = None) -> dict:
    """
    Query pharma import exposure, compute HHI concentration, return suppliers ranked by risk.

    Parameters
    ----------
    country : str or None — filter to a specific exporting country name (e.g. 'Ireland').
    hts_code : int or None — filter by HTS code: 3002, 3003, or 3004.

    Returns
    -------
    dict with keys: data (supplier risk records + HHI), chart_spec (Plotly figure dict)
    """
    try:
        p   = _pharma()
        exp = p["exp"].copy()       # rank, country, iso3, import_value_usd, import_share_pct, tariff_pct, post_tariff_share_pct, delta_share_pp
        risk = p["risk"].copy()     # rank, country, import_value_bn, pre_tariff_share_pct, tariff_pct, post_tariff_share_pct, share_shift_pp, risk_tier
        hts_exp = p["hts_exp"].copy()  # hts, import_value_usd, import_value_bn, import_share_pct

        # HTS code filter — applies to raw Pharma1 data
        if hts_code is not None:
            raw = p["raw"]
            filtered = raw[raw["HTS Number"] == float(hts_code)]
            total = filtered["annual_total"].sum()
            by_country = (
                filtered.groupby("Country")["annual_total"].sum()
                .reset_index()
                .rename(columns={"Country": "country", "annual_total": "import_value_usd"})
            )
            by_country["import_share_pct"] = by_country["import_value_usd"] / total * 100
            by_country = by_country.sort_values("import_share_pct", ascending=False).head(20)
            # Compute HHI on this subset
            shares = by_country["import_share_pct"].values
            hhi = float(np.sum(shares ** 2))
            result_records = by_country.to_dict("records")
        else:
            # Use full 2024 exposure
            shares = exp["import_share_pct"].values
            hhi = float(np.sum(shares ** 2))
            result_records = risk.to_dict("records")

        # Country filter (post-HHI)
        if country:
            result_records = [r for r in result_records
                              if r.get("country", "").lower() == country.lower()]

        # Append HHI summary
        hhi_label = "Concentrated" if hhi > 2500 else "Moderate" if hhi > 1500 else "Competitive"
        summary = {"hhi": round(hhi, 1), "hhi_label": hhi_label,
                   "top_supplier_share_pct": round(float(shares[0]), 2) if len(shares) else None}

        # Chart: horizontal bar colored by risk_tier or tariff level
        if result_records and "risk_tier" in result_records[0]:
            tier_color = {"Very high": "#f87171", "High": "#fb923c",
                          "Moderate": "#fbbf24", "Low": "#22d3a0"}
            bar_colors = [tier_color.get(r.get("risk_tier","Low"), "#60a5fa") for r in result_records]
            y_vals  = [r["country"] for r in result_records]
            x_vals  = [r.get("pre_tariff_share_pct", r.get("import_share_pct", 0)) for r in result_records]
            text_vals = [f"{r.get('tariff_pct',0):.0f}% tariff | {r.get('risk_tier','')}" for r in result_records]
        else:
            bar_colors = "#2563eb"
            y_vals  = [r.get("country","") for r in result_records]
            x_vals  = [r.get("import_share_pct", 0) for r in result_records]
            text_vals = [f"{v:.1f}%" for v in x_vals]

        traces = [{"type": "bar", "x": x_vals, "y": y_vals, "orientation": "h",
                   "marker": {"color": bar_colors},
                   "text": text_vals, "textposition": "outside"}]
        title = f"Pharma Supplier Risk{' — HTS ' + str(hts_code) if hts_code else ''} | HHI={hhi:.0f} ({hhi_label})"
        layout = {"title": title, "xaxis": {"title": "Import Share %"},
                  "yaxis": {"autorange": "reversed"}, "height": 440}

        return {"data": {"suppliers": result_records, "summary": summary},
                "chart_spec": _make_chart(traces, layout)}

    except Exception as e:
        return {"data": {"error": str(e)}, "chart_spec": {}}


# ─────────────────────────────────────────────────────────────────────────────
def get_quintile_burden(category: str = None) -> dict:
    """
    Return tariff incidence by income quintile from pharma + retail incidence data.

    Parameters
    ----------
    category : str or None — retail product category filter:
               'Grocery', 'Clothing', 'Footwear', 'Home Appliances', 'Electronics'.
               None returns pharma quintile burden (from pharma1_objective3_consumer_burden_2025.csv).

    Returns
    -------
    dict with keys: data (quintile records), chart_spec (Plotly bar figure dict)
    """
    try:
        q = _quintile()

        if category:
            retail = q["retail"]
            valid_cats = retail["Product Type"].unique().tolist()
            if category not in valid_cats:
                return {"data": {"error": f"Invalid category '{category}'. Valid: {valid_cats}"}, "chart_spec": {}}

            filtered = retail[retail["Product Type"] == category]
            avg_before = float(filtered["Price Before Tariff"].mean())
            avg_after  = float(filtered["Price After Tariff"].mean())
            avg_pct    = float(filtered["pct_increase"].mean())

            # Distribute burden by quintile using BLS CEX goods share (matches dashboard hardcoded quintile_data)
            goods_shares = [0.420, 0.390, 0.370, 0.345, 0.300]
            quintile_names = ["Q1 (Poorest)", "Q2", "Q3", "Q4", "Q5 (Richest)"]
            avg_share = sum(goods_shares) / len(goods_shares)
            records = []
            for i, (q_name, gs) in enumerate(zip(quintile_names, goods_shares)):
                burden = avg_pct * (gs / avg_share)
                records.append({
                    "quintile": q_name,
                    "goods_share_pct": round(gs * 100, 1),
                    "price_increase_pct": round(avg_pct, 2),
                    "adjusted_burden_pct": round(burden, 2),
                })
            chart_title = f"Tariff Burden by Quintile — {category}"
            y_key = "adjusted_burden_pct"
        else:
            burd = q["burd"]
            records = burd.rename(columns={
                "group": "quintile",
                "annual_drug_spending_usd": "annual_drug_spending_usd",
                "extra_cost_from_tariffs_usd": "extra_cost_usd",
                "burden_pct_income": "burden_pct_income",
            }).to_dict("records")
            # Convert to percentage for display
            for r in records:
                r["burden_pct_income_display"] = round(r["burden_pct_income"] * 100, 3)
            chart_title = "Pharma Tariff Burden as % of Income by Quintile"
            y_key = "burden_pct_income_display"

        quintile_labels = [r["quintile"] for r in records]
        y_vals = [r[y_key] for r in records]
        bar_colors = ["#f87171","#fb923c","#fbbf24","#a3e635","#22d3a0"]

        traces = [{"type": "bar", "x": quintile_labels, "y": y_vals,
                   "marker": {"color": bar_colors},
                   "text": [f"{v:.3f}%" for v in y_vals], "textposition": "outside"}]
        layout = {"title": chart_title, "xaxis": {"title": "Income Quintile"},
                  "yaxis": {"title": "Burden %"}, "height": 360}

        regressivity = y_vals[0] / y_vals[-1] if y_vals[-1] else None
        return {
            "data": {"quintiles": records,
                     "regressivity_ratio": round(regressivity, 2) if regressivity else None},
            "chart_spec": _make_chart(traces, layout),
        }

    except Exception as e:
        return {"data": {"error": str(e)}, "chart_spec": {}}


# ─────────────────────────────────────────────────────────────────────────────
def get_manufacturing_shock(naics_code: str = None, top_n: int = 10) -> dict:
    """
    Return sector-level tariff shocks and gross output ranked by impact.

    Parameters
    ----------
    naics_code : str or None — filter to a specific NAICS code string (e.g. '3364').
                 None returns top_n sectors by 2021 gross output.
    top_n : int — number of top sectors to return (default 10, max 50).

    Returns
    -------
    dict with keys: data (sector records), chart_spec (Plotly horizontal bar dict)
    """
    try:
        m      = _manufacturing()
        naics  = m["naics"].copy()          # Line, NAICS Code, Name, 2016..2021
        shocks = m["shocks"].copy()         # scenario, model_sector, tariff_rate, source_tariff_file
        mfg_stats = m["mfg_stats"]

        top_n = min(top_n, 50)

        # Liberation Day shocks by model sector
        ld_shocks = shocks[shocks["scenario"] == "liberation_day_schedule"][
            ["model_sector", "tariff_rate"]
        ].set_index("model_sector")["tariff_rate"].to_dict()

        # Prepare NAICS table
        naics["go_2021"] = pd.to_numeric(naics["2021"], errors="coerce")
        naics["NAICS Code"] = naics["NAICS Code"].astype(str)
        naics_clean = naics[["NAICS Code","Name","go_2021"]].dropna(subset=["go_2021"])

        if naics_code:
            naics_clean = naics_clean[naics_clean["NAICS Code"].str.startswith(str(naics_code))]
            if naics_clean.empty:
                return {"data": {"error": f"No NAICS sectors found matching '{naics_code}'"}, "chart_spec": {}}
        else:
            naics_clean = naics_clean.nlargest(top_n, "go_2021")

        records = []
        for _, row in naics_clean.iterrows():
            rec = {
                "naics_code": row["NAICS Code"],
                "name": row["Name"],
                "gross_output_2021_mn": round(float(row["go_2021"]), 0),
                "liberation_day_tariff_rate_pct": round(
                    ld_shocks.get("manufacturing_other", 0) * 100, 2
                ),
            }
            records.append(rec)

        # Aggregate stats
        total_go = naics_clean["go_2021"].sum()
        summary = {
            "total_gross_output_bn": round(float(total_go) / 1e6, 2),
            "avg_tariff_mfg_pct": round(mfg_stats.get("tau_mfg_avg", 0) * 100, 2),
            "cpi_contribution_pp": round(mfg_stats.get("cpi_mfg_contribution", 0), 3),
            "io_multiplier": round(mfg_stats.get("io_mult_mfg", 0), 4),
        }

        # Chart: horizontal bar of gross output
        sorted_recs = sorted(records, key=lambda x: x["gross_output_2021_mn"], reverse=True)
        y_names = [r["name"][:35] for r in sorted_recs]
        x_vals  = [r["gross_output_2021_mn"] / 1e6 for r in sorted_recs]

        traces = [{"type": "bar", "x": x_vals, "y": y_names, "orientation": "h",
                   "marker": {"color": "#2563eb"},
                   "text": [f"${v:.1f}T" for v in x_vals], "textposition": "outside"}]
        layout = {"title": f"Top {top_n} NAICS Sectors by Gross Output (2021)",
                  "xaxis": {"title": "Gross Output ($T)"},
                  "yaxis": {"autorange": "reversed"}, "height": max(360, top_n * 24)}

        return {"data": {"sectors": records, "summary": summary},
                "chart_spec": _make_chart(traces, layout)}

    except Exception as e:
        return {"data": {"error": str(e)}, "chart_spec": {}}


# ─────────────────────────────────────────────────────────────────────────────
def run_tariff_scenario(tariff_overrides: dict, countries: list = None) -> dict:
    """
    Re-run a partial equilibrium approximation with custom tariff rates, return welfare
    delta vs the baseline USTR scenario (scenario index 0).

    This is a fast linearised approximation — not a full GE solve.
    Formula: Δwelfare ≈ (τ_new - τ_ustr) * import_share * (-elasticity / (1 + τ_ustr))

    Parameters
    ----------
    tariff_overrides : dict — country ISO3 → new tariff rate (e.g. {"CHN": 0.60, "DEU": 0.20}).
                       Countries not listed keep the USTR tariff.
    countries : list of ISO3 codes to return results for.
                Defaults to US + the overridden countries.

    Returns
    -------
    dict with keys: data (country-level welfare delta records), chart_spec (Plotly bar dict)
    """
    try:
        b  = _baseline()
        cl = b["country_labels"]
        id_US = b["id_US"]   # 184

        # Load tariff baseline
        tariff_path = os.path.join(DATA, "base_data", "tariffs.csv")
        ustr_tariffs = pd.read_csv(tariff_path, header=0).values.flatten()
        ustr_tariffs = pd.to_numeric(ustr_tariffs, errors="coerce")

        # Build new tariff vector
        new_tariffs = ustr_tariffs.copy()
        override_indices = {}
        for iso3, rate in tariff_overrides.items():
            iso3_upper = iso3.upper()
            match = cl[cl["iso3"] == iso3_upper]
            if match.empty:
                continue
            country_idx = int(match.index[0])
            new_tariffs[country_idx] = float(rate)
            override_indices[iso3_upper] = country_idx

        # Partial equilibrium approximation
        # Δwelfare_US ≈ sum_j [ (τ_j_new - τ_j_ustr) * λ_j_US * φ * (-ε / (1+τ_j_ustr)) ]
        # We use trade-share-weighted formula as a first-order approximation
        eps = 4.0
        trade_path = os.path.join(DATA, "base_data", "trade_cepii.csv")
        X_ji = pd.read_csv(trade_path, header=0).values
        X_ji = pd.DataFrame(X_ji).apply(pd.to_numeric, errors="coerce").fillna(0).values
        total_imports_US = X_ji[:, id_US].sum()
        import_shares    = X_ji[:, id_US] / (total_imports_US + 1e-12)

        # Welfare delta for US
        d_tau          = new_tariffs - ustr_tariffs
        d_tau[id_US]   = 0
        welfare_delta_US = float(np.sum(
            d_tau * import_shares * (-eps / (1 + ustr_tariffs + 1e-12)) * 100
        ))

        # Per-country welfare delta (effect of being taxed more)
        # For exporting countries: losing export market share proportional to tariff increase
        welfare_delta_countries = {}
        for iso3_upper, cidx in override_indices.items():
            dt = new_tariffs[cidx] - ustr_tariffs[cidx]
            share = float(import_shares[cidx])
            # Exporter welfare ~ -dt * share * eps (loses market access)
            welfare_delta_countries[iso3_upper] = round(-float(dt) * share * eps * 100, 4)

        # Baseline US welfare from USTR scenario (index 0)
        us_baseline_welfare = float(b["results"][id_US, COL["welfare"], 0])
        us_new_welfare = round(us_baseline_welfare + welfare_delta_US, 4)

        # Compile results
        if countries is None:
            countries = ["USA"] + list(tariff_overrides.keys())
        countries_upper = [c.upper() for c in countries]

        records = []
        for iso3 in countries_upper:
            match = cl[cl["iso3"] == iso3]
            country_name = match["CountryName"].iloc[0] if not match.empty else iso3
            if iso3 == "USA":
                records.append({
                    "iso3": "USA", "country": "United States",
                    "baseline_welfare_pct": us_baseline_welfare,
                    "welfare_delta_pct": round(welfare_delta_US, 4),
                    "new_welfare_pct": us_new_welfare,
                    "tariff_override": None,
                    "note": "Partial equilibrium approximation",
                })
            else:
                cidx = override_indices.get(iso3)
                override_rate = tariff_overrides.get(iso3, tariff_overrides.get(iso3.lower()))
                baseline_w = float(b["results"][match.index[0], COL["welfare"], 0]) if not match.empty else None
                delta_w    = welfare_delta_countries.get(iso3, 0.0)
                records.append({
                    "iso3": iso3, "country": country_name,
                    "baseline_welfare_pct": round(baseline_w, 4) if baseline_w is not None else None,
                    "welfare_delta_pct": delta_w,
                    "new_welfare_pct": round((baseline_w or 0) + delta_w, 4),
                    "tariff_override": override_rate,
                    "note": "Partial equilibrium approximation",
                })

        # Chart: before vs after welfare for selected countries
        country_names = [r["country"] for r in records]
        baseline_vals = [r["baseline_welfare_pct"] or 0 for r in records]
        new_vals      = [r["new_welfare_pct"] for r in records]

        traces = [
            {"type": "bar", "name": "USTR Baseline", "x": country_names, "y": baseline_vals,
             "marker": {"color": "#2563eb"}, "text": [f"{v:+.2f}%" for v in baseline_vals],
             "textposition": "outside"},
            {"type": "bar", "name": "Custom Scenario", "x": country_names, "y": new_vals,
             "marker": {"color": "#22d3a0"}, "text": [f"{v:+.2f}%" for v in new_vals],
             "textposition": "outside"},
        ]
        layout = {"title": "Welfare: USTR Baseline vs Custom Tariff Scenario",
                  "barmode": "group", "yaxis": {"title": "Welfare % Change"},
                  "legend": {"bgcolor": "#1a1d2e"}, "height": 380}

        return {"data": {"countries": records,
                         "tariff_overrides": tariff_overrides,
                         "us_welfare_delta": round(welfare_delta_US, 4),
                         "method": "Partial equilibrium linear approximation (ε=4)"},
                "chart_spec": _make_chart(traces, layout)}

    except Exception as e:
        return {"data": {"error": str(e)}, "chart_spec": {}}


# ─────────────────────────────────────────────────────────────────────────────
def get_tariff_rates(country: str = None, top_n: int = 25, sort_by: str = "tariff_desc") -> dict:
    """
    Return US Liberation Day tariff rates for all countries or a specific one.

    Parameters
    ----------
    country : str or None — ISO3 code (e.g. 'CHN') or None for all countries.
    top_n   : int — when country is None, return top N by tariff rate (default 25, max 194).
    sort_by : str — 'tariff_desc' (highest first), 'tariff_asc', or 'name'.

    Returns
    -------
    dict with keys: data (list of records), chart_spec (Plotly bar dict)
    """
    try:
        df = _tariff_country_df().copy()
        df["tariff_pct"] = (df["applied_tariff"] * 100).round(2)

        if country:
            iso = country.upper()
            row = df[df["iso3"] == iso]
            if row.empty:
                return {"data": {"error": f"Country '{iso}' not found."}, "chart_spec": {}}
            records = row[["iso3", "CountryName", "tariff_pct"]].to_dict("records")
        else:
            top_n = min(top_n, 194)
            if sort_by == "tariff_asc":
                df = df.nsmallest(top_n, "tariff_pct")
            elif sort_by == "name":
                df = df.sort_values("CountryName").head(top_n)
            else:
                df = df.nlargest(top_n, "tariff_pct")
            records = df[["iso3", "CountryName", "tariff_pct"]].to_dict("records")

        colors = ["#f87171" if r["tariff_pct"] >= 50 else
                  "#fb923c" if r["tariff_pct"] >= 25 else
                  "#fbbf24" if r["tariff_pct"] >= 10 else "#22d3a0"
                  for r in records]
        traces = [{"type": "bar", "x": [r["tariff_pct"] for r in records],
                   "y": [r["CountryName"] for r in records], "orientation": "h",
                   "marker": {"color": colors},
                   "text": [f"{r['tariff_pct']:.1f}%" for r in records],
                   "textposition": "outside"}]
        layout = {"title": "US Liberation Day Tariff Rates",
                  "xaxis": {"title": "Applied Tariff (%)"},
                  "yaxis": {"autorange": "reversed"}, "height": max(360, len(records) * 22)}
        return {"data": records, "chart_spec": _make_chart(traces, layout)}
    except Exception as e:
        return {"data": {"error": str(e)}, "chart_spec": {}}


# ─────────────────────────────────────────────────────────────────────────────
def get_country_profile(country: str, scenario: str = "ustr_no_retaliation") -> dict:
    """
    Full country profile: GE welfare/trade outcomes, tariff rate, GDP, and region.

    Parameters
    ----------
    country  : str — ISO3 code (e.g. 'CHN', 'DEU', 'IND').
    scenario : str — welfare scenario (default ustr_no_retaliation).

    Returns
    -------
    dict with keys: data (profile dict), chart_spec (radar-style bar dict)
    """
    try:
        iso = country.upper()
        b   = _baseline()
        cl  = b["country_labels"]
        match = cl[cl["iso3"] == iso]
        if match.empty:
            return {"data": {"error": f"Country '{iso}' not found."}, "chart_spec": {}}

        idx  = int(match.index[0])
        name = match["CountryName"].iloc[0]

        # GE results
        sc_idx = SCENARIO_MAP.get(scenario)
        if scenario == "flat_15pct":
            ge_row = b["results_15pct"][idx, :]
        elif sc_idx is not None:
            ge_row = b["results"][idx, :, sc_idx]
        else:
            return {"data": {"error": f"Unknown scenario '{scenario}'"}, "chart_spec": {}}

        metric_names = ["welfare_pct", "deficit_pct", "exports_gdp_pct",
                        "imports_gdp_pct", "employment_pct", "cpi_pct", "rev_gdp_pct"]

        # Tariff
        tdf = _tariff_country_df()
        tariff_row = tdf[tdf["iso3"] == iso]
        tariff_pct = float(tariff_row["applied_tariff"].iloc[0]) * 100 if not tariff_row.empty else None

        # GDP
        gdp_val = float(_gdp_df().loc[idx, "gdp"])

        # Welfare in absolute terms (GDP × welfare_pct / 100)
        welfare_pct = float(ge_row[0])
        welfare_abs_bn = round(gdp_val * welfare_pct / 100 / 1e9, 3)

        profile = {
            "iso3": iso,
            "country": name,
            "region": _REGION_MAP.get(iso, "Other"),
            "scenario": scenario,
            "tariff_pct": round(tariff_pct, 2) if tariff_pct is not None else None,
            "gdp_bn": round(gdp_val / 1e9, 2),
            "welfare_abs_bn": welfare_abs_bn,
        }
        for name_m, val in zip(metric_names, ge_row):
            profile[name_m] = round(float(val), 4)

        metrics_display = metric_names[:6]
        vals_display    = [profile[m] for m in metrics_display]
        bar_colors      = ["#22d3a0" if v >= 0 else "#f87171" for v in vals_display]
        traces = [{"type": "bar", "x": metrics_display, "y": vals_display,
                   "marker": {"color": bar_colors},
                   "text": [f"{v:+.3f}%" for v in vals_display],
                   "textposition": "outside"}]
        layout = {"title": f"{profile['country']} — Country Profile ({scenario})",
                  "yaxis": {"title": "% Change"}, "height": 380}

        return {"data": profile, "chart_spec": _make_chart(traces, layout)}
    except Exception as e:
        return {"data": {"error": str(e)}, "chart_spec": {}}


# ─────────────────────────────────────────────────────────────────────────────
def get_global_welfare_summary(top_n: int = 15, scenario: str = "ustr_no_retaliation") -> dict:
    """
    Return top N gainers and top N losers globally under a given scenario.

    Parameters
    ----------
    top_n    : int — number of gainers and losers to return (default 15).
    scenario : str — welfare scenario name.

    Returns
    -------
    dict with keys: data (gainers, losers, summary), chart_spec (horizontal bar dict)
    """
    try:
        b  = _baseline()
        cl = b["country_labels"]
        sc_idx = SCENARIO_MAP.get(scenario)

        if scenario == "flat_15pct":
            welfare_arr = b["results_15pct"][:, COL["welfare"]]
        elif sc_idx is not None:
            welfare_arr = b["results"][:, COL["welfare"], sc_idx]
        else:
            return {"data": {"error": f"Unknown scenario '{scenario}'"}, "chart_spec": {}}

        df = cl.copy()
        df["welfare_pct"] = welfare_arr
        df["region"] = df["iso3"].map(_REGION_MAP).fillna("Other")

        gainers = df.nlargest(top_n, "welfare_pct")[["iso3","CountryName","welfare_pct","region"]]
        losers  = df.nsmallest(top_n, "welfare_pct")[["iso3","CountryName","welfare_pct","region"]]

        n_gainers  = int((df["welfare_pct"] > 0).sum())
        n_losers   = int((df["welfare_pct"] < 0).sum())
        avg_welfare = float(df["welfare_pct"].mean())

        # Combined chart: bottom N losers → top N gainers
        plot_df = pd.concat([losers.iloc[::-1], gainers]).drop_duplicates("iso3")
        colors  = ["#22d3a0" if v >= 0 else "#f87171" for v in plot_df["welfare_pct"]]
        traces  = [{"type": "bar", "x": plot_df["welfare_pct"].tolist(),
                    "y": plot_df["CountryName"].tolist(), "orientation": "h",
                    "marker": {"color": colors},
                    "text": [f"{v:+.2f}%" for v in plot_df["welfare_pct"]],
                    "textposition": "outside"}]
        layout  = {"title": f"Global Welfare Winners & Losers — {scenario}",
                   "xaxis": {"title": "Welfare % Change"},
                   "yaxis": {"autorange": "reversed"}, "height": max(500, len(plot_df) * 22)}

        return {
            "data": {
                "gainers": gainers.to_dict("records"),
                "losers":  losers.to_dict("records"),
                "summary": {"n_gainers": n_gainers, "n_losers": n_losers,
                            "avg_welfare_pct": round(avg_welfare, 4)},
            },
            "chart_spec": _make_chart(traces, layout),
        }
    except Exception as e:
        return {"data": {"error": str(e)}, "chart_spec": {}}


# ─────────────────────────────────────────────────────────────────────────────
def get_regional_summary(scenario: str = "ustr_no_retaliation", metric: str = "welfare") -> dict:
    """
    Aggregate GE outcomes by world region (Asia-Pacific, Europe, Americas, MENA, Africa, South Asia).

    Parameters
    ----------
    scenario : str — welfare scenario name.
    metric   : str — one of 'welfare', 'cpi', 'exports', 'imports', 'employment'.

    Returns
    -------
    dict with keys: data (region records), chart_spec (bar dict)
    """
    try:
        b  = _baseline()
        cl = b["country_labels"]
        sc_idx = SCENARIO_MAP.get(scenario)

        metric_col = COL.get(metric)
        if metric_col is None:
            return {"data": {"error": f"Unknown metric '{metric}'. Valid: {list(COL)}"}, "chart_spec": {}}

        if scenario == "flat_15pct":
            arr = b["results_15pct"][:, metric_col]
        elif sc_idx is not None:
            arr = b["results"][:, metric_col, sc_idx]
        else:
            return {"data": {"error": f"Unknown scenario '{scenario}'"}, "chart_spec": {}}

        df = cl.copy()
        df["value"] = arr
        df["region"] = df["iso3"].map(_REGION_MAP).fillna("Other")
        df["gdp"]    = _gdp_df()["gdp"].values

        # GDP-weighted average per region
        def _wavg(grp):
            w = grp["gdp"]
            v = grp["value"]
            return float((v * w).sum() / w.sum()) if w.sum() > 0 else float(v.mean())

        agg = df.groupby("region").apply(_wavg).reset_index()
        agg.columns = ["region", "gdp_weighted_avg"]
        agg["n_countries"] = df.groupby("region")["iso3"].count().values
        agg["simple_mean"] = df.groupby("region")["value"].mean().values
        agg = agg.sort_values("gdp_weighted_avg", ascending=False)
        records = [{
            "region": r["region"],
            "gdp_weighted_avg_pct": round(r["gdp_weighted_avg"], 4),
            "simple_mean_pct": round(r["simple_mean"], 4),
            "n_countries": int(r["n_countries"]),
        } for _, r in agg.iterrows()]

        colors = ["#22d3a0" if r["gdp_weighted_avg_pct"] >= 0 else "#f87171" for r in records]
        traces = [{"type": "bar",
                   "x": [r["region"] for r in records],
                   "y": [r["gdp_weighted_avg_pct"] for r in records],
                   "marker": {"color": colors},
                   "text": [f"{r['gdp_weighted_avg_pct']:+.3f}%" for r in records],
                   "textposition": "outside"}]
        layout = {"title": f"Regional {metric.capitalize()} — {scenario} (GDP-weighted)",
                  "yaxis": {"title": f"{metric.capitalize()} % Change"}, "height": 380}

        return {"data": records, "chart_spec": _make_chart(traces, layout)}
    except Exception as e:
        return {"data": {"error": str(e)}, "chart_spec": {}}


# ─────────────────────────────────────────────────────────────────────────────
def compare_countries(countries: list, scenario: str = "ustr_no_retaliation") -> dict:
    """
    Side-by-side comparison of 2–10 countries across all GE metrics + tariff + GDP.

    Parameters
    ----------
    countries : list of ISO3 codes, e.g. ["CHN", "DEU", "IND", "MEX"].
    scenario  : str — welfare scenario name.

    Returns
    -------
    dict with keys: data (comparison table as list of metric dicts), chart_spec
    """
    try:
        b  = _baseline()
        cl = b["country_labels"]
        tdf = _tariff_country_df()
        gdp = _gdp_df()
        sc_idx = SCENARIO_MAP.get(scenario)

        countries_upper = [c.upper() for c in countries[:10]]
        metric_names = ["welfare_pct", "deficit_pct", "exports_gdp_pct",
                        "imports_gdp_pct", "employment_pct", "cpi_pct", "rev_gdp_pct"]

        rows = []
        for iso in countries_upper:
            match = cl[cl["iso3"] == iso]
            if match.empty:
                continue
            idx  = int(match.index[0])
            name = match["CountryName"].iloc[0]

            if scenario == "flat_15pct":
                ge_row = b["results_15pct"][idx, :]
            elif sc_idx is not None:
                ge_row = b["results"][idx, :, sc_idx]
            else:
                return {"data": {"error": f"Unknown scenario '{scenario}'"}, "chart_spec": {}}

            tariff_row = tdf[tdf["iso3"] == iso]
            tariff_pct = float(tariff_row["applied_tariff"].iloc[0]) * 100 if not tariff_row.empty else None
            gdp_val    = float(gdp.loc[idx, "gdp"]) if idx < len(gdp) else None

            row = {"iso3": iso, "country": name,
                   "tariff_pct": round(tariff_pct, 2) if tariff_pct is not None else None,
                   "gdp_bn": round(gdp_val / 1e9, 2) if gdp_val else None,
                   "region": _REGION_MAP.get(iso, "Other")}
            for m, v in zip(metric_names, ge_row):
                row[m] = round(float(v), 4)
            rows.append(row)

        if not rows:
            return {"data": {"error": "No valid countries found."}, "chart_spec": {}}

        # Grouped bar: one group per country, bars = welfare, cpi, employment
        key_metrics = ["welfare_pct", "cpi_pct", "employment_pct"]
        colors_map  = {"welfare_pct": "#2563eb", "cpi_pct": "#f87171", "employment_pct": "#22d3a0"}
        traces = []
        for m in key_metrics:
            traces.append({
                "type": "bar", "name": m,
                "x": [r["country"] for r in rows],
                "y": [r[m] for r in rows],
                "marker": {"color": colors_map[m]},
                "text": [f"{r[m]:+.2f}%" for r in rows],
                "textposition": "outside",
            })
        layout = {"title": f"Country Comparison — {scenario}", "barmode": "group",
                  "yaxis": {"title": "% Change"},
                  "legend": {"bgcolor": "#1a1d2e"}, "height": 400}

        return {"data": rows, "chart_spec": _make_chart(traces, layout)}
    except Exception as e:
        return {"data": {"error": str(e)}, "chart_spec": {}}


# ─────────────────────────────────────────────────────────────────────────────
def get_gdp_context(scenario: str = "ustr_no_retaliation", top_n: int = 20) -> dict:
    """
    Welfare losses/gains in absolute USD billions (welfare_pct × GDP).

    Parameters
    ----------
    scenario : str — welfare scenario name.
    top_n    : int — top N countries by absolute welfare loss (default 20).

    Returns
    -------
    dict with keys: data (records with absolute welfare), chart_spec
    """
    try:
        b  = _baseline()
        cl = b["country_labels"]
        gdp = _gdp_df()
        sc_idx = SCENARIO_MAP.get(scenario)

        if scenario == "flat_15pct":
            welfare_arr = b["results_15pct"][:, COL["welfare"]]
        elif sc_idx is not None:
            welfare_arr = b["results"][:, COL["welfare"], sc_idx]
        else:
            return {"data": {"error": f"Unknown scenario '{scenario}'"}, "chart_spec": {}}

        df = cl.copy()
        df["welfare_pct"] = welfare_arr
        df["gdp"]         = gdp["gdp"].values
        df["welfare_abs_bn"] = (df["welfare_pct"] / 100 * df["gdp"] / 1e9).round(3)
        df["region"]      = df["iso3"].map(_REGION_MAP).fillna("Other")

        biggest_losses = df.nsmallest(top_n, "welfare_abs_bn")
        total_global_loss_bn = float(df[df["welfare_abs_bn"] < 0]["welfare_abs_bn"].sum())
        total_global_gain_bn = float(df[df["welfare_abs_bn"] > 0]["welfare_abs_bn"].sum())

        records = biggest_losses[["iso3","CountryName","welfare_pct","welfare_abs_bn","gdp","region"]].copy()
        records["gdp_bn"] = (records["gdp"] / 1e9).round(2)
        records = records.drop(columns=["gdp"]).to_dict("records")

        colors = ["#22d3a0" if r["welfare_abs_bn"] >= 0 else "#f87171" for r in records]
        traces = [{"type": "bar", "x": [r["welfare_abs_bn"] for r in records],
                   "y": [r["CountryName"] for r in records], "orientation": "h",
                   "marker": {"color": colors},
                   "text": [f"${r['welfare_abs_bn']:.1f}B ({r['welfare_pct']:+.2f}%)" for r in records],
                   "textposition": "outside"}]
        layout = {"title": f"Top {top_n} Welfare Losses in Absolute $ — {scenario}",
                  "xaxis": {"title": "Welfare Change ($B)"},
                  "yaxis": {"autorange": "reversed"}, "height": max(400, top_n * 24)}

        return {
            "data": {
                "countries": records,
                "summary": {
                    "total_global_loss_bn": round(total_global_loss_bn, 2),
                    "total_global_gain_bn": round(total_global_gain_bn, 2),
                    "net_global_bn": round(total_global_loss_bn + total_global_gain_bn, 2),
                },
            },
            "chart_spec": _make_chart(traces, layout),
        }
    except Exception as e:
        return {"data": {"error": str(e)}, "chart_spec": {}}


# ─────────────────────────────────────────────────────────────────────────────
def get_trade_flows(exporter: str = None, importer: str = None, top_n: int = 15) -> dict:
    """
    Return bilateral trade flows from the CEPII trade matrix.

    Parameters
    ----------
    exporter : str or None — ISO3 of exporting country (rows). None = aggregate.
    importer : str or None — ISO3 of importing country (cols). None = aggregate.
    top_n    : int — top N trading partners to return (default 15).

    Returns
    -------
    dict with keys: data (trade flow records), chart_spec
    """
    try:
        b  = _baseline()
        cl = b["country_labels"]
        trade_path = os.path.join(DATA, "base_data", "trade_cepii.csv")
        X = pd.read_csv(trade_path, header=0).apply(pd.to_numeric, errors="coerce").fillna(0)
        X_mat = X.values  # (194, 194): X_mat[i, j] = exports from i to j

        if exporter and importer:
            # Single bilateral flow
            exp_iso = exporter.upper()
            imp_iso = importer.upper()
            exp_match = cl[cl["iso3"] == exp_iso]
            imp_match = cl[cl["iso3"] == imp_iso]
            if exp_match.empty or imp_match.empty:
                return {"data": {"error": "One or both countries not found."}, "chart_spec": {}}
            ei = int(exp_match.index[0])
            ii = int(imp_match.index[0])
            val = float(X_mat[ei, ii])
            return {
                "data": {"exporter": exp_iso, "importer": imp_iso,
                         "trade_value": round(val, 2), "unit": "model units"},
                "chart_spec": {},
            }

        elif exporter:
            iso = exporter.upper()
            match = cl[cl["iso3"] == iso]
            if match.empty:
                return {"data": {"error": f"Country '{iso}' not found."}, "chart_spec": {}}
            ei = int(match.index[0])
            row = X_mat[ei, :]
            top_idx = np.argsort(row)[-top_n:][::-1]
            records = [{"importer_iso3": cl.iloc[j]["iso3"],
                        "importer": cl.iloc[j]["CountryName"],
                        "trade_value": round(float(row[j]), 2)} for j in top_idx]
            y_key, x_key, title = "importer", "trade_value", f"Top {top_n} Import Destinations for {iso}"

        elif importer:
            iso = importer.upper()
            match = cl[cl["iso3"] == iso]
            if match.empty:
                return {"data": {"error": f"Country '{iso}' not found."}, "chart_spec": {}}
            ii = int(match.index[0])
            col = X_mat[:, ii]
            top_idx = np.argsort(col)[-top_n:][::-1]
            records = [{"exporter_iso3": cl.iloc[j]["iso3"],
                        "exporter": cl.iloc[j]["CountryName"],
                        "trade_value": round(float(col[j]), 2)} for j in top_idx]
            y_key, x_key, title = "exporter", "trade_value", f"Top {top_n} Exporters to {iso}"

        else:
            # Global top traders by total exports
            total_exports = X_mat.sum(axis=1)
            top_idx = np.argsort(total_exports)[-top_n:][::-1]
            records = [{"iso3": cl.iloc[j]["iso3"],
                        "country": cl.iloc[j]["CountryName"],
                        "total_exports": round(float(total_exports[j]), 2)} for j in top_idx]
            y_key, x_key, title = "country", "total_exports", f"Top {top_n} Exporters (Global)"

        traces = [{"type": "bar",
                   "x": [r[x_key] for r in records],
                   "y": [r[y_key] for r in records], "orientation": "h",
                   "marker": {"color": "#2563eb"},
                   "text": [str(round(r[x_key], 1)) for r in records],
                   "textposition": "outside"}]
        layout = {"title": title, "xaxis": {"title": "Trade Value (model units)"},
                  "yaxis": {"autorange": "reversed"}, "height": max(360, top_n * 24)}
        return {"data": records, "chart_spec": _make_chart(traces, layout)}
    except Exception as e:
        return {"data": {"error": str(e)}, "chart_spec": {}}


# ─────────────────────────────────────────────────────────────────────────────
def get_cavallo_price_data(country: str = None, window_days: int = 30) -> dict:
    """
    Return empirical daily price index data from Cavallo et al. for Canada, Mexico, China, USA.

    Parameters
    ----------
    country     : str or None — one of 'canada', 'mexico', 'china', 'usa' (case-insensitive).
                  None returns all four.
    window_days : int — last N days to highlight (default 30). Use 0 for full series.

    Returns
    -------
    dict with keys: data (records), chart_spec (line chart dict), summary (key stats)
    """
    try:
        df = _cavallo().copy()
        country_cols = {
            "canada": "index_canada", "mexico": "index_mexico",
            "china":  "index_china",  "usa":    "index_usa",
        }

        if country:
            ckey = country.lower()
            if ckey not in country_cols:
                return {"data": {"error": f"Country must be one of: {list(country_cols)}"}, "chart_spec": {}}
            cols = [country_cols[ckey]]
            display = [ckey.capitalize()]
        else:
            cols    = list(country_cols.values())
            display = [c.capitalize() for c in country_cols]

        if window_days > 0:
            df = df.tail(window_days)

        records = df[["date"] + cols].to_dict("records")

        # Summary stats
        summary = {}
        for col, label in zip(cols, display):
            series = df[col].dropna()
            first_val = float(series.iloc[0])
            last_val  = float(series.iloc[-1])
            summary[label] = {
                "start_index": round(first_val, 6),
                "end_index":   round(last_val, 6),
                "total_change_pct": round((last_val / first_val - 1) * 100, 4),
                "max_index": round(float(series.max()), 6),
                "min_index": round(float(series.min()), 6),
            }

        # Line chart
        colors_palette = ["#2563eb", "#22d3a0", "#f87171", "#fbbf24"]
        traces = []
        for i, (col, label) in enumerate(zip(cols, display)):
            traces.append({
                "type": "scatter", "mode": "lines", "name": label,
                "x": df["date"].tolist(), "y": df[col].tolist(),
                "line": {"color": colors_palette[i % len(colors_palette)], "width": 2},
            })
        window_label = f"Last {window_days} Days" if window_days > 0 else "Full Series"
        layout = {"title": f"Cavallo Price Indices — {window_label}",
                  "xaxis": {"title": "Date"}, "yaxis": {"title": "Price Index (Oct 2024 = 1.0)"},
                  "legend": {"bgcolor": "#1a1d2e"}, "height": 380}

        return {"data": records, "summary": summary, "chart_spec": _make_chart(traces, layout)}
    except Exception as e:
        return {"data": {"error": str(e)}, "chart_spec": {}}


# ─────────────────────────────────────────────────────────────────────────────
def get_retail_impact() -> dict:
    """
    Return retail-sector GE results: CPI impact, welfare, quintile incidence, and Cavallo empirical stats.

    No parameters required.

    Returns
    -------
    dict with keys: data (all retail metrics), chart_spec (grouped bar dict)
    """
    try:
        r = _retail_npz()

        ge_cpi_noretal   = float(r["ge_cpi_noretal"])
        ge_cpi_retal     = float(r["ge_cpi_retal"])
        ge_welfare_noretal = float(r["ge_welfare_noretal"])
        ge_welfare_retal   = float(r["ge_welfare_retal"])
        first_order_cpi  = float(r["first_order_cpi"])
        ge_amplification = float(r["ge_amplification"])
        regress_ratio    = float(r["regress_ratio"])
        q_noretal = r["quintile_incidence_noretal"].tolist()
        q_retal   = r["quintile_incidence_retal"].tolist()

        cavallo_summary = {}
        for k in ["cavallo_usa_30d", "cavallo_usa_90d", "cavallo_china_90d"]:
            if k in r:
                cavallo_summary[k] = round(float(r[k]), 6)

        retail_passthrough = None
        if "retail_product_passthrough" in r:
            pt = r["retail_product_passthrough"]
            if hasattr(pt, "tolist"):
                retail_passthrough = pt.tolist()
            else:
                retail_passthrough = float(pt)

        quintile_labels = ["Q1 (Poorest)", "Q2", "Q3", "Q4", "Q5 (Richest)"]
        data = {
            "cpi": {
                "first_order_pct": round(first_order_cpi, 4),
                "ge_no_retaliation_pct": round(ge_cpi_noretal, 4),
                "ge_retaliation_pct": round(ge_cpi_retal, 4),
                "ge_amplification_factor": round(ge_amplification, 4),
            },
            "welfare": {
                "ge_no_retaliation_pct": round(ge_welfare_noretal, 4),
                "ge_retaliation_pct": round(ge_welfare_retal, 4),
            },
            "quintile_incidence_noretal": [
                {"quintile": q, "burden_pct": round(v, 4)}
                for q, v in zip(quintile_labels, q_noretal)
            ],
            "quintile_incidence_retal": [
                {"quintile": q, "burden_pct": round(v, 4)}
                for q, v in zip(quintile_labels, q_retal)
            ],
            "regressivity_ratio_noretal": round(regress_ratio, 4),
            "cavallo_empirical": cavallo_summary,
            "retail_passthrough": retail_passthrough,
        }

        # Chart: quintile incidence comparison
        colors_no   = ["#2563eb"] * 5
        colors_ret  = ["#f87171"] * 5
        traces = [
            {"type": "bar", "name": "No Retaliation", "x": quintile_labels, "y": q_noretal,
             "marker": {"color": colors_no},
             "text": [f"{v:.4f}%" for v in q_noretal], "textposition": "outside"},
            {"type": "bar", "name": "With Retaliation", "x": quintile_labels, "y": q_retal,
             "marker": {"color": colors_ret},
             "text": [f"{v:.4f}%" for v in q_retal], "textposition": "outside"},
        ]
        layout = {"title": "Retail Tariff Incidence by Quintile",
                  "barmode": "group", "yaxis": {"title": "Burden (% of income)"},
                  "legend": {"bgcolor": "#1a1d2e"}, "height": 380}

        return {"data": data, "chart_spec": _make_chart(traces, layout)}
    except Exception as e:
        return {"data": {"error": str(e)}, "chart_spec": {}}


# ─────────────────────────────────────────────────────────────────────────────
def get_sector_tariff_shocks(scenario: str = None, sector: str = None) -> dict:
    """
    Return tariff shock rates by model sector across all policy scenarios.

    Parameters
    ----------
    scenario : str or None — filter to one scenario. Valid options:
               'baseline_no_tariffs', 'liberation_day_schedule',
               'optimal_uniform_19', 'industry_focused', 'supply_chain_disruption'.
               None returns all scenarios.
    sector   : str or None — filter to one sector. Valid options:
               'steel_aluminum', 'pharma', 'retail_consumer',
               'manufacturing_other', 'services_other', 'energy_primary'.
               None returns all sectors.

    Returns
    -------
    dict with keys: data (shock records), chart_spec (grouped bar dict)
    """
    try:
        m = _manufacturing()
        shocks = m["shocks"].copy()

        if scenario:
            shocks = shocks[shocks["scenario"] == scenario]
            if shocks.empty:
                valid = m["shocks"]["scenario"].unique().tolist()
                return {"data": {"error": f"Unknown scenario '{scenario}'. Valid: {valid}"}, "chart_spec": {}}
        if sector:
            shocks = shocks[shocks["model_sector"] == sector]
            if shocks.empty:
                valid = m["shocks"]["model_sector"].unique().tolist()
                return {"data": {"error": f"Unknown sector '{sector}'. Valid: {valid}"}, "chart_spec": {}}

        shocks["tariff_pct"] = (shocks["tariff_rate"] * 100).round(3)
        records = shocks[["scenario", "model_sector", "tariff_pct"]].to_dict("records")

        # Pivot for chart: x=sector, bars grouped by scenario
        scenarios = shocks["scenario"].unique().tolist()
        sectors   = shocks["model_sector"].unique().tolist()
        colors_palette = ["#2563eb", "#22d3a0", "#fbbf24", "#f87171", "#a78bfa"]
        traces = []
        for i, sc in enumerate(scenarios):
            sc_data = shocks[shocks["scenario"] == sc]
            sc_map  = sc_data.set_index("model_sector")["tariff_pct"].to_dict()
            traces.append({
                "type": "bar", "name": sc,
                "x": sectors,
                "y": [sc_map.get(s, 0) for s in sectors],
                "marker": {"color": colors_palette[i % len(colors_palette)]},
            })
        layout = {"title": "Tariff Shock Rates by Sector & Scenario",
                  "barmode": "group",
                  "xaxis": {"title": "Model Sector"},
                  "yaxis": {"title": "Tariff Rate (%)"},
                  "legend": {"bgcolor": "#1a1d2e"}, "height": 400}

        return {"data": records, "chart_spec": _make_chart(traces, layout)}
    except Exception as e:
        return {"data": {"error": str(e)}, "chart_spec": {}}


# ─────────────────────────────────────────────────────────────────────────────
def get_manufacturing_stats() -> dict:
    """
    Return key manufacturing sector statistics: average tariff, import penetration,
    CPI contribution, IO multipliers, import share, and pre-Liberation Day HTS8 rates.

    No parameters required.

    Returns
    -------
    dict with keys: data (all manufacturing stats), chart_spec (bar comparison dict)
    """
    try:
        m = _manufacturing()
        s = m["mfg_stats"]

        data = {
            "tariffs": {
                "avg_liberation_day_tariff_pct": round(s.get("tau_mfg_avg", 0) * 100, 3),
                "hts8_pre_liberation_mfg_rate_pct": round(s.get("hts8_mfg_rate", 0) * 100, 3),
                "hts8_pre_liberation_steel_rate_pct": round(s.get("hts8_steel_rate", 0) * 100, 3),
            },
            "trade": {
                "import_penetration_pct": round(s.get("import_penetration_mfg", 0) * 100, 3),
                "import_share_mfg_pct": round(s.get("imp_share_mfg", 0) * 100, 3),
                "import_share_steel_pct": round(s.get("imp_share_steel", 0) * 100, 3),
                "mfg_import_change_pct": round(s.get("mfg_import_change", 0), 3),
            },
            "macro_impact": {
                "cpi_manufacturing_contribution_pp": round(s.get("cpi_mfg_contribution", 0), 4),
                "io_multiplier_manufacturing": round(s.get("io_mult_mfg", 0), 4),
                "io_multiplier_steel": round(s.get("io_mult_steel", 0), 4),
            },
        }

        # HTS product-level: top 20 highest MFN tariff manufacturing lines (exclude specific-rate placeholders)
        hts = m["hts"].copy()
        mfg_hts = hts[(hts["mfn_rate"] > 0) & (hts["mfn_rate"] < 5)].nlargest(20, "mfn_rate")[
            ["hts8", "brief_description", "mfn_text_rate", "mfn_rate"]
        ].copy()
        mfg_hts["mfn_rate_pct"] = (mfg_hts["mfn_rate"] * 100).round(2)
        data["top_tariffed_hts_lines"] = mfg_hts[
            ["hts8", "brief_description", "mfn_text_rate", "mfn_rate_pct"]
        ].to_dict("records")

        # Chart: key metrics bar
        labels = ["Avg Tariff", "Import Penetration", "Import Share", "CPI Contribution (pp)",
                  "IO Multiplier Mfg", "IO Multiplier Steel"]
        values = [
            data["tariffs"]["avg_liberation_day_tariff_pct"],
            data["trade"]["import_penetration_pct"],
            data["trade"]["import_share_mfg_pct"],
            data["macro_impact"]["cpi_manufacturing_contribution_pp"],
            data["macro_impact"]["io_multiplier_manufacturing"],
            data["macro_impact"]["io_multiplier_steel"],
        ]
        bar_colors = ["#f87171","#fbbf24","#fb923c","#2563eb","#22d3a0","#a78bfa"]
        traces = [{"type": "bar", "x": labels, "y": values,
                   "marker": {"color": bar_colors},
                   "text": [str(round(v, 3)) for v in values],
                   "textposition": "outside"}]
        layout = {"title": "Manufacturing Sector — Key Tariff & Trade Statistics",
                  "yaxis": {"title": "Value (% or multiplier)"}, "height": 400,
                  "xaxis": {"tickangle": -20}}

        return {"data": data, "chart_spec": _make_chart(traces, layout)}
    except Exception as e:
        return {"data": {"error": str(e)}, "chart_spec": {}}


# ─────────────────────────────────────────────────────────────────────────────
def get_hts_tariff_lookup(keyword: str = None, hts_prefix: str = None, top_n: int = 20) -> dict:
    """
    Search the US HTS8 tariff schedule by product keyword or HTS code prefix.

    Parameters
    ----------
    keyword    : str or None — search term in product description (e.g. 'steel', 'semiconductor').
    hts_prefix : str or None — HTS chapter/heading prefix (e.g. '72' for steel, '8541' for semiconductors).
    top_n      : int — max results to return (default 20).

    Returns
    -------
    dict with keys: data (matching HTS lines), chart_spec (bar of tariff rates)
    """
    try:
        m   = _manufacturing()
        hts = m["hts"].copy()
        hts["hts8_str"] = hts["hts8"].astype(str).str.zfill(8)

        if keyword:
            mask = hts["brief_description"].str.contains(keyword, case=False, na=False)
            hts  = hts[mask]
        if hts_prefix:
            hts = hts[hts["hts8_str"].str.startswith(str(hts_prefix))]

        # Filter out specific-rate placeholders (coded as very large numbers)
        hts = hts[hts["mfn_rate"] < 5]

        if hts.empty:
            return {"data": {"error": "No HTS lines matched the query."}, "chart_spec": {}}

        hts = hts.nlargest(top_n, "mfn_rate")[
            ["hts8_str", "brief_description", "mfn_text_rate", "mfn_rate"]
        ].copy()
        hts["mfn_rate_pct"] = (hts["mfn_rate"] * 100).round(2)
        records = hts.rename(columns={"hts8_str": "hts8"}).to_dict("records")

        colors = ["#f87171" if r["mfn_rate_pct"] >= 20 else
                  "#fbbf24" if r["mfn_rate_pct"] >= 10 else "#2563eb"
                  for r in records]
        traces = [{"type": "bar",
                   "x": [r["mfn_rate_pct"] for r in records],
                   "y": [r["brief_description"][:45] for r in records],
                   "orientation": "h",
                   "marker": {"color": colors},
                   "text": [f"{r['mfn_text_rate']} (HTS {r['hts8']})" for r in records],
                   "textposition": "outside"}]
        query_label = f"'{keyword}'" if keyword else f"prefix {hts_prefix}"
        layout = {"title": f"HTS Tariff Schedule — {query_label}",
                  "xaxis": {"title": "MFN Ad Valorem Rate (%)"},
                  "yaxis": {"autorange": "reversed"},
                  "height": max(360, len(records) * 26)}

        return {"data": records, "chart_spec": _make_chart(traces, layout)}
    except Exception as e:
        return {"data": {"error": str(e)}, "chart_spec": {}}


# ── Entry point ───────────────────────────────────────────────────────────────
# FastMCP is only imported here so that importing this module in the dashboard
# does NOT trigger the FastMCP server runtime.
if __name__ == "__main__":
    from fastmcp import FastMCP
    mcp = FastMCP(
        name="liberation-day-tariffs",
        instructions=(
            "Tools for analysing the April 2025 US Liberation Day tariffs via a "
            "194-country General Equilibrium model, pharma supply chain data, "
            "retail incidence, and manufacturing sector shocks."
        ),
    )
    mcp.tool()(get_welfare_results)
    mcp.tool()(get_scenario_comparison)
    mcp.tool()(get_pharma_supplier_risk)
    mcp.tool()(get_quintile_burden)
    mcp.tool()(get_manufacturing_shock)
    mcp.tool()(run_tariff_scenario)
    mcp.tool()(get_tariff_rates)
    mcp.tool()(get_country_profile)
    mcp.tool()(get_global_welfare_summary)
    mcp.tool()(get_regional_summary)
    mcp.tool()(compare_countries)
    mcp.tool()(get_gdp_context)
    mcp.tool()(get_trade_flows)
    mcp.tool()(get_cavallo_price_data)
    mcp.tool()(get_retail_impact)
    mcp.tool()(get_sector_tariff_shocks)
    mcp.tool()(get_manufacturing_stats)
    mcp.tool()(get_hts_tariff_lookup)
    mcp.run()
