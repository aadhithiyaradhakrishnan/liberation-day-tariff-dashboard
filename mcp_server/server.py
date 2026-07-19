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
        r = np.load(os.path.join(OUT, "sector_retail_results.npz"), allow_pickle=True)
        retail_stats = {k: (float(r[k]) if r[k].ndim == 0 else r[k].tolist()) for k in r.keys()}
        _cache["quintile"] = dict(burd=burd, retail_stats=retail_stats)
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

def _measured():
    """Post-Liberation Day measured datasets: USITC monthly imports + calculated
    duties by HTS chapter (2022-2025), BEA quarterly output (2024Q1-2026Q1),
    and official BLS CPI series (2022 - May 2026)."""
    if "measured" not in _cache:
        import json as _json
        months = ["January","February","March","April","May","June",
                  "July","August","September","October","November","December"]

        def _long(sheet):
            df = pd.read_excel(os.path.join(DATA, "USITC_Mfg_Imports_Monthly_2022_2025.xlsx"),
                               sheet_name=sheet, header=2)
            df.columns = [str(c).strip() for c in df.columns]
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
            df["HTS Number"] = pd.to_numeric(df["HTS Number"], errors="coerce")
            df = df.dropna(subset=["Year", "HTS Number"])
            rows = []
            for _, r in df.iterrows():
                for mi, m in enumerate(months, 1):
                    v = pd.to_numeric(r[m], errors="coerce")
                    if pd.notna(v) and v > 0:
                        rows.append({"chapter": int(r["HTS Number"]),
                                     "date": pd.Timestamp(int(r["Year"]), mi, 1),
                                     "value": float(v)})
            return pd.DataFrame(rows)

        imp = _long("Customs Value")
        dut = _long("Calculated Duties")

        bea = pd.read_excel(os.path.join(DATA, "BEA_Gross_Output_by_Industry_latest.xlsx"), header=None)
        bea_sub = bea.iloc[7:45, :11].copy()
        bea_sub.columns = ["line", "industry", "2024Q1", "2024Q2", "2024Q3", "2024Q4",
                           "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1"]
        bea_sub["industry"] = bea_sub["industry"].astype(str).str.strip()
        for c in bea_sub.columns[2:]:
            bea_sub[c] = pd.to_numeric(bea_sub[c], errors="coerce")

        with open(os.path.join(DATA, "BLS_Retail_CPI_Monthly_2022_2026.json"), encoding="utf-8") as f:
            raw = _json.load(f)
        bls_names = {
            "CUSR0000SA0":    ("All items (headline CPI)", "headline"),
            "CUSR0000SAF11":  ("Food at home",             "exposed"),
            "CUSR0000SAA":    ("Apparel",                  "exposed"),
            "CUSR0000SAH3":   ("Household furnishings",    "exposed"),
            "CUSR0000SETA01": ("New vehicles",             "exposed"),
            "CUSR0000SETA02": ("Used cars & trucks",       "exposed"),
            "CUSR0000SEEE01": ("Computers & peripherals",  "exposed"),
            "CUSR0000SERA02": ("Cable & streaming TV",     "service"),
            "CUSR0000SAM2":   ("Medical care services",    "service"),
            "CUSR0000SAS4":   ("Transportation services",  "service"),
        }
        brows = []
        for s in raw["Results"]["series"]:
            sid = s["seriesID"]
            if sid not in bls_names:
                continue
            for p in s["data"]:
                if p["value"] == "-":
                    continue
                brows.append({"sid": sid, "name": bls_names[sid][0],
                              "group": bls_names[sid][1],
                              "date": pd.Timestamp(int(p["year"]), int(p["period"][1:]), 1),
                              "value": float(p["value"])})
        bls = pd.DataFrame(brows).sort_values(["sid", "date"]).reset_index(drop=True)

        _cache["measured"] = dict(imp=imp, dut=dut, bea=bea_sub, bls=bls)
    return _cache["measured"]


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
    category : str or None — pass any value (e.g. 'retail') for the retail GE
               quintile incidence (from sector_retail_results.npz);
               None returns the pharma quintile burden.

    Returns
    -------
    dict with keys: data (quintile records), chart_spec (Plotly bar figure dict)
    """
    try:
        q = _quintile()

        if category:
            rs = q["retail_stats"]
            q_no = rs["quintile_incidence_noretal"]
            q_re = rs["quintile_incidence_retal"]
            quintile_names = ["Q1 (Poorest)", "Q2", "Q3", "Q4", "Q5 (Richest)"]
            records = []
            for name, v_no, v_re in zip(quintile_names, q_no, q_re):
                records.append({
                    "quintile": name,
                    "burden_no_retaliation_pct": round(float(v_no), 2),
                    "burden_with_retaliation_pct": round(float(v_re), 2),
                    "adjusted_burden_pct": round(float(v_no), 2),
                })
            chart_title = "Retail Consumer Price Burden by Income Quintile (GE model)"
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
def get_manufacturing_reality(chapter: int = None) -> dict:
    """
    Measured post-Liberation Day manufacturing outcomes from USITC customs data
    and BEA quarterly output (NOT model predictions).

    Parameters
    ----------
    chapter : int or None — HTS chapter (39, 72, 73, 84, 85, 87, 94, 95).
              If given, returns that chapter's monthly import series.
              None returns the cross-chapter summary: import change and
              effective duty rate per chapter (Apr-Dec 2025 vs Apr-Dec 2024),
              plus aggregate effective-rate and BEA output changes.

    Returns
    -------
    dict with keys: data, chart_spec
    """
    try:
        m = _measured()
        imp, dut, bea = m["imp"], m["dut"], m["bea"]
        names = {39: "Plastics", 72: "Iron & Steel", 73: "Steel Articles",
                 84: "Machinery", 85: "Electronics", 87: "Vehicles",
                 94: "Furniture", 95: "Toys & Sports"}

        if chapter is not None:
            sub = imp[imp["chapter"] == int(chapter)].sort_values("date")
            if sub.empty:
                return {"data": {"error": f"Unknown chapter {chapter}. Valid: {sorted(names)}"}, "chart_spec": {}}
            recs = [{"month": d.strftime("%Y-%m"), "imports_usd": v}
                    for d, v in zip(sub["date"], sub["value"])]
            traces = [{"type": "scatter", "x": [r["month"] for r in recs],
                       "y": [r["imports_usd"] / 1e9 for r in recs],
                       "line": {"color": "#2563eb", "width": 2}}]
            layout = {"title": f"Monthly US Imports — {names[int(chapter)]} (USITC, $B)",
                      "yaxis": {"title": "$B/month"}, "height": 360}
            return {"data": {"chapter": int(chapter), "name": names[int(chapter)],
                             "monthly": recs},
                    "chart_spec": _make_chart(traces, layout)}

        pre_i = imp[(imp["date"] >= "2024-04-01") & (imp["date"] <= "2024-12-01")]
        post_i = imp[(imp["date"] >= "2025-04-01") & (imp["date"] <= "2025-12-01")]
        pre_d = dut[(dut["date"] >= "2024-04-01") & (dut["date"] <= "2024-12-01")]
        post_d = dut[(dut["date"] >= "2025-04-01") & (dut["date"] <= "2025-12-01")]

        records = []
        for ch in sorted(names):
            v1 = float(pre_i[pre_i["chapter"] == ch]["value"].sum())
            v2 = float(post_i[post_i["chapter"] == ch]["value"].sum())
            d1 = float(pre_d[pre_d["chapter"] == ch]["value"].sum())
            d2 = float(post_d[post_d["chapter"] == ch]["value"].sum())
            if v1 <= 0 or v2 <= 0:
                continue
            records.append({
                "chapter": ch, "name": names[ch],
                "import_change_pct": round((v2 / v1 - 1) * 100, 1),
                "imports_2024_bn": round(v1 / 1e9, 1),
                "imports_2025_bn": round(v2 / 1e9, 1),
                "effective_rate_2024_pct": round(d1 / v1 * 100, 1),
                "effective_rate_2025_pct": round(d2 / v2 * 100, 1),
                "calculated_duties_2025_bn": round(d2 / 1e9, 1),
            })

        tot1, tot2 = float(pre_i["value"].sum()), float(post_i["value"].sum())
        dtot1, dtot2 = float(pre_d["value"].sum()), float(post_d["value"].sum())
        bea_mfg = bea[bea["industry"] == "Manufacturing"]
        bea_chg = (float(bea_mfg["2026Q1"].iloc[0] / bea_mfg["2025Q1"].iloc[0] - 1) * 100
                   if not bea_mfg.empty else None)
        summary = {
            "total_import_change_pct": round((tot2 / tot1 - 1) * 100, 1),
            "duty_multiple_vs_prior_year": round(dtot2 / max(dtot1, 1), 1),
            "calculated_duties_2025_bn": round(dtot2 / 1e9, 1),
            "effective_rate_2024_pct": round(dtot1 / tot1 * 100, 1),
            "effective_rate_2025_pct": round(dtot2 / tot2 * 100, 1),
            "bea_nominal_output_change_2025Q1_2026Q1_pct": round(bea_chg, 1) if bea_chg is not None else None,
            "period": "Apr-Dec 2025 vs Apr-Dec 2024; USITC calculated duties, 8 mfg chapters",
        }

        srt = sorted(records, key=lambda r: r["import_change_pct"])
        traces = [{"type": "bar", "orientation": "h",
                   "x": [r["import_change_pct"] for r in srt],
                   "y": [r["name"] for r in srt],
                   "marker": {"color": ["#f87171" if r["import_change_pct"] < 0 else "#22d3a0" for r in srt]},
                   "text": [f"{r['import_change_pct']:+.1f}%" for r in srt],
                   "textposition": "outside"}]
        layout = {"title": "Measured Import Change by Product (Apr-Dec 2025 vs 2024)",
                  "xaxis": {"title": "% change"}, "height": 380}
        return {"data": {"chapters": records, "summary": summary},
                "chart_spec": _make_chart(traces, layout)}
    except Exception as e:
        return {"data": {"error": str(e)}, "chart_spec": {}}


# ─────────────────────────────────────────────────────────────────────────────
def get_retail_price_reality(series_name: str = None) -> dict:
    """
    Official BLS consumer price outcomes after Liberation Day (measured).

    Parameters
    ----------
    series_name : str or None — category name filter, e.g. 'Apparel',
                  'Computers & peripherals'. None returns all categories.

    Returns
    -------
    dict with per-category price change since Mar 2025 and inflation
    acceleration vs the pre-tariff trend; chart_spec bar chart.
    """
    try:
        bls = _measured()["bls"]
        ld = pd.Timestamp("2025-03-01")
        pre0 = pd.Timestamp("2023-01-01")
        latest = bls["date"].max()

        records = []
        for sid in bls["sid"].unique():
            s = bls[bls["sid"] == sid].set_index("date")["value"]
            row0 = bls[bls["sid"] == sid].iloc[0]
            if ld not in s.index or latest not in s.index or pre0 not in s.index:
                continue
            m_pre = (ld.year - pre0.year) * 12 + (ld.month - pre0.month)
            m_post = (latest.year - ld.year) * 12 + (latest.month - ld.month)
            pre_ann = ((s[ld] / s[pre0]) ** (12 / m_pre) - 1) * 100
            post_ann = ((s[latest] / s[ld]) ** (12 / m_post) - 1) * 100
            records.append({
                "category": row0["name"], "group": row0["group"],
                "price_change_since_LD_pct": round(float((s[latest] / s[ld] - 1) * 100), 1),
                "inflation_before_pct_yr": round(float(pre_ann), 1),
                "inflation_after_pct_yr": round(float(post_ann), 1),
                "acceleration_pp": round(float(post_ann - pre_ann), 1),
            })

        if series_name:
            match = [r for r in records if series_name.lower() in r["category"].lower()]
            if not match:
                return {"data": {"error": f"No category matching '{series_name}'. Valid: {[r['category'] for r in records]}"}, "chart_spec": {}}
            records = match

        srt = sorted(records, key=lambda r: r["price_change_since_LD_pct"], reverse=True)
        colors = {"exposed": "#f87171", "headline": "#e2e8f0", "service": "#38bdf8"}
        traces = [{"type": "bar", "orientation": "h",
                   "x": [r["price_change_since_LD_pct"] for r in srt],
                   "y": [r["category"] for r in srt],
                   "marker": {"color": [colors[r["group"]] for r in srt]},
                   "text": [f"{r['price_change_since_LD_pct']:+.1f}%" for r in srt],
                   "textposition": "outside"}]
        layout = {"title": "Price Change Since Liberation Day (BLS, Mar 2025 to latest)",
                  "xaxis": {"title": "% change"}, "height": 380}
        summary = {
            "window": f"Mar 2025 to {latest.strftime('%b %Y')}",
            "note": "red = tariff-exposed goods, blue = domestic services (control), white = headline CPI",
        }
        return {"data": {"categories": records, "summary": summary},
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
            "retail incidence, manufacturing sector shocks, and measured post-tariff "
            "outcomes from USITC customs, BEA output and BLS price data."
        ),
    )
    mcp.tool()(get_welfare_results)
    mcp.tool()(get_scenario_comparison)
    mcp.tool()(get_pharma_supplier_risk)
    mcp.tool()(get_quintile_burden)
    mcp.tool()(get_manufacturing_shock)
    mcp.tool()(get_manufacturing_reality)
    mcp.tool()(get_retail_price_reality)
    mcp.tool()(run_tariff_scenario)
    mcp.run()
