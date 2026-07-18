import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# ── Auto-download large files from Hugging Face if missing ───────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_LARGE_FILES = [
    os.path.join(_ROOT, "data", "processed", "icio_2022", "io_coeff_matrix.npy"),
    os.path.join(_ROOT, "data", "processed", "icio_2022", "io_intermediate_matrix.npy"),
    os.path.join(_ROOT, "data", "code_and_release_data", "301 model", "D_all_data.zip"),
]
if any(not os.path.exists(f) for f in _LARGE_FILES):
    with st.spinner("Downloading large data files from Hugging Face (first run only)..."):
        import download_data
        download_data.main()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Liberation Day Tariff Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Font & background */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background-color: #0f1117; }

  /* Hide streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #1a1d2e;
    padding: 8px 12px;
    border-radius: 12px;
    border-bottom: none;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #8b93a7;
    font-weight: 500;
    font-size: 14px;
    padding: 8px 20px;
    border: none;
  }
  .stTabs [aria-selected="true"] {
    background: #2563eb !important;
    color: white !important;
  }

  /* KPI card */
  .kpi-card {
    background: linear-gradient(135deg, #1e2235 0%, #252a40 100%);
    border: 1px solid #2d3250;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
  }
  .kpi-label {
    font-size: 12px;
    font-weight: 600;
    color: #8b93a7;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
  }
  .kpi-value {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 4px;
    line-height: 1.1;
  }
  .kpi-sub {
    font-size: 12px;
    color: #6b7280;
  }
  .positive { color: #22d3a0; }
  .negative { color: #f87171; }
  .warning  { color: #fbbf24; }
  .neutral  { color: #60a5fa; }

  /* Section header */
  .section-header {
    font-size: 18px;
    font-weight: 600;
    color: #e2e8f0;
    margin: 24px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #2d3250;
  }

  /* Page title */
  .page-title {
    font-size: 28px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 4px;
  }
  .page-subtitle {
    font-size: 14px;
    color: #64748b;
    margin-bottom: 24px;
  }

  /* Insight box */
  .insight-box {
    background: #1e2235;
    border-left: 3px solid #2563eb;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #94a3b8;
    margin: 8px 0 16px 0;
  }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "python_output")

PLOTLY_THEME = dict(
    paper_bgcolor="#0f1117",
    plot_bgcolor="#0f1117",
    font=dict(family="Inter", color="#cbd5e1"),
    xaxis=dict(gridcolor="#1e2235", linecolor="#2d3250", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#1e2235", linecolor="#2d3250", tickfont=dict(size=11)),
    margin=dict(l=20, r=20, t=40, b=20),
    colorway=["#2563eb","#22d3a0","#f87171","#fbbf24","#a78bfa","#fb923c","#38bdf8"],
)

def apply_theme(fig, height=380):
    fig.update_layout(**PLOTLY_THEME, height=height)
    return fig

def _explain(text):
    """Plain-English caption rendered directly under a chart."""
    st.markdown(
        f'<div style="background:#141824;border-left:2px solid #475569;border-radius:0 6px 6px 0;'
        f'padding:8px 14px;margin:-6px 0 18px 0;font-size:12.5px;color:#8b93a7;line-height:1.5">'
        f'📖 {text}</div>', unsafe_allow_html=True)

# ── Data loaders (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_baseline():
    b = np.load(os.path.join(OUT, "baseline_results.npz"), allow_pickle=True)
    cl = pd.read_csv(os.path.join(DATA, "base_data", "country_labels.csv"))
    results = b["results"]          # (194, 7, 9)
    Y_i     = b["Y_i"]             # (194,)
    id_US   = int(b["id_US"])
    # Load 15% flat tariff scenario
    sc15 = np.load(os.path.join(OUT, "scenario_15pct.npz"), allow_pickle=True)
    results_15pct = sc15["results"]   # (194, 7)
    d_trade_15pct = float(sc15["d_trade"])
    return results, Y_i, id_US, cl, results_15pct, d_trade_15pct

@st.cache_data
def load_pharma1():
    xl = pd.ExcelFile(os.path.join(DATA, "Pharma1.xlsx"))
    df = xl.parse("Query Results", header=2)
    monthly = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
    df["annual_total"] = df[monthly].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["HTS Number"] = pd.to_numeric(df["HTS Number"], errors="coerce")
    return df, monthly

@st.cache_data
def load_pharma_outputs():
    dep  = pd.read_csv(os.path.join(OUT, "pharma1_objective1_dependence_2025.csv"))
    src  = pd.read_csv(os.path.join(OUT, "pharma1_objective2_sourcing_shifts_2025.csv"))
    burd = pd.read_csv(os.path.join(OUT, "pharma1_objective3_consumer_burden_2025.csv"))
    exp  = pd.read_csv(os.path.join(OUT, "pharma1_country_exposure_2024.csv"))
    return dep, src, burd, exp

@st.cache_data
def load_retail():
    cavallo = pd.read_csv(os.path.join(DATA, "daily_price_indices_cavallo_etal.csv"))
    cavallo["date"] = pd.to_datetime(cavallo["date"], format="%d%b%Y")
    r = np.load(os.path.join(OUT, "sector_retail_results.npz"), allow_pickle=True)
    retail_stats = {}
    for k in r.keys():
        v = r[k]
        retail_stats[k] = float(v) if v.ndim == 0 else v.tolist()
    return cavallo, retail_stats

@st.cache_data
def load_manufacturing():
    naics = pd.read_excel(os.path.join(DATA, "code_and_release_data", "301 model", "D_GO_by_NAICS.xlsx"))
    price_idx = pd.read_excel(os.path.join(DATA, "code_and_release_data", "301 model", "D_price_indices.xlsx"))
    shocks = pd.read_csv(os.path.join(DATA, "processed", "shocks", "sector_tariff_shocks.csv"))
    hts = pd.read_csv(os.path.join(DATA, "us_tariff_schedule_2025_hts8.csv"), low_memory=False)
    hts["mfn_rate"] = pd.to_numeric(hts["mfn_ad_val_rate"], errors="coerce").fillna(0)
    r = np.load(os.path.join(OUT, "sector_manufacturing_results.npz"), allow_pickle=True)
    mfg_stats = {k: float(r[k]) for k in r.keys()}
    return naics, price_idx, shocks, hts, mfg_stats

@st.cache_data
def load_pharma1():
    """Load Pharma1.xlsx (USITC monthly import data) and return pharma_df + monthly column names."""
    import warnings
    warnings.filterwarnings("ignore")
    df = pd.read_excel(os.path.join(DATA, "Pharma1.xlsx"),
                       sheet_name="Query Results", header=2)
    df.columns = [str(c).strip() for c in df.columns]
    month_cols = ["January","February","March","April","May","June",
                  "July","August","September","October","November","December"]
    month_cols = [m for m in month_cols if m in df.columns]
    for m in month_cols:
        df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0)
    df["annual_total"] = df[month_cols].sum(axis=1)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["HTS Number"] = pd.to_numeric(df["HTS Number"], errors="coerce")
    df = df.dropna(subset=["Year","HTS Number"]).copy()
    return df, month_cols

@st.cache_data
def load_pharma_outputs():
    """Load pharma analysis output CSVs: dependence, sourcing shifts, consumer burden, country exposure."""
    dep = pd.read_csv(os.path.join(OUT, "pharma1_objective1_dependence_top_suppliers.csv"))
    src = pd.read_csv(os.path.join(OUT, "pharma1_objective2_sourcing_shifts_2025.csv"))
    burd = pd.read_csv(os.path.join(OUT, "pharma1_objective3_consumer_burden_2025.csv"))
    exp  = pd.read_csv(os.path.join(OUT, "pharma1_country_exposure_2024.csv"))
    return dep, src, burd, exp

@st.cache_data
def load_tariffs():
    tariffs = pd.read_csv(os.path.join(DATA, "base_data", "tariffs.csv"))
    cl = pd.read_csv(os.path.join(DATA, "base_data", "country_labels.csv"))
    df = cl.copy()
    df["applied_tariff"] = tariffs["applied_tariff"].values
    df["tariff_pct"] = df["applied_tariff"] * 100
    return df.sort_values("tariff_pct", ascending=False).reset_index(drop=True)

@st.cache_data
def load_bls_cpi():
    """Official BLS CPI series (monthly, 2022 - May 2026): headline + tariff-exposed
    goods categories + domestic services controls. Oct 2025 missing (govt shutdown)."""
    import json
    with open(os.path.join(DATA, "BLS_Retail_CPI_Monthly_2022_2026.json"), encoding="utf-8") as f:
        raw = json.load(f)
    names = {
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
    rows = []
    for s in raw["Results"]["series"]:
        sid = s["seriesID"]
        if sid not in names:
            continue
        for p in s["data"]:
            if p["value"] == "-":
                continue  # Oct 2025 shutdown gap
            rows.append({
                "sid": sid,
                "name": names[sid][0],
                "group": names[sid][1],
                "date": pd.Timestamp(int(p["year"]), int(p["period"][1:]), 1),
                "value": float(p["value"]),
            })
    return pd.DataFrame(rows).sort_values(["sid", "date"]).reset_index(drop=True)

@st.cache_data
def load_mfg_reality():
    """Post-Liberation Day reality check data:
    - USITC monthly imports + calculated duties by HTS chapter, 2022-2025
    - BEA quarterly gross output by industry, 2024Q1-2026Q1 (nominal, SAAR)"""
    import warnings
    warnings.filterwarnings("ignore")
    months = ["January","February","March","April","May","June",
              "July","August","September","October","November","December"]

    def _parse_usitc(sheet):
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

    imports_long = _parse_usitc("Customs Value")
    duties_long  = _parse_usitc("Calculated Duties")

    bea = pd.read_excel(os.path.join(DATA, "BEA_Gross_Output_by_Industry_latest.xlsx"), header=None)
    bea_sub = bea.iloc[7:45, :11].copy()
    bea_sub.columns = ["line", "industry", "2024Q1", "2024Q2", "2024Q3", "2024Q4",
                       "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1"]
    bea_sub["industry"] = bea_sub["industry"].astype(str).str.strip()
    for c in bea_sub.columns[2:]:
        bea_sub[c] = pd.to_numeric(bea_sub[c], errors="coerce")
    return imports_long, duties_long, bea_sub

@st.cache_data
def load_bilateral():
    """US bilateral trade vectors from the 194x194 CEPII matrix (units: $1000s).
    Column id_US = each country's exports TO the US (US imports by partner);
    row id_US = US exports to each partner."""
    t = pd.read_csv(os.path.join(DATA, "base_data", "trade_cepii.csv")).values.astype(float)
    b = np.load(os.path.join(OUT, "baseline_results.npz"), allow_pickle=True)
    id_us = int(b["id_US"])
    us_imports = np.nan_to_num(t[:, id_us])   # partner -> US
    us_exports = np.nan_to_num(t[id_us, :])   # US -> partner
    d_trade = b["d_trade"]                     # global trade change per scenario (%)
    d_employment = b["d_employment"]           # US employment change per scenario (%)
    return us_imports, us_exports, d_trade, d_employment

@st.cache_data
def load_pharma_risk():
    risk    = pd.read_csv(os.path.join(OUT, "pharma1_supply_chain_risk_2024.csv"))
    top_sup = pd.read_csv(os.path.join(OUT, "pharma1_objective1_dependence_top_suppliers.csv"))
    hts_exp = pd.read_csv(os.path.join(OUT, "pharma1_hts_exposure_2024.csv"))
    r = np.load(os.path.join(OUT, "sector_pharma_results.npz"), allow_pickle=True)
    pharma_stats = {}
    for k in r.keys():
        v = r[k]
        pharma_stats[k] = float(v) if v.ndim == 0 else v.tolist()
    return risk, top_sup, hts_exp, pharma_stats

# ── Import run_tariff_scenario for the scenario builder ──────────────────────
_srv_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _srv_root not in sys.path:
    sys.path.insert(0, _srv_root)
from mcp_server.server import run_tariff_scenario as _run_tariff_scenario

def _compute_custom_scenario(_frozen_rates):
    overrides = {iso3: rate / 100.0 for iso3, rate in _frozen_rates}
    return _run_tariff_scenario(tariff_overrides=overrides, countries=["USA"] + list(overrides.keys()))

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">🏛️ Liberation Day Tariff Impact Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Replication of Ignatenko, Macedoni, Lashkaripour & Simonovska (2025) · April 2025 US Tariff Announcements</div>', unsafe_allow_html=True)

# Load baseline data at top level so sidebar can access country names + results
_g_results, _g_Y_i, _g_id_US, _g_cl, _g_results_15pct, _ = load_baseline()
_g_tariff_df = load_tariffs()

# These are set inside the sidebar block (no new scope in Python with-blocks) and read in Tab 1
_country_profile = None
_sb_live         = None   # single CE scenario result — computed once in sidebar, reused in Tab 1

with st.sidebar:
    st.markdown("### 🔍 Country Explorer")
    st.markdown('<div style="font-size:12px;color:#64748b;margin-bottom:10px">Select any country to see its Liberation Day tariff stats and build a custom scenario.</div>', unsafe_allow_html=True)

    _country_options = ["(Select a country)"] + sorted(_g_cl["CountryName"].tolist())
    _sel_country = st.selectbox("Country", _country_options, key="country_explorer", label_visibility="collapsed")

    if _sel_country != "(Select a country)":
        _crow = _g_cl[_g_cl["CountryName"] == _sel_country]
        if not _crow.empty:
            _cidx    = int(_crow.index[0])
            _ciso3   = str(_crow["iso3"].iloc[0])
            _ct_row  = _g_tariff_df[_g_tariff_df["CountryName"] == _sel_country]
            _ctariff = float(_ct_row["tariff_pct"].iloc[0]) if not _ct_row.empty else 0.0

            # GE model stats — USTR no retaliation (scenario 0)
            _c_welfare = float(_g_results[_cidx, 0, 0])
            _c_cpi     = float(_g_results[_cidx, 5, 0])
            _c_imp     = float(_g_results[_cidx, 3, 0])
            _c_exp     = float(_g_results[_cidx, 2, 0])
            _c_emp     = float(_g_results[_cidx, 4, 0])

            st.markdown(f'<div style="background:#1a1d2e;border:1px solid #2d3250;border-radius:8px;padding:12px;margin-bottom:10px">'
                f'<div style="color:#94a3b8;font-size:11px;font-weight:700;letter-spacing:1px;margin-bottom:8px">{_sel_country.upper()}</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">'
                f'<div><div style="color:#64748b;font-size:10px">Applied Tariff</div><div style="color:#f87171;font-size:18px;font-weight:700">{_ctariff:.0f}%</div></div>'
                f'<div><div style="color:#64748b;font-size:10px">Welfare Change</div><div style="color:{"#22d3a0" if _c_welfare>0 else "#f87171"};font-size:18px;font-weight:700">{_c_welfare:+.2f}%</div></div>'
                f'<div><div style="color:#64748b;font-size:10px">Price Change</div><div style="color:#fbbf24;font-size:16px;font-weight:600">{_c_cpi:+.1f}%</div></div>'
                f'<div><div style="color:#64748b;font-size:10px">Import Vol.</div><div style="color:#60a5fa;font-size:16px;font-weight:600">{_c_imp:+.1f}%</div></div>'
                f'<div><div style="color:#64748b;font-size:10px">Export Vol.</div><div style="color:#60a5fa;font-size:16px;font-weight:600">{_c_exp:+.1f}%</div></div>'
                f'<div><div style="color:#64748b;font-size:10px">Employment</div><div style="color:{"#22d3a0" if _c_emp>0 else "#f87171"};font-size:16px;font-weight:600">{_c_emp:+.2f}%</div></div>'
                f'</div></div>', unsafe_allow_html=True)

            st.markdown("**What if we change the tariff?**")
            # Reset flag must be checked before slider is instantiated
            if st.session_state.pop("_reset_ce", False):
                st.session_state["ce_scenario_slider"] = int(_ctariff)
            _c_scenario_rate = st.slider(
                f"New tariff for {_sel_country}", 0, 100, int(_ctariff), 1, format="%d%%",
                key="ce_scenario_slider"
            )
            if st.button("↺ Reset to Liberation Day tariff", use_container_width=True, key="ce_reset"):
                st.session_state["_reset_ce"] = True
                st.rerun()

            _country_profile = {
                "country": _sel_country, "iso3": _ciso3, "idx": _cidx,
                "tariff": _ctariff, "scenario_rate": _c_scenario_rate,
                "welfare": _c_welfare, "cpi": _c_cpi,
                "imp": _c_imp, "exp": _c_exp, "emp": _c_emp,
            }

            # ── Inline live result right below the slider ──────────────────
            if _c_scenario_rate != int(_ctariff):
                _sb_frozen = ((_ciso3, _c_scenario_rate),)
                _sb_live   = _compute_custom_scenario(_sb_frozen)  # also used in Tab 1
                _sb_ctries = _sb_live.get("data", {}).get("countries", [])
                _sb_us     = next((r for r in _sb_ctries if r.get("iso3") == "USA"), {})
                _sb_wd     = _sb_us.get("welfare_delta_pct") or 0
                _sb_nw     = _sb_us.get("new_welfare_pct") or 0
                _sb_bw     = _sb_us.get("baseline_welfare_pct") or 0
                _sb_dpp    = _c_scenario_rate - int(_ctariff)
                _sb_clr    = "#22d3a0" if _sb_wd > 0 else "#f87171"
                _sb_dir    = "cut" if _sb_dpp < 0 else "raised"
                _sb_effect = "better off" if _sb_wd > 0 else "worse off"
                st.markdown(
                    f'<div style="background:#0f172a;border:1px solid {_sb_clr};border-radius:8px;padding:12px;margin-top:8px">'
                    f'<div style="color:{_sb_clr};font-size:10px;font-weight:700;letter-spacing:1px;margin-bottom:8px">LIVE SCENARIO RESULT</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px">'
                    f'<div><div style="color:#64748b;font-size:9px">US economy today</div><div style="color:#60a5fa;font-size:16px;font-weight:700">{_sb_bw:+.2f}%</div></div>'
                    f'<div><div style="color:#64748b;font-size:9px">US economy after change</div><div style="color:{"#22d3a0" if _sb_nw>0 else "#f87171"};font-size:16px;font-weight:700">{_sb_nw:+.2f}%</div></div>'
                    f'</div>'
                    f'<div style="background:#1a1d2e;border-radius:6px;padding:8px;text-align:center">'
                    f'<div style="color:#64748b;font-size:9px">Change in US wellbeing</div>'
                    f'<div style="color:{_sb_clr};font-size:22px;font-weight:700">{_sb_wd:+.2f}%</div>'
                    f'<div style="color:#475569;font-size:10px">America is <b style="color:{_sb_clr}">{_sb_effect}</b> if tariff is {_sb_dir} by {abs(_sb_dpp)}pp</div>'
                    f'</div>'
                    f'</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🌐 Macro Overview", "💊 Pharma Supply Chain", "🛒 Retail & Consumer Prices", "🏭 Manufacturing Exposure", "🤖 AI Analyst", "🔌 MCP Analyst", "🎛️ Build Your Scenario"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — MACRO OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    results, Y_i, id_US, cl, results_15pct, d_trade_15pct = load_baseline()
    tariff_df = load_tariffs()

    SCENARIOS = {
        "USTR + No Retaliation":         0,
        "USTR + Lump-Sum Rebate":        7,
        "Optimal Tariff":                3,
        "USTR + Reciprocal Retaliation": 5,
        "USTR + Optimal Retaliation":    4,
        "Flat 15% Tariff (Custom)":      None,
    }

    # ── Chapter header helper ────────────────────────────────────────────────
    def _chapter(n, title, sub=""):
        st.markdown(
            f'<div style="display:flex;align-items:baseline;gap:14px;margin:38px 0 6px 0;padding-top:8px;border-top:2px solid #2d3250">'
            f'<div style="color:#2563eb;font-size:13px;font-weight:800;letter-spacing:2px">CHAPTER {n}</div>'
            f'<div style="color:#f1f5f9;font-size:22px;font-weight:700">{title}</div>'
            f'</div>'
            + (f'<div style="color:#64748b;font-size:13px;margin-bottom:14px">{sub}</div>' if sub else ''),
            unsafe_allow_html=True)

    # ── Hero numbers (measured + model) ──────────────────────────────────────
    _us_imports_vec, _us_exports_vec, _d_trade_sc, _d_emp_sc = load_bilateral()
    _tariff_by_idx = tariff_df.set_index("iso3").reindex(cl["iso3"])["tariff_pct"].fillna(0).values
    _wt_avg_tariff = float((_us_imports_vec * _tariff_by_idx).sum() / max(_us_imports_vec.sum(), 1))

    _imp_h, _dut_h, _bea_h = load_mfg_reality()
    _mv_h = _imp_h.groupby("date")["value"].sum().rename("value").to_frame()
    _mv_h["duty"] = _dut_h.groupby("date")["value"].sum()
    _mv_h["rate"] = _mv_h["duty"] / _mv_h["value"] * 100
    _rate_pre_h  = float(_mv_h.loc[(_mv_h.index >= "2024-01-01") & (_mv_h.index <= "2024-12-01"), "rate"].mean())
    _rate_peak_h = float(_mv_h.loc[_mv_h.index >= "2025-04-01", "rate"].max())
    _dut_pre_h   = float(_dut_h[(_dut_h["date"] >= "2024-04-01") & (_dut_h["date"] <= "2024-12-01")]["value"].sum())
    _dut_post_h  = float(_dut_h[(_dut_h["date"] >= "2025-04-01") & (_dut_h["date"] <= "2025-12-01")]["value"].sum())
    _imp_pre_h   = float(_imp_h[(_imp_h["date"] >= "2024-04-01") & (_imp_h["date"] <= "2024-12-01")]["value"].sum())
    _imp_post_h  = float(_imp_h[(_imp_h["date"] >= "2025-04-01") & (_imp_h["date"] <= "2025-12-01")]["value"].sum())
    _bea_mfg_h = _bea_h[_bea_h["industry"] == "Manufacturing"]
    _bea_chg_h = float((_bea_mfg_h["2026Q1"].iloc[0] / _bea_mfg_h["2025Q1"].iloc[0] - 1) * 100) if not _bea_mfg_h.empty else 0

    _bls_h = load_bls_cpi()
    _ld_ts_h, _pre_ts_h = pd.Timestamp("2025-03-01"), pd.Timestamp("2023-01-01")
    _latest_ts_h = _bls_h["date"].max()
    _head_s_h = _bls_h[_bls_h["sid"] == "CUSR0000SA0"].set_index("date")["value"]
    _m_pre_h  = (_ld_ts_h.year - _pre_ts_h.year) * 12 + (_ld_ts_h.month - _pre_ts_h.month)
    _m_post_h = (_latest_ts_h.year - _ld_ts_h.year) * 12 + (_latest_ts_h.month - _ld_ts_h.month)
    _infl_pre_h  = ((_head_s_h[_ld_ts_h] / _head_s_h[_pre_ts_h]) ** (12 / _m_pre_h) - 1) * 100
    _infl_post_h = ((_head_s_h[_latest_ts_h] / _head_s_h[_ld_ts_h]) ** (12 / _m_post_h) - 1) * 100
    _comp_s_h = _bls_h[_bls_h["sid"] == "CUSR0000SEEE01"].set_index("date")["value"]
    _comp_pre_h  = ((_comp_s_h[_ld_ts_h] / _comp_s_h[_pre_ts_h]) ** (12 / _m_pre_h) - 1) * 100
    _comp_post_h = ((_comp_s_h[_latest_ts_h] / _comp_s_h[_ld_ts_h]) ** (12 / _m_post_h) - 1) * 100

    # Model headline (USTR no-retaliation, scenario 0)
    _w0_h = results[:, 0, 0]
    _us_welfare_h = float(_w0_h[id_US])
    _global_bn_h = float((_w0_h / 100 * Y_i).sum() / 1e6)
    _n_lose_h = int((_w0_h < 0).sum())

    # ── CHAPTER 1: The opening ────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#111827,#1a1d2e 60%,#1e1b31);border:1px solid #2d3250;border-radius:16px;padding:28px 32px;margin-bottom:8px">'
        f'<div style="color:#f87171;font-size:13px;font-weight:800;letter-spacing:3px;margin-bottom:6px">APRIL 2, 2025 · "LIBERATION DAY"</div>'
        f'<div style="color:#f1f5f9;font-size:30px;font-weight:800;line-height:1.2;margin-bottom:8px">The largest US tariff increase in a century</div>'
        f'<div style="color:#94a3b8;font-size:14px;max-width:900px;margin-bottom:22px">A minimum 10 percent tariff on every one of 194 countries, with China facing the highest rate at 54 percent. This dashboard follows what actually happened next, using official customs, price and output data.</div>'
        f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:14px">'
        f'<div style="background:#0f172a;border-radius:10px;padding:14px"><div style="color:#f87171;font-size:30px;font-weight:800">2.4% → {_wt_avg_tariff:.0f}%</div><div style="color:#94a3b8;font-size:12px;font-weight:600">the tariff wall</div><div style="color:#475569;font-size:10px;margin-top:3px">The average tariff on goods America buys rose roughly tenfold overnight.</div></div>'
        f'<div style="background:#0f172a;border-radius:10px;padding:14px"><div style="color:#fbbf24;font-size:30px;font-weight:800">~{_rate_peak_h:.0f}% peak</div><div style="color:#94a3b8;font-size:12px;font-weight:600">what importers actually paid</div><div style="color:#475569;font-size:10px;margin-top:3px">Exemptions and carve-outs meant the real rate came in well below the headlines.</div></div>'
        f'<div style="background:#0f172a;border-radius:10px;padding:14px"><div style="color:#22d3a0;font-size:30px;font-weight:800">${_dut_post_h/1e9:,.0f}B</div><div style="color:#94a3b8;font-size:12px;font-weight:600">estimated tariff burden</div><div style="color:#475569;font-size:10px;margin-top:3px">Collected in the nine months after the tariffs, nearly five times the year before.</div></div>'
        f'<div style="background:#0f172a;border-radius:10px;padding:14px"><div style="color:#fbbf24;font-size:30px;font-weight:800">{_infl_pre_h:.1f}% → {_infl_post_h:.1f}%</div><div style="color:#94a3b8;font-size:12px;font-weight:600">consumer inflation, per year</div><div style="color:#475569;font-size:10px;margin-top:3px">The increase was concentrated in tariffed goods, while services actually cooled.</div></div>'
        f'<div style="background:#0f172a;border-radius:10px;padding:14px"><div style="color:#60a5fa;font-size:30px;font-weight:800">{_us_welfare_h:+.2f}% / −${abs(_global_bn_h):,.0f}B</div><div style="color:#94a3b8;font-size:12px;font-weight:600">US vs the world, if no one retaliates</div><div style="color:#475569;font-size:10px;margin-top:3px">The model finds America gains slightly while {_n_lose_h} countries end up worse off.</div></div>'
        f'</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Country spotlight (top of page when a country is selected) ─────────
    if _country_profile:
        cp = _country_profile

        # Plain-English labels and interpretations
        _w_label  = "Economy grew" if cp["welfare"] > 0 else "Economy shrank"
        _w_sub    = f"{'Gained' if cp['welfare']>0 else 'Lost'} {abs(cp['welfare']):.2f}% in real income under Liberation Day tariffs"
        _cpi_label = "Prices fell" if cp["cpi"] < 0 else "Prices rose"
        _cpi_sub  = f"{'Cheaper' if cp['cpi']<0 else 'More expensive'} goods & services — consumer price change"
        _imp_label = "Bought more from US" if cp["imp"] > 0 else "Bought less from US"
        _imp_sub  = f"Change in how much {cp['country']} imports from the US"
        _emp_label = "Jobs gained" if cp["emp"] > 0 else "Jobs lost"
        _emp_sub  = f"Estimated employment change due to tariff shock"

        st.markdown(
            f'<div style="background:linear-gradient(135deg,#0d2218,#1a1d2e);border:2px solid #22d3a0;border-radius:12px;padding:20px;margin-bottom:16px">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">'
            f'<div style="background:#22d3a0;color:#0d2218;font-size:11px;font-weight:800;letter-spacing:1px;padding:3px 10px;border-radius:20px">COUNTRY FOCUS</div>'
            f'<div style="color:#e2e8f0;font-size:24px;font-weight:700">{cp["country"]}</div>'
            f'<div style="color:#64748b;font-size:13px;margin-left:auto">US tariff rate: <span style="color:#f87171;font-weight:700">{cp["tariff"]:.0f}%</span></div>'
            f'</div>'
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px">'
            f'<div style="background:#0f172a;border-radius:8px;padding:12px">'
            f'  <div style="color:#94a3b8;font-size:11px;margin-bottom:2px">{_w_label}</div>'
            f'  <div style="color:{"#22d3a0" if cp["welfare"]>0 else "#f87171"};font-size:26px;font-weight:700">{cp["welfare"]:+.2f}%</div>'
            f'  <div style="color:#475569;font-size:10px;margin-top:4px">{_w_sub}</div>'
            f'</div>'
            f'<div style="background:#0f172a;border-radius:8px;padding:12px">'
            f'  <div style="color:#94a3b8;font-size:11px;margin-bottom:2px">{_cpi_label}</div>'
            f'  <div style="color:#fbbf24;font-size:26px;font-weight:700">{cp["cpi"]:+.1f}%</div>'
            f'  <div style="color:#475569;font-size:10px;margin-top:4px">{_cpi_sub}</div>'
            f'</div>'
            f'<div style="background:#0f172a;border-radius:8px;padding:12px">'
            f'  <div style="color:#94a3b8;font-size:11px;margin-bottom:2px">{_imp_label}</div>'
            f'  <div style="color:#60a5fa;font-size:26px;font-weight:700">{cp["imp"]:+.1f}%</div>'
            f'  <div style="color:#475569;font-size:10px;margin-top:4px">{_imp_sub}</div>'
            f'</div>'
            f'<div style="background:#0f172a;border-radius:8px;padding:12px">'
            f'  <div style="color:#94a3b8;font-size:11px;margin-bottom:2px">{_emp_label}</div>'
            f'  <div style="color:{"#22d3a0" if cp["emp"]>0 else "#f87171"};font-size:26px;font-weight:700">{cp["emp"]:+.2f}%</div>'
            f'  <div style="color:#475569;font-size:10px;margin-top:4px">{_emp_sub}</div>'
            f'</div>'
            f'</div></div>', unsafe_allow_html=True)

        # Scenario result if slider was moved
        if _sb_live:
            _ce_countries = _sb_live.get("data", {}).get("countries", [])
            _ce_us   = next((r for r in _ce_countries if r.get("iso3") == "USA"), {})
            _ce_ctry = next((r for r in _ce_countries if r.get("iso3") == cp["iso3"]), {})
            _delta_pp  = cp["scenario_rate"] - int(cp["tariff"])
            _wd_us     = (_ce_us.get("welfare_delta_pct") or 0)
            _nw_us     = (_ce_us.get("new_welfare_pct") or 0)
            _bw_us     = (_ce_us.get("baseline_welfare_pct") or 0)
            _wd_ctry   = (_ce_ctry.get("welfare_delta_pct") or 0)
            _clr_delta = "#22d3a0" if _wd_us > 0 else "#f87171"
            _direction = "cut" if _delta_pp < 0 else "raised"
            _us_effect = "better off" if _wd_us > 0 else "worse off"
            if _wd_us > 0:
                _insight = (f"{'Cutting' if _delta_pp < 0 else 'Changing'} {cp['country']}'s tariff by {abs(_delta_pp)}pp "
                            f"is estimated to improve US living standards by {abs(_wd_us):.2f}pp — "
                            f"cheaper imports benefit American consumers.")
            else:
                _insight = (f"{'Raising' if _delta_pp > 0 else 'Changing'} {cp['country']}'s tariff by {abs(_delta_pp)}pp "
                            f"is estimated to reduce US living standards by {abs(_wd_us):.2f}pp — "
                            f"higher tariffs make imports more expensive for American consumers.")

            st.markdown(
                f'<div style="background:#1a1d2e;border:1px solid {"#22d3a0" if _wd_us>0 else "#f87171"};border-radius:10px;padding:18px;margin-bottom:14px">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">'
                f'<div style="background:{"#0d2218" if _wd_us>0 else "#2a0f0f"};color:{"#22d3a0" if _wd_us>0 else "#f87171"};font-size:11px;font-weight:800;letter-spacing:1px;padding:3px 10px;border-radius:20px">WHAT HAPPENS IF WE CHANGE THIS?</div>'
                f'<div style="color:#e2e8f0;font-size:14px">{cp["country"]} tariff: <b>{cp["tariff"]:.0f}%</b> → <span style="color:{_clr_delta};font-weight:700">{cp["scenario_rate"]}%</span> ({_delta_pp:+d}pp)</div>'
                f'</div>'
                f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:14px">'

                f'<div style="background:#0f172a;border-radius:8px;padding:12px">'
                f'  <div style="color:#94a3b8;font-size:11px;margin-bottom:2px">US economy today (Liberation Day)</div>'
                f'  <div style="color:#60a5fa;font-size:24px;font-weight:700">{_bw_us:+.2f}%</div>'
                f'  <div style="color:#475569;font-size:10px;margin-top:4px">Current estimated change in US living standards</div>'
                f'</div>'

                f'<div style="background:#0f172a;border-radius:8px;padding:12px">'
                f'  <div style="color:#94a3b8;font-size:11px;margin-bottom:2px">US economy under your scenario</div>'
                f'  <div style="color:{"#22d3a0" if _nw_us>0 else "#f87171"};font-size:24px;font-weight:700">{_nw_us:+.2f}%</div>'
                f'  <div style="color:#475569;font-size:10px;margin-top:4px">Estimated US living standards if tariff changes</div>'
                f'</div>'

                f'<div style="background:#0f172a;border-radius:8px;padding:12px;border:1px solid {_clr_delta}">'
                f'  <div style="color:#94a3b8;font-size:11px;margin-bottom:2px">Change in US wellbeing</div>'
                f'  <div style="color:{_clr_delta};font-size:24px;font-weight:700">{_wd_us:+.2f}%</div>'
                f'  <div style="color:#475569;font-size:10px;margin-top:4px">America is <b style="color:{_clr_delta}">{_us_effect}</b> if this tariff is {_direction}</div>'
                f'</div>'

                f'</div>'
                f'<div style="background:#0f172a;border-radius:8px;padding:12px;border-left:3px solid {_clr_delta}">'
                f'  <div style="color:#94a3b8;font-size:11px;margin-bottom:4px">💡 What this means</div>'
                f'  <div style="color:#e2e8f0;font-size:13px;line-height:1.5">{_insight} '
                f'(Uses a partial equilibrium model — directionally correct, not a full GE re-solve.)</div>'
                f'</div>'
                f'</div>', unsafe_allow_html=True)

    # ── CHAPTER 2: Did it work? Promises vs measured reality ────────────────
    _chapter(2, "Did it work? — promises vs measured reality",
             "Five things the tariffs were supposed to do, graded against official post-April-2025 data. Sources tagged on every row.")

    def _verdict_row(claim, verdict, chip_bg, chip_fg, evidence, source):
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:16px;background:#1a1d2e;border:1px solid #2d3250;border-radius:10px;padding:14px 18px;margin-bottom:8px">'
            f'<div style="flex:0 0 260px;color:#e2e8f0;font-size:14px;font-weight:600">“{claim}”</div>'
            f'<div style="flex:0 0 150px"><span style="background:{chip_bg};color:{chip_fg};font-size:11px;font-weight:800;letter-spacing:1px;padding:4px 12px;border-radius:20px;white-space:nowrap">{verdict}</span></div>'
            f'<div style="flex:1;color:#94a3b8;font-size:13px;line-height:1.45">{evidence} <span style="color:#475569;font-size:11px">· {source}</span></div>'
            f'</div>', unsafe_allow_html=True)

    _verdict_row("Tariffs will raise massive revenue",
        "✓ DELIVERED", "#0d2218", "#22d3a0",
        f"USITC calculated duties on eight manufacturing chapters ran <b>{_dut_post_h/max(_dut_pre_h,1):.1f}× higher</b> than the year before — <b>${_dut_post_h/1e9:,.0f}B</b> during Apr–Dec 2025 (includes all trade measures in force, not Liberation Day alone).",
        "USITC customs data, measured")
    _verdict_row("Prices won't go up for Americans",
        "✗ DIDN'T HAPPEN", "#2a0f0f", "#f87171",
        f"Headline inflation accelerated from <b>{_infl_pre_h:.1f}%</b> to <b>{_infl_post_h:.1f}%/yr</b>. Computers flipped from getting cheaper ({_comp_pre_h:+.1f}%/yr) to inflating ({_comp_post_h:+.1f}%/yr); apparel and furniture accelerated ~3pp — while untariffed services cooled.",
        "BLS CPI, measured")
    _verdict_row("Manufacturing will come back",
        "◐ MIXED", "#2a230f", "#fbbf24",
        f"US factory output grew <b>{_bea_chg_h:+.1f}%</b> (nominal) in the year after tariffs and steel imports fell 24% — but machinery imports <b>rose 26%</b>: America still buys the equipment it can't make.",
        "BEA + USITC, measured")
    _verdict_row("Imports will collapse, the deficit will shrink",
        "◐ MIXED", "#2a230f", "#fbbf24",
        f"Targeted categories were hit hard — steel −24%, vehicles −18%, toys −21% — but <b>total</b> manufacturing imports came in {(_imp_post_h/max(_imp_pre_h,1)-1)*100:+.1f}% vs the year before. Trade rerouted more than it collapsed.",
        "USITC customs data, measured")
    _verdict_row("America will come out ahead",
        "◐ MODEL SAYS BARELY", "#2a230f", "#fbbf24",
        f"The 194-country GE model puts US welfare at <b>{_us_welfare_h:+.2f}%</b> — a small net gain (tariff revenue outweighing consumer costs) — while the world loses <b>${abs(_global_bn_h):,.0f}B</b> and {_n_lose_h} countries shrink.",
        "GE model estimate")

    # ── CHAPTER 3: The tariff wall ────────────────────────────────────────────
    _chapter(3, "The tariff wall — who got hit",
             "Total US applied tariff rate as of April 2, 2025, including pre-existing tariffs. China's 54% = 34% Liberation Day reciprocal + 20% imposed earlier in 2025.")

    def _tariff_color(r):
        if r >= 50: return "#f87171"
        elif r >= 25: return "#fb923c"
        elif r >= 10: return "#fbbf24"
        return "#22d3a0"

    # Build chart data — include selected country even if outside top 25
    top25 = tariff_df.head(25).copy()
    _sel_name = _country_profile["country"] if _country_profile else None
    if _sel_name and _sel_name not in top25["CountryName"].values:
        _sel_row = tariff_df[tariff_df["CountryName"] == _sel_name]
        if not _sel_row.empty:
            top25 = pd.concat([top25, _sel_row]).reset_index(drop=True)

    bar_colors_t1 = [
        "#ffffff" if (_sel_name and row["CountryName"] == _sel_name) else _tariff_color(row["tariff_pct"])
        for _, row in top25.iterrows()
    ]
    bar_widths = [1.0 if (_sel_name and row["CountryName"] == _sel_name) else 0.7 for _, row in top25.iterrows()]

    fig_tariff = go.Figure(go.Bar(
        x=top25["tariff_pct"], y=top25["CountryName"],
        orientation="h", marker_color=bar_colors_t1,
        marker_line_color=["#22d3a0" if (_sel_name and row["CountryName"] == _sel_name) else "rgba(0,0,0,0)" for _, row in top25.iterrows()],
        marker_line_width=[3 if (_sel_name and row["CountryName"] == _sel_name) else 0 for _, row in top25.iterrows()],
        text=[f"◀ {v:.0f}% ← selected" if (_sel_name and row["CountryName"] == _sel_name) else f"{v:.0f}%" for v, (_, row) in zip(top25["tariff_pct"], top25.iterrows())],
        textposition="outside",
    ))
    _chart_title = f"Total US Applied Tariff Rate — {_sel_name} highlighted" if _sel_name else "Top 25 Countries by Total US Applied Tariff Rate (as of April 2025)"
    fig_tariff.update_layout(**PLOTLY_THEME, height=560 if _sel_name and _sel_name not in tariff_df.head(25)["CountryName"].values else 520,
        title=_chart_title, xaxis_title="Tariff Rate (%)")
    fig_tariff.update_xaxes(range=[0, 75])
    fig_tariff.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_tariff, use_container_width=True)
    _explain("Each bar is the total tariff rate a country now faces when selling to the US. Red bars (50%+) are the hardest hit; China leads because its 34% Liberation Day rate stacks on a pre-existing 20%. If a country is selected in the sidebar it appears in white.")

    # ── CHAPTER 4: The world pays ─────────────────────────────────────────────
    _chapter(4, "The world pays — 194 economies, one map",
             "GE model estimate of each country's real-income change. Pick a scenario to see how retaliation changes the picture.")

    _c_sel4, _ = st.columns([2, 5])
    with _c_sel4:
        scenario_name = st.selectbox("Scenario", list(SCENARIOS.keys()), index=0)
    sc = SCENARIOS[scenario_name]
    is_15pct = (sc is None)
    welfare_vals = results_15pct[:, 0] if is_15pct else results[:, 0, sc]

    # Stat strip for the selected scenario
    _n_lose_4 = int((welfare_vals < 0).sum())
    _n_gain_4 = int((welfare_vals > 0).sum())
    _global_bn_4 = float((welfare_vals / 100 * Y_i).sum() / 1e6)
    st.markdown(
        f'<div style="display:flex;gap:26px;background:#1a1d2e;border:1px solid #2d3250;border-radius:10px;padding:12px 20px;margin-bottom:12px">'
        f'<div><span style="color:{"#f87171" if _global_bn_4 < 0 else "#22d3a0"};font-size:22px;font-weight:800">{_global_bn_4:+,.0f}B</span> <span style="color:#64748b;font-size:12px">global welfare change</span></div>'
        f'<div><span style="color:#f87171;font-size:22px;font-weight:800">{_n_lose_4}</span> <span style="color:#64748b;font-size:12px">countries lose</span></div>'
        f'<div><span style="color:#22d3a0;font-size:22px;font-weight:800">{_n_gain_4}</span> <span style="color:#64748b;font-size:12px">countries gain</span></div>'
        f'<div style="margin-left:auto;color:#475569;font-size:11px;align-self:center">scenario: {scenario_name}</div>'
        f'</div>', unsafe_allow_html=True)
    map_df = cl.copy()
    map_df["welfare"] = welfare_vals

    fig_map = px.choropleth(
        map_df, locations="iso3", color="welfare",
        hover_name="CountryName",
        color_continuous_scale=["#f87171","#fca5a5","#fef3c7","#6ee7b7","#22d3a0"],
        color_continuous_midpoint=0,
        range_color=[-5, 5],
        labels={"welfare": "Welfare %"},
    )
    fig_map.update_traces(
        marker_line_color="#0f1117", marker_line_width=0.3,
        hovertemplate="<b>%{hovertext}</b><br>Welfare: %{z:.2f}%<extra></extra>"
    )
    fig_map.update_layout(
        **PLOTLY_THEME, height=420,
        geo=dict(
            bgcolor="#0f1117", showframe=False,
            showcoastlines=True, coastlinecolor="#2d3250",
            showland=True, landcolor="#1a1d2e",
            showocean=True, oceancolor="#0f1117",
            showlakes=False, projection_type="natural earth",
        ),
        coloraxis_colorbar=dict(
            title="Welfare %", tickfont=dict(color="#94a3b8"),
            title_font=dict(color="#94a3b8"), bgcolor="#1a1d2e", bordercolor="#2d3250",
        ),
    )
    # Highlight selected country on the map
    if _country_profile:
        _hl = _g_cl[_g_cl["CountryName"] == _country_profile["country"]]
        if not _hl.empty:
            fig_map.add_trace(go.Choropleth(
                locations=[_hl["iso3"].iloc[0]],
                z=[1], colorscale=[[0,"#22d3a0"],[1,"#22d3a0"]],
                showscale=False, marker_line_color="#ffffff", marker_line_width=2,
                hovertemplate=f"<b>{_country_profile['country']}</b><br>Welfare: {_country_profile['welfare']:+.2f}%<extra></extra>",
            ))
    st.plotly_chart(fig_map, use_container_width=True)
    _explain("Green countries gained economically, red countries lost. Hover over any country for its exact welfare change. The scenario selector above changes which policy world you are looking at - retaliation scenarios turn much more of the map red.")

    # ── Top 20 winners / losers ────────────────────────────────────────────
    st.markdown('<div class="section-header">Biggest Winners and Biggest Losers</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Small open economies that trade heavily with the US face the largest losses. Countries that compete with US exports in third markets may actually gain.</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    map_df_sorted = map_df.sort_values("welfare")
    _hl_name = _country_profile["country"] if _country_profile else None

    with c3:
        losers = map_df_sorted.head(20).copy()
        # If selected country is a loser, ensure it's included
        if _hl_name and _hl_name not in losers["CountryName"].values:
            _hl_row = map_df_sorted[map_df_sorted["CountryName"] == _hl_name]
            if not _hl_row.empty and float(_hl_row["welfare"].iloc[0]) < 0:
                losers = pd.concat([losers, _hl_row]).sort_values("welfare").reset_index(drop=True)
        _loser_colors = ["#ffffff" if (_hl_name and n == _hl_name) else "#f87171" for n in losers["CountryName"]]
        fig_l = go.Figure(go.Bar(
            x=losers["welfare"], y=losers["CountryName"],
            orientation="h", marker_color=_loser_colors,
            marker_line_color=["#22d3a0" if (_hl_name and n == _hl_name) else "rgba(0,0,0,0)" for n in losers["CountryName"]],
            marker_line_width=[2 if (_hl_name and n == _hl_name) else 0 for n in losers["CountryName"]],
            text=[f"◀ {v:.2f}%" if (_hl_name and n == _hl_name) else f"{v:.2f}%" for v, n in zip(losers["welfare"], losers["CountryName"])],
            textposition="outside",
        ))
        _loser_title = f"Top 20 Biggest Losers — {_hl_name} highlighted" if _hl_name else "Top 20 Biggest Losers"
        fig_l.update_layout(**PLOTLY_THEME, height=480, title=_loser_title, xaxis_title="Welfare Change %")
        st.plotly_chart(fig_l, use_container_width=True)
        _explain("The 20 countries whose economies shrank the most. They are almost all small, open economies that send a large share of their exports to the US - when the tariff wall went up, they had nowhere else to sell.")

    with c4:
        winners = map_df_sorted.tail(20).sort_values("welfare", ascending=False).copy()
        # If selected country is a winner, ensure it's included
        if _hl_name and _hl_name not in winners["CountryName"].values:
            _hl_row = map_df_sorted[map_df_sorted["CountryName"] == _hl_name]
            if not _hl_row.empty and float(_hl_row["welfare"].iloc[0]) >= 0:
                winners = pd.concat([winners, _hl_row]).sort_values("welfare", ascending=False).reset_index(drop=True)
        _winner_colors = ["#ffffff" if (_hl_name and n == _hl_name) else "#22d3a0" for n in winners["CountryName"]]
        fig_w = go.Figure(go.Bar(
            x=winners["welfare"], y=winners["CountryName"],
            orientation="h", marker_color=_winner_colors,
            marker_line_color=["#22d3a0" if (_hl_name and n == _hl_name) else "rgba(0,0,0,0)" for n in winners["CountryName"]],
            marker_line_width=[2 if (_hl_name and n == _hl_name) else 0 for n in winners["CountryName"]],
            text=[f"◀ {v:.2f}%" if (_hl_name and n == _hl_name) else f"{v:.2f}%" for v, n in zip(winners["welfare"], winners["CountryName"])],
            textposition="outside",
        ))
        _winner_title = f"Top 20 Biggest Winners — {_hl_name} highlighted" if _hl_name else "Top 20 Biggest Winners"
        fig_w.update_layout(**PLOTLY_THEME, height=480, title=_winner_title, xaxis_title="Welfare Change %")
        st.plotly_chart(fig_w, use_container_width=True)
        _explain("The 20 countries that actually gained. Most had little direct US trade to lose, and picked up business as buyers rerouted orders away from tariffed suppliers like China and Vietnam.")

    # ── Sectoral tariff rates ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Sectoral tariff rates — before vs after Liberation Day</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">How hard each sector\'s imports are taxed. Pharma and manufacturing are model-based effective rates; retail is <b>measured</b> from actual duties paid on consumer goods (furniture & toys chapters, USITC). Every sector saw a 5–8× jump. Full sector detail lives in the 💊 Pharma, 🛒 Retail and 🏭 Manufacturing tabs.</div>', unsafe_allow_html=True)

    # Measured retail effective rate from USITC consumer-goods chapters (94 furniture, 95 toys)
    _ret_chs = [94, 95]
    _rv_pre  = float(_imp_h[(_imp_h["chapter"].isin(_ret_chs)) & (_imp_h["date"] >= "2024-01-01") & (_imp_h["date"] <= "2024-12-01")]["value"].sum())
    _rd_pre  = float(_dut_h[(_dut_h["chapter"].isin(_ret_chs)) & (_dut_h["date"] >= "2024-01-01") & (_dut_h["date"] <= "2024-12-01")]["value"].sum())
    _rv_post = float(_imp_h[(_imp_h["chapter"].isin(_ret_chs)) & (_imp_h["date"] >= "2025-04-01")]["value"].sum())
    _rd_post = float(_dut_h[(_dut_h["chapter"].isin(_ret_chs)) & (_dut_h["date"] >= "2025-04-01")]["value"].sum())
    _ret_pre_rate  = _rd_pre / max(_rv_pre, 1) * 100
    _ret_post_rate = _rd_post / max(_rv_post, 1) * 100

    _risk_sx, _tsup_sx, _htsx_sx, _ph_stats_sx = load_pharma_risk()
    _naics_sx, _pidx_sx, _shocks_sx, _hts_sx, _mfg_stats_sx = load_manufacturing()
    _ph_pre_sx,  _ph_post_sx  = _ph_stats_sx["hts8_pharma_rate"] * 100, _ph_stats_sx["tau_pharma_eff"] * 100
    _mfg_pre_sx, _mfg_post_sx = _mfg_stats_sx["hts8_mfg_rate"] * 100,   _mfg_stats_sx["tau_mfg_avg"] * 100

    _sx1, _sx2, _sx3 = st.columns(3)
    with _sx1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">💊 Pharmaceuticals</div>
          <div class="kpi-value negative" style="font-size:26px">{_ph_pre_sx:.1f}% → {_ph_post_sx:.1f}%</div>
          <div class="kpi-sub">{_ph_post_sx/max(_ph_pre_sx,0.1):.1f}× jump · model effective rate</div>
        </div>""", unsafe_allow_html=True)
    with _sx2:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">🛒 Retail consumer goods</div>
          <div class="kpi-value negative" style="font-size:26px">{_ret_pre_rate:.1f}% → {_ret_post_rate:.1f}%</div>
          <div class="kpi-sub">{_ret_post_rate/max(_ret_pre_rate,0.1):.1f}× jump · measured, USITC duties paid</div>
        </div>""", unsafe_allow_html=True)
    with _sx3:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">🏭 Manufacturing</div>
          <div class="kpi-value negative" style="font-size:26px">{_mfg_pre_sx:.1f}% → {_mfg_post_sx:.1f}%</div>
          <div class="kpi-sub">{_mfg_post_sx/max(_mfg_pre_sx,0.1):.1f}× jump · model effective (measured avg paid ~{_rate_peak_h:.0f}%)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    fig_sect = go.Figure()
    _sect_names = ["Pharmaceuticals", "Retail consumer goods", "Manufacturing"]
    _sect_pre   = [_ph_pre_sx, _ret_pre_rate, _mfg_pre_sx]
    _sect_post  = [_ph_post_sx, _ret_post_rate, _mfg_post_sx]
    fig_sect.add_trace(go.Bar(
        name="Before Liberation Day", x=_sect_names, y=_sect_pre,
        marker_color="#2563eb",
        text=[f"{v:.1f}%" for v in _sect_pre], textposition="outside",
    ))
    fig_sect.add_trace(go.Bar(
        name="After Liberation Day", x=_sect_names, y=_sect_post,
        marker_color="#f87171",
        text=[f"{v:.1f}%" for v in _sect_post], textposition="outside",
    ))
    fig_sect.update_layout(**PLOTLY_THEME, height=340,
        title="Effective Tariff Rate by Sector: Before vs After April 2, 2025",
        barmode="group",
        legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250"))
    fig_sect.update_yaxes(title_text="Effective Tariff Rate (%)", range=[0, 32])
    st.plotly_chart(fig_sect, use_container_width=True)
    _explain("Blue is what each sector paid before April 2025; red is after. All three sectors jumped 5-8x. Retail is measured from real duties paid on furniture and toys; pharma and manufacturing are model-based effective rates.")

    # ── CHAPTER 5: Your turn ──────────────────────────────────────────────────
    _chapter(5, "Your turn", "")
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0d1a2e,#1a1d2e);border:1px solid #2563eb;border-radius:12px;padding:20px 24px;margin-bottom:10px">'
        f'<div style="color:#e2e8f0;font-size:17px;font-weight:700;margin-bottom:6px">Think you could design a better tariff policy?</div>'
        f'<div style="color:#94a3b8;font-size:13px">Open the <b style="color:#60a5fa">🎛️ Build Your Scenario</b> tab to set your own rates for any of the 194 countries — and get a live verdict on wellbeing, prices, revenue and who gets hurt. Or pick a country in the sidebar\'s <b style="color:#60a5fa">🔍 Country Explorer</b> to see this whole page through that country\'s eyes.</div>'
        f'</div>', unsafe_allow_html=True)

    # ── Country Deep Dive — PE scenario chart + global rank ───────────────────
    if _country_profile:
        cp = _country_profile
        st.markdown(f'<div class="section-header">🔍 {cp["country"]}: Scenario Analysis & Global Ranking</div>', unsafe_allow_html=True)

        _ch1, _ch2 = st.columns(2)
        with _ch1:
            if _sb_live and _sb_live.get("chart_spec"):
                fig_ce = go.Figure(_sb_live["chart_spec"])
                fig_ce.update_layout(**PLOTLY_THEME, height=320,
                    title=f"Welfare Impact: {cp['country']} tariff {cp['tariff']:.0f}% → {cp['scenario_rate']}%")
                st.plotly_chart(fig_ce, use_container_width=True)
                _explain("The estimated welfare change for each country if the tariff you set in the sidebar took effect - computed live by the partial-equilibrium model.")
            else:
                st.markdown('<div class="insight-box">Move the scenario slider in the sidebar to see estimated welfare impact.</div>', unsafe_allow_html=True)

        with _ch2:
            _wdf = _g_cl.copy()
            _wdf["welfare"] = _g_results[:, 0, 0]
            _wdf_sorted = _wdf.sort_values("welfare").reset_index(drop=True)
            _rank_idx = _wdf_sorted[_wdf_sorted["CountryName"] == cp["country"]].index
            _rank_num = int(_rank_idx[0]) + 1 if len(_rank_idx) > 0 else 0
            _total = len(_wdf_sorted)
            # Show only nearest 20 countries around the selected one
            _lo = max(0, _rank_num - 11)
            _hi = min(_total, _rank_num + 9)
            _slice = _wdf_sorted.iloc[_lo:_hi]
            _colors_rank = ["#22d3a0" if n == cp["country"] else ("#f87171" if w < 0 else "#2563eb")
                            for n, w in zip(_slice["CountryName"], _slice["welfare"])]
            fig_rank = go.Figure(go.Bar(
                x=_slice["welfare"], y=_slice["CountryName"],
                orientation="h", marker_color=_colors_rank,
                text=[f"◀ {w:+.2f}%" if n == cp["country"] else f"{w:+.2f}%" for n, w in zip(_slice["CountryName"], _slice["welfare"])],
                textposition="outside",
            ))
            fig_rank.update_layout(**PLOTLY_THEME, height=320,
                title=f"{cp['country']} ranks #{_rank_num} of {_total} by welfare change (green = selected)",
                xaxis_title="Welfare Change (%)", showlegend=False)
            st.plotly_chart(fig_rank, use_container_width=True)
            _explain("Where your selected country sits in the global league table of welfare outcomes, shown with its 20 nearest neighbours. Green is your country.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — PHARMA SUPPLY CHAIN
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    pharma_df, monthly_cols = load_pharma1()
    dep, src, burd, exp = load_pharma_outputs()
    risk, top_sup, hts_exp, pharma_stats = load_pharma_risk()

    total_2025 = pharma_df[pharma_df["Year"] == 2025]["annual_total"].sum()
    total_2024 = pharma_df[pharma_df["Year"] == 2024]["annual_total"].sum()

    kpi2 = [
        ("Total US Medicine Imports",      f"${total_2025/1e9:.0f}B",                     "neutral",   "2025 customs value"),
        ("Year-on-Year Growth",             f"{(total_2025/total_2024-1)*100:+.1f}%",      "positive",  "2024 → 2025"),
        ("Effective Pharma Tariff",         f"{pharma_stats['tau_pharma_eff']*100:.1f}%",  "negative",  "vs 2.4% pre-tariff"),
        ("Medicine Import Drop (model)",    f"{pharma_stats['import_chg_noretal']:+.1f}%", "negative",  "GE model estimate"),
        ("Drug Price Increase (model)",     f"+{pharma_stats['price_noretal']:.2f}%",      "negative",  "From tariff pass-through"),
        ("Q1 vs Q5 Drug Cost Gap",          "6.9×",                                        "negative",  "Poorest pay 6.9× more"),
    ]
    cols2 = st.columns(6)
    for col, (label, val, cls, sub) in zip(cols2, kpi2):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value {cls}" style="font-size:22px">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Country spotlight for Pharma tab ──────────────────────────────────────
    if _country_profile:
        cp = _country_profile
        _ph_exp  = exp[exp["country"].str.lower().str.contains(cp["country"][:5].lower(), na=False)]
        _ph_src  = src[src["country"].str.lower().str.contains(cp["country"][:5].lower(), na=False)]
        _ph_risk = risk[risk["country"].str.lower().str.contains(cp["country"][:5].lower(), na=False)]
        _ph_share  = float(_ph_exp["import_share_pct"].iloc[0])  if not _ph_exp.empty  else None
        _ph_tariff = float(_ph_exp["tariff_pct"].iloc[0])        if not _ph_exp.empty  else cp["tariff"]
        _ph_delta  = float(_ph_exp["delta_share_pp"].iloc[0])    if not _ph_exp.empty  else None
        _ph_shift  = float(_ph_src["change_pp"].iloc[0])         if not _ph_src.empty  else None
        _ph_tier   = _ph_risk["risk_tier"].iloc[0]               if not _ph_risk.empty else None
        _tier_clr  = {"Very high": "#f87171", "High": "#fb923c", "Low": "#22d3a0"}.get(_ph_tier, "#60a5fa") if _ph_tier else "#60a5fa"

        _ph_cells = []
        if _ph_share is not None:
            _ph_cells.append(f'<div style="background:#0f172a;border-radius:8px;padding:10px"><div style="color:#64748b;font-size:10px">Pharma Import Share</div><div style="color:#60a5fa;font-size:22px;font-weight:700">{_ph_share:.1f}%</div><div style="color:#475569;font-size:10px">of total US pharma imports</div></div>')
        else:
            _ph_cells.append(f'<div style="background:#0f172a;border-radius:8px;padding:10px"><div style="color:#64748b;font-size:10px">Pharma Import Share</div><div style="color:#475569;font-size:14px">Not a top supplier</div></div>')
        if _ph_delta is not None:
            _delta_clr = "#f87171" if _ph_delta < 0 else "#22d3a0"
            _ph_cells.append(f'<div style="background:#0f172a;border-radius:8px;padding:10px"><div style="color:#64748b;font-size:10px">Post-Tariff Share Change</div><div style="color:{_delta_clr};font-size:22px;font-weight:700">{_ph_delta:+.2f}pp</div><div style="color:#475569;font-size:10px">shift in US pharma import share</div></div>')
        else:
            _ph_cells.append(f'<div style="background:#0f172a;border-radius:8px;padding:10px"><div style="color:#64748b;font-size:10px">Post-Tariff Share Change</div><div style="color:#475569;font-size:14px">No model data</div></div>')
        if _ph_shift is not None:
            _shift_clr = "#22d3a0" if _ph_shift > 0 else "#f87171"
            _ph_cells.append(f'<div style="background:#0f172a;border-radius:8px;padding:10px"><div style="color:#64748b;font-size:10px">Supply Share Change</div><div style="color:{_shift_clr};font-size:22px;font-weight:700">{_ph_shift:+.2f}pp</div><div style="color:#475569;font-size:10px">shift in US pharma business</div></div>')
        else:
            _ph_cells.append(f'<div style="background:#0f172a;border-radius:8px;padding:10px"><div style="color:#64748b;font-size:10px">Supply Share Change</div><div style="color:#475569;font-size:14px">No sourcing data</div></div>')
        _tier_txt = _ph_tier if _ph_tier else "Not classified"
        _ph_cells.append(f'<div style="background:#0f172a;border-radius:8px;padding:10px;border-left:3px solid {_tier_clr}"><div style="color:#64748b;font-size:10px">Supply Chain Risk</div><div style="color:{_tier_clr};font-size:18px;font-weight:700">{_tier_txt}</div><div style="color:#475569;font-size:10px">risk tier classification</div></div>')

        st.markdown(
            f'<div style="background:linear-gradient(135deg,#0d2218,#1a1d2e);border:2px solid #22d3a0;border-radius:12px;padding:16px;margin-bottom:16px">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">'
            f'<div style="background:#22d3a0;color:#0d2218;font-size:11px;font-weight:800;letter-spacing:1px;padding:3px 10px;border-radius:20px">PHARMA FOCUS</div>'
            f'<div style="color:#e2e8f0;font-size:20px;font-weight:700">{cp["country"]}</div>'
            f'<div style="color:#64748b;font-size:13px;margin-left:auto">US pharma tariff on this country: <span style="color:#f87171;font-weight:700">{_ph_tariff:.0f}%</span></div>'
            f'</div>'
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px">'
            + "".join(_ph_cells) +
            f'</div></div>', unsafe_allow_html=True)

        if _ph_shift is not None:
            _gain_lose = "gaining" if _ph_shift > 0 else "losing"
            _reason = "lower tariff makes it more competitive" if _ph_shift > 0 else "higher tariff is pushing US buyers to cheaper alternatives"
            st.markdown(f'<div class="insight-box"><b>{cp["country"]} is {_gain_lose} US pharma business after Liberation Day</b> — {_reason}. The sourcing share changed by {_ph_shift:+.2f} percentage points.</div>', unsafe_allow_html=True)

    # ── #2 Where does the US buy its medicines? (dependence only) ───────────
    st.markdown('<div class="section-header">Where does the US buy its medicines?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">The 15 countries supplying most US medicine imports, with the Liberation Day tariff each now faces. Color = supply-chain risk tier (red = very high concentration risk). Ireland alone supplies about a quarter of US medicines — at a 20% tariff.</div>', unsafe_allow_html=True)

    tier_colors = {"Very high": "#f87171", "High": "#fb923c", "Low": "#22d3a0"}
    exp_top = exp.head(15).copy()
    _sel_pharma_name = _country_profile["country"] if _country_profile else None
    if _sel_pharma_name and not exp_top["country"].str.lower().str.contains(_sel_pharma_name[:5].lower(), na=False).any():
        _sel_ph_row = exp[exp["country"].str.lower().str.contains(_sel_pharma_name[:5].lower(), na=False)]
        if not _sel_ph_row.empty:
            exp_top = pd.concat([exp_top, _sel_ph_row]).reset_index(drop=True)
    bar_c_risk = []
    for ctry in exp_top["country"]:
        if _sel_pharma_name and _sel_pharma_name[:5].lower() in ctry.lower():
            bar_c_risk.append("#ffffff")
        else:
            tier = risk[risk["country"] == ctry]["risk_tier"].values
            bar_c_risk.append(tier_colors.get(tier[0] if len(tier) > 0 else "Low", "#60a5fa"))
    _dep_hl_lines = ["#22d3a0" if (_sel_pharma_name and _sel_pharma_name[:5].lower() in ctry.lower()) else "rgba(0,0,0,0)" for ctry in exp_top["country"]]
    _dep_hl_widths = [3 if (_sel_pharma_name and _sel_pharma_name[:5].lower() in ctry.lower()) else 0 for ctry in exp_top["country"]]
    fig_dep = go.Figure(go.Bar(
        x=exp_top["import_share_pct"], y=exp_top["country"],
        orientation="h", marker_color=bar_c_risk,
        marker_line_color=_dep_hl_lines, marker_line_width=_dep_hl_widths,
        text=[f"◀ {s:.1f}% share | {t:.0f}% tariff — selected" if (_sel_pharma_name and _sel_pharma_name[:5].lower() in ctry.lower()) else f"{s:.1f}% share | {t:.0f}% tariff" for s, t, ctry in zip(exp_top["import_share_pct"], exp_top["tariff_pct"], exp_top["country"])],
        textposition="outside",
    ))
    _dep_title = f"Top 15 Medicine Suppliers — {_sel_pharma_name} highlighted" if _sel_pharma_name else "Top 15 Medicine Suppliers (red = very high risk, orange = high, green = low)"
    fig_dep.update_layout(**PLOTLY_THEME, height=440,
        title=_dep_title)
    fig_dep.update_xaxes(title_text="Share of US medicine imports (%)", range=[0, exp_top["import_share_pct"].max() * 1.55])
    fig_dep.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_dep, use_container_width=True)
    _explain("Each bar is a country's share of US medicine imports; the label also shows the tariff it now faces. Red and orange bars mark the risky combination the tariffs expose: heavy dependence on a supplier whose goods just got more expensive.")

    # ── #3 How did sourcing change after the tariffs? (estimated) ────────────
    st.markdown('<div class="section-header">How did sourcing change after the tariffs?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box"><b>Estimated</b> reallocation of US medicine sourcing under Liberation Day tariffs, using a gravity-style substitution response (share × (1+tariff)^−ε) — not observed customs flows. Low-tariff suppliers like Singapore gain share; high-tariff suppliers like Switzerland and India lose. The table shows the accompanying 2024→2025 rank reshuffle.</div>', unsafe_allow_html=True)

    _c_shift1, _c_shift2 = st.columns([3, 2])
    with _c_shift1:
        src_plot = src.head(20).copy()
        fig_src = go.Figure(go.Bar(
            x=src_plot["change_pp"], y=src_plot["country"],
            orientation="h",
            marker_color=["#22d3a0" if v >= 0 else "#f87171" for v in src_plot["change_pp"]],
            text=[f"{v:+.2f}pp" for v in src_plot["change_pp"]],
            textposition="outside",
        ))
        fig_src.add_vline(x=0, line_color="#4b5563", line_width=1)
        fig_src.update_layout(**PLOTLY_THEME, height=460,
            title="Estimated Change in US Pharma Import Share (pp)")
        fig_src.update_xaxes(title_text="Change in import share (pp)")
        fig_src.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_src, use_container_width=True)
        _explain("Green suppliers gain US pharma business under the tariffs, red suppliers lose it. Estimated with the elasticity-based sourcing model described above - lower-tariff countries become relatively cheaper, so orders shift toward them.")
    with _c_shift2:
        arrow_fn = lambda d: "▲" if d > 0 else ("▼" if d < 0 else "→")
        ts_rank = top_sup.copy()
        ts_rank["Rank Change"] = ts_rank["rank_change"].apply(lambda x: f"{arrow_fn(x)} {abs(int(x))}" if x != 0 else "→ 0")
        ts_rank["Imports"] = ts_rank.apply(lambda r: f"${r['imports_2024_bn']:.0f}B → ${r['imports_2025_bn']:.0f}B", axis=1)
        disp_sup = ts_rank[["country","rank_2024","rank_2025","Rank Change","Imports"]].copy()
        disp_sup.columns = ["Country","2024 Rank","2025 Rank","Rank Δ","Imports ($B)"]
        st.markdown("<br><b>Supplier rankings, 2024 → 2025</b>", unsafe_allow_html=True)
        st.dataframe(disp_sup, use_container_width=True, hide_index=True)


    # ── Measured: medicine imports by supplier, before vs after LD ──────────
    st.markdown('<div class="section-header">⚡ Measured: how medicine imports shifted by supplier</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">The <b>observed</b> counterpart to the estimate above — actual USITC customs value of medicine imports per supplier, comparing April–December 2025 with the same months of 2024 (seasonality-controlled). This is what really happened to pharma sourcing after Liberation Day.</div>', unsafe_allow_html=True)

    _MONTHS_PH = ["January","February","March","April","May","June",
                  "July","August","September","October","November","December"]
    _POST_MONTHS_PH = _MONTHS_PH[3:]  # April..December

    def _ctry_window_sum(df, year):
        sub = df[df["Year"] == year]
        g = sub.groupby("Country")[_POST_MONTHS_PH].sum().sum(axis=1)
        return g

    _ph24 = _ctry_window_sum(pharma_df, 2024)
    _ph25 = _ctry_window_sum(pharma_df, 2025)
    _ph_cmp = pd.DataFrame({"v24": _ph24, "v25": _ph25}).dropna()
    _ph_cmp = _ph_cmp[_ph_cmp["v24"] > 1e9]           # suppliers above $1B to avoid noise
    _ph_cmp["chg"] = (_ph_cmp["v25"] / _ph_cmp["v24"] - 1) * 100
    _ph_cmp = _ph_cmp.sort_values("v24", ascending=False).head(12).sort_values("chg")

    fig_phm = go.Figure(go.Bar(
        x=_ph_cmp["chg"], y=_ph_cmp.index,
        orientation="h",
        marker_color=["#f87171" if v < 0 else "#22d3a0" for v in _ph_cmp["chg"]],
        text=[f"{v:+.0f}%  (${a/1e9:,.1f}B → ${b/1e9:,.1f}B)" for v, a, b in
              zip(_ph_cmp["chg"], _ph_cmp["v24"], _ph_cmp["v25"])],
        textposition="outside",
    ))
    fig_phm.add_vline(x=0, line_color="#4b5563", line_width=1)
    fig_phm.update_layout(**PLOTLY_THEME, height=440,
        title="Change in US Medicine Imports by Supplier (Apr–Dec 2025 vs Apr–Dec 2024, USITC)")
    fig_phm.update_xaxes(title_text="% change in import value")
    st.plotly_chart(fig_phm, use_container_width=True)
    _explain("Actual customs value of medicine imports from each major supplier, same nine months compared in both years. Green suppliers shipped more to the US after Liberation Day, red shipped less - the measured reshuffling of the pharmaceutical supply chain, to compare against the model estimate above.")


    # ── Drug price impact ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Do tariffs raise drug prices?</div>', unsafe_allow_html=True)
    c_p1, c_p2, c_p3 = st.columns(3)
    with c_p1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Average Tariff Before Liberation Day</div>
          <div class="kpi-value neutral" style="font-size:28px">{pharma_stats['hts8_pharma_rate']*100:.1f}%</div>
          <div class="kpi-sub">HTS8 average, pre-tariff</div>
        </div>""", unsafe_allow_html=True)
    with c_p2:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Effective Tariff After Liberation Day</div>
          <div class="kpi-value negative" style="font-size:28px">{pharma_stats['tau_pharma_eff']*100:.1f}%</div>
          <div class="kpi-sub">8× jump in the effective rate</div>
        </div>""", unsafe_allow_html=True)
    with c_p3:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Resulting Drug Price Increase</div>
          <div class="kpi-value negative" style="font-size:28px">+{pharma_stats['price_noretal']:.2f}%</div>
          <div class="kpi-sub">IO multiplier: {pharma_stats['io_multiplier']:.2f}× amplifies the tariff cost</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('<div class="insight-box">The supply chain multiplier of 1.07 means that every $1 in tariff costs gets amplified: drugs depend on imported chemical precursors that also face tariffs. Medicine imports are projected to fall by 38.2%.</div>', unsafe_allow_html=True)


    # ── Who pays the most in higher drug costs? ──────────────────────────────
    st.markdown('<div class="section-header">Who pays the most in higher drug costs?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">The pharma tariff burden is deeply regressive — the poorest households pay 6.9× more of their income on drug cost increases than the wealthiest. Low-income households spend a larger share of income on prescription drugs and cannot substitute.</div>', unsafe_allow_html=True)

    quintile_labels_p = ["Lowest 20%", "Lower-Middle", "Middle", "Upper-Middle", "Top 20%"]
    c3, c4 = st.columns(2)
    with c3:
        fig_burd = go.Figure(go.Bar(
            x=quintile_labels_p,
            y=list(burd["burden_pct_income"]),
            marker_color=["#f87171","#fb923c","#fbbf24","#a3e635","#22d3a0"],
            text=[f"{v:.3f}%" for v in burd["burden_pct_income"]],
            textposition="outside",
        ))
        fig_burd.update_layout(**PLOTLY_THEME, height=320,
            title="Drug Tariff Burden as % of Annual Income",
            yaxis_title="% of Annual Income", xaxis_title="Income Group")
        st.plotly_chart(fig_burd, use_container_width=True)
        _explain("Extra drug cost from tariffs as a share of household income, by income group. The bars fall from left to right - the poorest fifth carries about 7x the relative burden of the richest fifth.")

    with c4:
        fig_spend = go.Figure()
        fig_spend.add_trace(go.Bar(
            name="Annual Drug Spending",
            x=quintile_labels_p, y=burd["annual_drug_spending_usd"],
            marker_color="#2563eb",
        ))
        fig_spend.add_trace(go.Bar(
            name="Extra Cost from Tariffs",
            x=quintile_labels_p, y=burd["extra_cost_from_tariffs_usd"],
            marker_color="#f87171",
        ))
        fig_spend.update_layout(**PLOTLY_THEME, height=320,
            title="Drug Spending vs Tariff Cost per Household (USD/year)",
            barmode="group", yaxis_title="USD per Household",
            legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250"))
        st.plotly_chart(fig_spend, use_container_width=True)
        _explain("Blue is what each income group spends on medicines per year; red is the extra cost the tariffs add. The red slice is similar in dollars for everyone - which is why it hurts low-income households the most.")

    # ── Sector verdict ───────────────────────────────────────────────────────
    st.markdown(
        '<div style="background:linear-gradient(135deg,#141824,#1a1d2e);border:2px solid #fbbf24;border-radius:12px;padding:22px 26px;margin-top:30px">'
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">'
        '<span style="background:#2a230f;color:#fbbf24;font-size:11px;font-weight:800;letter-spacing:2px;padding:4px 12px;border-radius:20px">THE BOTTOM LINE</span>'
        '<span style="color:#f1f5f9;font-size:19px;font-weight:700">Did the tariffs work for pharma? For the most part, no.</span>'
        '</div>'
        '<div style="display:flex;gap:10px;margin-bottom:8px;align-items:baseline"><span style="color:#22d3a0;font-weight:800;font-size:14px">✓</span><span style="color:#cbd5e1;font-size:13px;line-height:1.55">The tariffs did change behavior. Medicine imports from China fell 65 percent and imports from Switzerland fell 31 percent, based on measured customs data.</span></div><div style="display:flex;gap:10px;margin-bottom:8px;align-items:baseline"><span style="color:#f87171;font-weight:800;font-size:14px">✗</span><span style="color:#cbd5e1;font-size:13px;line-height:1.55">But the dependence did not go away. It simply moved. Germany picked up 52 percent more volume and India 22 percent more, and supply actually became a little more concentrated. America still imports its medicines, just from different places.</span></div><div style="display:flex;gap:10px;margin-bottom:8px;align-items:baseline"><span style="color:#f87171;font-weight:800;font-size:14px">✗</span><span style="color:#cbd5e1;font-size:13px;line-height:1.55">Patients carried the cost. Drug prices rose, and the burden landed about seven times harder on the lowest income households than on the highest.</span></div>'
        '<div style="color:#94a3b8;font-size:13px;margin-top:10px;padding-top:10px;border-top:1px solid #2d3250;line-height:1.55">In short, the tariffs redirected the pharmaceutical supply chain rather than bringing it home, and the cost of that shift fell mostly on patients.</div>'
        '</div>', unsafe_allow_html=True)





# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — RETAIL & CONSUMER PRICES
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    cavallo, retail_stats = load_retail()

    ge_cpi        = retail_stats["ge_cpi_noretal"]
    ge_cpi_retal  = retail_stats["ge_cpi_retal"]
    ge_welfare    = retail_stats["ge_welfare_noretal"]
    ge_welfare_r  = retail_stats["ge_welfare_retal"]
    first_order   = retail_stats["first_order_cpi"]
    passthrough   = retail_stats["retail_product_passthrough"]
    regress_ratio = retail_stats["regress_ratio"]
    q_noretal     = retail_stats["quintile_incidence_noretal"]
    q_retal       = retail_stats["quintile_incidence_retal"]
    cav_30d       = retail_stats["cavallo_usa_30d"] * 100
    cav_90d       = retail_stats["cavallo_usa_90d"] * 100

    kpi3 = [
        ("Expected Price Rise\n(before trade adjusts)",  f"+{first_order:.1f}%",        "negative", "First-order naïve estimate"),
        ("Actual Modelled Price Rise",                   f"+{ge_cpi:.1f}%",             "warning",  "General equilibrium model"),
        ("Share Passed to Consumers",                    f"{passthrough*100:.1f}%",     "neutral",  "Of tariff cost"),
        ("Extra Burden on Lowest Earners",               f"{regress_ratio:.2f}×",       "negative", "Q1 pays more than Q5"),
        ("US Prices — First 30 Days",                    f"+{cav_30d:.2f}%",            "warning",  "Cavallo et al. data"),
        ("US Prices — First 90 Days",                    f"+{cav_90d:.2f}%",            "warning",  "Cavallo et al. data"),
    ]
    cols3 = st.columns(6)
    for col, (label, val, cls, sub) in zip(cols3, kpi3):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value {cls}" style="font-size:20px">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Country spotlight for Retail tab ──────────────────────────────────────
    if _country_profile:
        cp = _country_profile
        # Cavallo tracks USA, Canada, Mexico, China — flag if selected is one of these
        _cav_map = {"United States": "index_usa", "Canada": "index_canada", "Mexico": "index_mexico", "China": "index_china"}
        _in_cavallo = cp["country"] in _cav_map

        _w_dir  = "grew" if cp["welfare"] > 0 else "shrank"
        _p_dir  = "fell" if cp["cpi"] < 0 else "rose"
        _x_dir  = "rose" if cp["exp"] > 0 else "fell"

        _cells_t3 = [
            f'<div style="background:#0f172a;border-radius:8px;padding:10px"><div style="color:#64748b;font-size:10px">Economy {_w_dir}</div><div style="color:{"#22d3a0" if cp["welfare"]>0 else "#f87171"};font-size:22px;font-weight:700">{cp["welfare"]:+.2f}%</div><div style="color:#475569;font-size:10px">real income change (GE model)</div></div>',
            f'<div style="background:#0f172a;border-radius:8px;padding:10px"><div style="color:#64748b;font-size:10px">Domestic prices {_p_dir}</div><div style="color:#fbbf24;font-size:22px;font-weight:700">{cp["cpi"]:+.1f}%</div><div style="color:#475569;font-size:10px">consumer price change</div></div>',
            f'<div style="background:#0f172a;border-radius:8px;padding:10px"><div style="color:#64748b;font-size:10px">Exports to US {_x_dir}</div><div style="color:#60a5fa;font-size:22px;font-weight:700">{cp["exp"]:+.1f}%</div><div style="color:#475569;font-size:10px">change in exports/GDP</div></div>',
            f'<div style="background:#0f172a;border-radius:8px;padding:10px;border-left:3px solid {"#22d3a0" if _in_cavallo else "#2d3250"}"><div style="color:#64748b;font-size:10px">Cavallo price tracker</div><div style="color:{"#22d3a0" if _in_cavallo else "#475569"};font-size:16px;font-weight:700">{"Tracked ↗" if _in_cavallo else "Not tracked"}</div><div style="color:#475569;font-size:10px">{"in daily price chart below" if _in_cavallo else "daily data for US, CA, MX, CN only"}</div></div>',
        ]
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#0d1a2e,#1a1d2e);border:2px solid #2563eb;border-radius:12px;padding:16px;margin-bottom:16px">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">'
            f'<div style="background:#2563eb;color:#fff;font-size:11px;font-weight:800;letter-spacing:1px;padding:3px 10px;border-radius:20px">RETAIL FOCUS</div>'
            f'<div style="color:#e2e8f0;font-size:20px;font-weight:700">{cp["country"]}</div>'
            f'<div style="color:#64748b;font-size:13px;margin-left:auto">US tariff on this country: <span style="color:#f87171;font-weight:700">{cp["tariff"]:.0f}%</span></div>'
            f'</div>'
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px">'
            + "".join(_cells_t3) +
            f'</div>'
            f'<div style="margin-top:10px;color:#64748b;font-size:12px;line-height:1.5">'
            f'The charts below show US consumer price data. {cp["country"]}\'s economy {_w_dir} by {abs(cp["welfare"]):.2f}% and its domestic prices {_p_dir} by {abs(cp["cpi"]):.1f}% — these are the spillover effects of US tariff policy on trading partners.'
            + (f' {cp["country"]} appears in the daily price tracker chart — look for it in the line chart below.' if _in_cavallo else '') +
            f'</div>'
            f'</div>', unsafe_allow_html=True)

    # ── Official BLS consumer prices after Liberation Day ────────────────────
    st.markdown('<div style="font-size:13px;font-weight:700;color:#94a3b8;letter-spacing:1px;margin:18px 0 6px 0">⚡ AT A GLANCE — OFFICIAL BLS DATA THROUGH MAY 2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">The cleanest test of whether tariffs raised prices: compare <b>tariff-exposed goods</b> (apparel, furniture, computers — things America imports) against <b>domestic services</b> (medical care, transportation — things that never cross a border). After Liberation Day the exposed categories accelerated sharply while services cooled. Computers had been getting <i>cheaper</i> for years (−3.4%/yr) and flipped to <i>inflating</i> (+2.8%/yr) — a textbook tariff signature. <span style="color:#64748b">(Oct 2025 missing — government shutdown paused BLS data collection.)</span></div>', unsafe_allow_html=True)

    _bls = load_bls_cpi()
    _LD_TS_R = pd.Timestamp("2025-03-01")   # last pre-tariff reading
    _PRE_TS_R = pd.Timestamp("2023-01-01")
    _latest_ts_r = _bls["date"].max()

    # Pre vs post annualized inflation per category
    _accel_rows_r = []
    for _sid_r in _bls["sid"].unique():
        _s_r = _bls[_bls["sid"] == _sid_r].set_index("date")["value"]
        if _PRE_TS_R in _s_r.index and _LD_TS_R in _s_r.index and _latest_ts_r in _s_r.index:
            _m_pre = (_LD_TS_R.year - _PRE_TS_R.year) * 12 + (_LD_TS_R.month - _PRE_TS_R.month)
            _m_post = (_latest_ts_r.year - _LD_TS_R.year) * 12 + (_latest_ts_r.month - _LD_TS_R.month)
            _pre_ann = ((_s_r[_LD_TS_R] / _s_r[_PRE_TS_R]) ** (12 / _m_pre) - 1) * 100
            _post_ann = ((_s_r[_latest_ts_r] / _s_r[_LD_TS_R]) ** (12 / _m_post) - 1) * 100
            _row0_r = _bls[_bls["sid"] == _sid_r].iloc[0]
            _accel_rows_r.append({"name": _row0_r["name"], "group": _row0_r["group"],
                                  "pre": _pre_ann, "post": _post_ann, "accel": _post_ann - _pre_ann})
    _accel_df_r = pd.DataFrame(_accel_rows_r)

    _head_r = _accel_df_r[_accel_df_r["group"] == "headline"].iloc[0]
    _comp_r = _accel_df_r[_accel_df_r["name"] == "Computers & peripherals"].iloc[0]
    _app_r  = _accel_df_r[_accel_df_r["name"] == "Apparel"].iloc[0]
    _svc_acc_r = _accel_df_r[_accel_df_r["group"] == "service"]["accel"].mean()

    _bk1, _bk2, _bk3, _bk4 = st.columns(4)
    with _bk1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Headline inflation</div>
          <div class="kpi-value negative" style="font-size:24px">{_head_r["pre"]:.1f}% → {_head_r["post"]:.1f}%</div>
          <div class="kpi-sub">annualized, before → after Liberation Day</div>
        </div>""", unsafe_allow_html=True)
    with _bk2:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Computers flipped to inflating</div>
          <div class="kpi-value negative" style="font-size:24px">{_comp_r["accel"]:+.1f}pp</div>
          <div class="kpi-sub">{_comp_r["pre"]:+.1f}%/yr before → {_comp_r["post"]:+.1f}%/yr after</div>
        </div>""", unsafe_allow_html=True)
    with _bk3:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Apparel acceleration</div>
          <div class="kpi-value negative" style="font-size:24px">{_app_r["accel"]:+.1f}pp</div>
          <div class="kpi-sub">{_app_r["pre"]:+.1f}%/yr before → {_app_r["post"]:+.1f}%/yr after</div>
        </div>""", unsafe_allow_html=True)
    with _bk4:
        _svc_cls_r = "positive" if _svc_acc_r < 0 else "warning"
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Domestic services (control group)</div>
          <div class="kpi-value {_svc_cls_r}" style="font-size:24px">{_svc_acc_r:+.1f}pp</div>
          <div class="kpi-sub">services cooled — the rise is goods-specific</div>
        </div>""", unsafe_allow_html=True)

    # ── #1 Which retail prices changed since Liberation Day? ────────────────
    st.markdown('<div class="section-header">Which retail prices changed since Liberation Day?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Official BLS price change per category over the 14 months after Liberation Day (March 2025 to May 2026). Apparel and household furnishings lead the increases; vehicles stayed flat as dealers absorbed costs and sold pre-tariff inventory.</div>', unsafe_allow_html=True)

    _chg_rows_r = []
    for _sid_r in _bls["sid"].unique():
        _s_r = _bls[_bls["sid"] == _sid_r].set_index("date")["value"]
        if _LD_TS_R in _s_r.index and _latest_ts_r in _s_r.index:
            _row0 = _bls[_bls["sid"] == _sid_r].iloc[0]
            _chg_rows_r.append({"name": _row0["name"], "group": _row0["group"],
                                "chg": (_s_r[_latest_ts_r] / _s_r[_LD_TS_R] - 1) * 100})
    _chg_df_r = pd.DataFrame(_chg_rows_r).sort_values("chg", ascending=False)
    _chg_colors_r = ["#f87171" if g == "exposed" else "#e2e8f0" if g == "headline" else "#38bdf8"
                     for g in _chg_df_r["group"]]
    fig_bls = go.Figure(go.Bar(
        x=_chg_df_r["chg"], y=_chg_df_r["name"],
        orientation="h", marker_color=_chg_colors_r,
        text=[f"{v:+.1f}%" for v in _chg_df_r["chg"]], textposition="outside",
    ))
    fig_bls.add_vline(x=0, line_color="#4b5563", line_width=1)
    fig_bls.update_layout(**PLOTLY_THEME, height=400,
        title="Price Change Since Liberation Day (Mar 2025 → May 2026, official BLS)")
    fig_bls.update_xaxes(title_text="% change in prices", range=[-5, 6])
    fig_bls.update_yaxes(tickfont=dict(size=11))
    st.plotly_chart(fig_bls, use_container_width=True)
    _explain("One bar per category: how much prices rose or fell in the 14 months after Liberation Day. Red = tariff-exposed goods, white = overall CPI, blue = domestic services. Read top to bottom - the biggest increases are almost all imported goods.")

    # ── #2 Did tariff-exposed goods accelerate more than services? ──────────
    st.markdown('<div class="section-header">Did tariff-exposed goods accelerate more than services?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">The deeper test: each category\'s inflation rate after Liberation Day compared with its own pre-tariff trend. If the price rise were general inflation, goods and services would accelerate together. Instead the acceleration concentrates in imported goods — computers swung +6.1pp (from falling prices to rising) — while untariffed services cooled. That divergence is the tariff signature.</div>', unsafe_allow_html=True)

    _acc_plot_r = _accel_df_r[_accel_df_r["group"] != "headline"].sort_values("accel")
    fig_acc = go.Figure(go.Bar(
        x=_acc_plot_r["accel"], y=_acc_plot_r["name"],
        orientation="h",
        marker_color=["#f87171" if g == "exposed" else "#38bdf8" for g in _acc_plot_r["group"]],
        text=[f"{v:+.1f}pp" for v in _acc_plot_r["accel"]], textposition="outside",
    ))
    fig_acc.add_vline(x=0, line_color="#4b5563", line_width=1)
    fig_acc.update_layout(**PLOTLY_THEME, height=400,
        title="Inflation Acceleration After Liberation Day (red = tariff-exposed goods, blue = services)")
    fig_acc.update_xaxes(title_text="Change in annualized inflation vs pre-tariff trend (pp)", range=[-7, 8])
    fig_acc.update_yaxes(tickfont=dict(size=11))
    st.plotly_chart(fig_acc, use_container_width=True)
    _explain("Each bar is how much a category\'s yearly inflation rate changed relative to its own pre-Liberation Day trend. Red bars to the right = tariffed goods speeding up. Blue bars to the left = domestic services slowing down. The split pattern is what identifies tariffs, rather than general inflation, as the cause.")


    # ── How did prices actually change? ─────────────────────────────────────
    st.markdown('<div class="section-header">Daily retail-price movement around Liberation Day (Cavallo tracker)</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Prices indexed to Oct 2024 = 0%. After Liberation Day (Apr 2, 2025) US prices rose while Chinese export prices fell — Chinese manufacturers cut prices to stay competitive despite the 54% US tariff.</div>', unsafe_allow_html=True)

    fig_cav = go.Figure()
    country_traces_t3 = [
        ("index_usa",    "United States", "#2563eb", 2.5),
        ("index_canada", "Canada",        "#22d3a0", 1.5),
        ("index_mexico", "Mexico",        "#fbbf24", 1.5),
        ("index_china",  "China",         "#f87171", 1.5),
    ]
    for col_name, name, color, width in country_traces_t3:
        pct = (cavallo[col_name] - 1) * 100
        fig_cav.add_trace(go.Scatter(
            x=cavallo["date"], y=pct, name=name,
            line=dict(color=color, width=width),
            hovertemplate=f"<b>{name}</b><br>%{{x|%b %d, %Y}}<br>%{{y:.3f}}%<extra></extra>"
        ))
    fig_cav.add_vline(x="2025-04-02", line_dash="dash", line_color="#f87171",
        annotation_text="Liberation Day (Apr 2)", annotation_position="top right",
        annotation_font_color="#f87171")
    fig_cav.update_layout(**PLOTLY_THEME, height=380,
        title="Daily Price Tracker vs Trading Partners (Oct 2024 = 0%)",
        yaxis_title="% Change from Baseline",
        legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250"))
    st.plotly_chart(fig_cav, use_container_width=True)
    _explain("Daily online prices from the Cavallo et al. research tracker, indexed to October 2024. After Liberation Day the US line drifts up while China's falls - Chinese exporters cut prices to stay competitive.")


    # ── Who pays more — rich or poor? ─────────────────────────────────────
    st.markdown('<div class="section-header">Who pays more — rich or poor households?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">The poorest fifth of households face an 8.4% consumer price burden vs 5.9% for the wealthiest (no retaliation scenario). With retaliation, all quintiles pay less but the US economy shrinks.</div>', unsafe_allow_html=True)

    quintile_labels_t3 = ["Lowest 20%", "Lower-Middle", "Middle", "Upper-Middle", "Top 20%"]
    c_q1, c_q2 = st.columns(2)

    with c_q1:
        fig_q = go.Figure()
        fig_q.add_trace(go.Bar(
            name="No Retaliation",
            x=quintile_labels_t3, y=q_noretal,
            marker_color="#f87171",
            text=[f"{v:.2f}%" for v in q_noretal], textposition="outside",
        ))
        fig_q.add_trace(go.Bar(
            name="With Retaliation",
            x=quintile_labels_t3, y=q_retal,
            marker_color="#fbbf24",
            text=[f"{v:.2f}%" for v in q_retal], textposition="outside",
        ))
        fig_q.update_layout(**PLOTLY_THEME, height=340,
            title="Consumer Price Burden by Income Group",
            barmode="group", yaxis_title="% Burden",
            legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250"))
        st.plotly_chart(fig_q, use_container_width=True)
        _explain("Consumer price burden by income group under two worlds - with and without foreign retaliation. In both, the poorest fifth pays the largest share of income. Retaliation lowers prices but shrinks the economy.")

    with c_q2:
        qdf_table = pd.DataFrame({
            "Household Group":   quintile_labels_t3,
            "No Retaliation":    [f"{v:.2f}%" for v in q_noretal],
            "With Retaliation":  [f"{v:.2f}%" for v in q_retal],
            "Difference":        [f"{(n-r):+.2f}pp" for n, r in zip(q_noretal, q_retal)],
        })
        st.markdown("<br><b>Detailed burden table</b>", unsafe_allow_html=True)
        st.dataframe(qdf_table, use_container_width=True, hide_index=True)

    # ── What if other countries retaliate? ─────────────────────────────────
    st.markdown('<div class="section-header">What if other countries retaliate?</div>', unsafe_allow_html=True)
    c_r1, c_r2, c_r3, c_r4 = st.columns(4)
    with c_r1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Consumer Prices — No Retaliation</div>
          <div class="kpi-value negative" style="font-size:26px">+{ge_cpi:.1f}%</div>
          <div class="kpi-sub">US CPI increase</div>
        </div>""", unsafe_allow_html=True)
    with c_r2:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Consumer Prices — With Retaliation</div>
          <div class="kpi-value warning" style="font-size:26px">+{ge_cpi_retal:.1f}%</div>
          <div class="kpi-sub">Lower prices, but economy shrinks</div>
        </div>""", unsafe_allow_html=True)
    with c_r3:
        welfare_cls = "positive" if ge_welfare > 0 else "negative"
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">US Living Standards — No Retaliation</div>
          <div class="kpi-value {welfare_cls}" style="font-size:26px">{ge_welfare:+.2f}%</div>
          <div class="kpi-sub">Real income change</div>
        </div>""", unsafe_allow_html=True)
    with c_r4:
        welfare_r_cls = "positive" if ge_welfare_r > 0 else "negative"
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">US Living Standards — With Retaliation</div>
          <div class="kpi-value {welfare_r_cls}" style="font-size:26px">{ge_welfare_r:+.2f}%</div>
          <div class="kpi-sub">Real income change</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('<div class="insight-box">If trading partners retaliate, US consumer prices actually rise <i>less</i> (because import demand falls and exporters cut prices to compete). However, the US economy shrinks overall because US export markets shut down in response. <b>Retaliation may reduce measured consumer-price pressure while simultaneously shrinking income and economic activity — lower CPI here is not an unambiguous benefit.</b></div>', unsafe_allow_html=True)

    # ── Sector verdict ───────────────────────────────────────────────────────
    st.markdown(
        '<div style="background:linear-gradient(135deg,#141824,#1a1d2e);border:2px solid #f87171;border-radius:12px;padding:22px 26px;margin-top:30px">'
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">'
        '<span style="background:#2a0f0f;color:#f87171;font-size:11px;font-weight:800;letter-spacing:2px;padding:4px 12px;border-radius:20px">THE BOTTOM LINE</span>'
        '<span style="color:#f1f5f9;font-size:19px;font-weight:700">Did the tariffs work for retail consumers? Not really. They ended up paying the bill.</span>'
        '</div>'
        '<div style="display:flex;gap:10px;margin-bottom:8px;align-items:baseline"><span style="color:#f87171;font-weight:800;font-size:14px">✗</span><span style="color:#cbd5e1;font-size:13px;line-height:1.55">Prices of tariffed goods rose in a visible way. Apparel and furnishings climbed about 4 percent in 14 months, and computers went from years of falling prices to rising ones, while untariffed services cooled off. That pattern points to the tariffs rather than general inflation.</span></div><div style="display:flex;gap:10px;margin-bottom:8px;align-items:baseline"><span style="color:#fbbf24;font-weight:800;font-size:14px">◐</span><span style="color:#cbd5e1;font-size:13px;line-height:1.55">The good news is that it could have been worse. Only about 30 percent of tariff costs reached store shelves, because exporters cut their prices and retailers absorbed part of the hit.</span></div><div style="display:flex;gap:10px;margin-bottom:8px;align-items:baseline"><span style="color:#f87171;font-weight:800;font-size:14px">✗</span><span style="color:#cbd5e1;font-size:13px;line-height:1.55">The burden was uneven. The lowest income households carried about 1.4 times the price burden of the highest, which makes the tariff work like a consumption tax on the people least able to pay it.</span></div>'
        '<div style="color:#94a3b8;font-size:13px;margin-top:10px;padding-top:10px;border-top:1px solid #2d3250;line-height:1.55">Retail is where tariff costs reach everyday households. Shoppers received no real benefit in return, though the increase was smaller and slower than many feared.</div>'
        '</div>', unsafe_allow_html=True)





# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — MANUFACTURING EXPOSURE
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    naics, price_idx, shocks, hts, mfg_stats = load_manufacturing()

    # Measured post-Liberation Day data (USITC customs + BEA quarterly output)
    _imp_rl, _dut_rl, _bea_rl = load_mfg_reality()
    _mv_rl = _imp_rl.groupby("date")["value"].sum().rename("value").to_frame()
    _mv_rl["duty"] = _dut_rl.groupby("date")["value"].sum()
    _mv_rl["rate"] = _mv_rl["duty"] / _mv_rl["value"] * 100
    _rate_2024_rl = float(_mv_rl.loc[(_mv_rl.index >= "2024-01-01") & (_mv_rl.index <= "2024-12-01"), "rate"].mean())
    _rate_peak_rl = float(_mv_rl.loc[_mv_rl.index >= "2025-04-01", "rate"].max())

    kpi4 = [
        ("Average tariff on factory imports",     f"{mfg_stats['tau_mfg_avg']*100:.1f}%",          "negative", "Trade-weighted avg"),
        ("Imports as % of factory output",        f"{mfg_stats['import_penetration_mfg']*100:.1f}%","warning",  "Import penetration"),
        ("Factory tariffs' share of price rises", f"+{mfg_stats['cpi_mfg_contribution']:.1f}pp",   "negative", "Of +7.1pp total CPI"),
        ("Supply chain multiplier",               f"{mfg_stats['io_mult_mfg']:.2f}×",              "warning",  "IO amplification"),
        ("Effective tariff actually paid",        f"{_rate_2024_rl:.1f}% → {_rate_peak_rl:.0f}%",  "negative", "US customs collections, 2024 → 2025 peak"),
        ("Pre-tariff HTS8 avg rate",              f"{mfg_stats['hts8_mfg_rate']*100:.1f}%",        "neutral",  "Before Liberation Day"),
    ]
    cols4 = st.columns(6)
    for col, (label, val, cls, sub) in zip(cols4, kpi4):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value {cls}" style="font-size:22px">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Country spotlight for Manufacturing tab ────────────────────────────────
    if _country_profile:
        cp = _country_profile
        _ct_row_mfg = _g_tariff_df[_g_tariff_df["CountryName"] == cp["country"]]
        _mfg_tariff = float(_ct_row_mfg["tariff_pct"].iloc[0]) if not _ct_row_mfg.empty else cp["tariff"]

        # US manufacturers' perspective: this country is a supplier/importer
        _imp_dir = "more" if cp["imp"] > 0 else "less"
        _exp_dir = "more" if cp["exp"] > 0 else "less"
        _emp_dir = "gained" if cp["emp"] > 0 else "lost"

        # Estimated cost increase for US factories importing from this country
        _cost_increase = _mfg_tariff * mfg_stats.get("import_penetration_mfg", 0.323)

        _cells_t4 = [
            f'<div style="background:#0f172a;border-radius:8px;padding:10px"><div style="color:#64748b;font-size:10px">US tariff on goods from here</div><div style="color:#f87171;font-size:22px;font-weight:700">{_mfg_tariff:.0f}%</div><div style="color:#475569;font-size:10px">rate US factories pay to import</div></div>',
            f'<div style="background:#0f172a;border-radius:8px;padding:10px"><div style="color:#64748b;font-size:10px">Input cost uplift (est.)</div><div style="color:#fbbf24;font-size:22px;font-weight:700">+{_cost_increase:.1f}pp</div><div style="color:#475569;font-size:10px">tariff × import penetration</div></div>',
            f'<div style="background:#0f172a;border-radius:8px;padding:10px"><div style="color:#64748b;font-size:10px">This country imports {_imp_dir} from US</div><div style="color:#60a5fa;font-size:22px;font-weight:700">{cp["imp"]:+.1f}%</div><div style="color:#475569;font-size:10px">their import vol change (GE model)</div></div>',
            f'<div style="background:#0f172a;border-radius:8px;padding:10px"><div style="color:#64748b;font-size:10px">Jobs {_emp_dir} there</div><div style="color:{"#22d3a0" if cp["emp"]>0 else "#f87171"};font-size:22px;font-weight:700">{cp["emp"]:+.2f}%</div><div style="color:#475569;font-size:10px">employment change (GE model)</div></div>',
        ]
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1a0d0d,#1a1d2e);border:2px solid #fb923c;border-radius:12px;padding:16px;margin-bottom:16px">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">'
            f'<div style="background:#fb923c;color:#0d0d0d;font-size:11px;font-weight:800;letter-spacing:1px;padding:3px 10px;border-radius:20px">MANUFACTURING FOCUS</div>'
            f'<div style="color:#e2e8f0;font-size:20px;font-weight:700">{cp["country"]}</div>'
            f'<div style="color:#64748b;font-size:13px;margin-left:auto">US factory import tariff: <span style="color:#f87171;font-weight:700">{_mfg_tariff:.0f}%</span></div>'
            f'</div>'
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px">'
            + "".join(_cells_t4) +
            f'</div>'
            f'<div style="margin-top:10px;color:#64748b;font-size:12px;line-height:1.5">'
            f'US factories importing from {cp["country"]} now pay a {_mfg_tariff:.0f}% tariff on those inputs. '
            f'At 32% average import penetration, this translates to an estimated {_cost_increase:.1f}pp cost uplift on factory inputs sourced from {cp["country"]}. '
            f'The industrial tariff shock affects the entire supply chain through the IO multiplier ({mfg_stats.get("io_mult_mfg", 1.09):.2f}×).'
            f'</div>'
            f'</div>', unsafe_allow_html=True)

    # ── Measured impact at a glance (USITC + BEA) ────────────────────────────
    st.markdown('<div style="font-size:13px;font-weight:700;color:#94a3b8;letter-spacing:1px;margin:18px 0 6px 0">⚡ AT A GLANCE — USITC CUSTOMS + BEA OUTPUT, MEASURED</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Everything below is measured data — USITC customs records (monthly imports and calculated duties through Dec 2025) and BEA quarterly output (through Q1 2026). No model outputs on this tab.</div>', unsafe_allow_html=True)

    _CH_NAMES_RL = {39: "Plastics", 72: "Iron & Steel", 73: "Steel Articles",
                    84: "Machinery", 85: "Electronics", 87: "Vehicles",
                    94: "Furniture", 95: "Toys & Sports"}

    _pre_rl  = _imp_rl[(_imp_rl["date"] >= "2024-04-01") & (_imp_rl["date"] <= "2024-12-01")]
    _post_rl = _imp_rl[(_imp_rl["date"] >= "2025-04-01") & (_imp_rl["date"] <= "2025-12-01")]
    _tot_chg_rl = (_post_rl["value"].sum() / _pre_rl["value"].sum() - 1) * 100
    _dpre_rl  = _dut_rl[(_dut_rl["date"] >= "2024-04-01") & (_dut_rl["date"] <= "2024-12-01")]["value"].sum()
    _dpost_rl = _dut_rl[(_dut_rl["date"] >= "2025-04-01") & (_dut_rl["date"] <= "2025-12-01")]["value"].sum()
    _duty_mult_rl = _dpost_rl / max(_dpre_rl, 1)
    _bea_mfg_rl = _bea_rl[_bea_rl["industry"] == "Manufacturing"]
    _bea_chg_rl = float((_bea_mfg_rl["2026Q1"].iloc[0] / _bea_mfg_rl["2025Q1"].iloc[0] - 1) * 100) if not _bea_mfg_rl.empty else 0

    _rc1, _rc2, _rc3, _rc4 = st.columns(4)
    with _rc1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Calculated duties since tariffs (USITC)</div>
          <div class="kpi-value negative" style="font-size:26px">{_duty_mult_rl:.1f}×</div>
          <div class="kpi-sub">${_dpost_rl/1e9:,.0f}B Apr–Dec 2025 vs ${_dpre_rl/1e9:,.0f}B in 2024</div>
        </div>""", unsafe_allow_html=True)
    with _rc2:
        _tc_cls = "positive" if _tot_chg_rl > 0 else "negative"
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Total mfg imports (Apr–Dec 2025 vs 2024)</div>
          <div class="kpi-value {_tc_cls}" style="font-size:26px">{_tot_chg_rl:+.1f}%</div>
          <div class="kpi-sub">USITC customs data, 8 mfg chapters</div>
        </div>""", unsafe_allow_html=True)
    with _rc3:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Effective tariff rate jump</div>
          <div class="kpi-value negative" style="font-size:26px">{_rate_2024_rl:.1f}% → {_rate_peak_rl:.0f}%</div>
          <div class="kpi-sub">duties ÷ import value, actual collections</div>
        </div>""", unsafe_allow_html=True)
    with _rc4:
        _bc_cls = "positive" if _bea_chg_rl > 0 else "negative"
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Nominal factory output since tariffs</div>
          <div class="kpi-value {_bc_cls}" style="font-size:26px">{_bea_chg_rl:+.1f}%</div>
          <div class="kpi-sub">BEA nominal output, 2025Q1 → 2026Q1</div>
        </div>""", unsafe_allow_html=True)

    # ── #1 Manufacturing imports, year by year ───────────────────────────────
    st.markdown('<div class="section-header">Manufacturing imports, year by year</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Annual US manufacturing imports, with 2025 as the tariff year. The total barely moved, but the mix underneath changed: machinery kept growing on data-center demand, while tariff-heavy vehicles and steel shrank.</div>', unsafe_allow_html=True)

    _yr_groups = [
        ("Machinery", [84], "#22d3a0"),
        ("Electronics", [85], "#a78bfa"),
        ("Vehicles", [87], "#fbbf24"),
        ("Steel", [72, 73], "#f87171"),
        ("Other (plastics, furniture, toys)", [39, 94, 95], "#475569"),
    ]
    _imp_yr = _imp_rl.copy()
    _imp_yr["year"] = _imp_yr["date"].dt.year
    _years_b4 = sorted(_imp_yr["year"].unique())

    fig_mi = go.Figure()
    _totals_yr = [_imp_yr[_imp_yr["year"] == y]["value"].sum() / 1e9 for y in _years_b4]
    fig_mi.add_trace(go.Scatter(
        name="Total (8 products)", x=[str(y) for y in _years_b4], y=_totals_yr,
        line=dict(color="#e2e8f0", width=3.5), mode="lines+markers+text",
        text=[f"${v:,.0f}B" for v in _totals_yr], textposition="top center",
        textfont=dict(color="#e2e8f0", size=12),
    ))
    for _gname, _chs, _clr in _yr_groups:
        _vals = [_imp_yr[(_imp_yr["year"] == y) & (_imp_yr["chapter"].isin(_chs))]["value"].sum() / 1e9
                 for y in _years_b4]
        fig_mi.add_trace(go.Scatter(
            name=_gname, x=[str(y) for y in _years_b4], y=_vals,
            line=dict(color=_clr, width=2), mode="lines+markers+text",
            text=[f"{v:,.0f}" for v in _vals], textposition="top center",
            textfont=dict(color=_clr, size=10),
        ))
    fig_mi.add_vline(x=2.75, line_dash="dash", line_color="#f87171",
        annotation_text="Liberation Day (Apr 2025)", annotation_position="top left",
        annotation_font_color="#f87171")
    fig_mi.update_layout(**PLOTLY_THEME, height=420,
        title="US Manufacturing Imports by Year ($B, USITC customs value; 2025 = tariff year)",
        legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250"))
    fig_mi.update_xaxes(type="category")
    fig_mi.update_yaxes(title_text="Imports ($B/year)")
    st.plotly_chart(fig_mi, use_container_width=True)
    _explain("The white line is total manufacturing imports per year, with the product groups below it. Compare 2025 with 2024: the total held steady, yet machinery (green) kept climbing while vehicles (yellow) and steel (red) declined. The tariffs reshaped what America buys more than how much.")

    # ── #2 The tariff rate factories actually paid ──────────────────────────
    st.markdown('<div class="section-header">The tariff rate factories actually paid</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Calculated duties divided by import value, month by month — the average tariff importers really faced after every exemption and carve-out. It quintupled within months of Liberation Day, yet stayed far below the ~25% headline announcements.</div>', unsafe_allow_html=True)

    _mv_plot_rl = _mv_rl.reset_index().sort_values("date")
    fig_rate_rl = go.Figure(go.Scatter(
        x=_mv_plot_rl["date"], y=_mv_plot_rl["rate"],
        line=dict(color="#f87171", width=2.5),
        fill="tozeroy", fillcolor="rgba(248,113,113,0.12)",
    ))
    fig_rate_rl.add_vline(x="2025-04-02", line_dash="dash", line_color="#e2e8f0",
        annotation_text="Liberation Day", annotation_position="top left",
        annotation_font_color="#e2e8f0")
    fig_rate_rl.update_layout(**PLOTLY_THEME, height=360,
        title="Effective Tariff Rate on Manufacturing Imports (calculated duties ÷ customs value)")
    fig_rate_rl.update_yaxes(title_text="Effective rate (%)")
    st.plotly_chart(fig_rate_rl, use_container_width=True)
    _explain("The average duty rate actually paid on the eight manufacturing categories. Flat around 2-3% for years, then a cliff at the red line - proof the tariffs were real, and a measure of how much exemptions softened the headline rates.")

    # ── #3 Import change by product: who fell, who rose ─────────────────────
    st.markdown('<div class="section-header">Import change by product: who fell, who rose</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Each product category\'s imports in the nine months after Liberation Day versus the same months a year earlier (seasonality-controlled). The tariffs bit hardest exactly where they were aimed — steel, toys, vehicles — while machinery rose on AI data-center demand America cannot source domestically.</div>', unsafe_allow_html=True)

    _chg_rows_rl = []
    for _ch_rl in sorted(_imp_rl["chapter"].unique()):
        _p1_rl = _pre_rl[_pre_rl["chapter"] == _ch_rl]["value"].sum()
        _p2_rl = _post_rl[_post_rl["chapter"] == _ch_rl]["value"].sum()
        if _p1_rl > 0:
            _chg_rows_rl.append({"name": _CH_NAMES_RL.get(_ch_rl, str(_ch_rl)),
                                 "chg": (_p2_rl / _p1_rl - 1) * 100})
    _chg_df_rl = pd.DataFrame(_chg_rows_rl).sort_values("chg")
    fig_chg_rl = go.Figure(go.Bar(
        x=_chg_df_rl["chg"], y=_chg_df_rl["name"],
        orientation="h",
        marker_color=["#f87171" if v < 0 else "#22d3a0" for v in _chg_df_rl["chg"]],
        text=[f"{v:+.1f}%" for v in _chg_df_rl["chg"]], textposition="outside",
    ))
    fig_chg_rl.add_vline(x=0, line_color="#4b5563", line_width=1)
    fig_chg_rl.update_layout(**PLOTLY_THEME, height=380,
        title="Change in Imports by Product (Apr–Dec 2025 vs Apr–Dec 2024, USITC)")
    fig_chg_rl.update_xaxes(title_text="% change", range=[-40, 40])
    st.plotly_chart(fig_chg_rl, use_container_width=True)
    _explain("Red bars fell after the tariffs - steel, toys, vehicles and furniture, the directly targeted goods. The one big green bar is machinery, which rose 26% despite tariffs. Same months compared in both years, so seasonality is controlled.")


    # ── Which products face the highest tariffs now? (measured) ─────────────
    st.markdown('<div class="section-header">⚡ Which products got hit hardest — the tariff rate actually paid</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">The effective tariff rate importers really paid on each manufacturing product category: USITC calculated duties divided by customs value, comparing April–December 2024 with the same months of 2025. Fully measured — this is the tariff wall as it actually landed, after every exemption and carve-out. Steel articles jumped hardest; machinery, despite huge duty totals, carries one of the lowest rates because its import base is enormous.</div>', unsafe_allow_html=True)

    _er_rows4 = []
    for _ch_e in sorted(_imp_rl["chapter"].unique()):
        _v_pre_e = float(_imp_rl[(_imp_rl["chapter"] == _ch_e) & (_imp_rl["date"] >= "2024-04-01") & (_imp_rl["date"] <= "2024-12-01")]["value"].sum())
        _v_post_e = float(_imp_rl[(_imp_rl["chapter"] == _ch_e) & (_imp_rl["date"] >= "2025-04-01") & (_imp_rl["date"] <= "2025-12-01")]["value"].sum())
        _d_pre_e = float(_dut_rl[(_dut_rl["chapter"] == _ch_e) & (_dut_rl["date"] >= "2024-04-01") & (_dut_rl["date"] <= "2024-12-01")]["value"].sum())
        _d_post_e = float(_dut_rl[(_dut_rl["chapter"] == _ch_e) & (_dut_rl["date"] >= "2025-04-01") & (_dut_rl["date"] <= "2025-12-01")]["value"].sum())
        if _v_pre_e > 0 and _v_post_e > 0:
            _er_rows4.append({"name": _CH_NAMES_RL.get(_ch_e, str(_ch_e)),
                              "pre": _d_pre_e / _v_pre_e * 100,
                              "post": _d_post_e / _v_post_e * 100})
    _er_df4 = pd.DataFrame(_er_rows4).sort_values("post", ascending=False)

    fig_er4 = go.Figure()
    fig_er4.add_trace(go.Bar(
        name="Apr–Dec 2024 (before)", y=_er_df4["name"], x=_er_df4["pre"],
        orientation="h", marker_color="#475569",
        text=[f"{v:.1f}%" for v in _er_df4["pre"]], textposition="outside",
    ))
    fig_er4.add_trace(go.Bar(
        name="Apr–Dec 2025 (after)", y=_er_df4["name"], x=_er_df4["post"],
        orientation="h", marker_color="#f87171",
        text=[f"{v:.1f}%" for v in _er_df4["post"]], textposition="outside",
    ))
    fig_er4.update_layout(**PLOTLY_THEME, height=420,
        title="Effective Tariff Rate Paid by Product Category (calculated duties ÷ customs value)",
        barmode="group",
        legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250"))
    fig_er4.update_xaxes(title_text="Effective rate (%)")
    fig_er4.update_yaxes(autorange="reversed", tickfont=dict(size=11))
    st.plotly_chart(fig_er4, use_container_width=True)
    _explain("Gray = the duty rate each product category actually carried before Liberation Day, red = after. Steel articles and toys saw the steepest jumps; vehicles and furniture also multiplied. Rates are USITC calculated duties over customs value for the same nine months of each year, so they reflect every exemption, carve-out and trade measure actually in force - not the headline announcement.")

    # ── How tariff costs travel through supply chains to consumer prices ────
    st.markdown('<div class="section-header">How tariff costs travel through supply chains to reach consumer prices</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">A tariff does not stop at the port. A factory pays more for imported steel directly, and then pays more again for every part made with that steel. The model captures this with a supply chain multiplier of <b>1.09</b>, and the result is striking: manufacturing alone accounts for <b>6.8 points of the 7.1 point</b> rise in consumer prices. In other words, the supply chain is the main road tariff costs take on their way to shoppers.</div>', unsafe_allow_html=True)

    _amp_stages = [
        "Direct tariff on<br>factory imports",
        f"After supply chain<br>amplification (×{mfg_stats['io_mult_mfg']:.2f})",
        "Manufacturing share of the<br>total consumer price rise",
    ]
    _amp_vals = [
        mfg_stats["tau_mfg_avg"] * 100,
        mfg_stats["tau_mfg_avg"] * 100 * mfg_stats["io_mult_mfg"],
        mfg_stats["cpi_mfg_contribution"],
    ]
    _amp_notes = [
        f"{_amp_vals[0]:.1f}%",
        f"{_amp_vals[1]:.1f}%",
        f"{_amp_vals[2]:.1f}pp of 7.1pp",
    ]
    fig_amp = go.Figure(go.Bar(
        x=_amp_stages, y=_amp_vals,
        marker_color=["#2563eb", "#fbbf24", "#f87171"],
        text=_amp_notes, textposition="outside",
    ))
    fig_amp.update_layout(**PLOTLY_THEME, height=340,
        title="From the Port to the Checkout: How the Tariff Cost Builds")
    fig_amp.update_yaxes(title_text="% impact")
    st.plotly_chart(fig_amp, use_container_width=True)
    _explain("Read the bars left to right. The first is the average tariff factories pay on imported inputs. The second adds the ripple effect through supply chains, since parts made with tariffed materials also cost more. The third shows where it ends up: manufacturing supplies nearly all of the total rise in consumer prices, about 96 percent of it.")

    # ── #5 Nominal US factory output after Liberation Day ───────────────────
    st.markdown('<div class="section-header">Nominal US factory output after Liberation Day</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box"><b>Nominal</b> BEA output indexed to Q1 2025 = 100 — the index combines changes in production volume AND tariff-driven price increases, so growth here is not proof of a production boom. Primary metals (steel) rose most, consistent with tariff protection; motor vehicles dipped in late 2025.</div>', unsafe_allow_html=True)

    _bea_keys_rl = ["Manufacturing", "Primary metals", "Motor vehicles, bodies and trailers, and parts",
                    "Machinery", "Computer and electronic products", "Chemical products"]
    _bea_labels_rl = {"Manufacturing": "All Manufacturing", "Primary metals": "Primary Metals (steel)",
                      "Motor vehicles, bodies and trailers, and parts": "Motor Vehicles",
                      "Machinery": "Machinery", "Computer and electronic products": "Computers & Electronics",
                      "Chemical products": "Chemicals"}
    _qtrs_rl = ["2024Q1","2024Q2","2024Q3","2024Q4","2025Q1","2025Q2","2025Q3","2025Q4","2026Q1"]
    fig_bea_rl = go.Figure()
    _colors_bea_rl = ["#e2e8f0","#f87171","#fbbf24","#22d3a0","#a78bfa","#38bdf8"]
    for _i_rl, _k_rl in enumerate(_bea_keys_rl):
        _row_rl = _bea_rl[_bea_rl["industry"] == _k_rl]
        if _row_rl.empty:
            continue
        _base_rl = float(_row_rl["2025Q1"].iloc[0])
        _vals_rl = [float(_row_rl[q].iloc[0]) / _base_rl * 100 for q in _qtrs_rl]
        fig_bea_rl.add_trace(go.Scatter(
            x=_qtrs_rl, y=_vals_rl, name=_bea_labels_rl[_k_rl],
            line=dict(color=_colors_bea_rl[_i_rl % len(_colors_bea_rl)],
                      width=3 if _k_rl == "Manufacturing" else 1.6),
        ))
    fig_bea_rl.add_vline(x=4.5, line_dash="dash", line_color="#f87171",
        annotation_text="Liberation Day", annotation_position="top left",
        annotation_font_color="#f87171")
    fig_bea_rl.update_layout(**PLOTLY_THEME, height=380,
        title="Nominal Factory Output by Industry (BEA quarterly, 2025Q1 = 100)",
        legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250", font=dict(size=10)))
    fig_bea_rl.update_yaxes(title_text="Index (2025Q1 = 100)")
    st.plotly_chart(fig_bea_rl, use_container_width=True)
    _explain("US factory output by industry in nominal dollars, indexed so Q1 2025 = 100. Lines above 100 grew after the tariffs - but remember part of that growth is higher prices, not more production. Steel (red) benefited most from protection; motor vehicles (yellow) dipped in late 2025.")

    # ── Sector verdict ───────────────────────────────────────────────────────
    st.markdown(
        '<div style="background:linear-gradient(135deg,#141824,#1a1d2e);border:2px solid #fbbf24;border-radius:12px;padding:22px 26px;margin-top:30px">'
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">'
        '<span style="background:#2a230f;color:#fbbf24;font-size:11px;font-weight:800;letter-spacing:2px;padding:4px 12px;border-radius:20px">THE BOTTOM LINE</span>'
        '<span style="color:#f1f5f9;font-size:19px;font-weight:700">Did the tariffs work for manufacturing? Partly. Protection yes, revival not yet.</span>'
        '</div>'
        '<div style="display:flex;gap:10px;margin-bottom:8px;align-items:baseline"><span style="color:#22d3a0;font-weight:800;font-size:14px">✓</span><span style="color:#cbd5e1;font-size:13px;line-height:1.55">Where the tariffs were aimed, they landed. Steel imports fell 24 percent, vehicles 18 percent and toys 21 percent, and protected industries such as primary metals posted the strongest output gains.</span></div><div style="display:flex;gap:10px;margin-bottom:8px;align-items:baseline"><span style="color:#22d3a0;font-weight:800;font-size:14px">✓</span><span style="color:#cbd5e1;font-size:13px;line-height:1.55">The revenue was real. Calculated duties ran nearly five times higher, about 146 billion dollars in nine months on these categories, and the rate importers actually paid rose from around 2 percent to about 13 percent.</span></div><div style="display:flex;gap:10px;margin-bottom:8px;align-items:baseline"><span style="color:#f87171;font-weight:800;font-size:14px">✗</span><span style="color:#cbd5e1;font-size:13px;line-height:1.55">A broad return of manufacturing has not shown up in the data yet. Total imports stayed roughly flat as trade rerouted, machinery imports rose 26 percent, and part of the output growth reflects higher prices rather than more production.</span></div>'
        '<div style="color:#94a3b8;font-size:13px;margin-top:10px;padding-top:10px;border-top:1px solid #2d3250;line-height:1.55">The tariffs worked as targeted protection and as a source of revenue. What the data does not yet show is a wider manufacturing revival, since imports changed shape rather than shrinking.</div>'
        '</div>', unsafe_allow_html=True)











# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — AI ANALYST
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    import anthropic as _anthropic
    import json as _json
    import sys as _sys

    # ── Anthropic client ──────────────────────────────────────────────────
    _api_key = None
    try:
        _api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        _api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not _api_key:
        st.error("No Anthropic API key found. Set `ANTHROPIC_API_KEY` in `.streamlit/secrets.toml` or as an environment variable.")
        st.stop()

    # Network uses SSL inspection (corporate/university proxy with a custom CA cert
    # that Python doesn't trust). verify=False bypasses certificate validation.
    # trust_env=False prevents httpx reading SSLKEYLOGFILE (Windows AV named pipe).
    os.environ.pop("SSLKEYLOGFILE", None)
    import httpx as _httpx
    _client = _anthropic.Anthropic(
        api_key=_api_key,
        http_client=_httpx.Client(trust_env=False, verify=False),
    )

    # ── Import MCP tool functions directly ───────────────────────────────
    _mcp_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _mcp_root not in _sys.path:
        _sys.path.insert(0, _mcp_root)
    from mcp_server.server import (
        get_welfare_results,
        get_scenario_comparison,
        get_pharma_supplier_risk,
        get_quintile_burden,
        get_manufacturing_shock,
        run_tariff_scenario,
    )

    # ── Tool definitions for the Anthropic API ────────────────────────────
    _TOOLS = [
        {
            "name": "get_welfare_results",
            "description": "Query GE welfare/CPI/trade results for one or all countries under a given scenario.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "country":  {"type": "string",  "description": "ISO3 country code e.g. 'USA', 'CHN'. Omit for all countries."},
                    "scenario": {"type": "string",  "description": "One of: ustr_no_retaliation, ustr_lump_sum, optimal_tariff, ustr_reciprocal_retaliation, ustr_optimal_retaliation, flat_15pct."},
                },
            },
        },
        {
            "name": "get_scenario_comparison",
            "description": "Pivot welfare outcomes across multiple countries and scenarios.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "countries": {"type": "array", "items": {"type": "string"}, "description": "List of ISO3 codes."},
                    "scenarios": {"type": "array", "items": {"type": "string"}, "description": "List of scenario names."},
                },
            },
        },
        {
            "name": "get_pharma_supplier_risk",
            "description": "Query pharma import exposure, compute HHI concentration, return suppliers ranked by risk.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "country":  {"type": "string",  "description": "Exporting country name e.g. 'Ireland'. Omit for all."},
                    "hts_code": {"type": "integer", "description": "HTS code: 3002, 3003, or 3004. Omit for all."},
                },
            },
        },
        {
            "name": "get_quintile_burden",
            "description": "Return tariff incidence by income quintile from pharma + retail incidence data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Retail category: Grocery, Clothing, Footwear, Home Appliances, Electronics. Omit for pharma quintile data."},
                },
            },
        },
        {
            "name": "get_manufacturing_shock",
            "description": "Return sector-level tariff shocks and gross output ranked by impact.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "naics_code": {"type": "string",  "description": "NAICS code prefix e.g. '3364'. Omit for top sectors."},
                    "top_n":      {"type": "integer", "description": "Number of top sectors (default 10, max 50)."},
                },
            },
        },
        {
            "name": "run_tariff_scenario",
            "description": "Run a partial equilibrium approximation with custom tariff rates and return welfare delta vs USTR baseline.",
            "input_schema": {
                "type": "object",
                "required": ["tariff_overrides"],
                "properties": {
                    "tariff_overrides": {"type": "object",  "description": "ISO3 → tariff rate mapping e.g. {\"CHN\": 0.60, \"DEU\": 0.20}."},
                    "countries":        {"type": "array", "items": {"type": "string"}, "description": "Countries to include in output."},
                },
            },
        },
    ]

    _TOOL_FN_MAP = {
        "get_welfare_results":     get_welfare_results,
        "get_scenario_comparison": get_scenario_comparison,
        "get_pharma_supplier_risk": get_pharma_supplier_risk,
        "get_quintile_burden":     get_quintile_burden,
        "get_manufacturing_shock": get_manufacturing_shock,
        "run_tariff_scenario":     run_tariff_scenario,
    }

    def _call_tool(name: str, inputs: dict) -> dict:
        fn = _TOOL_FN_MAP.get(name)
        if fn is None:
            return {"data": {"error": f"Unknown tool: {name}"}, "chart_spec": {}}
        return fn(**inputs)

    def _run_agentic(messages: list) -> tuple[str, list]:
        """Run the agentic loop: send messages, handle tool calls, return (text, charts)."""
        charts = []
        loop_msgs = list(messages)
        text_response = ""

        while True:
            response = _client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                tools=_TOOLS,
                messages=loop_msgs,
                system=(
                    "You are an expert trade economist analysing the Liberation Day tariff impacts. "
                    "You have access to tools that query GE model results, pharma supply chain data, "
                    "income quintile incidence, and manufacturing sector shocks. "
                    "When asked quantitative questions, always call the relevant tool first, "
                    "then synthesise the numbers into a clear, structured answer with policy implications. "
                    "Format responses in markdown. Use bullet points and headers for readability."
                ),
            )

            # Collect any text content
            for block in response.content:
                if hasattr(block, "text"):
                    text_response += block.text

            if response.stop_reason != "tool_use":
                break

            # Handle tool calls
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool_result = _call_tool(block.name, block.input)
                # Stash chart spec if present
                if tool_result.get("chart_spec"):
                    charts.append((block.name, tool_result["chart_spec"]))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _json.dumps(tool_result["data"], default=str),
                })

            loop_msgs.append({"role": "assistant", "content": response.content})
            loop_msgs.append({"role": "user",      "content": tool_results})

        return text_response, charts

    # ── Session state ─────────────────────────────────────────────────────
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = []  # list of {"role", "content", "charts": [...]}

    # ── CSS additions for chat UI ─────────────────────────────────────────
    st.markdown("""
    <style>
      .chat-user   { background:#1e2235; border-left:3px solid #2563eb; padding:12px 16px; border-radius:0 8px 8px 0; margin:8px 0; color:#e2e8f0; }
      .chat-ai     { background:#151824; border-left:3px solid #22d3a0; padding:12px 16px; border-radius:0 8px 8px 0; margin:8px 0; color:#cbd5e1; }
      .chat-label  { font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:6px; }
      .user-label  { color:#2563eb; }
      .ai-label    { color:#22d3a0; }
    </style>
    """, unsafe_allow_html=True)

    # ── Generate Briefing button ──────────────────────────────────────────
    st.markdown('<div class="section-header">🤖 AI Analyst — Liberation Day Tariff Intelligence</div>', unsafe_allow_html=True)

    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        briefing_scenario = st.selectbox(
            "Briefing scenario",
            ["ustr_no_retaliation","ustr_reciprocal_retaliation","flat_15pct"],
            key="briefing_scenario_sel",
        )
        run_briefing = st.button("📋 Generate Policy Briefing", use_container_width=True)
    with col_info:
        st.markdown("""
        <div class="insight-box">
        Ask any question about tariff impacts, pharma supply chain risk, income distributional effects,
        or manufacturing exposure. The AI analyst calls live data tools and returns structured analysis with charts.
        </div>""", unsafe_allow_html=True)

    if run_briefing:
        briefing_prompt = (
            f"Generate a structured policy briefing for the scenario **{briefing_scenario}**. "
            "Chain these three analyses:\n"
            "1. Call get_welfare_results to get US and top-5 affected countries' welfare & CPI.\n"
            "2. Call get_pharma_supplier_risk to assess supply chain concentration (HHI) and top risk countries.\n"
            "3. Call get_quintile_burden to quantify distributional incidence across income groups.\n\n"
            "Synthesise into a policy memo with: Executive Summary, Key Findings (bullet points), "
            "Supply Chain Risks, Distributional Impacts, and Policy Recommendations."
        )
        st.session_state.ai_messages.append({"role": "user", "content": briefing_prompt, "charts": []})

    # ── Chat history ──────────────────────────────────────────────────────
    history_container = st.container()
    with history_container:
        for msg in st.session_state.ai_messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user"><div class="chat-label user-label">You</div>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-ai"><div class="chat-label ai-label">AI Analyst</div>', unsafe_allow_html=True)
                st.markdown(msg["content"])
                st.markdown("</div>", unsafe_allow_html=True)
                for chart_name, chart_spec in msg.get("charts", []):
                    if chart_spec:
                        try:
                            st.plotly_chart(go.Figure(chart_spec), use_container_width=True)
                        except Exception:
                            pass

    # ── Trigger AI response if last message is from user ─────────────────
    if st.session_state.ai_messages and st.session_state.ai_messages[-1]["role"] == "user":
        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.ai_messages
            if m["role"] in ("user", "assistant")
        ]
        with st.spinner("Analysing..."):
            try:
                text_out, charts_out = _run_agentic(api_messages)
                st.session_state.ai_messages.append({
                    "role": "assistant",
                    "content": text_out,
                    "charts": charts_out,
                })
                st.rerun()
            except Exception as e:
                st.error(f"API error: {e}")

    # ── Chat input ────────────────────────────────────────────────────────
    st.markdown("---")
    user_input = st.chat_input("Ask about tariff impacts, pharma risk, income effects, manufacturing…")
    if user_input:
        st.session_state.ai_messages.append({"role": "user", "content": user_input, "charts": []})
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — MCP ANALYST
# Connects to mcp_server/server.py over stdio via the MCP protocol.
# Tools are discovered dynamically; every tool call is routed through the
# subprocess — no direct Python imports of the tool functions.
# ═════════════════════════════════════════════════════════════════════════════
with tab6:
    import asyncio as _asyncio
    import json as _json6
    import sys as _sys6
    import anthropic as _anthropic6
    import httpx as _httpx6

    from mcp import ClientSession as _MCPSession
    from mcp.client.stdio import stdio_client as _stdio_client, StdioServerParameters as _StdioParams

    # ── Anthropic client (same SSL bypass as Tab 5) ───────────────────────
    _mcp6_api_key = None
    try:
        _mcp6_api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        _mcp6_api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not _mcp6_api_key:
        st.error("No Anthropic API key found. Set `ANTHROPIC_API_KEY` in `.streamlit/secrets.toml`.")
        st.stop()

    os.environ.pop("SSLKEYLOGFILE", None)
    _mcp6_client = _anthropic6.Anthropic(
        api_key=_mcp6_api_key,
        http_client=_httpx6.Client(trust_env=False, verify=False),
    )

    # ── Path to the MCP server script ────────────────────────────────────
    _MCP_SERVER_SCRIPT = os.path.join(ROOT, "mcp_server", "server.py")
    _MCP_SERVER_PARAMS = _StdioParams(
        command=_sys6.executable,
        args=[_MCP_SERVER_SCRIPT],
        cwd=ROOT,
    )

    # ── Async agentic loop: spawns subprocess, discovers tools, routes calls ─
    async def _run_mcp_agentic(messages: list) -> tuple[str, list]:
        charts = []
        text_response = ""

        async with _stdio_client(_MCP_SERVER_PARAMS) as (read_stream, write_stream):
            async with _MCPSession(read_stream, write_stream) as session:
                await session.initialize()

                # Discover tools from the live MCP server
                tools_resp = await session.list_tools()
                anthropic_tools = [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "input_schema": t.inputSchema,
                    }
                    for t in tools_resp.tools
                ]

                loop_msgs = list(messages)

                while True:
                    response = _mcp6_client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=4096,
                        tools=anthropic_tools,
                        messages=loop_msgs,
                        system=(
                            "You are an expert trade economist analysing the Liberation Day tariff impacts. "
                            "You have access to tools that query GE model results, pharma supply chain data, "
                            "income quintile incidence, and manufacturing sector shocks. "
                            "When asked quantitative questions, always call the relevant tool first, "
                            "then synthesise the numbers into a clear, structured answer with policy implications. "
                            "Format responses in markdown. Use bullet points and headers for readability.\n\n"
                            "IMPORTANT — HOW THIS SYSTEM WORKS:\n"
                            "You are running inside the MCP Analyst tab of a Streamlit dashboard. "
                            "When you make a tool call, it is NOT executed locally in Python. Instead, the dashboard "
                            "routes your call through the Model Context Protocol (MCP) over stdio to a live "
                            "mcp_server/server.py subprocess (FastMCP 3.4.2). That server executes the function "
                            "and returns the result back through the protocol. You never import or call the "
                            "functions directly — every tool call travels over MCP stdio transport. "
                            "The tools were also discovered dynamically via session.list_tools() at the start "
                            "of this session, not hardcoded. "
                            "If asked about your architecture, explain this accurately: you use Anthropic's tool "
                            "use API as the LLM interface, but the execution layer is a real MCP server subprocess "
                            "connected over stdio — making this a genuine MCP-backed agentic system."
                        ),
                    )

                    for block in response.content:
                        if hasattr(block, "text"):
                            text_response += block.text

                    if response.stop_reason != "tool_use":
                        break

                    # Route each tool call through the MCP subprocess
                    tool_results = []
                    for block in response.content:
                        if block.type != "tool_use":
                            continue

                        mcp_result = await session.call_tool(block.name, block.input)

                        # FastMCP serialises tool return dicts to JSON in TextContent
                        raw_text = ""
                        for item in mcp_result.content:
                            if hasattr(item, "text"):
                                raw_text += item.text

                        chart_spec = {}
                        try:
                            parsed = _json6.loads(raw_text)
                            chart_spec = parsed.get("chart_spec", {})
                            data_payload = parsed.get("data", parsed)
                        except Exception:
                            data_payload = raw_text

                        if chart_spec:
                            charts.append((block.name, chart_spec))

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": _json6.dumps(data_payload, default=str),
                        })

                    loop_msgs.append({"role": "assistant", "content": response.content})
                    loop_msgs.append({"role": "user",      "content": tool_results})

        return text_response, charts

    def _run_mcp_sync(messages: list) -> tuple[str, list]:
        """Bridge async MCP loop into Streamlit's synchronous context."""
        return _asyncio.run(_run_mcp_agentic(messages))

    # ── Session state ─────────────────────────────────────────────────────
    if "mcp6_messages" not in st.session_state:
        st.session_state.mcp6_messages = []

    # ── UI — matches Tab 5 layout exactly ────────────────────────────────
    st.markdown('<div class="section-header">&#128299; MCP Analyst &#8212; Live Protocol Connection</div>', unsafe_allow_html=True)

    col_btn6, col_info6 = st.columns([2, 5])
    with col_btn6:
        briefing_scenario6 = st.selectbox(
            "Briefing scenario",
            ["ustr_no_retaliation", "ustr_reciprocal_retaliation", "flat_15pct"],
            key="mcp6_briefing_scenario_sel",
        )
        run_briefing6 = st.button("📋 Generate Policy Briefing", use_container_width=True, key="mcp6_briefing_btn")
        verify_mcp = st.button("🔬 Verify MCP Connection", use_container_width=True, key="mcp6_verify_btn")
    with col_info6:
        st.markdown("""
        <div class="insight-box">
        This tab connects to <b>mcp_server/server.py</b> as a live subprocess over the MCP stdio protocol.
        Tools are discovered dynamically from the running server &#8212; no direct Python imports.
        Every tool call travels through the protocol, exactly as an external AI agent would use it.
        </div>""", unsafe_allow_html=True)

    # ── Verify: spawn server, list_tools(), show raw server response ─────
    if verify_mcp:
        async def _verify_mcp():
            import time
            t0 = time.time()
            async with _stdio_client(_MCP_SERVER_PARAMS) as (r, w):
                async with _MCPSession(r, w) as session:
                    init_result = await session.initialize()
                    tools_resp  = await session.list_tools()
                    elapsed = round(time.time() - t0, 3)
                    return init_result, tools_resp, elapsed

        with st.spinner("Spawning MCP subprocess and running list_tools()…"):
            try:
                init_result, tools_resp, elapsed = _asyncio.run(_verify_mcp())
                st.success(f"Connected in {elapsed}s — server returned {len(tools_resp.tools)} tools")
                st.markdown("**Raw `initialize()` response from server:**")
                st.json(init_result.model_dump())
                st.markdown("**Raw `list_tools()` response from server:**")
                st.json([t.model_dump() for t in tools_resp.tools])
            except Exception as e:
                st.error(f"Verification failed: {e}")

    if run_briefing6:
        briefing_prompt6 = (
            f"Generate a structured policy briefing for the scenario **{briefing_scenario6}**. "
            "Chain these three analyses:\n"
            "1. Call get_welfare_results to get US and top-5 affected countries' welfare & CPI.\n"
            "2. Call get_pharma_supplier_risk to assess supply chain concentration (HHI) and top risk countries.\n"
            "3. Call get_quintile_burden to quantify distributional incidence across income groups.\n\n"
            "Synthesise into a policy memo with: Executive Summary, Key Findings (bullet points), "
            "Supply Chain Risks, Distributional Impacts, and Policy Recommendations."
        )
        st.session_state.mcp6_messages.append({"role": "user", "content": briefing_prompt6, "charts": []})

    # ── Chat history ──────────────────────────────────────────────────────
    with st.container():
        for msg in st.session_state.mcp6_messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user"><div class="chat-label user-label">You</div>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="chat-ai"><div class="chat-label ai-label">MCP Analyst</div>', unsafe_allow_html=True)
                st.markdown(msg["content"])
                st.markdown("</div>", unsafe_allow_html=True)
                for chart_name, chart_spec in msg.get("charts", []):
                    if chart_spec:
                        try:
                            st.plotly_chart(go.Figure(chart_spec), use_container_width=True)
                        except Exception:
                            pass

    # ── Trigger response when last message is from user ───────────────────
    if st.session_state.mcp6_messages and st.session_state.mcp6_messages[-1]["role"] == "user":
        api_messages6 = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.mcp6_messages
            if m["role"] in ("user", "assistant")
        ]
        with st.spinner("Connecting to MCP server and analysing…"):
            try:
                text_out6, charts_out6 = _run_mcp_sync(api_messages6)
                st.session_state.mcp6_messages.append({
                    "role": "assistant",
                    "content": text_out6,
                    "charts": charts_out6,
                })
                st.rerun()
            except Exception as e:
                st.error(f"MCP error: {e}")

    # ── Chat input ────────────────────────────────────────────────────────
    st.markdown("---")
    user_input6 = st.chat_input("Ask via MCP protocol &#8212; pharma risk, welfare impacts, custom scenarios…", key="mcp6_chat_input")
    if user_input6:
        st.session_state.mcp6_messages.append({"role": "user", "content": user_input6, "charts": []})
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — BUILD YOUR OWN SCENARIO (GE-powered)
# A stakeholder defines a country-level US tariff policy and runs the paper's
# ACTUAL 194-country general-equilibrium model (ge_scenario_runner.py — copy of
# the replication solver, validated to reproduce scenario 0 and the flat-15%
# run to <0.05pp). Every decision-facing number below comes from a completed
# GE solve; instant approximations are quarantined and labeled.
# ═════════════════════════════════════════════════════════════════════════════
with tab7:
    import io as _io7
    import json as _json7
    import zipfile as _zipf7
    from datetime import datetime as _dt7

    from ge_scenario_runner import run_ge_scenario as _ge_run_raw
    from ge_scenario_runner import MODEL_VERSION as _GE_MODEL_VERSION
    from ge_scenario_runner import DATA_VINTAGE as _GE_DATA_VINTAGE

    _res_g7, _Y_g7, _idus_g7, _cl_g7, _res15_g7, _ = load_baseline()
    _tdf_g7 = load_tariffs()
    _imp_vec_g7, _exp_vec_g7, _, _ = load_bilateral()
    _, _retail_g7 = load_retail()
    _, _, _, _pharma_exp_g7 = load_pharma_outputs()

    _iso_by_name_g7 = dict(zip(_cl_g7["CountryName"], _cl_g7["iso3"]))
    _name_by_iso_g7 = dict(zip(_cl_g7["iso3"], _cl_g7["CountryName"]))
    _idx_by_iso_g7 = {iso: i for i, iso in enumerate(_cl_g7["iso3"])}
    _ld_rate_by_iso_g7 = {iso: max(10.0, float(r)) for iso, r in
                          zip(_tdf_g7["iso3"], _tdf_g7["tariff_pct"])}

    _HH_CONSUMPTION = 77_280   # BLS Consumer Expenditure Survey 2023, avg annual expenditure
    _N_HH7 = 132_000_000       # US households (Census ~2023)
    _Q_INCOMES = [16_120, 44_762, 74_730, 117_002, 253_484]  # BLS CE 2023 avg pre-tax income by quintile
    _Q_LABELS7 = ["Lowest 20%", "Lower-Middle", "Middle", "Upper-Middle", "Top 20%"]

    @st.cache_data(show_spinner=False)
    def _ge_cached(frozen_rates):
        """Full GE solve, cached by exact tariff vector (iso3, pct) tuple.
        Param must NOT start with underscore: st.cache_data skips hashing
        underscore-prefixed args, which made every scenario return the first
        cached result (the LD baseline)."""
        overrides = {iso: rate / 100.0 for iso, rate in frozen_rates}
        return _ge_run_raw(overrides=overrides)

    st.markdown('<div class="section-header">🎛️ Build Your Own Tariff Policy — Full GE Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Define a future US tariff policy country by country, then run the paper\'s <b>actual 194-country general-equilibrium model</b> — the same solver that produced every scenario in this dashboard, validated to reproduce them to &lt;0.05pp. You get the full equilibrium: welfare, prices, employment, trade flows and revenue for every country, plus trade-diversion analysis no simple calculator can produce. Assumes no foreign retaliation (scenario-0 treatment).</div>', unsafe_allow_html=True)

    # ── A. Configure scenario ────────────────────────────────────────────────
    _scen_name7 = st.text_input("Scenario name", "My Tariff Policy", key="bys_name")

    _pending7 = st.session_state.pop("_bys_preset", None)
    if _pending7:
        for _k, _v in _pending7.items():
            st.session_state[_k] = _v

    _default7 = [n for n in ["China", "Vietnam", "Mexico", "Canada", "Germany", "Japan"]
                 if n in _iso_by_name_g7]
    _sel7 = st.multiselect("Countries in your scenario (others stay at Liberation Day rates)",
                           sorted(_cl_g7["CountryName"].tolist()),
                           default=_default7, key="bys_countries")

    _rates7, _lds7 = {}, {}
    if _sel7:
        _slider_cols7 = st.columns(3)
        for _i, _cname in enumerate(_sel7):
            _iso = _iso_by_name_g7[_cname]
            _ld = int(round(_ld_rate_by_iso_g7.get(_iso, 10)))
            with _slider_cols7[_i % 3]:
                _rates7[_iso] = st.slider(f"{_cname} (LD: {_ld}%)", 0, 100, _ld, 1,
                                          format="%d%%", key=f"bys_rate_{_iso}")
            _lds7[_iso] = _ld

    _p1, _p2, _p3, _p4, _p5 = st.columns(5)
    def _preset7(vals):
        st.session_state["_bys_preset"] = vals
        st.rerun()
    with _p1:
        if st.button("↺ Liberation Day", use_container_width=True, key="bys_p_ld"):
            _preset7({f"bys_rate_{iso}": _lds7[iso] for iso in _rates7})
    with _p2:
        if st.button("🕊️ Free Trade (0%)", use_container_width=True, key="bys_p_free"):
            _preset7({f"bys_rate_{iso}": 0 for iso in _rates7})
    with _p3:
        if st.button("🌍 Flat 10%", use_container_width=True, key="bys_p_10"):
            _preset7({f"bys_rate_{iso}": 10 for iso in _rates7})
    with _p4:
        if st.button("⚖️ Uniform 25%", use_container_width=True, key="bys_p_25"):
            _preset7({f"bys_rate_{iso}": 25 for iso in _rates7})
    with _p5:
        if st.button("🔥 Max Pressure (CHN 100%)", use_container_width=True, key="bys_p_max"):
            _vals = {f"bys_rate_{iso}": _lds7[iso] for iso in _rates7}
            if "CHN" in _rates7:
                _vals["bys_rate_CHN"] = 100
            _preset7(_vals)

    if not _sel7:
        st.info("Select at least one country to build a scenario.")
        st.stop()

    _frozen7 = tuple(sorted((iso, int(r)) for iso, r in _rates7.items()))
    _changed7 = {iso: r for iso, r in _rates7.items() if r != _lds7[iso]}

    # Config summary strip + instant PE preview (approximation only)
    _tot_imp7 = float(_imp_vec_g7.sum())
    _imp_touched7 = sum(float(_imp_vec_g7[_idx_by_iso_g7[i]]) for i in _changed7 if i in _idx_by_iso_g7)
    _twt_sel7 = (sum(float(_imp_vec_g7[_idx_by_iso_g7[i]]) * _rates7[i] for i in _rates7 if i in _idx_by_iso_g7) /
                 max(sum(float(_imp_vec_g7[_idx_by_iso_g7[i]]) for i in _rates7 if i in _idx_by_iso_g7), 1))
    _pe_prev7 = _compute_custom_scenario(_frozen7)
    _pe_us7 = next((r for r in _pe_prev7.get("data", {}).get("countries", []) if r.get("iso3") == "USA"), {})
    st.markdown(
        f'<div style="display:flex;gap:24px;background:#1a1d2e;border:1px solid #2d3250;border-radius:10px;padding:10px 18px;margin:8px 0 4px 0;flex-wrap:wrap">'
        f'<div style="color:#94a3b8;font-size:12px"><b style="color:#e2e8f0">{len(_changed7)}</b> countries changed</div>'
        f'<div style="color:#94a3b8;font-size:12px">imports affected <b style="color:#e2e8f0">${_imp_touched7/1e6:,.0f}B</b> ({_imp_touched7/max(_tot_imp7,1)*100:.1f}% of US imports)</div>'
        f'<div style="color:#94a3b8;font-size:12px">trade-weighted tariff on selection <b style="color:#e2e8f0">{_twt_sel7:.1f}%</b></div>'
        f'<div style="color:#64748b;font-size:11px;margin-left:auto">Instant approximation (not GE): US welfare Δ {float(_pe_us7.get("welfare_delta_pct") or 0):+.2f}pp</div>'
        f'</div>', unsafe_allow_html=True)

    # ── B. Run the model ─────────────────────────────────────────────────────
    _run_col7, _ = st.columns([2, 4])
    with _run_col7:
        _do_run7 = st.button("▶ Run Full GE Simulation", type="primary",
                             use_container_width=True, key="bys_run_ge")
    if _do_run7:
        with st.spinner("Solving 194-country general equilibrium…"):
            _ge_ld7 = _ge_cached(())            # LD baseline (cached after first run)
            _ge_res7 = _ge_cached(_frozen7)     # user scenario
        st.session_state["bys_ge"] = {"frozen": _frozen7, "ts": _dt7.now().strftime("%Y-%m-%d %H:%M")}

    if "bys_ge" not in st.session_state:
        st.markdown('<div class="insight-box">Configure your tariffs above, then press <b>▶ Run Full GE Simulation</b>. The solver computes the complete new world equilibrium — typically in a few seconds.</div>', unsafe_allow_html=True)
        st.stop()

    _stored7 = st.session_state["bys_ge"]
    _ge_res7 = _ge_cached(_stored7["frozen"])
    _ge_ld7 = _ge_cached(())
    _is_stale7 = (_stored7["frozen"] != _frozen7)
    if _is_stale7:
        st.markdown('<div style="background:#2a230f;border:1px solid #fbbf24;border-radius:8px;padding:10px 16px;margin:8px 0;color:#fbbf24;font-size:13px">⚠️ <b>Tariffs changed — these results are stale.</b> Run the model again to update.</div>', unsafe_allow_html=True)

    # Convergence gate — failed runs never render as analysis
    if not _ge_res7["converged"] or not np.isfinite(_ge_res7["resid_max"]):
        st.error(f"GE solver did not converge (ier={_ge_res7['fsolve_ier']}, "
                 f"residual={_ge_res7['resid_max']:.2e}). Results are not valid analysis. "
                 f"Try a less extreme tariff vector.")
        with st.expander("Solver diagnostics"):
            st.json({k: _ge_res7[k] for k in ["fsolve_ier", "fsolve_msg", "nfev",
                                              "resid_max", "resid_scaled", "runtime_sec"]})
        st.stop()

    # Shorthands
    _R7 = _ge_res7["results"]          # user scenario (194×7)
    _RLD7 = _ge_ld7["results"]         # LD GE baseline (validated vs paper scenario 0)
    _idus7 = _ge_res7["id_US"]
    _E7 = _ge_res7["E_i"]
    _run_frozen_map7 = dict(_stored7["frozen"])
    _run_changed7 = {iso: r for iso, r in _run_frozen_map7.items()
                     if r != int(round(_ld_rate_by_iso_g7.get(iso, 10)))}

    _scen_label7 = st.session_state.get("bys_name", "My Tariff Policy") or "My Tariff Policy"
    st.markdown(f'<div class="section-header">📋 GE results — “{_scen_label7}” <span style="color:#475569;font-size:12px;font-weight:400">(run {_stored7["ts"]}, {_ge_res7["runtime_sec"]:.1f}s, converged)</span></div>', unsafe_allow_html=True)

    # ── C. Executive scorecard: 8 outcomes, Δ vs Liberation Day GE ──────────
    # results columns: 0 welfare, 1 deficit, 2 exports/GDP, 3 imports/GDP, 4 employment, 5 CPI, 6 rev/E
    _rev_usd7 = _ge_res7["revenue_us_dollars"]
    _rev_ld_usd7 = _ge_ld7["revenue_us_dollars"]
    _sc_defs7 = [
        ("US welfare",     _R7[_idus7, 0], _RLD7[_idus7, 0], "%", True),
        ("US consumer prices (CPI)", _R7[_idus7, 5], _RLD7[_idus7, 5], "%", False),
        ("US employment",  _R7[_idus7, 4], _RLD7[_idus7, 4], "%", True),
        ("US imports",     _R7[_idus7, 3], _RLD7[_idus7, 3], "%", True),
        ("US exports",     _R7[_idus7, 2], _RLD7[_idus7, 2], "%", True),
        ("US trade deficit", _R7[_idus7, 1], _RLD7[_idus7, 1], "%", False),
        ("Tariff revenue", _rev_usd7 / 1e9, _rev_ld_usd7 / 1e9, "$B", True),
        ("Global trade",   _ge_res7["d_trade"], _ge_ld7["d_trade"], "%", True),
    ]
    _sc_cols7 = st.columns(4)
    for _i, (_lab, _val, _ldv, _unit, _up_good) in enumerate(_sc_defs7):
        _d = _val - _ldv
        _cls = ("positive" if ((_d > 0) == _up_good and abs(_d) > 1e-9) else
                "negative" if abs(_d) > 1e-9 else "neutral")
        _vtxt = f"${_val:,.0f}B" if _unit == "$B" else f"{_val:+.2f}%"
        _dtxt = f"{_d:+,.0f}$B vs LD" if _unit == "$B" else f"{_d:+.2f}pp vs LD"
        with _sc_cols7[_i % 4]:
            st.markdown(f"""<div class="kpi-card" style="margin-bottom:12px">
              <div class="kpi-label">{_lab}</div>
              <div class="kpi-value {_cls}" style="font-size:23px">{_vtxt}</div>
              <div class="kpi-sub">{_dtxt}</div>
            </div>""", unsafe_allow_html=True)

    # ── Stakeholder translations (estimated) ─────────────────────────────────
    _cpi_d7 = _R7[_idus7, 5] - _RLD7[_idus7, 5]
    _hh_cost_abs7 = _R7[_idus7, 5] / 100 * _HH_CONSUMPTION          # vs pre-tariff world
    _hh_cost_d7 = _cpi_d7 / 100 * _HH_CONSUMPTION                    # vs Liberation Day
    _m_new7 = float(np.sum(np.delete(_ge_res7["X_ji_new"][:, _idus7], _idus7))) * 1000
    _m_ld7 = float(np.sum(np.delete(_ge_ld7["X_ji_new"][:, _idus7], _idus7))) * 1000
    _x_new7 = float(np.sum(np.delete(_ge_res7["X_ji_new"][_idus7, :], _idus7))) * 1000
    _x_ld7 = float(np.sum(np.delete(_ge_ld7["X_ji_new"][_idus7, :], _idus7))) * 1000

    st.markdown('<div class="insight-box" style="margin-top:4px"><b>Estimated dollar translations</b> (derived from the GE run; household figures use BLS CE 2023 avg expenditure $77,280 × 132M households): '
        f'tariff revenue <b>${_rev_usd7/1e9:,.0f}B</b> ({(_rev_usd7-_rev_ld_usd7)/1e9:+,.0f}B vs LD) · '
        f'household purchasing-power cost <b>${_hh_cost_abs7:,.0f}/yr</b> vs pre-tariff world ({_hh_cost_d7:+,.0f} vs LD) · '
        f'aggregate household cost <b>${_hh_cost_abs7*_N_HH7/1e9:,.0f}B/yr</b> · '
        f'US imports <b>${_m_new7/1e9:,.0f}B</b> ({(_m_new7-_m_ld7)/1e9:+,.0f}B vs LD) · '
        f'US exports <b>${_x_new7/1e9:,.0f}B</b> ({(_x_new7-_x_ld7)/1e9:+,.0f}B vs LD). '
        'Note: welfare already accounts for tariff revenue — do not net revenue against household cost.</div>', unsafe_allow_html=True)

    # ── Policy footprint ─────────────────────────────────────────────────────
    _tv7 = _ge_res7["tariff_vector"]
    _imp_over25_7 = float(np.sum(_imp_vec_g7[np.array(_tv7) > 0.25])) if len(_tv7) == len(_imp_vec_g7) else 0.0
    _pharma_hit7 = 0.0
    for _iso, _r in _run_changed7.items():
        if _r > _ld_rate_by_iso_g7.get(_iso, 10) and "iso3" in _pharma_exp_g7.columns:
            _prow = _pharma_exp_g7[_pharma_exp_g7["iso3"] == _iso]
            if not _prow.empty:
                _pharma_hit7 += float(_prow["import_share_pct"].iloc[0])
    _twt_all7 = float(np.sum(_imp_vec_g7 * np.array(_tv7)) / max(_imp_vec_g7.sum(), 1)) * 100
    st.markdown(
        f'<div style="display:flex;gap:24px;background:#0f172a;border:1px solid #2d3250;border-radius:10px;padding:10px 18px;margin-bottom:10px;flex-wrap:wrap">'
        f'<div style="color:#94a3b8;font-size:12px">POLICY FOOTPRINT</div>'
        f'<div style="color:#94a3b8;font-size:12px"><b style="color:#e2e8f0">{len(_run_changed7)}</b> countries changed</div>'
        f'<div style="color:#94a3b8;font-size:12px">trade-weighted US tariff <b style="color:#e2e8f0">{_twt_all7:.1f}%</b></div>'
        f'<div style="color:#94a3b8;font-size:12px">imports facing &gt;25% <b style="color:#e2e8f0">${_imp_over25_7/1e6:,.0f}B</b></div>'
        f'<div style="color:#94a3b8;font-size:12px">pharma supply exposure <b style="color:{"#f87171" if _pharma_hit7 >= 3 else "#e2e8f0"}">{_pharma_hit7:.1f}%</b> of US medicine imports</div>'
        f'</div>', unsafe_allow_html=True)
    if _pharma_hit7 >= 3:
        st.markdown(f'<div class="insight-box" style="border-left-color:#f87171">⚠️ <b>Medicine supply risk</b> (linked sector analysis, not a direct GE output): you raised tariffs on countries supplying <b>{_pharma_hit7:.1f}%</b> of US pharmaceutical imports.</div>', unsafe_allow_html=True)

    # ── D. Global effects ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🌍 Global effects of your policy</div>', unsafe_allow_html=True)
    _w7 = _R7[:, 0]
    _harmed7 = int(np.sum(_w7 < -1.0))
    _ew_world7 = float(np.sum(_w7 * _E7) / np.sum(_E7))
    _g1, _g2, _g3, _g4 = st.columns(4)
    with _g1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Materially harmed countries</div>
          <div class="kpi-value negative" style="font-size:24px">{_harmed7}</div>
          <div class="kpi-sub">welfare below −1% (LD: {int(np.sum(_RLD7[:,0] < -1.0))})</div></div>""", unsafe_allow_html=True)
    with _g2:
        _ew_cls7 = "positive" if _ew_world7 > 0 else "negative"
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">World welfare (expenditure-weighted)</div>
          <div class="kpi-value {_ew_cls7}" style="font-size:24px">{_ew_world7:+.2f}%</div>
          <div class="kpi-sub">LD: {float(np.sum(_RLD7[:,0]*_E7)/np.sum(_E7)):+.2f}%</div></div>""", unsafe_allow_html=True)
    with _g3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Global trade change</div>
          <div class="kpi-value negative" style="font-size:24px">{_ge_res7["d_trade"]:+.1f}%</div>
          <div class="kpi-sub">LD: {_ge_ld7["d_trade"]:+.1f}%</div></div>""", unsafe_allow_html=True)
    with _g4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Countries losing / gaining</div>
          <div class="kpi-value neutral" style="font-size:24px">{int(np.sum(_w7 < 0))} / {int(np.sum(_w7 > 0))}</div>
          <div class="kpi-sub">of 194 modelled economies</div></div>""", unsafe_allow_html=True)

    _gm1, _gm2 = st.columns([3, 2])
    with _gm1:
        _map7 = _cl_g7.copy()
        _map7["welfare"] = _w7
        fig_gmap7 = px.choropleth(
            _map7, locations="iso3", color="welfare", hover_name="CountryName",
            color_continuous_scale=["#f87171", "#fca5a5", "#fef3c7", "#6ee7b7", "#22d3a0"],
            color_continuous_midpoint=0, range_color=[-5, 5], labels={"welfare": "Welfare %"})
        fig_gmap7.update_traces(marker_line_color="#0f1117", marker_line_width=0.3,
            hovertemplate="<b>%{hovertext}</b><br>Welfare: %{z:.2f}%<extra></extra>")
        fig_gmap7.update_layout(**PLOTLY_THEME, height=380,
            geo=dict(bgcolor="#0f1117", showframe=False, showcoastlines=True,
                     coastlinecolor="#2d3250", showland=True, landcolor="#1a1d2e",
                     showocean=True, oceancolor="#0f1117", projection_type="natural earth"),
            coloraxis_colorbar=dict(title="Welfare %", tickfont=dict(color="#94a3b8"),
                                    bgcolor="#1a1d2e"))
        st.plotly_chart(fig_gmap7, use_container_width=True)
        _explain("The full GE welfare map under YOUR tariff policy - all 194 countries re-solved simultaneously. Green gained, red lost. Compare mentally with Tab 1's Liberation Day map to see what your changes did to the world.")
    with _gm2:
        _wsort7 = _map7.sort_values("welfare")
        _wl7 = pd.concat([_wsort7.head(10), _wsort7.tail(10)])
        fig_wl7 = go.Figure(go.Bar(
            x=_wl7["welfare"], y=_wl7["CountryName"], orientation="h",
            marker_color=["#f87171" if v < 0 else "#22d3a0" for v in _wl7["welfare"]],
            text=[f"{v:+.2f}%" for v in _wl7["welfare"]], textposition="outside"))
        fig_wl7.add_vline(x=0, line_color="#4b5563", line_width=1)
        fig_wl7.update_layout(**PLOTLY_THEME, height=380,
            title="Top 10 losers & winners under your policy")
        fig_wl7.update_yaxes(autorange="reversed", tickfont=dict(size=10))
        st.plotly_chart(fig_wl7, use_container_width=True)
        _explain("The extremes of your policy: the ten hardest-hit and ten biggest-gaining economies, from the GE solution.")

    # ── Trade diversion (headline analysis) ──────────────────────────────────
    st.markdown('<div class="section-header">🔀 Trade diversion — what a GE model shows that a calculator can\'t</div>', unsafe_allow_html=True)
    _m_user7 = _ge_res7["X_ji_new"][:, _idus7].copy() * 1000
    _m_base7 = _ge_ld7["X_ji_new"][:, _idus7].copy() * 1000
    _m_user7[_idus7] = 0.0
    _m_base7[_idus7] = 0.0
    _dm7 = _m_user7 - _m_base7
    _losses7 = float(-_dm7[_dm7 < 0].sum())
    _gains7 = float(_dm7[_dm7 > 0].sum())
    _net7 = float(_dm7.sum())
    _redirected7 = min(_losses7, _gains7)
    _destroyed7 = max(0.0, -_net7)
    _created7 = max(0.0, _net7)
    _sh_u7 = _m_user7 / max(_m_user7.sum(), 1)
    _sh_b7 = _m_base7 / max(_m_base7.sum(), 1)
    _hhi_u7 = float(np.sum(_sh_u7 ** 2) * 10000)
    _hhi_b7 = float(np.sum(_sh_b7 ** 2) * 10000)

    st.markdown(f'<div class="insight-box">Versus Liberation Day, your policy moves <b>${(_losses7+_gains7)/2e9:,.0f}B</b> of US import flows: '
        f'<b>${_redirected7/1e9:,.0f}B redirected</b> to alternative suppliers, '
        + (f'<b>${_destroyed7/1e9:,.0f}B destroyed</b> (imports that simply stop)' if _destroyed7 >= _created7 else f'<b>${_created7/1e9:,.0f}B created</b> (net new imports)')
        + f'. Supplier concentration (HHI) moves from <b>{_hhi_b7:,.0f}</b> to <b>{_hhi_u7:,.0f}</b> — '
        f'{"more concentrated (riskier)" if _hhi_u7 > _hhi_b7 else "more diversified (safer)"}.</div>', unsafe_allow_html=True)

    _dv7 = _cl_g7.copy()
    _dv7["delta_bn"] = _dm7 / 1e9
    _dv_sorted7 = _dv7.sort_values("delta_bn")
    _dv_plot7 = pd.concat([_dv_sorted7.head(10), _dv_sorted7.tail(10)])
    _dv_plot7 = _dv_plot7[_dv_plot7["delta_bn"].abs() > 0.005]
    if not _dv_plot7.empty:
        fig_dv7 = go.Figure(go.Bar(
            x=_dv_plot7["delta_bn"], y=_dv_plot7["CountryName"], orientation="h",
            marker_color=["#f87171" if v < 0 else "#22d3a0" for v in _dv_plot7["delta_bn"]],
            text=[f"{v:+,.1f}B" for v in _dv_plot7["delta_bn"]], textposition="outside"))
        fig_dv7.add_vline(x=0, line_color="#4b5563", line_width=1)
        fig_dv7.update_layout(**PLOTLY_THEME, height=440,
            title="Change in US imports by supplier vs Liberation Day ($B, GE equilibrium)")
        fig_dv7.update_yaxes(autorange="reversed", tickfont=dict(size=10))
        st.plotly_chart(fig_dv7, use_container_width=True)
        _explain("Red countries lose US import business under your policy; green countries pick it up. This reallocation is solved inside the GE equilibrium - prices, wages and expenditure all adjust simultaneously. The top green bars are your policy's de-facto substitute suppliers.")
    else:
        st.markdown('<div class="insight-box">No material trade diversion vs Liberation Day — your rates match the LD schedule.</div>', unsafe_allow_html=True)

    # ── Country table + full CSV ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Country-by-country GE results</div>', unsafe_allow_html=True)
    _majors7 = ["USA", "CHN", "DEU", "JPN", "GBR", "CAN", "MEX"]
    _show_isos7 = list(dict.fromkeys(list(_run_frozen_map7.keys()) + _majors7))
    _tbl_rows7 = []
    for _iso in _show_isos7:
        _ci = _idx_by_iso_g7.get(_iso)
        if _ci is None:
            continue
        _tbl_rows7.append({
            "Country": _name_by_iso_g7.get(_iso, _iso),
            "US Tariff (%)": "—" if _iso == "USA" else round(float(_tv7[_ci]) * 100, 1),
            "Welfare (%)": round(float(_R7[_ci, 0]), 2),
            "Δ vs LD (pp)": round(float(_R7[_ci, 0] - _RLD7[_ci, 0]), 2),
            "CPI (%)": round(float(_R7[_ci, 5]), 2),
            "Employment (%)": round(float(_R7[_ci, 4]), 2),
            "Imports (%)": round(float(_R7[_ci, 3]), 2),
            "Exports (%)": round(float(_R7[_ci, 2]), 2),
        })
    st.dataframe(pd.DataFrame(_tbl_rows7), use_container_width=True, hide_index=True)

    _full7 = _cl_g7[["iso3", "CountryName"]].copy()
    _full7["us_tariff_pct"] = np.array(_tv7) * 100
    for _cidx, _cname in [(0, "welfare_pct"), (1, "deficit_pct"), (2, "exports_pct"),
                          (3, "imports_pct"), (4, "employment_pct"), (5, "cpi_pct"),
                          (6, "revenue_share")]:
        _full7[_cname] = _R7[:, _cidx]
        _full7[f"{_cname}_ld"] = _RLD7[:, _cidx]

    # ── Distributional impact (GE-scaled estimate) ───────────────────────────
    st.markdown('<div class="section-header">Who pays at home — GE-scaled distributional estimate</div>', unsafe_allow_html=True)
    _q_base7 = list(_retail_g7["quintile_incidence_noretal"])
    _cpi_ratio7 = (_R7[_idus7, 5] / _RLD7[_idus7, 5]) if abs(_RLD7[_idus7, 5]) > 1e-9 else 1.0
    _q_user7 = [v * _cpi_ratio7 for v in _q_base7]
    _q_cost7 = [inc * q / 100 for inc, q in zip(_Q_INCOMES, _q_user7)]
    _regress7 = (_q_user7[0] / _q_user7[4]) if _q_user7[4] else 0
    _dq1, _dq2 = st.columns([3, 2])
    with _dq1:
        fig_q7 = go.Figure(go.Bar(
            x=_Q_LABELS7, y=_q_user7,
            marker_color=["#f87171", "#fb923c", "#fbbf24", "#a3e635", "#22d3a0"],
            text=[f"{v:.1f}%<br>${c:,.0f}/yr" for v, c in zip(_q_user7, _q_cost7)],
            textposition="outside"))
        fig_q7.update_layout(**PLOTLY_THEME, height=340,
            title=f"Price burden by income group (regressivity {_regress7:.2f}×)")
        fig_q7.update_yaxes(title_text="Burden (% of income)")
        st.plotly_chart(fig_q7, use_container_width=True)
        _explain("The project's household-incidence results scaled by your scenario's GE consumer-price change relative to Liberation Day. This is a GE-scaled estimate, not a direct household-level GE result. Dollar figures use BLS CE 2023 average income per quintile.")
    with _dq2:
        st.dataframe(pd.DataFrame({
            "Income group": _Q_LABELS7,
            "Avg income": [f"${v:,.0f}" for v in _Q_INCOMES],
            "Burden % income": [f"{v:.2f}%" for v in _q_user7],
            "Cost $/yr": [f"${v:,.0f}" for v in _q_cost7],
        }), use_container_width=True, hide_index=True)

    # ── E. Ranking vs the paper's GE scenarios (all model-comparable) ───────
    st.markdown('<div class="section-header">Where does your policy rank? — all bars are full GE runs</div>', unsafe_allow_html=True)
    _rank7 = [
        ("USTR + No Retaliation", float(_res_g7[_idus_g7, 0, 0])),
        ("USTR + Lump-Sum Rebate", float(_res_g7[_idus_g7, 0, 7])),
        ("Optimal Tariff", float(_res_g7[_idus_g7, 0, 3])),
        ("USTR + Reciprocal Retaliation", float(_res_g7[_idus_g7, 0, 5])),
        ("USTR + Optimal Retaliation", float(_res_g7[_idus_g7, 0, 4])),
        ("Flat 15% Tariff", float(_res15_g7[_idus_g7, 0])),
        (f"⭐ {_scen_label7.upper()} (GE)", float(_R7[_idus7, 0])),
    ]
    _rank_df7 = pd.DataFrame(_rank7, columns=["scenario", "welfare"]).sort_values("welfare")
    fig_rk7 = go.Figure(go.Bar(
        x=_rank_df7["welfare"], y=_rank_df7["scenario"], orientation="h",
        marker_color=["#22d3a0" if s.startswith("⭐") else "#64748b" for s in _rank_df7["scenario"]],
        marker_line_color=["#ffffff" if s.startswith("⭐") else "rgba(0,0,0,0)" for s in _rank_df7["scenario"]],
        marker_line_width=[2 if s.startswith("⭐") else 0 for s in _rank_df7["scenario"]],
        text=[f"{v:+.2f}%" for v in _rank_df7["welfare"]], textposition="outside"))
    fig_rk7.add_vline(x=0, line_color="#4b5563", line_width=1)
    fig_rk7.update_layout(**PLOTLY_THEME, height=340,
        title="US welfare by policy — your GE run vs the paper's GE scenarios")
    st.plotly_chart(fig_rk7, use_container_width=True)
    _explain("Because your scenario is now solved with the same GE model, this comparison is apples-to-apples - no approximation caveat needed. Note where your policy lands relative to the Optimal Tariff benchmark.")

    # Retaliation stress boundary (illustrative, from the paper's GE runs)
    _pen_recip7 = float(_res_g7[_idus_g7, 0, 5] - _res_g7[_idus_g7, 0, 0])
    _pen_opt7 = float(_res_g7[_idus_g7, 0, 4] - _res_g7[_idus_g7, 0, 0])
    st.markdown(f'<div class="insight-box">🛡️ <b>Retaliation stress boundary (illustrative):</b> your run assumes no foreign retaliation. In the paper\'s GE runs, retaliation against the Liberation Day schedule cost the US <b>{_pen_recip7:+.2f}pp</b> (reciprocal) to <b>{_pen_opt7:+.2f}pp</b> (optimal) of welfare. Expect a penalty of similar sign and order if partners retaliate against your policy — a custom-retaliation GE run is a future extension, not included here.</div>', unsafe_allow_html=True)

    # ── PE exploratory tools (quarantined) ───────────────────────────────────
    with st.expander("🧪 Exploratory approximation — Laffer sweep (NOT part of the full GE run)"):
        st.markdown('<div style="color:#94a3b8;font-size:12px;margin-bottom:8px">Fast linearised sweep of tariff intensity on your selected countries (0×–2× LD). Directional only; the GE results above are the decision numbers.</div>', unsafe_allow_html=True)
        _EPS7 = 4.0
        def _rev_sweep7(mult):
            _rev = 0.0
            for _isoX, _ldX in _ld_rate_by_iso_g7.items():
                _ciX = _idx_by_iso_g7.get(_isoX)
                if _ciX is None or _ciX == _idus_g7:
                    continue
                _tau_ld = _ldX / 100.0
                _tau_new = _tau_ld * mult if _isoX in _rates7 else _tau_ld
                _m_adj = float(_imp_vec_g7[_ciX]) * ((1 + _tau_new) / (1 + _tau_ld)) ** (-_EPS7)
                _rev += _tau_new * _m_adj * 1000
            return _rev / 1e9
        _ms7 = [x / 20 for x in range(0, 41)]
        _revs7 = [_rev_sweep7(m) for m in _ms7]
        _peak7 = max(range(len(_revs7)), key=lambda i: _revs7[i])
        fig_lf7 = go.Figure(go.Scatter(x=_ms7, y=_revs7, line=dict(color="#22d3a0", width=2.5)))
        fig_lf7.add_vline(x=1.0, line_dash="dash", line_color="#94a3b8",
            annotation_text="Liberation Day", annotation_font_color="#94a3b8")
        fig_lf7.add_vline(x=_ms7[_peak7], line_dash="dot", line_color="#22d3a0",
            annotation_text=f"revenue peak ({_ms7[_peak7]:.1f}×)", annotation_position="bottom",
            annotation_font_color="#22d3a0")
        fig_lf7.update_layout(**PLOTLY_THEME, height=300,
            title="PE revenue sweep (approximation)")
        fig_lf7.update_xaxes(title_text="Tariff intensity (× LD rates on selection)")
        fig_lf7.update_yaxes(title_text="Revenue ($B, PE approx)")
        st.plotly_chart(fig_lf7, use_container_width=True)

    # ── F. Reproducibility: memo + scenario package + run-quality panel ─────
    st.markdown('<div class="section-header">📦 Take it with you</div>', unsafe_allow_html=True)

    _chg_lines7 = "\n".join(
        f"- {_name_by_iso_g7.get(iso, iso)}: {int(round(_ld_rate_by_iso_g7.get(iso, 10)))}% -> {r}%"
        for iso, r in sorted(_run_changed7.items())) or "- none (Liberation Day schedule)"
    _memo7 = f"""# Tariff Scenario Memo — {_scen_label7}
Generated: {_stored7['ts']}  ·  Model: {_GE_MODEL_VERSION}
Data: {_GE_DATA_VINTAGE}

## Tariff changes vs Liberation Day
{_chg_lines7}

## Executive results (full GE solve, no retaliation)
| Outcome | Your policy | Liberation Day | Δ |
|---|---|---|---|
| US welfare | {_R7[_idus7,0]:+.2f}% | {_RLD7[_idus7,0]:+.2f}% | {_R7[_idus7,0]-_RLD7[_idus7,0]:+.2f}pp |
| US CPI | {_R7[_idus7,5]:+.2f}% | {_RLD7[_idus7,5]:+.2f}% | {_cpi_d7:+.2f}pp |
| US employment | {_R7[_idus7,4]:+.2f}% | {_RLD7[_idus7,4]:+.2f}% | {_R7[_idus7,4]-_RLD7[_idus7,4]:+.2f}pp |
| US imports | {_R7[_idus7,3]:+.2f}% | {_RLD7[_idus7,3]:+.2f}% | {_R7[_idus7,3]-_RLD7[_idus7,3]:+.2f}pp |
| US exports | {_R7[_idus7,2]:+.2f}% | {_RLD7[_idus7,2]:+.2f}% | {_R7[_idus7,2]-_RLD7[_idus7,2]:+.2f}pp |
| Tariff revenue | ${_rev_usd7/1e9:,.0f}B | ${_rev_ld_usd7/1e9:,.0f}B | {(_rev_usd7-_rev_ld_usd7)/1e9:+,.0f}B |
| Global trade | {_ge_res7['d_trade']:+.1f}% | {_ge_ld7['d_trade']:+.1f}% | {_ge_res7['d_trade']-_ge_ld7['d_trade']:+.1f}pp |

Household translation: ~${_hh_cost_abs7:,.0f}/household/yr vs pre-tariff world ({_hh_cost_d7:+,.0f} vs LD), using BLS CE 2023 avg expenditure.

## Trade diversion vs Liberation Day
Redirected: ${_redirected7/1e9:,.1f}B · {"Destroyed" if _destroyed7 >= _created7 else "Created"}: ${max(_destroyed7,_created7)/1e9:,.1f}B · Supplier HHI: {_hhi_b7:,.0f} -> {_hhi_u7:,.0f}
Top substitute suppliers: {", ".join(_dv_sorted7.tail(5)["CountryName"].tolist()[::-1])}

## Distribution (GE-scaled estimate)
Regressivity ratio (bottom vs top quintile): {_regress7:.2f}x
{chr(10).join(f"- {l}: {b:.2f}% of income (~${c:,.0f}/yr)" for l, b, c in zip(_Q_LABELS7, _q_user7, _q_cost7))}

## Global effects
World welfare (expenditure-weighted): {_ew_world7:+.2f}% · Materially harmed (<-1%): {_harmed7} countries · Losing/gaining: {int(np.sum(_w7<0))}/{int(np.sum(_w7>0))}

## Methodology & limitations
- Full general-equilibrium solve of the Ignatenko-Macedoni-Lashkaripour-Simonovska (2025) replication model: 194 countries, eps=4, kappa=0.5, income-tax-relief revenue treatment, no foreign retaliation (scenario-0 structure).
- Engine validated to reproduce the paper's scenario 0 and flat-15% run to <0.05pp.
- Retaliation stress: paper's GE retaliation penalty on the LD schedule was {_pen_recip7:+.2f}pp (reciprocal) to {_pen_opt7:+.2f}pp (optimal) - illustrative bound only.
- Distributional section is a GE-scaled estimate; household dollars use BLS CE 2023 averages.

## Run quality
Converged: {_ge_res7['converged']} · fsolve ier={_ge_res7['fsolve_ier']} · nfev={_ge_res7['nfev']} · runtime {_ge_res7['runtime_sec']:.1f}s · max residual {_ge_res7['resid_max']:.2e}
"""

    _diag7 = {k: _ge_res7[k] for k in ["converged", "fsolve_ier", "fsolve_msg", "nfev",
                                       "resid_max", "resid_scaled", "runtime_sec",
                                       "unmatched_overrides", "model_version", "data_vintage"]}
    _baseline_err7 = float(np.max(np.abs(_RLD7[:, [0, 5]] - _res_g7[:, [0, 5], 0])))
    _diag7["baseline_reproduction_err_pp"] = _baseline_err7

    _tariff_csv7 = _cl_g7[["iso3", "CountryName"]].copy()
    _tariff_csv7["us_tariff_pct"] = np.array(_tv7) * 100
    _bilat7 = _cl_g7[["iso3", "CountryName"]].copy()
    _bilat7["us_imports_ld_usd"] = _m_base7
    _bilat7["us_imports_scenario_usd"] = _m_user7
    _bilat7["delta_usd"] = _dm7

    _zbuf7 = _io7.BytesIO()
    with _zipf7.ZipFile(_zbuf7, "w", _zipf7.ZIP_DEFLATED) as _zf:
        _zf.writestr("executive_memo.md", _memo7)
        _zf.writestr("input_tariff_schedule.csv", _tariff_csv7.to_csv(index=False))
        _zf.writestr("country_results_194.csv", _full7.to_csv(index=False))
        _zf.writestr("bilateral_us_import_changes.csv", _bilat7.to_csv(index=False))
        _zf.writestr("solver_diagnostics.json", _json7.dumps(_diag7, indent=2))
        _zf.writestr("metadata.json", _json7.dumps({
            "scenario_name": _scen_label7, "generated": _stored7["ts"],
            "model_version": _GE_MODEL_VERSION, "data_vintage": _GE_DATA_VINTAGE,
            "overrides": {k: v for k, v in _run_frozen_map7.items()},
        }, indent=2))

    _dl1, _dl2, _dl3 = st.columns(3)
    with _dl1:
        st.download_button("📄 Executive memo (.md)", _memo7.encode("utf-8"),
                           f"{_scen_label7.replace(' ', '_')}_memo.md", "text/markdown",
                           key="bys_dl_memo", use_container_width=True)
    with _dl2:
        st.download_button("🗂️ Full scenario package (.zip)", _zbuf7.getvalue(),
                           f"{_scen_label7.replace(' ', '_')}_package.zip", "application/zip",
                           key="bys_dl_zip", use_container_width=True)
    with _dl3:
        st.download_button("📊 194-country results (.csv)", _full7.to_csv(index=False).encode("utf-8"),
                           f"{_scen_label7.replace(' ', '_')}_results.csv", "text/csv",
                           key="bys_dl_csv", use_container_width=True)

    with st.expander("🔬 Run-quality panel — convergence, residuals, reproducibility"):
        st.json({
            "converged": _ge_res7["converged"],
            "fsolve_ier (1 = success)": _ge_res7["fsolve_ier"],
            "function_evaluations": _ge_res7["nfev"],
            "runtime_sec": round(_ge_res7["runtime_sec"], 2),
            "max_abs_residual": _ge_res7["resid_max"],
            "scaled_residual": _ge_res7["resid_scaled"],
            "baseline_reproduction_error_pp": _baseline_err7,
            "overrides_applied": len(_run_frozen_map7),
            "unmatched_overrides": _ge_res7["unmatched_overrides"],
            "model_version": _GE_MODEL_VERSION,
            "data_vintage": _GE_DATA_VINTAGE,
        })
