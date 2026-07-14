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

    col_ctrl, _ = st.columns([2, 5])
    with col_ctrl:
        scenario_name = st.selectbox("Scenario", list(SCENARIOS.keys()), index=0)
    sc = SCENARIOS[scenario_name]
    is_15pct = (sc is None)
    # Column order in results: [welfare, deficit, exports/GDP, imports/GDP, employment, CPI, rev/GDP]
    us_vals = results_15pct[id_US, :7] if is_15pct else results[id_US, :7, sc]

    # Bilateral trade vectors (US imports/exports by partner) + scenario scalars
    _us_imports_vec, _us_exports_vec, _d_trade_sc, _d_emp_sc = load_bilateral()
    # Import-weighted average tariff — what US buyers actually pay on average
    _tariff_by_idx = tariff_df.set_index("iso3").reindex(cl["iso3"])["tariff_pct"].fillna(0).values
    _wt_avg_tariff = float((_us_imports_vec * _tariff_by_idx).sum() / max(_us_imports_vec.sum(), 1))

    kpi_data = [
        ("Living Standards Change", f"{us_vals[0]:+.2f}%", "positive" if us_vals[0] > 0 else "negative", "US real income"),
        ("Consumer Price Change",   f"{us_vals[5]:+.1f}%", "negative" if us_vals[5] > 0 else "positive", "CPI"),
        ("Import Volume Change",    f"{us_vals[3]:+.1f}%", "negative" if us_vals[3] < 0 else "positive", "Imports / GDP"),
        ("Export Volume Change",    f"{us_vals[2]:+.1f}%", "negative" if us_vals[2] < 0 else "positive", "Exports / GDP"),
        ("Employment Change",       f"{us_vals[4]:+.2f}%", "positive" if us_vals[4] > 0 else "negative", "US labor market"),
        ("Import-Weighted Avg Tariff", f"{_wt_avg_tariff:.1f}%", "negative", "Weighted by what US actually buys"),
    ]
    cols = st.columns(6)
    for col, (label, val, cls, sub) in zip(cols, kpi_data):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value {cls}">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

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

    # ── Which countries face the highest US tariffs? ───────────────────────
    st.markdown('<div class="section-header">Which countries face the highest total US tariff rates?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Total US applied tariff rate as of Liberation Day (April 2, 2025), including any pre-existing tariffs. China\'s 54% = 34% Liberation Day reciprocal + 20% imposed earlier in 2025. Most other countries face 10–34%.</div>', unsafe_allow_html=True)

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

    # ── Was Liberation Day really "reciprocal"? ──────────────────────────────
    st.markdown('<div class="section-header">Was Liberation Day really "reciprocal"?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">The administration said tariffs were proportional to each country\'s trade imbalance with the US. This scatter tests that claim: if it were true, the dots would line up neatly. China (−$295B deficit → 54%) fits; but Vietnam (−$108B → 46%) is taxed far harder than Germany (−$85B → 20%). Bubble size = how much the US imports from that country.</div>', unsafe_allow_html=True)

    _recip = cl.copy()
    _recip["imports_from"] = _us_imports_vec
    _recip["exports_to"]   = _us_exports_vec
    _recip["deficit_bn"]   = (_us_exports_vec - _us_imports_vec) / 1e6
    _recip["imports_bn"]   = _us_imports_vec / 1e6
    _recip["tariff_pct"]   = _tariff_by_idx
    _recip = _recip[_recip.index != id_US]
    _recip_top = _recip.nlargest(40, "imports_from").copy()

    _sel_name_sc = _country_profile["country"] if _country_profile else None
    if _sel_name_sc and _sel_name_sc not in _recip_top["CountryName"].values:
        _extra_sc = _recip[_recip["CountryName"] == _sel_name_sc]
        if not _extra_sc.empty:
            _recip_top = pd.concat([_recip_top, _extra_sc])

    _sc_colors = ["#22d3a0" if (_sel_name_sc and n == _sel_name_sc) else
                  ("#f87171" if t >= 40 else "#fb923c" if t >= 20 else "#fbbf24")
                  for n, t in zip(_recip_top["CountryName"], _recip_top["tariff_pct"])]
    fig_recip = go.Figure(go.Scatter(
        x=_recip_top["deficit_bn"], y=_recip_top["tariff_pct"],
        mode="markers+text",
        marker=dict(
            size=np.sqrt(np.maximum(_recip_top["imports_bn"], 1)) * 2.2,
            color=_sc_colors, opacity=0.85,
            line=dict(color="#0f1117", width=1),
        ),
        text=[n if (abs(d) > 40 or t >= 40 or (_sel_name_sc and n == _sel_name_sc)) else ""
              for n, d, t in zip(_recip_top["CountryName"], _recip_top["deficit_bn"], _recip_top["tariff_pct"])],
        textposition="top center", textfont=dict(size=10, color="#94a3b8"),
        hovertemplate="<b>%{customdata[0]}</b><br>US bilateral balance: %{x:,.0f}B<br>Tariff: %{y:.0f}%<br>US imports: $%{customdata[1]:,.0f}B<extra></extra>",
        customdata=np.stack([_recip_top["CountryName"], _recip_top["imports_bn"]], axis=-1),
    ))
    fig_recip.add_vline(x=0, line_color="#4b5563", line_width=1, line_dash="dot")
    fig_recip.update_layout(**PLOTLY_THEME, height=440,
        title="US Bilateral Trade Balance vs Liberation Day Tariff (top 40 US import partners)")
    fig_recip.update_xaxes(title_text="US Trade Balance with Country ($B — negative = US deficit)")
    fig_recip.update_yaxes(title_text="Liberation Day Tariff (%)")
    st.plotly_chart(fig_recip, use_container_width=True)

    # ── Who depends most on the US market? ───────────────────────────────────
    st.markdown('<div class="section-header">Who depends most on the US market — and what happened to them?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">2023 merchandise exports to the United States as a share of exporter GDP (CEPII calibration); welfare change under <b>USTR tariffs + No Retaliation</b>, verified against the replication\'s output_map.csv. <b>Dependence predicts damage</b>: Vietnam sends 25% of its GDP to the US and lost 11%; Guyana sends 18% and lost 18.5%.<br><br>'
        '<b>Bar colors:</b> <span style="color:#22d3a0">■ green = welfare gain</span> · <span style="color:#fbbf24">■ yellow = loss under 1%</span> · <span style="color:#fb923c">■ orange = loss 1–5%</span> · <span style="color:#f87171">■ red = loss over 5%</span> · <span style="color:#ffffff">■ white = your selected country</span></div>', unsafe_allow_html=True)

    _dep_df = cl.copy()
    _dep_df["dep_pct"] = np.where(Y_i > 0, _us_imports_vec / Y_i * 100, 0)
    _dep_df["welfare"] = results[:, 0, 0]
    _dep_df = _dep_df[_dep_df.index != id_US]
    _dep_top = _dep_df.nlargest(20, "dep_pct").copy()

    _sel_name_dep = _country_profile["country"] if _country_profile else None
    if _sel_name_dep and _sel_name_dep not in _dep_top["CountryName"].values:
        _extra_dep = _dep_df[_dep_df["CountryName"] == _sel_name_dep]
        if not _extra_dep.empty:
            _dep_top = pd.concat([_dep_top, _extra_dep])

    _dep_colors = ["#ffffff" if (_sel_name_dep and n == _sel_name_dep) else
                   ("#f87171" if w < -5 else "#fb923c" if w < -1 else "#fbbf24" if w < 0 else "#22d3a0")
                   for n, w in zip(_dep_top["CountryName"], _dep_top["welfare"])]
    fig_dep_us = go.Figure(go.Bar(
        x=_dep_top["dep_pct"], y=_dep_top["CountryName"],
        orientation="h", marker_color=_dep_colors,
        marker_line_color=["#22d3a0" if (_sel_name_dep and n == _sel_name_dep) else "rgba(0,0,0,0)" for n in _dep_top["CountryName"]],
        marker_line_width=[3 if (_sel_name_dep and n == _sel_name_dep) else 0 for n in _dep_top["CountryName"]],
        text=[f"{d:.1f}% of GDP | welfare {w:+.1f}%" for d, w in zip(_dep_top["dep_pct"], _dep_top["welfare"])],
        textposition="outside",
    ))
    fig_dep_us.update_layout(**PLOTLY_THEME, height=520,
        title="2023 Exports to US as % of Own GDP — welfare under USTR + No Retaliation")
    fig_dep_us.update_xaxes(title_text="Exports to US / GDP (%)", range=[0, _dep_top["dep_pct"].max() * 1.45])
    fig_dep_us.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_dep_us, use_container_width=True)

    # ── World map ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">How each country\'s economy was affected</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Welfare change is the estimated effect on each country\'s real income. Green = economic gains; red = losses. All 194 countries are modelled simultaneously in a general equilibrium framework.</div>', unsafe_allow_html=True)

    welfare_vals = results_15pct[:, 0] if is_15pct else results[:, 0, sc]
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

    # ── Regional breakdown ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Which regions were hit hardest?</div>', unsafe_allow_html=True)

    _REGION_MAP_T1 = {
        "USA":"North America","CAN":"North America","MEX":"North America",
        "CHN":"East Asia","JPN":"East Asia","KOR":"East Asia","TWN":"East Asia","HKG":"East Asia","PRK":"East Asia","MNG":"East Asia","MAC":"East Asia",
        "DEU":"Europe","FRA":"Europe","GBR":"Europe","ITA":"Europe","NLD":"Europe","BEL":"Europe","ESP":"Europe","CHE":"Europe","AUT":"Europe","SWE":"Europe","DNK":"Europe","NOR":"Europe","FIN":"Europe","PRT":"Europe","IRL":"Europe","GRC":"Europe","POL":"Europe","CZE":"Europe","HUN":"Europe","ROU":"Europe","SVK":"Europe","BGR":"Europe","HRV":"Europe","SVN":"Europe","LTU":"Europe","LVA":"Europe","EST":"Europe","LUX":"Europe","MLT":"Europe","CYP":"Europe","RUS":"Europe","UKR":"Europe","BLR":"Europe","MDA":"Europe","GEO":"Europe","ARM":"Europe","AZE":"Europe","TUR":"Europe","MKD":"Europe","SRB":"Europe","BIH":"Europe","MNE":"Europe","ALB":"Europe","ISL":"Europe",
        "IND":"South Asia","PAK":"South Asia","BGD":"South Asia","LKA":"South Asia","NPL":"South Asia","AFG":"South Asia","MDV":"South Asia","BTN":"South Asia",
        "BRA":"Latin America","ARG":"Latin America","COL":"Latin America","CHL":"Latin America","PER":"Latin America","VEN":"Latin America","ECU":"Latin America","BOL":"Latin America","PRY":"Latin America","URY":"Latin America","GTM":"Latin America","HND":"Latin America","SLV":"Latin America","CRI":"Latin America","PAN":"Latin America","DOM":"Latin America","CUB":"Latin America","HTI":"Latin America","JAM":"Latin America","TTO":"Latin America","BRB":"Latin America","BHS":"Latin America","GUY":"Latin America","SUR":"Latin America","BLZ":"Latin America","NIC":"Latin America",
        "SAU":"Middle East & Africa","ARE":"Middle East & Africa","IRN":"Middle East & Africa","ISR":"Middle East & Africa","EGY":"Middle East & Africa","ZAF":"Middle East & Africa","NGA":"Middle East & Africa","KEN":"Middle East & Africa","ETH":"Middle East & Africa","GHA":"Middle East & Africa","TZA":"Middle East & Africa","UGA":"Middle East & Africa","AGO":"Middle East & Africa","MOZ":"Middle East & Africa","ZMB":"Middle East & Africa","MAR":"Middle East & Africa","TUN":"Middle East & Africa","DZA":"Middle East & Africa","LBY":"Middle East & Africa","SDN":"Middle East & Africa","IRQ":"Middle East & Africa","SYR":"Middle East & Africa","LBN":"Middle East & Africa","JOR":"Middle East & Africa","KWT":"Middle East & Africa","BHR":"Middle East & Africa","QAT":"Middle East & Africa","OMN":"Middle East & Africa","YEM":"Middle East & Africa","CMR":"Middle East & Africa","CIV":"Middle East & Africa","SEN":"Middle East & Africa","MDG":"Middle East & Africa","MLI":"Middle East & Africa","BFA":"Middle East & Africa","NER":"Middle East & Africa","TCD":"Middle East & Africa","GIN":"Middle East & Africa","SLE":"Middle East & Africa","LBR":"Middle East & Africa","BEN":"Middle East & Africa","TGO":"Middle East & Africa","RWA":"Middle East & Africa","BDI":"Middle East & Africa","MWI":"Middle East & Africa","ZWE":"Middle East & Africa","BWA":"Middle East & Africa","NAM":"Middle East & Africa","LSO":"Middle East & Africa","SWZ":"Middle East & Africa","DJI":"Middle East & Africa","ERI":"Middle East & Africa","COD":"Middle East & Africa","CAF":"Middle East & Africa","COG":"Middle East & Africa","GAB":"Middle East & Africa","GNQ":"Middle East & Africa","STP":"Middle East & Africa","CPV":"Middle East & Africa","GMB":"Middle East & Africa","GNB":"Middle East & Africa","MRT":"Middle East & Africa","COM":"Middle East & Africa","MUS":"Middle East & Africa","SYC":"Middle East & Africa",
        "AUS":"Asia-Pacific","NZL":"Asia-Pacific","SGP":"Asia-Pacific","MYS":"Asia-Pacific","THA":"Asia-Pacific","IDN":"Asia-Pacific","PHL":"Asia-Pacific","VNM":"Asia-Pacific","MMR":"Asia-Pacific","KHM":"Asia-Pacific","LAO":"Asia-Pacific","BRN":"Asia-Pacific","PNG":"Asia-Pacific","FJI":"Asia-Pacific","KAZ":"Asia-Pacific","UZB":"Asia-Pacific","TKM":"Asia-Pacific","KGZ":"Asia-Pacific","TJK":"Asia-Pacific",
    }

    reg_data = []
    for i, row in cl.iterrows():
        iso = row["iso3"]
        region = _REGION_MAP_T1.get(iso, "Other")
        reg_data.append({"iso3": iso, "region": region, "welfare": float(welfare_vals[i]), "gdp": float(Y_i[i])})
    reg_df = pd.DataFrame(reg_data)

    def _gdp_wt_avg(g):
        w = g["gdp"]
        tot = w.sum()
        return (g["welfare"] * w).sum() / tot if tot > 0 else 0

    region_summary = reg_df.groupby("region").apply(
        lambda g: pd.Series({
            "Countries": len(g),
            "Simple Avg %": g["welfare"].mean(),
            "GDP-Weighted %": _gdp_wt_avg(g),
        })
    ).reset_index().sort_values("GDP-Weighted %")

    c_reg1, c_reg2 = st.columns([3, 2])
    with c_reg1:
        colors_reg = ["#f87171" if v < 0 else "#22d3a0" for v in region_summary["GDP-Weighted %"]]
        fig_reg = go.Figure(go.Bar(
            x=region_summary["GDP-Weighted %"], y=region_summary["region"],
            orientation="h", marker_color=colors_reg,
            text=[f"{v:+.2f}%" for v in region_summary["GDP-Weighted %"]],
            textposition="outside",
        ))
        fig_reg.add_vline(x=0, line_color="#4b5563", line_width=1)
        fig_reg.update_layout(**PLOTLY_THEME, height=340,
            title="GDP-Weighted Welfare Change by World Region",
            xaxis_title="Welfare % (GDP-weighted average)")
        fig_reg.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_reg, use_container_width=True)

    with c_reg2:
        disp_reg = region_summary.copy()
        disp_reg["Simple Avg %"] = disp_reg["Simple Avg %"].map(lambda x: f"{x:+.2f}%")
        disp_reg["GDP-Weighted %"] = disp_reg["GDP-Weighted %"].map(lambda x: f"{x:+.2f}%")
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(disp_reg.reset_index(drop=True), use_container_width=True, hide_index=True)

    # ── Scenario comparison ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Scenario Comparison — Different Tariff Approaches</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        sc_names = list(SCENARIOS.keys())
        sc_idxs  = list(SCENARIOS.values())
        welfare_by_sc = [results_15pct[id_US, 0] if i is None else results[id_US, 0, i] for i in sc_idxs]
        cpi_by_sc     = [results_15pct[id_US, 5] if i is None else results[id_US, 5, i] for i in sc_idxs]

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Bar(
            name="US Living Standards", x=sc_names, y=welfare_by_sc,
            marker_color=["#22d3a0" if v > 0 else "#f87171" for v in welfare_by_sc],
            text=[f"{v:+.2f}%" for v in welfare_by_sc], textposition="outside",
        ))
        fig_sc.add_trace(go.Bar(
            name="Consumer Price Change", x=sc_names, y=cpi_by_sc,
            marker_color="#fbbf24", opacity=0.7,
            text=[f"{v:+.1f}%" for v in cpi_by_sc], textposition="outside",
        ))
        fig_sc.update_layout(**PLOTLY_THEME, height=360,
            title="US Living Standards & Prices Across Scenarios",
            barmode="group", xaxis_tickangle=-25,
            legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250"))
        st.plotly_chart(fig_sc, use_container_width=True)

    with c2:
        METRICS_T1 = ["Living Standards", "Consumer Prices", "Imports/GDP", "Exports/GDP", "Employment", "Trade Deficit", "Tax Revenue"]
        col_order  = [0, 5, 3, 2, 4, 1, 6]
        metrics_vals = [results_15pct[id_US, c] if is_15pct else results[id_US, c, sc] for c in col_order]
        fig_met = go.Figure(go.Bar(
            x=METRICS_T1, y=metrics_vals,
            marker_color=["#22d3a0" if v >= 0 else "#f87171" for v in metrics_vals],
            text=[f"{v:+.2f}%" for v in metrics_vals], textposition="outside",
        ))
        fig_met.update_layout(**PLOTLY_THEME, height=360,
            title=f"US Economic Outcomes — {scenario_name}",
            yaxis_title="% Change")
        st.plotly_chart(fig_met, use_container_width=True)

    # ── What happens to world trade? ─────────────────────────────────────────
    st.markdown('<div class="section-header">What happens to world trade under each policy?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Every scenario shrinks global trade. Liberation Day alone cuts it by ~9.4%; if trading partners retaliate optimally, the damage deepens to ~12.3% — retaliation roughly doubles the trade destruction while barely helping anyone.</div>', unsafe_allow_html=True)

    _SC_SCALAR_MAP = [
        ("USTR + No Retaliation",         0),
        ("Optimal Tariff",                3),
        ("USTR + Reciprocal Retaliation", 5),
        ("USTR + Optimal Retaliation",    4),
    ]
    _sc_labels_ws  = [n for n, _ in _SC_SCALAR_MAP]
    _sc_trade_ws   = [float(_d_trade_sc[i]) for _, i in _SC_SCALAR_MAP]
    _sc_emp_ws     = [float(_d_emp_sc[i])   for _, i in _SC_SCALAR_MAP]

    c_ws1, c_ws2 = st.columns(2)
    with c_ws1:
        fig_wtrade = go.Figure(go.Bar(
            x=_sc_labels_ws, y=_sc_trade_ws,
            marker_color=["#f87171" if v < -10 else "#fb923c" for v in _sc_trade_ws],
            text=[f"{v:+.1f}%" for v in _sc_trade_ws], textposition="outside",
        ))
        fig_wtrade.add_hline(y=0, line_color="#4b5563", line_width=1)
        fig_wtrade.update_layout(**PLOTLY_THEME, height=340,
            title="Global Trade Volume Change by Scenario", xaxis_tickangle=-20)
        fig_wtrade.update_yaxes(title_text="% Change in World Trade", range=[-15, 2])
        st.plotly_chart(fig_wtrade, use_container_width=True)
    with c_ws2:
        fig_wemp = go.Figure(go.Bar(
            x=_sc_labels_ws, y=_sc_emp_ws,
            marker_color=["#22d3a0" if v >= 0 else "#f87171" for v in _sc_emp_ws],
            text=[f"{v:+.2f}%" for v in _sc_emp_ws], textposition="outside",
        ))
        fig_wemp.add_hline(y=0, line_color="#4b5563", line_width=1)
        fig_wemp.update_layout(**PLOTLY_THEME, height=340,
            title="US Employment Change by Scenario", xaxis_tickangle=-20)
        fig_wemp.update_yaxes(title_text="% Change in US Employment")
        st.plotly_chart(fig_wemp, use_container_width=True)

    # ── US vs major economies ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Did America come out ahead of its rivals?</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">Head-to-head outcomes for the world\'s major economies under <b>{scenario_name}</b>. Change the scenario selector at the top of this tab to compare.</div>', unsafe_allow_html=True)

    _MAJORS = [("USA", "United States"), ("CHN", "China"), ("DEU", "Germany"),
               ("JPN", "Japan"), ("GBR", "United Kingdom"), ("CAN", "Canada"), ("MEX", "Mexico")]
    _maj_rows = []
    for _iso_m, _name_m in _MAJORS:
        _midx = cl.index[cl["iso3"] == _iso_m]
        if len(_midx) == 0:
            continue
        _mi = int(_midx[0])
        _mvals = results_15pct[_mi, :7] if is_15pct else results[_mi, :7, sc]
        _maj_rows.append({"name": _name_m, "welfare": float(_mvals[0]),
                          "cpi": float(_mvals[5]), "exports": float(_mvals[2])})
    _maj_df = pd.DataFrame(_maj_rows)

    fig_maj = go.Figure()
    fig_maj.add_trace(go.Bar(
        name="Living Standards", x=_maj_df["name"], y=_maj_df["welfare"],
        marker_color=["#22d3a0" if v > 0 else "#f87171" for v in _maj_df["welfare"]],
        text=[f"{v:+.2f}%" for v in _maj_df["welfare"]], textposition="outside",
    ))
    fig_maj.add_trace(go.Bar(
        name="Consumer Prices", x=_maj_df["name"], y=_maj_df["cpi"],
        marker_color="#fbbf24", opacity=0.75,
        text=[f"{v:+.1f}%" for v in _maj_df["cpi"]], textposition="outside",
    ))
    fig_maj.add_trace(go.Bar(
        name="Export Volume", x=_maj_df["name"], y=_maj_df["exports"],
        marker_color="#60a5fa", opacity=0.75,
        text=[f"{v:+.1f}%" for v in _maj_df["exports"]], textposition="outside",
    ))
    fig_maj.add_hline(y=0, line_color="#4b5563", line_width=1)
    fig_maj.update_layout(**PLOTLY_THEME, height=400,
        title=f"Major Economies Head-to-Head — {scenario_name}",
        barmode="group",
        legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250"))
    fig_maj.update_yaxes(title_text="% Change")
    st.plotly_chart(fig_maj, use_container_width=True)

    # ── The global scoreboard ────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">The global scoreboard — how the world\'s welfare is distributed ({scenario_name})</div>', unsafe_allow_html=True)

    _wf_all = welfare_vals  # current scenario's welfare for all 194 countries
    _n_lose = int((_wf_all < 0).sum())
    _n_gain = int((_wf_all > 0).sum())
    _global_bn = float((_wf_all / 100 * Y_i).sum() / 1e6)  # Y_i in $1000s → $B

    c_gs1, c_gs2, c_gs3 = st.columns(3)
    with c_gs1:
        _g_cls = "negative" if _global_bn < 0 else "positive"
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Global welfare change</div>
          <div class="kpi-value {_g_cls}" style="font-size:28px">{_global_bn:+,.0f}B</div>
          <div class="kpi-sub">GDP-weighted dollar total, all 194 countries</div>
        </div>""", unsafe_allow_html=True)
    with c_gs2:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Countries that lose</div>
          <div class="kpi-value negative" style="font-size:28px">{_n_lose}</div>
          <div class="kpi-sub">welfare change below zero</div>
        </div>""", unsafe_allow_html=True)
    with c_gs3:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Countries that gain</div>
          <div class="kpi-value positive" style="font-size:28px">{_n_gain}</div>
          <div class="kpi-sub">mostly small economies picking up rerouted trade</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="insight-box" style="margin-top:12px">Roughly half the world\'s countries technically "gain" — but the gains are small and concentrated in minor economies, while the losses are large and concentrated in major US trading partners. The world as a whole is poorer.</div>', unsafe_allow_html=True)

    _hist_df = pd.DataFrame({"welfare": _wf_all})
    fig_hist_w = go.Figure()
    fig_hist_w.add_trace(go.Histogram(
        x=_hist_df[_hist_df["welfare"] < 0]["welfare"],
        xbins=dict(start=-20, end=6, size=0.5),
        marker_color="#f87171", name="Losers", opacity=0.85,
    ))
    fig_hist_w.add_trace(go.Histogram(
        x=_hist_df[_hist_df["welfare"] >= 0]["welfare"],
        xbins=dict(start=-20, end=6, size=0.5),
        marker_color="#22d3a0", name="Winners", opacity=0.85,
    ))
    fig_hist_w.add_vline(x=0, line_color="#e2e8f0", line_width=1, line_dash="dash")
    fig_hist_w.update_layout(**PLOTLY_THEME, height=320,
        title="Distribution of Welfare Changes Across 194 Countries",
        barmode="overlay",
        legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250"))
    fig_hist_w.update_xaxes(title_text="Welfare Change (%)")
    fig_hist_w.update_yaxes(title_text="# of Countries")
    st.plotly_chart(fig_hist_w, use_container_width=True)

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

    # ── What types of medicines does the US import? ─────────────────────────
    st.markdown('<div class="section-header">What types of medicines does the US import?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Three HTS product categories cover all US pharma imports. Biologics & vaccines and ready-made drugs together account for 99% of the total.</div>', unsafe_allow_html=True)

    hts_labels = {3002: "Biologics & Vaccines (HTS 3002)", 3003: "Chemical Compounds (HTS 3003)", 3004: "Ready-Made Medicines (HTS 3004)"}
    hts_exp_plot = hts_exp.copy()
    hts_exp_plot["label"] = hts_exp_plot["hts"].map(hts_labels)
    c_hts1, c_hts2 = st.columns([3, 2])
    with c_hts1:
        fig_hts_bar = go.Figure(go.Bar(
            x=hts_exp_plot["import_share_pct"], y=hts_exp_plot["label"],
            orientation="h",
            marker_color=["#2563eb","#fbbf24","#22d3a0"],
            text=[f"{v:.1f}%  (${bn:.0f}B)" for v, bn in zip(hts_exp_plot["import_share_pct"], hts_exp_plot["import_value_bn"])],
            textposition="outside",
        ))
        fig_hts_bar.update_layout(**PLOTLY_THEME, height=220,
            title=f"US Pharma Imports by Product Type (Total: ${hts_exp_plot['import_value_bn'].sum():.0f}B)",
            xaxis_title="Share of Total Imports (%)")
        fig_hts_bar.update_xaxes(range=[0, 70])
        fig_hts_bar.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_hts_bar, use_container_width=True)
    with c_hts2:
        st.markdown("""
        <div class="insight-box">
        <b>HTS 3002</b> — Biologics & vaccines: $108B (51%)<br>
        <b>HTS 3004</b> — Ready-made drugs: $100B (48%)<br>
        <b>HTS 3003</b> — Chemical compounds: $2.4B (1%)<br><br>
        All three face Liberation Day tariffs, with <b>no pharma exemption</b>.
        </div>""", unsafe_allow_html=True)

    # ── Monthly import trends ────────────────────────────────────────────────
    st.markdown('<div class="section-header">US medicine import spending, 2018–2025</div>', unsafe_allow_html=True)

    hts_filter = st.multiselect(
        "Filter by HTS code", [3002, 3003, 3004],
        default=[3002, 3003, 3004],
        format_func=lambda x: {3002:"3002 – Biologics & Vaccines", 3003:"3003 – Chemical Compounds", 3004:"3004 – Ready-Made Drugs"}[x]
    )
    ts_df = pharma_df[pharma_df["HTS Number"].isin(hts_filter)].copy()
    rows_ts = []
    for _, row in ts_df.iterrows():
        yr = int(row["Year"]) if not pd.isna(row["Year"]) else None
        if yr is None: continue
        for mi, m in enumerate(monthly_cols, 1):
            val = pd.to_numeric(row[m], errors="coerce")
            if pd.notna(val) and val > 0:
                rows_ts.append({"year": yr, "month": mi, "value": val, "hts": int(row["HTS Number"])})
    ts = pd.DataFrame(rows_ts)
    if not ts.empty:
        ts["date"] = pd.to_datetime(ts[["year","month"]].assign(day=1).rename(columns={"year":"year","month":"month"}))
        ts_agg = ts.groupby(["date","hts"])["value"].sum().reset_index()
        ts_agg["value_bn"] = ts_agg["value"] / 1e9
        fig_ts = px.line(ts_agg, x="date", y="value_bn", color="hts",
            color_discrete_sequence=["#2563eb","#22d3a0","#fbbf24"],
            labels={"value_bn":"Import Value ($B)", "date":"Date", "hts":"HTS Code"},
        )
        fig_ts.add_vrect(x0="2025-04-01", x1="2025-05-01",
            fillcolor="#f87171", opacity=0.15,
            annotation_text="Liberation Day", annotation_position="top left",
            annotation_font_color="#f87171")
        fig_ts.update_traces(line_width=2)
        apply_theme(fig_ts, 320)
        st.plotly_chart(fig_ts, use_container_width=True)

    # ── Supplier dependence + rank changes ──────────────────────────────────
    st.markdown('<div class="section-header">Where does the US buy its medicines — and who\'s moving up or down?</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        tier_colors = {"Very high": "#f87171", "High": "#fb923c", "Low": "#22d3a0"}
        exp_top = exp.head(15).copy()
        # Include selected country even if outside top 15
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
        _dep_title = f"Top 15 Suppliers — {_sel_pharma_name} highlighted" if _sel_pharma_name else "Top 15 Suppliers (red=Very High risk, orange=High, green=Low)"
        fig_dep.update_layout(**PLOTLY_THEME, height=420,
            title=_dep_title,
            xaxis_title="Import Share %")
        fig_dep.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_dep, use_container_width=True)

    with c2:
        arrow_fn = lambda d: "▲" if d > 0 else ("▼" if d < 0 else "→")
        ts_rank = top_sup.copy()
        ts_rank["Rank Change"] = ts_rank["rank_change"].apply(lambda x: f"{arrow_fn(x)} {abs(int(x))}" if x != 0 else "→ 0")
        ts_rank["Imports"] = ts_rank.apply(lambda r: f"${r['imports_2024_bn']:.0f}B → ${r['imports_2025_bn']:.0f}B", axis=1)
        disp_sup = ts_rank[["country","rank_2024","rank_2025","Rank Change","Imports"]].copy()
        disp_sup.columns = ["Country","2024 Rank","2025 Rank","Rank Δ","Imports ($B)"]
        st.markdown("<br><b>How suppliers are reshuffling after tariffs</b>", unsafe_allow_html=True)
        st.dataframe(disp_sup, use_container_width=True, hide_index=True)
        st.markdown('<div class="insight-box">Singapore dropped 5 places as its 10% tariff (low, but high absolute value) squeezes margins. Germany rose 2 places as trade reroutes. Ireland stays #1 but its share is shrinking.</div>', unsafe_allow_html=True)

    # ── Risk tier chart ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Which suppliers are riskiest for the US?</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">Risk = high import share + high tariff. Ireland supplies 26% of US medicines at a 20% tariff — the top supply chain concentration risk. The HHI score rose from {pharma_stats["hhi_pre"]:.0f} (pre-tariff) to {pharma_stats["hhi_post"]:.0f} (post-tariff), meaning supply is becoming more concentrated.</div>', unsafe_allow_html=True)

    tier_order = {"Very high": 0, "High": 1, "Low": 2}
    risk_sorted = risk.sort_values(by="risk_tier", key=lambda x: x.map(tier_order))
    risk_colors_t2 = [tier_colors.get(t, "#60a5fa") for t in risk_sorted["risk_tier"]]
    fig_risk = go.Figure(go.Bar(
        x=risk_sorted["pre_tariff_share_pct"], y=risk_sorted["country"],
        orientation="h", marker_color=risk_colors_t2,
        text=[f"{s:.1f}% share | {t:.0f}% tariff | {tier}" for s, t, tier in zip(
            risk_sorted["pre_tariff_share_pct"], risk_sorted["tariff_pct"], risk_sorted["risk_tier"])],
        textposition="outside",
    ))
    fig_risk.update_layout(**PLOTLY_THEME, height=400,
        title="Supply Chain Risk: Import Share vs Tariff Rate",
        xaxis_title="Pre-Tariff Import Share (%)")
    fig_risk.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_risk, use_container_width=True)

    # ── Sourcing shifts ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Who gains and who loses US pharma business after tariffs?</div>', unsafe_allow_html=True)
    src_plot = src.head(20).copy()
    _sel_src_name = _country_profile["country"] if _country_profile else None
    # Include selected country if not in top 20
    if _sel_src_name and not src_plot["country"].str.lower().str.contains(_sel_src_name[:5].lower(), na=False).any():
        _sel_src_row = src[src["country"].str.lower().str.contains(_sel_src_name[:5].lower(), na=False)]
        if not _sel_src_row.empty:
            src_plot = pd.concat([src_plot, _sel_src_row]).reset_index(drop=True)
    _src_colors = []
    _src_hl_lines = []
    _src_hl_widths = []
    for ctry, v in zip(src_plot["country"], src_plot["change_pp"]):
        is_sel = _sel_src_name and _sel_src_name[:5].lower() in ctry.lower()
        _src_colors.append("#ffffff" if is_sel else ("#22d3a0" if v >= 0 else "#f87171"))
        _src_hl_lines.append("#22d3a0" if is_sel else "rgba(0,0,0,0)")
        _src_hl_widths.append(3 if is_sel else 0)
    fig_src = go.Figure(go.Bar(
        x=src_plot["change_pp"], y=src_plot["country"],
        orientation="h",
        marker_color=_src_colors,
        marker_line_color=_src_hl_lines, marker_line_width=_src_hl_widths,
        text=[f"◀ {v:+.2f}pp — selected" if (_sel_src_name and _sel_src_name[:5].lower() in ctry.lower()) else f"{v:+.2f}pp" for ctry, v in zip(src_plot["country"], src_plot["change_pp"])],
        textposition="outside",
    ))
    fig_src.add_vline(x=0, line_color="#4b5563", line_width=1)
    _src_title = f"Change in US Pharma Import Share — {_sel_src_name} highlighted" if _sel_src_name else "Change in US Pharma Import Share After Tariffs (percentage points)"
    fig_src.update_layout(**PLOTLY_THEME, height=440,
        title=_src_title,
        xaxis_title="Change in Import Share (pp)")
    fig_src.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_src, use_container_width=True)

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

    # ── How did prices actually change? ─────────────────────────────────────
    st.markdown('<div class="section-header">How did prices actually change?</div>', unsafe_allow_html=True)
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

    # ── Why was the price rise smaller than expected? ──────────────────────
    st.markdown('<div class="section-header">Why was the price rise smaller than expected?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">The naïve estimate assumed all tariff costs passed directly to consumers. In reality, exporters cut prices, trade flows rerouted, and some purchases switched to domestic goods — reducing the actual impact from 13.6% to 7.1% in the model and to ~0.1% in early empirical data.</div>', unsafe_allow_html=True)

    estimates_t3 = ["First-order estimate\n(naïve, no trade response)", f"General equilibrium\nmodel (trade rerouting)", f"Empirical data\n(Cavallo, 90 days)"]
    estimate_vals_t3 = [first_order, ge_cpi, cav_90d]
    fig_est = go.Figure(go.Bar(
        x=estimates_t3, y=estimate_vals_t3,
        marker_color=["#f87171","#fbbf24","#22d3a0"],
        text=[f"+{v:.2f}%" for v in estimate_vals_t3],
        textposition="outside",
    ))
    fig_est.update_layout(**PLOTLY_THEME, height=320,
        title="Three Price Estimates: Naïve vs Model vs Empirical",
        yaxis_title="% Price Increase")
    fig_est.update_yaxes(range=[0, first_order * 1.25])
    st.plotly_chart(fig_est, use_container_width=True)

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
    st.markdown('<div class="insight-box">If trading partners retaliate, US consumer prices actually rise <i>less</i> (because import demand falls and exporters cut prices to compete). However, the US economy shrinks overall because US export markets shut down in response.</div>', unsafe_allow_html=True)

    # ── US vs China: price divergence ──────────────────────────────────────
    st.markdown('<div class="section-header">US vs China: prices moving in opposite directions</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">After Liberation Day, US prices rose while Chinese export prices fell. Chinese manufacturers absorbed much of the tariff cost by cutting prices to maintain competitiveness, rather than losing market share entirely.</div>', unsafe_allow_html=True)

    lib_day = pd.Timestamp("2025-04-02")
    cav_post = cavallo[cavallo["date"] >= lib_day].copy()
    fig_div = go.Figure()
    fig_div.add_trace(go.Scatter(
        x=cav_post["date"], y=(cav_post["index_usa"] - 1) * 100,
        name="United States", line=dict(color="#2563eb", width=2.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.10)",
    ))
    fig_div.add_trace(go.Scatter(
        x=cav_post["date"], y=(cav_post["index_china"] - 1) * 100,
        name="China", line=dict(color="#f87171", width=2.5),
        fill="tozeroy", fillcolor="rgba(248,113,113,0.10)",
    ))
    fig_div.add_hline(y=0, line_color="#4b5563", line_width=1, line_dash="dot")
    fig_div.update_layout(**PLOTLY_THEME, height=360,
        title="US Rising, China Falling — Price Index Post Liberation Day (Oct 2024 = 0%)",
        yaxis_title="% Change from Oct 2024 baseline",
        legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250"))
    st.plotly_chart(fig_div, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — MANUFACTURING EXPOSURE
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    naics, price_idx, shocks, hts, mfg_stats = load_manufacturing()

    kpi4 = [
        ("Average tariff on factory imports",     f"{mfg_stats['tau_mfg_avg']*100:.1f}%",          "negative", "Trade-weighted avg"),
        ("Imports as % of factory output",        f"{mfg_stats['import_penetration_mfg']*100:.1f}%","warning",  "Import penetration"),
        ("Factory tariffs' share of price rises", f"+{mfg_stats['cpi_mfg_contribution']:.1f}pp",   "negative", "Of +7.1pp total CPI"),
        ("Supply chain multiplier",               f"{mfg_stats['io_mult_mfg']:.2f}×",              "warning",  "IO amplification"),
        ("Drop in manufacturing imports",         f"{mfg_stats['mfg_import_change']:+.1f}%",        "negative", "GE model estimate"),
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

    # ── How much did manufacturing imports fall? ───────────────────────────
    st.markdown('<div class="section-header">How much did manufacturing imports fall?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">The model predicts US manufacturing imports would drop by <b>80.8%</b> — the single largest sectoral trade shock. High import penetration (32%) means factories heavily depend on imported inputs.</div>', unsafe_allow_html=True)

    c_imp1, c_imp2 = st.columns([1, 2])
    with c_imp1:
        import_changes_t4 = {"Manufacturing": mfg_stats["mfg_import_change"], "Pharma": -38.17}
        fig_imp = go.Figure(go.Bar(
            y=list(import_changes_t4.keys()), x=list(import_changes_t4.values()),
            orientation="h", marker_color=["#f87171","#fb923c"],
            text=[f"{v:+.1f}%" for v in import_changes_t4.values()], textposition="outside",
        ))
        fig_imp.update_layout(**PLOTLY_THEME, height=220,
            title="Projected Import Drop by Sector",
            xaxis_title="% Change")
        fig_imp.update_xaxes(range=[-100, 10])
        st.plotly_chart(fig_imp, use_container_width=True)
    with c_imp2:
        st.markdown('<div class="insight-box" style="margin-top:20px"><b>Manufacturing −80.8%</b>: The largest trade shock in the model. High import penetration (32%) combined with a 27% average tariff drives a near-collapse of import demand.<br><br><b>Pharma −38.2%</b>: Severe, but smaller because US domestic pharma partially substitutes for imports and prices are less elastic.</div>', unsafe_allow_html=True)

    # ── REALITY CHECK: what actually happened after Liberation Day ──────────
    st.markdown('<div class="section-header">⚡ Reality check: what ACTUALLY happened after Liberation Day</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">The section above is a <b>model prediction</b>. This section is <b>measured reality</b> — official USITC customs data (monthly imports & duties collected, through Dec 2025) and BEA quarterly output (through Q1 2026). The model predicted an 80.8% import collapse; actual manufacturing imports were roughly <b>flat</b>. The tariffs were real — duty collections quintupled — but trade proved far more resilient than the linearised model assumed.</div>', unsafe_allow_html=True)

    _imp_rl, _dut_rl, _bea_rl = load_mfg_reality()
    _CH_NAMES_RL = {39: "Plastics", 72: "Iron & Steel", 73: "Steel Articles",
                    84: "Machinery", 85: "Electronics", 87: "Vehicles",
                    94: "Furniture", 95: "Toys & Sports"}

    # Pre/post comparison: Apr-Dec 2025 vs Apr-Dec 2024 (same months = seasonality-controlled)
    _pre_rl  = _imp_rl[(_imp_rl["date"] >= "2024-04-01") & (_imp_rl["date"] <= "2024-12-01")]
    _post_rl = _imp_rl[(_imp_rl["date"] >= "2025-04-01") & (_imp_rl["date"] <= "2025-12-01")]
    _tot_chg_rl = (_post_rl["value"].sum() / _pre_rl["value"].sum() - 1) * 100

    # Effective tariff rate per month
    _mv_rl = _imp_rl.groupby("date")["value"].sum().rename("value").to_frame()
    _mv_rl["duty"] = _dut_rl.groupby("date")["value"].sum()
    _mv_rl["rate"] = _mv_rl["duty"] / _mv_rl["value"] * 100
    _rate_2024_rl = float(_mv_rl.loc[(_mv_rl.index >= "2024-01-01") & (_mv_rl.index <= "2024-12-01"), "rate"].mean())
    _rate_peak_rl = float(_mv_rl.loc[_mv_rl.index >= "2025-04-01", "rate"].max())

    # BEA manufacturing output growth 2025Q1 -> 2026Q1
    _bea_mfg_rl = _bea_rl[_bea_rl["industry"] == "Manufacturing"]
    _bea_chg_rl = float((_bea_mfg_rl["2026Q1"].iloc[0] / _bea_mfg_rl["2025Q1"].iloc[0] - 1) * 100) if not _bea_mfg_rl.empty else 0

    _rc1, _rc2, _rc3, _rc4 = st.columns(4)
    with _rc1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Model predicted imports</div>
          <div class="kpi-value negative" style="font-size:26px">−80.8%</div>
          <div class="kpi-sub">GE model estimate</div>
        </div>""", unsafe_allow_html=True)
    with _rc2:
        _tc_cls = "positive" if _tot_chg_rl > 0 else "negative"
        st.markdown(f"""<div class="kpi-card" style="border:1px solid #22d3a0">
          <div class="kpi-label">Actual imports (Apr–Dec 2025 vs 2024)</div>
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
          <div class="kpi-label">US factory output since tariffs</div>
          <div class="kpi-value {_bc_cls}" style="font-size:26px">{_bea_chg_rl:+.1f}%</div>
          <div class="kpi-sub">BEA nominal output, 2025Q1 → 2026Q1</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    _rr1, _rr2 = st.columns(2)
    with _rr1:
        # Monthly imports by chapter
        fig_imp_rl = go.Figure()
        _colors_rl = ["#2563eb","#f87171","#fb923c","#22d3a0","#a78bfa","#fbbf24","#38bdf8","#f472b6"]
        for _i_rl, _ch_rl in enumerate(sorted(_imp_rl["chapter"].unique())):
            _sub_rl = _imp_rl[_imp_rl["chapter"] == _ch_rl].sort_values("date")
            fig_imp_rl.add_trace(go.Scatter(
                x=_sub_rl["date"], y=_sub_rl["value"] / 1e9,
                name=_CH_NAMES_RL.get(_ch_rl, str(_ch_rl)),
                line=dict(color=_colors_rl[_i_rl % len(_colors_rl)], width=1.8),
            ))
        fig_imp_rl.add_vline(x="2025-04-02", line_dash="dash", line_color="#f87171",
            annotation_text="Liberation Day", annotation_position="top left",
            annotation_font_color="#f87171")
        fig_imp_rl.update_layout(**PLOTLY_THEME, height=380,
            title="Actual Monthly US Manufacturing Imports by Product (2022–2025)",
            legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250", font=dict(size=10)))
        fig_imp_rl.update_yaxes(title_text="Imports ($B/month)")
        st.plotly_chart(fig_imp_rl, use_container_width=True)
    with _rr2:
        # Effective tariff rate line
        _mv_plot_rl = _mv_rl.reset_index().sort_values("date")
        fig_rate_rl = go.Figure(go.Scatter(
            x=_mv_plot_rl["date"], y=_mv_plot_rl["rate"],
            line=dict(color="#f87171", width=2.5),
            fill="tozeroy", fillcolor="rgba(248,113,113,0.12)",
        ))
        fig_rate_rl.add_vline(x="2025-04-02", line_dash="dash", line_color="#e2e8f0",
            annotation_text="Liberation Day", annotation_position="top left",
            annotation_font_color="#e2e8f0")
        fig_rate_rl.update_layout(**PLOTLY_THEME, height=380,
            title="Effective Tariff Rate Actually Paid (duties ÷ import value)")
        fig_rate_rl.update_yaxes(title_text="Effective Rate (%)")
        st.plotly_chart(fig_rate_rl, use_container_width=True)

    _rr3, _rr4 = st.columns(2)
    with _rr3:
        # Per-chapter change diverging bar
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
        fig_chg_rl.update_layout(**PLOTLY_THEME, height=360,
            title="Actual Import Change by Product (Apr–Dec 2025 vs 2024)")
        fig_chg_rl.update_xaxes(title_text="% Change", range=[-40, 40])
        st.plotly_chart(fig_chg_rl, use_container_width=True)
    with _rr4:
        # BEA output indexed to 2025Q1
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
        # Categorical x-axis: quarters map to positions 0..8; Liberation Day (Apr 2, 2025)
        # falls at the start of 2025Q2 (index 5), so draw the line at 4.5
        fig_bea_rl.add_vline(x=4.5, line_dash="dash", line_color="#f87171",
            annotation_text="Liberation Day", annotation_position="top left",
            annotation_font_color="#f87171")
        fig_bea_rl.update_layout(**PLOTLY_THEME, height=360,
            title="US Factory Output After Tariffs (BEA quarterly, 2025Q1 = 100)",
            legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250", font=dict(size=10)))
        fig_bea_rl.update_yaxes(title_text="Index (2025Q1 = 100)")
        st.plotly_chart(fig_bea_rl, use_container_width=True)

    st.markdown('<div class="insight-box"><b>Why was the model so wrong on imports?</b> Three reasons: (1) the −80.8% is a <i>linearised</i> elasticity estimate that breaks down for large tariff shocks; (2) exemptions and USMCA carve-outs meant the effective rate peaked near 13%, not the 27% headline; (3) demand surged in exactly the categories America can\'t substitute — machinery imports <b>rose 26%</b> (data-center and AI capex boom) even as tariffed steel (−24%), vehicles (−18%) and toys (−21%) fell sharply. The tariffs were real — monthly duty collections roughly <b>5×</b> — but aggregate trade rerouted rather than collapsed. Note: BEA output is nominal, so part of the output "growth" is tariff-driven price increases.</div>', unsafe_allow_html=True)

    # ── What does US manufacturing actually make? ──────────────────────────
    st.markdown('<div class="section-header">What does US manufacturing actually make?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">235 manufacturing industries (NAICS 31–33) rolled up into 21 subsectors. Transportation equipment, food processing and chemicals dominate — together over 40% of all US factory output.</div>', unsafe_allow_html=True)

    _SUB3_NAMES = {
        "336": "Cars, planes & transport equipment", "311": "Food processing",
        "325": "Chemicals & pharma",                 "324": "Petroleum & coal products",
        "333": "Industrial machinery",               "334": "Computers & electronics",
        "332": "Fabricated metal products",          "331": "Primary metals (steel, aluminum)",
        "326": "Plastics & rubber",                  "322": "Paper",
        "312": "Beverages & tobacco",                "339": "Misc (medical devices, toys…)",
        "321": "Wood products",                      "327": "Cement, glass & ceramics",
        "335": "Electrical equipment",               "337": "Furniture",
        "323": "Printing",                           "313": "Textile mills",
        "314": "Textile products",                   "315": "Apparel",
        "316": "Leather & footwear",
    }
    _naics_all = naics.copy()
    _naics_all["NAICS Code"] = _naics_all["NAICS Code"].astype(str)
    _mfg_rows = _naics_all[_naics_all["NAICS Code"].str.startswith(("31", "32", "33"))].copy()
    _mfg_rows["go_2021"] = pd.to_numeric(_mfg_rows["2021"], errors="coerce")
    _mfg_rows = _mfg_rows.dropna(subset=["go_2021"])
    _mfg_rows["sub3"] = _mfg_rows["NAICS Code"].str[:3]

    _mfg_total_2021 = _mfg_rows["go_2021"].sum()
    _econ_total_2021 = pd.to_numeric(_naics_all["2021"], errors="coerce").sum()
    _sub_totals = _mfg_rows.groupby("sub3")["go_2021"].sum().sort_values(ascending=False)
    _largest_sub_code = _sub_totals.index[0]

    c_sub_k1, c_sub_k2, c_sub_k3 = st.columns(3)
    with c_sub_k1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Total US manufacturing output (2021)</div>
          <div class="kpi-value neutral" style="font-size:26px">${_mfg_total_2021/1e6:.2f}T</div>
          <div class="kpi-sub">{len(_mfg_rows)} industries, NAICS 31–33</div>
        </div>""", unsafe_allow_html=True)
    with c_sub_k2:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Largest subsector</div>
          <div class="kpi-value positive" style="font-size:22px">{_SUB3_NAMES.get(_largest_sub_code, _largest_sub_code)}</div>
          <div class="kpi-sub">${_sub_totals.iloc[0]/1e6:.2f}T in 2021</div>
        </div>""", unsafe_allow_html=True)
    with c_sub_k3:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Manufacturing share of US economy</div>
          <div class="kpi-value warning" style="font-size:26px">{_mfg_total_2021/_econ_total_2021*100:.1f}%</div>
          <div class="kpi-sub">of total gross output, all industries</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    _sub_plot = _sub_totals.reset_index()
    _sub_plot.columns = ["sub3", "go"]
    _sub_plot["label"] = _sub_plot["sub3"].map(_SUB3_NAMES).fillna(_sub_plot["sub3"])
    _sub_plot["go_bn"] = _sub_plot["go"] / 1e3
    fig_sub = go.Figure(go.Bar(
        x=_sub_plot["go_bn"], y=_sub_plot["label"],
        orientation="h",
        marker_color=["#2563eb" if i > 2 else "#22d3a0" for i in range(len(_sub_plot))],
        text=[f"${v:,.0f}B" for v in _sub_plot["go_bn"]], textposition="outside",
    ))
    fig_sub.update_layout(**PLOTLY_THEME, height=560,
        title="US Manufacturing Output by Subsector, 2021 (green = top 3)",
        xaxis_title="Gross Output ($B)")
    fig_sub.update_yaxes(autorange="reversed", tickfont=dict(size=11))
    st.plotly_chart(fig_sub, use_container_width=True)

    # ── Which industries face the biggest tariff shock? ────────────────────
    st.markdown('<div class="section-header">Which industries face the biggest tariff shock?</div>', unsafe_allow_html=True)

    SECTOR_NAMES_T4 = {
        "steel_aluminum":      "Steel & Aluminum",
        "pharma":              "Medicines (Pharma)",
        "retail_consumer":     "Consumer Goods (Retail)",
        "manufacturing_other": "Other Manufacturing",
        "services_other":      "Services",
        "energy_primary":      "Energy & Primary",
    }
    shock_scenario_t4 = st.selectbox(
        "Scenario",
        shocks["scenario"].unique().tolist(),
        format_func=lambda s: {
            "baseline_no_tariffs":      "No Tariffs (baseline)",
            "liberation_day_schedule":  "Liberation Day Schedule",
            "optimal_uniform_19":       "Optimal Uniform 19%",
            "industry_focused":         "Industry-Focused",
            "supply_chain_disruption":  "Supply Chain Disruption",
        }.get(s, s),
        key="mfg_shock_scenario_t4",
    )
    shock_plot = shocks[shocks["scenario"] == shock_scenario_t4].copy()
    shock_plot["sector_label"] = shock_plot["model_sector"].map(SECTOR_NAMES_T4).fillna(shock_plot["model_sector"])
    shock_plot["tariff_pct"]   = shock_plot["tariff_rate"] * 100

    fig_shock = go.Figure(go.Bar(
        x=shock_plot["tariff_pct"], y=shock_plot["sector_label"],
        orientation="h",
        marker_color=["#f87171" if v > 10 else "#fbbf24" if v > 3 else "#22d3a0" for v in shock_plot["tariff_pct"]],
        text=[f"{v:.1f}%" for v in shock_plot["tariff_pct"]], textposition="outside",
    ))
    fig_shock.update_layout(**PLOTLY_THEME, height=320,
        title="Effective Tariff Rate by Industry",
        xaxis_title="Effective Tariff Rate %")
    fig_shock.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_shock, use_container_width=True)

    # ── All scenarios at once: tariff heatmap ───────────────────────────────
    st.markdown('<div class="section-header">Every scenario at a glance — which policy hits which sector?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Each cell is the effective tariff rate a sector faces under a policy scenario. The Optimal Uniform 19% scenario hits everything equally; the Liberation Day schedule concentrates on consumer goods.</div>', unsafe_allow_html=True)

    _SCEN_NAMES_HM = {
        "baseline_no_tariffs":     "No Tariffs (baseline)",
        "liberation_day_schedule": "Liberation Day Schedule",
        "optimal_uniform_19":      "Optimal Uniform 19%",
        "industry_focused":        "Industry-Focused",
        "supply_chain_disruption": "Supply Chain Disruption",
    }
    _hm = shocks.copy()
    _hm["scen_label"] = _hm["scenario"].map(_SCEN_NAMES_HM).fillna(_hm["scenario"])
    _hm["sect_label"] = _hm["model_sector"].map(SECTOR_NAMES_T4).fillna(_hm["model_sector"])
    _hm["pct"] = _hm["tariff_rate"] * 100
    _hm_pivot = _hm.pivot_table(index="scen_label", columns="sect_label", values="pct", aggfunc="first")
    _scen_order = [v for v in _SCEN_NAMES_HM.values() if v in _hm_pivot.index]
    _hm_pivot = _hm_pivot.reindex(_scen_order)

    fig_hm = go.Figure(go.Heatmap(
        z=_hm_pivot.values,
        x=_hm_pivot.columns.tolist(),
        y=_hm_pivot.index.tolist(),
        colorscale=[[0, "#1a1d2e"], [0.3, "#7c2d12"], [1, "#f87171"]],
        text=[[f"{v:.1f}%" for v in row] for row in _hm_pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=12, color="#e2e8f0"),
        colorbar=dict(title="Tariff %", tickfont=dict(color="#94a3b8"), bgcolor="#1a1d2e"),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
    ))
    fig_hm.update_layout(**PLOTLY_THEME, height=340,
        title="Effective Tariff Rate: 5 Scenarios × 6 Sectors")
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── Before vs after: the tariff jump ────────────────────────────────────
    st.markdown('<div class="section-header">Before vs after: how big was the tariff jump?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Factory import costs jumped roughly <b>7×</b> overnight. Before Liberation Day, US manufacturers paid ~3.6% on imported inputs; the effective rate is now ~27%. Pharma jumped 8× — from 2.4% to 19.9%.</div>', unsafe_allow_html=True)

    _risk_pj, _tsup_pj, _htsx_pj, _ph_stats_pj = load_pharma_risk()
    _jump_sectors = ["Manufacturing", "Pharma"]
    _jump_pre  = [mfg_stats["hts8_mfg_rate"] * 100,  _ph_stats_pj["hts8_pharma_rate"] * 100]
    _jump_post = [mfg_stats["tau_mfg_avg"] * 100,    _ph_stats_pj["tau_pharma_eff"] * 100]

    c_pj1, c_pj2 = st.columns([3, 2])
    with c_pj1:
        fig_jump = go.Figure()
        fig_jump.add_trace(go.Bar(
            name="Before Liberation Day", x=_jump_sectors, y=_jump_pre,
            marker_color="#2563eb",
            text=[f"{v:.1f}%" for v in _jump_pre], textposition="outside",
        ))
        fig_jump.add_trace(go.Bar(
            name="After Liberation Day", x=_jump_sectors, y=_jump_post,
            marker_color="#f87171",
            text=[f"{v:.1f}%" for v in _jump_post], textposition="outside",
        ))
        fig_jump.update_layout(**PLOTLY_THEME, height=320,
            title="Effective Import Tariff Rate: Pre vs Post Liberation Day",
            barmode="group", yaxis_title="Tariff Rate (%)",
            legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250"))
        st.plotly_chart(fig_jump, use_container_width=True)
    with c_pj2:
        _mfg_mult = _jump_post[0] / max(_jump_pre[0], 0.01)
        _ph_mult  = _jump_post[1] / max(_jump_pre[1], 0.01)
        st.markdown(f"""
        <div class="kpi-card" style="margin-top:24px">
          <div class="kpi-label">Manufacturing tariff multiplied by</div>
          <div class="kpi-value negative" style="font-size:30px">{_mfg_mult:.1f}×</div>
          <div class="kpi-sub">{_jump_pre[0]:.1f}% → {_jump_post[0]:.1f}%</div>
        </div>
        <div class="kpi-card" style="margin-top:12px">
          <div class="kpi-label">Pharma tariff multiplied by</div>
          <div class="kpi-value negative" style="font-size:30px">{_ph_mult:.1f}×</div>
          <div class="kpi-sub">{_jump_pre[1]:.1f}% → {_jump_post[1]:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    # ── Why do tariffs raise prices more than expected? ────────────────────
    st.markdown('<div class="section-header">Why do tariffs raise prices more than you\'d expect?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">The supply chain multiplier (1.09×) captures how tariff costs on imported inputs ripple through manufacturing. A factory pays more for imported steel directly — but also pays more for components that use steel, compounding the total impact.</div>', unsafe_allow_html=True)

    c_io1, c_io2 = st.columns([2, 1])
    with c_io1:
        io_stages = [
            "Direct tariff rate\n(avg 27.0%)",
            f"After supply chain effects\n(×{mfg_stats['io_mult_mfg']:.2f})",
            f"Manufacturing share\nof total CPI ({mfg_stats['cpi_mfg_contribution']:.1f}pp of 7.1pp)",
        ]
        io_vals = [
            mfg_stats["tau_mfg_avg"] * 100,
            mfg_stats["tau_mfg_avg"] * 100 * mfg_stats["io_mult_mfg"],
            mfg_stats["cpi_mfg_contribution"],
        ]
        fig_io = go.Figure(go.Bar(
            x=io_stages, y=io_vals,
            marker_color=["#2563eb","#fbbf24","#f87171"],
            text=[f"{v:.1f}%" for v in io_vals], textposition="outside",
        ))
        fig_io.update_layout(**PLOTLY_THEME, height=300,
            title="How Supply Chain Amplification Works",
            yaxis_title="% Impact")
        st.plotly_chart(fig_io, use_container_width=True)
    with c_io2:
        cpi_bd = pd.DataFrame({
            "Sector": ["Manufacturing", "Retail", "Pharma", "Other"],
            "CPI Contribution (pp)": [
                round(mfg_stats["cpi_mfg_contribution"], 2),
                0.30,
                0.05,
                round(7.09 - mfg_stats["cpi_mfg_contribution"] - 0.30 - 0.05, 2),
            ],
        })
        st.markdown("<br><b>CPI breakdown by sector</b>", unsafe_allow_html=True)
        st.dataframe(cpi_bd, use_container_width=True, hide_index=True)

    # ── US Manufacturing Output by Industry (NAICS 31–33 only) ─────────────
    st.markdown('<div class="section-header">Top manufacturing industries by output</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Only true manufacturing industries (NAICS 31–33) are shown — 235 industries totalling $6.3T. Petroleum refineries, light trucks and pharmaceutical preparations lead the pack.</div>', unsafe_allow_html=True)

    year_sel = st.select_slider("Year", options=[2016,2017,2018,2019,2020,2021], value=2021)
    naics_mfg_only = naics.copy()
    naics_mfg_only["NAICS Code"] = naics_mfg_only["NAICS Code"].astype(str)
    naics_mfg_only = naics_mfg_only[naics_mfg_only["NAICS Code"].str.startswith(("31","32","33"))]
    naics_plot = naics_mfg_only[["NAICS Code","Name", str(year_sel)]].copy()
    naics_plot.columns = ["naics","name","go_value"]
    naics_plot["go_value"] = pd.to_numeric(naics_plot["go_value"], errors="coerce")
    naics_plot = naics_plot.dropna(subset=["go_value"])

    c1, c2 = st.columns(2)
    with c1:
        top20 = naics_plot.nlargest(20, "go_value")
        fig_naics = go.Figure(go.Bar(
            x=top20["go_value"] / 1e3,
            y=top20["name"].str[:40],
            orientation="h", marker_color="#2563eb",
            text=[f"${v/1e3:,.0f}B" for v in top20["go_value"]],
            textposition="outside",
        ))
        fig_naics.update_layout(**PLOTLY_THEME, height=500,
            title=f"Top 20 Manufacturing Industries ({year_sel}, NAICS 31–33)",
            xaxis_title="Gross Output ($B)")
        fig_naics.update_yaxes(autorange="reversed", tickfont=dict(size=10))
        st.plotly_chart(fig_naics, use_container_width=True)

    with c2:
        top8_names = naics_plot.nlargest(8, "go_value")["name"].tolist()
        years = [2016, 2017, 2018, 2019, 2020, 2021]
        fig_ts_n = go.Figure()
        colors_n = ["#2563eb","#22d3a0","#fbbf24","#f87171","#a78bfa","#fb923c","#38bdf8","#4ade80"]
        for i, name in enumerate(top8_names):
            row = naics_mfg_only[naics_mfg_only["Name"] == name]
            if row.empty: continue
            vals = [pd.to_numeric(row[str(y)].iloc[0], errors="coerce") / 1e3 for y in years]
            fig_ts_n.add_trace(go.Scatter(
                x=years, y=vals, name=name[:25],
                line=dict(color=colors_n[i % len(colors_n)], width=2),
                mode="lines+markers",
            ))
        fig_ts_n.update_layout(**PLOTLY_THEME, height=500,
            title="Output Trend — Top 8 Manufacturing Industries (2016–2021)",
            yaxis_title="Gross Output ($B)", xaxis_title="Year",
            legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250", font=dict(size=10)))
        st.plotly_chart(fig_ts_n, use_container_width=True)

    # ── Which manufacturing industries are growing or shrinking? ────────────
    st.markdown('<div class="section-header">Which manufacturing industries are growing — and which are dying?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Output growth 2016 → 2021, for industries with at least $10B output in 2021. Tariffs land very differently on a booming industry than on one already in decline.</div>', unsafe_allow_html=True)

    _gr = naics_mfg_only[["NAICS Code","Name","2016","2021"]].copy()
    _gr["go16"] = pd.to_numeric(_gr["2016"], errors="coerce")
    _gr["go21"] = pd.to_numeric(_gr["2021"], errors="coerce")
    _gr = _gr.dropna(subset=["go16","go21"])
    _gr = _gr[(_gr["go21"] >= 10000) & (_gr["go16"] > 0)]
    _gr["growth_pct"] = (_gr["go21"] / _gr["go16"] - 1) * 100

    _gr_top = _gr.nlargest(10, "growth_pct")
    _gr_bot = _gr.nsmallest(10, "growth_pct").sort_values("growth_pct", ascending=False)
    _gr_plot = pd.concat([_gr_top, _gr_bot])

    fig_gr = go.Figure(go.Bar(
        x=_gr_plot["growth_pct"], y=_gr_plot["Name"].str[:40],
        orientation="h",
        marker_color=["#22d3a0" if v >= 0 else "#f87171" for v in _gr_plot["growth_pct"]],
        text=[f"{v:+.0f}%" for v in _gr_plot["growth_pct"]], textposition="outside",
    ))
    fig_gr.add_vline(x=0, line_color="#4b5563", line_width=1)
    fig_gr.update_layout(**PLOTLY_THEME, height=560,
        title="Fastest-Growing vs Fastest-Shrinking Manufacturing Industries (2016–2021, min $10B)",
        xaxis_title="Output Growth %")
    fig_gr.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    st.plotly_chart(fig_gr, use_container_width=True)

    # ── Which products have the highest tariffs? ───────────────────────────
    st.markdown('<div class="section-header">Which products have the highest tariffs?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">The HTS-8 schedule covers 13,100 product lines. Most face 0–25% MFN rates. Specific-rate tariffs (tobacco, sugar, dairy) are excluded as their rate equivalents are not meaningful percentages.</div>', unsafe_allow_html=True)

    hts_search_t4 = st.text_input("Search product description", "", placeholder="e.g. steel, motor, apparel…")
    hts_valid = hts[hts["mfn_rate"] < 2.0].copy()
    hts_valid["mfn_pct"] = hts_valid["mfn_rate"] * 100

    desc_cols = [c for c in hts_valid.columns if any(kw in c.lower() for kw in ["description","product","brief"])]
    if hts_search_t4 and desc_cols:
        hts_filtered = hts_valid[hts_valid[desc_cols[0]].astype(str).str.contains(hts_search_t4, case=False, na=False)]
    else:
        hts_filtered = hts_valid

    c_h1, c_h2 = st.columns([2, 1])
    with c_h1:
        fig_hts_hist = px.histogram(
            hts_filtered, x="mfn_pct", nbins=50,
            color_discrete_sequence=["#2563eb"],
            labels={"mfn_pct": "MFN Tariff Rate (%)"},
        )
        fig_hts_hist.update_layout(**PLOTLY_THEME, height=280,
            title=f"Distribution of Tariff Rates ({len(hts_filtered):,} product lines)",
            xaxis_title="MFN Tariff Rate %", yaxis_title="# Product Lines")
        st.plotly_chart(fig_hts_hist, use_container_width=True)
    with c_h2:
        if desc_cols:
            hts_col = [c for c in hts_filtered.columns if "hts" in c.lower()]
            top_hts = hts_filtered.nlargest(12, "mfn_pct")[[hts_col[0] if hts_col else hts_filtered.columns[0], desc_cols[0], "mfn_pct"]].copy()
            top_hts.columns = ["HTS Code","Product","Rate %"]
            top_hts["Rate %"] = top_hts["Rate %"].round(1)
            st.dataframe(top_hts.reset_index(drop=True), use_container_width=True, hide_index=True)

    # ── Which products were protected BEFORE Liberation Day? ────────────────
    st.markdown('<div class="section-header">Which products were already protected BEFORE Liberation Day?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Pre-existing US tariffs (MFN rates) were concentrated in labor-intensive goods: footwear ~15%, apparel ~11%, textiles ~10%. Machinery, electronics and pharma were nearly duty-free — exactly the categories Liberation Day hit hardest.</div>', unsafe_allow_html=True)

    _CH_NAMES = {
        "64": "Footwear", "61": "Knitted apparel", "62": "Woven apparel",
        "55": "Man-made staple fibres", "54": "Man-made filaments", "60": "Knitted fabrics",
        "19": "Cereal & bakery products", "42": "Leather goods & bags", "52": "Cotton",
        "12": "Oilseeds", "58": "Special woven fabrics", "69": "Ceramics",
        "63": "Made-up textiles", "21": "Misc edible preparations", "04": "Dairy, eggs & honey",
        "17": "Sugars", "20": "Fruit & vegetable preparations", "16": "Meat & fish preparations",
        "96": "Misc manufactured articles", "91": "Clocks & watches", "24": "Tobacco",
        "87": "Vehicles", "84": "Machinery", "85": "Electronics", "30": "Pharmaceuticals",
        "72": "Iron & steel", "73": "Steel articles", "39": "Plastics", "29": "Organic chemicals",
    }
    _hts_ch = hts_valid.copy()
    _hts_ch["chapter"] = _hts_ch["hts8"].astype(str).str.zfill(8).str[:2]
    _ch_agg = _hts_ch.groupby("chapter").agg(
        n_lines=("mfn_pct", "size"), avg_rate=("mfn_pct", "mean")).reset_index()
    _ch_agg = _ch_agg[_ch_agg["n_lines"] >= 30]
    _ch_top = _ch_agg.nlargest(15, "avg_rate").copy()
    _ch_top["label"] = _ch_top["chapter"].map(_CH_NAMES).fillna("Chapter " + _ch_top["chapter"])

    fig_ch = go.Figure(go.Bar(
        x=_ch_top["avg_rate"], y=_ch_top["label"],
        orientation="h",
        marker_color=["#f87171" if v > 10 else "#fb923c" if v > 7 else "#fbbf24" for v in _ch_top["avg_rate"]],
        text=[f"{v:.1f}%  ({n:,} products)" for v, n in zip(_ch_top["avg_rate"], _ch_top["n_lines"])],
        textposition="outside",
    ))
    fig_ch.update_layout(**PLOTLY_THEME, height=460,
        title="Top 15 Product Categories by Pre-Liberation Day Tariff Rate (avg MFN)",
        xaxis_title="Average MFN Tariff Rate (%)")
    fig_ch.update_xaxes(range=[0, 20])
    fig_ch.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_ch, use_container_width=True)

    # ── Who gets duty-free access? (FTA analysis) ───────────────────────────
    st.markdown('<div class="section-header">Who gets duty-free access to the US market?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Share of all 13,100 product lines that enter the US at a 0% rate under each trade regime. Free-trade agreement partners like Mexico, Canada (USMCA), Korea and Australia negotiated near-universal duty-free access — <b>Liberation Day tariffs stack on top of these rates, effectively nullifying decades of trade agreements</b>.</div>', unsafe_allow_html=True)

    _fta_regimes = [
        ("MFN (everyone)",        "mfn_ad_val_rate"),
        ("USMCA (Mexico/Canada)", "usmca_ad_val_rate"),
        ("Korea FTA",             "korea_ad_val_rate"),
        ("Australia FTA",         "australia_ad_val_rate"),
        ("Chile FTA",             "chile_ad_val_rate"),
        ("Singapore FTA",         "singapore_ad_val_rate"),
        ("Japan Agreement",       "japan_ad_val_rate"),
    ]
    _fta_rows = []
    _n_total_hts = len(hts)
    _mfn_free_mask = pd.to_numeric(hts["mfn_ad_val_rate"], errors="coerce").fillna(1) == 0
    for _label, _col in _fta_regimes:
        if _col not in hts.columns:
            continue
        if _label.startswith("MFN"):
            _free = int(_mfn_free_mask.sum())
        else:
            # Duty-free for an FTA partner = already MFN-free OR explicit 0% preference
            _fta_zero = pd.to_numeric(hts[_col], errors="coerce") == 0
            _free = int((_mfn_free_mask | _fta_zero).sum())
        _fta_rows.append({"regime": _label, "free_pct": _free / _n_total_hts * 100, "n_free": _free})
    _fta_df = pd.DataFrame(_fta_rows).sort_values("free_pct", ascending=False)

    fig_fta = go.Figure(go.Bar(
        x=_fta_df["free_pct"], y=_fta_df["regime"],
        orientation="h",
        marker_color=["#2563eb" if r.startswith("MFN") else "#22d3a0" for r in _fta_df["regime"]],
        text=[f"{v:.0f}%  ({n:,} lines)" for v, n in zip(_fta_df["free_pct"], _fta_df["n_free"])],
        textposition="outside",
    ))
    fig_fta.update_layout(**PLOTLY_THEME, height=340,
        title="Share of Product Lines Entering Duty-Free, by Trade Regime (pre-Liberation Day)",
        xaxis_title="% of 13,100 HTS Lines at 0% Tariff")
    fig_fta.update_xaxes(range=[0, 100])
    fig_fta.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_fta, use_container_width=True)

    # ── Producer Price Index ───────────────────────────────────────────────
    st.markdown('<div class="section-header">How have factory prices changed over the last decade?</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Tech goods got dramatically <b>cheaper</b> for a decade — computer producer prices fell 22% and semiconductors fell 8% between 2012 and 2021, while furniture, machinery and plastics rose ~22%. Tariffs reverse the cheap-tech trend: import taxes now push electronics prices up instead.</div>', unsafe_allow_html=True)

    _PPI_NAMES = {
        3152: "Apparel",            3344: "Semiconductors",
        3341: "Computers",          3371: "Furniture",
        3363: "Motor vehicle parts",3359: "Electrical equipment",
        3399: "Misc manufacturing", 3343: "Audio & video equipment",
        3339: "Industrial machinery",3261: "Plastics products",
    }
    year_cols = [c for c in price_idx.columns if "Annual" in str(c)]
    colors_ppi = ["#2563eb","#22d3a0","#fbbf24","#f87171","#a78bfa","#fb923c","#38bdf8","#4ade80","#f472b6","#a3e635"]

    c_ppi1, c_ppi2 = st.columns([3, 2])
    with c_ppi1:
        fig_ppi = go.Figure()
        for i, (_, row) in enumerate(price_idx.iterrows()):
            vals = pd.to_numeric(row[year_cols], errors="coerce")
            yr_labels = [str(y).replace("Annual ","") for y in year_cols]
            _ppi_name = _PPI_NAMES.get(int(row["NAICS4"]), f"NAICS {int(row['NAICS4'])}")
            fig_ppi.add_trace(go.Scatter(
                x=yr_labels, y=vals.values,
                name=_ppi_name,
                line=dict(color=colors_ppi[i % len(colors_ppi)], width=1.8),
                mode="lines+markers",
            ))
        fig_ppi.update_layout(**PLOTLY_THEME, height=420,
            title="Producer Price Index by Industry, 2012–2022",
            yaxis_title="PPI Index", xaxis_title="Year",
            legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250", font=dict(size=10)))
        st.plotly_chart(fig_ppi, use_container_width=True)

    with c_ppi2:
        _dec_rows = []
        for _, row in price_idx.iterrows():
            _v12 = pd.to_numeric(row["Annual 2012"], errors="coerce")
            _v21 = pd.to_numeric(row["Annual 2021"], errors="coerce")
            if pd.notna(_v12) and pd.notna(_v21) and _v12 > 0:
                _dec_rows.append({
                    "name": _PPI_NAMES.get(int(row["NAICS4"]), f"NAICS {int(row['NAICS4'])}"),
                    "chg": (_v21 / _v12 - 1) * 100,
                })
        _dec_df = pd.DataFrame(_dec_rows).sort_values("chg")
        fig_dec = go.Figure(go.Bar(
            x=_dec_df["chg"], y=_dec_df["name"],
            orientation="h",
            marker_color=["#22d3a0" if v < 0 else "#f87171" for v in _dec_df["chg"]],
            text=[f"{v:+.0f}%" for v in _dec_df["chg"]], textposition="outside",
        ))
        fig_dec.add_vline(x=0, line_color="#4b5563", line_width=1)
        fig_dec.update_layout(**PLOTLY_THEME, height=420,
            title="Total Price Change 2012 → 2021 (green = got cheaper)",
            xaxis_title="% Change")
        fig_dec.update_xaxes(range=[-35, 35])
        st.plotly_chart(fig_dec, use_container_width=True)


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
# TAB 7 — BUILD YOUR OWN SCENARIO
# Users pick countries, set tariff rates, and get a full policy analysis:
# welfare, consumer prices, $/household, tariff revenue, sector risks, and a
# ranking against the paper's real GE scenarios. Engine: run_tariff_scenario
# (linearised PE approximation, eps=4) via _compute_custom_scenario.
# ═════════════════════════════════════════════════════════════════════════════
with tab7:
    _res_b7, _Y_b7, _idus_b7, _cl_b7, _res15_b7, _ = load_baseline()
    _tdf_b7 = load_tariffs()
    _imp_b7, _exp_b7, _, _ = load_bilateral()
    _, _retail_b7 = load_retail()
    _, _, _, _pharma_exp_b7 = load_pharma_outputs()

    _PASSTHROUGH_B7 = float(_retail_b7.get("retail_product_passthrough", 0.297))
    _EPS_B7 = 4.0
    _N_HOUSEHOLDS = 132_000_000  # US households (Census, ~2023)

    st.markdown('<div class="section-header">🎛️ Build Your Own Tariff Scenario</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Pick any countries, set your own tariff rates, and see what happens — to American wallets, to government revenue, and to the countries you tax. Results update live. (Fast partial-equilibrium approximation — directionally correct, not a full GE re-solve; assumes no retaliation.)</div>', unsafe_allow_html=True)

    # LD reference rates by iso3 / name
    _ld_rate_by_iso = dict(zip(_tdf_b7["iso3"], _tdf_b7["tariff_pct"]))
    _iso_by_name_b7 = dict(zip(_cl_b7["CountryName"], _cl_b7["iso3"]))
    _idx_by_iso_b7  = {iso: i for i, iso in enumerate(_cl_b7["iso3"])}

    # ── Apply pending preset BEFORE sliders instantiate ─────────────────────
    _pending_b7 = st.session_state.pop("_bys_preset", None)
    if _pending_b7:
        for _k7, _v7 in _pending_b7.items():
            st.session_state[_k7] = _v7

    # ── Country picker ───────────────────────────────────────────────────────
    _default_b7 = [n for n in ["China", "Vietnam", "Mexico", "Canada", "Germany", "Japan"]
                   if n in _iso_by_name_b7]
    _sel_b7 = st.multiselect("Countries in your scenario", sorted(_cl_b7["CountryName"].tolist()),
                             default=_default_b7, key="bys_countries")

    # ── Rate sliders (3 columns) ─────────────────────────────────────────────
    _rates_b7 = {}   # iso3 -> user rate (%)
    _lds_b7   = {}   # iso3 -> LD rate (%)
    if _sel_b7:
        _slider_cols = st.columns(3)
        for _i7, _cname7 in enumerate(_sel_b7):
            _iso7 = _iso_by_name_b7[_cname7]
            _ld7  = int(round(_ld_rate_by_iso.get(_iso7, 10)))
            with _slider_cols[_i7 % 3]:
                _rates_b7[_iso7] = st.slider(f"{_cname7} (LD: {_ld7}%)", 0, 100, _ld7, 1,
                                             format="%d%%", key=f"bys_rate_{_iso7}")
            _lds_b7[_iso7] = _ld7

    # ── Preset buttons ───────────────────────────────────────────────────────
    _p1, _p2, _p3, _p4, _p5 = st.columns(5)
    def _mk_preset_b7(vals):
        st.session_state["_bys_preset"] = vals
        st.rerun()
    with _p1:
        if st.button("↺ Liberation Day", use_container_width=True, key="bys_p_ld"):
            _mk_preset_b7({f"bys_rate_{iso}": _lds_b7[iso] for iso in _rates_b7})
    with _p2:
        if st.button("🕊️ Free Trade (0%)", use_container_width=True, key="bys_p_free"):
            _mk_preset_b7({f"bys_rate_{iso}": 0 for iso in _rates_b7})
    with _p3:
        if st.button("🌍 Flat 10%", use_container_width=True, key="bys_p_10"):
            _mk_preset_b7({f"bys_rate_{iso}": 10 for iso in _rates_b7})
    with _p4:
        if st.button("⚖️ Uniform 25%", use_container_width=True, key="bys_p_25"):
            _mk_preset_b7({f"bys_rate_{iso}": 25 for iso in _rates_b7})
    with _p5:
        if st.button("🔥 Max Pressure (CHN 100%)", use_container_width=True, key="bys_p_max"):
            _vals = {f"bys_rate_{iso}": _lds_b7[iso] for iso in _rates_b7}
            if "CHN" in _rates_b7:
                _vals["bys_rate_CHN"] = 100
            _mk_preset_b7(_vals)

    if not _sel_b7:
        st.info("Select at least one country above to build a scenario.")
    else:
        # ── Run the PE model ────────────────────────────────────────────────
        _frozen_b7 = tuple(sorted((iso, r) for iso, r in _rates_b7.items()))
        _pe_b7 = _compute_custom_scenario(_frozen_b7)
        _pe_ctys_b7 = _pe_b7.get("data", {}).get("countries", [])
        _pe_us_b7 = next((r for r in _pe_ctys_b7 if r.get("iso3") == "USA"), {})
        _wd_us_b7 = float(_pe_us_b7.get("welfare_delta_pct") or 0)
        _bw_us_b7 = float(_pe_us_b7.get("baseline_welfare_pct") or 0)
        _nw_us_b7 = float(_pe_us_b7.get("new_welfare_pct") or 0)

        _changed_b7 = {iso: r for iso, r in _rates_b7.items() if r != _lds_b7[iso]}
        _n_changed_b7 = len(_changed_b7)

        # ── Derived analysis: prices, household cost, revenue ───────────────
        _tot_imp_b7 = float(_imp_b7.sum())
        _dcpi_b7 = 0.0          # consumer price change (%)
        _hh_cost_b7 = 0.0       # extra $ per household per year
        for _iso7, _r7 in _rates_b7.items():
            _ci7 = _idx_by_iso_b7.get(_iso7)
            if _ci7 is None:
                continue
            _dtau7 = (_r7 - _lds_b7[_iso7]) / 100.0
            _share7 = float(_imp_b7[_ci7]) / max(_tot_imp_b7, 1)
            _dcpi_b7 += _share7 * _dtau7 * _PASSTHROUGH_B7 * 100
            _hh_cost_b7 += float(_imp_b7[_ci7]) * 1000 * _dtau7 * _PASSTHROUGH_B7
        _hh_cost_b7 /= _N_HOUSEHOLDS

        # Elasticity-consistent revenue: M(tau) = M_2023 * ((1+tau)/(1+tau_LD))^(-eps)
        def _revenue_b7(rate_fn):
            _rev = 0.0
            for _isoX, _ldX in _ld_rate_by_iso.items():
                _ciX = _idx_by_iso_b7.get(_isoX)
                if _ciX is None or _ciX == _idus_b7:
                    continue
                _tau_new = rate_fn(_isoX, _ldX) / 100.0
                _tau_ld  = _ldX / 100.0
                _m_adj = float(_imp_b7[_ciX]) * ((1 + _tau_new) / (1 + _tau_ld)) ** (-_EPS_B7)
                _rev += _tau_new * _m_adj * 1000  # $ (imports are in $1000s)
            return _rev
        _rev_user_b7 = _revenue_b7(lambda iso, ld: _rates_b7.get(iso, ld))
        _rev_ld_b7   = _revenue_b7(lambda iso, ld: ld)
        _drev_bn_b7  = (_rev_user_b7 - _rev_ld_b7) / 1e9

        # World impact across changed countries
        _world_deltas_b7 = [float(r.get("welfare_delta_pct") or 0) for r in _pe_ctys_b7
                            if r.get("iso3") != "USA" and r.get("iso3") in _changed_b7]
        _avg_world_b7 = sum(_world_deltas_b7) / len(_world_deltas_b7) if _world_deltas_b7 else 0.0
        _n_win_b7  = sum(1 for v in _world_deltas_b7 if v > 0)
        _n_lose_b7 = sum(1 for v in _world_deltas_b7 if v < 0)

        # ── Scorecard ────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Your scenario scorecard</div>', unsafe_allow_html=True)
        if _n_changed_b7 == 0:
            st.markdown('<div class="insight-box">You\'re currently at exact Liberation Day rates — drag any slider or hit a preset to see your scenario\'s impact.</div>', unsafe_allow_html=True)

        _sc1, _sc2, _sc3, _sc4 = st.columns(4)
        with _sc1:
            _w_cls7 = "positive" if _wd_us_b7 > 0 else ("neutral" if _wd_us_b7 == 0 else "negative")
            _w_verdict7 = "America better off" if _wd_us_b7 > 0 else ("No change" if _wd_us_b7 == 0 else "America worse off")
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-label">US Wellbeing vs Liberation Day</div>
              <div class="kpi-value {_w_cls7}" style="font-size:26px">{_wd_us_b7:+.2f}pp</div>
              <div class="kpi-sub">{_w_verdict7}</div>
            </div>""", unsafe_allow_html=True)
        with _sc2:
            _p_cls7 = "negative" if _dcpi_b7 > 0 else ("neutral" if _dcpi_b7 == 0 else "positive")
            _p_verdict7 = "Prices rise" if _dcpi_b7 > 0 else ("No change" if _dcpi_b7 == 0 else "Prices fall")
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-label">Consumer Prices</div>
              <div class="kpi-value {_p_cls7}" style="font-size:26px">{_dcpi_b7:+.2f}%</div>
              <div class="kpi-sub">{_p_verdict7} · {_hh_cost_b7:+,.0f} $/household/yr</div>
            </div>""", unsafe_allow_html=True)
        with _sc3:
            _r_cls7 = "positive" if _drev_bn_b7 > 0 else ("neutral" if _drev_bn_b7 == 0 else "negative")
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-label">Tariff Revenue vs Liberation Day</div>
              <div class="kpi-value {_r_cls7}" style="font-size:26px">{_drev_bn_b7:+,.0f}B</div>
              <div class="kpi-sub">demand-adjusted (imports shrink when taxed)</div>
            </div>""", unsafe_allow_html=True)
        with _sc4:
            _wl_cls7 = "positive" if _avg_world_b7 > 0 else ("neutral" if _avg_world_b7 == 0 else "negative")
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-label">Impact on Countries You Changed</div>
              <div class="kpi-value {_wl_cls7}" style="font-size:26px">{_avg_world_b7:+.2f}pp</div>
              <div class="kpi-sub">{_n_win_b7} gain · {_n_lose_b7} lose</div>
            </div>""", unsafe_allow_html=True)

        # ── Sector risk callouts ────────────────────────────────────────────
        _pharma_hit_b7 = 0.0
        for _iso7, _r7 in _changed_b7.items():
            if _r7 > _lds_b7[_iso7] and "iso3" in _pharma_exp_b7.columns:
                _prow7 = _pharma_exp_b7[_pharma_exp_b7["iso3"] == _iso7]
                if not _prow7.empty:
                    _pharma_hit_b7 += float(_prow7["import_share_pct"].iloc[0])
        _imp_touched_b7 = sum(float(_imp_b7[_idx_by_iso_b7[iso]]) for iso in _changed_b7 if iso in _idx_by_iso_b7)
        _imp_touched_bn_b7 = _imp_touched_b7 / 1e6
        _imp_touched_pct_b7 = _imp_touched_b7 / max(_tot_imp_b7, 1) * 100

        if _n_changed_b7 > 0:
            st.markdown(f'<div class="insight-box">Your changes touch <b>${_imp_touched_bn_b7:,.0f}B</b> of US imports (<b>{_imp_touched_pct_b7:.1f}%</b> of everything America buys abroad).</div>', unsafe_allow_html=True)
        if _pharma_hit_b7 >= 3:
            st.markdown(f'<div class="insight-box" style="border-left-color:#f87171">⚠️ <b>Medicine supply risk:</b> you raised tariffs on countries supplying <b>{_pharma_hit_b7:.1f}%</b> of US pharmaceutical imports — expect drug price pressure and sourcing shifts.</div>', unsafe_allow_html=True)

        # ── Where does your policy rank? ────────────────────────────────────
        st.markdown('<div class="section-header">Where does your policy rank against the real scenarios?</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-box">The gray bars are the paper\'s full general-equilibrium scenarios. Your scenario (highlighted) is a PE estimate layered on the Liberation Day baseline — compare direction and rough size, not exact decimals.</div>', unsafe_allow_html=True)

        _rank_scens_b7 = [
            ("USTR + No Retaliation", float(_res_b7[_idus_b7, 0, 0])),
            ("USTR + Lump-Sum Rebate", float(_res_b7[_idus_b7, 0, 7])),
            ("Optimal Tariff", float(_res_b7[_idus_b7, 0, 3])),
            ("USTR + Reciprocal Retaliation", float(_res_b7[_idus_b7, 0, 5])),
            ("USTR + Optimal Retaliation", float(_res_b7[_idus_b7, 0, 4])),
            ("Flat 15% Tariff", float(_res15_b7[_idus_b7, 0])),
            ("⭐ YOUR SCENARIO (PE)", _nw_us_b7),
        ]
        _rank_df_b7 = pd.DataFrame(_rank_scens_b7, columns=["scenario", "welfare"]).sort_values("welfare")
        _rank_colors_b7 = ["#22d3a0" if s.startswith("⭐") else ("#475569" if w < 0 else "#64748b")
                           for s, w in zip(_rank_df_b7["scenario"], _rank_df_b7["welfare"])]
        fig_rank_b7 = go.Figure(go.Bar(
            x=_rank_df_b7["welfare"], y=_rank_df_b7["scenario"],
            orientation="h", marker_color=_rank_colors_b7,
            marker_line_color=["#ffffff" if s.startswith("⭐") else "rgba(0,0,0,0)" for s in _rank_df_b7["scenario"]],
            marker_line_width=[2 if s.startswith("⭐") else 0 for s in _rank_df_b7["scenario"]],
            text=[f"{v:+.2f}%" for v in _rank_df_b7["welfare"]], textposition="outside",
        ))
        fig_rank_b7.add_vline(x=0, line_color="#4b5563", line_width=1)
        fig_rank_b7.update_layout(**PLOTLY_THEME, height=360,
            title="US Welfare Under Each Policy (your scenario in green)")
        fig_rank_b7.update_xaxes(title_text="US Welfare Change (%)")
        st.plotly_chart(fig_rank_b7, use_container_width=True)

        # ── Country-level results ───────────────────────────────────────────
        st.markdown('<div class="section-header">Country-by-country results</div>', unsafe_allow_html=True)
        _cb1, _cb2 = st.columns(2)
        _names_sel_b7 = [n for n in _sel_b7]
        with _cb1:
            fig_cmp_b7 = go.Figure()
            fig_cmp_b7.add_trace(go.Bar(
                name="Liberation Day", x=_names_sel_b7,
                y=[_lds_b7[_iso_by_name_b7[n]] for n in _names_sel_b7],
                marker_color="#2563eb",
                text=[f"{_lds_b7[_iso_by_name_b7[n]]}%" for n in _names_sel_b7], textposition="outside",
            ))
            fig_cmp_b7.add_trace(go.Bar(
                name="Your Scenario", x=_names_sel_b7,
                y=[_rates_b7[_iso_by_name_b7[n]] for n in _names_sel_b7],
                marker_color=["#22d3a0" if _rates_b7[_iso_by_name_b7[n]] < _lds_b7[_iso_by_name_b7[n]]
                              else ("#f87171" if _rates_b7[_iso_by_name_b7[n]] > _lds_b7[_iso_by_name_b7[n]] else "#64748b")
                              for n in _names_sel_b7],
                text=[f"{_rates_b7[_iso_by_name_b7[n]]}%" for n in _names_sel_b7], textposition="outside",
            ))
            fig_cmp_b7.update_layout(**PLOTLY_THEME, height=340,
                title="Tariff Rates: Liberation Day vs Yours", barmode="group",
                legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250"))
            fig_cmp_b7.update_yaxes(title_text="Tariff (%)")
            st.plotly_chart(fig_cmp_b7, use_container_width=True)
        with _cb2:
            _wdel_b7 = [(r.get("country", ""), float(r.get("welfare_delta_pct") or 0))
                        for r in _pe_ctys_b7]
            _wdel_df_b7 = pd.DataFrame(_wdel_b7, columns=["country", "delta"]).sort_values("delta")
            fig_wd_b7 = go.Figure(go.Bar(
                x=_wdel_df_b7["delta"], y=_wdel_df_b7["country"],
                orientation="h",
                marker_color=["#ffffff" if c == "United States" else ("#22d3a0" if v > 0 else "#f87171" if v < 0 else "#64748b")
                              for c, v in zip(_wdel_df_b7["country"], _wdel_df_b7["delta"])],
                text=[f"{v:+.2f}pp" for v in _wdel_df_b7["delta"]], textposition="outside",
            ))
            fig_wd_b7.add_vline(x=0, line_color="#4b5563", line_width=1)
            fig_wd_b7.update_layout(**PLOTLY_THEME, height=340,
                title="Welfare Change vs Liberation Day (US in white)")
            fig_wd_b7.update_xaxes(title_text="Welfare Δ (pp)")
            st.plotly_chart(fig_wd_b7, use_container_width=True)

        # ── World map of your scenario ──────────────────────────────────────
        _map_iso_b7, _map_z_b7, _map_txt_b7 = [], [], []
        for r in _pe_ctys_b7:
            if r.get("iso3") == "USA":
                continue
            _map_iso_b7.append(r["iso3"])
            _map_z_b7.append(float(r.get("welfare_delta_pct") or 0))
            _map_txt_b7.append(r.get("country", r["iso3"]))
        if _map_iso_b7:
            fig_map_b7 = go.Figure()
            fig_map_b7.add_trace(go.Choropleth(
                locations=_cl_b7["iso3"], z=[0] * len(_cl_b7),
                colorscale=[[0, "#1a1d2e"], [1, "#1a1d2e"]], showscale=False,
                marker_line_color="#0f1117", marker_line_width=0.3, hoverinfo="skip",
            ))
            _zmax_b7 = max(abs(min(_map_z_b7)), abs(max(_map_z_b7)), 0.5)
            fig_map_b7.add_trace(go.Choropleth(
                locations=_map_iso_b7, z=_map_z_b7,
                zmin=-_zmax_b7, zmax=_zmax_b7,
                colorscale=[[0, "#f87171"], [0.5, "#fef3c7"], [1, "#22d3a0"]],
                marker_line_color="#ffffff", marker_line_width=1,
                colorbar=dict(title="Welfare Δ", tickfont=dict(color="#94a3b8"), bgcolor="#1a1d2e"),
                customdata=_map_txt_b7,
                hovertemplate="<b>%{customdata}</b><br>Welfare Δ: %{z:+.2f}pp<extra></extra>",
            ))
            fig_map_b7.update_layout(**PLOTLY_THEME, height=380,
                title="Countries Touched by Your Scenario",
                geo=dict(bgcolor="#0f1117", showframe=False, showcoastlines=True,
                         coastlinecolor="#2d3250", showland=True, landcolor="#1a1d2e",
                         showocean=True, oceancolor="#0f1117", projection_type="natural earth"))
            st.plotly_chart(fig_map_b7, use_container_width=True)

        # ── Results table + CSV download ────────────────────────────────────
        _tbl_rows_b7 = []
        for _cname7 in _sel_b7:
            _iso7 = _iso_by_name_b7[_cname7]
            _rec7 = next((r for r in _pe_ctys_b7 if r.get("iso3") == _iso7), {})
            _ci7 = _idx_by_iso_b7.get(_iso7)
            _tbl_rows_b7.append({
                "Country": _cname7,
                "LD Tariff (%)": _lds_b7[_iso7],
                "Your Tariff (%)": _rates_b7[_iso7],
                "US Imports ($B)": round(float(_imp_b7[_ci7]) / 1e6, 1) if _ci7 is not None else 0,
                "Welfare Before (%)": round(float(_rec7.get("baseline_welfare_pct") or 0), 2),
                "Welfare After (%)": round(float(_rec7.get("new_welfare_pct") or 0), 2),
                "Welfare Δ (pp)": round(float(_rec7.get("welfare_delta_pct") or 0), 2),
            })
        _tbl_rows_b7.append({
            "Country": "🇺🇸 United States",
            "LD Tariff (%)": "—", "Your Tariff (%)": "—", "US Imports ($B)": "—",
            "Welfare Before (%)": round(_bw_us_b7, 2),
            "Welfare After (%)": round(_nw_us_b7, 2),
            "Welfare Δ (pp)": round(_wd_us_b7, 2),
        })
        _tbl_b7 = pd.DataFrame(_tbl_rows_b7)
        st.dataframe(_tbl_b7, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download scenario as CSV",
                           _tbl_b7.to_csv(index=False).encode("utf-8"),
                           "my_tariff_scenario.csv", "text/csv", key="bys_dl")

        # ── Plain-English verdict ───────────────────────────────────────────
        if _n_changed_b7 > 0:
            _chg_desc_b7 = ", ".join(
                f"{next(n for n in _sel_b7 if _iso_by_name_b7[n] == iso)} {_lds_b7[iso]}%→{r}%"
                for iso, r in list(_changed_b7.items())[:5])
            if _n_changed_b7 > 5:
                _chg_desc_b7 += f" (+{_n_changed_b7 - 5} more)"
            _dir_us_b7 = "better off" if _wd_us_b7 > 0 else "worse off"
            _price_b7 = "fall" if _dcpi_b7 < 0 else "rise"
            _rev_b7 = "raises" if _drev_bn_b7 > 0 else "loses"
            _worst_b7 = min(((r.get("country", ""), float(r.get("welfare_delta_pct") or 0))
                             for r in _pe_ctys_b7 if r.get("iso3") != "USA"),
                            key=lambda x: x[1], default=("", 0))
            st.markdown(
                f'<div class="insight-box" style="border-left-color:{"#22d3a0" if _wd_us_b7 > 0 else "#f87171"}">'
                f'<b>💡 The verdict:</b> You changed {_chg_desc_b7}. '
                f'America is <b>{_dir_us_b7}</b> by {abs(_wd_us_b7):.2f}pp: consumer prices {_price_b7} '
                f'{abs(_dcpi_b7):.2f}% ({_hh_cost_b7:+,.0f} $/household/yr) and the policy {_rev_b7} '
                f'${abs(_drev_bn_b7):,.0f}B in tariff revenue vs Liberation Day. '
                + (f'Hardest hit abroad: <b>{_worst_b7[0]}</b> ({_worst_b7[1]:+.2f}pp). ' if _worst_b7[0] and _worst_b7[1] < 0 else '')
                + 'Assumes no retaliation; linearised PE approximation.</div>',
                unsafe_allow_html=True)

        # ── Methodology ─────────────────────────────────────────────────────
        with st.expander("🔬 How is this calculated?"):
            st.markdown("""
- **Welfare**: linearised partial-equilibrium approximation on top of the Liberation Day GE baseline —
  `Δwelfare ≈ Δτ × import_share × (−ε / (1 + τ_LD))` with trade elasticity **ε = 4** (from `run_tariff_scenario` in the MCP server).
- **Consumer prices**: `ΔCPI ≈ Σ (import_share × Δτ) × passthrough`, using the **29.7%** consumer passthrough estimated in the retail analysis (`sector_retail_results.npz`). Household cost divides the total by 132M US households.
- **Tariff revenue**: demand-adjusted — imports shrink when taxed: `M(τ) = M₂₀₂₃ × ((1+τ)/(1+τ_LD))^(−ε)`, revenue = `τ × M(τ)`, summed over all countries and compared with the same formula at Liberation Day rates.
- **What this misses**: general-equilibrium feedback (wages, exchange rates), foreign retaliation, supply-chain IO amplification, and product-level substitution. Use the AI Analyst tab to run questions against the full model outputs.
            """)


