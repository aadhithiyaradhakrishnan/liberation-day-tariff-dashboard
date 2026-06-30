import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Liberation Day Tariff Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
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
    prices = pd.read_csv(os.path.join(DATA, "retail_prices_illustrative.csv"))
    prices["pct_increase"] = (prices["Price After Tariff"] - prices["Price Before Tariff"]) / prices["Price Before Tariff"] * 100
    cavallo = pd.read_csv(os.path.join(DATA, "daily_price_indices_cavallo_etal.csv"))
    cavallo["date"] = pd.to_datetime(cavallo["date"], format="%d%b%Y")
    return prices, cavallo

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

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">🏛️ Liberation Day Tariff Impact Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Replication of Ignatenko, Macedoni, Lashkaripour & Simonovska (2025) · April 2025 US Tariff Announcements</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🌐 Macro Overview", "💊 Pharma Supply Chain", "🛒 Retail & Consumer Prices", "🏭 Manufacturing Exposure", "🤖 AI Analyst", "🔌 MCP Analyst"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — MACRO OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    results, Y_i, id_US, cl, results_15pct, d_trade_15pct = load_baseline()

    # Scenario definitions — None signals the 15% custom scenario
    SCENARIOS = {
        "USTR + No Retaliation":         0,
        "USTR + Lump-Sum Rebate":        7,
        "Optimal Tariff":                3,
        "USTR + Reciprocal Retaliation": 5,
        "USTR + Optimal Retaliation":    4,
        "Flat 15% Tariff (Custom)":      None,
    }
    METRICS = ["Welfare", "CPI", "Imports", "Exports", "Real Wage", "Rev/GDP"]

    col_ctrl, _ = st.columns([2, 5])
    with col_ctrl:
        scenario_name = st.selectbox("Scenario", list(SCENARIOS.keys()), index=0)
    sc = SCENARIOS[scenario_name]

    is_15pct = (sc is None)
    # Column order in results: [welfare, deficit, exports/GDP, imports/GDP, employment, CPI, rev/GDP]
    us_vals = results_15pct[id_US, :7] if is_15pct else results[id_US, :7, sc]

    # KPI row — indices: 0=welfare, 1=deficit, 2=exports, 3=imports, 4=employment, 5=CPI, 6=rev/GDP
    kpi_data = [
        ("US Welfare",      f"{us_vals[0]:+.2f}%", "positive" if us_vals[0] > 0 else "negative", "Real income change"),
        ("US CPI",          f"{us_vals[5]:+.1f}%", "negative" if us_vals[5] > 0 else "positive", "Consumer price level"),
        ("US Imports/GDP",  f"{us_vals[3]:+.1f}%", "negative" if us_vals[3] < 0 else "positive", "Import volume / GDP"),
        ("US Exports/GDP",  f"{us_vals[2]:+.1f}%", "negative" if us_vals[2] < 0 else "positive", "Export volume / GDP"),
        ("Employment",      f"{us_vals[4]:+.2f}%", "positive" if us_vals[4] > 0 else "negative", "Labor employment"),
        ("Trade Deficit Δ", f"{us_vals[1]:+.1f}%", "neutral",  "Change in trade deficit"),
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

    # ── World welfare map ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Country-Level Welfare Change (194 Countries)</div>', unsafe_allow_html=True)

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
        **PLOTLY_THEME,
        height=420,
        geo=dict(
            bgcolor="#0f1117",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#2d3250",
            showland=True, landcolor="#1a1d2e",
            showocean=True, oceancolor="#0f1117",
            showlakes=False,
            projection_type="natural earth",
        ),
        coloraxis_colorbar=dict(
            title="Welfare %", tickfont=dict(color="#94a3b8"),
            title_font=dict(color="#94a3b8"), bgcolor="#1a1d2e",
            bordercolor="#2d3250",
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ── Scenario comparison bar + trade metrics ───────────────────────────
    st.markdown('<div class="section-header">Scenario Comparison</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        sc_names = list(SCENARIOS.keys())
        sc_idxs  = list(SCENARIOS.values())
        welfare_by_sc = [
            results_15pct[id_US, 0] if i is None else results[id_US, 0, i]
            for i in sc_idxs
        ]
        cpi_by_sc = [
            results_15pct[id_US, 5] if i is None else results[id_US, 5, i]
            for i in sc_idxs
        ]

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Bar(
            name="US Welfare", x=sc_names, y=welfare_by_sc,
            marker_color=["#22d3a0" if v > 0 else "#f87171" for v in welfare_by_sc],
            text=[f"{v:+.2f}%" for v in welfare_by_sc], textposition="outside",
        ))
        fig_sc.add_trace(go.Bar(
            name="CPI Change", x=sc_names, y=cpi_by_sc,
            marker_color="#fbbf24", opacity=0.7,
            text=[f"{v:+.1f}%" for v in cpi_by_sc], textposition="outside",
        ))
        fig_sc.update_layout(**PLOTLY_THEME, height=360,
            title="US Welfare & CPI Across Scenarios",
            barmode="group",
            xaxis_tickangle=-25,
            legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250"))
        st.plotly_chart(fig_sc, use_container_width=True)

    with c2:
        # Column order: [0=welfare, 1=deficit, 2=exports, 3=imports, 4=employment, 5=CPI, 6=rev/GDP]
        METRICS    = ["Welfare", "CPI", "Imports/GDP", "Exports/GDP", "Employment", "Trade Deficit", "Rev/GDP"]
        col_order  = [0, 5, 3, 2, 4, 1, 6]
        metrics_vals = [
            results_15pct[id_US, c] if is_15pct else results[id_US, c, sc]
            for c in col_order
        ]
        colors = ["#22d3a0" if v >= 0 else "#f87171" for v in metrics_vals]
        fig_met = go.Figure(go.Bar(
            x=METRICS, y=metrics_vals,
            marker_color=colors,
            text=[f"{v:+.2f}%" for v in metrics_vals],
            textposition="outside",
        ))
        fig_met.update_layout(**PLOTLY_THEME, height=360,
            title=f"US Metrics — {scenario_name}",
            yaxis_title="% Change")
        st.plotly_chart(fig_met, use_container_width=True)

    # ── Top 15 winners / losers ────────────────────────────────────────────
    st.markdown('<div class="section-header">Top 15 Winners & Losers (Welfare %)</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    map_df_sorted = map_df.sort_values("welfare")  # map_df["welfare"] already set above
    with c3:
        losers = map_df_sorted.head(15)
        fig_l = go.Figure(go.Bar(
            x=losers["welfare"], y=losers["CountryName"],
            orientation="h", marker_color="#f87171",
            text=[f"{v:.2f}%" for v in losers["welfare"]], textposition="outside",
        ))
        fig_l.update_layout(**PLOTLY_THEME, height=400, title="Top 15 Losers",
            xaxis_title="Welfare Change %")
        st.plotly_chart(fig_l, use_container_width=True)

    with c4:
        winners = map_df_sorted.tail(15).sort_values("welfare", ascending=False)
        fig_w = go.Figure(go.Bar(
            x=winners["welfare"], y=winners["CountryName"],
            orientation="h", marker_color="#22d3a0",
            text=[f"{v:.2f}%" for v in winners["welfare"]], textposition="outside",
        ))
        fig_w.update_layout(**PLOTLY_THEME, height=400, title="Top 15 Winners",
            xaxis_title="Welfare Change %")
        st.plotly_chart(fig_w, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — PHARMA SUPPLY CHAIN
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    pharma_df, monthly_cols = load_pharma1()
    dep, src, burd, exp = load_pharma_outputs()

    # KPIs
    total_2025 = pharma_df[pharma_df["Year"] == 2025]["annual_total"].sum()
    total_2024 = pharma_df[pharma_df["Year"] == 2024]["annual_total"].sum()
    top_country = dep.iloc[0]["country"]
    top_share   = dep.iloc[0]["share_2025_pct"]
    top_tariff  = dep.iloc[0]["tariff_pct"]
    burden_q1   = float(burd[burd["group"].str.contains("Q1")]["burden_pct_income"].iloc[0]) * 100
    burden_q5   = float(burd[burd["group"].str.contains("Q5")]["burden_pct_income"].iloc[0]) * 100

    kpi2 = [
        ("2025 US Pharma Imports", f"${total_2025/1e9:.1f}B", "neutral", "Customs value"),
        ("YoY Growth", f"{(total_2025/total_2024-1)*100:+.1f}%", "positive", "2024→2025"),
        ("Top Supplier", top_country, "neutral", f"{top_share:.1f}% share"),
        ("Top Supplier Tariff", f"{top_tariff:.0f}%", "warning", top_country),
        ("Q1 Drug Burden", f"{burden_q1:.2f}%", "negative", "% of income"),
        ("Q5 Drug Burden", f"{burden_q5:.2f}%", "positive", "% of income — 7x less"),
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

    # ── Time series: monthly US pharma imports ─────────────────────────────
    st.markdown('<div class="section-header">US Pharma Import Trends (2018–2025, Monthly)</div>', unsafe_allow_html=True)

    hts_filter = st.multiselect(
        "HTS Codes", [3002, 3003, 3004],
        default=[3002, 3003, 3004],
        format_func=lambda x: {3002:"3002 – Vaccines & Blood", 3003:"3003 – Unmixed Medicaments", 3004:"3004 – Formulated Drugs"}[x]
    )

    ts_df = pharma_df[pharma_df["HTS Number"].isin(hts_filter)].copy()
    # Build monthly time series
    rows = []
    for _, row in ts_df.iterrows():
        yr = int(row["Year"]) if not pd.isna(row["Year"]) else None
        if yr is None: continue
        for mi, m in enumerate(monthly_cols, 1):
            val = pd.to_numeric(row[m], errors="coerce")
            if pd.notna(val) and val > 0:
                rows.append({"year": yr, "month": mi, "value": val, "hts": int(row["HTS Number"])})
    ts = pd.DataFrame(rows)
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
        apply_theme(fig_ts, 340)
        st.plotly_chart(fig_ts, use_container_width=True)

    # ── Top suppliers + sourcing shifts ───────────────────────────────────
    st.markdown('<div class="section-header">Supplier Dependence & Sourcing Shifts</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig_dep = go.Figure(go.Bar(
            x=dep["share_2025_pct"], y=dep["country"],
            orientation="h",
            marker=dict(
                color=dep["tariff_pct"],
                colorscale=[[0,"#22d3a0"],[0.5,"#fbbf24"],[1,"#f87171"]],
                cmin=0, cmax=35,
                colorbar=dict(title="Tariff %", tickfont=dict(color="#94a3b8"),
                              title_font=dict(color="#94a3b8")),
            ),
            text=[f"{s:.1f}% | {t:.0f}% tariff" for s,t in zip(dep["share_2025_pct"], dep["tariff_pct"])],
            textposition="outside",
        ))
        fig_dep.update_layout(**PLOTLY_THEME, height=380,
            title="Top 10 Suppliers by Import Share (2025)",
            xaxis_title="Import Share %")
        fig_dep.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_dep, use_container_width=True)

    with c2:
        src_top = src.head(15).copy()
        colors_shift = ["#22d3a0" if v >= 0 else "#f87171" for v in src_top["change_pp"]]
        fig_src = go.Figure(go.Bar(
            x=src_top["change_pp"], y=src_top["country"],
            orientation="h",
            marker_color=colors_shift,
            text=[f"{v:+.2f}pp" for v in src_top["change_pp"]],
            textposition="outside",
        ))
        fig_src.add_vline(x=0, line_color="#4b5563", line_width=1)
        fig_src.update_layout(**PLOTLY_THEME, height=380,
            title="Sourcing Share Shift (Post-Tariff vs Pre-Tariff)",
            xaxis_title="Change in Share (percentage points)")
        fig_src.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_src, use_container_width=True)

    # ── Consumer burden by income quintile ────────────────────────────────
    st.markdown('<div class="section-header">Pharma Tariff Burden by Income Quintile</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">The tariff burden is highly regressive: Q1 (lowest income) bears <b>7× more</b> cost as a share of income than Q5. This is because lower-income households spend a higher fraction of their budget on prescription drugs.</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        fig_burd = go.Figure(go.Bar(
            x=burd["group"], y=burd["burden_pct_income"] * 100,
            marker_color=["#f87171","#fb923c","#fbbf24","#a3e635","#22d3a0"],
            text=[f"{v*100:.3f}%" for v in burd["burden_pct_income"]],
            textposition="outside",
        ))
        fig_burd.update_layout(**PLOTLY_THEME, height=320,
            title="Drug Tariff Burden as % of Income",
            yaxis_title="% of Annual Income",
            xaxis_title="Income Quintile")
        st.plotly_chart(fig_burd, use_container_width=True)

    with c4:
        fig_spend = go.Figure()
        fig_spend.add_trace(go.Bar(
            name="Annual Drug Spending",
            x=burd["group"], y=burd["annual_drug_spending_usd"],
            marker_color="#2563eb",
        ))
        fig_spend.add_trace(go.Bar(
            name="Extra Cost from Tariffs",
            x=burd["group"], y=burd["extra_cost_from_tariffs_usd"],
            marker_color="#f87171",
        ))
        fig_spend.update_layout(**PLOTLY_THEME, height=320,
            title="Annual Drug Spending vs Tariff Cost (USD)",
            barmode="group", yaxis_title="USD per Household",
            legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250"))
        st.plotly_chart(fig_spend, use_container_width=True)

    # ── Country exposure map ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Global Pharma Import Exposure (2024)</div>', unsafe_allow_html=True)
    fig_pmap = px.choropleth(
        exp, locations="iso3", color="import_share_pct",
        hover_name="country",
        hover_data={"tariff_pct": True, "import_value_usd": True, "iso3": False},
        color_continuous_scale=["#1a1d2e","#1e3a5f","#2563eb","#60a5fa","#bfdbfe"],
        labels={"import_share_pct": "Import Share %"},
    )
    fig_pmap.update_traces(
        marker_line_color="#0f1117", marker_line_width=0.3,
        hovertemplate="<b>%{hovertext}</b><br>Share: %{z:.2f}%<br>Tariff: %{customdata[0]:.0f}%<extra></extra>"
    )
    fig_pmap.update_layout(
        **PLOTLY_THEME, height=400,
        geo=dict(bgcolor="#0f1117", showframe=False, showcoastlines=True,
                 coastlinecolor="#2d3250", showland=True, landcolor="#1a1d2e",
                 showocean=True, oceancolor="#0f1117",
                 projection_type="natural earth"),
        coloraxis_colorbar=dict(title="Share %", tickfont=dict(color="#94a3b8"),
                                title_font=dict(color="#94a3b8")),
    )
    st.plotly_chart(fig_pmap, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — RETAIL & CONSUMER PRICES
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    prices, cavallo = load_retail()

    # KPIs
    avg_increase = prices["pct_increase"].mean()
    max_cat = prices.groupby("Product Type")["pct_increase"].mean().idxmax()
    max_cat_val = prices.groupby("Product Type")["pct_increase"].mean().max()
    cavallo_us_latest = (cavallo["index_usa"].iloc[-1] - 1) * 100
    cavallo_cn_latest = (cavallo["index_china"].iloc[-1] - 1) * 100
    regressivity = 8.40 / 5.94  # from paper results

    kpi3 = [
        ("Avg Retail Price Increase", f"+{avg_increase:.1f}%", "negative", "Across all categories"),
        ("Highest Category", max_cat, "warning", f"+{max_cat_val:.1f}% avg"),
        ("US Price Index Change", f"+{cavallo_us_latest:.2f}%", "negative", "Oct 2024 → Feb 2026"),
        ("China Price Index Change", f"+{cavallo_cn_latest:.2f}%", "neutral", "Same period"),
        ("Regressivity Ratio", f"{regressivity:.2f}×", "negative", "Q1 burden / Q5 burden"),
        ("GE Dampening Factor", "0.52×", "positive", "GE vs naïve first-order"),
    ]
    cols3 = st.columns(6)
    for col, (label, val, cls, sub) in zip(cols3, kpi3):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value {cls}" style="font-size:22px">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Cavallo daily price indices ────────────────────────────────────────
    st.markdown('<div class="section-header">Daily Price Indices: US vs Trading Partners (Cavallo et al.)</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Liberation Day was April 2, 2025. US prices are rising faster than Canada and Mexico but tracking China, suggesting tariff pass-through in goods categories.</div>', unsafe_allow_html=True)

    fig_cav = go.Figure()
    country_map = {
        "index_usa":    ("United States", "#2563eb", 2.5),
        "index_canada": ("Canada",        "#22d3a0", 1.5),
        "index_mexico": ("Mexico",        "#fbbf24", 1.5),
        "index_china":  ("China",         "#f87171", 1.5),
    }
    for col, (name, color, width) in country_map.items():
        pct = (cavallo[col] - 1) * 100
        fig_cav.add_trace(go.Scatter(
            x=cavallo["date"], y=pct, name=name,
            line=dict(color=color, width=width),
            hovertemplate=f"<b>{name}</b><br>%{{x|%b %d, %Y}}<br>%{{y:.3f}}%<extra></extra>"
        ))
    fig_cav.add_vline(x="2025-04-02", line_dash="dash", line_color="#f87171",
        annotation_text="Liberation Day (Apr 2)", annotation_position="top right",
        annotation_font_color="#f87171")
    fig_cav.update_layout(**PLOTLY_THEME, height=380,
        title="Price Index Level (Oct 2024 = 0%)",
        yaxis_title="% Change from Baseline",
        legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250"))
    st.plotly_chart(fig_cav, use_container_width=True)

    # ── Product price changes ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Retail Product Price Changes by Category</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])

    with c1:
        cat_stats = prices.groupby("Product Type").agg(
            avg_before=("Price Before Tariff","mean"),
            avg_after=("Price After Tariff","mean"),
            avg_pct=("pct_increase","mean"),
            count=("Product Name","count"),
        ).reset_index().sort_values("avg_pct", ascending=False)

        fig_cat = go.Figure()
        fig_cat.add_trace(go.Bar(
            name="Before Tariff", x=cat_stats["Product Type"], y=cat_stats["avg_before"],
            marker_color="#2563eb",
        ))
        fig_cat.add_trace(go.Bar(
            name="After Tariff", x=cat_stats["Product Type"], y=cat_stats["avg_after"],
            marker_color="#f87171",
        ))
        fig_cat.update_layout(**PLOTLY_THEME, height=340,
            title="Average Price Before vs After Tariff",
            barmode="group", yaxis_title="Average Price (₹)",
            legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250"))
        st.plotly_chart(fig_cat, use_container_width=True)

    with c2:
        fig_pct = go.Figure(go.Bar(
            x=cat_stats["avg_pct"], y=cat_stats["Product Type"],
            orientation="h",
            marker_color=["#f87171" if v > 35 else "#fbbf24" if v > 25 else "#22d3a0"
                          for v in cat_stats["avg_pct"]],
            text=[f"+{v:.1f}%" for v in cat_stats["avg_pct"]],
            textposition="outside",
        ))
        fig_pct.update_layout(**PLOTLY_THEME, height=340,
            title="Avg % Price Increase by Category",
            xaxis_title="% Increase")
        fig_pct.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_pct, use_container_width=True)

    # ── Price distribution ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Price Increase Distribution</div>', unsafe_allow_html=True)
    cat_sel = st.multiselect(
        "Filter by category", prices["Product Type"].unique().tolist(),
        default=prices["Product Type"].unique().tolist()
    )
    filtered = prices[prices["Product Type"].isin(cat_sel)]

    c3, c4 = st.columns(2)
    with c3:
        fig_hist = px.histogram(
            filtered, x="pct_increase", color="Product Type",
            nbins=40, opacity=0.75,
            color_discrete_sequence=["#2563eb","#22d3a0","#fbbf24","#f87171","#a78bfa"],
            labels={"pct_increase": "% Price Increase"},
        )
        fig_hist.update_layout(**PLOTLY_THEME, height=320,
            title="Distribution of % Price Increases",
            xaxis_title="% Price Increase", yaxis_title="Count",
            legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250"))
        st.plotly_chart(fig_hist, use_container_width=True)

    with c4:
        quintile_data = {
            "Quintile": ["Q1 (Poorest)", "Q2", "Q3", "Q4", "Q5 (Richest)"],
            "Burden":   [8.40, 7.20, 6.80, 6.30, 5.94],
        }
        qdf = pd.DataFrame(quintile_data)
        fig_q = go.Figure(go.Bar(
            x=qdf["Quintile"], y=qdf["Burden"],
            marker_color=["#f87171","#fb923c","#fbbf24","#a3e635","#22d3a0"],
            text=[f"{v:.2f}%" for v in qdf["Burden"]],
            textposition="outside",
        ))
        fig_q.update_layout(**PLOTLY_THEME, height=320,
            title="Tariff Burden as % of Household Budget",
            yaxis_title="% of Budget", xaxis_title="Income Quintile")
        st.plotly_chart(fig_q, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — MANUFACTURING EXPOSURE
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    naics, price_idx, shocks, hts, mfg_stats = load_manufacturing()

    # KPIs
    kpi4 = [
        ("Avg Mfg Tariff",      f"{mfg_stats['tau_mfg_avg']*100:.1f}%",    "negative", "Trade-weighted"),
        ("Import Penetration",  f"{mfg_stats['import_penetration_mfg']*100:.1f}%","warning","Mfg sector"),
        ("CPI Contribution",    f"+{mfg_stats['cpi_mfg_contribution']:.2f}pp","negative","Of +7.09% total"),
        ("IO Multiplier",       f"{mfg_stats['io_mult_mfg']:.3f}×",          "warning", "Intermediate import"),
        ("HTS8 Mfg Rate",       f"{mfg_stats['hts8_mfg_rate']*100:.1f}%",    "negative","Avg tariff rate"),
        ("HTS8 Steel Rate",     f"{mfg_stats['hts8_steel_rate']*100:.1f}%",   "negative","Steel/aluminum"),
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

    # ── NAICS gross output ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Gross Output by NAICS Sector (BEA, 2016–2021)</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">96% of the total +7.09pp CPI impact comes from manufacturing. The top 5 NAICS subsectors account for ~70% of the $6.32T manufacturing gross output base.</div>', unsafe_allow_html=True)

    year_sel = st.select_slider("Year", options=[2016,2017,2018,2019,2020,2021], value=2021)

    naics_plot = naics[["NAICS Code","Name", str(year_sel)]].copy()
    naics_plot.columns = ["naics","name","go_value"]
    naics_plot["go_value"] = pd.to_numeric(naics_plot["go_value"], errors="coerce")
    naics_plot = naics_plot.dropna(subset=["go_value"])

    c1, c2 = st.columns(2)
    with c1:
        top20 = naics_plot.nlargest(20, "go_value")
        fig_naics = go.Figure(go.Bar(
            x=top20["go_value"] / 1e6,
            y=top20["name"].str[:35],
            orientation="h",
            marker_color="#2563eb",
            text=[f"${v/1e6:.1f}T" for v in top20["go_value"]],
            textposition="outside",
        ))
        fig_naics.update_layout(**PLOTLY_THEME, height=500,
            title=f"Top 20 NAICS Sectors by Gross Output ({year_sel})",
            xaxis_title="Gross Output ($T)")
        fig_naics.update_yaxes(autorange="reversed", tickfont=dict(size=10))
        st.plotly_chart(fig_naics, use_container_width=True)

    with c2:
        # Gross output time series for top 8
        top8_names = naics_plot.nlargest(8, "go_value")["name"].tolist()
        years = [2016, 2017, 2018, 2019, 2020, 2021]
        fig_ts_n = go.Figure()
        colors_n = ["#2563eb","#22d3a0","#fbbf24","#f87171","#a78bfa","#fb923c","#38bdf8","#4ade80"]
        for i, name in enumerate(top8_names):
            row = naics[naics["Name"] == name]
            if row.empty: continue
            vals = [pd.to_numeric(row[str(y)].iloc[0], errors="coerce") / 1e6 for y in years]
            fig_ts_n.add_trace(go.Scatter(
                x=years, y=vals, name=name[:25],
                line=dict(color=colors_n[i % len(colors_n)], width=2),
                mode="lines+markers",
            ))
        fig_ts_n.update_layout(**PLOTLY_THEME, height=500,
            title="Gross Output Trend — Top 8 Sectors",
            yaxis_title="Gross Output ($T)", xaxis_title="Year",
            legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250", font=dict(size=10)))
        st.plotly_chart(fig_ts_n, use_container_width=True)

    # ── Sector tariff shocks ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Liberation Day Tariff Shocks by Sector</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        shock_ld = shocks[shocks["scenario"] == "liberation_day_schedule"].copy()
        shock_ld["tariff_pct"] = shock_ld["tariff_rate"] * 100
        fig_shock = go.Figure(go.Bar(
            x=shock_ld["tariff_pct"],
            y=shock_ld["model_sector"].str.replace("_"," ").str.title(),
            orientation="h",
            marker_color=["#f87171" if v > 5 else "#fbbf24" if v > 2 else "#22d3a0"
                         for v in shock_ld["tariff_pct"]],
            text=[f"{v:.2f}%" for v in shock_ld["tariff_pct"]],
            textposition="outside",
        ))
        fig_shock.update_layout(**PLOTLY_THEME, height=320,
            title="Liberation Day Effective Tariff Rate by Sector",
            xaxis_title="Effective Tariff Rate %")
        fig_shock.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_shock, use_container_width=True)

    with c4:
        # HTS8 tariff rate distribution (capped at 100%)
        hts_plot = hts[hts["mfn_rate"] <= 1.0].copy()
        hts_plot["mfn_pct"] = hts_plot["mfn_rate"] * 100
        fig_hts = px.histogram(
            hts_plot, x="mfn_pct", nbins=50,
            color_discrete_sequence=["#2563eb"],
            labels={"mfn_pct": "MFN Tariff Rate (%)"},
        )
        fig_hts.update_layout(**PLOTLY_THEME, height=320,
            title="Distribution of HTS-8 MFN Tariff Rates (13,100 product lines)",
            xaxis_title="MFN Tariff Rate %", yaxis_title="# Product Lines")
        st.plotly_chart(fig_hts, use_container_width=True)

    # ── Price index trends ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Producer Price Index by NAICS Subsector (BLS, 2012–2022)</div>', unsafe_allow_html=True)

    year_cols = [c for c in price_idx.columns if "Annual" in str(c)]
    fig_ppi = go.Figure()
    colors_ppi = ["#2563eb","#22d3a0","#fbbf24","#f87171","#a78bfa","#fb923c","#38bdf8","#4ade80","#f472b6","#a3e635"]
    for i, (_, row) in enumerate(price_idx.iterrows()):
        vals = pd.to_numeric(row[year_cols], errors="coerce")
        yr_labels = [str(y).replace("Annual ","") for y in year_cols]
        fig_ppi.add_trace(go.Scatter(
            x=yr_labels, y=vals.values,
            name=f"NAICS {int(row['NAICS4'])}",
            line=dict(color=colors_ppi[i % len(colors_ppi)], width=1.8),
            mode="lines+markers",
        ))
    fig_ppi.update_layout(**PLOTLY_THEME, height=360,
        title="Producer Price Index Trends (2012=base)",
        yaxis_title="PPI Index", xaxis_title="Year",
        legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3250", font=dict(size=10)))
    st.plotly_chart(fig_ppi, use_container_width=True)


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
