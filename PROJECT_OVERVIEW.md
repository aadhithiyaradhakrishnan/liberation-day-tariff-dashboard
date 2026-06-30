# Liberation Day Tariff Impact — Project Overview

**Paper:** Ignatenko, Macedoni, Lashkaripour & Simonovska (2025)
**Event:** April 2, 2025 — US "Liberation Day" tariff announcements
**Scope:** 194 countries, 9 tariff scenarios, 3 sector deep-dives, AI-powered dashboard

---

## What This Project Does

This project replicates a cutting-edge 2025 economics paper and extends it into a
full-stack interactive analysis platform. It models how the April 2025 US tariff
announcements ripple through the global economy — measuring welfare, prices,
employment, and trade for every country on earth — then wraps that analysis in an
interactive dashboard and exposes it to AI agents via the Model Context Protocol.

---

## Architecture

```
Paper (2025)
    ↓
Raw Data (CEPII, USITC, BEA, BLS, Cavallo)
    ↓
GE Model Solver (194 countries, 9 scenarios)
    ↓
Sector Analyses (Pharma, Retail, Manufacturing)
    ↓
Output Files (.npz, .csv)
    ↓
┌──────────────────┬─────────────────────┐
│   Dashboard      │    MCP Server       │
│   (Streamlit)    │    (FastMCP 3.4.2)  │
│   6 tabs         │    6 tools          │
│   + AI chat      │    stdio protocol   │
└──────────────────┴─────────────────────┘
         ↓                   ↓
   Human analyst      External AI agents
```

---

## Layer 1 — The Academic Foundation

The paper quantifies the global economic impact of Liberation Day tariffs using a
state-of-the-art multi-country, multi-sector General Equilibrium model.

The project replicates it in three languages:

| Language | Entry point         | Purpose                        |
|----------|---------------------|--------------------------------|
| MATLAB   | `run_all_matlab.m`  | Original model solver          |
| Stata    | `run_all_stata.do`  | Econometric tables             |
| Python   | `run_all_python.py` | Full Python replication        |

---

## Layer 2 — The Economic Model

A **194-country General Equilibrium model** (Caliendo-Parro framework):

- Inputs: bilateral trade flows, tariff schedules, input-output tables
- Solves for equilibrium prices and quantities under each tariff scenario
- Outputs welfare, CPI, employment, and trade changes for every country
- Results shape: (194 countries × 7 metrics × 9 scenarios)

**Metrics computed per country per scenario:**

| Index | Metric           |
|-------|------------------|
| 0     | Welfare (%)      |
| 1     | Trade deficit (%)  |
| 2     | Exports / GDP (%)  |
| 3     | Imports / GDP (%)  |
| 4     | Employment (%)   |
| 5     | CPI (%)          |
| 6     | Revenue / GDP (%)  |

**9 scenarios modelled:**

| Scenario | Description |
|----------|-------------|
| USTR + No Retaliation | Liberation Day tariffs, no foreign response (baseline) |
| USTR + Lump-Sum Rebate | Tariff revenue recycled via lump-sum transfers |
| Optimal Tariff | US maximises own welfare |
| USTR + Reciprocal Retaliation | Trading partners match US tariffs |
| USTR + Optimal Retaliation | Trading partners respond optimally |
| Flat 15% (Custom) | Uniform 15% tariff on all US imports — re-run GE solver |

**Key model files:**
```
code_python/
├── utils/solver_utils.py         ← GE solver (solve_nu)
├── analysis/main_baseline.py     ← baseline scenario
├── analysis/main_deficit.py      ← deficit model
├── analysis/main_io.py           ← input-output model
├── analysis/main_regional.py     ← regional breakdown
├── config.py                     ← model parameters
└── diagnostic_comparison.py      ← validation
```

---

## Layer 3 — Sector Analyses

### Pharma Supply Chain
**Data source:** USITC DataWeb (`Pharma1.xlsx`) — HTS codes 3002, 3003, 3004
**2024 US pharma imports:** $210.8B
**Trade-weighted Liberation Day tariff:** 21.28%

Key findings:
- Ireland is the top supplier (~20% share) with a 10% tariff
- HHI concentration: Highly concentrated (>2500) — supply chain is fragile
- Drug price pressure: +2.47% (production route) to +10.34% (import-dependence route)
- Implied import volume response: -40.4% (elasticity = 2.3)
- Burden is highly regressive: Q1 (lowest income) bears 7× more cost as % of income than Q5

**Outputs:**
```
python_output/
├── pharma1_objective1_dependence_2025.csv       ← top suppliers by share
├── pharma1_objective2_sourcing_shifts_2025.csv  ← pre/post tariff shifts
├── pharma1_objective3_consumer_burden_2025.csv  ← quintile incidence
├── pharma1_country_exposure_2024.csv            ← country-level exposure
├── pharma1_supply_chain_risk_2024.csv           ← risk tiers
└── pharma1_hts_exposure_2024.csv                ← product-level exposure
```

### Retail & Consumer Prices
**Data sources:** Illustrative retail prices + Cavallo et al. daily price indices

Key findings:
- Average retail price increase: varies by category (electronics and footwear hit hardest)
- Cavallo price index shows US prices rising faster than Canada/Mexico post-Liberation Day
- Tariff burden is regressive: Q1 bears 8.40% of budget vs Q5 at 5.94% (ratio: 1.41×)
- GE dampening factor: 0.52× — actual impact is roughly half the naïve first-order estimate

### Manufacturing Exposure
**Data sources:** BEA NAICS gross output + BLS PPI + sector tariff shocks

Key findings:
- Total manufacturing gross output: $6.32T (2021)
- Average manufacturing tariff (trade-weighted): significant jump post-Liberation Day
- CPI contribution from manufacturing: +7.09 percentage points (96% of total)
- Input-output multiplier: intermediate import tariffs amplify through supply chains

---

## Layer 4 — Data Sources

```
data/
├── base_data/
│   ├── trade_cepii.csv                     ← 194×194 bilateral trade matrix
│   ├── tariffs.csv                         ← USTR tariff schedule
│   └── country_labels.csv                  ← ISO3 codes + country names
├── Pharma1.xlsx                            ← USITC pharma imports (2018–2025)
├── retail_prices_illustrative.csv          ← retail price before/after tariff
├── daily_price_indices_cavallo_etal.csv    ← US, Canada, Mexico, China indices
├── us_tariff_schedule_2025_hts8.csv        ← 13,100 HTS-8 product lines
└── code_and_release_data/
    └── 301 model/
        ├── D_GO_by_NAICS.xlsx              ← BEA gross output by NAICS
        └── D_price_indices.xlsx            ← BLS PPI by NAICS subsector
```

---

## Layer 5 — The Dashboard

**Tech stack:** Streamlit + Plotly (dark theme)
**Entry point:** `dashboard/app.py`

### 6 Tabs

| Tab | Content |
|-----|---------|
| Macro Overview | World welfare choropleth map, country winners/losers, scenario comparison KPIs |
| Pharma Supply Chain | Import trends (2018–2025), supplier dependence, sourcing shifts, quintile burden |
| Retail & Consumer Prices | Cavallo price indices, category price increases, distribution, quintile burden |
| Manufacturing Exposure | NAICS gross output, tariff shocks by sector, PPI trends |
| AI Analyst | Claude chat — tools called directly via Python imports |
| MCP Analyst | Claude chat — tools called via live MCP subprocess over stdio protocol |

### Design System
- Background: `#0f1117`
- Accent blue: `#2563eb`
- Positive: `#22d3a0`
- Negative: `#f87171`
- Warning: `#fbbf24`
- Font: Inter

---

## Layer 6 — MCP Server

**File:** `mcp_server/server.py`
**Framework:** FastMCP 3.4.2
**Transport:** stdio (spawned as subprocess)

### How it works

```
Client connects (dashboard or external AI)
        ↓
initialize() — MCP handshake
        ↓
list_tools() — server returns 6 tool definitions dynamically
        ↓
tools/call — client sends tool name + arguments
        ↓
Server executes Python function, reads data files
        ↓
Returns {data: {...}, chart_spec: {...}} as JSON
        ↓
Client renders result (text + Plotly chart)
```

### 6 Tools

| Tool | Description |
|------|-------------|
| `get_welfare_results` | GE welfare/CPI/trade results by country and scenario |
| `get_scenario_comparison` | Pivot welfare across multiple countries × scenarios |
| `get_pharma_supplier_risk` | Supplier HHI concentration and risk tiers |
| `get_quintile_burden` | Regressive tariff incidence by income quintile |
| `get_manufacturing_shock` | Sector tariff shocks and gross output by NAICS |
| `run_tariff_scenario` | Custom tariff partial equilibrium welfare approximation |

### Running the MCP server standalone
```bash
python -m mcp_server.server
```
Any MCP-compatible client (Claude Desktop, Cursor) can then connect to it.

---

## Layer 7 — AI Analyst vs MCP Analyst

| Feature | AI Analyst (Tab 5) | MCP Analyst (Tab 6) |
|---------|-------------------|---------------------|
| Tool access | Direct Python import | Subprocess over stdio |
| Tool discovery | Hardcoded schemas | `session.list_tools()` — dynamic |
| Tool execution | `fn(**inputs)` in-process | `session.call_tool()` over MCP protocol |
| Server lifecycle | No server | Spawns + tears down per query |
| Chart rendering | From chart_spec dict | From chart_spec in MCP JSON response |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Economic model | Python (NumPy, SciPy) |
| Dashboard | Streamlit + Plotly |
| AI layer | Anthropic API (claude-sonnet-4-6) |
| MCP server | FastMCP 3.4.2 |
| Data formats | CSV, XLSX, NPZ (NumPy compressed) |
| SSL bypass | httpx `verify=False`, `trust_env=False` |

---

## Project Structure

```
liberation-day-replication-main/
├── code_python/                  ← Python replication of GE model
│   ├── analysis/                 ← scenario runners + sector analyses
│   ├── utils/                    ← solver utilities
│   └── config.py
├── code/                         ← MATLAB/R visualisation code
├── data/                         ← all raw data inputs
├── python_output/                ← all computed outputs
├── dashboard/
│   ├── app.py                    ← 6-tab Streamlit dashboard
│   └── compute_15pct_scenario.py ← runs GE solver for flat 15% scenario
├── mcp_server/
│   ├── server.py                 ← FastMCP server with 6 tools
│   └── __init__.py
├── .streamlit/
│   ├── secrets.toml              ← Anthropic API key
│   └── config.toml               ← headless mode
├── output/                       ← original paper outputs
├── capstone_report/              ← academic report
├── run_all_python.py             ← master runner
├── run_all_matlab.m
├── run_all_stata.do
├── requirements_mcp.txt
└── README.md
```

---

## Key Results (USTR No-Retaliation Baseline)

| Metric | US Value |
|--------|----------|
| Welfare change | Negative (costs outweigh gains) |
| CPI increase | +7.09 pp |
| Import volume | -35% (flat 15% scenario) |
| Pharma tariff (trade-weighted) | 21.28% |
| Drug price pressure | +2.47% to +10.34% |
| Q1 retail burden | 8.40% of budget |
| Q5 retail burden | 5.94% of budget |
| Regressivity ratio | 1.41× |

---

*Built on top of Ignatenko, Macedoni, Lashkaripour & Simonovska (2025)*
