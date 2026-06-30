# Liberation Day MCP Server

FastMCP server exposing 6 analysis tools over the GE model results and sector datasets.

## Install

```bash
pip install -r requirements_mcp.txt
```

## Start the MCP server

```bash
# From project root
python -m mcp_server.server

# Or via fastmcp CLI
fastmcp run mcp_server/server.py
```

Server runs on `stdio` by default (standard MCP transport).

## Connect to Claude.ai

1. Open Claude.ai → Settings → Integrations → Add MCP Server
2. Choose **stdio** transport
3. Command: `python`  
   Args: `-m mcp_server.server`  
   Working directory: path to this project root

## Tools

| Tool | Description |
|---|---|
| `get_welfare_results(country, scenario)` | GE welfare/CPI/trade results, filterable by ISO3 country and scenario |
| `get_scenario_comparison(countries, scenarios)` | Pivot welfare across multiple countries × scenarios |
| `get_pharma_supplier_risk(country, hts_code)` | Pharma import exposure + HHI concentration ranked by risk |
| `get_quintile_burden(category)` | Tariff incidence by income quintile (pharma or retail category) |
| `get_manufacturing_shock(naics_code, top_n)` | Sector tariff shocks and gross output by NAICS |
| `run_tariff_scenario(tariff_overrides, countries)` | Partial equilibrium welfare delta for custom tariff rates |

Every tool returns `{data, chart_spec}` — `chart_spec` is a valid Plotly figure dict.

## Scenarios

| Key | Description |
|---|---|
| `ustr_no_retaliation` | USTR Liberation Day tariffs, no retaliation (baseline) |
| `ustr_lump_sum` | USTR tariffs + lump-sum rebate |
| `optimal_tariff` | Optimal US tariffs |
| `ustr_reciprocal_retaliation` | USTR + full reciprocal retaliation |
| `ustr_optimal_retaliation` | USTR + optimal retaliation |
| `flat_15pct` | Custom flat 15% tariff on all US imports |

## AI Analyst Tab

Tab 5 in the Streamlit dashboard (`dashboard/app.py`) provides a chat interface powered by `claude-sonnet-4-6`.
It calls all 6 MCP tools directly and renders Plotly charts inline in chat responses.

Set your API key:
```toml
# .streamlit/secrets.toml
ANTHROPIC_API_KEY = "sk-ant-..."
```
or as an environment variable: `export ANTHROPIC_API_KEY=sk-ant-...`
