# Python Replication Package

Python implementation of the economic models from:

**"Making America Great Again? The Economic Impacts of Liberation Day Tariffs"**
by Anna Ignatenko, Luca Macedoni, Ahmad Lashkaripour, Ina Simonovska (2025)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline model (Tables 1-3, 8-9)
cd analysis
python main_baseline.py

# Run multi-sector models (Tables 4, 11)
python sub_multisector_baseline.py
python sub_multisector_io.py
python generate_table_4.py
python generate_table_11.py
```

## Replication Status

| Tables | Status | Description |
|--------|--------|-------------|
| 1, 2, 3 | ✅ Exact | Baseline policy & retaliation scenarios |
| 8 | ✅ Exact | Regional trade wars |
| 9 | ✅ Exact | Alternative specifications |
| 4 | ⚠️ Close | Multi-sector welfare (~0.1% diff) |
| 10 | ⚠️ Partial | Deficit framework (2/4 match) |
| 11 | ⚠️ Close | Single-sector exact, multi-sector close |

See `python_output/TABLE_OVERVIEW.md` for detailed comparison.

## Project Structure

```
code_python/
├── analysis/
│   ├── main_baseline.py           # Single-sector baseline (Tables 1-3, 9)
│   ├── main_io.py                 # Input-output model
│   ├── main_regional.py           # Regional trade wars (Table 8)
│   ├── main_deficit.py            # Deficit framework (Table 10)
│   ├── sub_multisector_baseline.py   # Multi-sector K=4 (Tables 4, 11)
│   ├── sub_multisector_io.py      # Multi-sector + IO linkages
│   ├── generate_table_4.py        # Table 4 generator
│   ├── generate_table_11.py       # Table 11 generator
│   ├── print_tables_baseline.py   # LaTeX table output
│   ├── _debug/                    # Debug scripts (development only)
│   └── _experimental/             # Experimental solvers
├── utils/                         # Utility functions
├── config.py                      # Configuration
└── requirements.txt               # Dependencies
```

## Output

Results are saved to `python_output/`:
- `Table_*.tex` - LaTeX formatted tables
- `*_results.npz` - NumPy compressed arrays
- `*.csv` - Parameter exports

## Model Overview

### Single-Sector Baseline (`main_baseline.py`)
- N = 194 countries
- 9 tariff scenarios (USTR, optimal, retaliation variants)
- Produces Tables 1, 2, 3, 9

### Multi-Sector (`sub_multisector_baseline.py`)
- K = 4 sectors (Agriculture, Manufacturing, Mining, Services)
- Sectoral labor reallocation
- Produces Tables 4, 11

### Input-Output (`main_io.py`, `sub_multisector_io.py`)
- Roundabout production structure
- Intermediate input linkages

## Key MATLAB-to-Python Notes

| MATLAB | Python | Notes |
|--------|--------|-------|
| 1-indexed | 0-indexed | `id_US = 185` → `id_US = 184` |
| `repmat()` | `np.tile()` | Array replication |
| `reshape(X,N,N,K)` | `X.reshape((N,N,K), order='F')` | Column-major order |
| `.` operator | `*` | Element-wise multiplication |
| `fsolve()` | `scipy.optimize.root()` | Nonlinear solver |

## Requirements

- Python 3.8+
- NumPy >= 1.20
- SciPy >= 1.7
- Pandas >= 1.3

## Known Differences

1. **Solver algorithms**: Python uses Levenberg-Marquardt; MATLAB uses trust-region-dogleg
2. **Multi-sector results**: Small numerical differences (~0.1-0.5%) in equilibrium solutions
3. **Multi+IO model**: Larger discrepancy (~2%) under investigation

## Contact

- Replication package questions: ahmadlp@gmail.com
- See original paper for model documentation
