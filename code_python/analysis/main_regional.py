"""
Regional trade war analysis.

This script replicates main_regional.m from the MATLAB replication package.
Analyzes regional trade war scenarios (US vs China, US vs EU+China, etc.)

Converted from MATLAB to Python for:
"Making America Great Again? The Economic Impacts of Liberation Day Tariffs"
by Ignatenko, Macedoni, Lashkaripour, Simonovska (2025)
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import get_output_dir
from utils.solver_utils import solve_nu


def balanced_trade_eq_regional(x, data, param):
    """
    System of equations for balanced trade equilibrium (regional version).
    Same as baseline model.
    """
    N, E_i, Y_i, lambda_ji, t_ji, nu, T_i = data.values()
    eps, kappa, psi, phi = param.values()

    # Extract variables
    w_i_h = np.abs(x[0:N])
    E_i_h = np.abs(x[N:2*N])
    L_i_h = np.abs(x[2*N:3*N])

    # Construct 2D matrices
    wi_h_2D = np.tile(w_i_h.reshape(-1, 1), (1, N))
    phi_2D = np.tile(phi.reshape(1, -1), (N, 1))

    # Construct new trade values
    AUX0 = lambda_ji * ((wi_h_2D / (L_i_h.reshape(-1, 1) ** psi)) ** (-eps)) * \
           ((1 + t_ji) ** (-eps * phi_2D))
    AUX1 = np.tile(np.sum(AUX0, axis=0, keepdims=True), (N, 1))
    lambda_ji_new = AUX0 / AUX1

    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h

    # Price index
    P_i_h = ((E_i_h / w_i_h) ** (1 - phi)) * (np.sum(AUX0, axis=0) ** (-1 / eps))

    # New trade flows
    X_ji_new = lambda_ji_new * np.tile(E_i_new.reshape(1, -1), (N, 1)) / (1 + t_ji)
    tariff_rev = np.sum(lambda_ji_new * (t_ji / (1 + t_ji)) * \
                       np.tile(E_i_new.reshape(1, -1), (N, 1)), axis=0)

    # Tax adjustment
    tau_i = tariff_rev / Y_i_new
    tau_i_new = 0
    tau_i_h = (1 - tau_i_new) / (1 - tau_i)

    # Equilibrium conditions
    nu_2D = np.tile(nu.reshape(1, -1), (N, 1))
    ERR1 = np.sum((1 - nu_2D) * X_ji_new, axis=1) + \
           np.sum(nu_2D * X_ji_new, axis=0) - w_i_h * L_i_h * Y_i
    ERR1[N-1] = np.mean((P_i_h - 1) * E_i)

    # ERR2: Total Income = Total Sales
    X_global = np.sum(Y_i)
    X_global_new = np.sum(Y_i_new)
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global) - E_i_new

    # ERR3: Labor supply
    ERR3 = L_i_h - (tau_i_h * w_i_h / P_i_h) ** kappa

    ceq = np.concatenate([ERR1, ERR2, ERR3])

    # Calculate welfare
    delta_i = E_i / (E_i - kappa * (1 - tau_i) * Y_i / (1 + kappa))
    W_i_h = delta_i * (E_i_h / P_i_h) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h)

    # Factual trade flows
    X_ji = lambda_ji * np.tile(E_i.reshape(1, -1), (N, 1))
    D_i = np.sum(X_ji, axis=0) - np.sum(X_ji, axis=1)  # imports - exports
    D_i_new = np.sum(X_ji_new, axis=0) - np.sum(X_ji_new, axis=1)  # imports - exports

    # Calculate changes
    d_welfare = 100 * (W_i_h - 1)
    d_export = 100 * (np.sum(X_ji_new * (1 - np.eye(N)), axis=1) / \
                     np.sum(X_ji * (1 - np.eye(N)), axis=1) - 1)
    d_import = 100 * (np.sum(X_ji_new * (1 - np.eye(N)), axis=0) / \
                     np.sum(X_ji * (1 - np.eye(N)), axis=0) - 1)
    d_employment = 100 * (L_i_h - 1)
    d_CPI = 100 * (P_i_h - 1)
    d_D_i = 100 * ((D_i_new - D_i) / np.abs(D_i))

    results = np.column_stack([d_welfare, d_D_i, d_export, d_import,
                              d_employment, d_CPI, tariff_rev / E_i])

    return ceq, results


def main():
    """Main function to run regional trade war analysis."""
    print("Running regional trade war analysis...")
    print(f"Current directory: {os.getcwd()}")

    # Read trade and GDP data
    base_path = os.path.join(os.path.dirname(__file__), '..', '..')

    data_path = os.path.join(base_path, 'data', 'base_data', 'trade_cepii.csv')
    X_ji = pd.read_csv(data_path, header=0).values  # header=0 to skip header row
    # Convert to float, replacing any non-numeric values with NaN, then replace NaN with 0
    X_ji = pd.DataFrame(X_ji).apply(pd.to_numeric, errors='coerce').fillna(0).values
    N = X_ji.shape[0]

    # Country IDs (convert from MATLAB 1-indexed to Python 0-indexed)
    id_US = 185 - 1
    id_CHN = 34 - 1
    id_CAN = 31 - 1
    id_MEX = 115 - 1
    id_EU = np.array([10, 13, 17, 45, 47, 50, 56, 57, 59, 61, 71, 78, 80, 83, 88,
                      107, 108, 109, 119, 133, 144, 145, 149, 164, 165]) - 1
    id_RoW = np.setdiff1d(np.arange(N), np.concatenate([[id_US, id_CHN], id_EU]))
    non_US = np.setdiff1d(np.arange(N), [id_US])

    # GDP data
    gdp_path = os.path.join(base_path, 'data', 'base_data', 'gdp.csv')
    Y_i = pd.read_csv(gdp_path, header=0).values.flatten()
    Y_i = pd.to_numeric(Y_i, errors='coerce')
    Y_i = Y_i / 1000

    # Calculate trade variables
    tot_exports = np.sum(X_ji, axis=1)
    tot_imports = np.sum(X_ji, axis=0)

    # Solve for nu
    nu_eq = solve_nu(X_ji, Y_i, id_US)
    nu = nu_eq[0] * np.ones(N)
    nu[id_US] = nu_eq[1]

    # Calculate transfers and expenditure
    T = (1 - nu) * (np.sum(X_ji, axis=1) - \
                    np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=0))
    E_i = Y_i + T
    X_ii = E_i - tot_imports
    X_ii[X_ii < 0] = 0
    X_ji = X_ji.copy()
    np.fill_diagonal(X_ji, X_ii)

    # Recalculate after updating diagonal
    # MATLAB line 36: E_i = sum(X_ji,1)' - column sums (imports), then transpose
    # MATLAB line 37: Y_i = sum(...,2) + nu.*sum(X_ji,1)' - row sums + column sums
    E_i = np.sum(X_ji, axis=0)  # sum over axis 0 = column sums (imports)
    Y_i = np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1) + \
          nu * np.sum(X_ji, axis=0)  # First term: axis=1 (row sums), second: axis=0 (column sums)
    T = E_i - Y_i
    lambda_ji = X_ji / np.tile(E_i.reshape(1, -1), (N, 1))

    # Read USTR tariffs
    tariff_path = os.path.join(base_path, 'data', 'base_data', 'tariffs.csv')
    new_ustariff = pd.read_csv(tariff_path, header=0).values.flatten()
    new_ustariff = pd.to_numeric(new_ustariff, errors='coerce')
    t_ji = np.zeros((N, N))
    # Exclude US entry from tariff array to match dimension (N countries in model, but tariff file has N+1 entries)
    new_ustariff_valid = np.delete(new_ustariff, id_US) if len(new_ustariff) > N else new_ustariff
    t_ji[:, id_US] = new_ustariff_valid
    t_ji[:, id_US] = np.maximum(0.1, t_ji[:, id_US])
    t_ji[id_US, id_US] = 0
    tariff_USTR = t_ji.copy()

    # Trade elasticity and parameters
    eps = 4
    kappa = 0.5
    psi = 0.67 / eps

    theta = eps / 0.67
    phi_tilde = (1 + theta) / ((1 - nu) * theta) - (1 / theta) - 1
    Phi = [1 + phi_tilde, 0.5 + phi_tilde]

    # Create arrays to save results
    results = np.zeros((N, 7, 3))

    # Case 1: USTR tariff on China/EU + 10% tariff on others (with reciprocal retaliation)
    print("\n=== Case 1: US trade war with EU & China ===")
    t_ji_new = np.zeros((N, N))
    t_ji_new[non_US, id_US] = 0.1
    t_ji_new[id_US, non_US] = 0.1

    t_ji_new[id_CHN, id_US] = tariff_USTR[id_CHN, id_US]
    t_ji_new[id_EU, id_US] = tariff_USTR[id_EU, id_US]

    t_ji_new[id_US, id_CHN] = tariff_USTR[id_CHN, id_US]
    t_ji_new[id_US, id_EU] = tariff_USTR[id_EU, id_US]
    t_ji_new[id_US, id_US] = 0

    phi = Phi[0]

    data = {
        'N': N, 'E_i': E_i, 'Y_i': Y_i, 'lambda_ji': lambda_ji,
        't_ji': t_ji_new, 'nu': nu, 'T_i': T
    }
    param = {'eps': eps, 'kappa': kappa, 'psi': psi, 'phi': phi}

    x0 = np.ones(3 * N)

    def syst(x):
        ceq, _ = balanced_trade_eq_regional(x, data, param)
        return ceq

    x_fsolve = fsolve(syst, x0, xtol=1e-6, maxfev=50000, factor=0.1)
    _, results[:, :, 0] = balanced_trade_eq_regional(x_fsolve, data, param)

    print(f"  US welfare change: {results[id_US, 0, 0]:.2f}%")
    print(f"  China welfare change: {results[id_CHN, 0, 0]:.2f}%")

    # Case 2: USTR tariff on China + 10% tariff on others (with reciprocal retaliation)
    print("\n=== Case 2: US trade war with China only ===")
    t_ji_new = np.zeros((N, N))
    t_ji_new[non_US, id_US] = 0.1
    t_ji_new[id_US, non_US] = 0.1

    t_ji_new[id_CHN, id_US] = tariff_USTR[id_CHN, id_US]
    t_ji_new[id_US, id_CHN] = tariff_USTR[id_CHN, id_US]
    t_ji_new[id_US, id_US] = 0

    phi = Phi[0]

    data = {
        'N': N, 'E_i': E_i, 'Y_i': Y_i, 'lambda_ji': lambda_ji,
        't_ji': t_ji_new, 'nu': nu, 'T_i': T
    }
    param = {'eps': eps, 'kappa': kappa, 'psi': psi, 'phi': phi}

    x0 = np.ones(3 * N)
    x_fsolve = fsolve(syst, x0, xtol=1e-6, maxfev=50000, factor=0.1)
    _, results[:, :, 1] = balanced_trade_eq_regional(x_fsolve, data, param)

    print(f"  US welfare change: {results[id_US, 0, 1]:.2f}%")
    print(f"  China welfare change: {results[id_CHN, 0, 1]:.2f}%")

    # Case 3: 108% tariff on China + 10% tariff on others (with reciprocal retaliation)
    print("\n=== Case 3: US trade war with China (108% tariff) ===")
    t_ji_new = np.zeros((N, N))
    t_ji_new[non_US, id_US] = 0.1
    t_ji_new[id_US, non_US] = 0.1

    t_ji_new[id_CHN, id_US] = 1.08
    t_ji_new[id_US, id_CHN] = 1.08
    t_ji_new[id_US, id_US] = 0

    phi = Phi[0]

    data = {
        'N': N, 'E_i': E_i, 'Y_i': Y_i, 'lambda_ji': lambda_ji,
        't_ji': t_ji_new, 'nu': nu, 'T_i': T
    }
    param = {'eps': eps, 'kappa': kappa, 'psi': psi, 'phi': phi}

    x0 = np.ones(3 * N)
    x_fsolve = fsolve(syst, x0, xtol=1e-6, maxfev=50000, factor=0.1)
    _, results[:, :, 2] = balanced_trade_eq_regional(x_fsolve, data, param)

    print(f"  US welfare change: {results[id_US, 0, 2]:.2f}%")
    print(f"  China welfare change: {results[id_CHN, 0, 2]:.2f}%")

    # Generate Table 8 (LaTeX)
    print("\nGenerating Table 8...")
    countries_path = os.path.join(base_path, 'data', 'base_data', 'country_labels.csv')
    countries = pd.read_csv(countries_path)
    country_names = countries['iso3'].values

    output_dir = get_output_dir()
    os.makedirs(os.path.join(base_path, output_dir), exist_ok=True)
    output_path = os.path.join(base_path, output_dir, 'Table_8.tex')
    with open(output_path, 'w') as f:
        # Table preamble
        f.write(r'\begin{tabular}{lcccccc}' + '\n')
        f.write(r'\toprule' + '\n')
        f.write(r'\multicolumn{4}{l}{\textbf{Case 1: US trade war with EU \& China}}  \\' + '\n')
        f.write(r'\midrule' + '\n')
        f.write('Country &\n')
        f.write(r'\specialcell{$\Delta$ welfare} &' + '\n')
        f.write(r'\specialcell{$\Delta$ deficit} &' + '\n')
        f.write(r'\specialcell{$\Delta$ employment} &' + '\n')
        f.write(r'\specialcell{$\Delta$ prices} \\' + '\n')
        f.write(r'\midrule' + '\n')

        # Case 1 results
        for i in [id_US, id_CHN]:
            f.write(f'{country_names[i]} & ')
            f.write(f'{results[i, 0, 0]:.2f}\\% & ')
            f.write(f'{results[i, 1, 0]:.1f}\\% & ')
            f.write(f'{results[i, 4, 0]:.2f}\\% & ')
            f.write(f'{results[i, 5, 0]:.1f}\\% \\\\ \n')
            f.write(r' \addlinespace[3pt]' + '\n')

        # EU average
        avg_EU = np.sum(E_i[id_EU].reshape(-1, 1) * results[id_EU, :, 0], axis=0) / np.sum(E_i[id_EU])
        f.write('EU & ')
        f.write(f'{avg_EU[0]:.2f}\\%  & ')
        f.write(f'{avg_EU[1]:.1f}\\% & ')
        f.write(f'{avg_EU[4]:.2f}\\% & ')
        f.write(f'{avg_EU[5]:.1f}\\% \\\\ \n')
        f.write(r' \addlinespace[3pt]' + '\n')

        # RoW average
        avg_RoW = np.sum(E_i[id_RoW].reshape(-1, 1) * results[id_RoW, :, 0], axis=0) / np.sum(E_i[id_RoW])
        f.write('RoW & ')
        f.write(f'{avg_RoW[0]:.2f}\\%  & ')
        f.write(f'{avg_RoW[1]:.1f}\\% & ')
        f.write(f'{avg_RoW[4]:.2f}\\% & ')
        f.write(f'{avg_RoW[5]:.1f}\\% \\\\ \n')

        # Case 2
        f.write(r'\midrule' + '\n')
        f.write(r'\addlinespace[10pt]' + '\n')
        f.write(r'\multicolumn{4}{l}{\textbf{Case 2: US trade war with China}}  \\ ' + '\n')
        f.write(r'\midrule' + '\n')

        for i in [id_US, id_CHN]:
            f.write(f'{country_names[i]} & ')
            f.write(f'{results[i, 0, 1]:.2f}\\% & ')
            f.write(f'{results[i, 1, 1]:.1f}\\% & ')
            f.write(f'{results[i, 4, 1]:.2f}\\% & ')
            f.write(f'{results[i, 5, 1]:.1f}\\% \\\\ \n')
            f.write(r' \addlinespace[3pt]' + '\n')

        avg_EU = np.sum(E_i[id_EU].reshape(-1, 1) * results[id_EU, :, 1], axis=0) / np.sum(E_i[id_EU])
        f.write('EU  & ')
        f.write(f'{avg_EU[0]:.2f}\\%  & ')
        f.write(f'{avg_EU[1]:.1f}\\% & ')
        f.write(f'{avg_EU[4]:.2f}\\% & ')
        f.write(f'{avg_EU[5]:.1f}\\% \\\\ \n')
        f.write(r' \addlinespace[3pt]' + '\n')

        avg_RoW = np.sum(E_i[id_RoW].reshape(-1, 1) * results[id_RoW, :, 1], axis=0) / np.sum(E_i[id_RoW])
        f.write('RoW & ')
        f.write(f'{avg_RoW[0]:.2f}\\%  & ')
        f.write(f'{avg_RoW[1]:.1f}\\% & ')
        f.write(f'{avg_RoW[4]:.2f}\\% & ')
        f.write(f'{avg_RoW[5]:.1f}\\% \\\\ \n')

        # Case 3
        f.write(r'\midrule' + '\n')
        f.write(r'\addlinespace[10pt]' + '\n')
        f.write(r'\multicolumn{4}{l}{\textbf{Case 3: US trade war with China (108\% tariff)}} \\ ' + '\n')
        f.write(r'\midrule' + '\n')

        for i in [id_US, id_CHN]:
            f.write(f'{country_names[i]} & ')
            f.write(f'{results[i, 0, 2]:.2f}\\% & ')
            f.write(f'{results[i, 1, 2]:.1f}\\% & ')
            f.write(f'{results[i, 4, 2]:.2f}\\% & ')
            f.write(f'{results[i, 5, 2]:.1f}\\% \\\\ \n')
            f.write(r' \addlinespace[3pt]' + '\n')

        avg_EU = np.sum(E_i[id_EU].reshape(-1, 1) * results[id_EU, :, 2], axis=0) / np.sum(E_i[id_EU])
        f.write('EU & ')
        f.write(f'{avg_EU[0]:.2f}\\%  & ')
        f.write(f'{avg_EU[1]:.1f}\\% & ')
        f.write(f'{avg_EU[4]:.2f}\\% & ')
        f.write(f'{avg_EU[5]:.1f}\\% \\\\ \n')
        f.write(r' \addlinespace[3pt]' + '\n')

        avg_RoW = np.sum(E_i[id_RoW].reshape(-1, 1) * results[id_RoW, :, 2], axis=0) / np.sum(E_i[id_RoW])
        f.write('RoW& ')
        f.write(f'{avg_RoW[0]:.2f}\\%  & ')
        f.write(f'{avg_RoW[1]:.1f}\\% & ')
        f.write(f'{avg_RoW[4]:.2f}\\% & ')
        f.write(f'{avg_RoW[5]:.1f}\\% \\\\ \n')

        # Table closing
        f.write(r' \bottomrule' + '\n')
        f.write(r'\end{tabular}' + '\n')

    print(f"  Table 8 saved to: {output_path}")
    print("\n=== Regional trade war analysis completed ===")

    return {
        'results': results,
        'id_US': id_US,
        'id_CHN': id_CHN,
        'id_EU': id_EU,
        'id_RoW': id_RoW,
        'E_i': E_i
    }


if __name__ == '__main__':
    results_dict = main()
