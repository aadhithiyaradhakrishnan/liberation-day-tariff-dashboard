"""
Alternative deficit framework analysis.

This script replicates main_deficit.m from the MATLAB replication package.
Analyzes different deficit closure assumptions (fixed vs balanced trade).

Converted from MATLAB to Python for:
"Making America Great Again? The Economic Impacts of Liberation Day Tariffs"
by Ignatenko, Macedoni, Lashkaripour, Simonovska (2025)
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve, least_squares, root
import sys
import os
import warnings

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import get_output_dir
from utils.solver_utils import solve_nu


def robust_solve(func, x0, methods=['hybr', 'lm'], n_restarts=2):
    """
    Try multiple solver methods and initial guesses, return best solution.
    This helps find the global minimum similar to MATLAB's trust-region-dogleg.
    """
    best_solution = None
    best_residual = np.inf

    for method in methods:
        for restart in range(n_restarts):
            # Perturb initial guess slightly for restarts > 0
            if restart == 0:
                x_init = x0.copy()
            else:
                np.random.seed(restart * 42)
                x_init = x0 * (1 + 0.05 * np.random.randn(len(x0)))
                x_init = np.maximum(x_init, 0.1)  # Keep positive

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if method == 'lm':
                        sol = root(func, x_init, method=method, tol=1e-10,
                                   options={'maxiter': 5000})
                    else:
                        sol = root(func, x_init, method=method, tol=1e-10,
                                   options={'maxfev': 50000})

                    residual = np.sum(func(sol.x)**2)

                    if residual < best_residual:
                        best_residual = residual
                        best_solution = sol.x
            except Exception:
                continue

    if best_solution is None:
        # Fallback to fsolve
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_solution = fsolve(func, x0, xtol=1e-10, maxfev=100000)

    return best_solution


def balanced_trade_eq_deficit(x, data, param, scale_equations=False):
    """
    System of equations for balanced trade equilibrium (deficit version).
    Same structure as baseline model.

    If scale_equations=True, normalizes ERR2 to have similar magnitude as ERR1/ERR3.
    This helps with solver convergence for Cases 2 & 4.
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

    # Scale ERR2 if requested (helps Cases 2 & 4 converge)
    if scale_equations:
        # Normalize ERR2 by E_i to match magnitude of ERR1/ERR3
        ERR2 = ERR2 / E_i

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
    d_export = 100 * ((np.sum(X_ji_new * (1 - np.eye(N)), axis=1) / Y_i_new) / \
                     (np.sum(X_ji * (1 - np.eye(N)), axis=1) / Y_i) - 1)
    d_import = 100 * ((np.sum(X_ji_new * (1 - np.eye(N)), axis=0) / Y_i_new) / \
                     (np.sum(X_ji * (1 - np.eye(N)), axis=0) / Y_i) - 1)
    d_employment = 100 * (L_i_h - 1)
    d_CPI = 100 * (P_i_h - 1)
    d_D_i = 100 * ((D_i_new - D_i) / np.abs(D_i))

    results = np.column_stack([d_welfare, d_D_i, d_export, d_import,
                              d_employment, d_CPI, tariff_rev / E_i])

    return ceq, results


def main():
    """Main function to run deficit framework analysis."""
    print("Running deficit framework analysis...")
    print(f"Current directory: {os.getcwd()}")

    # Read trade and GDP data
    base_path = os.path.join(os.path.dirname(__file__), '..', '..')

    data_path = os.path.join(base_path, 'data', 'base_data', 'trade_cepii.csv')
    X_ji = pd.read_csv(data_path, header=0).values  # header=0 to skip header row
    # Convert to float, replacing any non-numeric values with NaN, then replace NaN with 0
    X_ji = pd.DataFrame(X_ji).apply(pd.to_numeric, errors='coerce').fillna(0).values
    N = X_ji.shape[0]

    id_US = 185 - 1

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
    T = (1 - nu) * (np.sum(X_ji, axis=0) - \
                    np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1))
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

    # Trade elasticity and parameters
    eps = 4
    kappa = 0.5
    psi = 0.67 / eps
    theta = eps / 0.67
    phi = (1 + theta) / ((1 - nu) * theta) - (1 / theta)

    # Create arrays to save results
    results = np.zeros((N, 7, 4))

    # Case 1: Fixed Deficit (Dekle et al., 2008)
    print("\n=== Case 1: Fixed deficit (Dekle et al., 2008) ===")
    Y_i_EK = np.sum(X_ji, axis=1)  # sum over axis 1 = row sums (exports), matching MATLAB line 57
    T_EK = E_i - Y_i_EK
    nu_EK = np.zeros(N)  # vector of zeros (EK model has no intermediates)
    phi_EK = np.ones(N)  # vector of ones (EK model has full passthrough)
    t_ji_new = t_ji.copy()

    data = {
        'N': N, 'E_i': E_i, 'Y_i': Y_i_EK, 'lambda_ji': lambda_ji,
        't_ji': t_ji_new, 'nu': nu_EK, 'T_i': T_EK
    }
    param = {'eps': eps, 'kappa': kappa, 'psi': psi, 'phi': phi_EK}

    x0 = np.ones(3 * N)

    def syst(x):
        ceq, _ = balanced_trade_eq_deficit(x, data, param)
        return ceq

    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000, factor=0.1)
    _, results[:, :, 0] = balanced_trade_eq_deficit(x_fsolve, data, param)

    print(f"  US welfare change: {results[id_US, 0, 0]:.2f}%")

    # Case 2: Zero Deficit (Ossa, 2014)
    # MATLAB uses default trust-region-dogleg algorithm for this case
    print("\n=== Case 2: Zero deficit (Ossa, 2014) ===")

    # Balance trade with the US
    T_i_new = T - (X_ji[id_US, :] - X_ji[:, id_US])
    T_i_new[id_US] = 0

    # First solve without tariffs (baseline)
    data2a = {
        'N': N, 'E_i': E_i, 'Y_i': Y_i, 'lambda_ji': lambda_ji,
        't_ji': np.zeros((N, N)), 'nu': nu, 'T_i': T_i_new
    }
    param2 = {'eps': eps, 'kappa': kappa, 'psi': psi, 'phi': phi}

    def syst2a(x):
        ceq, _ = balanced_trade_eq_deficit(x, data2a, param2, scale_equations=False)
        return ceq

    x0 = np.ones(3 * N)
    # Use robust multi-method solver
    print("    Solving baseline (no tariffs)...")
    x_baseline = robust_solve(syst2a, x0)
    _, temp_a = balanced_trade_eq_deficit(x_baseline, data2a, param2)

    # Now solve with tariffs
    t_ji_new = t_ji.copy()
    data2b = {
        'N': N, 'E_i': E_i, 'Y_i': Y_i, 'lambda_ji': lambda_ji,
        't_ji': t_ji_new, 'nu': nu, 'T_i': T_i_new
    }

    def syst2b(x):
        ceq, _ = balanced_trade_eq_deficit(x, data2b, param2, scale_equations=False)
        return ceq

    # Use robust solver with baseline as initial guess
    print("    Solving with tariffs...")
    x_fsolve = robust_solve(syst2b, x_baseline)
    _, temp_b = balanced_trade_eq_deficit(x_fsolve, data2b, param2)

    results[:, :, 1] = temp_b - temp_a

    print(f"  US welfare change: {results[id_US, 0, 1]:.2f}%")

    # Case 3: Fixed Deficit + Retaliation
    print("\n=== Case 3: Fixed deficit + retaliation ===")
    Y_i_EK = np.sum(X_ji, axis=1)  # sum over axis 1 = row sums (exports), matching MATLAB line 107
    T_EK = E_i - Y_i_EK

    t_ji_new = t_ji.copy()
    t_ji_new[id_US, :] = 1 / ((1 + eps) * phi_EK - 1)
    t_ji_new[id_US, id_US] = 0

    data = {
        'N': N, 'E_i': E_i, 'Y_i': Y_i_EK, 'lambda_ji': lambda_ji,
        't_ji': t_ji_new, 'nu': nu_EK, 'T_i': T_EK
    }
    param = {'eps': eps, 'kappa': kappa, 'psi': psi, 'phi': phi_EK}

    x0 = np.ones(3 * N)
    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000, factor=0.1)
    _, results[:, :, 2] = balanced_trade_eq_deficit(x_fsolve, data, param)

    print(f"  US welfare change: {results[id_US, 0, 2]:.2f}%")

    # Case 4: Zero Deficit + Retaliation
    # MATLAB uses default trust-region-dogleg algorithm for this case
    print("\n=== Case 4: Zero deficit + retaliation ===")
    T_i_new = T - (X_ji[id_US, :] - X_ji[:, id_US])
    T_i_new[id_US] = 0

    # Baseline without tariffs
    data4a = {
        'N': N, 'E_i': E_i, 'Y_i': Y_i, 'lambda_ji': lambda_ji,
        't_ji': np.zeros((N, N)), 'nu': nu, 'T_i': T_i_new
    }
    param4 = {'eps': eps, 'kappa': kappa, 'psi': psi, 'phi': phi}

    def syst4a(x):
        ceq, _ = balanced_trade_eq_deficit(x, data4a, param4, scale_equations=False)
        return ceq

    x0 = np.ones(3 * N)
    # Use robust multi-method solver
    print("    Solving baseline (no tariffs)...")
    x_baseline = robust_solve(syst4a, x0)
    _, temp_a = balanced_trade_eq_deficit(x_baseline, data4a, param4)

    # With tariffs and retaliation
    t_ji_new = t_ji.copy()
    t_ji_new[id_US, :] = 1 / ((1 + eps) * phi[id_US] - 1)
    t_ji_new[id_US, id_US] = 0

    data4b = {
        'N': N, 'E_i': E_i, 'Y_i': Y_i, 'lambda_ji': lambda_ji,
        't_ji': t_ji_new, 'nu': nu, 'T_i': T_i_new
    }

    def syst4b(x):
        ceq, _ = balanced_trade_eq_deficit(x, data4b, param4, scale_equations=False)
        return ceq

    print("    Solving with tariffs + retaliation...")
    x_fsolve = robust_solve(syst4b, x_baseline)
    _, temp_b = balanced_trade_eq_deficit(x_fsolve, data4b, param4)

    results[:, :, 3] = temp_b - temp_a

    print(f"  US welfare change: {results[id_US, 0, 3]:.2f}%")

    # Generate Table 10
    print("\nGenerating Table 10...")
    countries_path = os.path.join(base_path, 'data', 'base_data', 'country_labels.csv')
    countries = pd.read_csv(countries_path)
    country_names = countries['iso3'].values

    output_dir = get_output_dir()
    os.makedirs(os.path.join(base_path, output_dir), exist_ok=True)
    output_path = os.path.join(base_path, output_dir, 'Table_10.tex')
    with open(output_path, 'w') as f:
        f.write(r'\begin{tabular}{lccccccc}' + '\n')
        f.write(r'\toprule' + '\n')
        f.write(r'\multicolumn{6}{l}{\textbf{(1) Pre-retaliation: fixed transfers to global GDP (Dekle et al., 2008) }}  \\' + '\n')
        f.write(r'\midrule' + '\n')
        f.write('Country &\n')
        f.write(r'\specialcell{$\Delta$ welfare} &' + '\n')
        f.write(r'\specialcell{$\Delta$ $\frac{\textrm{exports}}{\textrm{GDP}}$} & ' + '\n')
        f.write(r'\specialcell{$\Delta$ $\frac{\textrm{imports}}{\textrm{GDP}}$} &' + '\n')
        f.write(r'\specialcell{$\Delta$ employment} &' + '\n')
        f.write(r'\specialcell{$\Delta$ prices} \\' + '\n')
        f.write(r'\midrule' + '\n')

        # Case 1
        f.write(f'{country_names[id_US]} & ')
        f.write(f'{results[id_US, 0, 0]:.2f}\\% & ')
        f.write(f'{results[id_US, 2, 0]:.1f}\\% &')
        f.write(f'{results[id_US, 3, 0]:.1f}\\% & ')
        f.write(f'{results[id_US, 4, 0]:.2f}\\% & ')
        f.write(f'{results[id_US, 5, 0]:.1f}\\% \\\\ \n')

        f.write(r' \addlinespace[3pt]' + '\n')
        non_us = np.setdiff1d(np.arange(N), [id_US])
        avg_non_US = np.sum(E_i[non_us].reshape(-1, 1) * results[non_us, :, 0], axis=0) / np.sum(E_i[non_us])
        avg_non_US[0] = np.mean(results[non_us, 0, 0])

        f.write('non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.2f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\ \n')

        # Case 2
        f.write(r'\midrule' + '\n')
        f.write(r'\addlinespace[10pt]' + '\n')
        f.write(r'\multicolumn{6}{l}{\textbf{(2) Pre-retaliation: balanced trade (Ossa, 2014) }} \\ ' + '\n')
        f.write(r'\midrule' + '\n')

        f.write(f'{country_names[id_US]} & ')
        f.write(f'{results[id_US, 0, 1]:.2f}\\% & ')
        f.write(f'{results[id_US, 2, 1]:.1f}\\% &')
        f.write(f'{results[id_US, 3, 1]:.1f}\\% & ')
        f.write(f'{results[id_US, 4, 1]:.2f}\\% & ')
        f.write(f'{results[id_US, 5, 1]:.1f}\\% \\\\ \n')

        f.write(r' \addlinespace[3pt]' + '\n')
        avg_non_US = np.sum(E_i[non_us].reshape(-1, 1) * results[non_us, :, 1], axis=0) / np.sum(E_i[non_us])
        avg_non_US[0] = np.mean(results[non_us, 0, 1])

        f.write('non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.2f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\ \n')

        # Case 3
        f.write(r'\midrule' + '\n')
        f.write(r'\addlinespace[25pt]' + '\n')
        f.write(r'\multicolumn{6}{l}{\textbf{(3) Post-retaliation: fixed transfers to global GDP (Dekle et al., 2008) }} \\ ' + '\n')
        f.write(r'\midrule' + '\n')

        f.write(f'{country_names[id_US]} & ')
        f.write(f'{results[id_US, 0, 2]:.2f}\\% & ')
        f.write(f'{results[id_US, 2, 2]:.1f}\\% &')
        f.write(f'{results[id_US, 3, 2]:.1f}\\% & ')
        f.write(f'{results[id_US, 4, 2]:.2f}\\% & ')
        f.write(f'{results[id_US, 5, 2]:.1f}\\% \\\\ \n')

        f.write(r' \addlinespace[3pt]' + '\n')
        avg_non_US = np.sum(E_i[non_us].reshape(-1, 1) * results[non_us, :, 2], axis=0) / np.sum(E_i[non_us])
        avg_non_US[0] = np.mean(results[non_us, 0, 2])

        f.write('non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.2f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\ \n')

        # Case 4
        f.write(r'\midrule' + '\n')
        f.write(r'\addlinespace[10pt]' + '\n')
        f.write(r'\multicolumn{6}{l}{\textbf{(4) Post-retaliation: balanced trade (Ossa, 2014) }} \\ ' + '\n')
        f.write(r'\midrule' + '\n')

        f.write(f'{country_names[id_US]} & ')
        f.write(f'{results[id_US, 0, 3]:.2f}\\% & ')
        f.write(f'{results[id_US, 2, 3]:.1f}\\% &')
        f.write(f'{results[id_US, 3, 3]:.1f}\\% & ')
        f.write(f'{results[id_US, 4, 3]:.2f}\\% & ')
        f.write(f'{results[id_US, 5, 3]:.1f}\\% \\\\ \n')

        f.write(r' \addlinespace[3pt]' + '\n')
        avg_non_US = np.sum(E_i[non_us].reshape(-1, 1) * results[non_us, :, 3], axis=0) / np.sum(E_i[non_us])
        avg_non_US[0] = np.mean(results[non_us, 0, 2])  # Note: MATLAB uses case 3 avg for case 4

        f.write('non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.2f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\ \n')

        f.write(r' \bottomrule' + '\n')
        f.write(r'\end{tabular}' + '\n')

    print(f"  Table 10 saved to: {output_path}")
    print("\n=== Deficit framework analysis completed ===")

    return {
        'results': results,
        'id_US': id_US,
        'E_i': E_i
    }


if __name__ == '__main__':
    results_dict = main()
