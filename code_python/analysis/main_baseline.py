"""
Baseline trade model analysis.

This script replicates main_baseline.m from the MATLAB replication package.
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
from utils.solver_utils import solve_nu
from config import get_output_dir


def balanced_trade_eq(x, data, param, lump_sum=0):
    """
    System of equations for balanced trade equilibrium.

    Parameters:
    -----------
    x : np.ndarray
        Solution vector [w_i_h; E_i_h; L_i_h] (3N x 1)
    data : dict
        Data dictionary containing N, E_i, Y_i, lambda_ji, t_ji, nu, T_i
    param : dict
        Parameter dictionary containing eps, kappa, psi, phi
    lump_sum : int
        Whether to use lump-sum rebate (0 or 1)

    Returns:
    --------
    ceq : np.ndarray
        Residuals of equilibrium conditions
    results : np.ndarray
        Results matrix (N x 7) with welfare, deficit, exports, imports, employment, CPI, tariff revenue
    d_trade : float
        Change in global trade
    """
    N, E_i, Y_i, lambda_ji, t_ji, nu, T_i = data.values()
    eps, kappa, psi, phi = param.values()

    # Extract variables (use abs to avoid complex numbers)
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
    if lump_sum == 0:
        tau_i = tariff_rev / Y_i_new
        tau_i_new = 0
        tau_i_h = (1 - tau_i_new) / (1 - tau_i)
    elif lump_sum == 1:
        tau_i = 0
        tau_i_h = np.ones(N)

    # Equilibrium conditions
    # ERR1: Wage Income = Total Sales net of Taxes
    # MATLAB line 329: sum(...,2) + sum(...,1)' - row sums + column sums
    nu_2D = np.tile(nu.reshape(1, -1), (N, 1))
    ERR1 = np.sum((1 - nu_2D) * X_ji_new, axis=1) + \
           np.sum(nu_2D * X_ji_new, axis=0) - w_i_h * L_i_h * Y_i
    ERR1[N-1] = np.mean((P_i_h - 1) * E_i)  # Replace one excess equation

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
    d_export = 100 * ((np.sum(X_ji_new * (1 - np.eye(N)), axis=1) / Y_i_new) / \
                     (np.sum(X_ji * (1 - np.eye(N)), axis=1) / Y_i) - 1)
    d_import = 100 * ((np.sum(X_ji_new * (1 - np.eye(N)), axis=0) / Y_i_new) / \
                     (np.sum(X_ji * (1 - np.eye(N)), axis=0) / Y_i) - 1)
    d_employment = 100 * (L_i_h - 1)
    d_CPI = 100 * (P_i_h - 1)
    d_D_i = 100 * ((D_i_new - D_i) / np.abs(D_i))

    # Trade change
    trade = X_ji * (1 - np.eye(N))
    trade_new = X_ji_new * (1 + t_ji) * (1 - np.eye(N))
    d_trade = 100 * ((np.sum(trade_new) / np.sum(trade)) / \
                    (np.sum(Y_i_new) / np.sum(Y_i)) - 1)

    results = np.column_stack([d_welfare, d_D_i, d_export, d_import,
                              d_employment, d_CPI, tariff_rev / E_i])

    return ceq, results, d_trade


def main():
    """Main function to run baseline analysis."""
    print("Running baseline analysis...")
    print(f"Current directory: {os.getcwd()}")

    # Read trade and GDP data
    # Note: Adjust paths relative to where script is run from
    base_path = os.path.join(os.path.dirname(__file__), '..', '..')

    data_path = os.path.join(base_path, 'data', 'base_data', 'trade_cepii.csv')
    X_ji = pd.read_csv(data_path, header=0).values  # header=0 to skip header row
    # Convert to float, replacing any non-numeric values with NaN, then replace NaN with 0
    X_ji = pd.DataFrame(X_ji).apply(pd.to_numeric, errors='coerce').fillna(0).values
    N = X_ji.shape[0]

    # Country IDs (convert from MATLAB 1-indexed to Python 0-indexed)
    id_US = 185 - 1  # MATLAB was 185, Python is 184
    id_CAN = 31 - 1
    id_MEX = 115 - 1
    id_CHN = 34 - 1
    id_EU = np.array([10, 13, 17, 45, 47, 50, 56, 57, 59, 61, 71, 78, 80, 83, 88,
                      107, 108, 109, 119, 133, 144, 145, 149, 164, 165]) - 1
    id_RoW = np.setdiff1d(np.arange(N), np.concatenate([[id_US, id_CHN], id_EU]))
    non_US = np.setdiff1d(np.arange(N), [id_US])

    # GDP data
    gdp_path = os.path.join(base_path, 'data', 'base_data', 'gdp.csv')
    Y_i = pd.read_csv(gdp_path, header=0).values.flatten()
    Y_i = pd.to_numeric(Y_i, errors='coerce')
    Y_i = Y_i / 1000  # Trade flows are in 1000s of USD

    # Calculate trade variables
    tot_exports = np.sum(X_ji, axis=1)
    tot_imports = np.sum(X_ji, axis=0)

    # Solve for nu
    nu_eq = solve_nu(X_ji, Y_i, id_US)
    nu = nu_eq[0] * np.ones(N)
    nu[id_US] = nu_eq[1]

    # Calculate transfers and expenditure
    # MATLAB: T = (1-nu).*(sum(X_ji,1)' - sum(repmat((1-nu)',N,1).*X_ji,2));
    # sum(X_ji,1)' = column sums (imports) = axis=0
    # sum(...X_ji,2) = row sums (exports) = axis=1
    T = (1 - nu) * (np.sum(X_ji, axis=0) -
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

    # Read US tariffs
    tariff_path = os.path.join(base_path, 'data', 'base_data', 'tariffs.csv')
    new_ustariff = pd.read_csv(tariff_path, header=0).values.flatten()
    new_ustariff = pd.to_numeric(new_ustariff, errors='coerce')
    t_ji = np.zeros((N, N))
    t_ji[:, id_US] = new_ustariff
    t_ji[:, id_US] = np.maximum(0.1, t_ji[:, id_US])
    t_ji[id_US, id_US] = 0

    # Trade elasticity and parameters
    eps = 4
    kappa = 0.5
    psi = 0.67 / eps

    theta = eps / 0.67
    phi_tilde = (1 + theta) / ((1 - nu) * theta) - (1 / theta) - 1

    Phi = [1 + phi_tilde, 0.5 + phi_tilde, 0.25 + phi_tilde]

    # Create arrays to save results
    results = np.zeros((N, 7, 9))
    revenue = np.zeros(9)  # Match the number of scenarios
    d_trade = np.zeros(9)
    d_employment = np.zeros(9)

    # Baseline Analysis
    print("\n=== Running Baseline Analysis ===")
    for i in range(2):
        print(f"\nCase {i+1}:")
        if i == 0:
            phi = Phi[i]
        elif i == 1:
            phi = Phi[i+1]

        data = {
            'N': N, 'E_i': E_i, 'Y_i': Y_i, 'lambda_ji': lambda_ji,
            't_ji': t_ji, 'nu': nu, 'T_i': T
        }
        param = {'eps': eps, 'kappa': kappa, 'psi': psi, 'phi': phi}
        lump_sum = 0

        x0 = np.ones(3 * N)

        def syst(x):
            ceq, _, _ = balanced_trade_eq(x, data, param, lump_sum)
            return ceq

        x_fsolve = fsolve(syst, x0, xtol=1e-6, maxfev=50000, factor=0.1)
        _, results[:, :, i], d_trade[i] = balanced_trade_eq(x_fsolve, data, param, lump_sum)

        revenue[i] = results[id_US, 6, i]
        d_employment[i] = np.sum(results[:, 4, i] * Y_i) / np.sum(Y_i)
        print(f"  US welfare change: {results[id_US, 0, i]:.2f}%")

    # Eaton-Kortum Specification
    print("\nRunning Eaton-Kortum specification...")
    Y_i_EK = np.sum(X_ji, axis=1)  # sum over axis 1 = row sums (exports)
    T_EK = E_i - Y_i_EK
    phi_EK = np.ones(N)
    nu_EK = np.zeros(N)

    data = {
        'N': N, 'E_i': E_i, 'Y_i': Y_i_EK, 'lambda_ji': lambda_ji,
        't_ji': t_ji, 'nu': nu_EK, 'T_i': T_EK
    }
    param = {'eps': eps, 'kappa': kappa, 'psi': psi, 'phi': phi_EK}

    x0 = np.ones(3 * N)
    def syst(x):
        ceq, _, _ = balanced_trade_eq(x, data, param, 0)
        return ceq

    x_fsolve = fsolve(syst, x0, xtol=1e-6, maxfev=50000, factor=0.1)
    _, results[:, :, 2], d_trade[2] = balanced_trade_eq(x_fsolve, data, param, 0)

    revenue[2] = results[id_US, 6, 2]
    d_employment[2] = np.sum(results[:, 4, 2] * Y_i) / np.sum(Y_i)
    print(f"  US welfare change: {results[id_US, 0, 2]:.2f}%")

    # Lump-sum rebate of tariff revenue
    print("\nRunning lump-sum rebate scenario...")
    phi = Phi[0]
    data = {
        'N': N, 'E_i': E_i, 'Y_i': Y_i, 'lambda_ji': lambda_ji,
        't_ji': t_ji, 'nu': nu, 'T_i': T
    }
    param = {'eps': eps, 'kappa': kappa, 'psi': psi, 'phi': phi}
    lump_sum = 1

    x0 = np.ones(3 * N)
    def syst(x):
        ceq, _, _ = balanced_trade_eq(x, data, param, lump_sum)
        return ceq

    x_fsolve = fsolve(syst, x0, xtol=1e-6, maxfev=50000, factor=0.1)
    _, results[:, :, 7], _ = balanced_trade_eq(x_fsolve, data, param, lump_sum)
    print(f"  US welfare change: {results[id_US, 0, 7]:.2f}%")

    # Higher trade elasticity
    print("\nRunning higher trade elasticity scenario...")
    param = {'eps': 2*eps, 'kappa': kappa, 'psi': psi, 'phi': phi}
    lump_sum = 0

    x0 = np.ones(3 * N)
    def syst(x):
        ceq, _, _ = balanced_trade_eq(x, data, param, lump_sum)
        return ceq

    x_fsolve = fsolve(syst, x0, xtol=1e-6, maxfev=50000, factor=0.1)
    _, results[:, :, 8], _ = balanced_trade_eq(x_fsolve, data, param, lump_sum)
    print(f"  US welfare change: {results[id_US, 0, 8]:.2f}%")

    # Optimal Tariff without retaliation
    print("\nRunning optimal tariff (no retaliation)...")
    t_ji_new = np.zeros((N, N))
    phi = Phi[0]

    delta = (np.sum(X_ji * np.tile((1 - nu).reshape(1, -1), (N, 1)) * \
                   (1 - np.eye(N)) * (1 - lambda_ji), axis=1)) / \
            ((1 - nu) * (E_i - np.diag(X_ji)))
    t_ji_new[:, id_US] = 1 / ((1 + delta[id_US] * eps) * phi[id_US] - 1)
    t_ji_new[id_US, id_US] = 0

    data = {
        'N': N, 'E_i': E_i, 'Y_i': Y_i, 'lambda_ji': lambda_ji,
        't_ji': t_ji_new, 'nu': nu, 'T_i': T
    }
    param = {'eps': eps, 'kappa': kappa, 'psi': psi, 'phi': phi}
    lump_sum = 0

    x0 = np.ones(3 * N)
    def syst(x):
        ceq, _, _ = balanced_trade_eq(x, data, param, lump_sum)
        return ceq

    x_fsolve = fsolve(syst, x0, xtol=1e-6, maxfev=50000, factor=0.1)
    _, results[:, :, 3], d_trade[3] = balanced_trade_eq(x_fsolve, data, param, lump_sum)

    revenue[3] = results[id_US, 6, 3]
    d_employment[3] = np.sum(results[:, 4, 3] * Y_i) / np.sum(Y_i)
    print(f"  US welfare change: {results[id_US, 0, 3]:.2f}%")

    # Liberation Tariffs with optimal retaliation
    print("\nRunning Liberation tariffs with optimal retaliation...")
    t_ji_new = t_ji.copy()
    phi = Phi[0]

    AggI = np.zeros((2, N))
    AggI[0, :] = 1
    AggI[0, id_US] = 0
    AggI[1, id_US] = 1
    X = AggI @ X_ji @ AggI.T
    Y = AggI @ Y_i
    lambda_agg = X / np.tile(np.sum(X, axis=0, keepdims=True), (2, 1))
    delta = (np.sum(X * np.tile((1 - nu_eq).reshape(1, -1), (2, 1)) * \
                   (1 - np.eye(2)) * (1 - lambda_agg), axis=0)) / \
            ((1 - nu_eq) * (Y - np.diag(X)))

    t_ji_new[id_US, :] = 1 / ((1 + delta[0] * eps) * phi - 1)
    t_ji_new[id_US, id_US] = 0

    data = {
        'N': N, 'E_i': E_i, 'Y_i': Y_i, 'lambda_ji': lambda_ji,
        't_ji': t_ji_new, 'nu': nu, 'T_i': T
    }
    param = {'eps': eps, 'kappa': kappa, 'psi': psi, 'phi': phi}

    x0 = np.ones(3 * N)
    def syst(x):
        ceq, _, _ = balanced_trade_eq(x, data, param, lump_sum)
        return ceq

    x_fsolve = fsolve(syst, x0, xtol=1e-6, maxfev=50000, factor=0.1)
    _, results[:, :, 4], d_trade[4] = balanced_trade_eq(x_fsolve, data, param, lump_sum)

    revenue[4] = results[id_US, 6, 4]
    d_employment[4] = np.sum(results[:, 4, 4] * Y_i) / np.sum(Y_i)
    print(f"  US welfare change: {results[id_US, 0, 4]:.2f}%")

    # Liberation Tariffs with reciprocal retaliation
    print("\nRunning Liberation tariffs with reciprocal retaliation...")
    t_ji_new = t_ji.copy()
    phi = Phi[0]

    t_ji_new[id_US, :] = t_ji_new[:, id_US]
    t_ji_new[id_US, id_US] = 0

    data = {
        'N': N, 'E_i': E_i, 'Y_i': Y_i, 'lambda_ji': lambda_ji,
        't_ji': t_ji_new, 'nu': nu, 'T_i': T
    }
    param = {'eps': eps, 'kappa': kappa, 'psi': psi, 'phi': phi}

    x0 = np.ones(3 * N)
    def syst(x):
        ceq, _, _ = balanced_trade_eq(x, data, param, lump_sum)
        return ceq

    x_fsolve = fsolve(syst, x0, xtol=1e-6, maxfev=50000, factor=0.1)
    _, results[:, :, 5], d_trade[5] = balanced_trade_eq(x_fsolve, data, param, lump_sum)

    revenue[5] = results[id_US, 6, 5]
    d_employment[5] = np.sum(results[:, 4, 5] * Y_i) / np.sum(Y_i)
    print(f"  US welfare change: {results[id_US, 0, 5]:.2f}%")

    # Optimal Tariff with optimal retaliation
    print("\nRunning optimal tariff with optimal retaliation...")
    t_ji_new = np.zeros((N, N))
    phi = Phi[0]

    delta = (np.sum(X_ji * np.tile((1 - nu).reshape(1, -1), (N, 1)) * \
                   (1 - np.eye(N)) * (1 - lambda_ji), axis=1)) / \
            ((1 - nu) * (Y_i - np.diag(X_ji)))
    t_ji_new[:, id_US] = 1 / ((1 + delta[id_US] * eps) * phi[id_US] - 1)
    t_ji_new[id_US, id_US] = 0

    AggI = np.zeros((2, N))
    AggI[0, :] = 1
    AggI[0, id_US] = 0
    AggI[1, id_US] = 1
    X = AggI @ X_ji @ AggI.T
    Y = AggI @ Y_i
    lambda_agg = X / np.tile(np.sum(X, axis=0, keepdims=True), (2, 1))
    delta = (np.sum(X * np.tile((1 - nu_eq).reshape(1, -1), (2, 1)) * \
                   (1 - np.eye(2)) * (1 - lambda_agg), axis=0)) / \
            ((1 - nu_eq) * (Y - np.diag(X)))

    t_ji_new[id_US, :] = 1 / ((1 + delta[0] * eps) * phi - 1)
    t_ji_new[id_US, id_US] = 0

    data = {
        'N': N, 'E_i': E_i, 'Y_i': Y_i, 'lambda_ji': lambda_ji,
        't_ji': t_ji_new, 'nu': nu, 'T_i': T
    }
    param = {'eps': eps, 'kappa': kappa, 'psi': psi, 'phi': phi}

    x0 = np.ones(3 * N)
    def syst(x):
        ceq, _, _ = balanced_trade_eq(x, data, param, lump_sum)
        return ceq

    x_fsolve = fsolve(syst, x0, xtol=1e-6, maxfev=50000, factor=0.1)
    _, results[:, :, 6], d_trade[6] = balanced_trade_eq(x_fsolve, data, param, lump_sum)

    d_employment[6] = np.sum(results[:, 4, 6] * Y_i) / np.sum(Y_i)
    print(f"  US welfare change: {results[id_US, 0, 6]:.2f}%")

    # Save results for map
    countries_path = os.path.join(base_path, 'data', 'base_data', 'country_labels.csv')
    countries = pd.read_csv(countries_path)
    country_names = countries['iso3'].values

    Data_base = results[:, :, 0]
    # Create DataFrame with all 7 value columns
    Tab = pd.DataFrame({'Country': country_names})
    for i in range(7):
        Tab[f'Value_{i+1}'] = Data_base[:, i]
    output_dir = get_output_dir()
    os.makedirs(os.path.join(base_path, output_dir), exist_ok=True)
    output_path = os.path.join(base_path, output_dir, 'output_map.csv')
    Tab.to_csv(output_path, index=False)

    Data_retal = results[:, :, 4]
    # Create DataFrame with all 7 value columns
    Tab = pd.DataFrame({'Country': country_names})
    for i in range(7):
        Tab[f'Value_{i+1}'] = Data_retal[:, i]
    output_path = os.path.join(base_path, output_dir, 'output_map_retal.csv')
    Tab.to_csv(output_path, index=False)

    # Save parameters for multi-sector models
    print("\nSaving parameters for multi-sector models...")
    pd.DataFrame({'phi': Phi[1]}).to_csv(os.path.join(base_path, output_dir, 'phi_values.csv'), index=False)
    pd.DataFrame({'nu': nu}).to_csv(os.path.join(base_path, output_dir, 'nu_values.csv'), index=False)
    pd.DataFrame({'Y_i': Y_i}).to_csv(os.path.join(base_path, output_dir, 'Y_i_baseline.csv'), index=False)

    # Save baseline results (d_trade, d_employment) for Table 11
    np.savez(os.path.join(base_path, output_dir, 'baseline_results.npz'),
             results=results,
             d_trade=d_trade,
             d_employment=d_employment,
             revenue=revenue,
             Y_i=Y_i,
             E_i=E_i,
             id_US=id_US)
    print(f"  - Saved: phi_values.csv, nu_values.csv, Y_i_baseline.csv")
    print(f"  - Saved: baseline_results.npz")

    # Generate LaTeX tables
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from print_tables_baseline import generate_all_tables

    results_dict = {
        'results': results,
        'revenue': revenue,
        'd_trade': d_trade,
        'd_employment': d_employment,
        'Y_i': Y_i,
        'E_i': E_i,
        'Phi': Phi,
        'nu': nu,
        'id_US': id_US,
        'id_EU': id_EU,
        'id_CHN': id_CHN,
        'id_RoW': id_RoW,
        'non_US': non_US,
        'country_names': country_names,
        'N': N
    }

    generate_all_tables(results_dict, base_path=base_path)

    print("\n=== Baseline analysis completed ===")
    print(f"Results saved to: {os.path.join(base_path, output_dir)}")

    # Return results for further use
    return results_dict


if __name__ == '__main__':
    results_dict = main()
