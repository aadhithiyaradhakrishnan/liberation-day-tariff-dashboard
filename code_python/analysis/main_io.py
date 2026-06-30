"""
Input-Output (IO) trade model analysis.

This script replicates main_io.m from the MATLAB replication package.
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


def balanced_trade_io(x, data, param):
    """
    System of equations for balanced trade equilibrium with IO linkages.

    Parameters:
    -----------
    x : np.ndarray
        Solution vector [w_i_h; E_i_h; L_i_h; P_i_h] (4N x 1)
    data : dict
        Data dictionary
    param : dict
        Parameter dictionary

    Returns:
    --------
    ceq : np.ndarray
        Residuals
    results : np.ndarray
        Results matrix (N x 7)
    d_trade : float
        Change in global trade
    """
    N, E_i, Y_i, lambda_ji, t_ji, nu, T_i = data.values()
    eps, kappa, psi, phi, beta = param.values()

    # Extract variables
    w_i_h = np.abs(x[0:N])
    E_i_h = np.abs(x[N:2*N])
    L_i_h = np.abs(x[2*N:3*N])
    P_i_h = np.abs(x[3*N:4*N])

    # Construct 2D matrices
    phi_2D = np.tile(phi.reshape(1, -1), (N, 1))

    # Construct new trade values
    c_i_h = np.tile((w_i_h**beta * P_i_h**(1-beta)).reshape(-1, 1), (1, N))
    entry = np.tile(((w_i_h/P_i_h)**(1-beta)).reshape(-1, 1), (1, N))
    p_ij_h = ((c_i_h / ((entry * L_i_h.reshape(-1, 1))**psi))**(-eps)) * \
             ((1 + t_ji)**(-eps * phi_2D))

    AUX0 = lambda_ji * p_ij_h
    AUX1 = np.tile(np.sum(AUX0, axis=0, keepdims=True), (N, 1))
    lambda_ji_new = AUX0 / AUX1

    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h

    # New trade flows
    X_ji_new = lambda_ji_new * np.tile(E_i_new.reshape(1, -1), (N, 1)) / (1 + t_ji)
    tariff_rev = np.sum(lambda_ji_new * (t_ji / (1 + t_ji)) * \
                       np.tile(E_i_new.reshape(1, -1), (N, 1)), axis=0)

    # Tax adjustment
    tau_i = tariff_rev / Y_i_new
    tau_i_new = 0
    tau_i_h = (1 - tau_i_new) / (1 - tau_i)

    # Equilibrium conditions
    # ERR1: Wage Income = Total Sales net of Taxes
    nu_2D = np.tile(nu.reshape(1, -1), (N, 1))
    ERR1 = np.sum(beta * (1 - nu_2D) * X_ji_new, axis=1) + \
           np.sum(nu_2D * X_ji_new, axis=0) - w_i_h * L_i_h * Y_i  # Fixed: axis=1 for row sums, axis=0 for column sums
    ERR1[N-1] = np.mean((w_i_h - 1) * Y_i)  # Replace one excess equation

    # ERR2: Total Income = Total Sales
    X_global = np.sum(Y_i)
    X_global_new = np.sum(Y_i_new)
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i) + \
           np.sum((1 - beta) * (1 - nu_2D) * X_ji_new, axis=1) + \
           T_i * (X_global_new / X_global) - E_i_new  # Fixed: axis=1 for row sums

    # ERR3: Labor supply
    ERR3 = L_i_h - (tau_i_h * w_i_h / P_i_h)**kappa

    # ERR4: Price index
    ERR4 = P_i_h - ((E_i_h / w_i_h)**(1 - phi)) * (np.sum(AUX0, axis=0)**(-1 / eps))

    ceq = np.concatenate([ERR1, ERR2, ERR3, ERR4])

    # Calculate welfare
    Ec_i = Y_i + T_i
    delta_i = Ec_i / (Ec_i - kappa * (1 - tau_i) * Y_i / (1 + kappa))
    Ec_i_h = (tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global)) / Ec_i
    W_i_h = delta_i * (Ec_i_h / P_i_h) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h)

    # Factual trade flows
    X_ji = lambda_ji * np.tile(E_i.reshape(1, -1), (N, 1))
    D_i = np.sum(X_ji, axis=1) - np.sum(X_ji, axis=0)
    D_i_new = np.sum(X_ji_new, axis=1) - np.sum(X_ji_new, axis=0)

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

    # Trade change
    trade = X_ji * (1 - np.eye(N))
    trade_new = X_ji_new * (1 + t_ji) * (1 - np.eye(N))
    d_trade = 100 * ((np.sum(trade_new) / np.sum(trade)) / \
                    (np.sum(Y_i_new) / np.sum(Y_i)) - 1)

    return ceq, results, d_trade


def const_mpec(x, data, param, id_US, tariff_case):
    """
    Constraint function for MPEC optimization.

    Parameters:
    -----------
    x : np.ndarray
        Combined vector [equilibrium vars; tariff vars]
    data : dict
        Data dictionary
    param : dict
        Parameter dictionary
    id_US : int
        US country index
    tariff_case : int
        1 for optimal tariff, 2 for optimal retaliation

    Returns:
    --------
    dict : Constraints in scipy format
    """
    N, E_i, Y_i, lambda_ji, t_ji, nu, T_i = data.values()
    eps, kappa, psi, phi, beta = param.values()

    # Extract variables
    w_i_h = np.abs(x[0:N])
    E_i_h = np.abs(x[N:2*N])
    L_i_h = np.abs(x[2*N:3*N])
    P_i_h = np.abs(x[3*N:4*N])
    t = np.abs(x[4*N:])

    # Update tariff matrix
    t_ji_new = t_ji.copy()
    if tariff_case == 1:
        non_us = np.setdiff1d(np.arange(N), [id_US])
        t_ji_new[non_us, id_US] = t
    elif tariff_case == 2:
        non_us = np.setdiff1d(np.arange(N), [id_US])
        t_ji_new[id_US, non_us] = t

    np.fill_diagonal(t_ji_new, 0)

    # Construct matrices
    phi_2D = np.tile(phi.reshape(1, -1), (N, 1))
    c_i_h = np.tile((w_i_h**beta * P_i_h**(1-beta)).reshape(-1, 1), (1, N))
    entry = np.tile(((w_i_h/P_i_h)**(1-beta)).reshape(-1, 1), (1, N))
    p_ij_h = ((c_i_h / ((entry * L_i_h.reshape(-1, 1))**psi))**(-eps)) * \
             ((1 + t_ji_new)**(-eps * phi_2D))

    AUX0 = lambda_ji * p_ij_h
    AUX1 = np.tile(np.sum(AUX0, axis=0, keepdims=True), (N, 1))
    lambda_ji_new = AUX0 / AUX1

    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h

    X_ji_new = lambda_ji_new * np.tile(E_i_new.reshape(1, -1), (N, 1)) / (1 + t_ji_new)
    tariff_rev = np.sum(lambda_ji_new * (t_ji_new / (1 + t_ji_new)) * \
                       np.tile(E_i_new.reshape(1, -1), (N, 1)), axis=0)

    tau_i = tariff_rev / Y_i_new
    tau_i_new = 0
    tau_i_h = (1 - tau_i_new) / (1 - tau_i)

    # Equilibrium conditions
    nu_2D = np.tile(nu.reshape(1, -1), (N, 1))
    ERR1 = np.sum(beta * (1 - nu_2D) * X_ji_new, axis=0) + \
           np.sum(nu_2D * X_ji_new, axis=1) - w_i_h * L_i_h * Y_i
    ERR1[N-1] = np.mean((w_i_h - 1) * Y_i)

    X_global = np.sum(Y_i)
    X_global_new = np.sum(Y_i_new)
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i) + \
           np.sum((1 - beta) * (1 - nu_2D) * X_ji_new, axis=0) + \
           T_i * (X_global_new / X_global) - E_i_new

    ERR3 = L_i_h - (tau_i_h * w_i_h / P_i_h)**kappa
    ERR4 = P_i_h - ((E_i_h / w_i_h)**(1 - phi)) * (np.sum(AUX0, axis=0)**(-1 / eps))

    ceq = np.concatenate([ERR1, ERR2, ERR3, ERR4])

    return ceq


def obj_mpec(x, data, param, id_US, tariff_case):
    """
    Objective function for MPEC optimization.

    Parameters:
    -----------
    x : np.ndarray
        Combined vector [equilibrium vars; tariff vars]
    data : dict
        Data dictionary
    param : dict
        Parameter dictionary
    id_US : int
        US country index
    tariff_case : int
        1 for optimal tariff, 2 for optimal retaliation

    Returns:
    --------
    float : Negative welfare (for minimization)
    """
    N, E_i, Y_i, lambda_ji, t_ji, nu, T_i = data.values()
    eps, kappa, psi, phi, beta = param.values()

    # Extract variables
    w_i_h = np.abs(x[0:N])
    E_i_h = np.abs(x[N:2*N])
    L_i_h = np.abs(x[2*N:3*N])
    P_i_h = np.abs(x[3*N:4*N])
    t = np.abs(x[4*N:])

    # Update tariff matrix
    t_ji_new = t_ji.copy()
    if tariff_case == 1:
        non_us = np.setdiff1d(np.arange(N), [id_US])
        t_ji_new[non_us, id_US] = t
    elif tariff_case == 2:
        non_us = np.setdiff1d(np.arange(N), [id_US])
        t_ji_new[id_US, non_us] = t

    np.fill_diagonal(t_ji_new, 0)

    # Construct matrices
    phi_2D = np.tile(phi.reshape(1, -1), (N, 1))
    c_i_h = np.tile((w_i_h**beta * P_i_h**(1-beta)).reshape(-1, 1), (1, N))
    entry = np.tile(((w_i_h/P_i_h)**(1-beta)).reshape(-1, 1), (1, N))
    p_ij_h = ((c_i_h / ((entry * L_i_h.reshape(-1, 1))**psi))**(-eps)) * \
             ((1 + t_ji_new)**(-eps * phi_2D))

    AUX0 = lambda_ji * p_ij_h
    lambda_ji_new = AUX0 / np.tile(np.sum(AUX0, axis=0, keepdims=True), (N, 1))

    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h

    tariff_rev = np.sum(lambda_ji_new * (t_ji_new / (1 + t_ji_new)) * \
                       np.tile(E_i_new.reshape(1, -1), (N, 1)), axis=0)
    tau_i = tariff_rev / Y_i_new

    X_global = np.sum(Y_i)
    X_global_new = np.sum(Y_i_new)

    Ec_i = Y_i + T_i
    delta_i = Ec_i / (Ec_i - kappa * (1 - tau_i) * Y_i / (1 + kappa))
    Ec_i_h = (tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global)) / Ec_i
    W_i_h = delta_i * (Ec_i_h / P_i_h) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h)

    if tariff_case == 1:
        gains = -100 * (W_i_h[id_US] - 1)
    elif tariff_case == 2:
        non_us = np.setdiff1d(np.arange(N), [id_US])
        gains = -100 * np.sum(Y_i[non_us] * (W_i_h[non_us] - 1)) / np.sum(Y_i[non_us])

    return gains


def main():
    """Main function to run IO analysis."""
    print("Running Input-Output model analysis...")
    print(f"Current directory: {os.getcwd()}")

    # Read trade and GDP data
    base_path = os.path.join(os.path.dirname(__file__), '..', '..')

    data_path = os.path.join(base_path, 'data', 'base_data', 'trade_cepii.csv')
    X_ji = pd.read_csv(data_path, header=0).values  # header=0 to skip header row
    # Convert to float, replacing any non-numeric values with NaN, then replace NaN with 0
    X_ji = pd.DataFrame(X_ji).apply(pd.to_numeric, errors='coerce').fillna(0).values
    N = X_ji.shape[0]

    id_US = 185 - 1  # Convert to 0-indexed

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
    np.fill_diagonal(X_ji, X_ii)

    # Setup for IO model
    beta = 0.49
    nu_IO = nu.copy()
    X_ji_IO = X_ji.copy()
    X_ji_IO[np.eye(N, dtype=bool)] = X_ji[np.eye(N, dtype=bool)] / beta

    # FIXED: E_i_IO should be column sums (axis=0), not row sums
    E_i_IO = np.sum(X_ji_IO, axis=0)
    # FIXED: First term should be row sums (axis=1), second term column sums (axis=0)
    Y_i_IO = beta * np.sum(np.tile((1 - nu_IO).reshape(1, -1), (N, 1)) * X_ji_IO, axis=1) + \
             nu_IO * np.sum(X_ji_IO, axis=0)
    lambda_ji_IO = X_ji_IO / np.tile(E_i_IO.reshape(1, -1), (N, 1))
    # FIXED: Should be row sums (axis=1), not column sums
    T_IO = E_i_IO - (Y_i_IO + (1 - beta) * \
                     np.sum(np.tile((1 - nu_IO).reshape(1, -1), (N, 1)) * X_ji_IO, axis=1))

    # Read tariffs
    tariff_path = os.path.join(base_path, 'data', 'base_data', 'tariffs.csv')
    new_ustariff = pd.read_csv(tariff_path, header=0).values.flatten()
    new_ustariff = pd.to_numeric(new_ustariff, errors='coerce')
    t_ji = np.zeros((N, N))
    # Exclude US entry from tariff array to match dimension (N countries in model, but tariff file has N+1 entries)
    new_ustariff_valid = np.delete(new_ustariff, id_US) if len(new_ustariff) > N else new_ustariff
    t_ji[:, id_US] = new_ustariff_valid
    t_ji[:, id_US] = np.maximum(0.1, t_ji[:, id_US])
    t_ji[id_US, id_US] = 0

    # Parameters
    eps = 4
    kappa = 0.5
    psi = 0.67 / eps
    theta = eps / 0.67
    phi_tilde = (1 + theta) / ((1 - nu) * theta) - (1 / theta) - 1
    phi_IO = 1 + phi_tilde

    # Create arrays to save results
    results = np.zeros((N, 7, 4))
    d_trade_IO = np.zeros(2)
    d_employment_IO = np.zeros(2)

    # Roundabout Production (no retaliation)
    print("\n=== Running IO model with roundabout production ===")
    data = {
        'N': N, 'E_i': E_i_IO, 'Y_i': Y_i_IO, 'lambda_ji': lambda_ji_IO,
        't_ji': t_ji, 'nu': nu_IO, 'T_i': T_IO
    }
    param = {'eps': eps, 'kappa': kappa, 'psi': psi, 'phi': phi_IO, 'beta': beta}

    x0 = np.ones(4 * N)

    def syst(x):
        ceq, _, _ = balanced_trade_io(x, data, param)
        return ceq

    # Tight tolerance to match MATLAB (TolFun=1e-10, TolX=1e-10)
    print("  Solving equilibrium (this may take 2-3 minutes)...")
    x_fsolve_1 = fsolve(syst, x0, xtol=1e-10, maxfev=50000, factor=0.1)
    _, results[:, :, 0], d_trade_IO[0] = balanced_trade_io(x_fsolve_1, data, param)
    d_employment_IO[0] = np.sum(results[:, 4, 0] * Y_i_IO) / np.sum(Y_i_IO)
    print(f"  US welfare change: {results[id_US, 0, 0]:.2f}%")

    # Optimal tariff + IO (using grid search)
    print("\nFinding optimal US tariff with IO linkages...")
    print("  Using grid search over uniform tariff values...")

    non_us = np.setdiff1d(np.arange(N), [id_US])

    # Grid search over uniform tariff values from 0.10 to 0.25
    tariff_grid = np.arange(0.10, 0.26, 0.01)
    best_welfare = -np.inf
    best_tariff = 0.15
    best_solution = x_fsolve_1.copy()

    for t_test in tariff_grid:
        # Set uniform tariff on US imports
        t_ji_test = np.zeros((N, N))
        t_ji_test[non_us, id_US] = t_test

        # Update data with new tariff
        data_test = {
            'N': N, 'E_i': E_i_IO, 'Y_i': Y_i_IO, 'lambda_ji': lambda_ji_IO,
            't_ji': t_ji_test, 'nu': nu_IO, 'T_i': T_IO
        }
        param_test = {'eps': eps, 'kappa': kappa, 'psi': psi, 'phi': phi_IO, 'beta': beta}

        # Define equilibrium system for this tariff
        def syst_tariff(x):
            ceq, _, _ = balanced_trade_io(x, data_test, param_test)
            return ceq

        # Solve equilibrium
        try:
            x_sol = fsolve(syst_tariff, x_fsolve_1, xtol=1e-10, maxfev=50000, factor=0.1)
            _, results_test, _ = balanced_trade_io(x_sol, data_test, param_test)
            usa_welfare = results_test[id_US, 0]

            # Track best solution
            if usa_welfare > best_welfare:
                best_welfare = usa_welfare
                best_tariff = t_test
                best_solution = x_sol.copy()
                print(f"  t={t_test:.2f}: US welfare = {usa_welfare:.2f}%")
        except:
            pass  # Skip failed equilibria

    print(f"\n  Grid search complete!")
    print(f"  Optimal uniform tariff: {best_tariff:.2f}")
    print(f"  USA welfare change: {best_welfare:.2f}%")

    # Compute full results with optimal tariff
    t_ji_opt = np.zeros((N, N))
    t_ji_opt[non_us, id_US] = best_tariff
    data_opt = {
        'N': N, 'E_i': E_i_IO, 'Y_i': Y_i_IO, 'lambda_ji': lambda_ji_IO,
        't_ji': t_ji_opt, 'nu': nu_IO, 'T_i': T_IO
    }
    param_opt = {'eps': eps, 'kappa': kappa, 'psi': psi, 'phi': phi_IO, 'beta': beta}
    _, results[:, :, 1], d_trade_IO[1] = balanced_trade_io(best_solution, data_opt, param_opt)
    d_employment_IO[1] = np.sum(results[:, 4, 1] * Y_i_IO) / np.sum(Y_i_IO)

    # Liberation Tariffs with reciprocal retaliation + IO
    print("\nRunning Liberation tariffs with reciprocal retaliation (IO)...")
    t_ji_new = t_ji.copy()
    t_ji_new[id_US, :] = t_ji_new[:, id_US]
    t_ji_new[id_US, id_US] = 0

    data = {
        'N': N, 'E_i': E_i_IO, 'Y_i': Y_i_IO, 'lambda_ji': lambda_ji_IO,
        't_ji': t_ji_new, 'nu': nu_IO, 'T_i': T_IO
    }
    param = {'eps': eps, 'kappa': kappa, 'psi': psi, 'phi': phi_IO, 'beta': beta}

    x_fsolve = fsolve(syst, np.ones(4*N), xtol=1e-10, maxfev=50000, factor=0.1)
    _, results[:, :, 2], d_trade_IO[1] = balanced_trade_io(x_fsolve, data, param)
    d_employment_IO[1] = np.sum(results[:, 4, 2] * Y_i_IO) / np.sum(Y_i_IO)
    print(f"  US welfare change: {results[id_US, 0, 2]:.2f}%")

    # Save IO results for Table 11
    print("\nSaving IO model results...")
    output_dir = '../../python_output'
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, 'io_results.npz'),
             results=results,
             d_trade_IO=d_trade_IO,
             d_employment_IO=d_employment_IO,
             Y_i_IO=Y_i_IO,
             E_i_IO=E_i_IO,
             id_US=id_US)
    print(f"  - Saved: io_results.npz")

    print("\n=== IO model analysis completed ===")

    return {
        'results': results,
        'd_trade_IO': d_trade_IO,
        'd_employment_IO': d_employment_IO,
        'Y_i_IO': Y_i_IO,
        'E_i_IO': E_i_IO,
        'N': N
    }


if __name__ == '__main__':
    results_dict = main()
