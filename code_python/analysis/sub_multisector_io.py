"""
Multi-sector IO model (K=4 sectors with roundabout production)
Generates results for Table 11 columns: "multi + IO" (before & after retaliation)
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def balanced_trade_io_multisector(x, data, param):
    """
    Multi-sector IO equilibrium system with K=4 sectors

    Variables (4N + NK):
    - w_i: wages (N)
    - E_i: expenditure (N)
    - L_i: labor (N)
    - P_i: price index (N)
    - ell_ik: sectoral labor shares (N*K)

    Parameters:
    - data: [N, K, E_i, Y_i, lambda_ji, e_i, ell_ik, t_ji, nu, T_i]
    - param: [eps_3D, kappa, psi, phi, beta_3D]
    """
    N, K, E_i, Y_i, lambda_ji, e_i, ell_ik, t_ji, nu, T_i = data
    eps, kappa, psi, phi, beta = param

    # Extract variables (use abs to avoid complex numbers)
    w_i_h = np.abs(x[:N])
    E_i_h = np.abs(x[N:2*N])
    L_i_h = np.abs(x[2*N:3*N])
    P_i_h = np.abs(x[3*N:4*N])
    # CRITICAL: Use Fortran order ('F') to match MATLAB's reshape behavior
    ell_ik_h = np.abs(x[4*N:]).reshape((N, 1, K), order='F')

    # Construct 3D arrays
    w_i_3D = np.tile(w_i_h.reshape(-1, 1, 1), (1, N, K))
    P_i_3D = np.tile(P_i_h.reshape(-1, 1, 1), (1, N, K))
    L_ik_3D = np.tile(L_i_h.reshape(-1, 1, 1), (1, N, K)) * np.tile(ell_ik_h, (1, N, 1))
    phi_3D = np.tile(phi.reshape(1, -1, 1), (N, 1, K))

    # Price and trade
    c_i_h = (w_i_3D**beta) * (P_i_3D**(1 - beta))
    entry = np.tile((w_i_h / P_i_h).reshape(-1, 1, 1), (1, N, K))**(1 - beta)

    p_ij_h = (c_i_h / ((entry * L_ik_3D)**psi))**(-eps) * (1 + t_ji)**(-eps * phi_3D)
    AUX0 = lambda_ji * p_ij_h
    AUX1 = np.tile(np.sum(AUX0, axis=0, keepdims=True), (N, 1, 1))
    lambda_ji_new = AUX0 / AUX1

    # Income and expenditure
    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h

    # New trade flows
    X_ji_new = lambda_ji_new * e_i * np.tile(E_i_new.reshape(1, -1, 1), (N, 1, K)) / (1 + t_ji)

    # Tariff revenue
    tariff_rev = np.sum(np.sum(t_ji * X_ji_new, axis=2), axis=0)

    tau_i = tariff_rev / Y_i
    tau_i_new = 0
    tau_i_h = (1 - tau_i_new) / (1 - tau_i)

    # Equilibrium equations
    nu_3D = np.tile(nu.reshape(1, -1, 1), (N, 1, K))

    # ERR1: Sectoral income balance (N*K equations)
    Y_ik_h = w_i_3D[:, 0:1, :] * L_ik_3D[:, 0:1, :]
    Y_ik = ell_ik * np.tile(Y_i.reshape(-1, 1, 1), (1, 1, K))
    Y_ik_cf = np.sum((1 - nu_3D) * beta * X_ji_new, axis=1, keepdims=True) + \
              np.transpose(np.sum(nu_3D * X_ji_new, axis=0, keepdims=True), (1, 0, 2))
    # CRITICAL: Use Fortran order ('F') to match MATLAB's reshape behavior
    ERR1 = (Y_ik_cf - Y_ik * Y_ik_h).reshape(N*K, order='F')
    ERR1[N-1] = np.sum((E_i_h - 1) * E_i)  # Replace one redundant equation

    # ERR2: Income = Sales + Transfers (N equations)
    X_global = np.sum(Y_i)
    X_global_new = np.sum(Y_i_new)
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i) + \
           np.sum(np.sum((1 - beta) * (1 - nu_3D) * X_ji_new, axis=1), axis=1) + \
           T_i * (X_global_new / X_global) - E_i_new

    # ERR3: Labor supply (N equations)
    ERR3 = L_i_h - (tau_i_h * w_i_h / P_i_h)**kappa

    # ERR4: Price index (N equations)
    ERR4 = P_i_h - (E_i_h / w_i_h)**(1 - phi) * \
           np.prod(np.sum(AUX0, axis=0)**(-e_i[0, :, :] / eps[0, :, :]), axis=1)

    # ERR5: Sectoral labor shares sum to 1 (N equations)
    ERR5 = 100 * (np.sum(ell_ik * ell_ik_h, axis=2).reshape(N) - 1)

    ceq = np.concatenate([ERR1, ERR2, ERR3, ERR4, ERR5])

    # Calculate results
    X_ji = lambda_ji * e_i * np.tile(E_i.reshape(1, -1, 1), (N, 1, K))
    D_i = np.sum(np.sum(X_ji, axis=2), axis=0) - np.sum(np.sum(X_ji, axis=2), axis=1)
    D_i_new = np.sum(np.sum(X_ji_new, axis=2), axis=0) - np.sum(np.sum(X_ji_new, axis=2), axis=1)

    Ec_i = Y_i + T_i
    delta_i = Ec_i / (Ec_i - kappa * (1 - tau_i) * Y_i / (1 + kappa))
    Ec_i_h = (tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global)) / Ec_i
    W_i_h = delta_i * (Ec_i_h / P_i_h) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h)

    d_welfare = 100 * (W_i_h - 1)
    d_export = 100 * ((np.sum(np.sum(X_ji_new, axis=2) * (1 - np.eye(N)), axis=1) / Y_i_new) / \
                      (np.sum(np.sum(X_ji, axis=2) * (1 - np.eye(N)), axis=1) / Y_i) - 1)
    d_import = 100 * ((np.sum(np.sum(X_ji_new, axis=2) * (1 - np.eye(N)), axis=0) / Y_i_new) / \
                      (np.sum(np.sum(X_ji, axis=2) * (1 - np.eye(N)), axis=0) / Y_i) - 1)
    d_employment = 100 * (L_i_h - 1)
    d_CPI = 100 * (P_i_h - 1)
    d_D_i = 100 * ((D_i_new - D_i) / np.abs(D_i))

    results = np.column_stack([d_welfare, d_D_i, d_export, d_import, d_employment, d_CPI, tariff_rev / E_i])

    # Global trade change
    trade = X_ji * np.tile((1 - np.eye(N)).reshape(N, N, 1), (1, 1, K))
    trade_new = X_ji_new * (1 + t_ji) * np.tile((1 - np.eye(N)).reshape(N, N, 1), (1, 1, K))
    d_trade = 100 * ((np.sum(trade_new) / np.sum(trade)) / (np.sum(Y_i_new) / np.sum(Y_i)) - 1)

    return ceq, results, d_trade


def main():
    print("=" * 80)

    # Set up base path
    base_path = os.path.join(os.path.dirname(__file__), "..", "..")
    output_dir = os.path.join(base_path, "python_output")
    print("Multi-Sector IO Model (K=4 sectors with roundabout production)")
    print("=" * 80)

    # Load sectoral trade data
    print("\nLoading sectoral trade data...")
    data_path = os.path.join(base_path, 'data', 'ITPDS', 'trade_ITPD.csv')
    trade_data = pd.read_csv(data_path, header=None)
    X = trade_data.iloc[:, 3].values
    N = 194
    K = 4
    # CRITICAL: Use Fortran order ('F') to match how the CSV data is stored
    X_ji = X.reshape((N, N, K), order='F')

    # Remove countries with no trade FIRST
    problematic_id = np.sum(np.all(X_ji == 0, axis=0), axis=1)
    ID = np.where(problematic_id == 1)[0]
    idx = np.setdiff1d(np.arange(N), ID)
    N = len(idx)

    X_new = np.zeros((N, N, K))
    for k in range(K):
        X_new[:, :, k] = X_ji[np.ix_(idx, idx, [k])].reshape(N, N)
    X_ji = X_new

    # Load and filter tariffs AFTER removing problematic countries
    print("Loading tariff data...")
    tariff_data_full = pd.read_csv(os.path.join(base_path, 'data', 'base_data', 'tariffs.csv'))
    new_ustariff_full = tariff_data_full.values
    # Filter tariffs to match filtered countries
    new_ustariff = new_ustariff_full[idx, :]

    # US is at index 184 in the original 0-indexed array
    id_US = 184
    # After filtering, find where US (index 184) is in the filtered idx array
    id_US_new = np.where(idx == id_US)[0][0]

    t_ji = np.zeros((N, N, K))
    # new_ustariff is (N, 1), tile to (N, K-1) for assignment
    t_ji[:, id_US_new, :K-1] = np.tile(new_ustariff, (1, K-1))
    t_ji[:, id_US_new, :K-1] = np.maximum(0.1, t_ji[:, id_US_new, :K-1])
    t_ji[id_US_new, id_US_new, :K-1] = 0

    # Load parameters from baseline model
    print("Loading baseline parameters...")
    phi_data = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))
    phi = phi_data['phi'].values[idx]
    nu_data = pd.read_csv(os.path.join(output_dir, 'nu_values.csv'))
    nu = nu_data['nu'].values[idx]

    # Sectoral IO parameters
    beta = np.array([0.51, 0.32, 0.49, 0.56])
    beta_3D = np.tile(beta.reshape(1, 1, -1), (N, N, 1))
    nu_3D = np.tile(nu.reshape(1, -1, 1), (N, 1, K))

    # Calculate initial values
    E_i_multi = np.sum(np.sum(X_ji, axis=0), axis=1)
    Y_i_multi = np.sum(np.sum((1 - nu_3D) * beta_3D * X_ji, axis=2), axis=1) + \
                np.sum(np.sum(nu_3D * X_ji, axis=0), axis=1)
    T = E_i_multi - (Y_i_multi + np.sum(np.sum((1 - beta_3D) * (1 - nu_3D) * X_ji, axis=2), axis=1))

    lambda_ji = X_ji / np.tile(np.sum(X_ji, axis=0, keepdims=True), (N, 1, 1))
    e_i = np.tile(np.sum(X_ji, axis=0, keepdims=True), (N, 1, 1)) / \
          np.tile(E_i_multi.reshape(1, -1, 1), (N, 1, K))

    Y_ik_p = np.sum((1 - nu_3D) * beta_3D * X_ji, axis=1, keepdims=True)
    Y_ik_f = np.transpose(np.sum(nu_3D * X_ji, axis=0, keepdims=True), (1, 0, 2))
    Y_ik = Y_ik_p + Y_ik_f
    ell_ik = Y_ik / np.tile(Y_i_multi.reshape(-1, 1, 1), (1, 1, K))

    # Trade elasticities
    # MATLAB line 69: phi_avg=sum(Phi{1}.*Y_i)./sum(Y_i) uses UNFILTERED values and Phi{1}
    # phi_values.csv contains Phi{2} = 0.5 + phi_tilde, but we need Phi{1} = 1 + phi_tilde
    # So we add 0.5 to convert from Phi{2} to Phi{1}
    Y_i_baseline_full = pd.read_csv(os.path.join(output_dir, 'Y_i_baseline.csv'))['Y_i'].values
    phi_baseline_full = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))['phi'].values
    phi_avg = np.sum((phi_baseline_full + 0.5) * Y_i_baseline_full) / np.sum(Y_i_baseline_full)
    eps = np.array([3.3, 3.8, 4.1]) / phi_avg
    eps = np.append(eps, 3.0)  # Services sector
    eps_3D = np.tile(eps.reshape(1, 1, -1), (N, N, 1))

    # Standard parameters
    kappa = 0.5
    psi = 0.67 / 4

    results_multi = np.zeros((N, 7, 2))
    d_trade_IO_multi = np.zeros(2)
    d_employment_IO_multi = np.zeros(2)

    # Scenario 1: No retaliation
    print("\n" + "-" * 80)
    print("Scenario 1: USTR tariffs (no retaliation)")
    print("-" * 80)

    data = [N, K, E_i_multi, Y_i_multi, lambda_ji, e_i, ell_ik, t_ji, nu, T]
    param = [eps_3D, kappa, psi, phi, beta_3D]

    x0 = np.concatenate([np.ones(N), np.ones(N), np.ones(N), np.ones(N), np.ones(N*K)])

    def syst(x):
        ceq, _, _ = balanced_trade_io_multisector(x, data, param)
        return ceq

    print("Solving equilibrium...")
    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=50000)

    _, results_multi[:, :, 0], d_trade_IO_multi[0] = balanced_trade_io_multisector(x_fsolve, data, param)
    d_employment_IO_multi[0] = np.sum(results_multi[:, 4, 0] * Y_i_multi) / np.sum(Y_i_multi)

    print(f"USA welfare change: {results_multi[id_US_new, 0, 0]:.2f}%")
    print(f"Global trade-to-GDP change: {d_trade_IO_multi[0]:.2f}%")
    print(f"Global employment change: {d_employment_IO_multi[0]:.2f}%")

    # Scenario 2: Reciprocal retaliation
    print("\n" + "-" * 80)
    print("Scenario 2: Reciprocal retaliation")
    print("-" * 80)

    for k in range(K-1):
        t_ji[id_US_new, :, k] = t_ji[:, id_US_new, k]
    t_ji[id_US_new, id_US_new, :] = 0

    data = [N, K, E_i_multi, Y_i_multi, lambda_ji, e_i, ell_ik, t_ji, nu, T]

    print("Solving equilibrium...")
    x_fsolve = fsolve(syst, x_fsolve, xtol=1e-10, maxfev=50000)

    _, results_multi[:, :, 1], d_trade_IO_multi[1] = balanced_trade_io_multisector(x_fsolve, data, param)
    d_employment_IO_multi[1] = np.sum(results_multi[:, 4, 1] * Y_i_multi) / np.sum(Y_i_multi)

    print(f"USA welfare change: {results_multi[id_US_new, 0, 1]:.2f}%")
    print(f"Global trade-to-GDP change: {d_trade_IO_multi[1]:.2f}%")
    print(f"Global employment change: {d_employment_IO_multi[1]:.2f}%")

    # Save results
    print("\n" + "-" * 80)
    print("Saving results...")
    print("-" * 80)

    np.savez(os.path.join(output_dir, 'multisector_io_results.npz'),
             results_multi=results_multi,
             d_trade_IO_multi=d_trade_IO_multi,
             d_employment_IO_multi=d_employment_IO_multi,
             id_US=id_US_new)

    print("\nResults saved to: python_output/multisector_io_results.npz")
    print("  - d_trade_IO_multi[0] (no retaliation): {:.2f}%".format(d_trade_IO_multi[0]))
    print("  - d_trade_IO_multi[1] (retaliation): {:.2f}%".format(d_trade_IO_multi[1]))
    print("  - d_employment_IO_multi[0] (no retaliation): {:.2f}%".format(d_employment_IO_multi[0]))
    print("  - d_employment_IO_multi[1] (retaliation): {:.2f}%".format(d_employment_IO_multi[1]))

    print("\n" + "=" * 80)
    print("Multi-Sector IO Model Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
