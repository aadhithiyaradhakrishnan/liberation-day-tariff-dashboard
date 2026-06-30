"""
Diagnostic script to compare MATLAB and Python implementations step-by-step.
This will help identify where the numerical discrepancies originate.
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from utils.solver_utils import solve_nu

def main():
    print("="*70)
    print(" DIAGNOSTIC COMPARISON: MATLAB vs Python")
    print("="*70)

    base_path = os.path.join(os.path.dirname(__file__), '..')

    # 1. Read and compare trade data
    print("\n1. Reading trade data (trade_cepii.csv)...")
    data_path = os.path.join(base_path, 'data', 'base_data', 'trade_cepii.csv')
    X_ji = pd.read_csv(data_path, header=0).values
    X_ji = pd.DataFrame(X_ji).apply(pd.to_numeric, errors='coerce').fillna(0).values
    N = X_ji.shape[0]
    print(f"   Shape: {X_ji.shape}")
    print(f"   Sum of all trade flows: {np.sum(X_ji):.2f}")
    print(f"   Number of countries (N): {N}")

    # 2. Country indices
    print("\n2. Country indices (converted from MATLAB 1-indexed to Python 0-indexed)...")
    id_US = 185 - 1  # MATLAB: 185, Python: 184
    id_CHN = 34 - 1
    id_CAN = 31 - 1
    id_MEX = 115 - 1
    print(f"   US index: {id_US}")
    print(f"   China index: {id_CHN}")

    # 3. Read and compare GDP data
    print("\n3. Reading GDP data (gdp.csv)...")
    gdp_path = os.path.join(base_path, 'data', 'base_data', 'gdp.csv')
    Y_i = pd.read_csv(gdp_path, header=0).values.flatten()
    Y_i = pd.to_numeric(Y_i, errors='coerce')
    Y_i = Y_i / 1000  # Convert to thousands
    print(f"   Shape: {Y_i.shape}")
    print(f"   US GDP: {Y_i[id_US]:.2f}")
    print(f"   China GDP: {Y_i[id_CHN]:.2f}")
    print(f"   Total world GDP: {np.sum(Y_i):.2f}")

    # 4. Calculate trade aggregates
    print("\n4. Trade aggregates...")
    tot_exports = np.sum(X_ji, axis=1)
    tot_imports = np.sum(X_ji, axis=0)
    print(f"   US total exports: {tot_exports[id_US]:.2f}")
    print(f"   US total imports: {tot_imports[id_US]:.2f}")

    # 5. Solve for nu parameters
    print("\n5. Solving for nu (expenditure shares)...")
    nu_eq = solve_nu(X_ji, Y_i, id_US)
    print(f"   nu_eq result: {nu_eq}")
    nu = nu_eq[0] * np.ones(N)
    nu[id_US] = nu_eq[1]
    print(f"   nu (non-US): {nu_eq[0]:.6f}")
    print(f"   nu (US): {nu_eq[1]:.6f}")

    # 6. Calculate trade balance and expenditure
    print("\n6. Trade balance and expenditure...")
    T = (1 - nu) * (np.sum(X_ji, axis=0) -
                    np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1))
    E_i = Y_i + T
    print(f"   US trade balance (T): {T[id_US]:.2f}")
    print(f"   US expenditure (E_i): {E_i[id_US]:.2f}")
    print(f"   US Y_i: {Y_i[id_US]:.2f}")

    # 7. Update diagonal of trade matrix
    print("\n7. Updating domestic trade (diagonal)...")
    X_ii = E_i - tot_imports
    X_ii[X_ii < 0] = 0
    np.fill_diagonal(X_ji, X_ii)
    print(f"   US domestic trade (X_ii): {X_ii[id_US]:.2f}")

    # 8. Recalculate E_i, Y_i, T after diagonal update
    print("\n8. Recalculating after diagonal update...")
    # MATLAB line 36: E_i = sum(X_ji,1)' - column sums (imports), then transpose
    # MATLAB line 37: Y_i = sum(...,2) + nu.*sum(X_ji,1)' - row sums + column sums
    E_i = np.sum(X_ji, axis=0)  # sum over axis 0 = column sums (imports)
    Y_i = np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1) + \
          nu * np.sum(X_ji, axis=0)  # First term: axis=1 (row sums), second: axis=0 (column sums)
    T = E_i - Y_i
    lambda_ji = X_ji / np.tile(E_i.reshape(1, -1), (N, 1))
    print(f"   US E_i (expenditure): {E_i[id_US]:.2f}")
    print(f"   US Y_i (income): {Y_i[id_US]:.2f}")
    print(f"   US T (trade balance): {T[id_US]:.2f}")

    # 9. Read and apply tariffs
    print("\n9. Reading tariff data...")
    tariff_path = os.path.join(base_path, 'data', 'base_data', 'tariffs.csv')
    new_ustariff = pd.read_csv(tariff_path, header=0).values.flatten()
    new_ustariff = pd.to_numeric(new_ustariff, errors='coerce')
    print(f"   Tariff data shape: {new_ustariff.shape}")
    print(f"   China -> US tariff (before min): {new_ustariff[id_CHN]:.4f}")
    print(f"   Canada -> US tariff (before min): {new_ustariff[id_CAN]:.4f}")

    t_ji = np.zeros((N, N))
    t_ji[:, id_US] = new_ustariff
    t_ji[:, id_US] = np.maximum(0.1, t_ji[:, id_US])  # Minimum 10% tariff
    t_ji[id_US, id_US] = 0  # No tariff on domestic trade

    print(f"   China -> US tariff (after min): {t_ji[id_CHN, id_US]:.4f}")
    print(f"   Canada -> US tariff (after min): {t_ji[id_CAN, id_US]:.4f}")
    print(f"   Number of non-zero tariffs: {np.count_nonzero(t_ji)}")

    # 10. Model parameters
    print("\n10. Model parameters...")
    eps = 4
    kappa = 0.5
    psi = 0.67 / eps
    theta = eps / 0.67
    phi_tilde = (1 + theta) / ((1 - nu) * theta) - (1 / theta) - 1

    print(f"   Trade elasticity (eps): {eps}")
    print(f"   Labor supply elasticity (kappa): {kappa}")
    print(f"   psi: {psi:.6f}")
    print(f"   theta: {theta:.6f}")
    print(f"   phi_tilde (US): {phi_tilde[id_US]:.6f}")
    print(f"   phi_tilde (China): {phi_tilde[id_CHN]:.6f}")

    Phi = [1 + phi_tilde, 0.5 + phi_tilde, 0.25 + phi_tilde]
    print(f"   Phi[0] (US): {Phi[0][id_US]:.6f}")
    print(f"   Phi[1] (US): {Phi[1][id_US]:.6f}")

    # 11. Trade share matrix
    print("\n11. Trade share matrix (lambda_ji)...")
    print(f"   lambda_ji shape: {lambda_ji.shape}")
    print(f"   China -> US share: {lambda_ji[id_CHN, id_US]:.6f}")
    print(f"   US -> US (domestic) share: {lambda_ji[id_US, id_US]:.6f}")
    print(f"   Sum of lambda_ji to US: {np.sum(lambda_ji[:, id_US]):.6f} (should be 1.0)")

    print("\n" + "="*70)
    print(" DIAGNOSTIC COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Compare these values with MATLAB output")
    print("2. Check if solver initial conditions match")
    print("3. Verify equilibrium equations are identical")

    return {
        'N': N,
        'X_ji': X_ji,
        'Y_i': Y_i,
        'E_i': E_i,
        'T': T,
        'nu': nu,
        't_ji': t_ji,
        'lambda_ji': lambda_ji,
        'Phi': Phi,
        'eps': eps,
        'kappa': kappa,
        'psi': psi
    }

if __name__ == '__main__':
    results = main()
