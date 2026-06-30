"""
Compute a flat 15% US tariff scenario using the GE model from main_baseline.py.
Saves results to python_output/scenario_15pct.npz.
"""
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import sys
import os

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code_python"))

from utils.solver_utils import solve_nu

def balanced_trade_eq(x, data, param, lump_sum=0):
    N, E_i, Y_i, lambda_ji, t_ji, nu, T_i = data.values()
    eps, kappa, psi, phi = param.values()

    w_i_h = np.abs(x[0:N])
    E_i_h = np.abs(x[N:2*N])
    L_i_h = np.abs(x[2*N:3*N])

    wi_h_2D = np.tile(w_i_h.reshape(-1, 1), (1, N))
    phi_2D  = np.tile(phi.reshape(1, -1), (N, 1))

    AUX0 = lambda_ji * ((wi_h_2D / (L_i_h.reshape(-1, 1) ** psi)) ** (-eps)) * \
           ((1 + t_ji) ** (-eps * phi_2D))
    AUX1 = np.tile(np.sum(AUX0, axis=0, keepdims=True), (N, 1))
    lambda_ji_new = AUX0 / AUX1

    Y_i_h   = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h

    P_i_h = ((E_i_h / w_i_h) ** (1 - phi)) * (np.sum(AUX0, axis=0) ** (-1 / eps))

    X_ji_new  = lambda_ji_new * np.tile(E_i_new.reshape(1, -1), (N, 1)) / (1 + t_ji)
    tariff_rev = np.sum(lambda_ji_new * (t_ji / (1 + t_ji)) *
                        np.tile(E_i_new.reshape(1, -1), (N, 1)), axis=0)

    if lump_sum == 0:
        tau_i    = tariff_rev / Y_i_new
        tau_i_h  = (1 - 0) / (1 - tau_i)
    else:
        tau_i_h  = np.ones(N)

    nu_2D = np.tile(nu.reshape(1, -1), (N, 1))
    ERR1  = np.sum((1 - nu_2D) * X_ji_new, axis=1) + \
            np.sum(nu_2D * X_ji_new, axis=0) - w_i_h * L_i_h * Y_i
    ERR1[N-1] = np.mean((P_i_h - 1) * E_i)

    X_global     = np.sum(Y_i)
    X_global_new = np.sum(Y_i_new)
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global) - E_i_new
    ERR3 = L_i_h - (tau_i_h * w_i_h / P_i_h) ** kappa

    ceq = np.concatenate([ERR1, ERR2, ERR3])

    tau_i    = tariff_rev / Y_i_new
    delta_i  = E_i / (E_i - kappa * (1 - tau_i) * Y_i / (1 + kappa))
    W_i_h    = delta_i * (E_i_h / P_i_h) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h)

    X_ji     = lambda_ji * np.tile(E_i.reshape(1, -1), (N, 1))
    D_i      = np.sum(X_ji, axis=0) - np.sum(X_ji, axis=1)
    D_i_new  = np.sum(X_ji_new, axis=0) - np.sum(X_ji_new, axis=1)

    d_welfare    = 100 * (W_i_h - 1)
    d_D_i        = 100 * ((D_i_new - D_i) / np.abs(D_i))
    d_export     = 100 * ((np.sum(X_ji_new * (1 - np.eye(N)), axis=1) / Y_i_new) /
                          (np.sum(X_ji * (1 - np.eye(N)), axis=1) / Y_i) - 1)
    d_import     = 100 * ((np.sum(X_ji_new * (1 - np.eye(N)), axis=0) / Y_i_new) /
                          (np.sum(X_ji * (1 - np.eye(N)), axis=0) / Y_i) - 1)
    d_employment = 100 * (L_i_h - 1)
    d_CPI        = 100 * (P_i_h - 1)

    trade     = X_ji * (1 - np.eye(N))
    trade_new = X_ji_new * (1 + t_ji) * (1 - np.eye(N))
    d_trade   = 100 * ((np.sum(trade_new) / np.sum(trade)) /
                       (np.sum(Y_i_new) / np.sum(Y_i)) - 1)

    results = np.column_stack([d_welfare, d_D_i, d_export, d_import,
                               d_employment, d_CPI, tariff_rev / E_i])
    return ceq, results, d_trade


def main():
    print("Computing 15% flat tariff scenario...")

    X_ji = pd.read_csv(os.path.join(ROOT, "data", "base_data", "trade_cepii.csv"),
                       header=0).values
    X_ji = pd.DataFrame(X_ji).apply(pd.to_numeric, errors="coerce").fillna(0).values.copy()
    N    = X_ji.shape[0]

    id_US = 184  # 0-indexed

    Y_i = pd.read_csv(os.path.join(ROOT, "data", "base_data", "gdp.csv"),
                      header=0).values.flatten()
    Y_i = pd.to_numeric(Y_i, errors="coerce") / 1000

    nu_eq = solve_nu(X_ji, Y_i, id_US)
    nu    = nu_eq[0] * np.ones(N)
    nu[id_US] = nu_eq[1]

    T    = (1 - nu) * (np.sum(X_ji, axis=0) -
                       np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1))
    E_i  = Y_i + T
    X_ii = E_i - np.sum(X_ji, axis=0)
    X_ii[X_ii < 0] = 0
    np.fill_diagonal(X_ji, X_ii)

    E_i      = np.sum(X_ji, axis=0)
    Y_i      = np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1) + \
               nu * np.sum(X_ji, axis=0)
    T        = E_i - Y_i
    lambda_ji = X_ji / np.tile(E_i.reshape(1, -1), (N, 1))

    # ── Build 15% flat tariff on all US imports ──────────────────────────
    t_ji = np.zeros((N, N))
    t_ji[:, id_US] = 0.15
    t_ji[id_US, id_US] = 0

    eps   = 4
    kappa = 0.5
    psi   = 0.67 / eps
    theta = eps / 0.67
    phi_tilde = (1 + theta) / ((1 - nu) * theta) - (1 / theta) - 1
    phi   = 1 + phi_tilde

    data  = {"N": N, "E_i": E_i, "Y_i": Y_i, "lambda_ji": lambda_ji,
             "t_ji": t_ji, "nu": nu, "T_i": T}
    param = {"eps": eps, "kappa": kappa, "psi": psi, "phi": phi}

    x0 = np.ones(3 * N)

    def syst(x):
        ceq, _, _ = balanced_trade_eq(x, data, param, 0)
        return ceq

    print("  Solving GE equilibrium (this takes ~1-2 minutes)...")
    x_sol = fsolve(syst, x0, xtol=1e-6, maxfev=50000, factor=0.1)
    _, results, d_trade = balanced_trade_eq(x_sol, data, param, 0)

    print(f"  US welfare:    {results[id_US, 0]:.2f}%")
    print(f"  US CPI:        {results[id_US, 5]:.2f}%")
    print(f"  US imports:    {results[id_US, 3]:.2f}%")
    print(f"  Global trade:  {d_trade:.2f}%")

    out_path = os.path.join(ROOT, "python_output", "scenario_15pct.npz")
    np.savez(out_path, results=results, d_trade=d_trade, id_US=id_US)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
