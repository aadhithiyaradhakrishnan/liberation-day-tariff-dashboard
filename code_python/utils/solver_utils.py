"""
Utility functions for solving trade equilibrium models.
Converted from MATLAB to Python for replication of:
"Making America Great Again? The Economic Impacts of Liberation Day Tariffs"
by Ignatenko, Macedoni, Lashkaripour, Simonovska (2025)
"""

import numpy as np
from scipy.optimize import fsolve


def solve_nu(X, Y, id_US):
    """
    Solve for nu parameters (share of factor income in production).

    Parameters:
    -----------
    X : np.ndarray
        Trade flow matrix (N x N)
    Y : np.ndarray
        GDP vector (N x 1)
    id_US : int
        Index of US (0-indexed in Python, was 185 in MATLAB which is 1-indexed)

    Returns:
    --------
    nu : np.ndarray
        Vector of nu values (2 x 1)
    """
    N = X.shape[0]

    # Aggregate to US vs non-US
    AggI = np.zeros((2, N))
    AggI[0, :] = 1
    AggI[0, id_US] = 0
    AggI[1, id_US] = 1

    X_agg = AggI @ X @ AggI.T
    Y_agg = AggI @ Y

    # Initial guess for nu
    nu0 = np.array([0.1, 0.24])

    # Solve system of equations
    nu = fsolve(lambda nu_val: eq_fun(nu_val, X_agg, Y_agg), nu0,
                xtol=1e-10, maxfev=100000)

    # Ensure non-negative
    nu[nu < 0] = 0

    return nu


def eq_fun(nu, X, Y):
    """
    System of equations for solving nu.

    Parameters:
    -----------
    nu : np.ndarray
        Vector of nu values (2 x 1)
    X : np.ndarray
        Aggregated trade flow matrix (2 x 2)
    Y : np.ndarray
        Aggregated GDP vector (2 x 1)

    Returns:
    --------
    F : np.ndarray
        Residuals (2 x 1)
    """
    # Compute expenditure
    E_i = Y + (1 - nu) * (np.sum(X, axis=1) - np.sum(np.tile((1 - nu), (2, 1)).T * X, axis=1))

    # Compute ratios
    r_11 = (E_i[0] - X[1, 0]) / (E_i[0] - X[1, 0] + X[0, 1])
    r_22 = (E_i[1] - X[0, 1]) / (E_i[1] - X[0, 1] + X[1, 0])

    # System of equations
    F = np.zeros(2)
    F[0] = (1 - r_11) * nu[1] + r_11 * nu[0] - 0.12
    F[1] = r_22 * nu[1] + (1 - r_22) * nu[0] - 0.26

    return F
