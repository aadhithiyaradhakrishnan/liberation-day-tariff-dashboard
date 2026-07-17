"""
GE scenario runner — runs the paper's actual 194-country general-equilibrium
model (Ignatenko, Macedoni, Lashkaripour & Simonovska 2025 replication) on a
custom US tariff vector.

The equilibrium system is copied from dashboard/compute_15pct_scenario.py
(which itself mirrors code_python/analysis/main_baseline.py) and EXTENDED to
also return bilateral flows (X_ji_new, lambda_ji_new) and tariff revenue so
trade-diversion analysis is possible. The originals are left untouched as
reference implementations.

Scenario treatment matches the paper's scenario 0 (USTR schedule, no
retaliation, income-tax-relief revenue treatment / lump_sum=0):
  t_ji[:, id_US] = max(0.10, applied tariff)   # 10% Liberation Day floor
  t_ji[id_US, id_US] = 0
User overrides are applied AFTER the floor (an explicit user rate below 10%
is honored — that is the point of a counterfactual).

Validation gates (run scripts/validate via validate() below):
  1. overrides={}        must reproduce baseline_results.npz results[:, :, 0]
  2. flat 15% everywhere must reproduce scenario_15pct.npz
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

MODEL_VERSION = "ge-runner-1.0 (eps=4, kappa=0.5, phi=1+phi_tilde, lump_sum=0)"
DATA_VINTAGE = "CEPII 2023 trade matrix; tariffs.csv USTR schedule w/ 10% floor"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code_python"))

from utils.solver_utils import solve_nu  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Equilibrium system — copy of compute_15pct_scenario.balanced_trade_eq,
# extended to return bilateral flows and tariff revenue.
# ─────────────────────────────────────────────────────────────────────────────
def balanced_trade_eq_ext(x, data, param, lump_sum=0):
    N, E_i, Y_i, lambda_ji, t_ji, nu, T_i = data.values()
    eps, kappa, psi, phi = param.values()

    w_i_h = np.abs(x[0:N])
    E_i_h = np.abs(x[N:2 * N])
    L_i_h = np.abs(x[2 * N:3 * N])

    wi_h_2D = np.tile(w_i_h.reshape(-1, 1), (1, N))
    phi_2D = np.tile(phi.reshape(1, -1), (N, 1)) if np.ndim(phi) else np.full((N, N), phi)

    AUX0 = lambda_ji * ((wi_h_2D / (L_i_h.reshape(-1, 1) ** psi)) ** (-eps)) * \
           ((1 + t_ji) ** (-eps * phi_2D))
    AUX1 = np.tile(np.sum(AUX0, axis=0, keepdims=True), (N, 1))
    lambda_ji_new = AUX0 / AUX1

    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h

    P_i_h = ((E_i_h / w_i_h) ** (1 - phi)) * (np.sum(AUX0, axis=0) ** (-1 / eps))

    X_ji_new = lambda_ji_new * np.tile(E_i_new.reshape(1, -1), (N, 1)) / (1 + t_ji)
    tariff_rev = np.sum(lambda_ji_new * (t_ji / (1 + t_ji)) *
                        np.tile(E_i_new.reshape(1, -1), (N, 1)), axis=0)

    if lump_sum == 0:
        tau_i = tariff_rev / Y_i_new
        tau_i_h = (1 - 0) / (1 - tau_i)
    else:
        tau_i_h = np.ones(N)

    nu_2D = np.tile(nu.reshape(1, -1), (N, 1))
    ERR1 = np.sum((1 - nu_2D) * X_ji_new, axis=1) + \
           np.sum(nu_2D * X_ji_new, axis=0) - w_i_h * L_i_h * Y_i
    ERR1[N - 1] = np.mean((P_i_h - 1) * E_i)

    X_global = np.sum(Y_i)
    X_global_new = np.sum(Y_i_new)
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global) - E_i_new
    ERR3 = L_i_h - (tau_i_h * w_i_h / P_i_h) ** kappa

    ceq = np.concatenate([ERR1, ERR2, ERR3])

    tau_i = tariff_rev / Y_i_new
    delta_i = E_i / (E_i - kappa * (1 - tau_i) * Y_i / (1 + kappa))
    W_i_h = delta_i * (E_i_h / P_i_h) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h)

    X_ji = lambda_ji * np.tile(E_i.reshape(1, -1), (N, 1))
    D_i = np.sum(X_ji, axis=0) - np.sum(X_ji, axis=1)
    D_i_new = np.sum(X_ji_new, axis=0) - np.sum(X_ji_new, axis=1)

    d_welfare = 100 * (W_i_h - 1)
    d_D_i = 100 * ((D_i_new - D_i) / np.abs(D_i))
    d_export = 100 * ((np.sum(X_ji_new * (1 - np.eye(N)), axis=1) / Y_i_new) /
                      (np.sum(X_ji * (1 - np.eye(N)), axis=1) / Y_i) - 1)
    d_import = 100 * ((np.sum(X_ji_new * (1 - np.eye(N)), axis=0) / Y_i_new) /
                      (np.sum(X_ji * (1 - np.eye(N)), axis=0) / Y_i) - 1)
    d_employment = 100 * (L_i_h - 1)
    d_CPI = 100 * (P_i_h - 1)

    trade = X_ji * (1 - np.eye(N))
    trade_new = X_ji_new * (1 + t_ji) * (1 - np.eye(N))
    d_trade = 100 * ((np.sum(trade_new) / np.sum(trade)) /
                     (np.sum(Y_i_new) / np.sum(Y_i)) - 1)

    results = np.column_stack([d_welfare, d_D_i, d_export, d_import,
                               d_employment, d_CPI, tariff_rev / E_i])
    extras = {
        "X_ji_new": X_ji_new,
        "lambda_ji_new": lambda_ji_new,
        "tariff_rev": tariff_rev,
        "X_ji_base": X_ji,
        "P_i_h": P_i_h,
        "E_i_new": E_i_new,
    }
    return ceq, results, d_trade, extras


# ─────────────────────────────────────────────────────────────────────────────
# Model data (built once per process)
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_CACHE = {}


def build_model_data():
    """Rebuild model inputs exactly as main_baseline.py / compute_15pct_scenario."""
    if _MODEL_CACHE:
        return _MODEL_CACHE

    X_ji = pd.read_csv(os.path.join(ROOT, "data", "base_data", "trade_cepii.csv"),
                       header=0).values
    X_ji = pd.DataFrame(X_ji).apply(pd.to_numeric, errors="coerce").fillna(0).values.copy()
    N = X_ji.shape[0]
    id_US = 184

    Y_i = pd.read_csv(os.path.join(ROOT, "data", "base_data", "gdp.csv"),
                      header=0).values.flatten()
    Y_i = pd.to_numeric(Y_i, errors="coerce") / 1000  # trade flows are in $1000s

    nu_eq = solve_nu(X_ji, Y_i, id_US)
    nu = nu_eq[0] * np.ones(N)
    nu[id_US] = nu_eq[1]

    T = (1 - nu) * (np.sum(X_ji, axis=0) -
                    np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1))
    E_i = Y_i + T
    X_ii = E_i - np.sum(X_ji, axis=0)
    X_ii[X_ii < 0] = 0
    np.fill_diagonal(X_ji, X_ii)

    E_i = np.sum(X_ji, axis=0)
    Y_i = np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1) + \
          nu * np.sum(X_ji, axis=0)
    T = E_i - Y_i
    lambda_ji = X_ji / np.tile(E_i.reshape(1, -1), (N, 1))

    # Liberation Day US tariff vector with the paper's 10% floor
    ustr = pd.read_csv(os.path.join(ROOT, "data", "base_data", "tariffs.csv"),
                       header=0).values.flatten()
    ustr = pd.to_numeric(ustr, errors="coerce")
    ld_tariffs = np.maximum(0.10, ustr)

    eps = 4
    kappa = 0.5
    psi = 0.67 / eps
    theta = eps / 0.67
    phi_tilde = (1 + theta) / ((1 - nu) * theta) - (1 / theta) - 1
    phi = 1 + phi_tilde

    cl = pd.read_csv(os.path.join(ROOT, "data", "base_data", "country_labels.csv"))

    _MODEL_CACHE.update({
        "N": N, "id_US": id_US, "E_i": E_i, "Y_i": Y_i, "T": T,
        "lambda_ji": lambda_ji, "nu": nu, "ld_tariffs": ld_tariffs,
        "eps": eps, "kappa": kappa, "psi": psi, "phi": phi,
        "country_labels": cl,
        "iso_to_idx": {iso: i for i, iso in enumerate(cl["iso3"])},
    })
    return _MODEL_CACHE


# ─────────────────────────────────────────────────────────────────────────────
# Scenario runner
# ─────────────────────────────────────────────────────────────────────────────
def run_ge_scenario(overrides=None, flat_rate=None):
    """
    Solve the full GE model for a custom US tariff schedule.

    Parameters
    ----------
    overrides : dict {iso3: rate} — decimal rates (0.54 = 54%). Applied AFTER
                the Liberation Day 10% floor; an explicit rate below 10% is
                honored (that is the counterfactual's point).
    flat_rate : float — if given, ALL countries get this rate (overrides wins
                if both supplied for a country). Used by the flat-15% gate.

    Returns
    -------
    dict with results (194x7), d_trade, d_employment_wt, bilateral flows,
    revenue, runtime and convergence diagnostics.
    """
    m = build_model_data()
    N, id_US = m["N"], m["id_US"]

    t_us = m["ld_tariffs"].copy()
    if flat_rate is not None:
        t_us[:] = float(flat_rate)
    unmatched = []
    if overrides:
        for iso3, rate in overrides.items():
            idx = m["iso_to_idx"].get(str(iso3).upper())
            if idx is None:
                unmatched.append(iso3)
                continue
            t_us[idx] = float(rate)

    t_ji = np.zeros((N, N))
    t_ji[:, id_US] = t_us
    t_ji[id_US, id_US] = 0.0

    data = {"N": N, "E_i": m["E_i"], "Y_i": m["Y_i"],
            "lambda_ji": m["lambda_ji"], "t_ji": t_ji,
            "nu": m["nu"], "T_i": m["T"]}
    param = {"eps": m["eps"], "kappa": m["kappa"],
             "psi": m["psi"], "phi": m["phi"]}

    def syst(x):
        ceq, _, _, _ = balanced_trade_eq_ext(x, data, param, 0)
        return ceq

    x0 = np.ones(3 * N)
    t_start = time.time()
    x_sol, info, ier, msg = fsolve(syst, x0, xtol=1e-6, maxfev=50000,
                                   factor=0.1, full_output=True)
    runtime = time.time() - t_start

    ceq, results, d_trade, extras = balanced_trade_eq_ext(x_sol, data, param, 0)

    resid_max = float(np.max(np.abs(ceq)))
    resid_scaled = float(np.max(np.abs(ceq)) / max(np.max(np.abs(m["E_i"])), 1))
    # fsolve ier==1 means converged; also require a sane residual
    converged = (ier == 1) and np.isfinite(resid_max)

    d_employment_wt = float(np.sum(results[:, 4] * m["Y_i"]) / np.sum(m["Y_i"]))
    # Revenue in dollars: tariff_rev is in model units ($1000s)
    revenue_us_dollars = float(extras["tariff_rev"][id_US] * 1000)

    return {
        "results": results,                     # (194, 7): welfare, deficit, exp, imp, emp, CPI, rev/E
        "d_trade": float(d_trade),
        "d_employment_wt": d_employment_wt,
        "revenue_us_dollars": revenue_us_dollars,
        "X_ji_new": extras["X_ji_new"],         # bilateral flows, model units ($1000s)
        "X_ji_base": extras["X_ji_base"],
        "lambda_ji_new": extras["lambda_ji_new"],
        "tariff_vector": t_us,                  # decimal rates applied on US imports
        "id_US": id_US,
        "country_labels": m["country_labels"],
        "E_i": m["E_i"], "Y_i": m["Y_i"],
        "converged": bool(converged),
        "fsolve_ier": int(ier),
        "fsolve_msg": str(msg),
        "nfev": int(info.get("nfev", -1)),
        "resid_max": resid_max,
        "resid_scaled": resid_scaled,
        "runtime_sec": runtime,
        "unmatched_overrides": unmatched,
        "model_version": MODEL_VERSION,
        "data_vintage": DATA_VINTAGE,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validation gates
# ─────────────────────────────────────────────────────────────────────────────
def validate(tol_pp=0.05, verbose=True):
    """
    Gate 1: overrides={} reproduces baseline_results.npz results[:, :, 0]
    Gate 2: flat 15% reproduces scenario_15pct.npz
    Compares welfare + CPI columns (0 and 5) for all 194 countries.
    Returns dict with pass/fail and max abs errors.
    """
    out_dir = os.path.join(ROOT, "python_output")
    report = {}

    if verbose:
        print("Gate 1: Liberation Day defaults vs baseline_results.npz scenario 0")
    base = np.load(os.path.join(out_dir, "baseline_results.npz"), allow_pickle=True)
    ref0 = base["results"][:, :, 0]
    r1 = run_ge_scenario(overrides={})
    err1 = np.max(np.abs(r1["results"][:, [0, 5]] - ref0[:, [0, 5]]))
    report["gate1_max_err_pp"] = float(err1)
    report["gate1_pass"] = bool(err1 < tol_pp and r1["converged"])
    report["gate1_runtime_sec"] = r1["runtime_sec"]
    if verbose:
        print(f"  converged={r1['converged']} nfev={r1['nfev']} "
              f"runtime={r1['runtime_sec']:.0f}s max|err|={err1:.5f}pp "
              f"-> {'PASS' if report['gate1_pass'] else 'FAIL'}")

    if verbose:
        print("Gate 2: flat 15% vs scenario_15pct.npz")
    ref15 = np.load(os.path.join(out_dir, "scenario_15pct.npz"), allow_pickle=True)["results"]
    r2 = run_ge_scenario(flat_rate=0.15)
    err2 = np.max(np.abs(r2["results"][:, [0, 5]] - ref15[:, [0, 5]]))
    report["gate2_max_err_pp"] = float(err2)
    report["gate2_pass"] = bool(err2 < tol_pp and r2["converged"])
    report["gate2_runtime_sec"] = r2["runtime_sec"]
    if verbose:
        print(f"  converged={r2['converged']} nfev={r2['nfev']} "
              f"runtime={r2['runtime_sec']:.0f}s max|err|={err2:.5f}pp "
              f"-> {'PASS' if report['gate2_pass'] else 'FAIL'}")

    report["all_pass"] = report["gate1_pass"] and report["gate2_pass"]
    if verbose:
        print("VALIDATION:", "ALL GATES PASS" if report["all_pass"] else "FAILED")
    return report


if __name__ == "__main__":
    validate()
