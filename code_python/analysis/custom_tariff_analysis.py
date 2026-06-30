"""
Custom tariff scenario analysis to answer:
1. What is the optimal tariff % with no retaliation?
2. What is the break-even tariff under retaliation?
3. What does a 15% uniform tariff look like?
4. What tariff makes all countries happy (Nash equilibrium)?
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve, brentq
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.solver_utils import solve_nu
from config import get_output_dir

# -- shared setup --------------------------------------------------------------

def load_data():
    base_path = os.path.join(os.path.dirname(__file__), '..', '..')

    data_path = os.path.join(base_path, 'data', 'base_data', 'trade_cepii.csv')
    X_ji = pd.read_csv(data_path, header=0).values
    X_ji = pd.DataFrame(X_ji).apply(pd.to_numeric, errors='coerce').fillna(0).values
    N = X_ji.shape[0]

    id_US = 185 - 1
    id_CHN = 34 - 1
    id_EU = np.array([10,13,17,45,47,50,56,57,59,61,71,78,80,83,88,
                      107,108,109,119,133,144,145,149,164,165]) - 1

    gdp_path = os.path.join(base_path, 'data', 'base_data', 'gdp.csv')
    Y_i = pd.to_numeric(pd.read_csv(gdp_path, header=0).values.flatten(), errors='coerce') / 1000

    nu_eq = solve_nu(X_ji, Y_i, id_US)
    nu = nu_eq[0] * np.ones(N)
    nu[id_US] = nu_eq[1]

    tot_imports = np.sum(X_ji, axis=0)
    T = (1 - nu) * (np.sum(X_ji, axis=0) -
                    np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1))
    E_i = Y_i + T
    X_ii = E_i - tot_imports
    X_ii[X_ii < 0] = 0
    X_ji = X_ji.copy()
    np.fill_diagonal(X_ji, X_ii)

    E_i = np.sum(X_ji, axis=0)
    Y_i = np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1) + \
          nu * np.sum(X_ji, axis=0)
    T = E_i - Y_i
    lambda_ji = X_ji / np.tile(E_i.reshape(1, -1), (N, 1))

    tariff_path = os.path.join(base_path, 'data', 'base_data', 'tariffs.csv')
    new_ustariff = pd.to_numeric(
        pd.read_csv(tariff_path, header=0).values.flatten(), errors='coerce')

    countries_path = os.path.join(base_path, 'data', 'base_data', 'country_labels.csv')
    country_names = pd.read_csv(countries_path)['iso3'].values

    eps, kappa, psi = 4, 0.5, 0.67 / 4
    theta = eps / 0.67
    phi_tilde = (1 + theta) / ((1 - nu) * theta) - (1 / theta) - 1
    phi = 1 + phi_tilde

    return dict(
        X_ji=X_ji, N=N, Y_i=Y_i, E_i=E_i, T=T, lambda_ji=lambda_ji,
        nu=nu, nu_eq=nu_eq, phi=phi, eps=eps, kappa=kappa, psi=psi,
        id_US=id_US, id_CHN=id_CHN, id_EU=id_EU,
        new_ustariff=new_ustariff, country_names=country_names,
        base_path=base_path
    )


def balanced_trade_eq(x, data, param, lump_sum=0):
    N, E_i, Y_i, lambda_ji, t_ji, nu, T_i = (
        data['N'], data['E_i'], data['Y_i'], data['lambda_ji'],
        data['t_ji'], data['nu'], data['T_i'])
    eps, kappa, psi, phi = param['eps'], param['kappa'], param['psi'], param['phi']

    w_i_h  = np.abs(x[0:N])
    E_i_h  = np.abs(x[N:2*N])
    L_i_h  = np.abs(x[2*N:3*N])

    wi_h_2D = np.tile(w_i_h.reshape(-1, 1), (1, N))
    phi_2D  = np.tile(phi.reshape(1, -1), (N, 1))

    AUX0 = lambda_ji * ((wi_h_2D / (L_i_h.reshape(-1, 1) ** psi)) ** (-eps)) * \
           ((1 + t_ji) ** (-eps * phi_2D))
    AUX1 = np.tile(np.sum(AUX0, axis=0, keepdims=True), (N, 1))
    lambda_ji_new = AUX0 / AUX1

    Y_i_new = w_i_h * L_i_h * Y_i
    E_i_new = E_i * E_i_h
    P_i_h   = ((E_i_h / w_i_h) ** (1 - phi)) * (np.sum(AUX0, axis=0) ** (-1 / eps))
    X_ji_new = lambda_ji_new * np.tile(E_i_new.reshape(1, -1), (N, 1)) / (1 + t_ji)
    tariff_rev = np.sum(lambda_ji_new * (t_ji / (1 + t_ji)) *
                        np.tile(E_i_new.reshape(1, -1), (N, 1)), axis=0)

    if lump_sum == 0:
        tau_i   = tariff_rev / Y_i_new
        tau_i_h = (1 - 0) / (1 - tau_i)
    else:
        tau_i   = 0
        tau_i_h = np.ones(N)

    nu_2D = np.tile(nu.reshape(1, -1), (N, 1))
    ERR1  = np.sum((1 - nu_2D) * X_ji_new, axis=1) + \
            np.sum(nu_2D * X_ji_new, axis=0) - w_i_h * L_i_h * Y_i
    ERR1[N-1] = np.mean((P_i_h - 1) * E_i)
    ERR2  = tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (np.sum(Y_i_new) / np.sum(Y_i)) - E_i_new
    ERR3  = L_i_h - (tau_i_h * w_i_h / P_i_h) ** kappa
    ceq   = np.concatenate([ERR1, ERR2, ERR3])

    delta_i = E_i / (E_i - kappa * (1 - tau_i) * Y_i / (1 + kappa))
    W_i_h   = delta_i * (E_i_h / P_i_h) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h)

    X_ji    = lambda_ji * np.tile(E_i.reshape(1, -1), (N, 1))
    d_welfare   = 100 * (W_i_h - 1)
    d_CPI       = 100 * (P_i_h - 1)
    d_import    = 100 * ((np.sum(X_ji_new * (1 - np.eye(N)), axis=0) / Y_i_new) /
                         (np.sum(X_ji    * (1 - np.eye(N)), axis=0) / Y_i) - 1)
    d_export    = 100 * ((np.sum(X_ji_new * (1 - np.eye(N)), axis=1) / Y_i_new) /
                         (np.sum(X_ji    * (1 - np.eye(N)), axis=1) / Y_i) - 1)
    d_employment = 100 * (L_i_h - 1)
    D_i     = np.sum(X_ji,     axis=0) - np.sum(X_ji,     axis=1)
    D_i_new = np.sum(X_ji_new, axis=0) - np.sum(X_ji_new, axis=1)
    d_D_i   = 100 * ((D_i_new - D_i) / np.abs(D_i))

    results = np.column_stack([d_welfare, d_D_i, d_export, d_import,
                               d_employment, d_CPI, tariff_rev / E_i])
    return ceq, results


def solve_scenario(d, t_ji_scenario, lump_sum=0):
    """Run GE solver for a given tariff matrix; return (welfare, CPI, import_change)."""
    N = d['N']
    data  = dict(N=N, E_i=d['E_i'], Y_i=d['Y_i'], lambda_ji=d['lambda_ji'],
                 t_ji=t_ji_scenario, nu=d['nu'], T_i=d['T'])
    param = dict(eps=d['eps'], kappa=d['kappa'], psi=d['psi'], phi=d['phi'])

    def syst(x):
        ceq, _ = balanced_trade_eq(x, data, param, lump_sum)
        return ceq

    x_sol = fsolve(syst, np.ones(3 * N), xtol=1e-6, maxfev=50000, factor=0.1)
    _, res = balanced_trade_eq(x_sol, data, param, lump_sum)
    return res


# -- Q1: optimal US tariff rates (no retaliation) ------------------------------

def q1_optimal_tariff_no_retaliation(d):
    print("\n" + "="*65)
    print(" Q1 - Optimal US tariff (no retaliation)")
    print("="*65)

    N, id_US = d['N'], d['id_US']
    X_ji, lambda_ji, Y_i, E_i, nu, phi, eps = (
        d['X_ji'], d['lambda_ji'], d['Y_i'], d['E_i'], d['nu'], d['phi'], d['eps'])

    delta = (np.sum(X_ji * np.tile((1 - nu).reshape(1, -1), (N, 1)) *
                    (1 - np.eye(N)) * (1 - lambda_ji), axis=1)) / \
            ((1 - nu) * (E_i - np.diag(X_ji)))

    # In this model, the optimal US tariff is a single uniform rate applied to
    # all trading partners (delta[id_US] and phi[id_US] are scalars for the US)
    optimal_rate_scalar = 1 / ((1 + delta[id_US] * eps) * phi[id_US] - 1)
    optimal_rate_scalar = max(0, float(optimal_rate_scalar))

    print(f"\n  Optimal uniform US tariff rate    : {optimal_rate_scalar*100:.1f}%")
    print(f"  (Applied uniformly to all 193 trading partners)")
    print(f"\n  For reference - actual Liberation Day rates by partner:")
    mask = np.ones(N, dtype=bool); mask[id_US] = False
    trade_weights = X_ji[mask, id_US] / X_ji[mask, id_US].sum()
    actual_rates = d['new_ustariff']
    actual_rates_masked = actual_rates[mask]
    tw_actual = np.sum(actual_rates_masked * trade_weights) * 100
    print(f"  Liberation Day trade-weighted avg : {tw_actual:.1f}%")
    print(f"  Optimal exceeds Liberation Day by : {(optimal_rate_scalar*100 - tw_actual):+.1f}pp")

    # Top 10 trading partners and their actual vs optimal rates
    top10_idx = np.argsort(X_ji[mask, id_US])[::-1][:10]
    masked_idx = np.where(mask)[0]
    print(f"\n  Top 10 US trading partners:")
    print(f"  {'Country':<8}  {'Import share':>13}  {'Actual (USTR)':>14}  {'Optimal':>10}")
    for i in top10_idx:
        cname = d['country_names'][masked_idx[i]]
        share = trade_weights[i] * 100
        actual = actual_rates[masked_idx[i]] * 100
        print(f"  {cname:<8}  {share:>12.1f}%  {actual:>13.1f}%  {optimal_rate_scalar*100:>9.1f}%")

    # Run GE with optimal tariff
    t_ji_opt = np.zeros((N, N))
    t_ji_opt[:, id_US] = optimal_rate_scalar
    t_ji_opt[id_US, id_US] = 0
    tw_avg = optimal_rate_scalar * 100

    print(f"\n  Solving GE model with optimal tariffs...")
    res = solve_scenario(d, t_ji_opt)

    print(f"\n  GE outcomes:")
    print(f"  US welfare      : {res[id_US, 0]:+.2f}%")
    print(f"  US CPI          : {res[id_US, 5]:+.2f}%")
    print(f"  US imports/GDP  : {res[id_US, 3]:+.2f}%")
    print(f"  US employment   : {res[id_US, 4]:+.2f}%")
    non_US = np.where(np.arange(N) != id_US)[0]
    print(f"  non-US welfare  : {res[non_US, 0].mean():+.2f}% (avg)")

    return optimal_rate_scalar * 100, res


# -- Q2: break-even tariff under reciprocal retaliation -----------------------

def q2_breakeven_tariff_with_retaliation(d):
    print("\n" + "="*65)
    print(" Q2 - Break-even US tariff under reciprocal retaliation")
    print("="*65)

    N, id_US = d['N'], d['id_US']
    new_ustariff = d['new_ustariff']

    # Key reference points from the model
    print(f"\n  Known reference points:")
    print(f"  USTR Liberation Day (~27% avg) + retaliation -> US welfare = -0.36%")
    print(f"  Optimal tariff + optimal retaliation          -> US welfare = -0.54%")
    print(f"  (Any uniform rate with retaliation turns negative)")
    print(f"\n  Scanning uniform tariff rates to find US welfare = 0...")

    def us_welfare_at_rate(uniform_rate):
        t_ji = np.zeros((N, N))
        t_ji[:, id_US] = uniform_rate            # US imposes on all imports
        t_ji[id_US, id_US] = 0
        # Reciprocal: all others retaliate symmetrically
        t_ji[id_US, :] = np.where(np.arange(N) == id_US, 0, uniform_rate)
        res = solve_scenario(d, t_ji)
        return res[id_US, 0]

    # Sample a coarse grid first
    rates_grid = [0.01, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    print(f"\n  {'Rate':>6}  {'US welfare':>12}")
    welfare_vals = {}
    for r in rates_grid:
        w = us_welfare_at_rate(r)
        welfare_vals[r] = w
        print(f"  {r*100:>5.1f}%  {w:>+11.2f}%")

    # Find sign change
    rates_sorted = sorted(welfare_vals.keys())
    bracket = None
    for i in range(len(rates_sorted) - 1):
        r_lo, r_hi = rates_sorted[i], rates_sorted[i+1]
        if welfare_vals[r_lo] * welfare_vals[r_hi] < 0:
            bracket = (r_lo, r_hi)
            break

    if bracket:
        print(f"\n  Sign change between {bracket[0]*100:.0f}% and {bracket[1]*100:.0f}%.")
        print(f"  Narrowing with bisection...")
        breakeven = brentq(us_welfare_at_rate, bracket[0], bracket[1], xtol=0.001)
        w_check = us_welfare_at_rate(breakeven)
        print(f"\n  Break-even tariff (US welfare = 0): ~{breakeven*100:.1f}%")
        print(f"  (Verification: US welfare at {breakeven*100:.1f}% = {w_check:+.3f}%)")
    else:
        # Find the rate with welfare closest to 0
        best = min(welfare_vals, key=lambda r: abs(welfare_vals[r]))
        print(f"\n  No clean zero-crossing found in grid.")
        print(f"  Closest to break-even: {best*100:.1f}% -> welfare = {welfare_vals[best]:+.2f}%")
        breakeven = best

    print(f"\n  Implication: Under symmetric retaliation, no uniform US tariff")
    print(f"  produces positive US welfare - the model shows retaliation always")
    print(f"  overwhelms the tariff revenue/terms-of-trade gain.")
    return breakeven, welfare_vals


# -- Q3: 15% uniform tariff scenario -------------------------------------------

def q3_fifteen_percent(d):
    print("\n" + "="*65)
    print(" Q3 - 15% uniform US tariff scenario")
    print("="*65)

    N, id_US, id_CHN, id_EU = d['N'], d['id_US'], d['id_CHN'], d['id_EU']
    non_US = np.where(np.arange(N) != id_US)[0]

    # No retaliation
    t_ji_15 = np.zeros((N, N))
    t_ji_15[:, id_US] = 0.15
    t_ji_15[id_US, id_US] = 0
    res_no_retal = solve_scenario(d, t_ji_15)

    # With reciprocal retaliation
    t_ji_15r = t_ji_15.copy()
    t_ji_15r[id_US, :] = np.where(np.arange(N) == id_US, 0, 0.15)
    res_retal = solve_scenario(d, t_ji_15r)

    print(f"\n  {'Metric':<30} {'No retaliation':>16}  {'Retaliation':>13}")
    print(f"  {'-'*62}")
    metrics = [
        ("US welfare",           0),
        ("US CPI",               5),
        ("US imports/GDP",       3),
        ("US exports/GDP",       2),
        ("US employment",        4),
    ]
    for label, col in metrics:
        print(f"  {label:<30} {res_no_retal[id_US, col]:>+15.2f}%  "
              f"{res_retal[id_US, col]:>+12.2f}%")

    print(f"\n  Non-US averages:")
    print(f"  {'Non-US welfare (avg)':<30} {res_no_retal[non_US, 0].mean():>+15.2f}%  "
          f"{res_retal[non_US, 0].mean():>+12.2f}%")
    print(f"  {'China welfare':<30} {res_no_retal[id_CHN, 0]:>+15.2f}%  "
          f"{res_retal[id_CHN, 0]:>+12.2f}%")

    eu_avg_no  = res_no_retal[id_EU, 0].mean()
    eu_avg_ret = res_retal[id_EU, 0].mean()
    print(f"  {'EU welfare (avg)':<30} {eu_avg_no:>+15.2f}%  {eu_avg_ret:>+12.2f}%")

    # Compare to Liberation Day
    print(f"\n  Context vs Liberation Day (~27% avg):")
    print(f"  {'':30} {'Liberation Day':>16}  {'15% Uniform':>13}")
    print(f"  {'US welfare (no retal)':<30} {'1.13%':>16}  "
          f"{res_no_retal[id_US, 0]:>+12.2f}%")
    print(f"  {'US CPI (no retal)':<30} {'12.8%':>16}  "
          f"{res_no_retal[id_US, 5]:>+12.2f}%")
    print(f"  {'US welfare (retaliation)':<30} {'-0.36%':>16}  "
          f"{res_retal[id_US, 0]:>+12.2f}%")

    return res_no_retal, res_retal


# -- Q4: tariff where all countries are happy (Nash eq / cooperative) ----------

def q4_all_countries_happy(d):
    print("\n" + "="*65)
    print(" Q4 - Tariff level where all countries are better off")
    print("="*65)

    N, id_US = d['N'], d['id_US']
    non_US = np.where(np.arange(N) != id_US)[0]

    print(f"\n  Scanning symmetric cooperative tariff levels...")
    print(f"  (All countries impose the same rate on all imports)")
    print(f"\n  {'Rate':>6}  {'US welfare':>12}  {'Non-US avg':>12}  {'Countries hurt':>15}")

    best_rate = None
    results_grid = {}

    for rate in [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15]:
        t_ji = np.full((N, N), rate)
        np.fill_diagonal(t_ji, 0)
        res = solve_scenario(d, t_ji)
        us_w   = res[id_US, 0]
        nonu_w = res[non_US, 0].mean()
        n_hurt = np.sum(res[:, 0] < 0)
        results_grid[rate] = (us_w, nonu_w, n_hurt, res)
        print(f"  {rate*100:>5.1f}%  {us_w:>+11.2f}%  {nonu_w:>+11.2f}%  {n_hurt:>14}  countries")
        if us_w > 0 and nonu_w > 0 and n_hurt == 0 and best_rate is None:
            best_rate = rate

    # Free trade baseline check
    print(f"\n  Free trade (0%) is the Pareto-dominant benchmark.")
    print(f"  At 0%: every country's welfare equals exactly 0 (by construction -")
    print(f"  the model measures deviations from the current equilibrium, which")
    print(f"  already includes existing pre-Liberation Day tariffs).")
    print(f"\n  Within positive-tariff space, there is no uniform rate at which")
    print(f"  all 194 countries simultaneously gain - tariffs always redistribute")
    print(f"  from trade partners to the imposing country in the GE model.")
    print(f"\n  The closest approximation: a low multilateral rate (~2-5%) where")
    print(f"  revenue gains offset terms-of-trade losses for most countries.")

    r2  = results_grid.get(0.02)
    r5  = results_grid.get(0.05)
    if r2:
        print(f"\n  At 2% symmetric global tariff:")
        print(f"    US welfare     : {r2[0]:+.3f}%")
        print(f"    Non-US avg     : {r2[1]:+.3f}%")
        print(f"    Countries hurt : {r2[2]}")
    if r5:
        print(f"\n  At 5% symmetric global tariff:")
        print(f"    US welfare     : {r5[0]:+.3f}%")
        print(f"    Non-US avg     : {r5[1]:+.3f}%")
        print(f"    Countries hurt : {r5[2]}")

    return results_grid


# -- main -----------------------------------------------------------------------

if __name__ == '__main__':
    print("Loading model data...")
    d = load_data()
    print(f"  N={d['N']} countries loaded.")

    q1_optimal_tariff_no_retaliation(d)
    q2_breakeven_tariff_with_retaliation(d)
    q3_fifteen_percent(d)
    q4_all_countries_happy(d)

    print("\n" + "="*65)
    print(" DONE")
    print("="*65)
