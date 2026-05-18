"""
Estimate per-day transmission rate beta from ABM output using NB regression.

Three variants of the regressor z_vec are provided:
  - z_vec_ABM   : ABM mean-field force of infection
  - z_vec_MPM   : classical residence-time MPM
  - z_vec_MPM_2 : one-beta MPM with home-home, away-home, home-away pathways

The estimation loop is identical for all three; only z_vec differs.
"""
import datetime as dt
import pickle
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import optimize, special

from estimation import rivm_loader
from utils.constants import CoronaConstants

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

MAX_ITER = int(1e5)
BOOTSTRAP_MAXITER = 100  # bootstrap samples


@lru_cache(maxsize=None)
def _load_mob(scenario, date_str):
    with open(f'../data/Mobility/MobMats/{scenario}{date_str}.p', 'rb') as f:
        mob = pickle.load(f)
    row_sums = mob.sum(axis=1, keepdims=True)
    return mob / np.maximum(row_sums, 1e-10)


# ---------------------------------------------------------------------
# z_vec builders. NB regression is Delta_S_i ~ NB(beta * z_i, r).
# ---------------------------------------------------------------------

def z_vec_ABM(mob, S_live, I_live, N_live):
    """ABM mean-field FOI.

    pressure_j = sum_k mob[k,j] * I_k / sum_l mob[l,j] * N_l
    mobility_term_i = sum_j mob[i,j] * pressure_j
    z_i = S_i * mobility_term_i
    """
    infect_frac = I_live / np.maximum(N_live, 1e-10)
    numerator = mob * N_live[:, None]
    denominator = numerator.sum(axis=0)
    w = numerator / np.maximum(denominator, 1e-10)
    pressure_j = w.T @ infect_frac
    mobility_term_i = mob @ pressure_j
    return S_live * mobility_term_i


def z_vec_MPM(mob, S_live, I_live, N_live):
    """Classical MPM.

    Resident of i spends mob[i,j] of their time in j and is exposed
    to prevalence I_j/N_j there.
        foi_i = sum_j mob[i,j] * I_j/N_j
        z_i   = S_i * foi_i
    """
    infect_frac = I_live / np.maximum(N_live, 1e-10)
    foi = mob @ infect_frac
    return S_live * foi


def z_vec_MPM_2(mob, S_live, I_live, N_live):
    """One-beta MPM covering home-home + away-home + home-away pathways.

    mob[i,j] = fraction of i's residents present in j (rows sum to 1).
    The 'visitor pressure' uses M_raw[j,i]/N_i = mob[j,i]*N_j/N_i.
    """
    infect_frac = I_live / np.maximum(N_live, 1e-10)
    out_pressure = mob @ infect_frac  # p^hh + p^ah

    M_in = mob.T * (N_live[:, None] / np.maximum(N_live[None, :], 1e-10))
    in_pressure = M_in @ infect_frac
    # subtract j=i diagonal to avoid double-counting p^hh
    in_pressure -= np.diag(M_in) * infect_frac
    return S_live * (out_pressure + in_pressure)


# ---------------------------------------------------------------------
# NB log-likelihood. The loggamma(Delta_S + 1) term is constant in
# (beta, r) so we drop it from the optimization.
# ---------------------------------------------------------------------

def _neg_log_lik(x, Delta_S, z_vec, logz_vec):
    beta, r = x
    beta = max(beta, 1e-10)
    r = max(r, 1e-10)

    mu = beta * z_vec
    log_mu = np.log(beta) + logz_vec
    log_r_plus_mu = np.log(r + mu)

    term = (
            special.loggamma(Delta_S + r)
            + Delta_S * log_mu
            - Delta_S * log_r_plus_mu
            - r * log_r_plus_mu
    )
    n = Delta_S.shape[0]
    scalar = n * (r * np.log(r) - special.loggamma(r))
    return -(term.sum() + scalar)


# ---------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------

def _fit_one_sample(sample_nr, Delta_S_original, z_vec, logz_vec,
                    beta_init, r_init, pre_drawn_dS=None):
    if pre_drawn_dS is None:
        rng = np.random.default_rng(sample_nr)
        dS = rng.poisson(Delta_S_original).astype(np.float64)
    else:
        dS = pre_drawn_dS
    dS = np.maximum(0, dS)

    res = optimize.minimize(
        _neg_log_lik,
        np.array([beta_init, r_init]),
        args=(dS, z_vec, logz_vec),
        method="L-BFGS-B",
        bounds=[(1e-6, None), (1e-6, None)],
        tol=1e-8,
        options={'maxiter': BOOTSTRAP_MAXITER},
    )
    return max(res.x[0], 1e-10), max(res.x[1], 1e-10)


def _run_bootstrap(Delta_S, z_vec, logz_vec, beta_hat, r_hat,
                   nr_samples, n_jobs=-1):
    """Bootstrap CI for (beta, r). Parallel via joblib if available."""
    # pre-draw all Poisson resamples in one go
    rng = np.random.default_rng(0)
    pre_drawn = rng.poisson(
        Delta_S, size=(nr_samples, Delta_S.shape[0])
    ).astype(np.float64)

    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, prefer='processes')(
            delayed(_fit_one_sample)(
                i, Delta_S, z_vec, logz_vec, beta_hat, r_hat, pre_drawn[i]
            )
            for i in range(nr_samples)
        )
    except ImportError:
        results = [
            _fit_one_sample(i, Delta_S, z_vec, logz_vec, beta_hat, r_hat, pre_drawn[i])
            for i in range(nr_samples)
        ]
    return np.array(results)


# ---------------------------------------------------------------------
# Shared estimation core
# ---------------------------------------------------------------------

def _run_estimation(start_date, z_builder, confidence=False,
                    conf_level=.95, nr_bootstrap_samples=100,
                    scenario='low', seed=0, run=0,
                    initialization=True, plot=True, n_jobs=-1):
    """Common per-day estimation loop. Only z_builder differs across variants."""
    start_date_dt = dt.datetime.strptime(start_date, '%d-%m-%Y')
    rivm = rivm_loader.RivmLoader(scenario, start_date_dt, seed, run)

    CoronaConstants.population_nl = float(rivm.N_arr[0, :].sum())

    S_hat_arr = rivm.S_hat_arr
    I_hat_arr = rivm.I_hat_arr
    S_true_arr = rivm.S_true_arr.astype(np.float64)
    I_true_arr = rivm.I_true_arr.astype(np.float64)
    N_per_muni = rivm.N_arr[0, :].astype(np.float64)

    if initialization:
        S_series, I_series = S_hat_arr, I_hat_arr
    else:
        S_series, I_series = S_true_arr, I_true_arr

    Delta_S_full = np.maximum(0.0, S_series[:-1, :] - S_series[1:, :])
    Delta_S_true_full = np.maximum(0.0, S_true_arr[:-1, :] - S_true_arr[1:, :])

    # for the diagnostic plot
    S_delta = Delta_S_full.sum(axis=1)
    S_delta_true = Delta_S_true_full.sum(axis=1)

    initial_guess = np.array([0.5, 20.0])
    bounds = [(1e-6, None), (1e-6, None)]
    point_rows = []
    ci_rows = []

    for d in range(rivm.T - 1):
        current_date_dt = start_date_dt + dt.timedelta(days=d)
        Delta_S = Delta_S_full[d, :]
        mob = _load_mob(scenario, current_date_dt.strftime('%d%m%Y'))

        S_live = S_series[d, :]
        I_live = I_series[d, :]

        z_vec = z_builder(mob, S_live, I_live, N_per_muni)
        logz_vec = np.log(np.maximum(z_vec, 1e-300))

        if not np.any(z_vec > 0):
            beta_hat, r_hat = initial_guess
        else:
            res = optimize.minimize(
                _neg_log_lik, initial_guess,
                args=(Delta_S, z_vec, logz_vec),
                method='L-BFGS-B', bounds=bounds, tol=1e-10,
                options={'maxiter': MAX_ITER},
            )
            beta_hat = max(res.x[0], 1e-10)
            r_hat = max(res.x[1], 1e-10)

        point_rows.append({
            'date': current_date_dt, 'beta': beta_hat, 'dispersion': r_hat,
        })
        initial_guess = [beta_hat, r_hat]

        if confidence:
            samples = _run_bootstrap(
                Delta_S, z_vec, logz_vec,
                beta_hat, r_hat, nr_bootstrap_samples, n_jobs,
            )
            q_lo = (1 - conf_level) / 2
            q_hi = (1 + conf_level) / 2
            ci_rows.append({
                'date': current_date_dt,
                'lower_beta': np.quantile(samples[:, 0], q_lo),
                'beta': np.quantile(samples[:, 0], 0.5),
                'upper_beta': np.quantile(samples[:, 0], q_hi),
                'lower_disp': np.quantile(samples[:, 1], q_lo),
                'disp': np.quantile(samples[:, 1], 0.5),
                'upper_disp': np.quantile(samples[:, 1], q_hi),
            })

    Transmission_rate_df = pd.DataFrame(point_rows)
    if not confidence:
        Transmission_rate_CI = Transmission_rate_df.copy()
    else:
        Transmission_rate_CI = pd.DataFrame(ci_rows).set_index('date')

    if plot:
        plt.plot(range(len(S_delta_true)), S_delta_true, label='True delta S')
        plt.plot(range(len(S_delta)), S_delta, ':', label='Estimated delta S')
        plt.xlabel('Day')
        plt.ylabel(r'$\sum_{i \in \mathcal{M}} \Delta S_i(t)$')
        plt.legend()
        plt.show()

    return Transmission_rate_df, Transmission_rate_CI


# Functions to run
def estimate_rates_per_day(start_date, confidence=False, conf_level=.95,
                           nr_bootstrap_samples=100, scenario='low',
                           seed=0, run=0, initialization=True,
                           plot=False, n_jobs=-1):
    """ABM mean-field NB regression."""
    return _run_estimation(
        start_date, z_vec_ABM, confidence, conf_level,
        nr_bootstrap_samples, scenario, seed, run,
        initialization, plot, n_jobs,
    )


def estimate_rates_per_day_MPM(start_date, confidence=False, conf_level=.95,
                               nr_bootstrap_samples=100, scenario='low',
                               seed=0, run=0, initialization=True,
                               plot=False, n_jobs=-1):
    """Classical residence-time MPM."""
    return _run_estimation(
        start_date, z_vec_MPM, confidence, conf_level,
        nr_bootstrap_samples, scenario, seed, run,
        initialization, plot, n_jobs,
    )


def estimate_rates_per_day_MPM2(start_date, confidence=False, conf_level=.95,
                                nr_bootstrap_samples=100, scenario='low',
                                seed=0, run=0, initialization=True,
                                plot=False, n_jobs=1):
    """One-beta MPM with three pathways (hh, ah, ha)."""
    return _run_estimation(
        start_date, z_vec_MPM_2, confidence, conf_level,
        nr_bootstrap_samples, scenario, seed, run,
        initialization, plot, n_jobs,
    )
