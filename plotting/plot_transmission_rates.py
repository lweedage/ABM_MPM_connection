"""
Plot pooled + per-seed transmission rate estimates across all (seed, run)
realizations produced by run_estimation.py. Run after the CI_*.csv files
exist for the (seed, run) pairs you want.
"""
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

from estimation.pathways import INITIALIZATION
from utils.util import *

# ------------------------------------------------------------
# settings
# ------------------------------------------------------------

_rocket = seaborn.color_palette('rocket', 8)
COLOR_ABM = _rocket[1]
COLOR_MPM = _rocket[5]
COLOR_MPM2 = _rocket[4]

markers = ['o', 's', 'p', 'd', '*']

# pick exactly one or none
MPM = False
MPM2 = False
NO_MOBILITY = False
POISSON = False

GRANULARITY = 'Medium'
INITIALIZATION = False

if MPM:
    METHOD_COLOR = COLOR_MPM
elif MPM2:
    METHOD_COLOR = COLOR_MPM2
elif NO_MOBILITY:
    METHOD_COLOR = _rocket[2]
elif POISSON:
    METHOD_COLOR = _rocket[3]
else:
    METHOD_COLOR = COLOR_ABM

N_SEEDS = 10
N_RUNS = 10
INIT_VAL = 4

if INIT_VAL == 4:
    TRUE_BETA = 0.5
else:
    TRUE_BETA = 0.25

START_DATE = '01-01-2021'
END_DATE = '11-04-2021'

INIT_DAYS = 14  # drop first 14 days (warmup)
END_DAYS = 7  # drop last 7 days (lookahead window too short)

FIG_DIR = '../Output/Plots'
DATA_DIR = '../Output/PlotData'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

if MPM:
    METHOD_TAG = f'MPM_init_{INITIALIZATION}_{INIT_VAL}'
elif NO_MOBILITY:
    METHOD_TAG = f'no_mob_init_{INITIALIZATION}_{INIT_VAL}'
elif POISSON:
    METHOD_TAG = f'Poisson_init_{INITIALIZATION}_{INIT_VAL}'
else:
    METHOD_TAG = f'ABM_init_{INITIALIZATION}_{INIT_VAL}'


# ------------------------------------------------------------
# load all realizations into (n_seeds, n_runs, n_days) arrays
# ------------------------------------------------------------
def _ci_path(seed, run):
    if MPM:
        suffix = f'_MPM_init_{INITIALIZATION}'
    elif MPM2:
        suffix = f'_MPM2_init_{INITIALIZATION}'
    elif NO_MOBILITY:
        suffix = f'_no_mob_init_{INITIALIZATION}'
    elif POISSON:
        suffix = f'_Poisson_init_{INITIALIZATION}'
    else:
        suffix = f'_init_{INITIALIZATION}'
    return (f'../ABM_data/{GRANULARITY}/transmission_rates/Initialization{INIT_VAL}/'
            f'CI_{START_DATE}-{END_DATE}_seed_{seed}_perday{run}{suffix}.csv')


def _load_all():
    """Read all CI_*.csv files. Files come in two flavors:
        - with bootstrap CI: beta + lower/upper + disp + lower/upper
        - point-est only:    beta + dispersion
    Detected by column sniffing.
    """
    # discover n_days from the first successful file
    n_days = None
    for s in range(N_SEEDS):
        for r in range(N_RUNS):
            p = _ci_path(s, r)

            if os.path.exists(p):
                n_days = len(pd.read_csv(p, index_col=0))
                break
        if n_days is not None:
            break
    if n_days is None:
        raise FileNotFoundError(
            f"No CI files found under ABM_data/{GRANULARITY}/transmission_rates/."
        )

    cols = ['beta', 'lower_beta', 'upper_beta',
            'disp', 'lower_disp', 'upper_disp']
    arrs = {c: np.full((N_SEEDS, N_RUNS, n_days), np.nan) for c in cols}

    n_loaded = n_with_ci = n_missing = 0
    for s in range(N_SEEDS):
        for r in range(N_RUNS):
            p = _ci_path(s, r)
            if not os.path.exists(p):
                n_missing += 1
                continue

            df = pd.read_csv(p, index_col=0)
            if len(df) != n_days:
                warnings.warn(f"{p}: expected {n_days} rows, got {len(df)} skipping.")
                n_missing += 1
                continue

            if 'disp' not in df.columns and 'dispersion' in df.columns:
                df = df.rename(columns={'dispersion': 'disp'})

            if 'beta' in df.columns: arrs['beta'][s, r, :] = df['beta'].values
            if 'disp' in df.columns: arrs['disp'][s, r, :] = df['disp'].values

            has_ci = all(c in df.columns for c in
                         ['lower_beta', 'upper_beta', 'lower_disp', 'upper_disp'])
            if has_ci:
                for c in ['lower_beta', 'upper_beta', 'lower_disp', 'upper_disp']:
                    arrs[c][s, r, :] = df[c].values
                n_with_ci += 1

            n_loaded += 1

    print(f"Loaded {n_loaded} realizations "
          f"({n_with_ci} with CI, {n_loaded - n_with_ci} point-only, "
          f"{n_missing} missing) with {n_days} days each.")
    return arrs, n_days, n_with_ci, n_loaded


arrs, n_days, n_with_ci, n_loaded_total = _load_all()

t_full = np.arange(n_days)

SHOW_BOOTSTRAP_BAND = n_with_ci >= 1


# ------------------------------------------------------------
# aggregations
# ------------------------------------------------------------
def _pooled_stats(arr3d):
    """Returns (median, q025, q975) across all realizations per day."""
    flat = arr3d.reshape(-1, arr3d.shape[-1])
    return (np.nanmedian(flat, axis=0),
            np.nanquantile(flat, 0.025, axis=0),
            np.nanquantile(flat, 0.975, axis=0))


def _per_seed_median(arr3d):
    return np.nanmedian(arr3d, axis=1)


beta_med, beta_q025, beta_q975 = _pooled_stats(arrs['beta'])
beta_boot_lo = np.nanmedian(arrs['lower_beta'].reshape(-1, n_days), axis=0)
beta_boot_hi = np.nanmedian(arrs['upper_beta'].reshape(-1, n_days), axis=0)

disp_med, disp_q025, disp_q975 = _pooled_stats(arrs['disp'])
disp_boot_lo = np.nanmedian(arrs['lower_disp'].reshape(-1, n_days), axis=0)
disp_boot_hi = np.nanmedian(arrs['upper_disp'].reshape(-1, n_days), axis=0)

beta_by_seed = _per_seed_median(arrs['beta'])
disp_by_seed = _per_seed_median(arrs['disp'])

lo, hi = INIT_DAYS, n_days - END_DAYS
t_crop = t_full[lo:hi]


def _crop(x):
    return x[lo:hi]


# ------------------------------------------------------------
# plots
# ------------------------------------------------------------
def _pooled_panel(ax, t, med, q025, q975, ylabel, hline=None, log_y=False):
    ax.fill_between(t, q025, q975, color=METHOD_COLOR, alpha=0.25)
    ax.plot(t, med, color=METHOD_COLOR, marker=markers[0],
            linewidth=1.5, markersize=4, label='Median across runs')
    if hline is not None:
        ax.axhline(hline, color='red', linestyle=':', label=r'True $\beta$')
    if log_y:
        ax.set_yscale('log')
    ax.set_ylabel(ylabel)


# beta
fig, ax = plt.subplots(figsize=(10, 5))
_pooled_panel(ax, t_crop, _crop(beta_med), _crop(beta_q025), _crop(beta_q975),
              ylabel=r'$\beta$', hline=TRUE_BETA)
ax.set_xlim(lo, hi - 1)
ax.set_ylim(TRUE_BETA - 0.2, TRUE_BETA + 0.2)
ax.set_xlabel('day')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f'Estimated_beta_all_runs_{GRANULARITY}_{METHOD_TAG}.png'), dpi=150)
plt.show()

# dispersion
fig, ax = plt.subplots(figsize=(10, 5))
_pooled_panel(ax, t_crop, _crop(disp_med), _crop(disp_q025), _crop(disp_q975),
              ylabel=r'dispersion $r$', log_y=True)
ax.set_xlim(lo, hi - 1)
ax.set_xlabel('day')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f'Estimated_dispersion_all_runs_{GRANULARITY}_{METHOD_TAG}.png'), dpi=150)
# plt.show()
plt.close(fig)

# by-seed plots — useful for spotting per-mobility-seed bias
def _by_seed_panel(ax, t, by_seed_arr, ylabel, hline=None, log_y=False):
    palette = seaborn.light_palette(METHOD_COLOR, n_colors=N_SEEDS + 2,
                                    reverse=False)[2:]
    for s in range(N_SEEDS):
        ax.plot(t, by_seed_arr[s], color=palette[s], linewidth=1.5,
                alpha=1, label=f'seed {s}')
    if hline is not None:
        ax.axhline(hline, color='red', linestyle=':', label=r'True $\beta$')
    if log_y:
        ax.set_yscale('log')
    ax.set_ylabel(ylabel)


fig, ax = plt.subplots(figsize=(10, 5))
_by_seed_panel(ax, t_crop, beta_by_seed[:, lo:hi], ylabel=r'$\beta$', hline=TRUE_BETA)
ax.set_xlim(lo, hi - 1)
# ax.set_ylim(-0.1, 1)
ax.set_ylim(TRUE_BETA - 0.2, TRUE_BETA + 0.2)

ax.set_xlabel('day')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f'Estimated_beta_by_seed_{GRANULARITY}_{METHOD_TAG}.png'), dpi=150)
# plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
_by_seed_panel(ax, t_crop, disp_by_seed[:, lo:hi], ylabel=r'$k$', log_y=True)
ax.set_xlim(lo, hi - 1)
ax.set_xlabel('day')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f'Estimated_dispersion_by_seed_{GRANULARITY}_{METHOD_TAG}.png'), dpi=150)
# plt.show()
plt.close(fig)

# ------------------------------------------------------------
# CSVs to reproduce figures
# ------------------------------------------------------------
beta_mean_pool = np.nanmean(arrs['beta'].reshape(-1, n_days), axis=0)
disp_mean_pool = np.nanmean(arrs['disp'].reshape(-1, n_days), axis=0)

pooled_df = pd.DataFrame({
    'day': t_full,
    'beta_median': beta_med, 'beta_mean': beta_mean_pool,
    'beta_q025': beta_q025, 'beta_q975': beta_q975,
    'beta_boot_lo_median': beta_boot_lo, 'beta_boot_hi_median': beta_boot_hi,
    'disp_median': disp_med, 'disp_mean': disp_mean_pool,
    'disp_q025': disp_q025, 'disp_q975': disp_q975,
    'disp_boot_lo_median': disp_boot_lo, 'disp_boot_hi_median': disp_boot_hi,
})
pooled_df.to_csv(os.path.join(DATA_DIR, f'Estimated_all_runs_{METHOD_TAG}.csv'), index=False)

by_seed_rows = [
    {'seed': s, 'day': d,
     'beta_median': beta_by_seed[s, d],
     'disp_median': disp_by_seed[s, d]}
    for s in range(N_SEEDS) for d in range(n_days)
]
pd.DataFrame(by_seed_rows).to_csv(
    os.path.join(DATA_DIR, f'Estimated_by_seed_{METHOD_TAG}.csv'), index=False)

print(f"Saved plots + CSVs ({'MPM' if MPM else 'ABM-NB'} variant).")
