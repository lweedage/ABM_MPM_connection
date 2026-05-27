"""
Decompose the ABM mean-field force of infection into the four pathways
of eq. (2.11) and plot how much each contributes per day.

The paper's claim is that the classical MPM is missing p^ha and p^aa.
This script measures, on actual ABM output, what fraction of the
right-hand side of dS/dt those missing pathways actually account for —
which is *not* the same as the total probability mass of p^aa across
all (i, j, k) triples.

Specifically:
    p^aa MASS:         sum over all (i, j!=i, k!=j) of p^aa_{ij,kj}.
                       Big in NL data because hubs have many co-visitors.
    p^aa CONTRIBUTION: weights each (i, j, k) by I_k/N_k. A triple only
                       contributes to infections when k actually has
                       prevalence. Early in the epidemic this is small.

Run from the project root:
    python -m src.estimation.pathway_decomposition

Outputs (under Output/PlotData/ and the project root):
    - pathway_fractions.csv  : per-day fraction of FOI from each pathway
    - pathway_fractions.png  : line plot of those fractions over time
    - pathway_by_muni.csv    : per-muni p^aa fraction at a few snapshots
"""
import os
import pickle
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from estimation import rivm_loader
from utils.constants import CoronaConstants

# ------------------------------------------------------------
# settings — match what run_estimation.py uses
# ------------------------------------------------------------
START_DATE = '01-01-2021'
SCENARIO = 'Medium100'
SEED = 9
RUN = 0
INITIALIZATION = True

# Short label for this stay-home / diagonal setting, appended to the output
# filenames so different settings don't overwrite each other. For example set
# this to 'diag30' for the ~30% stay-home run and 'diag80' for the ~80% run,
# then point make_pathways_figure.py at the two resulting CSVs. Leave '' for
# the old single-file behaviour.
DIAGONAL_TAG = '30'

OUT_DIR = '../Output/PlotData'
os.makedirs(OUT_DIR, exist_ok=True)

# '' -> 'pathway_fractions.csv'; 'diag30' -> 'pathway_fractions_diag30.csv'
_SUFFIX = f'_{DIAGONAL_TAG}' if DIAGONAL_TAG else ''


# ------------------------------------------------------------
# pathway decomposition of the ABM mean-field FOI
# ------------------------------------------------------------
def decompose_foi(mob, S_live, I_live, N_live):
    """Split z_i = S_i * sum_{j,k} M_ij * w_kj * I_k/N_k into four pieces.

    Returns dict with arrays of length M for:
        phh: j=i, k=i
        pah: j!=i, k=j
        pha: j=i, k!=i
        paa: j!=i, k!=j

    Their sum equals z_vec_ABM(mob, S, I, N) up to numerical noise.
    """
    M = mob.shape[0]
    infect_frac = I_live / np.maximum(N_live, 1e-10)  # f_k

    # row-normalized 'visitor share' matrix w_kj = P(home=k | currently in j)
    numer = mob * N_live[:, None]  # (M, M); numer[k, j] = M_kj * N_k
    denom = numer.sum(axis=0)  # (M,);  denom[j] = sum_l M_lj * N_l
    W = numer / np.maximum(denom, 1e-10)  # (M, M)

    diagA = np.diag(mob)  # M_ii
    diagW = np.diag(W)  # w_ii

    # p^hh: j=i, k=i
    phh = S_live * diagA * diagW * infect_frac

    # p^ah: j!=i, k=j  → S_i * sum_{j!=i} M_ij * w_jj * f_j
    A_off = mob - np.diag(diagA)  # zero diagonal
    pah = S_live * (A_off @ (diagW * infect_frac))

    # p^ha: j=i, k!=i  → S_i * M_ii * sum_{k!=i} w_ki * f_k
    W_off_rows = W - np.diag(diagW)  # zero diagonal
    # we need sum_k w_ki * f_k with k!=i; that's (W_off_rows.T @ f)_i
    pha = S_live * diagA * (W_off_rows.T @ infect_frac)

    # p^aa: j!=i, k!=j  → S_i * sum_{j!=i} M_ij * sum_{k!=j} w_kj * f_k
    # inner sum: sum_{k!=j} w_kj * f_k = (W_off_rows.T @ f)_j   (using same W with zeroed diag)
    inner = W_off_rows.T @ infect_frac  # (M,); inner[j] = sum_{k!=j} w_kj * f_k
    paa = S_live * (A_off @ inner)

    total = phh + pah + pha + paa
    return {'phh': phh, 'pah': pah, 'pha': pha, 'paa': paa, 'total': total}


def z_vec_ABM_reference(mob, S_live, I_live, N_live):
    """Reference computation, identical to estimate_rates.z_vec_ABM,
    used to sanity-check that the four pathway pieces sum to it."""
    infect_frac = I_live / np.maximum(N_live, 1e-10)
    numerator = mob * N_live[:, None]
    denominator = numerator.sum(axis=0)
    w = numerator / np.maximum(denominator, 1e-10)
    pressure_j = w.T @ infect_frac
    mobility_term_i = mob @ pressure_j
    return S_live * mobility_term_i


def _load_mob_normalized(scenario, date_str):
    """Mirror estimate_rates._load_mob: row-normalize the raw mobility matrix."""
    with open(f'../data/Mobility/MobMats/{scenario}{date_str}.p', 'rb') as f:
        mob = pickle.load(f)
    row_sums = mob.sum(axis=1, keepdims=True)
    return mob / np.maximum(row_sums, 1e-10)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    start_date_dt = dt.datetime.strptime(START_DATE, '%d-%m-%Y')

    print(f"Loading RIVM data (scenario={SCENARIO}, seed={SEED}, run={RUN})...")
    rivm = rivm_loader.RivmLoader(SCENARIO, start_date_dt, SEED, RUN)
    CoronaConstants.population_nl = float(rivm.N_arr[0, :].sum())

    if INITIALIZATION:
        S_series = rivm.S_hat_arr
        I_series = rivm.I_hat_arr
    else:
        S_series = rivm.S_true_arr.astype(np.float64)
        I_series = rivm.I_true_arr.astype(np.float64)
    N_per_muni = rivm.N_arr[0, :].astype(np.float64)
    T = rivm.T

    # storage
    daily = {'phh': [], 'pah': [], 'pha': [], 'paa': [], 'total': []}
    # per-muni snapshots at a few interesting days
    snapshot_days = [10, 20, 30, 50, 80]
    snapshot_days = [d for d in snapshot_days if d < T]
    per_muni_snapshots = {d: {} for d in snapshot_days}

    # sanity check accumulator
    max_relerr = 0.0

    print(f"Running decomposition over {T} days...")
    for d in range(T):
        date_str = (start_date_dt + dt.timedelta(days=d)).strftime('%d%m%Y')
        try:
            mob = _load_mob_normalized(SCENARIO, date_str)
        except FileNotFoundError as e:
            print(f"  day {d}: missing mobility file ({e}); stopping.")
            T = d
            break

        S_live = S_series[d, :]
        I_live = I_series[d, :]

        parts = decompose_foi(mob, S_live, I_live, N_per_muni)

        # sanity check vs reference z_vec
        ref = z_vec_ABM_reference(mob, S_live, I_live, N_per_muni)
        denom = max(np.abs(ref).sum(), 1e-12)
        relerr = float(np.abs(parts['total'] - ref).sum() / denom)
        max_relerr = max(max_relerr, relerr)

        # national-aggregate contributions on this day
        for k in daily:
            daily[k].append(float(parts[k].sum()))

        # per-muni snapshot
        if d in snapshot_days:
            for k in ['phh', 'pah', 'pha', 'paa', 'total']:
                per_muni_snapshots[d][k] = parts[k].copy()

    print(f"\nDecomposition sanity check: max relative error vs z_vec_ABM = {max_relerr:.2e}")
    print("  (should be ~1e-12; larger means a bug in the decomposition)\n")

    # ------------------------------------------------------------
    # per-day aggregate fractions
    # ------------------------------------------------------------
    df_daily = pd.DataFrame(daily)
    df_daily.insert(0, 'day', np.arange(len(df_daily)))

    total = df_daily['total'].replace(0, np.nan)
    for k in ['phh', 'pah', 'pha', 'paa']:
        df_daily[f'{k}_frac'] = df_daily[k] / total

    df_daily['missing_by_MPM_frac'] = df_daily['pha_frac'] + df_daily['paa_frac']

    out_csv = os.path.join(OUT_DIR, f'pathway_fractions{_SUFFIX}.csv')
    df_daily.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    # ------------------------------------------------------------
    # plots
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))

    # top: stacked fractions
    days = df_daily['day'].values
    fracs = df_daily[['phh_frac', 'pah_frac', 'pha_frac', 'paa_frac']].values.T
    labels = [
        r'$hh$',
        r'$ah$',
        r'$ha$',
        r'$aa$',
    ]
    colors = ['#4daf4a', '#377eb8', '#ff7f00', '#e41a1c']
    ax.stackplot(days, fracs, labels=labels, colors=colors, alpha=0.85)
    ax.set_ylabel('fraction of the force of infection')
    ax.set_ylim(0, 1)
    ax.set_xlabel('day')
    ax.legend(fontsize=12)
    # ax.set_title('Decomposition of the force of infection by pathway')

    fig.tight_layout()
    fig_path = f'pathway_fractions{_SUFFIX}.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {fig_path}")

    # ------------------------------------------------------------
    # per-muni snapshots (which munis are most affected by p^aa?)
    # ------------------------------------------------------------
    rows = []
    munis = rivm.munis
    for d, parts in per_muni_snapshots.items():
        if not parts:
            continue
        total_i = parts['total']
        safe_total = np.where(total_i > 0, total_i, np.nan)
        for k in ['phh', 'pah', 'pha', 'paa']:
            frac_k = parts[k] / safe_total
            for m_idx, muni in enumerate(munis):
                rows.append({
                    'day': d,
                    'muni': muni,
                    'pathway': k,
                    'absolute': float(parts[k][m_idx]),
                    'fraction': float(frac_k[m_idx])
                    if np.isfinite(frac_k[m_idx]) else np.nan,
                })

    df_muni = pd.DataFrame(rows)
    out_muni_csv = os.path.join(OUT_DIR, f'pathway_by_muni{_SUFFIX}.csv')
    df_muni.to_csv(out_muni_csv, index=False)
    print(f"Wrote {out_muni_csv}")

    # ------------------------------------------------------------
    # console summary
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Per-day national-aggregate fractions (selected days)")
    print("=" * 60)
    print(df_daily[['day', 'phh_frac', 'pah_frac', 'pha_frac', 'paa_frac',
                    'missing_by_MPM_frac']].iloc[::10].to_string(
        index=False, float_format='%.4f'))

    print("\nTop 10 munis by p^aa fraction at each snapshot day:")
    for d in snapshot_days:
        sub = df_muni[(df_muni['day'] == d) & (df_muni['pathway'] == 'paa')]
        sub = sub.sort_values('fraction', ascending=False).head(10)
        print(f"\n  day {d}:")
        print(sub[['muni', 'fraction', 'absolute']].to_string(
            index=False, float_format='%.4f'))


if __name__ == '__main__':
    main()
