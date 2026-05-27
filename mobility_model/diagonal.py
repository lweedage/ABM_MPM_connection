"""
Visualize the size of the diagonal (stay-at-home fraction) versus the
off-diagonal entries in the row-normalized mobility matrix.

Motivates Section 1 of the meeting note: TomTom only records off-diagonal
trips, so the diagonal must be constructed as N_i - sum of trips out.
We need to show that this diagonal dominates the row mass (≈80-90%) to
justify the construction.

Run from project root:
    python -m src.estimation.diagonal_size_diagnostic
"""
import datetime as dt
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# settings
# ------------------------------------------------------------
SCENARIO = 'Medium100'
START_DATE = '01-01-2021'

# sample a handful of days across the simulation window
SAMPLE_DAYS = [0, 25, 50, 75, 99]

OUT_FIG = f'mobility_diagonal_distribution_{SCENARIO}.png'
OUT_CSV = f'Output/PlotData/mobility_diagonal_summary_{SCENARIO}.csv'
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)


def _load_mob(scenario, date_str):
    """Load and row-normalize one MobMat pickle."""
    with open(f'../data/Mobility/MobMats/{scenario}{date_str}.p', 'rb') as f:
        mob = pickle.load(f)
    row_sums = mob.sum(axis=1, keepdims=True)
    return mob / np.maximum(row_sums, 1e-10)


def main():
    start_date_dt = dt.datetime.strptime(START_DATE, '%d-%m-%Y')

    # collect diagonals + off-diagonals across sample days
    diags_per_day = {}
    offdiags_per_day = {}
    for d in SAMPLE_DAYS:
        date_str = (start_date_dt + dt.timedelta(days=d)).strftime('%d%m%Y')
        try:
            mob = _load_mob(SCENARIO, date_str)
        except FileNotFoundError as e:
            print(f"day {d}: {e}; skipping.")
            continue
        diag = np.diag(mob)
        offdiag = mob - np.diag(diag)
        offdiag_flat = offdiag[offdiag > 1e-10]  # drop the structural zeros

        diags_per_day[d] = diag
        offdiags_per_day[d] = offdiag_flat

    if not diags_per_day:
        print("No data loaded; aborting.")
        return

    # ------------------------------------------------------------
    # console summary
    # ------------------------------------------------------------
    print(f"\nMobility diagonal summary (scenario={SCENARIO}):")
    print(f"{'day':>5}{'mean diag':>12}{'median':>10}"
          f"{'min':>10}{'max':>10}{'mean off-diag':>16}{'>0.5 frac':>12}")
    summary_rows = []
    for d, diag in diags_per_day.items():
        offdiag = offdiags_per_day[d]
        row = {
            'day': d,
            'mean_diag': float(np.mean(diag)),
            'median_diag': float(np.median(diag)),
            'min_diag': float(np.min(diag)),
            'max_diag': float(np.max(diag)),
            'mean_offdiag': float(np.mean(offdiag)) if offdiag.size > 0 else 0.0,
            'frac_diag_gt_0p5': float((diag > 0.5).mean()),
        }
        summary_rows.append(row)
        print(f"{d:>5}{row['mean_diag']:>12.3f}{row['median_diag']:>10.3f}"
              f"{row['min_diag']:>10.3f}{row['max_diag']:>10.3f}"
              f"{row['mean_offdiag']:>16.5f}{row['frac_diag_gt_0p5']:>12.2%}")

    pd.DataFrame(summary_rows).to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    # ------------------------------------------------------------
    # plot
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: histogram of diagonal weights (stay-home fraction per muni),
    # one day for clarity
    ax = axes[0]
    day_for_hist = SAMPLE_DAYS[len(SAMPLE_DAYS) // 2]  # middle day
    if day_for_hist in diags_per_day:
        diag = diags_per_day[day_for_hist]
        ax.hist(diag, bins=40, color='C0', edgecolor='white', alpha=0.85)
        ax.axvline(np.mean(diag), color='red', linestyle='--', linewidth=1.5,
                   label=f'mean = {np.mean(diag):.3f}')
        ax.axvline(np.median(diag), color='black', linestyle=':', linewidth=1.5,
                   label=f'median = {np.median(diag):.3f}')
        ax.set_xlabel(r'diagonal weight $\hat{M}_{ii}$ (stay-home fraction)')
        ax.set_ylabel('number of municipalities')
        ax.set_title(f'A. Distribution of stay-home fractions  (day {day_for_hist})')
        ax.set_xlim(0, 1)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Panel B: diagonal vs off-diagonal entries on a log scale
    ax = axes[1]
    all_diag = np.concatenate(list(diags_per_day.values()))
    all_offdiag = np.concatenate(list(offdiags_per_day.values()))

    # log-scale bins covering both
    lo = max(1e-6, min(all_diag.min(), all_offdiag.min()))
    bins = np.logspace(np.log10(lo), 0, 60)

    ax.hist(all_offdiag, bins=bins, alpha=0.6, label='off-diagonal entries',
            color='C1', edgecolor='white')
    ax.hist(all_diag, bins=bins, alpha=0.6, label='diagonal entries',
            color='C0', edgecolor='white')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\hat{M}_{ij}$  (log scale)')
    ax.set_ylabel('count (pooled across sample days)')
    ax.set_title('B. Diagonal vs off-diagonal magnitudes')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    print(f"Wrote {OUT_FIG}")


if __name__ == '__main__':
    main()