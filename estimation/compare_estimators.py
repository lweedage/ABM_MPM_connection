"""
Pool per-(seed, run) beta estimates into one tidy CSV for the estimation
figures. This script is now a *pure reader/aggregator*: it reads the CSVs
that run_estimation.py wrote and produces

    Output/PlotData/estimator_comparison_pooled.csv

It does NOT plot — make_figures.py does that from the CSV alone. Run order:
    run_estimation.py  ->  compare_estimators.py  ->  make_figures.py

The grid has three axes:
    estimator       'ABM' (mean-field FOI)   |  'MPM' (classical residence-time)
    init_val         4 -> true beta 0.5       |  5 -> true beta 0.25   (beta scenario / folder)
    initialization   False -> oracle true S,I |  True -> reconstructed S,I from incidence

Michiel's Figure 1 = the initialization=False panels (idealised, true S/I).
Michiel's Figure 3 = the initialization=True  panels (reconstructed, "in practice").
Both come from this one CSV; make_figures.py filters on `initialization`.

BAND SEMANTICS (relabeled, kept as-is by choice):
    beta_q025 / beta_q975 are the 2.5th / 97.5th PERCENTILES of the per-realization
    point estimates across the pooled (seed, run) replicates -- i.e. the spread of
    the estimator across simulated epidemics. They are NOT bootstrap confidence
    intervals. The on-disk 'CI_' filename prefix is legacy from run_estimation.py.
"""
import os
import warnings

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# settings -- must match run_estimation.py for filename compatibility
# ------------------------------------------------------------
START_DATE = '01-01-2021'
END_DATE = '11-04-2021'
SCENARIO = 'High'

N_SEEDS = 10
N_RUNS = 100

# which slices of the grid to pool. Edit these to taste.
ESTIMATORS = ('ABM', 'MPM')
INIT_VALS = (4, 5)  # 4 -> beta 0.5, 5 -> beta 0.25
INITIALIZATIONS = (False, True)  # False -> Fig 1 (true S/I), True -> Fig 3 (reconstructed)

# true beta per init_val (single source of truth, also written into the CSV)
TRUE_BETA = {4: 0.5, 5: 0.25}

DATA_DIR = '../Output/PlotData'
OUT_CSV = os.path.join(DATA_DIR, 'estimator_comparison_pooled.csv')
os.makedirs(DATA_DIR, exist_ok=True)

# legacy on-disk filename prefix written by run_estimation.py
LEGACY_FILE_PREFIX = 'CI'

# stable window (after transient, before the look-ahead edge) for the bias summary
BIAS_WINDOW = slice(40, 80)


# ------------------------------------------------------------
# file pathing -- mirrors run_estimation._fn exactly
#   folder : Initialization{init_val}
#   suffix : 'MPM_init' (MPM) | 'init' (ABM), then '_{initialization}'
# ------------------------------------------------------------
def _realization_csv_path(seed, run, estimator, init_val, initialization):
    if estimator == 'MPM':
        stem = 'MPM_init'
    elif estimator == 'ABM':
        stem = 'init'
    else:
        raise ValueError(f"estimator must be 'ABM' or 'MPM', got {estimator!r}")
    suffix = f'{stem}_{initialization}'
    return (f'../ABM_data/{SCENARIO}/transmission_rates/Initialization{init_val}/'
            f'{LEGACY_FILE_PREFIX}_{START_DATE}-{END_DATE}_seed_{seed}_perday{run}'
            f'_{suffix}.csv')


# ------------------------------------------------------------
# load all realizations for one (estimator, init_val, initialization) cell
# ------------------------------------------------------------
def _load_cell(estimator, init_val, initialization):
    """Return (arr, n_loaded) where arr is (N_SEEDS, N_RUNS, n_days) of beta
    estimates (NaN where missing), or (None, 0) if the cell has no files."""
    n_days = None
    for s in range(N_SEEDS):
        for r in range(N_RUNS):
            p = _realization_csv_path(s, r, estimator, init_val, initialization)
            if os.path.exists(p):
                n_days = len(pd.read_csv(p, index_col=0))
                break
        if n_days is not None:
            break
    if n_days is None:
        return None, 0

    arr = np.full((N_SEEDS, N_RUNS, n_days), np.nan)
    n_loaded = 0
    for s in range(N_SEEDS):
        for r in range(N_RUNS):
            p = _realization_csv_path(s, r, estimator, init_val, initialization)
            if not os.path.exists(p):
                continue
            df = pd.read_csv(p, index_col=0)
            if len(df) != n_days:
                warnings.warn(f"{p}: row count {len(df)} != {n_days}; skipping.")
                continue
            if 'beta' in df.columns:
                arr[s, r, :] = df['beta'].values
                n_loaded += 1
    return arr, n_loaded


def _pooled_stats(arr3d):
    """Per-day median and 2.5/97.5 percentiles across all pooled realizations."""
    flat = arr3d.reshape(-1, arr3d.shape[-1])
    return (np.nanmedian(flat, axis=0),
            np.nanquantile(flat, 0.025, axis=0),
            np.nanquantile(flat, 0.975, axis=0))


# ------------------------------------------------------------
# build the tidy long-format table
# ------------------------------------------------------------
def main():
    rows = []
    summary = []  # (estimator, init_val, initialization, n, mean_beta, bias)

    for initialization in INITIALIZATIONS:
        for init_val in INIT_VALS:
            for estimator in ESTIMATORS:
                arr, n = _load_cell(estimator, init_val, initialization)
                tag = f"{estimator} init_val={init_val} initialization={initialization}"
                if arr is None:
                    print(f"  [skip] no files for {tag}")
                    continue

                med, q025, q975 = _pooled_stats(arr)
                tb = TRUE_BETA[init_val]
                print(f"  [ok]   {tag}: {n}/{N_SEEDS * N_RUNS} realizations, "
                      f"{arr.shape[-1]} days")

                for d in range(arr.shape[-1]):
                    rows.append({
                        'estimator': estimator,
                        'init_val': init_val,
                        'true_beta': tb,
                        'initialization': initialization,
                        'day': d,
                        'beta_median': med[d],
                        'beta_q025': q025[d],
                        'beta_q975': q975[d],
                        'n_realizations': n,
                    })

                mean_b = float(np.nanmean(med[BIAS_WINDOW]))
                summary.append((estimator, init_val, initialization, n,
                                mean_b, mean_b - tb))

    if not rows:
        raise SystemExit(
            "No data found. Run run_estimation.py first (for the INITIALIZATION "
            "and INIT_VAL settings you want), then re-run this script."
        )

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(rows)} rows)")

    # ------------------------------------------------------------
    # console summary: mean estimated beta over the stable window
    # ------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"Mean estimated beta over days {BIAS_WINDOW.start}-{BIAS_WINDOW.stop} "
          f"(bias = mean - true beta)")
    print("=" * 78)
    print(f"{'estimator':<6}{'init_val':>9}{'initialization':>16}"
          f"{'n':>6}{'mean beta':>12}{'bias':>10}")
    for est, iv, ini, n, mean_b, bias in summary:
        print(f"{est:<6}{iv:>9}{str(ini):>16}{n:>6}{mean_b:>12.4f}{bias:>+10.4f}")
    print()


if __name__ == '__main__':
    main()
