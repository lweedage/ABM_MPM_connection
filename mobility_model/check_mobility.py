"""
Diagnostic: does the ABM's realized agent positioning match MobMat?

For each saved Positions file, count how many agents from home i ended up
in muni j. Compare this realized matrix to MobMat (the input to the
mean-field equation).

If realized_diagonal_fraction != MobMat_diagonal_fraction, the mean-field
is being fed the wrong M̂ and any β̂ comparison is moot until we fix it.
"""
import datetime
import os
import pickle

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# settings — match plot_trajectories.py
# ------------------------------------------------------------
DIVISION = 'Medium100'
START_DATE = '01-01-2021'
NDAYS = 100
MOBILITY_SEEDS = range(10)
SAMPLE_DAYS = [0, 25, 50, 75, 99]      # days to spot-check

DATA_DIR = '../Output/Data'
MOBMAT_DIR = '../Data/Mobility/MobMats'

OUT_CSV = f'realized_vs_input_{DIVISION}.csv'

start_dt = datetime.datetime.strptime(START_DATE, '%d-%m-%Y')


def _load_mobmat(date_str):
    """MobMat[i, j] from disk: number of i-residents present in j."""
    with open(f'{MOBMAT_DIR}/{DIVISION}{date_str}.p', 'rb') as f:
        return np.asarray(pickle.load(f), dtype=float)


def _load_positions(seed, date_str):
    """Positions[p] = muni index of agent p on this day."""
    path = f'{DATA_DIR}/{DIVISION}/Seed_{seed}/Positions{date_str}.npy'
    pos = np.load(path)
    return np.array([p[0] for p in pos.astype(int)])


def _build_realized_matrix(positions, home_idx, n_munis):
    """realized[i, j] = #{agents with home i, currently in j}."""
    R = np.zeros((n_munis, n_munis), dtype=np.int64)
    np.add.at(R, (home_idx, positions), 1)
    return R


def _diag_share(M):
    s = M.sum()
    return float(np.diag(M).sum() / s) if s > 0 else float('nan')


def _row_normalize(M):
    rs = M.sum(axis=1, keepdims=True)
    return M / np.maximum(rs, 1e-12)


def main():
    rows = []
    print(f"{'seed':>5}{'day':>5}{'MobMat diag':>15}{'realized diag':>17}"
          f"{'Δ diag':>10}{'L1 row-norm':>14}{'max |Δ ij|':>13}")
    print('-' * 80)

    for seed in MOBILITY_SEEDS:
        # build home_idx once per seed
        people_df = pd.read_pickle(f'{DATA_DIR}/{DIVISION}/Seed_{seed}/PeopleDF.pkl')
        gemeenten = pd.read_pickle(f'{DATA_DIR}/{DIVISION}/Seed_{seed}/Gemeenten.pkl')
        gemeente_list = gemeenten.values.ravel()
        gem_to_idx = {g: i for i, g in enumerate(gemeente_list)}
        home_idx = np.array([gem_to_idx[h] for h in people_df['Home'].values])
        n_munis = len(gemeente_list)

        for d in SAMPLE_DAYS:
            date_str = (start_dt + datetime.timedelta(days=d)).strftime('%d%m%Y')

            try:
                M_input = _load_mobmat(date_str)
                positions = _load_positions(seed, date_str)
            except FileNotFoundError as e:
                print(f"seed={seed} day={d}: missing file ({e}); skipping")
                continue

            M_realized = _build_realized_matrix(positions, home_idx, n_munis)

            diag_in = _diag_share(M_input)
            diag_re = _diag_share(M_realized)

            # compare normalized matrices entry-wise
            Mbar_in = _row_normalize(M_input)
            Mbar_re = _row_normalize(M_realized.astype(float))
            l1 = float(np.abs(Mbar_in - Mbar_re).sum(axis=1).mean())
            max_dev = float(np.abs(Mbar_in - Mbar_re).max())

            rows.append({
                'seed': seed, 'day': d,
                'mobmat_diag_share': diag_in,
                'realized_diag_share': diag_re,
                'diag_diff': diag_re - diag_in,
                'mean_l1_row_distance': l1,
                'max_abs_entry_diff': max_dev,
            })

            print(f"{seed:>5}{d:>5}{diag_in:>15.4f}{diag_re:>17.4f}"
                  f"{diag_re - diag_in:>+10.4f}{l1:>14.4f}{max_dev:>13.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    # summary
    print("\n--- summary across all (seed, day) ---")
    print(f"mean MobMat diagonal share:     {df['mobmat_diag_share'].mean():.4f}")
    print(f"mean realized diagonal share:   {df['realized_diag_share'].mean():.4f}")
    print(f"mean diagonal difference:       {df['diag_diff'].mean():+.4f}")
    print(f"mean L1 row distance:           {df['mean_l1_row_distance'].mean():.4f}")
    print(f"max single-entry deviation:     {df['max_abs_entry_diff'].max():.4f}")


if __name__ == '__main__':
    main()